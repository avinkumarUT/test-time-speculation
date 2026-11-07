# DFlash full-weight TTS: design

This document describes the full-weight mode of test-time speculation (TTS)
for the DFlash speculator. The code is in `speculative/dflash_tts.py` (state,
loss, optimizer, the differentiable block forward),
`speculative/dflash_worker.py` (integration into the speculate/verify loop)
and `models/dflash.py` (the draft's sliding-window switch).

## Overview

In speculative decoding, the target model verifies every drafted token, and
the verification pass yields the target's distribution at each drafted
position. TTS uses that signal to update the draft model online, while the
response is being generated, so that later drafts are accepted more often.

DFlash is a block-diffusion style speculator: each round it drafts one block
of tokens (16 for `z-lab/Qwen3.6-27B-DFlash`) in a single non-causal forward
pass. In the full-weight mode, an update is one Adam step on the weights of
the draft model itself, distilling the target's verify distribution for a
block that has just been drafted and verified. The adapted weights belong to
the request that is being generated: they are restored to the pretrained
weights at the start of every request.

Two TTS modes exist for DFlash:

| | Full-weight (this document) | Per-request LoRA (`DFLASH_LORA_TTS_DESIGN.md`) |
|---|---|---|
| What is updated | Every weight of the draft block forward | Low-rank adapters on selected projections; base weights frozen |
| Concurrency | One in-flight request | Batched serving; every request owns its adapters |
| State | One set of weights, a snapshot and Adam moments, plus float32 master weights in the default precision | Adapters and Adam moments per request |
| Draft forward on update rounds | Differentiable block forward, run eagerly; every other round uses the unchanged serving forward and its CUDA graphs | Unchanged serving forward plus a replay |
| Enable with | `--speculative-dflash-tts-enable` | `--speculative-dflash-lora-tts-enable` |

Use the full-weight mode to adapt the whole speculator to a single long
generation. Use the LoRA mode whenever more than one request is served at a
time. The two modes are mutually exclusive.

Correctness of the output is never at stake in either mode: the target model
still decides every token. A failed or skipped update only costs acceptance
length.

## Data flow of one round

```
main stream                                            side stream
-----------                                            -----------
fence (wait for the previous optimizer step)
append committed tokens to the draft KV cache
draft forward
    update round: differentiable block forward,
                  hidden states kept with their graph
    other rounds: serving forward (CUDA graph)
sample the draft tokens with the target's head
target verify forward
snapshot the verify logits
verify (accept / reject, bonus token)
append the committed tokens to the draft KV cache
loss, backward,
gradients to float32,
gradient completion (TP)  ----------------------------> clip, Adam step,
advance the stride counter                              results stored in the
                                                        weights
next round ...
```

## Which weights are updated

Every position that the DFlash draft model attends to as context holds K/V
projected from the *target's* hidden states: `project_target_hidden` (`fc`,
then `hidden_norm`) followed by the K/V projection, `k_norm` and RoPE writes
them into the draft KV cache. The block forward then attends from the block's
own positions to that context and to the block itself.

| Weights | Role | Updated |
|---|---|---|
| `fc`, `hidden_norm` | Run only when context K/V is written. The loss has no gradient path to them. | Never |
| K/V slices of `qkv_proj`, `k_norm` | Two roles: project the target's hidden states into context K/V, and produce the block's own K/V. The loss observes the second role. | Yes |
| Q slice of `qkv_proj`, `q_norm`, `o_proj`, `gate_up_proj`, `down_proj`, all layer norms, the final norm | Used only in the block forward. | Yes |

The optimizer holds every draft parameter (or its float32 master, see
[Precision](#precision)); `fc` and `hidden_norm` never receive a gradient and
therefore never change.

The cached context K/V are constants of an update: they are read back from
the draft KV cache and are not rewritten when the weights change. Each cached
token keeps the K/V projection that was current when it was appended. For
this reason the fused KV materialization helper, which stacks a copy of the
K/V projection weights when it is built, is not used in this mode: context
K/V are materialized layer by layer from the live weights.

## The loss

The draft block is non-causal: hidden state `k` of a block predicts the token
at block position `k`. Position 0 holds the bonus token that the target has
already emitted, so positions `1..B-1` are the drafted ones. The target
verifies the block causally: its logits at block row `k` are the distribution
of position `k + 1`. Draft positions `1..B-1` are therefore scored against
target rows `0..B-2`.

Draft logits are the block's hidden states multiplied by the target's
`lm_head` weight, in the head's dtype, exactly as when the draft tokens are
sampled. With draft distribution `q` and target distribution `p` at a
position:

```
lk (default):  loss = -log( sum_x min(p(x), q(x)) )
kl:            loss = lambda * KL(p || q) + (1 - lambda) * CE(q, argmax q)
```

`--speculative-tts-loss` selects the objective. `sum_x min(p, q)` is the
speculative acceptance probability. The second term of `kl` is a cross-entropy
of the draft against its own prediction; it keeps the draft's predictions sharp
while the first term moves them towards the target. The loss is averaged over
the drafted positions with weights `exp(-k / gamma)`, `k = 0..B-2`, normalized by their sum. Acceptance
is a chain, so an early position gates every position behind it and counts
most. The softmaxes and the loss are computed in fp32.

The target's logits are copied before verification modifies them, and they
and the `lm_head` weight enter the loss as constants.

With `--speculative-dflash-tts-kl-threshold > 0`, an update is skipped when
the divergence term of its block (the `lk` loss, or the KL of `kl`) is below the
threshold. Reading that value on the host
synchronizes with the GPU on every update round.

## Update stride and request lifecycle

An update fires on the rounds whose index within the request is a multiple of
`--speculative-dflash-tts-freq` (the update stride `S`), starting with the
request's first decode round. The stride counter advances on every verify
round. All other rounds draft and verify as usual.

An update fires only while the request is alone in the batch, and the update
of a round in which the request finishes is skipped.

* **Prefill.** The pending optimizer step, if any, is awaited. If the weights
  have changed, they are restored from a snapshot of the pretrained weights
  taken at startup, the float32 masters, if the mode has them, are set to the
  float32 value of that snapshot, and the optimizer state (Adam's moments and
  step count) is cleared. Every request therefore starts from the same state, bit for bit.
  The stride counter restarts at 0. The request's prompt is then written into
  the draft KV cache with the pretrained weights.
* **Update round.** See [The differentiable block forward](#the-differentiable-block-forward)
  and [Asynchronous stepping](#asynchronous-stepping).
* **Request end.** A one-line summary is logged (see [Log lines](#log-lines)).

## The differentiable block forward

SGLang is an inference engine, and the serving forward of the draft model
cannot be differentiated:

* the attention backends (FlashInfer, Triton, FA) have no backward;
* the fused RMSNorm, SwiGLU, Q/K-norm and RoPE kernels have no backward, and
  several of them write their result in place;
* the model's `forward` is decorated with `torch.no_grad()`, and the
  scheduler loop runs with autograd disabled (optionally in inference mode,
  whose tensors cannot be saved for backward);
* the tensor-parallel collectives sever or zero the gradient;
* draft CUDA graphs replay recorded kernels and build no autograd graph;
* with deterministic inference, `torch.bmm` is replaced by a function that
  bypasses autograd.

None of these raises an error. Each leaves some or all parameters without a
gradient, and the update silently does nothing for them.

On an update round the block is therefore drafted by a separate forward,
`dflash_tts.DFlashBlockForward`. It computes the same function as the serving
forward from the model's own weights, using only operations with a backward:

* every layer's pure-PyTorch implementation (`forward_native`) for the norms,
  RoPE and SwiGLU, and `F.linear` on the projection weights;
* `F.scaled_dot_product_attention` over the concatenation of the cached
  prefix K/V and the block's own K/V. The prefix is gathered from the draft
  KV cache through the request's token table and is a constant; the block's
  K/V stay on the autograd tape, so the K and V projections receive
  gradients;
* conjugate pairs of collectives under tensor parallelism (see
  [Tensor parallelism](#tensor-parallelism));
* an explicit exit from inference mode, and private copies of the inputs
  that are later rewritten in place.

One forward serves both purposes: its hidden states are detached and used to
sample the draft tokens, and the same hidden states, with their autograd
graph, are kept until the target has verified the block. No second forward is
needed for the update.

The forward does not go through the draft's model runner, so it is never
replayed from a CUDA graph. It contains no `torch.bmm` call and works
unchanged with `--enable-deterministic-inference`.

It runs on update rounds only. Every other round is drafted by the serving
forward of the draft's model runner, which is set up exactly as without TTS,
including its CUDA graphs. Updates change the draft's parameters in place and
never their dtype or storage, so a captured graph reads the current weights.
If the block forward fails on an update round (for example, out of memory
under autograd) and `tp_size == 1`, the serving forward drafts that round and
the round counts as a failed update.

### Sliding-window layers

DFlash draft checkpoints can be trained with sliding-window attention on some
layers (for example, `z-lab/Qwen3.6-27B-DFlash` uses a 2048-token window on 4
of its 5 layers). With TTS enabled, those layers are served with their trained
window and the draft uses the triton attention backend, as in the LoRA mode.
The block forward reads each layer's serving window: for a windowed layer it
gathers only the last `window` prefix locations and masks them per query row
exactly like the serving kernel, so the update is computed on the function
that is served, and the prefix gather of a windowed layer does not grow with
the response. The window must not be shorter than a block.

A windowed draft that is captured in CUDA graphs on a target with a hybrid
sliding-window KV pool is rejected at startup; use `--disable-cuda-graph` in
that configuration.

## Asynchronous stepping

With asynchronous stepping (the default) the optimizer step is taken off the
critical path between verification and the response to the scheduler.

* `loss.backward()` runs its kernels on the stream of the forward, which is
  the main stream, whatever stream is current when it is called. With
  `tp_size > 1` backward contains collectives, and so does the completion of
  the gradients. All of them are issued on the main stream, and so is the
  conversion of the gradients to float32, which comes between the two.
* The rank-local remainder runs on a side CUDA stream after that stream has
  been made to wait for the main stream: scaling the gradients for clipping,
  the Adam step, and the store of its results into the draft's parameters.
  Tensors that are produced on the main stream and consumed on the side
  stream (the gradients, the clipping coefficient, the finiteness flag) are
  marked with `record_stream`. With the `bfloat16` optimizer dtype the float32
  weight and moments of a tensor, and the stored moments, are allocated and
  consumed on the side stream alone, so they need no mark.
* A CUDA event is recorded on the side stream after the results are stored.
  Before draft weights are read again (at the start of the next decode round and at
  a prefill) the host waits for that event.

The wait is a host-side `event.synchronize()` and not a `wait_event` on the
main stream. The next draft forward is usually replayed from a CUDA graph on
PyTorch's internal stream, and a wait recorded on the host's current stream
does not propagate into a graph replay: the replay would read partially
written weights without any error. The host-side wait is correct whichever
stream reads the weights. It can block only in the round that follows an
update.

With `--speculative-dflash-tts-no-async` the whole update is issued on the
main stream.

## Precision

The draft's parameters are served in the dtype they were loaded in (bfloat16
for the published drafts) and keep that dtype and their storage at all times.
The serving forward, the fused kernels and the draft CUDA graphs read them in
place, and the differentiable block forward of an update round runs on the
same parameters. `--speculative-dflash-tts-optimizer-dtype` selects the
precision in which the optimizer state is stored. The arithmetic of a step is
float32 in both modes, and clipping and the tensor-parallel gradient completion
are the same. The option means the same as
`--speculative-eagle-tts-optimizer-dtype` and
`--speculative-dflash-lora-tts-optimizer-dtype`.

### `float32` (default)

Every draft parameter has a float32 master copy. The cast from the served
dtype is exact, so a master starts at the value of its parameter. Adam steps
on the masters, with float32 moments. One update is:

1. backward through the served parameters, which yields gradients in the
   served dtype;
2. the gradients are cast to float32 and handed to the masters, and the
   served-dtype gradients are released;
3. gradient completion across ranks, the global norm and clipping, on the
   float32 gradients;
4. one Adam step on the masters;
5. every stepped master is copied back into its parameter, in place, which
   rounds it to the served dtype.

A bfloat16 value has 8 significant bits, so neighbouring values are 0.4% to
0.8% of the value's magnitude apart, and Adam's steps have a magnitude of
about the learning rate. An update smaller than half that spacing leaves the
served weight unchanged, but it is kept in the master, and the served weight
moves once the accumulated change crosses a rounding boundary. The served
weights are always the masters rounded to the served dtype. The request
summary reports the largest change of a served weight and warns when updates
changed none.

The snapshot for the per-request reset is held once, in the served dtype. A
reset copies it into the parameters and, cast to float32, into the masters.

A draft whose parameters are all float32 has no masters: the parameters are
their own masters and Adam steps on them in place.

### `bfloat16`

The parameters are their own masters and the moments are bfloat16, so no float32
tensor outlives a step. One update is:

1. backward through the served parameters, which yields gradients in the
   served dtype;
2. the gradients are cast to float32, and the served-dtype gradients are
   released;
3. gradient completion across ranks, the global norm and clipping, on the
   float32 gradients;
4. for one tensor at a time, the weight and the moments are widened to float32,
   which is exact, and torch's functional Adam (the function
   `torch.optim.Adam.step` dispatches to) steps them (`tts_optim.Bf16StateAdam`);
5. the weight and the moments are rounded once and stored, the weight in place.

With the same state and gradients, a `bfloat16` update is therefore the update of
the `float32` mode, rounded. `torch.optim.Adam` is not stepped on the bfloat16
tensors directly: that would round after every operation of the update instead of
once, and its clipping norm would be a bfloat16 value.

The cost of the mode is resolution. An update smaller than half the spacing of
the weight it is added to is rounded away, and is not remembered anywhere. With a
small learning rate most weights do not change at all, and the mode needs a
correspondingly larger learning rate to have an effect. The stored second moment
does not decay: without a new gradient `v` changes by `(1 - beta2) v` per update,
0.1% of `v` with `beta2 = 0.999`, and half the spacing of a bfloat16 value is 0.2%
to 0.4% of its magnitude, so the rounding undoes the decay. A large gradient
therefore lowers the step size for the rest of the request. The first moment is
not affected: it changes by 10% per update. The request summary warns when
updates changed no weight.

`bfloat16` is accepted only for a draft model whose parameters are bfloat16;
this is checked at startup.

### Serving in both modes

Rounds without an update run the serving forward, including draft CUDA
graphs. As in the LoRA mode, windowed draft layers are served with their
window and context K/V are written by the per-layer path, so the weights that
are read are always the live ones.

## Tensor parallelism

`tp_size >= 1` is supported. The draft model is sharded like any SGLang
model: `qkv_proj` and `gate_up_proj` are column-parallel, `o_proj` and
`down_proj` are row-parallel, the norms are replicated, and the target's
`lm_head` is sharded along the vocabulary. An update is exact under tensor
parallelism: every rank applies the update that a single-GPU run would apply
to its shard, and replicated parameters receive identical updates on every
rank.

* **Row-parallel projections.** The partial sums are all-reduced in the
  forward, and the gradient passes through the reduce unchanged.
* **Column-parallel projections and the head.** Their input is replicated
  and their output sharded, so each rank computes only its shard's
  contribution to the input's gradient. The forward marks those inputs with
  an operator that is the identity in the forward and all-reduces the
  gradient in backward.
* **Head.** Each rank computes its shard of the draft logits, and the shards
  are gathered into the full vocabulary with a gather whose backward returns
  the rank's own slice.
* **`q_norm` and `k_norm`.** These replicated weights act on the heads a rank
  owns, so their gradients are per-rank partial sums. They are all-reduced
  after backward.
* **Clipping.** The gradient norm is the global one: sharded parameters
  contribute the sum over ranks, replicated parameters are counted once.
  The partial gradients and the norm travel in a single all-reduce.
* **KL gate.** The ranks compare a value that has passed through an
  all-reduce, so they always take the same branch.

The draft's key/value heads must be sharded, not replicated: the mode
requires at least `tp_size` key/value heads.

A failure in an update raises with `tp_size > 1`. A rank that skipped its
part would hold different weights and leave the ranks' collectives out of
step, which is much harder to diagnose than a stopped server.

## Safeguards

* **The target model is never modified.** The DFlash draft owns no embedding
  and no head; it is fed the target's embeddings and scored with the target's
  `lm_head` weight, which the loss detaches. At startup the state verifies
  that no draft parameter shares storage with a target parameter.
* **Gradient coverage.** After the first backward the state checks that every
  parameter of the block forward received a gradient. If one did not, TTS is
  disabled with an error that names the parameters, instead of training a
  part of the model silently. A block whose hidden states carry no autograd
  graph raises as well.
* **Non-finite updates.** A flag computed on the GPU records whether the loss
  and the gradient norm of an update were finite. It is read at the next
  fence, where the update has completed. After a non-finite update the
  pretrained weights are restored, the masters, if the mode has them, are reset
  to them, and the optimizer state is cleared.
* **Failed updates** (`tp_size == 1`). A failure of the differentiable block
  forward, of the loss, of backward or of the optimizer step is logged once
  and decoding continues; a round whose block forward failed is drafted by
  the serving forward. After three consecutive failures TTS disables itself
  and restores the pretrained weights. A step that failed part-way may have
  written some weights, masters or moments; they are restored at the next
  prefill.

## Memory

All TTS memory is allocated outside the KV cache budget
(`--mem-fraction-static`), so leave headroom for it. With `N` draft
parameters in total across ranks, for a bfloat16 draft:

| | Weights | Snapshot | Masters | Adam moments | Standing total | Gradients during an update | Float32 copies during an update |
|---|---|---|---|---|---|---|---|
| `float32` | 2N bytes | 2N bytes | 4N bytes | 8N bytes | 16N bytes | 4N bytes, and 2N bytes while they are converted | none |
| `bfloat16` | 2N bytes | 2N bytes | none | 4N bytes | 8N bytes | 4N bytes, and 2N bytes while they are converted | 12 bytes per element of the tensor being stepped |

A draft served without TTS takes the 2N bytes of its weights.
`z-lab/Qwen3.6-27B-DFlash` has about 1.73 billion parameters, of which the largest
tensor, a `gate_up_proj`, has 178 million. The mode holds about 27.7 GB (`float32`)
or 13.8 GB (`bfloat16`) between updates and up to about 38 GB or 26.4 GB at the
peak of an update, spread over the ranks. The weights, the snapshot and the
masters are allocated at startup, the moments at the first update of a request.
The gradients are released when the optimizer step has completed, and with
`bfloat16` the float32 copies of a tensor when its step has been stored.

Backward produces the gradients in the weights' dtype (2N bytes). They are
converted to float32 one tensor at a time (4N bytes) and each is released once
converted, but PyTorch's allocator keeps the released blocks cached for the next
update, so both count towards the peak.

On CUDA, Adam also holds the square root of the second moment during a step, 4
bytes per element of the tensors it steps at once. That is 4N bytes (6.9 GB for
the draft above) on top of the `float32` figures, and the size of one tensor (0.7
GB) on top of the `bfloat16` figures.

Moments and gradients exist only for the parameters the block forward uses,
which excludes `fc` and `hidden_norm` (8% of the parameters of the draft
above); the figures in the table are upper bounds by that margin.

An update round additionally holds the autograd graph of one block between
the draft forward and the update, and the loss works on
`[block_size, vocab]` fp32 tensors.

## Configuration

| Flag | Default | Meaning |
|---|---|---|
| `--speculative-dflash-tts-enable` | off | Enable full-weight TTS for DFlash. |
| `--speculative-dflash-tts-lr` | 1e-5 | Adam learning rate. |
| `--speculative-tts-loss {lk,kl}` | `lk` | Objective, for every speculator. |
| `--speculative-tts-lambda` | 0.9 | Weight of the KL term of the `kl` loss, in [0, 1]. |
| `--speculative-dflash-tts-freq` | 10 | Update stride `S`, in verify rounds; must be >= 1. |
| `--speculative-dflash-tts-pos-weight-gamma` | 7.0 | Decay of the per-position loss weights. |
| `--speculative-dflash-tts-kl-threshold` | 0.0 | Skip an update whose KL term is below this value (nats); 0 disables the gate. |
| `--speculative-dflash-tts-grad-clip` | 1.0 | Global gradient-norm limit; `<= 0` disables clipping. |
| `--speculative-dflash-tts-no-async` | async on | Run the optimizer step on the main stream. |
| `--speculative-dflash-tts-optimizer-dtype` | `float32` | Precision in which the optimizer state is stored: `float32` (float32 master weights and moments, written back into the served weights) or `bfloat16` (the served weights and bfloat16 moments, widened for a step and rounded once when stored). |

## Log lines

At startup (TP rank 0):

```
DFlash TTS enabled (full-weight). optimizer_dtype=float32 fp32_master=True lr=... freq=... async=... tp_size=... trainable_params=...
DFLASH full-weight TTS ready. draft_dtype=torch.bfloat16, draft_cuda_graph=True, draft_forward=differentiable block forward on update rounds, context_kv=per-layer from live weights
```

`fp32_master` tells whether float32 masters are held (the `float32` optimizer
dtype on a draft that is not float32 itself); with `optimizer_dtype=bfloat16` it
is `False`. `draft_dtype` is the dtype the
draft is served in, and `draft_cuda_graph` whether the draft's model runner
captured CUDA graphs, which depends only on the server's CUDA graph settings.

At the first update (the counts depend on the draft):

```
DFlash TTS: the first backward reached <n> of <m> draft parameter tensors; not reached (...): ['fc.weight', 'hidden_norm.weight']
```

When a request finishes:

```
DFlash TTS request summary: rounds=... updates=... kl_skipped=... max_abs_weight_change=...
```

`max_abs_weight_change` is the largest absolute difference between a current
and a pretrained served weight on rank 0. A value of 0 after one or more
updates means that no served weight changed, and a warning is logged: with
`bfloat16` every update was below half the spacing of the weights and is lost, and
with `float32` the updates accumulated in the masters have not yet moved a served
weight across a rounding boundary.

## Requirements and limitations

Checked at startup:

* `--speculative-algorithm DFLASH`, and a block size of at least 2.
* A single in-flight request. `--max-running-requests` defaults to 1 in this
  mode, and a larger explicit value is rejected.
* Not combinable with `--speculative-dflash-lora-tts-enable`, nor with
  `--enable-lora` / `--lora-paths`.
* A target whose KV pool is a hybrid sliding-window pool (a target with
  sliding-window attention layers of its own) only without CUDA graphs
  (`--disable-cuda-graph`). Targets with a plain KV pool, such as the Qwen
  models, keep all CUDA graphs.
* `--kv-cache-dtype auto`: the block forward reads the draft KV cache
  directly.
* Unquantized (fp16 / bf16 / fp32) draft weights and target `lm_head`; no
  added vocabulary; with `tp_size > 1`, unpadded `lm_head` shards and at
  least `tp_size` key/value heads in the draft.
* `bfloat16` optimizer dtype only for a bfloat16 draft.
* The draft's KV pool is the plain multi-head pool, and a sliding window is
  not shorter than a block.
* DFlash's own restrictions apply unchanged (for example, no dp-attention
  and `pp_size == 1`).

Behavior to be aware of:

* Each request starts from the pretrained draft and adapts from its first
  decode round; the weights and the optimizer state are reset at every
  prefill.
* Update rounds draft the block with the differentiable forward, which runs
  eagerly and is slower than the serving forward of the draft. All other
  rounds, and the target, run as without TTS.
* With the `float32` optimizer dtype an update that is smaller than the
  spacing of a served weight changes the master only; the served weight
  follows once the accumulated change is large enough. With `bfloat16` such an
  update is lost.
* Context K/V are not rewritten when the weights change (see
  [Which weights are updated](#which-weights-are-updated)). With the radix
  cache enabled, a later request that shares a prefix containing generated
  tokens (a multi-turn conversation) reuses draft K/V that were written with
  the earlier request's adapted K/V projections. This affects only acceptance
  length; `--disable-radix-cache` avoids it.
* The target's final logit soft-capping, if the target has one, is not
  applied to the draft logits in the loss.
* TTS memory is allocated outside the KV cache budget (see
  [Memory](#memory)). With `tp_size == 1`, updates that keep running out of
  memory disable TTS and decoding continues with the pretrained draft. With
  `tp_size > 1` a failed update stops the server.

## Tests

`test/registered/spec/dflash_tts/` includes these tests. They
need no checkpoint and no server:

* `test_dflash_tts.py`: the state on a toy draft. The loss against an
  independent computation, the positional weights, the update stride, the KL
  gate, the bit-exact reset of weights, masters and optimizer state and the
  reproduction of a request after it, a float32-master update against a
  reference (float32 Adam on the float32 weights with the gradient of the
  bfloat16 forward, rounded), `bfloat16` updates against a reference (float32
  Adam with the same clip on the widened state, rounded once per update), the
  state's dtypes and byte counts, updates below the bfloat16 spacing in both
  optimizer dtypes, synchronous and asynchronous updates, target safety, and
  the failure handling. `TTS_TEST_DEV=cpu` runs it without a GPU, skips the
  asynchronous comparison and checks the stream order of an update with stand-ins
  for the CUDA stream API.
* `test_dflash_block_forward.py`: the differentiable block forward on a small
  `DFlashDraftModel` with a real KV pool. Forward and gradients against an
  independent implementation, for full-attention and windowed layers; the
  forward inside no-grad and inference-mode contexts and with `torch.bmm`
  replaced; both optimizer dtypes over a bfloat16 draft against the same
  references; the worker's hooks in both optimizer dtypes (block forward on
  update rounds only, fallback to the serving forward when it fails); and
  agreement with the replay forward of the LoRA mode.
* `test_dflash_tts_tp.py`: a `tp_size` 2 update emulated with one thread per
  rank, compared with the single-GPU update, with negative controls, for a
  float32 draft (stepped in place) and for a bfloat16 draft in both optimizer
  dtypes.
