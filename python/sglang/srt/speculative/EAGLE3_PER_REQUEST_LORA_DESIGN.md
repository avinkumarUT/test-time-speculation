# Per-request LoRA adapters for EAGLE-3 Test-Time Speculation

This document describes how Test-Time Speculation (TTS) is implemented for
EAGLE-3 speculative decoding in batched serving. The code lives in

- `eagle_tts.py`: update engine (schedule, loss, optimizer step, async
  stepping, the differentiable draft forward),
- `eagle_lora_tts.py`: low-rank adapters, the per-request adapter pool, row
  selection, and the pooled optimizer,
- `eagle_worker_v2.py`: integration into the EAGLE draft/verify loop.

## Background

In speculative decoding a small draft model (the "speculator") proposes
several tokens and the target model verifies them in one forward pass. That
verification pass also yields the target's next-token distribution at every
drafted position. TTS uses these distributions to update the draft model
online, while the response is being generated, so that the draft adapts to the
sequence at hand and more of its proposals are accepted.

EAGLE-3 is a single-layer autoregressive draft that proposes a chain of tokens
from the target's hidden states. Its per-step logits during the draft rollout
are the proposal distributions, and those are what TTS fits to the target.

## Goals

1. **Correct under batching.** Concurrent requests must adapt independently:
   one request's updates, resets and gradient scale must not affect another's.
2. **Target unchanged.** The target model's weights stay bit-identical, so
   decoding remains lossless.
3. **Compatible with CUDA graphs.** Non-firing rounds keep replaying the
   captured draft graphs and must see the current adapter weights and the
   current batch composition without recapture.
4. **Correct under tensor parallelism.** All TP ranks must keep holding the
   same model after every update.
5. **Bounded memory.** The per-request state must be small enough to
   preallocate for every request slot.

## Why per-request adapters

TTS has two update modes (`--speculative-eagle-tts-mode`):

- `full` updates the draft's own weights (all but `fc` and its input norms). The trainable state is the whole draft
  plus a master copy, Adam moments and a restore snapshot, held once per
  worker. The weights are restored at every prefill, so with concurrent
  requests one request's prefill would discard the adaptation of the others.
  This mode is kept as a single-request reference path.
- `lora` freezes the base weights and updates low-rank adapters
  `delta = (alpha / rank) * (x @ A^T) @ B^T` on selected projections.

With `lora` and a single shared adapter the same problem remains: the adapter
is reset whenever any request starts. `--speculative-eagle-tts-per-request`
gives every in-flight request its own adapter slot, optimizer state and reset.
This is only practical with LoRA, because the per-request state is
`rank * (in + out)` elements per adapted matrix instead of a copy of the draft.

Freezing the base has two further consequences:

- **No snapshot.** B is initialized to zero, so the delta is exactly zero
  before the first update, and resetting A to its initial value and B to zero
  restores the pretrained draft exactly.
- **The target is safe by construction.** EAGLE-3 binds the draft's
  `embed_tokens` (and `lm_head`, for drafts without a head of their own) to
  the target's tensors. A frozen base never writes them. `EagleTTSState`
  additionally checks that no trainable tensor shares storage with the target.

The adapters are applied in the module forward and are never merged into the
base weights. Merging would need either a pristine copy of every adapted
matrix or incremental re-merging, which accumulates bf16 rounding error and
makes the reset inexact; it also could not be done per request.

## Which projections can carry adapters

`--speculative-eagle-tts-lora-targets` takes a comma-separated subset of:

| target | module | TP layout | gradient crosses attention | feeds the draft KV cache |
|---|---|---|---|---|
| `q` | `qkv_proj` (q slice) | column-parallel | yes | no |
| `k` | `qkv_proj` (k slice) | column-parallel | yes | yes |
| `v` | `qkv_proj` (v slice) | column-parallel | yes | yes |
| `o` | `o_proj` | row-parallel | no | no |
| `gate_up` | `mlp.gate_up_proj` | column-parallel | no | no |
| `down` | `mlp.down_proj` | row-parallel | no | no |
| `fc` | `model.fc` | replicated | yes | yes |
| `lm_head` | `lm_head` | vocab-parallel | no | no |

Two properties of a single-layer draft drive the choice:

**Attention backward.** The fused attention kernels have no backward. A target
at or above the attention output (`o`, `gate_up`, `down`, `lm_head`) needs no
gradient through attention, so the fused kernel stays in place on firing
rounds. A target below it (`q`, `k`, `v`, `fc`) requires the differentiable
attention rebuild in `eagle_tts._sdpa_attention`, which gathers the full prefix
K/V and runs `scaled_dot_product_attention` once per rollout step. It is only
supported for the plain multi-head KV pool (`MHATokenToKVPool`).

**KV-cache consistency.** The draft's KV cache is written every round and read
by every later round. An update to `k`, `v` or `fc` changes what is written,
so entries cached before the update become inconsistent with those written
after it, while the loss only sees the current block. These targets are
accepted but log a warning.

The default set is `q,o,gate_up,down,lm_head`: every target that leaves the KV
cache consistent. Dropping `q` avoids the attention rebuild. `o` alone needs
no gradient through attention and is the target set used for the paper's
batched results.

When the pool is not in use (single shared adapter), the two post-attention
RMSNorm weight vectors are trained in full alongside the adapters. They are
ordinary shared model parameters, so they stay frozen whenever the pool is
active.

Restrictions on `lm_head`: it cannot be adapted when the draft ties `lm_head`
to `embed_tokens`, nor together with `--enable-fp32-lm-head`.

## The adapter pool

For every adapter, A and B carry a leading pool dimension:

```
A : [P, rank, in_local]        B : [P, out_local, rank]
```

**The slot is the request's `req_pool_idx`.** SGLang's request-to-token table
hands out dense indices, unique among running requests, so they are used
directly as slot ids. No side table is needed, and no device-to-host sync
(`.tolist()`) is introduced on the decode path.

**Sizing.** `P` is the number of rows the request table allocates
(`max_running_requests + 1` for the standard pool; row 0 is a padding row that
CUDA-graph padded batches index). Every slot carries adapter and optimizer
state whether or not a request occupies it, so `--max-running-requests` should
be set to the concurrency that is actually served. The worker refuses to build
the pool if that value is unset, because the size SGLang would then derive
from KV capacity can be in the thousands.

**Initialization.** A is Kaiming-uniform, generated on CPU from a seed derived
from the adapter name, identically for every slot and every TP rank (each rank
takes its own slice of the full tensor). B is zero.

**Reset on reuse.** Request indices are recycled. When a prefill batch arrives,
`EAGLEWorkerV2.forward_batch_generation` resets exactly the slots that batch
claims: A back to its template, B to zero, and the slot's optimizer state
(master, moments, step count) cleared. Other requests keep their state.
A whole-pool reset is refused in pooled mode.

**Per-row application.** Row `i` of a forward uses slot `rows[i]`:

```
h_i     = x_i @ A[rows[i]]^T                 # [N, 1, rank]
delta_i = (h_i @ B[rows[i]]^T) * alpha/rank  # [N, 1, out]
```

implemented as two batched matmuls over `index_select`-gathered factors. A row
touches only its own slot, so one batched backward yields each request's
gradient with no cross-talk.

The batched matmul is called as `torch.ops.aten.bmm`, not `torch.bmm`.
`--enable-deterministic-inference` replaces `torch.bmm` with a Python-level
function that bypasses the dispatcher and therefore autograd; going through
the aten op keeps the batch-invariant kernel and records the autograd node.

With `pool_size == 1` (no `--speculative-eagle-tts-per-request`) the adapter
uses plain matmuls on slot 0 and no gather.

## Row selection under CUDA graphs

The adapter wrappers only receive the layer input, not the `ForwardBatch`, so
the row-to-slot mapping is published out of band through a `_RowSelector`:

- The mapping lives in a **static index buffer written in place**. A captured
  CUDA graph records the buffer's address, so every replay reads the mapping
  the current round wrote. A freshly allocated index tensor would be ignored
  by a replay.
- The pooled/unpooled branch is **fixed at construction**. CUDA graph capture
  runs at worker initialization, before any request exists, and the Python
  branch taken during capture is fixed in the graph.
- Before the first mapping is published (graph warmup and capture), the
  selector returns the static buffer as is, so the gather is traced. After the
  first publish, a forward whose row count differs from the published mapping
  raises.
- A forward under grad gets a **snapshot** of the buffer, because
  `index_select` version-checks its index during backward and the buffer is
  rewritten by the next bind.

Every draft forward publishes its own mapping first:

| forward | rows | mapping |
|---|---|---|
| draft rollout (`draft`) | one per request | `req_pool_indices` |
| draft prefill (`_draft_extend_for_prefill`) | prompt tokens | `req_pool_indices` repeated by each request's token count |
| draft extend after verify (`_draft_extend_for_decode`) | accept tokens | same, repeated by token count |

`lm_head` has a separate selector. `LogitsProcessor` selects the positions that
need logits before calling the head, so the head sees a block of
`num_tokens_for_logprob_per_req` rows per request rather than the token rows;
the worker publishes that mapping explicitly.

## What is trained on

On a firing round the draft rollout runs eagerly under `tts_grad_mode`, and
its per-step logits are kept. Rollout step `i` produces the distribution for
tree node `i + 2`; target verification row `j` predicts node `j + 1`. Node
`i + 2` is therefore paired with target row `i + 1`.

The loss (`--speculative-tts-loss`, shared with DFlash) is either

```
kl:  lambda * KL(target || draft) + (1 - lambda) * CE(draft, argmax draft)
lk:  -log( sum_x min(p_target(x), q_draft(x)) )
```

with positions weighted by `depth_decay ** depth`. `sum_x min(p, q)` is the
speculative acceptance probability. For drafts with a reduced vocabulary the
target logits are restricted to the draft's columns before the softmax.

Every drafted position enters the loss. A row past the prefix of correct drafts is
conditioned on the draft's own tokens, and its label is the target's
distribution given that drafted prefix. `--speculative-eagle-tts-correct-drafts-only`
restricts the loss to the prefix of correct drafts of the chain.

**Per-request normalization.** In pooled mode the loss is the sum over
requests of each request's weighted mean. A single mean over the batch would
divide each request's gradient by the total row weight of the round, making a
request's update depend on what the other requests contributed. With
per-request means every slot receives the gradient it would have received had
its request run alone.

## Pooled optimizer and per-request clipping

`PoolAdam` replaces `torch.optim.Adam` in pooled mode:

- It holds the **master weights** and both Adam moments as `[P, ...]` tensors
  and writes the result back to the bf16 pool parameters. The state is fp32 by
  default, because small steps applied directly in bf16 would be lost to
  rounding; see [Precision](#precision).
- It keeps a **step count per slot**. Requests start at different times, and
  Adam's bias correction depends on the number of steps a slot has taken.
- It steps only the slots of the current batch, and among those only the ones
  with a **non-zero gradient**. A request that contributed no rows to the loss
  keeps its weights, moments and step count unchanged.
- Duplicate slots in one step raise, since they would mean two requests were
  mapped to one pool row.

**Gradient clipping is per slot.** The gradient norm is computed over each
slot's rows separately and each slot gets its own clip coefficient. A global
norm would let one request with a large gradient shrink every other request's
update.

Non-finite losses and non-finite gradients skip the step.

## Precision

The trained tensors (the adapter parameters, or the draft weights in `full`
mode) are served in bfloat16 and keep that dtype and their storage at all times.
The serving forward and the captured draft graphs read them in place.
`--speculative-eagle-tts-optimizer-dtype` selects the precision in which the
optimizer state is stored. The arithmetic of a step is float32 in both modes,
and clipping and the tensor-parallel gradient completion are the same.

### `float32` (default)

Every trained tensor has a float32 master copy and float32 moments. A step
updates the master and copies it into the served tensor, which rounds it to
bfloat16. An update smaller than the spacing of a served weight stays in the
master, and the served weight moves once the accumulated change crosses a
rounding boundary.

### `bfloat16`

The served tensors are their own masters and the moments are bfloat16, so no
float32 tensor outlives a step. A step widens the state to float32, which is
exact, runs the same arithmetic as the `float32` mode, and rounds each result
once when it stores it, with the cast's round-to-nearest-even.

| path | optimizer | how a step runs |
|---|---|---|
| per-request LoRA | `PoolAdam(fp32_state=False)` | The rows of the slots in the batch are widened, stepped with the `PoolAdam` formula and rounded. A slot that did not fire, or fired with a zero gradient, keeps every bit of its weights, moments and step count. |
| shared LoRA, `full` | `Bf16StateAdam` | The weights and the gradients are widened to float32 leaves, and the leaves' gradients are clipped by `_clip_grads`, as those of float32 masters are. Then, one tensor at a time, the moments are widened, torch's functional Adam (the function `torch.optim.Adam.step` dispatches to) steps the leaf, and the weight and the moments are rounded. A tensor without a gradient is skipped. |

The modes share everything around that step. Non-finite gradients skip it before
any state is touched, and with `tp_size > 1` all ranks skip together. A reset
restores the served tensors from the adapter template or the snapshot and clears
the moments and step counts. The per-slot clip of the pooled path works on the
bfloat16 gradients in both modes, so the two modes consume the same gradient. In
the unpooled paths both modes clip float32 gradients and get the same coefficient.
With the same state and gradients, a `bfloat16` step is therefore the `float32`
step, rounded.

`torch.optim.Adam` is not stepped on the bfloat16 tensors directly. That would
round after every operation of the update instead of once, so its moments and
weights depart from this result within a few steps, and its clipping norm would
be a bfloat16 value.

The cost of the mode is resolution.

**Weights.** An update smaller than half the spacing of the weight it is added
to is rounded away, and is not remembered anywhere. Adam's steps have a magnitude
of about the learning rate, so a learning rate has a weight magnitude above which
its steps are lost:

| learning rate | steps are lost from about `|w|` = |
|---|---|
| 1e-3 | 0.5 |
| 1e-4 | 0.031 |
| 1e-5 | 0.0039 |

Adapter factors are small: `B` starts at zero and `A` is drawn with magnitude at
most `1 / sqrt(in_features)` (0.0156 for the 4096 input features of `o`). At a
learning rate of 1e-4 their updates survive, with a resolution of one or a few
spacings. At the default 1e-5 most updates to `A` are lost. The draft's own weights
in `full` mode are generally larger than adapter factors, so more of their updates
are lost at a given rate, and the mode needs a correspondingly larger learning rate.

**Second moment.** Without a new gradient `v` changes by `(1 - beta2) v` per step,
0.1% of `v` with `beta2 = 0.999`. Half the spacing of a bfloat16 value is 0.2% to
0.4% of its magnitude, so this decay rounds away and `v` does not decay. `v` grows
only while the increment `0.001 * g^2` exceeds half its spacing. A large gradient
therefore lowers the step size for the rest of the request (a spike leaves `v`
unchanged through any later small gradients, where float32 decays it to 0.7% in
5000 steps), and under a constant gradient `v` stops at a fifth to a third of
`g^2`, depending on where the value lies between two powers of two, so that after
several thousand steps Adam's update is about twice as large as with float32 (the
same at first, about 1.6 times at 1000 steps). The first moment is
not affected: it changes by 10% per step.

`bfloat16` is accepted only when every trained tensor is bfloat16; this is checked
at startup.

## Update schedule and asynchronous stepping

An update fires when `round_idx % freq == 0`
(`--speculative-eagle-tts-freq`, the update stride). In pooled mode the round
counter belongs to the worker and keeps running across prefills, so **all
requests in the batch fire on the same rounds**.

The training forward is the draft rollout itself: on a
firing round the whole batch leaves the CUDA graphs and runs an eager rollout
with autograd enabled. That switch applies to the batch, not to individual
rows, so the cost of a firing round is paid once per round regardless of how
many requests are updated in it. Synchronized firing keeps the number of
eager rounds at one in `freq`; giving requests different phases would make
most rounds eager.

Between firing rounds the decode loop is unchanged. The rollout replays its
captured graph, and because the optimizer writes A and B in place, the replay
uses the updated adapters without recapture.

**Asynchronous stepping** (default; `--speculative-eagle-tts-no-async`
disables it). Autograd runs backward nodes on the stream their forward ran on,
so backward is issued on the main stream. The side stream is then made to wait
for the main stream, and the optimizer step (gradient reduction, clipping,
Adam, write-back) is issued on the side stream. The gradients and the slot
index tensor are registered with `record_stream` so their memory is not reused
while the side stream reads them. Immediately before the next draft forward,
including graph replays, the host synchronizes on a CUDA event recorded after
the step, so no forward reads partially written adapters.

## Tensor parallelism

SGLang's TP layers have no backward. `tts_grad_mode` installs the missing
pieces for the duration of a training forward: an identity backward for
all-reduce, a slice backward for all-gather, and an all-reduce backward for
the replicated input of column-parallel and vocab-parallel layers. Every rank
computes the same loss.

Adapter layout follows the wrapped layer:

| layout | targets | A | B |
|---|---|---|---|
| column-parallel / vocab-parallel | `q`, `k`, `v`, `gate_up`, `lm_head` | replicated | sharded |
| row-parallel | `o`, `down` | sharded | replicated |
| replicated | `fc` | replicated | replicated |

- For row-parallel layers the input is sharded, so each rank holds a partial
  `x @ A^T`. The `[*, rank]` intermediate is all-reduced in the forward, which
  is cheaper than reducing the `[*, out]` result.
- For column-parallel and vocab-parallel layers the gradient of the replicated
  A depends on the local shard of B and is a per-rank partial sum. It is
  all-reduced before the optimizer step.
- The clip norm is assembled so that it is identical on every rank: sharded
  tensors contribute their shard, replicated tensors contribute `1 / tp_size`
  each, and the sum of squares is all-reduced.

`--enable-dp-attention` is rejected with TTS, because the collectives it uses
are not the ones made differentiable here.

## Memory per slot

Per TP rank, one slot of one adapter holds `rank * (in_local + out_local)`
elements, where `in_local` / `out_local` are the local (possibly sharded)
dimensions from the table above. The resident state per element depends on the
optimizer dtype (see [Precision](#precision)):

| tensor | `float32` | `bfloat16` |
|---|---|---|
| pool parameter (forward), bf16 | 2 | 2 |
| master weight | 4 (fp32) | 0 (the pool parameter) |
| Adam first and second moment | 8 (fp32) | 4 (bf16) |
| total, bytes per element | 14 | 6 |

The resident total is `bytes per element * elements per slot * P`. Once updates
run, gradient tensors with the shape of the pool parameters are allocated in
addition. A step also holds float32 copies of the fired rows, of one tensor at a
time, in both modes. The startup log line
`EAGLE3 LoRA-TTS: ... params/slot x ... slot(s)` reports the element counts and
the resident size per rank for the configured targets, pool and optimizer dtype.

For `AngelSlim/Qwen3-8B_eagle3` at rank 16 and `tp_size = 4`, per rank:

| targets | elements per slot | `float32` per slot | `bfloat16` per slot | `float32` for `P = 65` | `bfloat16` for `P = 65` |
|---|---|---|---|---|---|
| `o` | 81,920 | 1.15 MB | 0.49 MB | 75 MB | 32 MB |
| `q,o,gate_up,down,lm_head` | 701,440 | 9.8 MB | 4.2 MB | 638 MB | 274 MB |

The unpooled paths cost, per element, 14 bytes against 6 for shared LoRA and,
with its snapshot, 16 against 8 for `full` mode (weight, snapshot, master, moments;
a tensor that never receives a gradient has no moments). During a step the
`bfloat16` unpooled paths hold in addition float32 leaves and gradients, 8 bytes
per element of all trained tensors, and the float32 moments of one tensor at a
time. The `float32` paths hold the float32 gradients, 4 bytes per element.

## Flags and environment variables

| flag | default | meaning |
|---|---|---|
| `--speculative-eagle-tts-enable` | off | enable TTS for the EAGLE / EAGLE-3 draft |
| `--speculative-eagle-tts-mode {full,lora}` | `full` | update the draft's own weights, or LoRA adapters with the base frozen |
| `--speculative-eagle-tts-per-request` | off | one adapter slot per request (requires `lora`; the pool is sized from `--max-running-requests`) |
| `--speculative-eagle-tts-lora-targets` | `q,o,gate_up,down,lm_head` | projections that carry an adapter |
| `--speculative-eagle-tts-lora-rank` | 16 | adapter rank |
| `--speculative-eagle-tts-lora-alpha` | 32.0 | adapter scale is `alpha / rank` |
| `--speculative-tts-loss {lk,kl}` | `lk` | objective, for every speculator |
| `--speculative-tts-lambda` | 0.9 | KL / self-CE mixing weight of the `kl` loss |
| `--speculative-eagle-tts-lr` | 1e-5 | Adam learning rate |
| `--speculative-eagle-tts-freq` | 10 | update stride, in decode rounds |
| `--speculative-eagle-tts-depth-decay` | 0.8 | per-depth loss weight `decay ** depth` |
| `--speculative-eagle-tts-accum-steps` | 1 | firing rounds accumulated per optimizer step; above 1 is rejected with `--speculative-eagle-tts-per-request` |
| `--speculative-eagle-tts-grad-clip` | 1.0 | gradient-norm clip, per slot in pooled mode (0 disables) |
| `--speculative-eagle-tts-kl-threshold` | 0.0 | skip the update when the divergence term is below this (0 disables) |
| `--speculative-eagle-tts-no-async` | off | run the optimizer step synchronously |
| `--speculative-eagle-tts-optimizer-dtype {float32,bfloat16}` | `float32` | precision in which the masters and moments are stored; see [Precision](#precision) |

| environment variable | meaning |
|---|---|
| `SGLANG_ENABLE_TORCH_INFERENCE_MODE` | must not be enabled; TTS needs gradients through the draft forward |

The batched experiments in the paper use `--speculative-eagle-tts-mode lora
--speculative-eagle-tts-per-request --speculative-eagle-tts-lora-targets o
--speculative-tts-loss lk --speculative-eagle-tts-lr 1e-4
--speculative-eagle-tts-freq 10` with asynchronous stepping and all drafted
positions in the loss.

## Limitations

- **Chain drafts only.** `--speculative-eagle-topk` must be 1. The pairing of
  rollout steps with target rows is not defined for a draft tree.
- **Spec-v2 worker only.** TTS is integrated in `EAGLEWorkerV2`
  (`eagle_worker_v2.py`), which SGLang selects by default at topk 1. The v1
  worker and the multi-layer EAGLE workers do not run TTS.
- **Unsupported combinations.** `--enable-dp-attention` and
  `--speculative-adaptive` are rejected, as is
  `SGLANG_ENABLE_TORCH_INFERENCE_MODE`.
- **The first drafted token is not trained.** The rollout steps cover tree
  nodes 2 and later. The proposal for node 1 comes from the previous round's
  draft-extend pass and is not part of the loss.
- **Firing rounds are eager for the whole batch.** The rollout cannot replay
  its CUDA graph on a firing round, for any request in the batch.
- **Single-layer drafts.** Adapters are installed on every layer of a
  multi-layer draft, but the attention and KV-cache reasoning behind the
  target table assumes one layer.
- **`k`, `v` targets** leave the draft KV cache inconsistent across an
  update and are not recommended.
- **`fc` does not train.** `fc` and its input norms run only in the
  draft-extend pass, which the update does not differentiate, so no gradient
  reaches them: an adapter on `fc` stays at zero, and `full` mode leaves
  these weights at their pretrained values.
- **Differentiable attention** (needed for `q`, `k`, `v`, `fc` and for `full`
  mode) requires the draft's KV pool to be `MHATokenToKVPool`.
- **Norm training** applies only without the pool, so the pooled and unpooled
  configurations do not update exactly the same parameter set.
- **`full` mode** is single-request: its state is shared by the worker and
  restored at every prefill.

## Tests

`test/registered/spec/eagle_tts/` covers the pieces described here; see its
README. In particular `test_pool_gather.py` exercises the per-row gather and
the row selector, `test_pool_adam.py` the pooled optimizer, the per-request
loss reduction and the per-slot clip, `test_lora_grad_tp.py` the TP gradient
consistency, and `test_bf16_optimizer.py` the `bfloat16` optimizer dtype on all
three optimizer paths.
