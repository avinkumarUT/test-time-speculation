# DFlash LoRA-TTS: design

This document describes how test-time speculation (TTS) is implemented for
the DFlash speculator in batched serving. The code is in
`speculative/dflash_lora_tts.py` (adapter state, updates),
`speculative/dflash_worker.py` (integration into the speculate/verify loop)
and `models/dflash.py` (the serving hook in the draft model).

## Overview

In speculative decoding, the target model verifies every drafted token, and
the verification pass yields the target's distribution at each drafted
position. TTS uses that signal to update the draft model online, while the
response is being generated, so that later drafts are accepted more often.

DFlash is a block-diffusion style speculator: each round it drafts one block
of tokens (16 for `z-lab/Qwen3.6-27B-DFlash`) in a single non-causal forward
pass. In batched serving, one request's update must not change the draft
model that other requests use. The draft model's weights therefore stay
frozen and shared, and each in-flight request owns a set of LoRA adapters on
the draft model's linear projections. An update is one or more Adam steps on
a request's adapters, distilling the target's verify distribution for a block
that the request has just drafted. For a single in-flight request, the
full-weight mode described in `DFLASH_TTS_DESIGN.md` updates the draft
model's own weights instead.

Goals:

* Per-request adaptation with a small memory cost per request.
* No change to the serving path other than an additive low-rank delta: the
  fused attention kernels, the fused QKV projection and draft CUDA graphs
  stay in use.
* Updates that stay off the critical path: they run every `S` rounds (the
  update stride) on a side CUDA stream and overlap the following rounds.
* Tensor parallelism (`tp_size >= 1`).
* Correctness of the output is never at stake: the target model still
  decides every token. A failed or skipped update only costs acceptance
  length.

## Background: which projections are adapted

Every position that the DFlash draft model attends to as context holds K/V
projected from the *target's* hidden states: `project_target_hidden`
(`fc`, then `hidden_norm`) followed by `kv_proj_only`, `k_norm` and RoPE
writes them into the draft KV cache. The draft model does not attend to its
own token representations for context.

The projections fall into three groups:

| Projection | Role | Adapted |
|---|---|---|
| `fc`, `hidden_norm` | Run only when context K/V is written. There is no gradient path from the loss. | Never |
| `k_proj`, `v_proj`, `k_norm` | Two roles: project target hidden states into context K/V, and produce the block's own K/V. The loss observes only the second role. | Not by default (see [Adapting k and v](#adapting-k-and-v)) |
| `q` (the Q slice of `qkv_proj`), `o` (`o_proj`), `gate_up` (`gate_up_proj`), `down` (`down_proj`) | Used only in the block forward. | Yes (default targets) |

With the default targets, the context K/V in the draft cache is produced by
weights that never change. Three properties follow:

* cached prefix K/V stay valid across updates, so no KV rewrite is needed;
* the serving attention kernel is unchanged;
* an update can read the prefix K/V back from the cache as constants.

## Data flow of one round

```
main stream                                            side stream
-----------                                            -----------
fence (wait for in-flight updates)
append committed tokens to the draft KV cache
capture replay inputs of the requests due for an update
publish the batch's adapters to the static buffers
draft forward (base weights + per-row LoRA delta)
target verify forward
snapshot the verify logits of the captured rows
verify (accept / reject, bonus token)
gather prefix K/V of the captured requests
enqueue the update  ---------------------------------> replay forward, loss,
advance cadence counters, release finished requests    backward, clip, Adam,
next round ...                                         refresh pool rows
```

A request "fires" on the rounds in which its adapters are updated. All other
rounds only publish and draft.

## Per-request state

State is keyed by the request's slot (`req_pool_idx`) and is created lazily
the first time the request is published or captured. For each targeted
`(layer, projection)`:

```
A: [rank, in_features]    copied from a fixed per-(layer, projection) template
                          (uniform init, fixed seed)
B: [out_features, rank]   zeros
```

`B = 0` makes a new adapter exactly inert. As in standard LoRA, the gradient
of `A` is zero while `B` is zero, so `A` starts to move one step after `B`.
The template makes a request's initial adapters independent of the order in
which requests arrive.

The matrices are stored as slices of four flat tensors per request (fp32 by
default; see [Precision](#precision)), grouped by their role under tensor parallelism (`a_repl`, `b_col`, `a_row`,
`b_repl`; see [Tensor parallelism](#tensor-parallelism)). The optimizer,
gradient sanitizing, clipping and collectives then handle four tensors per
request instead of one per matrix. Adam is elementwise and all tensors of a
request step together, so this layout does not change the update. Tensors of
different requests are never merged, because their step counts differ.

Each request also has Adam moments, a round counter and step counter, an
update phase, and one row in the serving pools.

Lifecycle:

* **Prefill.** `reset()` clears any state left in the slot by a previous
  request.
* **After each verify round.** `advance()` increments the round counter of
  every request in the batch. `drop()` releases the state of requests that
  finished in this round (they are also skipped by the round's update).
  `prune_except()` releases the state of requests that left the batch
  without finishing (aborted or retracted).
* **Flush / sleep.** All state and the pools are released.

## The adapter pool and serving

### Pool rows

The adapters that serving reads are snapshots in the serving dtype, held in
state-wide pools `{(layer, projection): [capacity, rank, in]}` and
`{...: [capacity, out, rank]}`. Each live request owns one row, taken from a
free list. Rows are not indexed by request slot: SGLang cycles through its
whole slot space, so slot-indexed pools would grow to the number of slots
rather than the number of live requests. A row is written when the request
is created and after each of its updates. The pools grow geometrically with
the peak number of live requests.

### Static buffers and the serving hook

The draft model reads adapters from *static buffers*
(`models/dflash._StaticLoRABuffers`): tensors
`a[(layer, projection)]: [max_bs, rank, in]` and
`b[(layer, projection)]: [max_bs, out, rank]`, where row `r` belongs to batch
position `r`. The worker allocates them once at startup, before the draft
model is constructed, from the draft model's config. Inside
`DFlashAttention.forward` and `DFlashMLP.forward`, each targeted projection
adds

```
x3    = x.view(bs, block, in)
delta = bmm(bmm(x3, A[:bs].transpose(1, 2)), B[:bs].transpose(1, 2)) * alpha / rank
y     = y_base + delta.view(bs * block, out)
```

The delta is applied only when the token count is exactly `bs * block` and
`bs <= max_bs`, so other forwards through the same modules (such as the
context-KV projection) are unaffected.

Before every draft forward, `publish_static` copies the batch's pool rows
into the buffers with one `index_select` per `(layer, projection)` and zeroes
any rows left over from a larger batch.

### CUDA graphs

Because the buffers exist before the draft model's CUDA graphs are captured
and keep their addresses, the captured graphs contain the delta computation
reading those buffers. Publishing only rewrites buffer contents, so draft
graphs stay enabled with TTS. A zero row contributes a zero delta, which
covers:

* padding rows, when a batch runs in a graph captured for a larger batch
  size;
* requests that have not been updated yet (`B = 0`);
* the state after TTS has been disabled or flushed (the rows are zeroed,
  since the delta cannot be removed from a captured graph);
* a round that serves without adapters after a publish failure.

(A zero delta leaves values unchanged. Adding `+0.0` can turn a `-0.0` into
`+0.0`, which compares equal.)

Buffer shapes are computed from the model config and cross-checked against
the loaded modules at init (`adopt_static_buffers`); a mismatch, or a failure
to allocate the buffers, is a startup error. `max_bs` is derived from
`--max-running-requests` (falling back to the draft's `cuda_graph_max_bs`,
then to 160), capped at 256 and rounded up to a multiple of 8 plus 8 rows of
headroom. The
draft's list of captured batch sizes is capped to match, and startup fails if
a captured size still exceeds the buffers. A batch larger than `max_bs` is
served without adapters (the decision is the same on every rank, and a
warning is logged once).

`--speculative-dflash-lora-tts-no-draft-graphs` runs the draft eagerly over
the same buffers. With `--speculative-dflash-lora-tts-freq 0` no buffers are
installed and the hook is absent; the draft then behaves as it does without
TTS.

## Computing an update

### Capture

On a round in which a request fires, the worker captures, before the draft
forward:

* the block's input embeddings `[block, hidden]` (cloned before the forward,
  because the serving fused add-RMSNorm modifies them in place),
* the block's positions `[block]`,
* the request's prefix locations in the draft KV cache.

After the target's verify forward and before `verify()` modifies the logits
in place, the worker copies the verify logits of the captured rows. Only rows
`0..block-2` are copied: the target's row `k` is the distribution for block
position `k+1`, and the last row only produces the bonus token. The copy is
taken before sampling-related logit changes, so updates always distill the
target's unmodified distribution, for greedy and non-greedy verification
alike.

### Replay forward

The update re-runs the draft block forward eagerly with autograd
(`_replay_block_hidden` and its batched form). It follows
`DFlashDraftModel.forward`, with the native RMSNorm, RoPE and
`scaled_dot_product_attention` implementations in place of the fused serving
kernels, and reads the draft model's frozen weights in place. The request's
fp32 leaves are cast to the compute dtype inside the graph, so that the replay
matches what serving computed while the update itself runs in fp32.

Prefix K/V are read from the draft KV cache at the captured locations and
treated as constants. They are gathered on the main stream when the update is
enqueued, so the side stream never reads the KV pool. (SGLang's
`release_memory_occupation` can unmap KV memory before the TTS fence runs; an
update that read the pool lazily could then touch unmapped memory.)

### Loss

The block is non-causal: draft hidden state `k` predicts block position `k`,
and position 0 holds the already-verified bonus token. Draft rows
`1..block-1` are therefore scored against target rows `0..block-2`. Draft
logits are computed with the target's `lm_head` (detached), with the target's
final logit soft-capping applied if it has one.

`--speculative-tts-loss` selects the objective:

```
lk (default):  -log( sum_x min(p_target(x), q_draft(x)) )
kl:            lambda * KL(target || draft) + (1 - lambda) * CE(draft, argmax(draft))
```

`sum_x min(p, q)` is the speculative acceptance probability. Each position is
weighted by `w_k = exp(-k / gamma)`, normalized over the block. The KL is computed as
`sum(p_t * log p_t) - sum(p_t * log p_d)`; the first term is a constant of
the capture and is computed once.

### Gradient step

1. `backward()`.
2. Non-finite gradient elements are set to zero. This precedes clipping,
   because a single NaN would make the clipping norm NaN.
3. With `tp_size > 1`, the gradients of replicated `A` matrices are completed
   with an all-reduce (see [Tensor parallelism](#tensor-parallelism)).
4. The gradient norm is clipped per request (`--speculative-dflash-lora-tts-grad-clip`).
5. One Adam step per request.

`--speculative-dflash-lora-tts-steps N` repeats the forward and the gradient
step `N` times on the same capture.

Non-finite losses: in synchronous mode a step whose loss is not finite is
skipped (with `tp_size > 1` the decision is made collectively, so all ranks
agree). In asynchronous mode the check is omitted, because reading the loss
would synchronize the side stream; step 2 then turns such a step into a
near-no-op, and the number of non-finite losses is counted on the GPU and
logged when TTS is flushed or disabled.

### Batching several requests

Requests that fire in the same round are replayed together, in chunks of up
to `--speculative-dflash-lora-tts-replay-chunk` (default 8). Base linears, norms,
RoPE and the `lm_head` run once on the flattened `[R * block, ...]` rows, and
the LoRA deltas use a stacked `bmm` over the per-request adapters. Attention
is bucketed by prefix length: requests with equal prefix length share one
batched attention call.

This is equivalent to updating the requests one at a time, up to
floating-point reassociation. The total loss is the sum of the per-request
losses, and each request's loss depends only on its own adapters, so the
batched backward yields per-request gradients with no cross terms. Clipping
and Adam remain per request. Chunking bounds the transient memory of the loss
and does not change the result. A chunk of one request uses the unbatched
code path.

Deterministic inference mode replaces `torch.bmm` with a Python-level
function that bypasses autograd. The batched replay therefore calls
`torch.ops.aten.bmm`, which keeps gradients and uses the same kernel.

### Failure handling

With `tp_size == 1`, a failure in capture, publish, the logits snapshot or
the update is contained: the round serves without adapters or skips the
update, and a warning is logged once. After three consecutive failed rounds
TTS is disabled for the lifetime of the server and all adapter state is
freed. With `tp_size > 1` such failures are raised instead (see
[Tensor parallelism](#tensor-parallelism)).

## Scheduling updates

### Stride and phase stagger

`--speculative-dflash-lora-tts-freq S` sets the update stride: each request
is updated once every `S` verify rounds. A heavier speculator such as DFlash
favors a larger stride.

Requests admitted in the same prefill have the same round counter. If they
all fired when the counter is a multiple of `S`, the whole batch would be
updated on the same round: the peak memory of the update and the latency of
that round would grow with the batch size, and the other rounds would do no
TTS work. By default the rounds are staggered: a request's phase is
`req_pool_idx % S`, and it fires when `round_ct % S == phase`. About
`batch_size / S` requests then fire per round. The first update of a request
honors the phase as well, so it can come up to `S - 1` rounds after
admission. `--speculative-dflash-lora-tts-no-phase-stagger` turns staggering
off.

The round counter advances for every request on every verify round, whether
or not the request fired.

### Asynchronous execution

By default a round's updates are enqueued on a dedicated side CUDA stream and
overlap the next round's serving (`--speculative-dflash-lora-tts-no-async`
runs them synchronously on the main stream). The computation is the same in
both modes. The following rules keep the two streams consistent:

* **Fence.** After a round's update has been enqueued, one CUDA event is
  recorded on the side stream (in a `finally` block, so that partially
  enqueued work is covered too). The fence (`wait_for_pending()`)
  synchronizes on the outstanding events from the host. It runs at the start
  of every decode round and again before publishing, before a prefill,
  before the next update is enqueued, and in `disable()`, flush and the
  weight-update entry points. Adapter state is read or reassigned only
  behind this fence.
* **Stream order.** Before each chunk is enqueued, the side stream waits for
  the main stream, so it sees the captures, the logits snapshot and the
  gathered prefix K/V.
* **Tensor lifetime.** Main-stream tensors that the side stream consumes
  (captured embeddings and positions, gathered K/V, the logits snapshot) are
  marked with `record_stream`, so the allocator does not reuse their memory
  while the side stream reads them.
* **Deferred frees.** Releasing a request's state does not fence. If an
  update may still be writing the state, it is parked and freed at a later
  fence, so that a finishing request does not stall the round.
* **Collectives.** With `tp_size > 1`, the start-of-round fence also keeps
  serving collectives on the main stream from interleaving with update
  collectives on the side stream on the same communicator.

With this scheme the host can block in the fence only in the round after an
update.

## Delayed publish

`--speculative-dflash-lora-tts-delayed-publish` (off by default) changes when
an update takes effect: adapters trained in round N are used for drafting
from round N+2 instead of N+1. Without it, round N+1's publish reads the pool
rows that round N's update writes, so the host must wait for that update at
the start of round N+1. With it, the update has a full serving round to
finish and does not block the next round. The option requires asynchronous
updates.

* **Double-buffered pool rows.** The pools hold two generations per row (a
  flat `[2 * capacity, ...]` layout). Each update writes the request's
  *other* generation, never the one that publish can currently read, and
  then flips the request's front generation. The flip is per request and per
  update rather than tied to round parity: with an even stride a request
  fires on one parity only, and a parity-indexed buffer would never be
  refreshed.
* **Lag-1 fence.** Step events are queued with an enqueue sequence number.
  The per-round fence synchronizes all events except the newest, which have
  normally completed already. Publish then reads, for each request, the
  newest generation whose writing update has been fenced. A new request's
  initial snapshot is written to both generations.
* **Visibility.** With `tp_size == 1` the fence also polls the newest event
  without blocking; if that update has already finished, its result is used
  in the next round. With `tp_size > 1` the poll is skipped, because its
  result depends on device timing and every rank must select the same
  generation. Visibility is then determined by event order alone.
* **Full fence.** Pool growth, `disable()`, flush, the weight-update entry
  points and prefill still wait for all updates. Freed pool rows and
  pinned-CPU blocks are parked until the update that could write them has
  been fenced, and only then reused.
* **Dedicated communicator with `tp_size > 1`.** Under the lag-1 fence,
  serving collectives run while the previous round's update collectives may
  still be executing. To keep the two streams off one NCCL communicator, a
  second communicator over the same TP ranks is created at init and all
  update collectives use it. This requires the `torch.distributed` world to
  be exactly the TP group; otherwise delayed publish is turned off with a
  warning.
* **Cost.** Pool-row memory doubles, and drafting uses adapters that are one
  round older. The latter matters little when the stride is large, since
  adapters then change only once every `S` rounds.

## Tensor parallelism

Adapters are sharded like their base layers:

* **Column-parallel targets (`q`, `k`, `v`, `gate_up`).** `A` is replicated
  (`[rank, in]`) and `B` is sharded (`[out_local, rank]`). The serving delta
  is local to each rank. In the backward pass each rank sees only its own
  `B` shard's contribution to the gradient of `A`, so the gradients of `A`
  are all-reduced after backward.
* **Row-parallel targets (`o`, `down`).** `A` is sharded
  (`[rank, in_local]`) and `B` is replicated (`[out, rank]`). The rank-`r`
  intermediate `z = A x_local` is a partial sum and is all-reduced, in both
  serving and replay. The gradients of `A` and `B` are then complete without
  further collectives.

The replay makes the collectives differentiable in the Megatron style: an
operator that is the identity forward and an all-reduce backward on
replicated activations entering a column-parallel projection, and an
all-reduce forward with identity backward on row-parallel partial sums. The
same identity-forward operator is applied to the input of the vocab-sharded
`lm_head`; the local logits are then gathered to the full vocabulary with a
differentiable all-gather. The `lm_head` shards must be equal and unpadded.

Gradient clipping uses a global norm: squared norms of sharded tensors are
all-reduced and those of replicated tensors are counted once, so every rank
applies the same scale. The `A` templates are drawn as one global tensor from
a fixed seed and each rank takes its slice. When several requests are
updated together, their collectives are coalesced: one all-reduce completes
the `A` gradients of the whole chunk and one carries all clipping norms.

Every rank must enqueue the same sequence of collectives. All decisions that
affect that sequence (which requests fire, chunk boundaries, attention
buckets, skipping a non-finite step, the published generation) are derived
from state that is identical on every rank or are made with a collective. For
the same reason, failures in the TTS path are raised with `tp_size > 1`
rather than contained: a rank that skipped work while its peers continued
would leave the collectives out of step. If a failure occurs on one rank
only, that rank raises and the others may block in their next collective, so
the server has to be restarted.

## Precision

The adapters are served in the draft's dtype (bfloat16 for the published drafts)
from the pool rows. An update always computes in float32: its leaves, gradients
and moment workspaces are fp32, with the clipping and the collectives described
above. `--speculative-dflash-lora-tts-optimizer-dtype` selects the dtype in which
a request's masters and moments are stored between updates.

### `float32` (default)

The masters and moments are fp32 and an update stores its fp32 results exactly.
The pool row is a rounded copy of the master. An update smaller than the spacing
of a served adapter weight stays in the master, and the served weight moves once
the accumulated change crosses a rounding boundary.

### `bfloat16`

The masters and moments are stored as bfloat16 blocks. One update is:

1. **Materialize.** The master blocks are copied to the GPU and widened to fp32
   leaves. With offload the transfer from the pinned block carries the block's
   own bytes, and the widening runs on the update's stream. The cast is exact.
2. **Step.** The moments are loaded the same way when first used, and the update
   runs with the fp32 arithmetic of the `float32` mode: backward, gradient
   sanitizing, completion, clipping and torch's functional Adam. All inner steps
   of `--speculative-dflash-lora-tts-steps` carry on in fp32 on the leaves and
   workspaces.
3. **Store.** The results are rounded once per update, on the GPU, and copied into
   the blocks. The refresh writes the same rounded values into the pool row, so
   serving and the next update's starting point hold the same bits.

A request's initial `A` is its fp32 template, rounded. A leaf group that received
no gradient, an update whose loss was not finite and a failure before the first
Adam step all store back the bits they started from, because widening and
rounding are exact inverses on bfloat16 values. Phase stagger and delayed publish
do not depend on the state's dtype: an update still reads its blocks and writes
its results on the same stream, in the same order. Recycled blocks stay gated by
`moments_valid`.

The state's location is independent of its dtype. With
`--speculative-dflash-lora-tts-no-offload` the blocks are bfloat16 GPU tensors and
an update runs the same way without the host transfers. (The `float32` mode without
offload holds fp32 GPU Parameters and a resident `torch.optim.Adam`; a resident
optimizer over bfloat16 Parameters would round after every operation of an update
instead of once, which is a different update.) Results are bit-identical with
and without offload, and for synchronous, asynchronous and delayed-publish
execution.

The cost of the mode is resolution.

**Weights.** An update smaller than half the spacing of the weight it is added to
is rounded away, and is not remembered anywhere. Across the inner steps of one
update the fp32 leaves do accumulate; the rounding applies to what the update
stores. Adam's steps have a magnitude of about the learning rate, so a learning
rate has a weight magnitude above which its steps are lost:

| learning rate | steps are lost from about `|w|` = |
|---|---|
| 1e-3 | 0.5 |
| 1e-4 | 0.031 |
| 1e-5 | 0.0039 |

`B` starts at zero and `A` is drawn with magnitude at most `1 / sqrt(in_features)`,
0.014 for hidden size 5120. At the default learning rate of 1e-4 their updates
survive, with a resolution of one or a few spacings, until a weight has grown to
about 0.03.

**Second moment.** Without a new gradient `v` changes by `(1 - beta2) v` per step,
0.1% of `v` with `beta2 = 0.999`. Half the spacing of a bfloat16 value is 0.2% to
0.4% of its magnitude, so this decay rounds away and `v` does not decay. `v` grows
only while the increment `0.001 * g^2` exceeds half its spacing. A large gradient
therefore lowers the step size for the rest of the request (a spike leaves `v`
unchanged through any later small gradients, where float32 decays it to 0.7% in
5000 steps), and under a constant gradient `v` stops at a fifth to a third of
`g^2`, depending on where the value lies between two powers of two, so that after
several thousand steps Adam's update is about twice as large as with float32 (the
same at first, about 1.6 times at 1000 steps). The first moment is not
affected: it changes by 10% per step.

`bfloat16` is accepted only for a bfloat16 draft model; this is checked at startup.
Serving reads the pool rows and the next update widens the stored masters, and
they hold the same values only when the pool has the dtype of the state.

## Memory accounting

Let `P` be the number of adapter parameters per request: the sum of
`rank * (in + out)` over the targeted `(layer, projection)` pairs. `P` is
logged at startup as `trainable_params_per_req`.

| Item | Size | Where |
|---|---|---|
| Masters | `4 P` bytes per request (`2 P` with the `bfloat16` optimizer dtype) | pinned CPU (GPU with `--speculative-dflash-lora-tts-no-offload`) |
| Adam moments | `8 P` bytes per request (`4 P` with the `bfloat16` optimizer dtype) | pinned CPU (GPU with `--speculative-dflash-lora-tts-no-offload`) |
| Serving pool row | `P * sizeof(serving dtype)` per live request; twice that with delayed publish | GPU |
| Static buffers | `max_bs * P * sizeof(serving dtype)` | GPU, allocated at startup |
| Update workspace | With offload, or with the `bfloat16` optimizer dtype, about `16 P` bytes per request being updated (fp32 leaves, gradients and moments), in both optimizer dtypes, plus with `bfloat16` a bfloat16 staging copy of one leaf group at a time; in addition the replay's activations, the gathered prefix K/V and `block * vocab` fp32 target probabilities | GPU, transient |

For example, `z-lab/Qwen3.6-27B-DFlash` (5 layers, hidden size 5120) at rank
16 with the default targets and `tp_size = 1` has `P` = 6,471,680. With
tensor parallelism the per-rank sizes are smaller, since sharded matrices are
divided across ranks (`P` = 2,846,720 per rank at `tp_size = 4`):

| per request | `float32` | `bfloat16` | `float32`, `tp_size = 4`, per rank | `bfloat16`, `tp_size = 4`, per rank |
|---|---|---|---|---|
| Masters | 24.7 MiB | 12.3 MiB | 10.9 MiB | 5.4 MiB |
| Adam moments | 49.4 MiB | 24.7 MiB | 21.7 MiB | 10.9 MiB |
| Stored state, total | 74.1 MiB | 37.0 MiB | 32.6 MiB | 16.3 MiB |
| Host-GPU transfers per update, offloaded (state read and written) | 148 MiB | 74 MiB | 65 MiB | 33 MiB |
| Serving pool row (bfloat16) | 12.3 MiB | 12.3 MiB | 5.4 MiB | 5.4 MiB |

At 64 live requests the pinned state is 4.6 GiB against 2.3 GiB per rank at
`tp_size = 1`, and 2.0 GiB against 1.0 GiB per rank at `tp_size = 4`.

With offload (the default), a request's masters and moments are in pinned CPU
memory between updates and on the GPU only during its update, so the
steady-state GPU cost of a request is its pool row. The pinned blocks are
recycled across requests; up to 32 freed blocks are kept for reuse. The
transient update memory is bounded by the replay chunk size, and the
prefix-K/V gather grows with the context length (windowed layers gather at
most their window).

These allocations are separate from the KV cache budget, so
`--mem-fraction-static` must leave room for them.

## Optional modes

### Adapting k and v

`--speculative-dflash-lora-tts-targets q,k,v,o,gate_up,down` adds adapters on
`W_k` / `W_v`. One adapter per `(layer, k/v)` is then applied wherever the
weight is used: in the block forward (before `k_norm`) and in the context-KV
append, where each request's appended tokens are projected with that
request's adapters. Two consequences:

1. Cached context K/V is not rewritten when adapters change, so each cached
   token keeps the adapter state of the time it was appended.
2. The replay reads cached prefix K/V as constants, so gradients reach the
   k/v adapters only through the block forward; their effect on context K/V
   is not part of the loss.

This mode requires `--disable-radix-cache`, because context K/V then depends
on the request. The decode-round context append uses the non-fused
projection path; the prompt's append at prefill keeps the fused path, since
a new request's delta is zero.

### Layer coverage

`--speculative-dflash-lora-tts-layers` restricts adapters to a subset of
draft layers (`last<K>`, `first<K>`, or a list of layer ids). Layers outside
the subset have no adapter state and run as frozen base layers.

### Sliding-window draft attention

DFlash draft checkpoints can be trained with sliding-window attention on some
layers (for example, `z-lab/Qwen3.6-27B-DFlash` uses a 2048-token window on 4
of its 5 layers). With TTS enabled, those layers are served with their trained
window, and the draft uses the triton attention backend.

The replay reads each layer's serving window, so serving and replay always
agree. For a windowed layer the replay gathers only the last
`min(prefix_len, window)` prefix locations and applies the same per-row mask
as the serving kernel, which also bounds the size of the prefix-K/V gather.
The window must not be shorter than a block.

A windowed draft with CUDA graphs on a target that uses a hybrid
sliding-window KV pool is rejected at startup; use
`--speculative-dflash-lora-tts-no-draft-graphs` or `--disable-cuda-graph` in
that configuration.

## Configuration

| Flag | Default | Meaning |
|---|---|---|
| `--speculative-dflash-lora-tts-enable` | off | Enable TTS for DFlash. |
| `--speculative-dflash-lora-tts-freq` | 1 | Update stride `S`. `<= 0` never updates. |
| `--speculative-dflash-lora-tts-rank` | 16 | LoRA rank. |
| `--speculative-dflash-lora-tts-alpha` | 32.0 | LoRA alpha; the delta is scaled by `alpha / rank`. |
| `--speculative-dflash-lora-tts-lr` | 1e-4 | Adam learning rate. |
| `--speculative-tts-loss {lk,kl}` | `lk` | Objective, for every speculator. |
| `--speculative-tts-lambda` | 0.9 | Weight of the KL term of the `kl` loss. |
| `--speculative-dflash-lora-tts-steps` | 1 | Adam steps per update. |
| `--speculative-dflash-lora-tts-pos-weight-gamma` | 7.0 | Decay of the per-position loss weights. |
| `--speculative-dflash-lora-tts-grad-clip` | 1.0 | Per-request gradient-norm limit; `<= 0` disables clipping. |
| `--speculative-dflash-lora-tts-targets` | `q,o,gate_up,down` | Projections that carry adapters. |
| `--speculative-dflash-lora-tts-layers` | all | Draft layers that carry adapters. |
| `--speculative-dflash-lora-tts-delayed-publish` | off | Use updated adapters from round N+2. |
| `--speculative-dflash-lora-tts-no-async` | async on | Run updates synchronously on the main stream. |
| `--speculative-dflash-lora-tts-no-phase-stagger` | stagger on | Update all requests on the same rounds. |
| `--speculative-dflash-lora-tts-no-offload` | offload on | Keep masters and Adam moments on the GPU. |
| `--speculative-dflash-lora-tts-optimizer-dtype` | `float32` | Dtype in which masters and Adam moments are stored: `float32` or `bfloat16`. See [Precision](#precision). |
| `--speculative-dflash-lora-tts-no-draft-graphs` | graphs on | Run the draft model eagerly. |
| `--speculative-dflash-lora-tts-replay-chunk` | 8 | Maximum number of requests replayed together. Lower it to reduce the transient memory of an update. |

See the repository README for example launch commands.

## Requirements and limitations

Checked at startup or at runtime:

* `--speculative-algorithm DFLASH`, and a block size of at least 2.
* `--kv-cache-dtype auto`: the replay reads the draft KV cache directly.
* Unquantized (fp16 / bf16 / fp32) draft weights and target `lm_head`. An
  unsupported `lm_head` layout (quantized, added vocabulary, or padded or
  unequal shards under TP) disables TTS at runtime with a warning.
* Not combinable with `--enable-lora` / `--lora-paths`.
* k/v targets require `--disable-radix-cache`.
* Delayed publish requires asynchronous updates.
* DFlash's own restrictions apply unchanged (for example, no dp-attention
  and `pp_size == 1`).

Behavior to be aware of:

* Adapters live only as long as their request. Nothing is carried over
  between requests, and a flush or sleep releases all TTS state.
* Batches larger than the static buffers are served without adapters.
* TTS memory is allocated outside the KV cache budget (see
  [Memory accounting](#memory-accounting)). With `tp_size == 1`, updates that
  keep running out of memory disable TTS; with `tp_size > 1` they stop the
  server.
* In-place weight updates (`update_weights_*`): the DFlash worker fences TTS
  before delegating to the target worker, but the scheduler updates the
  target worker before it calls the DFlash worker. An update that is in
  flight at that moment may read partially updated `lm_head` weights for one
  step. Serving is not affected.
* `release_memory_occupation` fences TTS only when it releases the KV cache
  (through the cache flush). Releasing only the model weights does not pass
  through the fence, so an asynchronous update that is still in flight could
  read released weights. Flush the cache first, or run with
  `--speculative-dflash-lora-tts-no-async`.

## Tests

`test/registered/spec/dflash_tts/` contains unit tests that run without a
model; those specific to the LoRA mode are `test_aten_bmm_autograd.py` (the
batched delta stays differentiable under deterministic inference),
`test_offload_adam_parity.py` (the offloaded Adam step is bitwise identical to
a resident `torch.optim.Adam`) and `test_lora_bf16_optimizer.py` (the `bfloat16`
optimizer dtype).
