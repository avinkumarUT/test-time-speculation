"""Test-Time Speculation (TTS) for EAGLE-3 speculative decoding.

In speculative decoding, the target model's verification pass yields the
target's next-token distribution at every drafted position. TTS uses those
distributions to update the EAGLE-3 draft model (the "speculator") online,
while the response is being generated, so the draft adapts to the sequence at
hand and more of its proposals are accepted.

This module is the update engine. `eagle_worker_v2.py` drives it once per
decode round, and `eagle_lora_tts.py` supplies the low-rank parameterization.

Main pieces
-----------
`EagleTTSConfig`
    Options, read from the `--speculative-eagle-tts-*` server arguments.
`EagleTTSState`
    Per-worker state: the update schedule, the loss, the optimizer, the
    asynchronous step, and the reset at the start of each request.
`tts_grad_mode`
    Context manager that makes one draft forward differentiable inside the
    inference-only serving stack.
`privatize_target_aliases`
    Gives the draft private copies of the tensors it shares with the target
    ("full" mode only).

Update modes
------------
"full"
    Updates the draft's own weights: every parameter the rollout
    differentiates, i.e. all of them except the input projection (`fc` and
    its norms), which runs only in the draft-extend pass. The state is per
    worker: the weights are snapshotted at construction and restored at every
    prefill, so this mode serves one request at a time.
"lora"
    Keeps the base weights frozen and updates low-rank adapters. With
    `per_request`, every in-flight request owns one slot of an adapter pool and
    the optimizer (`eagle_lora_tts.PoolAdam`) keeps per-slot state, so
    concurrent requests adapt independently. This is the configuration for
    batched serving.

Objective
---------
For a drafted position with draft distribution q and target distribution p:

    "kl":  lambda * KL(p || q) + (1 - lambda) * CE(q, argmax q)
    "lk":  -log(sum_x min(p(x), q(x)))

`sum_x min(p, q)` is the probability that speculative sampling accepts a token
proposed from q, so "lk" optimizes acceptance directly. Positions are weighted
by `depth_decay ** depth`. When the draft uses a reduced vocabulary, the target
logits are restricted to the draft's columns (`hot_token_id`) before the
softmax. Both distributions are taken at temperature 1. The loss covers every
drafted position, or with `correct_drafts_only` just the drafts the target
accepted.

Schedule
--------
An update fires every `freq` decode rounds (the update stride). On a firing
round the draft rollout runs eagerly under `tts_grad_mode`, so its logits carry
an autograd graph; on every other round decoding is unchanged, including CUDA
graph replay. With `async_step`, backward is issued on the main CUDA stream and
the optimizer step on a side stream. The host waits on a CUDA event immediately
before the next draft forward, the first point at which the updated weights are
read.

Invariants
----------
* The target model stays bit-identical. EAGLE-3 binds the draft's
  `embed_tokens` (and, for drafts without a head of their own, `lm_head`) to
  the target's tensors. "full" mode clones them before updating; "lora" mode
  writes only its adapters and, without per-request slots, the draft's
  `post_attention_layernorm` and final `norm`. `EagleTTSState` refuses to start if any tensor it would
  update shares storage with the target.
* A draft forward is differentiable only inside `tts_grad_mode` and an enclosing
  `torch.enable_grad()`. SGLang is inference-only: model forwards are decorated
  with `torch.no_grad()`, the scheduler loop disables grad globally, the fused
  kernels and attention backends have no autograd registration, and the
  tensor-parallel collectives have no backward. Each of these yields missing or
  zero gradients without an error, so the context manager substitutes a
  differentiable equivalent for each and raises when it cannot.
* With the float32 optimizer dtype, updates accumulate in fp32 master weights
  and are written back to the (bf16) tensors the forward reads. With bfloat16
  the bf16 tensors are their own master, so a step smaller than half the spacing
  of a weight is lost to rounding.
* Under tensor parallelism every rank computes the same loss, and the gradients
  of replicated tensors are made identical across ranks
  (`_ReplicatedInputGrad`, `_reduce_partial_grads`, `_clip_grads`), so the
  ranks keep holding one consistent model.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.speculative.tts_loss import lk_per_position
from sglang.srt.speculative.tts_optim import Bf16StateAdam

logger = logging.getLogger(__name__)


class _DifferentiableAllReduce(torch.autograd.Function):
    """Sum all-reduce with an explicit backward.

    `c10d::allreduce_` has no autograd kernel: backward through it warns and
    yields a zero gradient, which zeroes the gradient of every parameter
    upstream of a `RowParallelLinear` (`o_proj`, `down_proj`) and of
    `VocabParallelEmbedding`.

    For `y = sum_r x_r`, every rank holds the same `y` and, because the loss is
    computed identically on every rank, the same `dL/dy`. Hence
    `dL/dx_r = dL/dy`: backward is the identity and needs no communication.
    This is the `g` half of the Megatron-style `f`/`g` conjugate pair.
    """

    @staticmethod
    def forward(ctx, x, all_reduce_fn):
        return all_reduce_fn(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class _DifferentiableAllGather(torch.autograd.Function):
    """All-gather with an explicit (slice) backward.

    `c10d::_allgather_base_` has no autograd kernel either. For
    `y = concat_r(x_r)` along `dim`, `dL/dx_r` is this rank's slice of `dL/dy`.
    The loss is identical on every rank, so backward needs no communication.
    """

    @staticmethod
    def forward(ctx, x, dim, rank, world_size, all_gather_fn):
        ctx.dim = dim
        ctx.rank = rank
        ctx.world_size = world_size
        ctx.shard = x.shape[dim]
        return all_gather_fn(x, dim)

    @staticmethod
    def backward(ctx, grad_output):
        start = ctx.rank * ctx.shard
        grad = grad_output.narrow(ctx.dim, start, ctx.shard)
        return grad.contiguous(), None, None, None, None


def _patch_differentiable_collectives(stack):
    """Swap the TP collectives for autograd-aware versions during a TTS forward.

    `linear.py`, `logits_processor.py` and `vocab_parallel_embedding.py` each do
    `from sglang.srt.distributed import tensor_model_parallel_all_*` at import
    time, so the names live in those modules' namespaces and patching the
    source module would have no effect. Each consumer is patched instead, and
    the originals are pushed onto `stack` for the caller to restore.
    """
    import sglang.srt.distributed.parallel_state as _pstate
    import sglang.srt.layers.linear as _linear
    import sglang.srt.layers.logits_processor as _logits
    import sglang.srt.layers.vocab_parallel_embedding as _vocab

    # Single-process (unit tests) or TP=1: no collectives on the forward, so
    # there is nothing to make differentiable.
    if _pstate._TP is None:
        return
    world_size = _pstate.get_tensor_model_parallel_world_size()
    if world_size <= 1:
        return
    rank = _pstate.get_tensor_model_parallel_rank()

    def make_all_reduce(orig):
        def all_reduce(input_):
            if not torch.is_grad_enabled() or not input_.requires_grad:
                return orig(input_)
            return _DifferentiableAllReduce.apply(input_, orig)

        return all_reduce

    def make_all_gather(orig):
        def all_gather(input_, dim=-1):
            if not torch.is_grad_enabled() or not input_.requires_grad:
                return orig(input_, dim)
            return _DifferentiableAllGather.apply(
                input_, dim, rank, world_size, orig
            )

        return all_gather

    targets = [
        (_linear, "tensor_model_parallel_all_reduce", make_all_reduce),
        (_linear, "tensor_model_parallel_all_gather", make_all_gather),
        (_logits, "tensor_model_parallel_all_gather", make_all_gather),
        (_vocab, "tensor_model_parallel_all_reduce", make_all_reduce),
    ]
    for module, name, factory in targets:
        orig = getattr(module, name, None)
        if orig is None:
            raise RuntimeError(
                f"EAGLE3 TTS: expected {module.__name__}.{name} to exist; sglang's "
                "collective plumbing has changed and TTS would silently train on "
                "zero gradients."
            )
        setattr(module, name, factory(orig))
        stack.append((module, name, orig))


class _ReplicatedInputGrad(torch.autograd.Function):
    """Megatron's `f`: identity forward, all-reduce backward.

    A column-parallel layer (`qkv_proj`, `gate_up_proj`) and the vocab-parallel
    `lm_head` take a replicated input and produce a sharded output, so each
    rank computes only its own shard's contribution to `dL/d(input)`; the true
    gradient is the sum over ranks. Without this conjugate every gradient
    upstream of such a layer is a per-rank partial sum, each rank optimizes a
    different objective, and the ranks diverge.

    The forward is the identity, so the op costs nothing outside a backward
    pass and can stay in place during CUDA graph capture.
    """

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        from sglang.srt.layers import linear as _linear

        return _linear.tensor_model_parallel_all_reduce(grad_output.contiguous())


def all_reduce_input_grad(x: torch.Tensor) -> torch.Tensor:
    """Apply the `f` conjugate to a replicated input.

    Returns `x` unchanged when TP is inactive, grad is disabled, or `x` does
    not require grad.
    """
    import sglang.srt.distributed.parallel_state as _pstate

    if (
        _pstate._TP is None
        or _pstate.get_tensor_model_parallel_world_size() <= 1
        or not torch.is_grad_enabled()
        or not x.requires_grad
    ):
        return x
    return _ReplicatedInputGrad.apply(x)


def _patch_column_parallel_input_grad(stack):
    """Route column-parallel layers' replicated input through the `f` conjugate.

    Patched at the class level for the duration of one TTS forward, alongside
    the `MultiPlatformOp` swap. `ColumnParallelLinear` covers `QKVParallelLinear`
    and `MergedColumnParallelLinear` by inheritance. The vocab-parallel head is
    a bare matmul inside `LogitsProcessor._compute_lm_head` with no module
    forward to wrap, so that method is patched as well; the patch also covers
    the adapted head installed by `eagle_lora_tts._wrap_lm_head`, which is
    reached through the same method.
    """
    import sglang.srt.distributed.parallel_state as _pstate

    if _pstate._TP is None or _pstate.get_tensor_model_parallel_world_size() <= 1:
        return

    from sglang.srt.layers.linear import ColumnParallelLinear
    from sglang.srt.layers.logits_processor import LogitsProcessor

    col_forward = ColumnParallelLinear.forward

    def forward(self, input_, *args, **kwargs):
        return col_forward(self, all_reduce_input_grad(input_), *args, **kwargs)

    ColumnParallelLinear.forward = forward
    stack.append((ColumnParallelLinear, "forward", col_forward))

    head_fn = LogitsProcessor._compute_lm_head

    def compute_lm_head(self, hidden_states, lm_head, embedding_bias=None):
        return head_fn(
            self, all_reduce_input_grad(hidden_states), lm_head, embedding_bias
        )

    LogitsProcessor._compute_lm_head = compute_lm_head
    stack.append((LogitsProcessor, "_compute_lm_head", head_fn))


def _sdpa_attention(m, q, k, v, forward_batch):
    """Differentiable stand-in for the fused attention kernel.

    The fused backends (FA3, FlashInfer, Triton) have no backward. With them,
    `qkv_proj` and everything feeding it (`embed_tokens`, `input_layernorm`,
    `hidden_norm`) receive no gradient, while everything downstream of
    attention still does.

    This recomputes the same attention with `F.scaled_dot_product_attention`
    over [cached prefix K/V] ++ [this block's live K/V], so the block's K/V
    stay on the autograd tape and the k and v projections receive gradients
    too.

    q/k/v are flat `[num_tokens, n_heads * head_dim]` and already rotated by
    RoPE. The cache holds post-RoPE K, so the prefix must not be rotated again.
    """
    # The KV pool and the request-to-token table are carried on the ForwardBatch.
    kv_pool = forward_batch.token_to_kv_pool
    req_to_token = forward_batch.req_to_token_pool.req_to_token
    layer_id = m.attn.layer_id
    k_cache = kv_pool.get_key_buffer(layer_id)
    v_cache = kv_pool.get_value_buffer(layer_id)
    # The rebuild needs a KV buffer that is [num_tokens, kv_heads, head_dim]
    # (NHD) and indexable with the full-pool locations stored in
    # `req_to_token`. A shape check alone does not establish that: SWAKVPool
    # returns a 3-D NHD buffer for its sliding layers but expects
    # SWA-translated indices, and MLATokenToKVPool returns
    # [size, 1, lora_rank + rope_dim], which passes a shape test whenever
    # num_kv_heads == 1. Either would produce plausible but wrong attention, so
    # require the plain MHA pool explicitly.
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    if not isinstance(kv_pool, MHATokenToKVPool):
        raise RuntimeError(
            f"EAGLE3 TTS: draft KV pool is {type(kv_pool).__name__}; the "
            "differentiable attention path only supports MHATokenToKVPool, "
            "whose buffers are NHD and indexed by the same full-pool locations "
            "stored in req_to_token."
        )
    if k_cache.dim() != 3 or k_cache.shape[1] != m.num_kv_heads:
        raise RuntimeError(
            f"EAGLE3 TTS: draft KV cache buffer has shape {tuple(k_cache.shape)}, "
            f"which is not the [num_tokens, {m.num_kv_heads}, head_dim] NHD layout "
            "the differentiable attention path requires."
        )
    k_scale = getattr(m.attn, "k_scale_float", None)
    v_scale = getattr(m.attn, "v_scale_float", None)

    n_tok = q.shape[0]
    qh = q.view(n_tok, m.num_heads, m.head_dim)
    kh = k.view(n_tok, m.num_kv_heads, m.head_dim)
    vh = v.view(n_tok, m.num_kv_heads, m.head_dim)
    n_rep = m.num_heads // m.num_kv_heads
    window = getattr(m.attn, "sliding_window_size", -1)

    if forward_batch.extend_seq_lens is None:
        # Decode (the draft rollout): exactly one query token per request.
        #
        # The prefix length comes from `positions`, not `seq_lens`.
        # `draft_forward` advances `forward_batch.positions` on every rollout
        # step but leaves `seq_lens` untouched, so `seq_lens - 1` is correct
        # only at step 0 and would truncate the prefix on every later step.
        # `positions[b]` is the query's absolute index, so exactly that many
        # tokens precede it.
        ext = [1] * forward_batch.batch_size
        pre = [int(x) for x in forward_batch.positions.tolist()]
    else:
        # Extend. The lengths come from the extend fields, not `seq_lens`: for
        # DRAFT_EXTEND_V2 `seq_lens` has already been advanced by
        # num_draft_tokens, so it is the post-write length.
        ext = forward_batch.extend_seq_lens_cpu
        if ext is None:
            ext = forward_batch.extend_seq_lens.tolist()
        pre = forward_batch.extend_prefix_lens_cpu
        if pre is None:
            pre = forward_batch.extend_prefix_lens.tolist()

    out = qh.new_empty(n_tok, m.num_heads, m.head_dim)
    start = 0
    for i in range(forward_batch.batch_size):
        seq_len = int(ext[i])
        if seq_len == 0:
            continue
        prefix = int(pre[i])
        idx = req_to_token[forward_batch.req_pool_indices[i], :prefix].long()
        kp = k_cache[idx].to(qh.dtype)
        vp = v_cache[idx].to(qh.dtype)
        if k_scale is not None:
            kp = kp * k_scale
            vp = vp * v_scale

        full_k = torch.cat([kp, kh[start : start + seq_len]], dim=0)
        full_v = torch.cat([vp, vh[start : start + seq_len]], dim=0)

        qs = qh[start : start + seq_len].transpose(0, 1).unsqueeze(0)
        ks = full_k.transpose(0, 1).unsqueeze(0)
        vs = full_v.transpose(0, 1).unsqueeze(0)
        if n_rep > 1:
            ks = ks.repeat_interleave(n_rep, dim=1)
            vs = vs.repeat_interleave(n_rep, dim=1)

        total = prefix + seq_len
        kpos = torch.arange(total, device=q.device).unsqueeze(0)
        qpos = torch.arange(prefix, total, device=q.device).unsqueeze(1)
        # Bottom-right aligned causal mask: queries start at absolute position
        # `prefix`, so SDPA's built-in top-left `is_causal` would be wrong.
        allowed = kpos <= qpos
        if window is not None and window > -1:
            # Matches FA3's inclusive window_size=(W, 0).
            allowed = allowed & (kpos >= qpos - window)
        mask = torch.zeros(allowed.shape, dtype=qs.dtype, device=q.device)
        mask.masked_fill_(~allowed, float("-inf"))

        o = F.scaled_dot_product_attention(
            qs, ks, vs,
            attn_mask=mask.unsqueeze(0).unsqueeze(0),
            scale=m.scaling,
            is_causal=False,
        )
        out[start : start + seq_len] = o.squeeze(0).transpose(0, 1)
        start += seq_len

    return out.reshape(n_tok, m.num_heads * m.head_dim)


def _patch_attention_for_grad(model, stack):
    """Route the draft's attention modules through `_sdpa_attention`.

    Installed per instance, so nothing else in the process is affected, and
    only for the duration of one TTS forward. The patched modules are appended
    to `stack` for the caller to restore.
    """
    from sglang.srt.layers.radix_attention import RadixAttention

    # Count the fused-attention layers so the check at the end is exact: a
    # model with none (a test stub) is fine, but a draft whose attention
    # modules are not recognized must raise rather than run with no gradient
    # reaching `qkv_proj`.
    expected = sum(isinstance(mod, RadixAttention) for mod in model.modules())
    patched = 0
    if expected == 0:
        # No fused attention (a test stub): nothing to patch.
        return

    for module in model.modules():
        if not all(
            hasattr(module, attr)
            for attr in ("attn", "qkv_proj", "o_proj", "rotary_emb", "q_size")
        ):
            continue

        def make_forward(m):
            def forward(positions, hidden_states, forward_batch):
                qkv, _ = m.qkv_proj(hidden_states)
                q, k, v = qkv.split([m.q_size, m.kv_size, m.kv_size], dim=-1)
                q, k = m.rotary_emb(positions, q, k)
                # Run the fused attention only for its KV-write side effect
                # (its output is discarded): later decode rounds read this
                # block from the cache. k/v are cloned because a KV pool with
                # k/v scaling rescales them in place, which would corrupt the
                # values still on the tape.
                with torch.no_grad():
                    m.attn(
                        q.detach(), k.detach().clone(), v.detach().clone(),
                        forward_batch,
                    )
                attn_output = _sdpa_attention(m, q, k, v, forward_batch)
                output, _ = m.o_proj(attn_output)
                return output

            return forward

        module.forward = make_forward(module)
        stack.append(module)
        patched += 1
    if patched != expected:
        raise RuntimeError(
            f"EAGLE3 TTS: draft has {expected} fused attention layer(s) but only "
            f"{patched} could be made differentiable. qkv_proj and the embedding "
            "would silently train on zero gradients."
        )


@contextlib.contextmanager
def tts_grad_mode(model=None, patch_attention: bool = True):
    """Make a draft forward differentiable for the duration of the context.

    Four substitutions are installed and removed again on exit:

    * Fused layers. SGLang's fused layers (`RMSNorm`, `SiluAndMul`, ...) call
      into `sgl_kernel`, which has no autograd registration: outputs come back
      with `grad_fn=None`, so the tape ends at the first such op and only the
      parameters downstream of it receive a gradient. Every fused layer derives
      from `MultiPlatformOp`, whose `forward` dispatches to a per-platform
      method chosen at init, and each also defines a pure-PyTorch
      `forward_native`. Routing that single entry point to `forward_native`
      covers all of them without an allow-list of ops.
    * TP collectives and column-parallel inputs
      (`_patch_differentiable_collectives`,
      `_patch_column_parallel_input_grad`).
    * Attention, when `model` is given and `patch_attention` is true
      (`_patch_attention_for_grad`).
    * The model forward, when `model` is given. SGLang decorates every
      `*ForCausalLM.forward` with `@torch.no_grad()` (`models/llama.py`,
      inherited by `LlamaForCausalLMEagle3`), so `enable_grad()` at the call
      site is not enough, and a `forward_pre_hook` does not help either: the
      hook runs before `forward`, and the decorator disables grad again on
      entry. `functools.wraps` leaves the undecorated function on
      `__wrapped__`, which is called directly under `enable_grad`.

    The forward and attention overrides are installed on the instance, not the
    class, so the target model is unaffected even when it shares a class with
    the draft.
    """
    from sglang.srt.layers.utils.multi_platform import MultiPlatformOp

    saved = MultiPlatformOp.forward

    def forward_native_dispatch(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    restore_stack = []
    attn_stack = []
    patched_forward = False

    # The patch helpers raise by design when they cannot make the forward
    # differentiable, so all of them run inside the try. Otherwise a raise
    # would leave `MultiPlatformOp.forward` routed to `forward_native` for the
    # whole process, including the target model's RMSNorm / SiluAndMul /
    # FusedMoE, which would then run unfused with nothing in the log.
    try:
        MultiPlatformOp.forward = forward_native_dispatch
        _patch_differentiable_collectives(restore_stack)
        _patch_column_parallel_input_grad(restore_stack)

        if model is not None and patch_attention:
            _patch_attention_for_grad(model, attn_stack)

        patched_forward = model is not None
        if patched_forward:
            class_forward = type(model).forward
            # The undecorated body if @torch.no_grad() wrapped it, else the
            # function itself. Wrapping is needed either way: the scheduler
            # event loop runs with grad disabled (`DynamicGradMode`).
            inner = getattr(class_forward, "__wrapped__", class_forward)
            model.forward = _make_grad_forward(model, inner)
        yield
    finally:
        MultiPlatformOp.forward = saved
        for module, name, orig in restore_stack:
            setattr(module, name, orig)
        for attn_module in attn_stack:
            del attn_module.forward
        if patched_forward:
            del model.forward


def _make_grad_forward(model, inner):
    def grad_forward(*args, **kwargs):
        # The embedding and rotary fallbacks save their integer inputs (token
        # ids, positions) for backward. Those arguments can alias long-lived
        # buffers that are later mutated in place: at tp=1
        # `VocabParallelEmbedding` forwards the ids unmasked (tp>1 builds a
        # masked copy), and the rollout advances positions with `add_`. The
        # backward's version check would then fail with "modified by an inplace
        # operation". Clone the integer arguments so the tape holds private
        # copies. Float arguments are large and are not mutated in place within
        # a round, so they are passed through.
        args = tuple(
            a.clone()
            if isinstance(a, torch.Tensor) and not a.is_floating_point()
            else a
            for a in args
        )
        with torch.enable_grad():
            return inner(model, *args, **kwargs)

    return grad_forward


def privatize_target_aliases(draft_model: nn.Module, target) -> List[str]:
    """Give the draft its own copy of every tensor it shares with the target.

    Must run before CUDA graph capture. Privatizing rebinds the attribute to a
    new `nn.Parameter` at a new address, whereas every other updated parameter
    is written in place. A graph captured earlier would keep reading the old
    (target) tensor, so graph-replayed drafting would not use the weights that
    TTS updates.

    `target` is either the target module or a set of its parameter data_ptrs.
    Returns the names of the privatized parameters.
    """
    ptrs = (
        target
        if isinstance(target, set)
        else {p.data_ptr() for p in target.parameters()}
    )
    privatized = []
    for name, param in list(draft_model.named_parameters()):
        if param.data_ptr() not in ptrs:
            continue
        module = draft_model
        *path, attr = name.split(".")
        for part in path:
            module = getattr(module, part)
        # requires_grad=False matches how sglang loads every other weight
        # (layers/parameter.py); CUDA graph capture, which runs next, assumes
        # inference-style parameters. `EagleTTSState` enables grad on the
        # trainable set afterwards.
        setattr(module, attr, nn.Parameter(param.detach().clone(), requires_grad=False))
        privatized.append(name)
    return privatized


def tts_loss_rows(
    bs: int,
    n_step: int,
    nd: int,
    accept_lens: Optional[Sequence[int]],
    correct_drafts_only: bool,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Pair draft rollout rows with target verify rows for the TTS loss.

    Rollout step i of request b produced the distribution for tree node i+2,
    stored at draft row `b * n_step + i`. Target verify row j predicts node
    j+1, so node i+2 pairs with target row `b * nd + i + 1` (`nd` is the number
    of draft tokens per request, including the root).

    `accept_lens[b]` counts the accept tokens of request b including the bonus
    token. With `correct_drafts_only`, position i is kept only when node i+2 is
    a correct draft, i.e. `i + 2 <= accept_lens[b] - 1`; otherwise every
    rollout position is kept and `accept_lens` is not read (it may be None).

    Returns `(keep, draft_rows, target_rows)`: the kept `(request, step)`
    pairs and the matching row indices into the stacked draft logits and the
    target verify logits.
    """
    keep = [
        (b, i)
        for b in range(bs)
        for i in range(n_step)
        if not correct_drafts_only or i + 2 <= int(accept_lens[b]) - 1
    ]
    draft_rows = [b * n_step + i for b, i in keep]
    target_rows = [b * nd + i + 1 for b, i in keep]
    return keep, draft_rows, target_rows


@dataclass
class EagleTTSConfig:
    enable: bool = False
    lr: float = 1e-5
    tts_lambda: float = 0.9  # "kl" loss: lambda*KL + (1-lambda)*self-CE
    # "kl" -> lambda*KL(target||draft) + (1-lambda)*self-CE
    # "lk" -> -log(sum_x min(p_target, q_draft)). sum_x min(p, q) equals
    #         1 - TV(p, q) and is the speculative acceptance probability, so
    #         this minimizes the quantity acceptance depends on directly.
    tts_loss: str = "lk"
    freq: int = 10  # update stride: fire when round_idx % freq == 0
    # Depth weighting w_d = depth_decay ** d. Acceptance is a chain: a shallow
    # position gates every position behind it, so closing the target/draft gap
    # at a shallow position matters more than at a deep one. A value outside
    # (0, 1) means uniform weighting.
    depth_decay: float = 0.8
    # Accumulate this many firing rounds per optimizer step. One round supplies
    # only a few token positions, so its gradient is noisy.
    accum_steps: int = 1
    grad_clip: float = 1.0
    kl_threshold: float = 0.0  # skip the step when KL < threshold (0 disables)
    # Train only on the correct drafts (the drafted positions the target
    # accepted). By default every drafted position enters the loss: a row past
    # the correct drafts is conditioned on the draft's own (rejected) tokens,
    # and its label is the target's distribution given that drafted prefix.
    correct_drafts_only: bool = False
    async_step: bool = True
    # Run Adam on an fp32 master copy and write the result back to the (bf16)
    # weights. A small-lr update applied directly in bf16 is lost to rounding.
    # In LoRA mode the master holds the adapters rather than the draft weights.
    # False (--speculative-eagle-tts-optimizer-dtype bfloat16) stores the
    # weights and the moments in bf16 only: a step widens them to fp32, runs
    # the same arithmetic and rounds once when it stores the results.
    fp32_master: bool = True

    # "full" -> update the draft's own weights (one request at a time).
    # "lora" -> update low-rank adapters over `lora_targets`; base weights frozen.
    mode: str = "full"
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_targets: str = ""  # "" -> eagle_lora_tts.DEFAULT_LORA_TARGETS
    # One adapter slot per concurrent request instead of one shared adapter.
    # Required for batched serving: with a shared adapter, one request's
    # prefill resets the adapter another request is still using. LoRA mode
    # only; a per-request copy of every draft weight would be far too large.
    per_request: bool = False

    @property
    def is_lora(self) -> bool:
        return self.mode == "lora"

    @classmethod
    def from_server_args(cls, server_args) -> "EagleTTSConfig":
        def _get(name, default):
            value = getattr(server_args, f"speculative_eagle_tts_{name}", None)
            return default if value is None else value

        return cls(
            mode=str(_get("mode", "full")),
            lora_rank=int(_get("lora_rank", 16)),
            lora_alpha=float(_get("lora_alpha", 32.0)),
            lora_targets=str(_get("lora_targets", "") or ""),
            per_request=bool(_get("per_request", False)),
            enable=bool(_get("enable", False)),
            lr=float(_get("lr", 1e-5)),
            tts_lambda=float(getattr(server_args, "speculative_tts_lambda", 0.9)),
            tts_loss=str(getattr(server_args, "speculative_tts_loss", "lk")),
            freq=int(_get("freq", 10)),
            depth_decay=float(_get("depth_decay", 0.8)),
            accum_steps=max(1, int(_get("accum_steps", 1))),
            grad_clip=float(_get("grad_clip", 1.0)),
            kl_threshold=float(_get("kl_threshold", 0.0)),
            correct_drafts_only=bool(_get("correct_drafts_only", False)),
            async_step=not bool(_get("no_async", False)),
            fp32_master=str(_get("optimizer_dtype", "float32")) != "bfloat16",
        )


class EagleTTSState:
    """TTS state for one EAGLE-3 draft model on one worker.

    Owns the update schedule (`should_fire` / `advance`), the loss and
    optimizer step (`step`), the asynchronous hand-off (`wait_for_pending`) and
    the reset at the start of each request (`reset`). In LoRA mode the
    trainable tensors are supplied by an `EagleLoRAAdapterSet`; in full mode
    they are the draft's own weights.
    """

    def __init__(
        self,
        *,
        draft_model: nn.Module,
        config: EagleTTSConfig,
        device: torch.device | str,
        tp_size: int = 1,
        target_model: Optional[nn.Module] = None,
        adapters=None,
    ):
        self.cfg = config
        self.draft_model = draft_model
        # `EagleLoRAAdapterSet` in LoRA mode, else None. It owns the trainable
        # tensors, the base-weight freeze and the adapter reset; this class
        # provides the loss, schedule, optimizer and async step over whatever
        # tensors it is handed.
        self.adapters = adapters
        # Pool rows of the requests in the current batch, published by the
        # worker before every draft forward. Pooled mode only; None otherwise.
        self.active_slots: Optional[torch.Tensor] = None
        self.device = device
        self.tp_size = int(tp_size)
        self.round_idx: int = 0
        self.step_ct: int = 0
        self.skip_ct: int = 0
        self._reported_coverage: bool = False
        self._warned_no_grad: bool = False
        self._accum_count: int = 0
        self.nonfinite_ct: int = 0

        # EAGLE-3 binds some draft tensors directly to the target's
        # (`set_embed` / `set_embed_and_head` in models/llama.py rebind
        # `embed_tokens.weight`, and `lm_head.weight` for drafts that set
        # `load_lm_head_from_target`). An optimizer step on those would modify
        # the target model, so record the target's storage pointers and check
        # every trainable tensor against them.
        self._target_ptrs = set()
        if target_model is not None:
            self._target_ptrs = {p.data_ptr() for p in target_model.parameters()}

        if self.adapters is not None:
            # LoRA mode. The adapter set freezes the base weights, so no draft
            # tensor is written, including those that alias the target.
            # Neither privatization nor a full-weight snapshot is needed: reset
            # restores A to its template and zeroes B, which restores the
            # pretrained function exactly.
            self._privatized = []
            self._param_names = self.adapters.parameter_names()
            self._params = self.adapters.parameters()
            self._snapshot = {}
            self._assert_target_is_safe(
                list(zip(self._param_names, self._params))
            )
        else:
            # Full mode: privatize the target-aliased tensors before anything
            # updates them.
            self._privatized = self._privatize_target_aliases(draft_model)
            named = [
                (name, p)
                for name, p in draft_model.named_parameters()
                if not p.is_inference()
            ]
            self._assert_target_is_safe(named)
            for _, p in named:
                p.requires_grad_(True)
            self._param_names = [n for n, p in named if p.requires_grad]
            self._params = [p for _, p in named if p.requires_grad]

            # Snapshot for the per-request reset. Kept on device: a restore
            # happens once per request, and a host round trip would stall the
            # prefill.
            self._snapshot = {name: p.detach().clone() for name, p in named}

        if self._privatized and config.enable:
            logger.info(
                "EAGLE3 TTS: gave the draft a private copy of %d target-aliased "
                "tensor(s) so they can train without touching the target: %s",
                len(self._privatized),
                self._privatized,
            )

        if not config.fp32_master:
            other = sorted(
                {str(p.dtype) for p in self._params if p.dtype != torch.bfloat16}
            )
            if other:
                raise ValueError(
                    "--speculative-eagle-tts-optimizer-dtype bfloat16 keeps the "
                    "optimizer state in the dtype of the trained tensors and is "
                    "only supported for a bfloat16 draft model; this draft has "
                    f"{other} trainable tensors. Use float32."
                )

        self._master: Optional[List[torch.Tensor]] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._build_optimizer()

        device_obj = (
            torch.device(device) if not isinstance(device, torch.device) else device
        )
        self._use_async = (
            bool(config.async_step)
            and device_obj.type == "cuda"
            and torch.cuda.is_available()
        )

        self._tts_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(device=device_obj) if self._use_async else None
        )
        self._step_event: Optional[torch.cuda.Event] = None

        if config.enable:
            logger.info(
                "EAGLE3 TTS enabled. mode=%s loss=%s lr=%.2e lambda=%.3f freq=%d "
                "gamma=%.2f kl_threshold=%.3f async=%s fp32_master=%s "
                "trainable=%d accum=%d",
                config.mode, config.tts_loss, config.lr, config.tts_lambda, config.freq,
                config.depth_decay, config.kl_threshold, self._use_async,
                config.fp32_master,
                sum(p.numel() for p in self._params), config.accum_steps,
            )

    # ------------------------------------------------------------------ aliasing

    def _privatize_target_aliases(self, draft_model: nn.Module) -> List[str]:
        """Replace draft tensors that alias the target with private copies.

        `init_lm_head` rebinds the draft's `embed_tokens.weight` (and, when the
        draft has no head of its own, i.e. `load_lm_head_from_target`, its
        `lm_head.weight`) to the target's tensor objects
        (`set_embed` / `set_embed_and_head`, models/llama.py). They share
        storage; they are not copies.

        Updating them in place would modify the target model during a request.
        In speculative decoding the target defines the emitted text, so the
        output would change and decoding would stop being lossless.

        Freezing them instead would exclude the embedding and the head from the
        update. Cloning keeps every draft weight trainable and leaves the
        target untouched.
        """
        if not self._target_ptrs:
            return []
        return privatize_target_aliases(draft_model, self._target_ptrs)

    def _assert_target_is_safe(self, named) -> None:
        """Raise if any tensor TTS would update still aliases the target."""
        leaked = [name for name, p in named if p.data_ptr() in self._target_ptrs]
        if leaked:
            raise RuntimeError(
                "EAGLE3 TTS: cannot start: these draft parameters share storage "
                "with the target model and an optimizer step would corrupt "
                f"it: {leaked}"
            )

    # ---------------------------------------------------------------- optimizer

    @property
    def is_pooled(self) -> bool:
        """True when each request owns its own adapter slot (batched serving)."""
        return self.adapters is not None and self.adapters.pool_size > 1

    def _build_optimizer(self):
        if self.is_pooled:
            # Pooled parameters are [P, ...] and every slot needs its own Adam
            # step count, because requests start at different times; the single
            # scalar `state['step']` of torch.optim.Adam cannot express that.
            # PoolAdam holds the master and the moments itself (fp32, or bf16
            # when `fp32_master` is off) and updates only the rows that fired.
            from sglang.srt.speculative.eagle_lora_tts import PoolAdam

            self._master = None
            self._optimizer = PoolAdam(
                self._params,
                self.adapters.pool_size,
                lr=self.cfg.lr,
                fp32_state=self.cfg.fp32_master,
            )
        elif self.cfg.fp32_master:
            self._master = [
                p.detach().clone().float().requires_grad_(True) for p in self._params
            ]
            self._optimizer = torch.optim.Adam(self._master, lr=self.cfg.lr)
        else:
            self._master = None
            self._optimizer = Bf16StateAdam(self._params, lr=self.cfg.lr)

    def _apply_master_update(self):
        """Copy grads into the fp32 master, step, and write back to the draft."""
        assert self._master is not None
        with torch.no_grad():
            for p, m in zip(self._params, self._master):
                m.grad = None if p.grad is None else p.grad.detach().float()
        if self.cfg.grad_clip and self.cfg.grad_clip > 0:
            grad_norm = self._clip_grads(self._master)
        else:
            grad_norm = None
        self._optimizer.step()
        with torch.no_grad():
            for p, m in zip(self._params, self._master):
                p.copy_(m.to(p.dtype))
        return grad_norm

    def _apply_bf16_state_update(self):
        """Clip and step on fp32 leaves; the optimizer stores the rounded results.

        The leaves take the place of the fp32 masters in `_clip_grads`, so the
        gradients, the clipping norm and the coefficient are those of
        `_apply_master_update`.
        """
        leaves = self._optimizer.widen()
        if self.cfg.grad_clip and self.cfg.grad_clip > 0:
            grad_norm = self._clip_grads(leaves)
        else:
            grad_norm = None
        self._optimizer.step(leaves)
        return grad_norm

    # ------------------------------------------------------------ stride / async

    def note_no_aligned_rows(self) -> None:
        """Count a firing round that contributed no positions to the loss."""
        self.skip_ct += 1

    def should_fire(self) -> bool:
        if not self.cfg.enable:
            return False
        return (self.round_idx % max(1, self.cfg.freq)) == 0

    def advance(self) -> None:
        self.round_idx += 1

    def wait_for_pending(self) -> None:
        """Block until the pending asynchronous optimizer step has completed.

        Called immediately before the next draft forward, the first point at
        which the updated weights are read.

        This is a CPU-side `event.synchronize()` rather than a GPU-side
        `wait_event`: the draft forward may be replayed from a captured CUDA
        graph on PyTorch's internal stream, and a wait recorded on the host's
        current stream does not propagate into a graph replay. The replay would
        then read weights that are only partially written, without any error.
        """
        if self._step_event is None or not self._use_async:
            return
        self._step_event.synchronize()
        self._step_event = None

    def reset(self, slots: "torch.Tensor | None" = None) -> None:
        """Restore the pretrained draft and clear optimizer state at a new prefill.

        `slots` (pooled mode only) restricts the reset to the pool rows of the
        requests that are starting. A whole-pool reset would discard the
        adaptation of every other request in flight, so the pooled path raises
        when `slots` is missing.
        """
        if not self.cfg.enable:
            return
        if self._step_event is not None:
            self._step_event.synchronize()
            self._step_event = None
        if self.is_pooled:
            if slots is None:
                raise RuntimeError(
                    "EAGLE3 LoRA-TTS: reset() with no slots while pooled. A "
                    "prefill must clear only the slots it is claiming."
                )
            self.adapters.reset(slots)
            # `reset_slots` indexes the pooled tensors itself, so it takes the
            # full parameters (already restored above), not a gathered copy.
            self._optimizer.reset_slots(slots, self._params)
            # `round_idx` keeps running: it drives the firing schedule for
            # every request in flight, and rewinding it would shift the phase
            # of the whole batch whenever one request prefills.
            return
        if self.adapters is not None:
            # O(adapter) rather than O(model): no full-weight restore.
            self.adapters.reset()
        else:
            with torch.no_grad():
                for name, p in self.draft_model.named_parameters():
                    # Parameters without a snapshot entry are not trained and
                    # must not be written.
                    snap = self._snapshot.get(name)
                    if snap is not None and snap.shape == p.shape:
                        p.copy_(snap)
        self._build_optimizer()
        self._accum_count = 0
        self.round_idx = 0

    # ------------------------------------------------------------------- update

    def step(
        self,
        *,
        draft_logits: torch.Tensor,
        target_logits: torch.Tensor,
        hot_token_id: Optional[torch.Tensor] = None,
        pos_weights: Optional[torch.Tensor] = None,
        row_groups: Optional[torch.Tensor] = None,
        num_groups: int = 0,
    ) -> Optional[float]:
        """Compute the TTS loss and apply one update.

        Args:
            draft_logits: [N, draft_vocab] draft-head logits for the drafted
                positions, with grad connected to the trainable tensors.
            target_logits: [N, target_vocab] target verification logits for
                the same positions, in the same order. Detached internally.
            hot_token_id: [draft_vocab] mapping draft-vocab index -> target-vocab
                index, for drafts with a reduced vocabulary. None when the two
                vocabularies match.
            pos_weights: [N] per-position loss weights (the depth weighting),
                or None for an unweighted mean.
            row_groups, num_groups: pooled mode only. `row_groups[i]` is the
                index of the request row i belongs to, out of `num_groups`, so
                the loss is normalized per request rather than over the whole
                batch.

        Returns the per-request mean of the divergence term if it was
        materialized (only when `kl_threshold` is enabled), else None.
        """
        if not self.cfg.enable:
            return None
        if not draft_logits.requires_grad:
            # The draft forward produced logits with no autograd history, so
            # there is nothing to update. Warn once: decoding keeps working in
            # this state, so TTS would otherwise be inactive without notice.
            self.skip_ct += 1
            if not self._warned_no_grad:
                self._warned_no_grad = True
                logger.warning(
                    "EAGLE3 TTS: draft logits have no grad_fn, so every step is a "
                    "no-op. Something on the draft forward is not differentiable "
                    "(a fused kernel without a backward, or an enclosing "
                    "inference_mode)."
                )
            return None
        if draft_logits.shape[0] == 0:
            return None

        n_pos = draft_logits.shape[0]
        t_logits = target_logits.detach()
        if t_logits.shape[0] != n_pos:
            raise ValueError(
                "EAGLE3 TTS: draft and target logits must have the same number "
                f"of rows, got {n_pos} and {t_logits.shape[0]}"
            )

        # Restrict the target distribution to the draft's vocabulary by
        # selecting the draft's columns before the softmax, which renormalizes
        # the target over the draft vocabulary. `hot_token_id` is the
        # draft-to-target index map built from `d2t`.
        if hot_token_id is not None and t_logits.shape[-1] != draft_logits.shape[-1]:
            t_logits = t_logits.index_select(-1, hot_token_id)

        d_logits_f = draft_logits.float()
        t_logits_f = t_logits.float()

        log_p_d = F.log_softmax(d_logits_f, dim=-1)
        log_p_t = F.log_softmax(t_logits_f, dim=-1)
        p_t = log_p_t.exp()
        if self.cfg.tts_loss == "lk":
            per_pos_kl = lk_per_position(p_t, log_p_d)  # [n_pos]
        else:
            # A token the target gives zero probability (a -inf logit)
            # contributes nothing; without the mask it would be 0 * inf.
            per_pos_kl = (
                (p_t * (log_p_t - log_p_d)).masked_fill(p_t == 0, 0.0).sum(dim=-1)
            )  # [n_pos]

        with torch.no_grad():
            self_targets = d_logits_f.argmax(dim=-1)
        per_pos_ce = F.cross_entropy(d_logits_f, self_targets, reduction="none")

        if pos_weights is None and row_groups is not None:
            # No depth weighting: the per-request means below still apply.
            pos_weights = torch.ones(n_pos, device=d_logits_f.device)
        if pos_weights is not None and pos_weights.numel() == n_pos:
            # Normalised, so a constant depth offset cancels.
            w = pos_weights.to(d_logits_f.dtype)
            if row_groups is not None and num_groups > 0:
                # Per-request weighted mean, summed across requests. A single
                # mean over the whole batch would divide every request's
                # gradient by the round's total row weight, so a request's
                # update would depend on how many rows the other requests
                # contributed that round, and a fixed `grad_clip` would bind
                # less and less as the batch grew. Each request owns its own
                # adapter slot, so summing per-request means gives every slot
                # the gradient it would have had if its request ran alone.
                idx = row_groups.to(torch.long)
                zeros = torch.zeros(
                    num_groups, dtype=d_logits_f.dtype, device=d_logits_f.device
                )
                wsum_g = zeros.index_add(0, idx, w).clamp_min(1e-6)
                kl_term = (zeros.index_add(0, idx, per_pos_kl * w) / wsum_g).sum()
                ce_term = (zeros.index_add(0, idx, per_pos_ce * w) / wsum_g).sum()
            else:
                wsum = w.sum().clamp_min(1e-6)
                kl_term = (per_pos_kl * w).sum() / wsum
                ce_term = (per_pos_ce * w).sum() / wsum
        else:
            kl_term = per_pos_kl.mean()
            ce_term = per_pos_ce.mean()

        kl_value: Optional[float] = None
        if self.cfg.kl_threshold > 0:
            # .item() forces a host sync; only pay for it when the gate is on.
            # Compare the per-request mean, not the sum across requests used
            # for backward: the threshold is a per-request convergence gate,
            # and gating on the sum would make the update cadence depend on
            # the batch size.
            kl_value = float(kl_term.detach().item()) / max(1, num_groups or 1)
            if kl_value < self.cfg.kl_threshold:
                self.skip_ct += 1
                return kl_value

        if self.cfg.tts_loss == "lk":
            loss = kl_term  # already the depth-weighted -log(alpha)
        else:
            loss = (
                self.cfg.tts_lambda * kl_term
                + (1.0 - self.cfg.tts_lambda) * ce_term
            )

        # A single non-finite loss would poison the master weights and
        # Adam's moments for the rest of the request (a reset only happens at
        # the next prefill), so skip the step instead.
        if not torch.isfinite(loss):
            self.nonfinite_ct += 1
            if self.nonfinite_ct <= 3:
                logger.warning(
                    "EAGLE3 TTS: non-finite loss (kl=%s ce=%s) at round %d; "
                    "skipping step (%d so far). Inputs finite: draft=%s "
                    "target=%s.",
                    float(kl_term), float(ce_term), self.round_idx,
                    self.nonfinite_ct,
                    bool(torch.isfinite(draft_logits).all()),
                    bool(torch.isfinite(t_logits).all()),
                )
            return None

        if self._use_async and self._tts_stream is not None:
            # Asynchronous step. `loss.backward()` runs its autograd nodes on
            # the stream that was current during the forward (the main stream),
            # regardless of any `torch.cuda.stream(...)` around the call. So
            # `_backward_and_step` first enqueues backward on the main stream,
            # then makes the side stream wait on the main stream, and only
            # then runs the optimizer step on the side stream. Issuing the
            # wait before backward would let the optimizer read `p.grad` while
            # the main stream is still writing it.
            self._backward_and_step(loss, async_opt=True)
            if self._accum_count == 0:  # an optimizer step actually ran
                if self._step_event is None:
                    self._step_event = torch.cuda.Event()
                self._step_event.record(self._tts_stream)
        else:
            self._backward_and_step(loss)

        self.step_ct += 1
        return kl_value

    def _backward_and_step(self, loss: torch.Tensor, async_opt: bool = False):
        """Run backward, then the optimizer step once the accumulation window is full.

        Backward is always issued on the current (main) stream. With
        `async_opt`, the optimizer step is issued on the side stream after that
        stream has been made to wait for the main stream.
        """
        accum = max(1, int(self.cfg.accum_steps))
        if self._accum_count == 0:
            # Clear gradients only at the start of an accumulation window, so
            # the earlier rounds' gradients are kept.
            self._zero_grad()
        (loss / accum).backward()
        self._report_grad_coverage_once()
        self._accum_count += 1
        if self._accum_count < accum:
            return None
        self._accum_count = 0
        opt_ctx = contextlib.nullcontext()
        if async_opt and self._tts_stream is not None:
            main_stream = torch.cuda.current_stream(device=self.device)
            # Backward has been enqueued on the main stream, so this wait
            # covers gradient production.
            self._tts_stream.wait_stream(main_stream)
            for prm in self._params:
                if prm.grad is not None:
                    # Gradients are allocated on the main stream but consumed
                    # on the side stream. Without record_stream the caching
                    # allocator may reuse the memory while the side stream is
                    # still reading it.
                    prm.grad.record_stream(self._tts_stream)
            if self.active_slots is not None:
                # The same hazard applies to `active_slots`, which aliases the
                # ForwardBatch's req_pool_indices: the next round's `draft()`
                # drops the last Python reference to it. The side stream uses
                # it as an index, so reused memory could produce an
                # out-of-range gather, which is a device-side fault rather
                # than a wrong value.
                self.active_slots.record_stream(self._tts_stream)
            opt_ctx = torch.cuda.stream(self._tts_stream)
        with opt_ctx:
            return self._finish_step()

    def _finish_step(self):
        self._reduce_partial_grads()
        # A finite loss can still yield non-finite gradients (overflow in
        # backward), and clip_grad_norm_ does not sanitize NaN/Inf, so check
        # before stepping.
        bad = [p for p in self._params if p.grad is not None
               and not torch.isfinite(p.grad).all()]
        skip = bool(bad)
        if self.tp_size > 1:
            # Sharded gradients differ between ranks, so a non-finite value
            # can appear on one rank only. The ranks must skip together: the
            # clip below contains a collective, and a rank that stepped alone
            # would hold different weights.
            from sglang.srt.layers.linear import tensor_model_parallel_all_reduce

            flag = torch.tensor(
                [1.0 if bad else 0.0], device=self.device, dtype=torch.float32
            )
            skip = bool(tensor_model_parallel_all_reduce(flag).item() > 0)
        if skip:
            self.nonfinite_ct += 1
            if self.nonfinite_ct <= 3:
                logger.warning(
                    "EAGLE3 TTS: %d parameter(s) on this rank had non-finite "
                    "gradients at round %d -- skipping step (%d so far).",
                    len(bad), self.round_idx, self.nonfinite_ct,
                )
            self._zero_grad()
            return None
        if self.is_pooled:
            slots = self.active_slots
            if slots is None:
                raise RuntimeError(
                    "EAGLE3 LoRA-TTS: a pooled step fired with no slot mapping. "
                    "`active_slots` must be published every round the draft runs."
                )
            grad_norm = None
            if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                grad_norm = self._clip_grads_pooled(slots)
            self._optimizer.step(slots)
            return grad_norm
        if self._master is not None:
            return self._apply_master_update()
        return self._apply_bf16_state_update()

    def _report_grad_coverage_once(self):
        """Log, once, which trainable parameters a backward actually reached.

        An op without an autograd registration ends the tape without raising:
        the step still runs and the loss looks normal, but the parameters
        upstream of the break never change. `tts_grad_mode` covers every
        `MultiPlatformOp`, but the attention backend is not one of those, so
        this reports what the live forward delivered.

        In LoRA mode the report is deferred until one optimizer step has been
        applied. B is initialized to zero, so for
        `delta = scale * (x @ A^T) @ B^T` the gradient
        `dL/dA = (dL/d_delta @ B)^T x` is identically zero until B has moved,
        and every `lora_a` would be reported as unreached on the first
        backward.
        """
        if self._reported_coverage:
            return
        if self.adapters is not None and self.step_ct < 1:
            return
        self._reported_coverage = True
        starved = [
            name
            for name, p in zip(self._param_names, self._params)
            if p.grad is None or float(p.grad.abs().sum()) == 0.0
        ]
        total = len(self._params)
        expected = [n for n in starved if self._outside_rollout(n)]
        unexpected = [n for n in starved if n not in expected]
        if expected:
            logger.info(
                "EAGLE3 TTS: %d/%d draft parameters are outside the update and "
                "keep their pretrained values: %s. The input projection (`fc` "
                "and its norms) runs only in the draft-extend pass, which the "
                "update does not differentiate; a tensor-parallel embedding "
                "shard trains only on rounds that draft one of its tokens.",
                len(expected), total, expected,
            )
        if unexpected:
            logger.warning(
                "EAGLE3 TTS: %d/%d draft parameters received no gradient and will "
                "not train: %s. This means an op on the draft forward has no "
                "backward and cut the autograd graph.",
                len(unexpected), total, unexpected,
            )
        elif not expected:
            logger.info(
                "EAGLE3 TTS: all %d draft parameters received gradients.", total
            )

    def _outside_rollout(self, name: str) -> bool:
        """Whether a parameter legitimately gets no gradient from an update.

        The update differentiates the draft rollout. `fc` and `fc_norm` map the target's hidden states to the draft's width and run
        only in the draft-extend pass, so no rollout gradient reaches them. At
        tp>1 the embedding table is sharded over the vocabulary, and a rank
        whose shard holds none of the round's draft tokens sees a zero
        gradient for it on that round.
        """
        parts = name.split(".")
        if any(part in ("fc", "fc_norm") for part in parts):
            return True
        return self.tp_size > 1 and "embed_tokens" in parts

    def _clip_grads(self, params):
        """Gradient-norm clip with a norm that is identical on every TP rank.

        `torch.nn.utils.clip_grad_norm_` sums over whatever it is given. At
        tp>1 that list mixes replicated tensors with rank-local shards, so each
        rank would compute a different total norm and hence a different clip
        coefficient. Applying different coefficients to a replicated tensor
        makes its copies diverge across ranks.

        Instead, the sum of squares is built so that all-reducing it is exact:
        sharded entries contribute their own shard, and replicated entries
        contribute 1/tp_size each so the all-reduce counts them once. Every
        rank then applies the same coefficient.

        `params` holds the tensors whose gradients are clipped, the fp32 masters
        or leaves, one for each entry of `self._params` and in the same order.
        """
        max_norm = float(self.cfg.grad_clip)
        grads = [p.grad for p in params if p.grad is not None]
        if not grads or max_norm <= 0:
            return None
        if self.tp_size <= 1:
            return torch.nn.utils.clip_grad_norm_(params, max_norm)

        from sglang.srt.layers.linear import tensor_model_parallel_all_reduce

        replicated = self._replicated_grad_ids()
        total_sq = torch.zeros((), device=grads[0].device, dtype=torch.float32)
        for live, p in zip(self._params, params):
            if p.grad is None:
                continue
            sq = p.grad.detach().float().pow(2).sum()
            total_sq += sq / self.tp_size if id(live) in replicated else sq
        total_sq = tensor_model_parallel_all_reduce(total_sq)
        total_norm = total_sq.sqrt()

        coef = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
        with torch.no_grad():
            for p in params:
                if p.grad is not None:
                    p.grad.mul_(coef)
        return total_norm

    def _clip_grads_pooled(self, slots):
        """Gradient-norm clip computed per slot.

        One global norm over the pool would couple requests: a request with a
        large gradient would shrink every other request's update on that round.
        Norms are therefore reduced over the trailing dims only, giving one
        norm and one coefficient per slot.

        TP handling matches `_clip_grads`: replicated entries contribute
        1/tp_size so the all-reduce counts them once, which keeps the
        coefficient identical on every rank.
        """
        max_norm = float(self.cfg.grad_clip)
        params = [p for p in self._params if p.grad is not None]
        if not params or max_norm <= 0:
            return None
        pool = self.adapters.pool_size
        replicated = self._replicated_grad_ids()
        total_sq = torch.zeros(pool, device=params[0].device, dtype=torch.float32)
        for p in params:
            if p.dim() == 0 or p.shape[0] != pool:
                # Every trainable must carry the pool on dim 0; a tensor shared
                # across slots would be updated by all requests at once.
                raise RuntimeError(
                    f"EAGLE3 LoRA-TTS: trainable of shape {tuple(p.shape)} is "
                    f"not pooled (expected leading dim {pool})."
                )
            sq = p.grad.detach().float().pow(2).flatten(1).sum(1)  # [P]
            total_sq += sq / self.tp_size if id(p) in replicated else sq
        if self.tp_size > 1:
            from sglang.srt.layers.linear import tensor_model_parallel_all_reduce

            total_sq = tensor_model_parallel_all_reduce(total_sq)
        total_norm = total_sq.sqrt()                               # [P]
        coef = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)     # [P]
        with torch.no_grad():
            for p in params:
                p.grad.mul_(coef.view((-1,) + (1,) * (p.grad.dim() - 1)))
        return total_norm.index_select(0, slots).max()

    def _replicated_grad_ids(self):
        """ids of trainables whose gradient is identical on every rank.

        LoRA mode knows its layout exactly. Full mode returns an empty set, so
        its replicated tensors (the norms, `fc`) are counted tp_size times in
        the norm. That makes clipping somewhat more aggressive but keeps the
        coefficient identical across ranks, which is the property required.
        """
        if self.adapters is None:
            return set()
        return {id(p) for p in self.adapters.replicated_params()}

    def _reduce_partial_grads(self):
        """All-reduce the gradients of which each rank holds only a partial sum.

        Runs once per optimizer step, on the A tensors of the column-parallel
        adapters. Without it the ranks would apply different updates to a
        replicated tensor. See `LoRAAdapter.a_grad_is_partial`.
        """
        if self.adapters is None or self.tp_size <= 1:
            return
        from sglang.srt.layers.linear import tensor_model_parallel_all_reduce

        with torch.no_grad():
            for p in self.adapters.partial_grad_params():
                if p.grad is not None:
                    p.grad = tensor_model_parallel_all_reduce(p.grad.contiguous())

    def _zero_grad(self):
        for p in self._params:
            p.grad = None
        if self._optimizer is not None:
            self._optimizer.zero_grad(set_to_none=True)
