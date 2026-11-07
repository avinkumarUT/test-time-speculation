"""Low-rank (LoRA) parameterization of EAGLE-3 Test-Time Speculation (TTS).

`eagle_tts.py` implements the online update of the EAGLE-3 draft model
(the "speculator"). In its "full" mode the update is applied to every draft
weight, which needs a master copy, Adam moments and a restore snapshot for the
whole draft, all shared by every request on the worker. This module
provides the parameterization used for batched serving: the base weights stay
frozen and bit-identical, and the update is applied to low-rank adapters on a
chosen subset of the draft's matrices.

Consequences of the frozen base:

  * The trainable state is small (rank * (in + out) elements per adapted
    matrix), so every in-flight request can own a private copy.
  * No snapshot is needed: resetting A to its initial value and B to zero
    restores the pretrained draft exactly.
  * The target model is never written. The draft's `embed_tokens` (and
    `lm_head`, for drafts that borrow the target's) alias the target's
    tensors; a frozen base never writes them, so they need not be cloned.

Main pieces
-----------
`LoRAAdapter`
    One adapter, `delta = (alpha / rank) * (x @ A^T) @ B^T`. A and B carry a
    leading pool dimension: one slot per concurrent request.
`_RowSelector`
    Static index buffer that tells the adapters which slot each row of a
    forward belongs to. It lets a captured CUDA graph follow the current batch.
`EagleLoRAAdapterSet`
    Locates the requested targets on a draft model, records each one's
    tensor-parallel layout, installs the adapters by wrapping the module
    forwards, freezes the base, and resets adapters per slot.
`PoolAdam`
    Adam over pooled parameters with per-slot master weights, moments and step
    counts, kept in fp32 or, with `fp32_state=False`, in the pool's bf16. It
    steps only the slots that produced a gradient.

Adapters stay in the forward
----------------------------
The adapters are applied inside the wrapped module forwards and are never
merged into the base weights. On non-firing rounds the draft rollout is
replayed from a captured CUDA graph; installing the adapters before capture
places the delta inside the graph, and because the optimizer writes A and B in
place, the replay reads the new values without recapture. Merging would instead
need either a pristine copy of every adapted matrix or incremental
`W += dW(t) - dW(t-1)` bookkeeping, which accumulates bf16 rounding error over
many updates and makes the reset inexact.

Per-request slots
-----------------
With `pool_size > 1`, A and B are `[P, ...]` tensors and row i of a forward
uses slot `rows[i]`: the slot's factors are gathered with `index_select` and
applied with a batched matmul. A row touches only its own slot, so one batched
backward yields every request's gradient without cross-talk. The slot index is
the request's `req_pool_idx`, and a slot is reset when a new request takes it
over.

Choosing targets
----------------
Two properties of a single-layer EAGLE-3 draft determine which matrices are
suitable targets:

**KV-cache consistency.** The draft's own KV cache is written every round and
read by every later round. An update to anything that feeds it (`k`, `v`, `fc`,
the input norms, `embed_tokens`) leaves the entries cached before the update
inconsistent with those written after it, and the loss only observes the
current block.

**Attention backward.** With one decoder layer, a target at or above the
attention output (`o`, `gate_up`, `down`, `lm_head`) needs no gradient path
through attention, so the fused attention kernel stays in place. A target below
it (`q`, `k`, `v`, `fc`) requires `eagle_tts._sdpa_attention`, which rebuilds
attention over the full prefix K/V once per rollout step.
`lora_needs_differentiable_attention()` reports which case a target set is in.
"""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)

# Every matrix of an EAGLE-3 draft that can carry an adapter. `q`/`k`/`v` are
# slices of the fused `qkv_proj`; `gate_up` is the whole fused `gate_up_proj`.
VALID_LORA_TARGETS: Tuple[str, ...] = (
    "q",
    "k",
    "v",
    "o",
    "gate_up",
    "down",
    "fc",
    "lm_head",
)

# Default: every matrix that leaves the draft KV cache consistent. `q` is the
# only member that requires the differentiable-attention path (see the module
# docstring).
DEFAULT_LORA_TARGETS: str = "q,o,gate_up,down,lm_head"

# Targets whose gradient must be backpropagated through the attention op, so
# the fused attention kernel has to be replaced by a differentiable rebuild.
_BELOW_ATTENTION = frozenset({"q", "k", "v", "fc"})

# Targets that change what is written into the draft KV cache, making entries
# already in the cache inconsistent with later ones.
_KV_CACHE_AFFECTING = frozenset({"k", "v", "fc"})

# 1-D RMSNorm weights downstream of attention. A low-rank factorization does
# not apply to a vector, so they are updated in full, and only when the pool is
# not in use (see `EagleLoRAAdapterSet`). Names are relative to the draft model
# root.
POST_ATTENTION_NORMS: Tuple[str, ...] = (
    "model.layers.0.post_attention_layernorm.weight",
    "model.norm.weight",
)


def parse_lora_targets(spec: str) -> Tuple[str, ...]:
    """Parse and validate a comma-separated target list."""
    targets = tuple(t.strip() for t in spec.split(",") if t.strip())
    if not targets:
        raise ValueError(
            "EAGLE3 LoRA-TTS targets must be a non-empty list; valid targets: "
            f"{VALID_LORA_TARGETS}."
        )
    seen = set()
    for t in targets:
        if t not in VALID_LORA_TARGETS:
            raise ValueError(
                f"Invalid EAGLE3 LoRA-TTS target {t!r}. Valid targets: "
                f"{VALID_LORA_TARGETS}."
            )
        if t in seen:
            raise ValueError(f"Duplicate EAGLE3 LoRA-TTS target {t!r}.")
        seen.add(t)
    return targets


def lora_needs_differentiable_attention(targets: Sequence[str]) -> bool:
    """True if any target's gradient must flow back through attention.

    When False, the caller can leave the fused attention kernel in place and
    skip `_patch_attention_for_grad`, which avoids the full-prefix K/V gather
    and SDPA rebuild that otherwise run once per rollout step.
    """
    return any(t.split(".", 1)[0] in _BELOW_ATTENTION for t in targets)


def lora_kv_cache_affecting(targets: Sequence[str]) -> Tuple[str, ...]:
    """Targets that make already-cached draft K/V stale (used for a warning)."""
    return tuple(t for t in targets if t.split(".", 1)[0] in _KV_CACHE_AFFECTING)


def _init_a_template(
    name: str, rank: int, in_features_full: int, dtype: torch.dtype, seed: int
) -> torch.Tensor:
    """Kaiming-uniform A for the full (unsharded) input dim, deterministically.

    Generated on CPU from a name-derived seed, so every TP rank produces the
    identical full tensor and then takes its own slice. Drawing from one shared
    generator would make the init depend on module traversal order. `hash()` is
    not used because it is salted per process.
    """
    digest = 0
    for ch in name:
        digest = (digest * 131 + ord(ch)) & 0xFFFFFFFF
    g = torch.Generator(device="cpu")
    g.manual_seed((seed * 1000003 + digest) & 0x7FFFFFFF)
    bound = 1.0 / math.sqrt(in_features_full)
    a = torch.empty(rank, in_features_full, dtype=torch.float32)
    a.uniform_(-bound, bound, generator=g)
    return a.to(dtype)


class PoolAdam:
    """Adam over pooled parameters, stepping only the slots that fired.

    `torch.optim.Adam` keeps one scalar step count per parameter tensor. Here
    every slot needs its own, because requests start at different times and
    bias correction depends on the number of steps a slot has taken.

    Applies the standard Adam update to `master[slots]` only, with `master` and
    the moments as `[P, ...]` tensors. Slots that did not fire keep their
    weights, moments and step counts.

    State precision (`fp32_state`):

    * True (default): the master and both moments are fp32 tensors. The pool
      parameters (bf16) are a rounded copy of the master, refreshed after
      every step. An update smaller than the spacing of a bf16 weight stays in
      the master and takes effect once the accumulated change crosses a
      rounding boundary.
    * False: the moments are bf16 tensors and the pool parameters are their
      own master (`self.master` is `self.params`), so no fp32 tensor outlives
      a step. A step widens the fired rows to fp32, runs the same fp32
      arithmetic, and rounds each result once when it is stored. An update
      smaller than half the spacing of a bf16 weight is rounded away, and the
      second moment stops decaying; see "Precision" in
      EAGLE3_PER_REQUEST_LORA_DESIGN.md. The pool parameters must be bf16
      (`EagleTTSState` checks this at startup).
    """

    def __init__(
        self, params, pool_size, lr, betas=(0.9, 0.999), eps=1e-8, fp32_state=True
    ):
        self.params = list(params)
        self.lr = float(lr)
        self.b1, self.b2 = betas
        self.eps = float(eps)
        self.fp32_state = bool(fp32_state)
        if self.fp32_state:
            self.master = [p.detach().float().clone() for p in self.params]
        else:
            self.master = self.params
        self.m = [torch.zeros_like(x) for x in self.master]
        self.v = [torch.zeros_like(x) for x in self.master]
        self.step_ct = torch.zeros(
            pool_size, dtype=torch.int64, device=self.params[0].device
        )

    @torch.no_grad()
    def step(self, slots: torch.Tensor) -> None:
        """Adam on `slots` (a 1-D LongTensor of pool rows)."""
        if slots.numel() == 0:
            return
        # Duplicate slots mean two requests were mapped to one pool row, whose
        # gradients would then be blended into a single update. Raise so the
        # upstream mapping error is visible.
        if slots.unique().numel() != slots.numel():
            raise RuntimeError(
                f"EAGLE3 LoRA-TTS: duplicate slots in one Adam step "
                f"({slots.tolist()}); two requests are mapped to one pool row."
            )
        live = [
            (p, mst, m, v)
            for p, mst, m, v in zip(self.params, self.master, self.m, self.v)
            if p.grad is not None
        ]
        if not live:
            # No parameter received a gradient: advance nothing. Incrementing
            # step_ct here would offset the slot's bias correction, so its
            # first real update would be computed at t=k+1 instead of t=1.
            return

        # Per-slot liveness. `live` only says that backward reached some slot.
        # A request that contributed no rows to this round's loss (for
        # example, when only accepted positions are trained and every drafted
        # node was rejected) has a gradient of exactly zero. Stepping it anyway
        # would advance its step count and apply an update from decayed
        # momentum alone, with no new signal. The mask is computed on device:
        # selecting the live subset would make the shapes data-dependent and
        # force a host sync.
        active = torch.zeros(slots.numel(), device=slots.device, dtype=torch.bool)
        for p, _, _, _ in live:
            active |= p.grad[slots].flatten(1).abs().sum(1) > 0
        step = active.to(self.step_ct.dtype)
        self.step_ct[slots] += step
        # Slots that have never stepped would give bc1 = 1 - b1^0 = 0 and divide
        # by zero; their update is masked out anyway, so floor t at 1.
        t = self.step_ct[slots].clamp(min=1).to(torch.float32)   # [K]
        bc1 = 1.0 - self.b1 ** t
        bc2 = 1.0 - self.b2 ** t

        for p, mst, m, v in live:
            g = p.grad[slots].float()
            # Broadcast the per-slot corrections over this tensor's trailing dims.
            shape = (-1,) + (1,) * (g.dim() - 1)
            keep = active.view(shape)
            # The arithmetic is fp32 for both state precisions. `.float()`
            # returns fp32 rows unchanged and widens bf16 rows exactly, and the
            # results are rounded once when they are stored below, so an
            # inactive row keeps its bits.
            m_old = m[slots].float()
            v_old = v[slots].float()
            w_old = mst[slots].float()
            m_new = torch.where(keep, self.b1 * m_old + (1 - self.b1) * g, m_old)
            v_new = torch.where(keep, self.b2 * v_old + (1 - self.b2) * g * g, v_old)
            upd = (m_new / bc1.view(shape)) / (
                (v_new / bc2.view(shape)).sqrt() + self.eps
            )
            w_new = torch.where(keep, w_old - self.lr * upd, w_old)
            m[slots] = m_new.to(m.dtype)
            v[slots] = v_new.to(v.dtype)
            mst[slots] = w_new.to(mst.dtype)
            if mst is not p:
                p[slots] = mst[slots].to(p.dtype)

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = True) -> None:
        """Same interface as `torch.optim.Optimizer.zero_grad`."""
        for p in self.params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    @torch.no_grad()
    def reset_slots(self, slots: torch.Tensor, params_after_reset) -> None:
        """Clear optimizer state for slots that are handed to new requests.

        Without this a new request would inherit the previous request's
        momentum and step count. `params_after_reset` are the pool parameters
        after the adapter reset; the master rows are re-read from them. With
        bf16 state the parameters are the master, so there is nothing to re-read.
        """
        self.step_ct[slots] = 0
        for mst, m, v, p in zip(self.master, self.m, self.v, params_after_reset):
            m[slots] = 0
            v[slots] = 0
            if mst is not p:
                mst[slots] = p[slots].to(mst.dtype)


class _RowSelector:
    """Maps each row of a forward to its adapter slot, in a CUDA-graph-safe way.

    The module wrappers only receive `input_`, not the ForwardBatch, so the
    current batch's row -> slot mapping is published here before the forward
    and read at call time.

    The index tensor is a static buffer written in place. A captured graph
    records the buffer's address, so a freshly allocated index tensor per round
    would be ignored on replay, and every replay would keep using the mapping
    that was live at capture.

    `active` selects the code path. False means one shared slot (pool_size 1):
    `rows_for` returns None and the adapter takes its plain-matmul path. The
    flag is fixed at construction rather than switched on by the first
    `set_rows`, because `LoRAAdapter.delta` branches on it in Python and CUDA
    graph capture runs at worker init, before any mapping has been published.
    The branch taken during capture is fixed for the life of the graph: had
    capture traced the plain `a[0]`/`b[0]` path, every replayed round would
    draft from slot 0, the unused padding row, and the per-request adapters
    would take effect only on the eager firing rounds.
    """

    def __init__(self, max_rows: int, device, active: bool = False):
        self.buf = torch.zeros(max_rows, dtype=torch.long, device=device)
        self.n = 0
        self.active = bool(active)

    @torch.no_grad()
    def set_rows(self, slots: torch.Tensor) -> None:
        n = slots.numel()
        if n > self.buf.numel():
            raise RuntimeError(
                f"EAGLE3 LoRA-TTS: {n} rows exceeds the adapter pool's "
                f"{self.buf.numel()} slots. Raise the pool size (it is sized "
                "from max_running_requests)."
            )
        self.buf[:n].copy_(slots.to(self.buf.dtype), non_blocking=True)
        self.n = n

    def rows_for(self, x: torch.Tensor):
        if not self.active:
            return None
        n = x.shape[0]
        if n > self.buf.numel():
            raise RuntimeError(
                f"EAGLE3 LoRA-TTS: forward has {n} rows, more than the row "
                f"buffer holds ({self.buf.numel()})."
            )
        if self.n == 0:
            # Nothing has been published yet: this is CUDA graph
            # initialization, either the eager warmup passes that run before
            # capture or the capture itself. Both call the draft forward
            # directly, without a bind. Return the static buffer so the traced
            # gather uses it as its index operand; a replay then reads whatever
            # the live round wrote into it.
            #
            # The condition is `n == 0` rather than
            # `is_current_stream_capturing()`, which is true only inside
            # `torch.cuda.graph(...)` and does not cover the warmup passes.
            # A real batch has at least one row, so `n == 0` does not recur
            # once serving starts. That relies on two guards together:
            # `bind_rows` skips empty publishes, and 0-row (idle) batches exist
            # only under DP attention, which server_args.py rejects in
            # combination with TTS. If that restriction is relaxed, this
            # branch must be revisited.
            return self.buf[:n]
        # After the first publish, a forward must have exactly the published
        # number of rows. A mismatch means the mapping was not refreshed for
        # this forward, and rows would be paired with the wrong slots.
        if n != self.n:
            raise RuntimeError(
                f"EAGLE3 LoRA-TTS: forward has {n} rows but the slot mapping "
                f"holds {self.n}. The mapping was not refreshed for this batch."
            )
        if torch.is_grad_enabled() and not torch.cuda.is_current_stream_capturing():
            # index_select saves its index for backward and version-checks it
            # there, but this buffer is rewritten in place by the next bind
            # (the extend path rebinds within the same round), which would
            # make backward fail with "modified by an inplace operation". A
            # forward under grad therefore gets a snapshot. Capture keeps the
            # static buffer, since a captured graph must re-read the live
            # mapping on every replay.
            return self.buf[:n].clone()
        return self.buf[:n]


class LoRAAdapter(nn.Module):
    """One low-rank adapter: delta = scale * (x @ A^T) @ B^T.

    `A` is [rank, in_local] and `B` is [out_local, rank]. Which of those dims is
    sharded depends on the wrapped layer's TP layout; `a_shard_offset` records
    where this rank's slice of the conceptually-full A starts, so the init is
    rank-symmetric and reproducible.
    """

    def __init__(
        self,
        *,
        name: str,
        in_features_full: int,
        in_features_local: int,
        a_shard_offset: int,
        out_features_local: int,
        rank: int,
        alpha: float,
        dtype: torch.dtype,
        device: torch.device,
        seed: int,
        tp_size: int,
        a_grad_is_partial: bool = False,
        layout: str = "column",
        pool_size: int = 1,
        rows: "_RowSelector | None" = None,
    ):
        super().__init__()
        self.name = name
        # column | lm_head -> A replicated, B sharded
        # row              -> A sharded,    B replicated
        # plain (fc)       -> both replicated
        self.layout = layout
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.tp_size = int(tp_size)
        # Column-parallel layouts keep A replicated but B sharded, so
        # `dL/dA_r = (dL/d_delta_r @ B_r)^T x` is rank-local and the true
        # gradient is the sum over ranks. Every other trainable is either
        # sharded (locally correct) or receives a gradient that is already
        # identical on all ranks once the `f` conjugate is applied; reducing
        # those would scale them by tp_size.
        self.a_grad_is_partial = bool(a_grad_is_partial) and self.tp_size > 1

        template = _init_a_template(name, self.rank, in_features_full, dtype, seed)
        self._a_template = template[
            :, a_shard_offset : a_shard_offset + in_features_local
        ].contiguous()

        # One slot per concurrent request. pool_size=1 is the shared-adapter
        # case and is mathematically identical to a single unpooled adapter.
        self.pool_size = int(pool_size)
        self._rows = rows

        # A starts Kaiming-uniform and B at zero, so the delta is exactly zero
        # before the first update: with TTS enabled and no update applied yet,
        # the draft is bit-identical to the pretrained one.
        self.a = nn.Parameter(
            self._a_template.clone().to(device).unsqueeze(0).repeat(self.pool_size, 1, 1)
        )
        self.b = nn.Parameter(
            torch.zeros(
                self.pool_size, out_features_local, self.rank,
                dtype=dtype, device=device,
            )
        )

    def delta(self, x: torch.Tensor, reduce_hidden: bool = False) -> torch.Tensor:
        """Low-rank delta for input `x`, using each row's own adapter slot.

        `reduce_hidden` is for row-parallel layers, whose input is sharded: each
        rank holds a partial `x @ A^T`, and the true product needs the sum over
        ranks. Reducing the [*, rank] intermediate rather than the [*, out]
        result keeps that collective at `rank` floats per token instead of
        `out_features`.
        """
        rows = self._rows.rows_for(x) if self._rows is not None else None

        if rows is None:
            # Every row shares slot 0 (pool_size 1): plain matmuls, no gather.
            h = F.linear(x.to(self.a.dtype), self.a[0])
            h = self._maybe_reduce(h, reduce_hidden)
            return F.linear(h, self.b[0]) * self.scale

        # Per-request: row i uses slot rows[i] and touches no other slot, so a
        # single batched backward yields correct per-request gradients.
        #
        # `torch.ops.aten.bmm` rather than `torch.bmm`: deterministic inference
        # mode (batch_invariant_ops.enable_batch_invariant_mode) replaces
        # `torch.bmm` with a Python-level function that calls a Triton kernel
        # directly. That bypasses the dispatcher and therefore autograd, so the
        # output would have no grad_fn and the update would receive no
        # gradient. The aten op goes through the dispatcher: autograd records
        # the node and then dispatches to the same batch-invariant
        # implementation, which that mode also registers for `aten::bmm`, so
        # determinism is preserved.
        xb = x.to(self.a.dtype).unsqueeze(1)                 # [N,1,in]
        h = torch.ops.aten.bmm(
            xb, self.a.index_select(0, rows).transpose(1, 2)
        )  # [N,1,r]
        h = self._maybe_reduce(h, reduce_hidden)
        out = torch.ops.aten.bmm(
            h, self.b.index_select(0, rows).transpose(1, 2)
        )  # [N,1,out]
        return out.squeeze(1) * self.scale

    def _maybe_reduce(self, h, reduce_hidden):
        if not (reduce_hidden and self.tp_size > 1):
            return h
        # Looked up at call time so that the differentiable replacement
        # installed by `tts_grad_mode` is used during a TTS forward.
        from sglang.srt.layers import linear as _linear

        return _linear.tensor_model_parallel_all_reduce(h)

    @torch.no_grad()
    def reset(self, slots: "torch.Tensor | None" = None) -> None:
        """Return slots to the pretrained function (A <- template, B <- 0).

        `slots=None` resets the whole pool. The per-slot reset is what makes
        slot reuse safe: sglang recycles `req_pool_idx` values, so a slot is
        later handed to another request, which must not inherit the previous
        request's adaptation.
        """
        tmpl = self._a_template.to(self.a.device)
        if slots is None:
            self.a.copy_(tmpl.unsqueeze(0).expand_as(self.a))
            self.b.zero_()
        else:
            self.a[slots] = tmpl
            self.b[slots] = 0

    def num_parameters(self) -> int:
        """Trainable elements for one slot, i.e. the per-request cost."""
        return self.a[0].numel() + self.b[0].numel()


@dataclass
class _Slot:
    """One adapter's placement on a module."""

    name: str
    in_full: int
    in_local: int
    a_offset: int
    out_local: int
    out_slice: Optional[Tuple[int, int]] = None  # fused sub-projection, local coords


@dataclass
class _ModuleSpec:
    """A module to wrap, plus every adapter slot that lands on it."""

    module: nn.Module
    kind: str  # column | row | plain | lm_head
    slots: List[_Slot] = field(default_factory=list)


def _install(module: nn.Module, fn, adapters: List[LoRAAdapter]) -> None:
    if getattr(module, "_tts_lora", None) is not None:
        raise RuntimeError(
            "EAGLE3 LoRA-TTS: module already carries adapter(s) "
            f"{[a.name for a in module._tts_lora]}."
        )
    module._tts_lora = adapters
    module.forward = fn


def _wrap_column_parallel(
    module: nn.Module, pairs: List[Tuple[LoRAAdapter, Optional[Tuple[int, int]]]]
) -> None:
    """Column-parallel (`qkv_proj`, `gate_up_proj`): input replicated, output sharded.

    Takes a list because `qkv_proj` is one fused module shared by the q/k/v
    targets: selecting more than one of them adds several slice-local deltas to
    a single output rather than wrapping the module several times.
    """
    base = module.forward

    def forward(input_, *args, **kwargs):
        # The `f` conjugate is applied here rather than by the class-level
        # patch in `_patch_column_parallel_input_grad`: `base` is a bound
        # method captured at install time, before that patch exists, so a
        # wrapped module bypasses it. The class patch covers the unwrapped
        # column-parallel layers; each layer gets exactly one of the two, so
        # the input gradient is never all-reduced twice.
        from sglang.srt.speculative.eagle_tts import all_reduce_input_grad

        input_ = all_reduce_input_grad(input_)
        out, bias = base(input_, *args, **kwargs)
        width = out.shape[-1]
        for adapter, sl in pairs:
            delta = adapter.delta(input_).to(out.dtype)
            if sl is not None:
                # Zero-pad the delta to full width instead of writing into a
                # slice of `out`: `out` may alias a CUDA-graph static buffer,
                # so an in-place write is unsafe, and F.pad is a single
                # differentiable op.
                delta = F.pad(delta, (sl[0], width - sl[1]))
            out = out + delta
        return out, bias

    _install(module, forward, [a for a, _ in pairs])


def _wrap_row_parallel(module: nn.Module, adapter: LoRAAdapter) -> None:
    """Row-parallel (`o_proj`, `down_proj`): input sharded, output all-reduced."""
    base = module.forward

    def forward(input_, *args, **kwargs):
        skip = kwargs.get("skip_all_reduce", False)
        if not skip and len(args) >= 1:
            skip = bool(args[0])
        if skip:
            # With skip_all_reduce the caller reduces the output itself, so
            # the already-reduced delta added here would be summed tp_size
            # times. Nothing on the EAGLE-3 draft path calls the layer this
            # way (llama_eagle3 calls `self.mlp(hidden_states)` without a
            # reduce-scatter), so raise rather than return a scaled result.
            raise RuntimeError(
                f"EAGLE3 LoRA-TTS: {adapter.name} was called with "
                "skip_all_reduce=True, which this adapter does not support."
            )
        out, bias = base(input_, *args, **kwargs)
        delta = adapter.delta(input_, reduce_hidden=True)
        return out + delta.to(out.dtype), bias

    _install(module, forward, [adapter])


def _wrap_plain_linear(module: nn.Module, adapter: LoRAAdapter) -> None:
    """Replicated `torch.nn.Linear` (`fc`): returns a bare tensor, not a tuple."""
    base = module.forward

    def forward(input_, *args, **kwargs):
        out = base(input_, *args, **kwargs)
        return out + adapter.delta(input_).to(out.dtype)

    _install(module, forward, [adapter])


def _wrap_lm_head(module: nn.Module, adapter: LoRAAdapter) -> None:
    """Vocab-parallel head.

    `LogitsProcessor._compute_lm_head` dispatches to `lm_head(hidden_states)`
    when the module carries `set_lora`/`apply_lora`, so installing those
    sentinels routes the head through this forward. The base branch it would
    otherwise take is `matmul(hidden_states.to(w.dtype), w.T)`, reproduced here.
    The delta lands in this rank's vocab shard, before the all-gather.
    """
    weight = module.weight

    def forward(hidden_states, *args, **kwargs):
        # No `f` conjugate here, unlike `_wrap_column_parallel`.
        # `eagle_tts._patch_column_parallel_input_grad` patches
        # `LogitsProcessor._compute_lm_head`, which applies the conjugate and
        # then dispatches into this wrapper through the set_lora/apply_lora
        # sentinels. Applying it again here would all-reduce dL/d(hidden)
        # twice and scale every gradient upstream of the head by tp_size. The
        # column-parallel wrappers are the opposite case: their captured
        # `base` bypasses the class patch, so they apply the conjugate
        # themselves.
        logits = torch.matmul(hidden_states.to(weight.dtype), weight.T)
        return logits + adapter.delta(hidden_states).to(logits.dtype)

    _install(module, forward, [adapter])
    # Sentinels read by LogitsProcessor._compute_lm_head.
    module.set_lora = True
    module.apply_lora = True


class EagleLoRAAdapterSet:
    """All adapters installed on one EAGLE3 draft model.

    Construction installs the adapters, and must happen before CUDA graph
    capture: the adapter ops have to be inside the captured graph, and
    installation rebinds `module.forward`, which a graph captured earlier would
    not reflect.
    """

    def __init__(
        self,
        *,
        draft_model: nn.Module,
        targets: Sequence[str],
        rank: int,
        alpha: float,
        tp_size: int = 1,
        tp_rank: int = 0,
        seed: int = 0,
        train_post_attention_norms: bool = True,
        pool_size: int = 1,
        max_rows: int = 0,
        head_rows_per_req: int = 1,
        fp32_master: bool = True,
    ):
        self.draft_model = draft_model
        # pool_size == 1: one adapter shared by every row, applied with plain
        # matmuls. pool_size > 1: one slot per concurrent request, which is
        # what makes batched serving correct.
        self.pool_size = int(pool_size)
        # The selector buffer is sized by rows, not slots. The rollout has one
        # row per request, but the two draft-extend paths write several tokens
        # per request, so their forwards can have more rows than the pool has
        # slots.
        self.rows = _RowSelector(
            max(1, self.pool_size, int(max_rows or 0)),
            next(draft_model.parameters()).device,
            active=self.pool_size > 1,
        )
        # Separate selector for lm_head. LogitsProcessor selects the logprob
        # positions before the head, so the head's rows are per-request blocks
        # (one row per request at prefill and in the rollout,
        # num_tokens_for_logprob rows per request in the decode extend), not
        # the body's token rows. Each call site publishes the expansion it
        # knows; this class does not infer it. Sized for the widest head
        # forward: pool_size * head_rows_per_req.
        self.head_rows = _RowSelector(
            max(1, self.pool_size * max(1, int(head_rows_per_req or 1))),
            next(draft_model.parameters()).device,
            active=self.pool_size > 1,
        )
        self.targets = tuple(targets)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.tp_size = int(tp_size)
        self.tp_rank = int(tp_rank)
        self.adapters: "OrderedDict[str, LoRAAdapter]" = OrderedDict()
        self._norm_params: List[Tuple[str, torch.Tensor]] = []
        self._norm_templates: Dict[str, torch.Tensor] = {}

        if self.rank <= 0:
            raise ValueError(f"EAGLE3 LoRA-TTS rank must be positive, got {rank}.")

        specs = self._resolve(draft_model)
        resolved = {s.name.split(".", 1)[0] for spec in specs for s in spec.slots}
        missing = [t for t in self.targets if t not in resolved]
        if missing:
            raise RuntimeError(
                f"EAGLE3 LoRA-TTS: could not locate target(s) {missing} on the "
                f"draft model ({type(draft_model).__name__}). Resolved: "
                f"{sorted(resolved)}."
            )

        device = next(draft_model.parameters()).device
        dtype = next(draft_model.parameters()).dtype

        for spec in specs:
            made: List[Tuple[LoRAAdapter, Optional[Tuple[int, int]]]] = []
            for slot in spec.slots:
                adapter = LoRAAdapter(
                    name=slot.name,
                    in_features_full=slot.in_full,
                    in_features_local=slot.in_local,
                    a_shard_offset=slot.a_offset,
                    out_features_local=slot.out_local,
                    rank=self.rank,
                    alpha=self.alpha,
                    dtype=dtype,
                    device=device,
                    seed=seed,
                    tp_size=self.tp_size,
                    a_grad_is_partial=spec.kind in ("column", "lm_head"),
                    layout=spec.kind,
                    pool_size=self.pool_size,
                    # lm_head runs after LogitsProcessor has selected the
                    # logprob positions, so its rows differ from the body's
                    # per-token rows and it uses its own selector.
                    rows=self.head_rows if spec.kind == "lm_head" else self.rows,
                )
                self.adapters[slot.name] = adapter
                made.append((adapter, slot.out_slice))

            if spec.kind == "column":
                _wrap_column_parallel(spec.module, made)
            elif spec.kind == "row":
                _wrap_row_parallel(spec.module, made[0][0])
            elif spec.kind == "plain":
                _wrap_plain_linear(spec.module, made[0][0])
            elif spec.kind == "lm_head":
                _wrap_lm_head(spec.module, made[0][0])
            else:  # pragma: no cover - _resolve emits only the four above
                raise AssertionError(spec.kind)

        # Freeze the base weights. This prunes the backward graph and keeps the
        # target-aliased tensors (embed_tokens, and lm_head for drafts that
        # borrow the target's) unmodified without cloning them.
        for p in draft_model.parameters():
            if not p.is_inference():
                p.requires_grad_(False)
        for a in self.adapters.values():
            a.a.requires_grad_(True)
            a.b.requires_grad_(True)

        if train_post_attention_norms and self.pool_size > 1:
            # The norms are plain 1-D model parameters, not pooled: updating
            # them with several requests in flight would let one request's
            # update change another request's drafts. Pooling them would need
            # a per-row gather inside RMSNorm, so they stay frozen when the
            # pool is active. As a result the pooled and unpooled
            # configurations do not train exactly the same parameter set.
            train_post_attention_norms = False
            logger.info(
                "EAGLE3 LoRA-TTS: post-attention norm training disabled because "
                "the adapter pool is active (pool_size=%d); norms are shared "
                "across requests and cannot be trained per request.",
                self.pool_size,
            )
        if train_post_attention_norms:
            self._claim_post_attention_norms(draft_model)

        stale = lora_kv_cache_affecting(self.targets)
        if stale:
            logger.warning(
                "EAGLE3 LoRA-TTS: target(s) %s change what is written into the "
                "draft KV cache. Entries cached before an update stay as they "
                "were, so the prefix becomes inconsistent with later positions. "
                "k/v/fc adapters are not recommended.",
                list(stale),
            )
        # Report bytes as well as element counts. The resident state per
        # element is 14 bytes with fp32 masters (bf16 pool + fp32 master + two
        # fp32 Adam moments), seven times the size of the bf16 adapter itself,
        # and 6 bytes with bf16 optimizer state (bf16 pool, which is its own
        # master, + two bf16 Adam moments).
        bytes_per_element = 14 if fp32_master else 6
        resident_gb = self.total_parameters() * bytes_per_element / 1e9
        logger.info(
            "EAGLE3 LoRA-TTS: %d adapter(s) %s rank=%d alpha=%.1f | "
            "%.3fM params/slot x %d slot(s) = %.3fM resident/rank "
            "(~%.2f GB/rank at %d bytes/element, optimizer state included) + "
            "%d norm params | differentiable attention: %s",
            len(self.adapters),
            list(self.adapters),
            self.rank,
            self.alpha,
            self.num_parameters() / 1e6,
            self.pool_size,
            self.total_parameters() / 1e6,
            resident_gb,
            bytes_per_element,
            sum(p.numel() for _, p in self._norm_params),
            self.needs_differentiable_attention(),
        )

    # ------------------------------------------------------------------ layout

    def _resolve(self, draft_model: nn.Module) -> List[_ModuleSpec]:
        """Locate each requested target and record its TP layout.

        Slots are grouped per module so a fused module shared by several targets
        (`qkv_proj` for q/k/v) is wrapped exactly once.
        """
        specs: List[_ModuleSpec] = []
        want = set(self.targets)
        model = getattr(draft_model, "model", draft_model)

        if "fc" in want and hasattr(model, "fc"):
            fc = model.fc
            # Plain torch.nn.Linear: replicated on every rank.
            specs.append(
                _ModuleSpec(
                    fc,
                    "plain",
                    [_Slot("fc", fc.in_features, fc.in_features, 0, fc.out_features)],
                )
            )

        if "lm_head" in want and hasattr(draft_model, "lm_head"):
            head = draft_model.lm_head
            if head is getattr(model, "embed_tokens", None):
                # A tied-embedding draft sets `lm_head = model.embed_tokens`,
                # the same module object. Wrapping its forward would replace
                # the embedding lookup with a logits matmul, so raise with the
                # reason instead. (A draft with a head of its own does not take
                # this branch.)
                raise RuntimeError(
                    "EAGLE3 LoRA-TTS: this draft ties lm_head to embed_tokens "
                    "(tie_word_embeddings=True), so 'lm_head' cannot be adapted "
                    "without also replacing the embedding lookup. Drop 'lm_head' "
                    "from --speculative-eagle-tts-lora-targets."
                )
            from sglang.srt.server_args import get_global_server_args

            if getattr(get_global_server_args(), "enable_fp32_lm_head", False):
                # `_compute_lm_head` checks the LoRA sentinels before its fp32
                # branch, so a wrapped head would run in bf16 and the
                # fp32-head option would be ignored without an error.
                raise RuntimeError(
                    "EAGLE3 LoRA-TTS: 'lm_head' adapter is incompatible with "
                    "--enable-fp32-lm-head (the adapter path would silently "
                    "replace the fp32 head with a bf16 matmul)."
                )
            w = head.weight
            specs.append(
                _ModuleSpec(
                    head,
                    "lm_head",
                    [_Slot("lm_head", w.shape[1], w.shape[1], 0, w.shape[0])],
                )
            )

        layers = getattr(model, "layers", None) or []
        if len(layers) > 1:
            logger.warning(
                "EAGLE3 LoRA-TTS: draft has %d layers; adapters are installed on "
                "every layer, but the attention/KV-cache reasoning behind the "
                "default target set assumes a single layer.",
                len(layers),
            )
        for li, layer in enumerate(layers):
            sfx = "" if len(layers) == 1 else f".{li}"
            attn = getattr(layer, "self_attn", None)
            mlp = getattr(layer, "mlp", None)

            if attn is not None:
                qkv = attn.qkv_proj
                in_full = qkv.weight.shape[1]
                q_size, kv_size = attn.q_size, attn.kv_size
                spans = (
                    ("q", 0, q_size),
                    ("k", q_size, q_size + kv_size),
                    ("v", q_size + kv_size, q_size + 2 * kv_size),
                )
                slots = [
                    _Slot(kind + sfx, in_full, in_full, 0, hi - lo, (lo, hi))
                    for kind, lo, hi in spans
                    if kind in want
                ]
                if slots:
                    specs.append(_ModuleSpec(qkv, "column", slots))
                if "o" in want:
                    o = attn.o_proj
                    in_local = o.weight.shape[1]
                    specs.append(
                        _ModuleSpec(
                            o,
                            "row",
                            [
                                _Slot(
                                    "o" + sfx,
                                    in_local * self.tp_size,
                                    in_local,
                                    self.tp_rank * in_local,
                                    o.weight.shape[0],
                                )
                            ],
                        )
                    )

            if mlp is not None:
                if "gate_up" in want:
                    gu = mlp.gate_up_proj
                    specs.append(
                        _ModuleSpec(
                            gu,
                            "column",
                            [
                                _Slot(
                                    "gate_up" + sfx,
                                    gu.weight.shape[1],
                                    gu.weight.shape[1],
                                    0,
                                    gu.weight.shape[0],
                                )
                            ],
                        )
                    )
                if "down" in want:
                    dn = mlp.down_proj
                    in_local = dn.weight.shape[1]
                    specs.append(
                        _ModuleSpec(
                            dn,
                            "row",
                            [
                                _Slot(
                                    "down" + sfx,
                                    in_local * self.tp_size,
                                    in_local,
                                    self.tp_rank * in_local,
                                    dn.weight.shape[0],
                                )
                            ],
                        )
                    )
        return specs

    def _claim_post_attention_norms(self, draft_model: nn.Module) -> None:
        named = dict(draft_model.named_parameters())
        for name in POST_ATTENTION_NORMS:
            p = named.get(name)
            if p is None or p.is_inference():
                continue
            p.requires_grad_(True)
            self._norm_params.append((name, p))
            self._norm_templates[name] = p.detach().clone()

    # ------------------------------------------------------------------- state

    def parameters(self) -> List[torch.Tensor]:
        """Trainable tensors, in a stable order (adapters, then norms)."""
        out: List[torch.Tensor] = []
        for adapter in self.adapters.values():
            out.append(adapter.a)
            out.append(adapter.b)
        out.extend(p for _, p in self._norm_params)
        return out

    def parameter_names(self) -> List[str]:
        out: List[str] = []
        for name in self.adapters:
            out.append(f"{name}.lora_a")
            out.append(f"{name}.lora_b")
        out.extend(n for n, _ in self._norm_params)
        return out

    def num_parameters(self) -> int:
        """Trainable adapter elements for one slot, per TP rank.

        This is the per-request cost. The resident total is this times
        `pool_size`; see `total_parameters()`.
        """
        return sum(a.num_parameters() for a in self.adapters.values())

    def total_parameters(self) -> int:
        """Adapter elements resident on this TP rank, over all slots."""
        return self.num_parameters() * self.pool_size

    def replicated_params(self) -> List[torch.Tensor]:
        """Trainables that must hold an identical value on every TP rank.

        Which factor is replicated depends on the layout: column-parallel keeps
        A replicated and shards B; row-parallel is the reverse; `fc` is a plain
        replicated Linear, so both are. The trained RMSNorm weights are 1-D and
        replicated. The gradient clip needs this list to compute a norm that is
        identical across ranks.
        """
        out: List[torch.Tensor] = []
        for a in self.adapters.values():
            if a.layout in ("column", "lm_head"):
                out.append(a.a)
            elif a.layout == "row":
                out.append(a.b)
            else:  # plain
                out.extend((a.a, a.b))
        out.extend(p for _, p in self._norm_params)
        return out

    def partial_grad_params(self) -> List[torch.Tensor]:
        """Tensors whose gradient is a per-rank partial sum and must be all-reduced.

        See `LoRAAdapter.a_grad_is_partial`. Empty at tp=1.
        """
        return [a.a for a in self.adapters.values() if a.a_grad_is_partial]

    @torch.no_grad()
    def reset(self, slots: "torch.Tensor | None" = None) -> None:
        """Return the draft to its pretrained function exactly.

        No base-weight snapshot is involved: A goes back to its template and B
        to zero, which makes every delta identically zero again. The trained
        norms are restored from their load-time values.

        `slots` restricts the reset to those pool rows, which is what a prefill
        does for the `req_pool_idx` values it takes over. The norms are not
        pooled, so they are restored only on a whole-pool reset; they are not
        trained at all when the pool is in use (see `__init__`).
        """
        for a in self.adapters.values():
            a.reset(slots)
        if slots is None:
            for name, p in self._norm_params:
                p.copy_(self._norm_templates[name])

    @torch.no_grad()
    def bind_rows(
        self,
        req_pool_indices: torch.Tensor,
        row_slots: "torch.Tensor | None" = None,
        head_slots: "torch.Tensor | None" = None,
    ) -> "torch.Tensor | None":
        """Publish a forward's row -> slot mapping. Returns the per-request slots.

        The slot is the `req_pool_idx`: sglang sizes `req_to_token_pool` from
        `max_running_requests` and hands out dense indices into it, so the
        identity mapping is collision-free across concurrent requests. That
        avoids a side table and any `.tolist()` on the decode path, which would
        sync device to host every round.

        `row_slots` carries the per-row mapping when a forward has more rows
        than requests: the two draft-extend paths write several tokens per
        request, so row i does not belong to request i there. None means one
        row per request, which is the rollout at topk=1. `head_slots` is the
        corresponding mapping for the rows `lm_head` sees.

        The return value is the per-request slot tensor the optimizer steps
        over. `PoolAdam` requires unique slots, so it must not be given the
        expanded per-row version.
        """
        if self.pool_size <= 1:
            return None
        if req_pool_indices.numel() == 0:
            # An idle round has no requests. Publishing an empty mapping would
            # set `n` back to 0, which `_RowSelector.rows_for` interprets as
            # CUDA graph initialization and exempts from the row-count check.
            # Keep the previous mapping; the idle forward's outputs are
            # discarded.
            return None
        self.rows.set_rows(req_pool_indices if row_slots is None else row_slots)
        # lm_head sees the rows left after logprob-position selection:
        # per-request blocks whose width the caller knows (1 at prefill and in
        # the rollout, num_tokens_for_logprob in the decode extend). The
        # caller passes the mapping; it is not inferred from tensor widths.
        self.head_rows.set_rows(
            req_pool_indices if head_slots is None else head_slots
        )
        return req_pool_indices

    def needs_differentiable_attention(self) -> bool:
        return lora_needs_differentiable_attention(self.targets)
