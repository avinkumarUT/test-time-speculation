"""Per-request LoRA test-time speculation (TTS) for the DFlash speculator.

During speculative decoding, the target model's verification pass yields the
target's distribution at every drafted position. TTS uses that signal to
update the draft model (the "speculator") online, while the response is being
generated. For batched serving the draft model's base weights stay frozen and
shared, and every in-flight request owns a set of LoRA adapters on the draft
model's linear projections. The adapters are trained by distilling the
target's verify logits. DFLASH_LORA_TTS_DESIGN.md in this directory describes
the design.

Main pieces
-----------
Adapter pool
    Each live request owns master weights and Adam moments (fp32, or bf16 with
    the bfloat16 optimizer dtype) and one row of a serving-dtype pool
    (`_ReqLoRAState`, `get`, `_refresh_cached`). Before
    every draft forward, `publish_static` copies the batch's pool rows into
    fixed buffers that the draft model reads (see `models/dflash.py`), so the
    same path works with and without draft CUDA graphs.
Capture of verified rounds
    A request "fires" on the rounds in which its adapters are updated
    (`should_fire`). On such a round `capture_for_replay` saves the block's
    input embeddings, positions and prefix KV locations before the draft
    forward, and the worker snapshots the target's verify logits for the
    same block.
Replay forward and loss
    `_replay_block_hidden` / `_replay_block_hidden_many` re-run the draft
    block forward eagerly with autograd. They read the frozen base weights in
    place and the prefix K/V from the draft KV cache as constants. The loss
    (`_compute_loss`, selected by `tts_loss`) is position-weighted: either the
    negative log acceptance probability, or a mix of KL(target || draft) and a
    self cross-entropy term.
Per-request clipping and optimizer
    Gradients are clipped per request and every request takes its own Adam
    step (`_adam_step`). By default the masters and moments are kept in pinned
    CPU memory between updates and are on the GPU only during an update. The
    arithmetic of an update is fp32 whatever the dtype of the stored state.
Asynchronous stepping
    `run_tts_steps` enqueues a round's updates on a side CUDA stream, where
    they overlap the next round's serving. A request is updated every `freq`
    rounds (the update stride), and requests are phase-staggered so that
    their updates fall on different rounds.
Tensor parallelism
    Adapters are sharded like their base layers. The replay uses
    differentiable collectives (`_CopyToTP`, `_AllReduceTP`,
    `_VocabAllGather`); completing the gradients and clipping them use
    all-reduces.
Delayed publish
    Optionally, adapters trained in round N become visible to drafting in
    round N+2 instead of N+1, so that an update never blocks the next round.
    Pool rows are double-buffered and the per-round fence waits only for the
    second-newest update.

Key invariants
--------------
* An adapter's B matrix starts at zero, so an adapter that has not been
  updated, and any all-zero buffer row, contributes an exactly-zero delta.
* With the default targets, the context K/V in the draft cache is produced by
  projections that are never adapted, so cached prefix K/V stay valid across
  updates.
* Adapter state is never read or reassigned while an update that may write it
  is in flight. Reads and reuse happen behind `wait_for_pending` /
  `fence_for_publish`, and frees are deferred until the writer is fenced.
* The side stream never reads the KV pool: prefix K/V are gathered on the main
  stream before an update is enqueued.
* With tp_size > 1, every rank makes the same host-side decisions and enqueues
  the same sequence of collectives. Failures in the TTS path are therefore
  raised. With tp_size == 1 they are contained, and TTS disables itself after
  repeated failures.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.speculative.tts_loss import lk_per_position

logger = logging.getLogger(__name__)

# Dedicated process group for the collectives of TTS updates. It is created
# only for delayed publish with tp_size > 1 (see DFlashLoRATTSState.__init__).
# In that mode the per-round fence leaves the newest update running, so
# serving collectives on the main stream execute while update collectives are
# still in flight on the side stream. Interleaving two streams on one NCCL
# communicator can deadlock or corrupt data. With a separate communicator for
# the updates, each communicator sees its collectives in the same order on
# every rank and on a single stream. When None, collectives go through SGLang's
# TP wrappers.
_TTS_PG = None


def _tts_all_reduce(x: torch.Tensor) -> torch.Tensor:
    """All-reduce (sum) for the collectives of TTS updates. Uses the dedicated
    group when one is installed (in place on `x`; every call site passes a
    fresh contiguous tensor), otherwise SGLang's TP wrapper."""
    if _TTS_PG is not None:
        torch.distributed.all_reduce(x, group=_TTS_PG)
        return x
    return tensor_model_parallel_all_reduce(x)


def _tts_all_gather(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """All-gather + concat along `dim` in rank order, matching
    tensor_model_parallel_all_gather's layout; dedicated group when
    installed."""
    if _TTS_PG is not None:
        ws = torch.distributed.get_world_size(group=_TTS_PG)
        outs = [torch.empty_like(x) for _ in range(ws)]
        torch.distributed.all_gather(outs, x.contiguous(), group=_TTS_PG)
        return torch.cat(outs, dim=dim)
    return tensor_model_parallel_all_gather(x, dim=dim)


class _CopyToTP(torch.autograd.Function):
    """Megatron f: identity forward, all-reduce(SUM) backward.

    Applied to replicated activations entering column-parallel projections in
    the replay so upstream gradients (previous layers' adapter leaves) sum
    the per-rank shard contributions instead of keeping only the local one.
    """

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        # clone(): an in-place NCCL reduce on an already-contiguous grad
        # would mutate an autograd-engine-owned buffer.
        return _tts_all_reduce(grad.contiguous().clone())


class _AllReduceTP(torch.autograd.Function):
    """Megatron g: all-reduce(SUM) forward, identity backward.

    Used for row-parallel partial sums in the replay. The inference
    all-reduce can dispatch to out-of-place custom collectives with no
    backward (severing the graph), so the reduce is wrapped explicitly.
    """

    @staticmethod
    def forward(ctx, x):
        return _tts_all_reduce(x.contiguous().clone())

    @staticmethod
    def backward(ctx, grad):
        return grad


class _VocabAllGather(torch.autograd.Function):
    """Gather vocab-sharded draft logits to full vocab, differentiably.

    Forward concatenates every rank's shard on the last dim (rank-contiguous
    vocab ranges); backward slices out the local rank's columns (vocab
    columns are disjoint, no reduction needed).
    """

    @staticmethod
    def forward(ctx, local_logits):
        ctx.local_width = local_logits.shape[-1]
        ctx.rank = get_tensor_model_parallel_rank()
        return _tts_all_gather(local_logits.contiguous(), dim=-1)

    @staticmethod
    def backward(ctx, grad):
        start = ctx.rank * ctx.local_width
        return grad[..., start : start + ctx.local_width].contiguous()

# Projections that can carry adapters. The default target set is q, o, gate_up
# and down.
#
# k_proj, v_proj and k_norm are excluded by default because they have two
# roles: besides producing the block's own K/V they project the target's
# hidden states into the draft's context KV cache, and the TTS loss never
# observes that second role. fc and hidden_norm run only when context KV is
# written and have no gradient path, so they are never adapted.
#
# "k" and "v" can be requested explicitly. One adapter per (layer, k/v) is
# then applied wherever W_k / W_v is used: in the draft block forward and in
# the context-KV append (kv_proj_only path; the worker bypasses the fused KV
# materializer for those appends). This has two consequences. Cached context
# KV is not rewritten when adapters change, so each cached token keeps the
# adapter state of the time it was appended. And gradients reach the k/v
# adapters only through the block forward, because cached prefix K/V are
# constants in the replay. See DFLASH_LORA_TTS_DESIGN.md.
_VALID_TARGETS = ("q", "k", "v", "o", "gate_up", "down")

# Accepted values of `--speculative-dflash-lora-tts-optimizer-dtype`.
OPTIMIZER_DTYPES: Dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


@dataclass
class DFlashLoRATTSConfig:
    rank: int = 16
    alpha: float = 32.0
    # Defaults mirror ServerArgs. The worker always passes every field
    # explicitly; the defaults exist for direct construction in tests.
    lr: float = 1e-4
    # "lk" -> -log(sum_x min(p_target, q_draft)), see tts_loss.py
    # "kl" -> tts_lambda * KL(target || draft) + (1 - tts_lambda) * self-CE
    tts_loss: str = "lk"
    tts_lambda: float = 0.9
    freq: int = 1
    steps: int = 1
    pos_weight_gamma: float = 7.0
    grad_clip: float = 1.0
    targets: Tuple[str, ...] = ("q", "o", "gate_up", "down")
    # Which draft layers carry adapters: "" or "all" for every layer,
    # "last<K>" / "first<K>" for the top / bottom K layers, or an explicit
    # list such as "0,2,4". Resolved by parse_lora_tts_layers. A layer outside
    # the set has no (layer, kind) keys, so the serving hook, the replay and
    # the context-KV path all treat it as a frozen base layer.
    layers_spec: str = ""
    # Run the replay, the backward pass and the Adam step on a side CUDA
    # stream so that the update overlaps the next round's serving. The
    # computation is the same as in synchronous mode. The next round fences on
    # the in-flight update before it reads adapters.
    async_step: bool = True
    # Stagger the update rounds of different requests when freq > 1. A
    # request's phase is req_pool_idx % freq, and it is updated on rounds
    # phase, phase + freq, phase + 2 * freq, ... (the first update honors the
    # phase too; see should_fire). Requests admitted in the same prefill share
    # round_ct == 0, so without staggering the whole batch would be updated on
    # the same round: peak replay memory and the latency of that round would
    # grow with the batch size, and the other rounds would do no TTS work.
    # With staggering about batch_size / freq requests are updated per round.
    # Every request is still updated once per freq rounds, from that round's
    # own capture.
    phase_stagger: bool = True
    # Keep each request's masters and Adam moments in pinned CPU memory between
    # updates. That state is used only on update rounds, so with offload the
    # steady-state GPU cost of a request is its serving-dtype pool row. For an
    # update the state is copied to the GPU on the stream the update runs on,
    # the step runs on GPU leaves with torch's functional Adam (the function
    # torch.optim.Adam.step dispatches to), and the results are copied back.
    # The copies preserve the values exactly, so the update is the same as with
    # GPU-resident state (see test_offload_adam_parity.py).
    offload_state: bool = True
    # Final logit soft-capping of the target model, if it has one (e.g.
    # Gemma). The target's verify logits have already been capped, so the
    # replay applies the same cap to the draft logits before the loss.
    logit_softcap: Optional[float] = None
    # Delayed publish (off by default; it changes when updates take effect).
    # Adapters trained in round N become visible to drafting in round N+2
    # instead of N+1, which gives the side-stream update a full serving round
    # to finish without blocking the host. Each request's pool row is
    # double-buffered: an update writes the buffer that publish is not
    # reading, and publish reads the newest buffer whose writer has been
    # fenced. The per-round fence waits only for the second-newest update.
    # Requires async_step. With tp_size > 1 it also needs a dedicated NCCL
    # communicator for the update collectives, created at init; when that is
    # not possible the option is turned off with a warning. It doubles the
    # pool-row memory. See "Delayed publish" in DFLASH_LORA_TTS_DESIGN.md.
    delayed_publish: bool = False
    # Maximum number of requests whose updates are replayed together. Every
    # rank gets it from the same server argument: chunk boundaries determine
    # the sequence of TP collectives in the batched replay.
    replay_chunk: int = 8
    # Dtype in which each request's masters and moments are stored between
    # updates: "float32" or "bfloat16". An update always computes in fp32; with
    # "bfloat16" it widens the stored state, and rounds it once when it stores
    # the results. See "Precision" in DFLASH_LORA_TTS_DESIGN.md.
    optimizer_dtype: str = "float32"

    @property
    def scaling(self) -> float:
        return float(self.alpha) / float(self.rank)


def parse_lora_tts_targets(spec: str) -> Tuple[str, ...]:
    targets = tuple(t.strip() for t in spec.split(",") if t.strip())
    if not targets:
        raise ValueError("DFLASH LoRA-TTS targets must be a non-empty list.")
    for t in targets:
        if t not in _VALID_TARGETS:
            raise ValueError(
                f"Invalid DFLASH LoRA-TTS target {t!r}. Valid targets: "
                f"{_VALID_TARGETS}."
            )
    return targets


def parse_lora_tts_layers(spec: str, n_layers: int):
    """Resolve the layer-coverage spec to a frozenset of draft layer ids.

    Returns None when the spec selects every layer.

    Accepted forms:
      ""  / "all"      -> None (all layers)
      "last<K>"        -> the top K layers, i.e. {n_layers-K, ..., n_layers-1}
      "first<K>"       -> the bottom K layers
      "0,2,4"          -> that explicit set

    The spec is resolved in two places: by DFlashLoRATTSState, from the loaded
    modules, and by build_static_lora_buffers, from the model config.
    adopt_static_buffers cross-checks the two key sets and raises at init if
    they differ.
    """
    s = (spec or "").strip().lower()
    if not s or s == "all":
        return None
    n_layers = int(n_layers)
    if s.startswith("last") or s.startswith("first"):
        head = "last" if s.startswith("last") else "first"
        try:
            k = int(s[len(head) :])
        except ValueError:
            raise ValueError(
                f"Invalid DFLASH LoRA-TTS layers spec {spec!r}: expected "
                f"'{head}<K>' with an integer K."
            )
        if not 1 <= k <= n_layers:
            raise ValueError(
                f"Invalid DFLASH LoRA-TTS layers spec {spec!r}: K must be in "
                f"[1, {n_layers}] for a {n_layers}-layer drafter."
            )
        if k == n_layers:
            return None
        ids = (
            range(n_layers - k, n_layers) if head == "last" else range(k)
        )
        return frozenset(int(i) for i in ids)
    ids = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            ids.append(int(tok))
        except ValueError:
            raise ValueError(
                f"Invalid DFLASH LoRA-TTS layers spec {spec!r}: {tok!r} is "
                "not an integer layer id."
            )
    if not ids:
        raise ValueError(
            f"Invalid DFLASH LoRA-TTS layers spec {spec!r}: empty layer set."
        )
    for i in ids:
        if not 0 <= i < n_layers:
            raise ValueError(
                f"Invalid DFLASH LoRA-TTS layer id {i}: the drafter has "
                f"{n_layers} layers."
            )
    if len(set(ids)) == n_layers:
        return None
    return frozenset(ids)


@dataclass
class _ReqLoRAState:
    """Per-request adapters, optimizer state and update-cadence counters.

    The trainable state is four flat 1-D leaves, fp32 during an update, grouped
    by TP role:
      'a_repl'  A of column-parallel targets (replicated; the one group whose
                gradients have to be completed with an all-reduce)
      'b_col'   B of column-parallel targets (sharded)
      'a_row'   A of row-parallel targets (sharded)
      'b_repl'  B of row-parallel targets (replicated; its gradients are
                already complete)
    The per-key [r, in] / [out, r] matrices are contiguous narrow+view slices
    of these leaves (DFlashLoRATTSState._leaf_views), so the optimizer,
    nan_to_num, clipping and the all-reduce each handle four tensors per
    request instead of one per matrix. Adam is elementwise and all of a
    request's leaves step together, so flattening does not change the update
    (the clipping norm differs only in summation order). Leaves are never
    flattened across requests, because their step counts differ.

    The serving-dtype snapshot of the adapters lives in the state-wide pools,
    in row `pool_row`. It is written at creation and after each update."""

    # Offload mode (default): flats / adam_m / adam_v are pinned-CPU blocks in
    # the optimizer dtype, recycled across requests, and GPU fp32 leaves exist
    # only during the request's update. GPU-resident mode (offload_state=False)
    # with the float32 optimizer dtype: flats are GPU Parameters, `opt` is a
    # torch.optim.Adam over them, and adam_m / adam_v stay None. GPU-resident
    # mode with the bfloat16 optimizer dtype: the blocks are bf16 GPU tensors
    # and an update runs as in offload mode, without the host transfers.
    flats: Dict[str, torch.Tensor]
    opt: Optional[torch.optim.Adam]
    adam_m: Optional[Dict[str, torch.Tensor]] = None
    adam_v: Optional[Dict[str, torch.Tensor]] = None
    # Leaf groups whose moments have been written back by an Adam step.
    # Recycled CPU blocks still hold another request's moments, which must
    # never be read. Validity is tracked per group, so a group that takes its
    # first step later than the others still starts from zero moments.
    moments_valid: Set[str] = field(default_factory=set)
    # Row of the serving pools owned by this request (see _acquire_row).
    pool_row: Optional[int] = None
    round_ct: int = 0
    step_ct: int = 0
    # Update phase within the freq window (0 when staggering is off or
    # freq <= 1). Derived from req_pool_idx, so it is the same on every rank.
    phase: int = 0
    # Delayed-publish double buffering. `front` is the pool generation (0/1)
    # that holds this request's newest enqueued snapshot, and `front_seq` is
    # the enqueue sequence number that wrote it (0 for the creation snapshot,
    # which is written to both generations). Publish reads `front` if
    # front_seq has been fenced (<= _synced_seq) and otherwise the other
    # generation, which holds the previous complete snapshot. Both fields stay
    # 0 when delayed publish is off. The bookkeeping is host-side and
    # identical on every rank.
    front: int = 0
    front_seq: int = 0


@dataclass
class _ReplayCapture:
    """Inputs captured at draft time to replay one request's block forward."""

    req_pool_idx: int
    row: int  # batch row at capture time (batch is immutable within the round)
    input_embeds: torch.Tensor  # [block_size, hidden], detached clone
    positions: torch.Tensor  # [block_size] int64, detached clone
    prefix_locs: torch.Tensor  # [prefix_len] int64 draft-cache locations


def build_static_lora_buffers(
    *,
    draft_hf_config,
    tp_size: int,
    rank: int,
    alpha: float,
    targets: Tuple[str, ...],
    block_size: int,
    max_bs: int,
    dtype: torch.dtype,
    device,
    layers_spec: str = "",
):
    """Build the static serving buffers read by the hook in models/dflash.py.

    The buffers are derived from the draft model's config because they must
    exist before the draft model is constructed: draft CUDA-graph capture
    records the delta bmms reading these fixed tensors. The shapes follow
    DFlashLoRATTSState._shapes, which is derived from the loaded modules
    (column-parallel outputs and row-parallel inputs are divided by tp_size).
    adopt_static_buffers cross-checks the two derivations at init and raises
    on any mismatch."""
    from sglang.srt.models.dflash import _StaticLoRABuffers

    hidden = int(draft_hf_config.hidden_size)
    heads = int(draft_hf_config.num_attention_heads)
    head_dim = int(
        getattr(draft_hf_config, "head_dim", hidden // heads)
    )
    inter = int(draft_hf_config.intermediate_size)
    n_layers = int(draft_hf_config.num_hidden_layers)
    if heads % tp_size or inter % tp_size:
        raise RuntimeError(
            "DFlash LoRA-TTS static buffers: draft heads/intermediate not "
            f"divisible by tp={tp_size}."
        )
    q_local = (heads // tp_size) * head_dim
    inter_local = inter // tp_size
    kv_heads = int(
        getattr(draft_hf_config, "num_key_value_heads", heads)
    )
    kv_local = max(1, kv_heads // tp_size) * head_dim
    kind_shapes = {
        "q": (hidden, q_local),
        "k": (hidden, kv_local),
        "v": (hidden, kv_local),
        "o": (q_local, hidden),
        "gate_up": (hidden, 2 * inter_local),
        "down": (inter_local, hidden),
    }
    a: Dict[Tuple[int, str], torch.Tensor] = {}
    b: Dict[Tuple[int, str], torch.Tensor] = {}
    layer_sel = parse_lora_tts_layers(layers_spec, n_layers)
    for layer_id in range(n_layers):
        if layer_sel is not None and layer_id not in layer_sel:
            continue
        for kind in targets:
            in_f, out_f = kind_shapes[kind]
            key = (layer_id, kind)
            a[key] = torch.zeros(
                max_bs, rank, in_f, dtype=dtype, device=device
            )
            b[key] = torch.zeros(
                max_bs, out_f, rank, dtype=dtype, device=device
            )
    return _StaticLoRABuffers(
        block_size=block_size,
        scaling=float(alpha) / float(rank),
        max_bs=max_bs,
        a=a,
        b=b,
    )


class DFlashLoRATTSState:
    """Per-request LoRA adapters for the DFlash draft model: adapter state and
    lifecycle, publication to the serving buffers, and the replay-based
    update step."""

    def __init__(
        self,
        *,
        config: DFlashLoRATTSConfig,
        draft_model: torch.nn.Module,
        token_to_kv_pool,
        block_size: int,
        device: torch.device,
    ):
        if config.optimizer_dtype not in OPTIMIZER_DTYPES:
            raise ValueError(
                f"DFlash LoRA-TTS: unknown optimizer dtype "
                f"{config.optimizer_dtype!r}; expected one of "
                f"{sorted(OPTIMIZER_DTYPES)}."
            )
        self.cfg = config
        self.draft_model = draft_model
        self.token_to_kv_pool = token_to_kv_pool
        self.block_size = int(block_size)
        self.device = device

        # The base draft model is frozen; only the LoRA A/B matrices train.
        for p in draft_model.parameters():
            p.requires_grad_(False)

        self.layers = list(draft_model.layers)
        # Per-layer prefix window, read from the serving RadixAttention
        # (SGLang's exclusive convention; -1 or absent means full attention).
        # The replay windows a layer exactly when serving does. For a
        # windowed layer the fused kernel exposes the last
        # min(prefix_len, window) prefix keys (update_sliding_window_buffer)
        # and masks them per query row (extend_attention.py: block row m
        # excludes tail keys with abs_q - abs_k > window). The replay gathers
        # the same tail and rebuilds the same mask (see _replay_block_hidden).
        self._layer_prefix_windows = []
        for layer in self.layers:
            swa = int(getattr(layer.self_attn.attn, "sliding_window_size", -1) or -1)
            if 0 < swa < self.block_size - 1:
                # The replay leaves block-to-block attention unmasked; a
                # window shorter than the block would need masking inside the
                # block as well. DFlash windows are far longer than a block
                # (e.g. a 2048-token window with a 16-token block).
                raise RuntimeError(
                    f"DFlash LoRA-TTS requires sliding window ({swa}) >= "
                    f"block_size - 1 ({self.block_size - 1})."
                )
            self._layer_prefix_windows.append(swa if swa > 0 else None)
        self.serving_dtype = next(draft_model.parameters()).dtype
        self._state_dtype = OPTIMIZER_DTYPES[config.optimizer_dtype]
        if (
            self._state_dtype == torch.bfloat16
            and self.serving_dtype != torch.bfloat16
        ):
            # Serving reads the pool rows and the next update widens the stored
            # masters. Both hold the same values only when the pool has the
            # dtype of the state.
            raise ValueError(
                "--speculative-dflash-lora-tts-optimizer-dtype bfloat16 stores "
                "the adapters in the dtype they are served in and is only "
                "supported for a bfloat16 draft model; this draft is "
                f"{self.serving_dtype}. Use float32."
            )
        self._reqs: Dict[int, _ReqLoRAState] = {}
        self._captures: List[_ReplayCapture] = []
        self._warned_nonfinite = False
        # Static serving buffers (models/dflash._StaticLoRABuffers), handed
        # over by the worker through adopt_static_buffers. None when TTS never
        # updates (freq <= 0); the serving hook is then absent from the draft
        # graphs.
        self._static = None
        self._static_rows_hw = 0
        self._warned_static_overflow = False
        # In async mode the per-step isfinite check is skipped, because
        # reading it would synchronize the side stream. A non-finite loss then
        # becomes a near-no-op step (its gradients pass through nan_to_num)
        # and would leave no trace in the log. Count such losses on the GPU
        # and report the count at the infrequent host-sync points (disable /
        # flush) through _report_nonfinite().
        self._nonfinite_ct_gpu = torch.zeros(
            1, dtype=torch.int64, device=device
        )
        self._warned_step_failed = False

        # Serving-dtype adapter pools: {key: [cap, r, in]} and
        # {key: [cap, out, r]}, with one row per live request (rs.pool_row). A
        # row is written only when the request's adapters change (at creation
        # and after an update). publish_static gathers the batch's rows into
        # the static serving buffers with one index_select per key.
        # The pool tensors need no record_stream: side-stream writes are
        # fenced by the step event before the next publish reads them,
        # main-stream reads precede the next enqueue through wait_stream, and
        # the pools are freed only when they grow (_ensure_pool_cap), which
        # happens behind a fence.
        self._pool_cap = 0
        self._a_pools: Dict[Tuple[int, str], torch.Tensor] = {}
        self._b_pools: Dict[Tuple[int, str], torch.Tensor] = {}
        # Pool rows are assigned from a free list and are not indexed by
        # req_pool_idx. SGLang's request-slot allocator cycles through the
        # whole slot space, so pools indexed by slot would grow to the total
        # number of slots however few requests are live. Freeing a row is
        # bookkeeping only: an update that is already enqueued resolved its
        # row indices at enqueue time. A freed row is reassigned only by
        # get(), after every update that could still write it has been fenced
        # (see _release_state).
        self._free_rows: List[int] = []
        self._next_row = 0

        # Replay chunk size. Requests that are updated in the same round are
        # replayed together in one batched forward/backward (base weights are
        # read once per layer per step). The batch is split into chunks to
        # bound the transient fp32 memory of the loss, which per request is
        # proportional to block_size * vocab: the target probabilities held
        # across inner steps (see _target_constants) plus the draft-side
        # tensors that autograd retains for backward. This memory is allocated
        # on the side stream while serving runs. Lower `replay_chunk` on
        # memory-tight deployments; a replay that keeps failing disables TTS
        # (see note_round_failure). Chunking does not change the result: adapter
        # leaves of different requests are disjoint, so nothing accumulates
        # across chunks.
        self._replay_chunk = int(config.replay_chunk)

        # Deferred frees. Dropping a request's masters / Adam state while
        # an in-flight async update may still write them would let the
        # allocator recycle that memory during the write. Fencing at every
        # release would give up the overlap each time a request finishes, so
        # the state is parked here instead. An entry is tagged with the
        # enqueue sequence number current at release time (the newest update
        # that could still write it) and is released once that sequence number
        # has been fenced. With delayed publish the entry also keeps its pool
        # row until then; otherwise the row is freed immediately.
        self._deferred_free: List[Tuple[int, _ReqLoRAState]] = []

        # The positional loss weights depend only on block_size and gamma, so
        # they are computed once.
        num_pos = self.block_size - 1
        ks = torch.arange(num_pos, device=device, dtype=torch.float32)
        pos_weights = torch.exp(-ks / max(config.pos_weight_gamma, 1e-6))
        self._pos_weights = pos_weights / pos_weights.sum().clamp_min(1e-6)

        # Async stepping. A round's whole update (chunks in sequence, each
        # chunk batched) is enqueued on one side stream, and a single CUDA
        # event recorded after it fences the next read of the state. The fence
        # is a host-side event.synchronize() rather than a stream wait, so it
        # is correct whichever stream reads the adapters afterwards. It can
        # block only in the round that follows an update.
        self._use_async = (
            bool(config.async_step)
            and device.type == "cuda"
            and torch.cuda.is_available()
        )
        self._tts_stream = (
            torch.cuda.Stream(device=device) if self._use_async else None
        )
        # Outstanding step events, oldest first, each tagged with its enqueue
        # sequence number. run_tts_steps records at most one event per decode
        # round, and every fence pops from the front. wait_for_pending()
        # synchronizes all of them (the full fence). fence_for_publish() in
        # delayed mode synchronizes all but the newest: the newest update may
        # still be running while this round publishes from the other pool
        # generation. _synced_seq is the highest fenced sequence number.
        self._step_events: List[Tuple[int, torch.cuda.Event]] = []
        self._enqueue_seq = 0
        self._synced_seq = 0
        # Number of consecutive rounds whose update failed. Without a limit, a
        # persistent failure (e.g. a long-context replay that runs out of
        # memory on every update) would be retried on every round.
        self._step_failure_streak = 0
        self._disabled = False

        # The replay reads module weights with plain F.linear, which would be
        # wrong for quantized weights (including fp8, whose dtype still
        # reports as floating point). The draft model is built without a
        # quant_config; verify that here.
        for layer in self.layers:
            for w in (
                layer.self_attn.qkv_proj.weight,
                layer.self_attn.o_proj.weight,
                layer.mlp.gate_up_proj.weight,
                layer.mlp.down_proj.weight,
            ):
                if w.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                    raise RuntimeError(
                        "DFlash LoRA-TTS requires unquantized fp16/bf16/fp32 "
                        f"draft weights, got dtype={w.dtype}."
                    )

        self.tp_size = int(get_tensor_model_parallel_world_size())
        self.tp_rank = int(get_tensor_model_parallel_rank())

        # ---- Delayed publish (see DFlashLoRATTSConfig.delayed_publish) ----
        # _delayed turns on the lag-1 per-round fence, double-buffered pool
        # rows (_num_gens == 2), deferred pool-row frees and, with
        # tp_size > 1, a dedicated NCCL communicator for the update
        # collectives. The decision is made here from inputs that are the
        # same on every rank, so all ranks agree on it.
        self._delayed = bool(config.delayed_publish)
        if self._delayed and not self._use_async:
            logger.warning(
                "DFlash LoRA-TTS: delayed publish requires async stepping "
                "(sync mode completes training before publish anyway); "
                "running with immediate publish."
            )
            self._delayed = False
        self._tts_pg = None
        if self._delayed and self.tp_size > 1:
            import torch.distributed as dist

            if (
                not dist.is_initialized()
                or dist.get_world_size() != self.tp_size
            ):
                # new_group must be called by every process of the default
                # group with the same ranks list. That holds when the
                # torch.distributed world is exactly this TP group. With
                # additional parallelism (dp / pp) the other TP groups' rank
                # lists are not visible from here and creating the group
                # would hang, so delayed publish is turned off instead.
                logger.warning(
                    "DFlash LoRA-TTS: delayed publish at tp>1 needs a "
                    "dedicated communicator, which requires the "
                    "torch.distributed world to be exactly the TP group "
                    "(world=%s, tp=%d); running with immediate publish.",
                    dist.get_world_size() if dist.is_initialized() else None,
                    self.tp_size,
                )
                self._delayed = False
            else:
                from sglang.srt.distributed import get_tp_group

                global _TTS_PG
                # Every rank creates the group from the ranks list of the
                # shared TP group. The warm-up all-reduce initializes the NCCL
                # communicator now, on the main stream, rather than lazily
                # inside the first update's side-stream enqueue.
                self._tts_pg = dist.new_group(
                    ranks=list(get_tp_group().ranks), backend="nccl"
                )
                warm = torch.ones(1, device=self.device)
                dist.all_reduce(warm, group=self._tts_pg)
                torch.cuda.synchronize(self.device)
                _TTS_PG = self._tts_pg
        # Pool-row generations: 2 with delayed publish (front / back double
        # buffering), otherwise 1.
        self._num_gens = 2 if self._delayed else 1

        # Local shapes per (layer, kind), read from the loaded (TP-sharded)
        # modules. Adapters are sharded like their base layer:
        #   column-parallel (q, k, v, gate_up): A replicated [r, in], B
        #     sharded [out_local, r]. The serving delta is rank-local. The
        #     gradient of A is partial on each rank and is all-reduced after
        #     backward.
        #   row-parallel (o, down): A sharded [r, in_local], B replicated
        #     [out, r]. The rank-r intermediate z = A x_local is a partial
        #     sum and is all-reduced in both serving and replay; the gradients
        #     of A and B are then complete without further collectives.
        _A_SHARDED_KINDS = ("o", "down")
        self._shapes: Dict[Tuple[int, str], Tuple[int, int]] = {}
        self._a_sharded: Dict[Tuple[int, str], bool] = {}
        # Layer coverage; None selects every layer. Unselected layers get no
        # keys, so the serving hook, the replay and the context-KV path (all
        # keyed off _shapes) treat them as frozen base layers.
        self._layer_sel = parse_lora_tts_layers(
            getattr(config, "layers_spec", ""), len(self.layers)
        )
        for layer_id, layer in enumerate(self.layers):
            if self._layer_sel is not None and layer_id not in self._layer_sel:
                continue
            attn = layer.self_attn
            mlp = layer.mlp
            hidden = int(attn.hidden_size)
            for kind in config.targets:
                if kind == "q":
                    in_f, out_f = hidden, int(attn.q_size)
                elif kind in ("k", "v"):
                    # Column-parallel (part of qkv_proj): A replicated,
                    # B sharded — same TP role as q.
                    in_f, out_f = hidden, int(attn.kv_size)
                elif kind == "o":
                    w = attn.o_proj.weight  # [hidden, q_size_local]
                    in_f, out_f = int(w.shape[1]), int(w.shape[0])
                elif kind == "gate_up":
                    w = mlp.gate_up_proj.weight  # [2 * inter_local, hidden]
                    in_f, out_f = int(w.shape[1]), int(w.shape[0])
                elif kind == "down":
                    w = mlp.down_proj.weight  # [hidden, inter_local]
                    in_f, out_f = int(w.shape[1]), int(w.shape[0])
                self._shapes[(layer_id, kind)] = (in_f, out_f)
                self._a_sharded[(layer_id, kind)] = kind in _A_SHARDED_KINDS

        # One A-init template per (layer, kind), drawn once from a fixed seed,
        # so that a request's initial adapters are reproducible and do not
        # depend on the order in which requests arrive. Every rank draws the
        # same global tensor. For kinds with a sharded A, each rank takes its
        # contiguous block of input columns, so the shards together form one
        # consistent global A.
        gen = torch.Generator(device="cpu")
        gen.manual_seed(0)
        self._offload = bool(config.offload_state)
        if self._offload and not (
            torch.device(device).type == "cuda" and torch.cuda.is_available()
        ):
            # Pinned memory needs a CUDA context. Fall back so that CPU-only
            # construction (unit tests) works.
            self._offload = False
        self._functional_adam = None
        if self._offload or self._state_dtype != torch.float32:
            try:
                # The function torch.optim.Adam.step dispatches to. Using it
                # keeps the offloaded update identical to that of a resident
                # optimizer.
                from torch.optim.adam import adam as _fadam

                self._functional_adam = _fadam
            except Exception:  # noqa: BLE001
                if self._state_dtype != torch.float32:
                    raise
                logger.warning(
                    "DFlash LoRA-TTS: torch functional Adam unavailable; "
                    "keeping optimizer state GPU-resident."
                )
                self._offload = False
        # Whether a request's state is stored in separate blocks and an update
        # runs on fp32 leaves filled from them. The alternative, for float32
        # GPU-resident state only, is a torch.optim.Adam over GPU Parameters.
        self._state_in_blocks = self._offload or self._state_dtype != torch.float32
        self._a_init: Dict[Tuple[int, str], torch.Tensor] = {}
        for key, (in_f, _out_f) in self._shapes.items():
            full_in = in_f * (self.tp_size if self._a_sharded[key] else 1)
            a = torch.empty(config.rank, full_in, dtype=torch.float32)
            # kaiming_uniform_ has no generator= on some torch versions; draw
            # uniformly by hand with the same bound (a=sqrt(5) fan_in mode).
            bound = 1.0 / math.sqrt(full_in)
            a.uniform_(-bound, bound, generator=gen)
            if self._a_sharded[key]:
                a = a[:, self.tp_rank * in_f : (self.tp_rank + 1) * in_f]
            # In offload mode the templates stay on the CPU: creation copies
            # them straight into the pinned-CPU masters.
            self._a_init[key] = (
                a.contiguous()
                if self._offload
                else a.contiguous().to(device)
            )

        # Recycled pinned-CPU state blocks (offload mode). Pinned allocation
        # is slow, so blocks are reused across requests. A block returns to
        # this cache only once no in-flight update can still be copying
        # results into it: immediately when nothing is pending, otherwise at a
        # later fence through the deferred-free list. The cache is capped at
        # _CPU_CACHE_MAX blocks.
        self._cpu_state_cache: List[
            Tuple[
                Dict[str, torch.Tensor],
                Dict[str, torch.Tensor],
                Dict[str, torch.Tensor],
            ]
        ] = []

        # Flat-leaf layout: per-key offsets into the four role-grouped 1-D
        # leaves (see _ReqLoRAState). _shapes iterates in layer / kind
        # insertion order, which is the same on every rank, so the layout and
        # every collective over a flat tensor are consistent across ranks.
        self._flat_layout: Dict[Tuple[int, str], Tuple[str, int, str, int]] = {}
        self._flat_sizes: Dict[str, int] = {
            "a_repl": 0,
            "b_col": 0,
            "a_row": 0,
            "b_repl": 0,
        }
        for key, (in_f, out_f) in self._shapes.items():
            ga, gb = (
                ("a_row", "b_repl") if self._a_sharded[key] else ("a_repl", "b_col")
            )
            off_a = self._flat_sizes[ga]
            self._flat_sizes[ga] += config.rank * in_f
            off_b = self._flat_sizes[gb]
            self._flat_sizes[gb] += out_f * config.rank
            self._flat_layout[key] = (ga, off_a, gb, off_b)

        num_params_per_req = sum(
            config.rank * (in_f + out_f) for (in_f, out_f) in self._shapes.values()
        )
        logger.info(
            "DFlash LoRA-TTS enabled. loss=%s rank=%d alpha=%.1f lr=%.2e lambda=%.3f "
            "freq=%d steps=%d gamma=%.2f grad_clip=%.2f targets=%s "
            "layers=%s trainable_params_per_req=%d optimizer_dtype=%s "
            "state_bytes_per_req=%d offload=%s stagger=%s delayed_publish=%s",
            config.tts_loss,
            config.rank,
            config.alpha,
            config.lr,
            config.tts_lambda,
            config.freq,
            config.steps,
            config.pos_weight_gamma,
            config.grad_clip,
            ",".join(config.targets),
            (
                "all"
                if self._layer_sel is None
                else ",".join(str(i) for i in sorted(self._layer_sel))
            ),
            num_params_per_req,
            config.optimizer_dtype,
            3 * num_params_per_req * torch.finfo(self._state_dtype).bits // 8,
            self._offload,
            config.phase_stagger,
            self._delayed,
        )

    # ---- Per-request lifecycle -----------------------------------------

    def _leaf_views(
        self, flats: Dict[str, torch.Tensor], key: Tuple[int, str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the per-key A [r, in] and B [out, r] views into the flat
        leaves (the GPU leaves during an update, or the pinned-CPU masters for
        the refresh at creation). The views are created in the caller's grad
        context: under enable_grad the narrow / view nodes route gradients
        into the flat leaf. They belong to one step's graph and must be
        re-created for every step, which is cheap (no kernels run)."""
        in_f, out_f = self._shapes[key]
        ga, off_a, gb, off_b = self._flat_layout[key]
        r = self.cfg.rank
        a = flats[ga].narrow(0, off_a, r * in_f).view(r, in_f)
        b = flats[gb].narrow(0, off_b, out_f * r).view(out_f, r)
        return a, b

    def _acquire_state_blocks(
        self,
    ) -> Tuple[
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
    ]:
        """Blocks for one request's masters and moments, in the optimizer dtype:
        pinned-CPU in offload mode (recycled ones first), otherwise on the
        device."""
        if self._cpu_state_cache:
            return self._cpu_state_cache.pop()

        def _mk() -> Dict[str, torch.Tensor]:
            return {
                g: torch.empty(
                    n,
                    dtype=self._state_dtype,
                    pin_memory=self._offload,
                    device=None if self._offload else self.device,
                )
                for g, n in self._flat_sizes.items()
            }

        return (_mk(), _mk(), _mk())

    def _recycle_cpu_state(self, rs: _ReqLoRAState) -> None:
        """Return a released request's pinned-CPU blocks to the cache. Call
        only when no in-flight update can still be copying results into them
        (when nothing is pending, or after a fence). The next owner must not
        read the old contents: reads of the moments are gated by
        moments_valid, and the masters are overwritten at creation."""
        if not self._offload or not rs.flats:
            return
        # The cache is capped. It would otherwise grow to the peak number of
        # concurrent requests and keep that much page-locked host memory for
        # the lifetime of the server. Beyond the cap the block is dropped (and
        # unpinned when it is collected); a later burst of requests pays for
        # the pinned allocation again.
        if len(self._cpu_state_cache) < self._CPU_CACHE_MAX:
            self._cpu_state_cache.append((rs.flats, rs.adam_m, rs.adam_v))
        rs.flats = {}
        rs.adam_m = None
        rs.adam_v = None

    def _acquire_row(self) -> int:
        if self._free_rows:
            return self._free_rows.pop()
        row = self._next_row
        # Grow before advancing _next_row: if the growth allocation raises, no
        # row index has been consumed.
        self._ensure_pool_cap(row + 1)
        self._next_row = row + 1
        return row

    def get(self, req_pool_idx: int) -> _ReqLoRAState:
        rs = self._reqs.get(int(req_pool_idx))
        if rs is not None:
            return rs
        with torch.inference_mode(False):
            if self._state_in_blocks:
                flats, adam_m, adam_v = self._acquire_state_blocks()
                # B must start at zero (the adapter is inert until its first
                # update). The A groups are fully covered by the template
                # copies below. Recycled moment blocks are left as they are;
                # moments_valid gates reads of them.
                for g in ("b_col", "b_repl"):
                    flats[g].zero_()
                for key in self._shapes:
                    ga, off_a, _gb, _off_b = self._flat_layout[key]
                    a0 = self._a_init[key]
                    flats[ga][off_a : off_a + a0.numel()].copy_(
                        a0.reshape(-1)
                    )
                opt = None
            else:
                init = {
                    g: torch.zeros(
                        n, dtype=torch.float32, device=self.device
                    )
                    for g, n in self._flat_sizes.items()
                }
                for key in self._shapes:
                    ga, off_a, _gb, _off_b = self._flat_layout[key]
                    a0 = self._a_init[key]
                    init[ga][off_a : off_a + a0.numel()].copy_(
                        a0.reshape(-1)
                    )
                flats = {
                    g: torch.nn.Parameter(t) for g, t in init.items()
                }
                opt = torch.optim.Adam(
                    list(flats.values()), lr=self.cfg.lr
                )
                adam_m = adam_v = None
        phase = (
            int(req_pool_idx) % self.cfg.freq
            if self.cfg.phase_stagger and self.cfg.freq > 1
            else 0
        )
        rs = _ReqLoRAState(
            flats=flats, opt=opt, adam_m=adam_m, adam_v=adam_v, phase=phase
        )
        try:
            rs.pool_row = self._acquire_row()
            # Creation-time snapshot. copy_ does the host-to-device transfer
            # and the fp32 -> serving-dtype cast in one op when the masters
            # are on the CPU (offload mode). With delayed publish both
            # generations are written (B is zero, so the delta is exactly
            # zero in either), so a new request publishes correctly whichever
            # generation the round selects. front_seq stays 0, which always
            # counts as fenced.
            for gen in range(self._num_gens):
                self._refresh_cached(rs, rs.flats, gen=gen)
        except Exception:
            # rs is not in _reqs yet, so nothing else would release its row or
            # recycle its CPU blocks. Do it here; no update can be in flight
            # for a state that has never been captured.
            if rs.pool_row is not None:
                self._free_rows.append(rs.pool_row)
                rs.pool_row = None
            self._recycle_cpu_state(rs)
            raise
        self._reqs[int(req_pool_idx)] = rs
        return rs

    def _ensure_pool_cap(self, cap: int) -> None:
        """Grow the pools to at least `cap` rows per generation.

        Pool tensors are flat across generations: [num_gens * cap, ...], with
        row index gen * cap + pool_row (num_gens is 1 unless delayed publish
        is on). Growth frees the old tensors, so it must not overlap the row
        writes of an in-flight update. Without delayed publish, get() already
        runs behind the full per-round fence. With delayed publish the
        per-round fence leaves the newest update in flight, so growth does a
        full fence here. Growth is rare: it happens only when the peak number
        of live requests rises."""
        if cap <= self._pool_cap:
            return
        if self._delayed:
            self.wait_for_pending()
        old_cap = self._pool_cap
        new_cap = max(cap, old_cap * 2 if old_cap else cap)
        with torch.no_grad():
            for key, (in_f, out_f) in self._shapes.items():
                a_new = torch.zeros(
                    self._num_gens * new_cap, self.cfg.rank, in_f,
                    dtype=self.serving_dtype, device=self.device,
                )
                b_new = torch.zeros(
                    self._num_gens * new_cap, out_f, self.cfg.rank,
                    dtype=self.serving_dtype, device=self.device,
                )
                if old_cap:
                    for g in range(self._num_gens):
                        a_new[g * new_cap : g * new_cap + old_cap].copy_(
                            self._a_pools[key][
                                g * old_cap : (g + 1) * old_cap
                            ]
                        )
                        b_new[g * new_cap : g * new_cap + old_cap].copy_(
                            self._b_pools[key][
                                g * old_cap : (g + 1) * old_cap
                            ]
                        )
                self._a_pools[key] = a_new
                self._b_pools[key] = b_new
        self._pool_cap = new_cap

    def _refresh_cached(
        self,
        rs: _ReqLoRAState,
        flats: Dict[str, torch.Tensor],
        gen: int = 0,
    ) -> None:
        """Write the request's serving-dtype adapter snapshot into its pool
        row (generation `gen`; always 0 without delayed publish) from `flats`:
        the update's GPU leaves, or the CPU masters at creation. copy_ casts
        and, for CPU sources, transfers in one op without allocating.

        Ordering invariant: a row is written only at creation or by that
        request's own update, and never while a publish may be reading it.

        Without delayed publish, creation-time writes, row reuse and pool
        growth all happen inside get(). get() is reached only from the
        worker's draft preparation, after the per-round fence (a full
        wait_for_pending()), and that same fence orders each update's row
        writes before the next publish.

        With delayed publish the per-round fence is lag-1, so the newest
        update may still be writing. The invariant then rests on three rules:
          (a) an update writes the request's other generation (the front
              flip), never the one a publish can read;
          (b) freed rows are parked until the update that could write them
              has been fenced (sequence-tagged deferred frees), so a reused
              row has no in-flight writer in either generation;
          (c) pool growth does a full fence (_ensure_pool_cap).
        The row index is resolved here, at enqueue time, against the current
        _pool_cap. That is consistent because growth cannot run while an
        update is in flight (rule c).

        All of this relies on the per-round fence running on every decode
        round, including rounds that publish nothing new."""
        row = gen * self._pool_cap + rs.pool_row
        with torch.no_grad():
            for key in self._shapes:
                a, b = self._leaf_views(flats, key)
                self._a_pools[key][row].copy_(a)
                self._b_pools[key][row].copy_(b)

    def _report_nonfinite(self) -> None:
        """Read and reset the count of non-finite losses seen in async mode.
        Call only from host-sync points (disable / flush): the read
        synchronizes with the device."""
        if not self._use_async:
            return
        ct = int(self._nonfinite_ct_gpu.item())
        if ct > 0:
            logger.warning(
                "DFlash LoRA-TTS: %d async training steps had a non-finite "
                "loss (grads were nan_to_num'd to ~zero-effect steps). "
                "Persistent counts mean the adapters are diverging — lower "
                "the lr or use --speculative-dflash-lora-tts-no-async to "
                "surface per-step skips.",
                ct,
            )
            self._nonfinite_ct_gpu.zero_()

    def release_pools(self) -> None:
        """Free the serving-dtype pools (flush / sleep path).

        Only valid when no per-request state is live (call after
        wait_for_pending() and prune_except([])): a live request's rows would
        otherwise read as a zero delta until its next update. The pools are
        recreated lazily by _ensure_pool_cap on the next prefill that
        allocates TTS state. New pools are zero-initialized, which means "no
        delta".
        """
        if self._reqs:
            raise RuntimeError(
                "release_pools() called with live TTS requests; "
                "prune_except([]) first"
            )
        self._a_pools.clear()
        self._b_pools.clear()
        self._pool_cap = 0
        self._free_rows.clear()
        self._next_row = 0
        self._zero_static()
        # Release the pinned host blocks too: page-locked memory should not
        # stay allocated across a flush / sleep.
        self._cpu_state_cache.clear()

    def _drain_deferred(self) -> None:
        """Release parked state whose potential writer has been fenced
        (seq <= _synced_seq): free its pool row (parked with the state in
        delayed mode) and return its pinned-CPU blocks to the cache. Entries
        whose writer is still outstanding stay parked."""
        if not self._deferred_free:
            return
        still: List[Tuple[int, _ReqLoRAState]] = []
        for seq, rs in self._deferred_free:
            if seq <= self._synced_seq:
                if rs.pool_row is not None:
                    self._free_rows.append(rs.pool_row)
                    rs.pool_row = None
                self._recycle_cpu_state(rs)
            else:
                still.append((seq, rs))
        self._deferred_free = still

    def _sync_events(self, keep: int) -> None:
        """Synchronize outstanding step events oldest-first until at most
        `keep` remain, advance _synced_seq, and drain newly-safe deferred
        frees."""
        while len(self._step_events) > keep:
            seq, ev = self._step_events.pop(0)
            ev.synchronize()
            if seq > self._synced_seq:
                self._synced_seq = seq
        self._drain_deferred()

    def wait_for_pending(self) -> None:
        """Full fence: wait until every in-flight async update has completed.
        No-op when async is off or nothing is in flight. Also releases state
        whose free was deferred. Required before pool rows are reused or the
        pools grow, before state is released or recycled, and in disable(),
        flush / clear_cache_pool, the weight-update overrides and the
        extend / prefill branch."""
        self._sync_events(0)

    def fence_for_publish(self) -> None:
        """The per-decode-round fence that precedes publishing adapters.

        Without delayed publish this is the full fence, because publish reads
        the rows that the newest update writes.

        With delayed publish it is lag-1: it waits only for the events older
        than the newest, which have normally completed already. The newest
        update keeps running on the side stream while this round publishes,
        for each request, the newest generation that has been fenced
        (_published_gen). The reader and the in-flight writer use different
        generations, so no read can be torn.

        With tp_size == 1 the newest event is additionally polled with a
        non-blocking query(): if that update has already finished, its
        snapshots become visible in this round. The poll also bounds the delay
        when updates are sparse (batch size < freq), where rounds without a
        new enqueue would otherwise leave the newest event unfenced for many
        rounds.

        With tp_size > 1 the poll is skipped. Its result depends on device
        timing, and every rank must select the same generation: a rank that
        publishes a different generation than its peers computes a different
        forward and breaks the TP-synchronized sampling. Visibility is then
        determined by event order alone: an update is fenced once the lag-1
        fence of a newer enqueue has retired it."""
        if not self._delayed:
            self._sync_events(0)
            return
        self._sync_events(1)
        if (
            self.tp_size == 1
            and self._step_events
            and self._step_events[0][1].query()
        ):
            seq, _ev = self._step_events.pop(0)
            if seq > self._synced_seq:
                self._synced_seq = seq
            self._drain_deferred()

    def _published_gen(self, rs: _ReqLoRAState) -> int:
        """The pool generation that publish and the context-KV path read for
        this request: the newest snapshot whose writer has been fenced.
        front_seq == 0 denotes the creation snapshot, which is written to
        both generations."""
        if not self._delayed:
            return 0
        return rs.front if rs.front_seq <= self._synced_seq else 1 - rs.front

    def _release_state(self, rs: Optional[_ReqLoRAState]) -> None:
        """Release a request's masters / Adam state without fencing. If an
        async update may still be writing them, park them (tagged with the
        newest enqueue sequence number) until that update has been fenced.
        Fencing here would give up the overlap every time a request finishes.

        Without delayed publish the pool row is freed immediately. That is
        bookkeeping only: an update that is already enqueued resolved its row
        indices at enqueue time, and the row is reassigned only by get(),
        behind the next full fence. With delayed publish the row is parked
        with the state, because the lag-1 fence leaves the newest update in
        flight and a row it writes must not be reassigned before it has been
        fenced."""
        if rs is None:
            return
        if rs.pool_row is not None and not self._delayed:
            self._free_rows.append(rs.pool_row)
            rs.pool_row = None
        if self._use_async and self._step_events:
            self._deferred_free.append((self._enqueue_seq, rs))
            return
        if rs.pool_row is not None:
            self._free_rows.append(rs.pool_row)
            rs.pool_row = None
        self._recycle_cpu_state(rs)

    def drop(self, req_pool_idx: int) -> None:
        """Release a finished request's adapters and Adam state right away
        (its request slot may not be prefilled again for a long time)."""
        self._release_state(self._reqs.pop(int(req_pool_idx), None))

    def reset(self, req_pool_indices: torch.Tensor) -> None:
        """A prefill means a (re)used pool slot: clear any stale state."""
        for idx in req_pool_indices.tolist():
            self._release_state(self._reqs.pop(int(idx), None))

    def prune_except(self, live_req_pool_indices: Iterable[int]) -> None:
        """Drop the state of slots that are no longer in the running batch.

        Retracted (evicted under memory pressure) and aborted requests never
        reach the finished() drop path. Without this their adapters and
        Adam state would stay allocated until the slot is prefilled again.
        Speculative decoding runs a single running batch, so every live
        decoding request appears in every decode round.
        """
        live = set(int(i) for i in live_req_pool_indices)
        stale = [idx for idx in self._reqs if idx not in live]
        for idx in stale:
            self._release_state(self._reqs.pop(idx, None))

    def should_fire(self, req_pool_idx: int) -> bool:
        if self.cfg.freq <= 0 or self._disabled:
            return False
        rs = self.get(req_pool_idx)
        # The first update honors the phase as well; round 0 is not a special
        # case. Otherwise all requests admitted in the same prefill would be
        # updated together at round 0 and allocate their update workspaces at
        # the same time, and the caching allocator would keep that peak
        # reserved, which defeats the memory saving of offloading. The cost is
        # that a request's first update comes up to freq - 1 rounds after
        # admission. With phase 0 (staggering off, freq 1, or
        # req_pool_idx % freq == 0) the condition is round_ct % freq == 0.
        return (rs.round_ct % self.cfg.freq) == rs.phase

    def advance(self, req_pool_idx: int) -> None:
        # Called for every request of the batch on every verify round, whether
        # or not the request was captured or updated in that round. The
        # counter must not be gated on a capture: captures depend on the
        # counter, so gating would stall the cadence after the first update.
        if self.cfg.freq <= 0 or self._disabled:
            # Never-train / disabled mode: don't materialize per-request state.
            return
        # Deliberately does not allocate: post-verify bookkeeping must not be
        # able to run out of memory. A request whose state is missing (e.g.
        # after an earlier allocation failure) restarts its cadence when the
        # next capture calls get(), like a new request.
        rs = self._reqs.get(int(req_pool_idx))
        if rs is not None:
            rs.round_ct += 1

    # ---- Serving publication -------------------------------------------

    def is_disabled(self) -> bool:
        return self._disabled

    def disable(self, reason: str) -> None:
        """Permanently stop capturing, publishing and updating (e.g. when an
        unsupported lm_head layout is found at runtime), and free all
        per-request state, which nothing would prune afterwards."""
        if not self._disabled:
            logger.warning("DFlash LoRA-TTS disabled: %s", reason)
        self._disabled = True
        self._captures = []
        self.wait_for_pending()
        # Also synchronize the TTS stream directly.
        if self._tts_stream is not None:
            self._tts_stream.synchronize()
        self._report_nonfinite()
        # Everything is fenced, so the releases below free rows and recycle
        # CPU blocks immediately.
        for rs in list(self._reqs.values()):
            self._release_state(rs)
        self._reqs.clear()
        self._drain_deferred()
        # The delta bmms recorded in the draft CUDA graphs cannot be removed.
        # Zero the static rows so they contribute exactly zero from now on.
        self._zero_static()
        # Repeated failures are often caused by memory pressure (e.g. replays
        # that keep running out of memory), so free the serving-dtype pools
        # too instead of holding them until an explicit flush.
        self.release_pools()

    def adopt_static_buffers(self, buffers) -> None:
        """Check the worker's static buffers against the per-key shapes
        derived from the loaded modules, then adopt them as the publish
        target. The buffers are built from the model config, before the draft
        graphs are captured; any mismatch between the two shape derivations
        raises here, at init."""
        for key, (in_f, out_f) in self._shapes.items():
            a = buffers.a.get(key)
            b = buffers.b.get(key)
            if (
                a is None
                or b is None
                or tuple(a.shape[1:]) != (self.cfg.rank, in_f)
                or tuple(b.shape[1:]) != (out_f, self.cfg.rank)
                or a.dtype != self.serving_dtype
                or b.dtype != self.serving_dtype
            ):
                raise RuntimeError(
                    f"DFlash LoRA-TTS static buffer drift for {key}: "
                    f"config-derived buffers do not match the module-derived "
                    f"shapes (expected A[*, {self.cfg.rank}, {in_f}] "
                    f"B[*, {out_f}, {self.cfg.rank}] {self.serving_dtype})."
                )
        extra = (set(buffers.a) | set(buffers.b)) - set(self._shapes)
        if extra:
            raise RuntimeError(
                f"DFlash LoRA-TTS static buffers carry unknown keys: {extra}"
            )
        if float(buffers.scaling) != float(self.cfg.scaling) or int(
            buffers.block_size
        ) != int(self.block_size):
            raise RuntimeError(
                "DFlash LoRA-TTS static buffer scaling/block_size drift."
            )
        self._static = buffers
        self._static_rows_hw = 0

    def publish_static(self, req_pool_idx_list: List[int]) -> None:
        """Write the batch's serving-dtype adapters into the static buffers
        (buffer row r holds batch position r) and zero the rows that a
        previous, larger batch wrote. With draft CUDA graphs the recorded
        delta bmms read these buffers, so their contents must be correct
        before every draft forward. An all-zero row gives an exactly-zero
        delta, and B is zero until a request's first update, so publishing a
        request that has not been updated yet is exact. Runs behind the
        per-round fence (see _refresh_cached)."""
        if self._static is None:
            return
        buf = self._static
        bs = len(req_pool_idx_list)
        with torch.no_grad():
            written = 0
            if bs > 0 and not (self.cfg.freq <= 0 or self._disabled):
                if bs > buf.max_bs:
                    # The serving hook skips the delta for the whole batch
                    # when the batch exceeds max_bs (every rank makes the same
                    # decision). Leave the buffers as they are and warn once.
                    if not self._warned_static_overflow:
                        logger.warning(
                            "DFlash LoRA-TTS: batch size %d exceeds the "
                            "static adapter buffers (max_bs=%d); serving "
                            "without adapters for oversized batches. Set "
                            "--max-running-requests at or below the buffer "
                            "cap (or restart with a larger value to size "
                            "the buffers up).",
                            bs,
                            buf.max_bs,
                        )
                        self._warned_static_overflow = True
                    written = self._static_rows_hw  # leave as-is
                else:
                    states = [self.get(i) for i in req_pool_idx_list]
                    # With delayed publish each request reads its newest
                    # fenced generation (row gen * cap + pool_row of the flat
                    # pools); an in-flight update writes only the other
                    # generation. Without it gen is 0.
                    cap = self._pool_cap
                    rows = torch.tensor(
                        [
                            self._published_gen(s) * cap + s.pool_row
                            for s in states
                        ],
                        dtype=torch.long,
                        device=self.device,
                    )
                    # Raise the high-water mark before writing. If the
                    # per-key loop raised midway while the mark was still the
                    # old one, rows [old_hw, bs) could already hold adapter
                    # content that _zero_static (failure path / disable)
                    # would not clear, and those rows would later be applied
                    # to whichever requests occupy those batch positions. The
                    # mark must cover every written row at all times.
                    self._static_rows_hw = max(self._static_rows_hw, bs)
                    for key in self._shapes:
                        torch.index_select(
                            self._a_pools[key], 0, rows, out=buf.a[key][:bs]
                        )
                        torch.index_select(
                            self._b_pools[key], 0, rows, out=buf.b[key][:bs]
                        )
                    written = bs
            if self._static_rows_hw > written:
                for key in self._shapes:
                    buf.a[key][written : self._static_rows_hw].zero_()
                    buf.b[key][written : self._static_rows_hw].zero_()
            self._static_rows_hw = written

    def has_kv_targets(self) -> bool:
        """True when adapters on W_k / W_v are in the target set. The worker
        then applies per-request deltas in the context-KV append and bypasses
        the fused KV materializer for those appends."""
        return any(kind in ("k", "v") for _lid, kind in self._shapes)

    def context_kv_delta(
        self,
        req_pool_idx: int,
        layer_id: int,
        kind: str,
        x: torch.Tensor,  # [seg_len, hidden], serving dtype
    ) -> Optional[torch.Tensor]:
        """Per-request delta for the context-KV append when W_k / W_v carry
        adapters (kv_proj_only path). Reads the request's serving-dtype pool
        row, i.e. the adapters as they are at append time; cached KV is not
        rewritten when the adapters change later. Returns None when the kind
        is not targeted or the request has not been updated yet (B = 0, so
        the delta is exactly zero). Callers run behind the decode / extend
        fences, so the row read cannot overlap the refresh of an in-flight
        update."""
        key = (int(layer_id), kind)
        if key not in self._shapes:
            return None
        rs = self._reqs.get(int(req_pool_idx))
        if rs is None or rs.step_ct == 0 or rs.pool_row is None:
            return None
        # Same generation selection as publish_static: the newest fenced
        # snapshot (with delayed publish the other one may be being written).
        # With tp_size == 1 and delayed publish, this append and the later
        # publish_static can select different generations for one request in
        # the same round, because the poll in fence_for_publish may advance
        # _synced_seq between the two. The round's cached KV then comes from
        # adapters one update older than those of the block forward, which is
        # consistent with the append-time semantics described above.
        row = self._published_gen(rs) * self._pool_cap + rs.pool_row
        a = self._a_pools[key][row]  # [r, in]
        b = self._b_pools[key][row]  # [out, r]
        if x.dtype != a.dtype:
            x = x.to(a.dtype)
        return (x @ a.t() @ b.t()) * self.cfg.scaling

    def _zero_static(self) -> None:
        """Zero all written static rows (disable / flush / failure paths), so
        that the delta bmms contribute exactly zero."""
        if self._static is None:
            return
        with torch.no_grad():
            hw = max(self._static_rows_hw, 0)
            if hw > 0:
                for key in self._shapes:
                    self._static.a[key][:hw].zero_()
                    self._static.b[key][:hw].zero_()
        self._static_rows_hw = 0

    # ---- Capture ---------------------------------------------------------

    def clear_captures(self) -> None:
        self._captures = []

    def capture_for_replay(
        self,
        *,
        req_pool_idx_list: List[int],
        block_input_embeds: torch.Tensor,  # [bs, block_size, hidden]
        positions_2d: torch.Tensor,  # [bs, block_size] int64
        draft_prefix_lens_cpu: torch.Tensor,  # [bs] int32 on CPU
        draft_req_to_token: torch.Tensor,
    ) -> None:
        """Capture the replay inputs of the rows that are updated this round.

        The clones are taken with inference mode disabled, because inference
        tensors cannot be saved for backward. The embeddings must be cloned
        before the draft forward: the serving fused add-RMSNorm modifies the
        residual, which is a view of the embeddings, in place.
        """
        self._captures = []
        if self.cfg.freq <= 0 or self._disabled:
            return
        with torch.inference_mode(False), torch.no_grad():
            for row, req_pool_idx in enumerate(req_pool_idx_list):
                if not self.should_fire(req_pool_idx):
                    continue
                prefix_len = int(draft_prefix_lens_cpu[row])
                if prefix_len <= 0:
                    continue
                self._captures.append(
                    _ReplayCapture(
                        req_pool_idx=req_pool_idx,
                        row=row,
                        input_embeds=block_input_embeds[row].detach().clone(),
                        positions=positions_2d[row].detach().clone(),
                        prefix_locs=draft_req_to_token[req_pool_idx, :prefix_len]
                        .detach()
                        .to(torch.int64)
                        .clone(),
                    )
                )

    def has_pending_captures(self) -> bool:
        return len(self._captures) > 0

    def pending_capture_rows(self) -> List[int]:
        """Batch rows (capture-time order) of the pending captures."""
        return [cap.row for cap in self._captures]

    # ---- Replay forward --------------------------------------------------

    def _lora_delta(
        self,
        flats: Dict[str, torch.Tensor],
        layer_id: int,
        kind: str,
        x: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        key = (layer_id, kind)
        if key not in self._flat_layout:
            return None
        a, b = self._leaf_views(flats, key)
        # Cast the fp32 leaves to the compute dtype inside the graph, so that
        # the replay matches serving numerics (serving reads snapshots in the
        # serving dtype, e.g. bf16). Gradients still reach the flat fp32
        # leaves through the cast and view nodes.
        a_c = a.to(x.dtype)
        b_c = b.to(x.dtype)
        z = F.linear(x, a_c)
        if self.tp_size > 1 and self._a_sharded[key]:
            # Row-parallel target: z is a partial sum over input shards.
            z = _AllReduceTP.apply(z)
        return F.linear(z, b_c) * self.cfg.scaling

    def _add_delta(
        self, y: torch.Tensor, delta: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return y if delta is None else y + delta

    def _replay_block_hidden(
        self,
        flats: Dict[str, torch.Tensor],
        input_embeds: torch.Tensor,  # [B, hidden]
        positions: torch.Tensor,  # [B] int64
        prefix_kv: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Re-run the draft block forward eagerly, with autograd.

        Follows DFlashDraftModel.forward / DFlashDecoderLayer, using the
        native (autograd-capable) RMSNorm, RoPE and SDPA implementations in
        place of the fused serving kernels. Prefix K/V are constants read
        from the draft cache. Gradients flow through the block's own
        activations into the LoRA leaves only; the base weights are frozen.
        """
        hidden = input_embeds
        residual: Optional[torch.Tensor] = None
        n = hidden.shape[0]

        tp = self.tp_size > 1
        for layer_id, layer in enumerate(self.layers):
            attn = layer.self_attn
            if residual is None:
                residual = hidden
                hidden = layer.input_layernorm.forward_native(hidden)
            else:
                hidden, residual = layer.input_layernorm.forward_native(
                    hidden, residual
                )

            # Megatron "f" on the replicated input of the column-parallel
            # projection. Without it dL/d(hidden) would contain only the
            # local shard's contribution, and every upstream adapter leaf
            # would receive a partial gradient.
            hidden_f = _CopyToTP.apply(hidden) if tp else hidden

            # Attention (non-causal over [prefix | block]).
            qkv = F.linear(hidden_f, attn.qkv_proj.weight, attn.qkv_proj.bias)
            q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
            q = self._add_delta(q, self._lora_delta(flats, layer_id, "q", hidden_f))
            # k / v deltas when those kinds are targeted (None otherwise).
            # The k delta is applied before k_norm, as in serving.
            k = self._add_delta(k, self._lora_delta(flats, layer_id, "k", hidden_f))
            v = self._add_delta(v, self._lora_delta(flats, layer_id, "v", hidden_f))
            head_dim = attn.head_dim
            q = attn.q_norm.forward_native(
                q.reshape(-1, head_dim)
            ).view(n, attn.q_size)
            k = attn.k_norm.forward_native(
                k.reshape(-1, head_dim)
            ).view(n, attn.kv_size)
            q, k = attn.rotary_emb.forward_native(positions, q, k)

            k_prefix, v_prefix = prefix_kv[layer_id]
            q_h = q.view(n, attn.num_heads, head_dim)
            k_h = torch.cat(
                [k_prefix.to(k.dtype), k.view(n, attn.num_kv_heads, head_dim)], dim=0
            )
            v_h = torch.cat(
                [v_prefix.to(v.dtype), v.view(n, attn.num_kv_heads, head_dim)], dim=0
            )

            # Windowed layers: the fused kernel masks per query row within
            # the gathered tail. With T tail keys, block row m (absolute
            # position prefix_len + m) excludes tail key n_k (absolute
            # position prefix_len - T + n_k) when
            # abs_q - abs_k = T + m - n_k > window. Row m therefore masks its
            # first (T + m - window) tail keys whenever that is positive;
            # with a full-length tail (T == window) these are its first m
            # keys.
            window = self._layer_prefix_windows[layer_id]
            n_tail = k_prefix.shape[0]
            attn_mask = None
            if window is not None and n_tail + (n - 1) > window:
                q_rows = torch.arange(n, device=hidden.device).unsqueeze(1)
                k_idx = torch.arange(n_tail, device=hidden.device).unsqueeze(0)
                allowed = torch.cat(
                    [
                        k_idx >= (n_tail + q_rows - window),  # tail keys
                        torch.ones(
                            n, n, dtype=torch.bool, device=hidden.device
                        ),  # block keys: always visible (non-causal)
                    ],
                    dim=1,
                )
                attn_mask = allowed  # [n, n_tail + n], broadcasts over heads

            # SDPA layout: [1, heads, seq, head_dim]. With no mask, enable_gqa
            # broadcasts KV heads; with a mask, materialize the GQA repeat
            # (enable_gqa + attn_mask support varies across torch versions).
            if attn_mask is None:
                q_s = q_h.transpose(0, 1).unsqueeze(0)
                k_s = k_h.transpose(0, 1).unsqueeze(0)
                v_s = v_h.transpose(0, 1).unsqueeze(0)
                enable_gqa = attn.num_heads != attn.num_kv_heads
            else:
                n_rep = attn.num_heads // attn.num_kv_heads
                q_s = q_h.transpose(0, 1).unsqueeze(0)
                k_s = (
                    k_h.repeat_interleave(n_rep, dim=1)
                    .transpose(0, 1)
                    .unsqueeze(0)
                )
                v_s = (
                    v_h.repeat_interleave(n_rep, dim=1)
                    .transpose(0, 1)
                    .unsqueeze(0)
                )
                enable_gqa = False
            attn_out = F.scaled_dot_product_attention(
                q_s,
                k_s,
                v_s,
                attn_mask=attn_mask,
                scale=attn.scaling,
                is_causal=False,
                enable_gqa=enable_gqa,
            )
            attn_out = (
                attn_out.squeeze(0).transpose(0, 1).reshape(n, attn.q_size)
            )
            # Row-parallel o_proj: the base matmul over the local heads is a
            # partial sum. Reduce it ("g"), then add the bias once after the
            # reduce (RowParallelLinear semantics) and the LoRA delta, whose
            # own partial sum was reduced at the rank-r bottleneck inside
            # _lora_delta.
            o = F.linear(attn_out, attn.o_proj.weight)
            if tp:
                o = _AllReduceTP.apply(o)
            if attn.o_proj.bias is not None:
                o = o + attn.o_proj.bias
            o = self._add_delta(o, self._lora_delta(flats, layer_id, "o", attn_out))

            hidden, residual = layer.post_attention_layernorm.forward_native(
                o, residual
            )

            # MLP (SwiGLU); gate_up is column-parallel (f on its replicated
            # input), down is row-parallel (g on its partial sum).
            mlp = layer.mlp
            hidden_f2 = _CopyToTP.apply(hidden) if tp else hidden
            gate_up = F.linear(hidden_f2, mlp.gate_up_proj.weight)
            gate_up = self._add_delta(
                gate_up, self._lora_delta(flats, layer_id, "gate_up", hidden_f2)
            )
            inter = gate_up.shape[-1] // 2
            act = F.silu(gate_up[..., :inter]) * gate_up[..., inter:]
            down = F.linear(act, mlp.down_proj.weight)
            if tp:
                down = _AllReduceTP.apply(down)
            down = self._add_delta(down, self._lora_delta(flats, layer_id, "down", act))
            hidden = down

        hidden, _ = self.draft_model.norm.forward_native(hidden, residual)
        return hidden

    def _gather_prefix_kv_many(
        self, caps: List[_ReplayCapture]
    ) -> List[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Read the context K/V of the given captures from the draft KV cache.

        The replay treats them as constants. With the default targets they
        were written by projections that are never adapted (fc, hidden_norm,
        k_proj, v_proj, k_norm). Windowed layers gather only the last
        `window` prefix locations, which is the tail that the fused serving
        kernel exposes to the block.

        Each layer does one gather for K and one for V over the concatenated
        locations of all captures and splits the result per request. Runs on
        the main stream at enqueue time (see run_tts_steps), so that the side
        stream never reads the KV pool.
        """
        out: List[List[Tuple[torch.Tensor, torch.Tensor]]] = [
            [] for _ in caps
        ]
        for layer, window in zip(self.layers, self._layer_prefix_windows):
            layer_id = layer.self_attn.attn.layer_id
            k_buf, v_buf = self.token_to_kv_pool.get_kv_buffer(layer_id)
            locs_list = [
                cap.prefix_locs
                if window is None
                else cap.prefix_locs[-window:]
                for cap in caps
            ]
            lens = [int(l.shape[0]) for l in locs_list]
            cat_locs = (
                locs_list[0] if len(locs_list) == 1 else torch.cat(locs_list)
            )
            k_all = k_buf[cat_locs].detach()
            v_all = v_buf[cat_locs].detach()
            for i, (k_i, v_i) in enumerate(
                zip(k_all.split(lens), v_all.split(lens))
            ):
                out[i].append((k_i, v_i))
        return out

    # ---- Loss / step -----------------------------------------------------

    def _target_constants(
        self, t_logits: torch.Tensor  # [..., vocab] fp32
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Precompute the target-side constants of the loss: p_t, and for the
        "kl" loss ent = sum(p_t * log p_t), the negative target entropy (masked
        where p_t == 0, since log p_t is -inf there). The KL then splits as
        KL = ent - sum(p_t * log p_d). ent has no gradient, and the cross
        term needs no mask: log p_d is a finite fp32 log_softmax, so entries
        with p_t == 0 contribute exactly 0. Only p_t and ent are kept across
        inner steps; log p_t is not.
        """
        log_p_t = F.log_softmax(t_logits, dim=-1)
        p_t = log_p_t.exp()
        if self.cfg.tts_loss == "lk":
            return p_t, None
        ent = torch.where(
            p_t > 0, p_t * log_p_t, torch.zeros_like(p_t)
        ).sum(dim=-1)
        return p_t, ent

    def _compute_loss(
        self,
        draft_logits: torch.Tensor,  # [B-1, vocab], grad
        p_t: torch.Tensor,  # [B-1, vocab], constant (precomputed per capture)
        ent: Optional[torch.Tensor],  # [B-1], constant; "kl" loss only
    ) -> torch.Tensor:
        draft_logits = draft_logits.float()
        if self.cfg.logit_softcap:
            # The target's verify logits were softcapped by its
            # LogitsProcessor; cap the draft side identically.
            cap = float(self.cfg.logit_softcap)
            draft_logits = cap * torch.tanh(draft_logits / cap)

        log_p_d = F.log_softmax(draft_logits, dim=-1)
        if self.cfg.tts_loss == "lk":
            return (lk_per_position(p_t, log_p_d) * self._pos_weights).sum()

        # KL split (see _target_constants). The gradient with respect to
        # log_p_d is -p_t per element, the same as for the masked
        # p_t * (log_p_t - log_p_d) form.
        per_pos_kl = ent - (p_t * log_p_d).sum(dim=-1)

        # Self cross-entropy: -log_softmax at the draft's own argmax, read
        # from the log_p_d computed above. F.cross_entropy would compute the
        # [B-1, vocab] log_softmax, and its backward graph, a second time.
        with torch.no_grad():
            self_targets = draft_logits.argmax(dim=-1)
        per_pos_ce = -log_p_d.gather(1, self_targets.unsqueeze(1)).squeeze(1)

        kl_term = (per_pos_kl * self._pos_weights).sum()
        ce_term = (per_pos_ce * self._pos_weights).sum()
        lam = self.cfg.tts_lambda
        return lam * kl_term + (1.0 - lam) * ce_term

    def _load_block(self, block: torch.Tensor) -> torch.Tensor:
        """A new fp32 tensor on the device with the contents of a state block.
        The transfer from a pinned block carries the block's own bytes, and
        widening is exact. Allocations happen on the current stream (the side
        stream in async mode), which orders their lifetime with the update."""
        if self._offload:
            staged = torch.empty(
                block.shape, dtype=block.dtype, device=self.device
            )
            staged.copy_(block, non_blocking=True)
            block = staged
        return block.to(torch.float32)

    def _store_block(
        self, block: torch.Tensor, value: torch.Tensor, non_blocking: bool
    ) -> None:
        """Store an fp32 result into a state block. This is the one rounding of
        the result. For a pinned block it happens on the device, so that the
        transfer carries the block's own bytes."""
        if self._offload:
            value = value.to(block.dtype)
        block.copy_(value, non_blocking=non_blocking)

    def _materialize_leaves(
        self, rs: _ReqLoRAState
    ) -> Dict[str, torch.Tensor]:
        """The fp32 leaves for this request's update. With a resident
        optimizer: the resident Parameters themselves. Otherwise: new tensors
        filled from the master blocks, which are true leaves rather than
        views, so the in-place Adam update is safe."""
        if rs.opt is not None:
            return rs.flats
        leaves: Dict[str, torch.Tensor] = {}
        for g, block in rs.flats.items():
            t = self._load_block(block)
            t.requires_grad_(True)
            leaves[g] = t
        return leaves

    def _adam_step(
        self,
        rs: _ReqLoRAState,
        leaves: Dict[str, torch.Tensor],
        adam_ws: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """One Adam step. With a resident optimizer: that optimizer. Otherwise:
        torch's functional Adam over fp32 moment workspaces in `adam_ws`. A
        workspace is filled from the moment blocks on its first use in this
        update (zeros on the request's first ever step) and is carried across
        inner steps; the caller writes the workspaces back after the update.
        The step counter passed in is rs.step_ct before the
        increment, because the functional increments it internally, as
        torch.optim.Adam does. Leaves without a gradient are skipped and
        their moments left untouched, as in the resident optimizer."""
        if rs.opt is not None:
            rs.opt.step()
            return
        params, grads, ms, vs, steps = [], [], [], [], []
        for g, leaf in leaves.items():
            if leaf.grad is None or leaf.numel() == 0:
                continue
            if g not in adam_ws:
                if g in rs.moments_valid:
                    m = self._load_block(rs.adam_m[g])
                    v = self._load_block(rs.adam_v[g])
                else:
                    m = torch.zeros_like(leaf)
                    v = torch.zeros_like(leaf)
                adam_ws[g] = (m, v)
            m, v = adam_ws[g]
            params.append(leaf)
            grads.append(leaf.grad)
            ms.append(m)
            vs.append(v)
            steps.append(torch.tensor(float(rs.step_ct)))
        if not params:
            return
        with torch.no_grad():
            # foreach and fused stay None, so that the functional selects the
            # same implementation as a default-constructed torch.optim.Adam.
            # On CUDA that is the foreach implementation, whose results are
            # not bitwise equal to those of the single-tensor one. Forcing
            # foreach=False would therefore break parity with the resident
            # optimizer (see test_offload_adam_parity.py).
            self._functional_adam(
                params,
                grads,
                ms,
                vs,
                [],
                steps,
                foreach=None,
                capturable=False,
                differentiable=False,
                fused=None,
                grad_scale=None,
                found_inf=None,
                has_complex=False,
                decoupled_weight_decay=False,
                amsgrad=False,
                beta1=0.9,
                beta2=0.999,
                lr=float(self.cfg.lr),
                weight_decay=0.0,
                eps=1e-8,
                maximize=False,
            )

    def _writeback_state(
        self,
        rs: _ReqLoRAState,
        leaves: Dict[str, torch.Tensor],
        adam_ws: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Store the update's results into the master and moment blocks (no-op
        with a resident optimizer, whose Parameters were updated in place).
        Called from a finally block: if a failure follows an Adam step that
        already changed some leaves, the changes must still be persisted, as
        they are with a resident optimizer.

        The results are rounded here, once per update, to the optimizer dtype:
        inner steps carry on from the fp32 leaves and workspaces. A leaf that
        did not step keeps its bits, since widening and rounding are exact
        inverses on bf16 values.

        In async mode the copies are non-blocking. That is safe because the
        pinned targets are next read at this request's next update, behind at
        least one host fence, and are recycled only after a fence (a release
        parks the state while a step event is outstanding). Sync mode has no
        step event, so _release_state recycles immediately and nothing would
        order a pending copy before the next host-side write to the block;
        the copies therefore block. That adds little, since sync mode already
        synchronizes with the device at every update for the isfinite
        check."""
        if rs.opt is not None:
            return
        nb = self._use_async
        with torch.no_grad():
            for g, leaf in leaves.items():
                self._store_block(rs.flats[g], leaf, nb)
            for g, (m, v) in adam_ws.items():
                self._store_block(rs.adam_m[g], m, nb)
                self._store_block(rs.adam_v[g], v, nb)
        for g in adam_ws:
            rs.moments_valid.add(g)

    def _step_one(
        self,
        cap: _ReplayCapture,
        target_rows: torch.Tensor,  # [B-1, vocab]
        prefix_kv: List[Tuple[torch.Tensor, torch.Tensor]],
        lm_head_weight: torch.Tensor,  # [num_org_vocab, hidden], detached
    ) -> None:
        rs = self._reqs.get(cap.req_pool_idx)
        if rs is None:
            return
        leaves = self._materialize_leaves(rs)
        adam_ws: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        # The target-side quantities are constant for the capture; compute
        # them once instead of once per inner step.
        p_t, ent = self._target_constants(target_rows.float())

        # The writeback and the snapshot refresh must run even if an inner
        # step stops early (non-finite loss) or raises after an earlier Adam
        # step changed the leaves. Otherwise serving would keep publishing the
        # old adapters while later replays start from the newer masters.
        try:
            self._run_inner_steps(
                rs, leaves, adam_ws, cap, prefix_kv, p_t, ent, lm_head_weight
            )
        finally:
            # The refresh must run even if the writeback raises. Both read the
            # same leaves, so they stay consistent with each other.
            try:
                self._writeback_state(rs, leaves, adam_ws)
            finally:
                self._refresh_after_fire(rs, leaves)

    def _refresh_after_fire(
        self, rs: _ReqLoRAState, leaves: Dict[str, torch.Tensor]
    ) -> None:
        """Refresh the request's serving snapshot after an update.

        With delayed publish the snapshot goes to the request's other
        generation (the one this round's publish cannot be reading), which
        then becomes `front`, tagged with the current enqueue sequence
        number. It can be published once that sequence number has been
        fenced. The flip is committed only after the refresh has been
        enqueued without error: after a failed refresh `front` still points
        at the old, complete snapshot, and the next update overwrites the
        partially written generation in full. Without delayed publish the
        snapshot goes to generation 0."""
        gen = (1 - rs.front) if self._delayed else 0
        self._refresh_cached(rs, leaves, gen=gen)
        if self._delayed:
            rs.front = gen
            rs.front_seq = self._enqueue_seq

    # ---- Batched replay of several requests ------------------------------

    def _lora_delta_many(
        self,
        flats_list: List[Dict[str, torch.Tensor]],
        layer_id: int,
        kind: str,
        x: torch.Tensor,  # [R, n, in_features]
    ) -> Optional[torch.Tensor]:
        key = (layer_id, kind)
        if key not in self._flat_layout:
            return None
        pairs = [self._leaf_views(f, key) for f in flats_list]
        # Stacking the per-request views keeps each request's gradient routed
        # to its own flat leaf (the backward of stack slices the gradient per
        # request, and the view nodes carry it into the leaf). The cast is
        # inside the graph so that the replay matches serving numerics, as in
        # _lora_delta.
        a_s = torch.stack([p[0] for p in pairs]).to(x.dtype)  # [R, r, in]
        b_s = torch.stack([p[1] for p in pairs]).to(x.dtype)  # [R, out, r]
        # torch.ops.aten.bmm rather than torch.bmm: deterministic inference
        # mode (batch_invariant_ops.enable_batch_invariant_mode) replaces
        # `torch.bmm` with a Python-level function that bypasses the
        # dispatcher and therefore autograd. Its output has no grad_fn, so an
        # update computed through it would silently have no effect. The aten
        # op records the autograd node and then dispatches to the same
        # batch-invariant kernel.
        z = torch.ops.aten.bmm(x, a_s.transpose(1, 2))  # [R, n, r]
        if self.tp_size > 1 and self._a_sharded[key]:
            # Row-parallel target: z is a partial sum over input shards.
            z = _AllReduceTP.apply(z)
        return torch.ops.aten.bmm(z, b_s.transpose(1, 2)) * self.cfg.scaling

    def _replay_block_hidden_many(
        self,
        flats_list: List[Dict[str, torch.Tensor]],
        input_embeds: torch.Tensor,  # [R, B, hidden]
        positions: torch.Tensor,  # [R, B] int64
        prefix_kvs: List[List[Tuple[torch.Tensor, torch.Tensor]]],
        buckets_per_layer: Optional[List[List[List[int]]]] = None,
    ) -> torch.Tensor:  # [R, B, hidden]
        """Batched replay of the block forwards of R requests.

        Computes, for each request, the same function as
        _replay_block_hidden. Base linears, norms and RoPE run once on the
        flattened [R*B, ...] rows (weights are read once per layer per step).
        LoRA deltas use a stacked bmm over the per-request adapters.
        Attention is bucketed by prefix length: requests with equal prefix
        length (common when they were admitted in the same prefill) share one
        batched SDPA call, and a request with a unique length forms a bucket
        of its own. Gradients reach each request's own fp32 leaves through
        the stack / slice nodes.
        """
        R, B, _ = input_embeds.shape
        n = R * B
        hidden = input_embeds.reshape(n, -1)
        pos_flat = positions.reshape(n)
        residual: Optional[torch.Tensor] = None

        tp = self.tp_size > 1
        for layer_id, layer in enumerate(self.layers):
            attn = layer.self_attn
            if residual is None:
                residual = hidden
                hidden = layer.input_layernorm.forward_native(hidden)
            else:
                hidden, residual = layer.input_layernorm.forward_native(
                    hidden, residual
                )

            # Megatron f on the replicated input of the column-parallel
            # projection (see _replay_block_hidden).
            hidden_f = _CopyToTP.apply(hidden) if tp else hidden

            qkv = F.linear(hidden_f, attn.qkv_proj.weight, attn.qkv_proj.bias)
            q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
            q_delta = self._lora_delta_many(
                flats_list, layer_id, "q", hidden_f.view(R, B, -1)
            )
            if q_delta is not None:
                q = q + q_delta.reshape(n, -1)
            # k / v deltas when targeted; the k delta precedes k_norm.
            k_delta = self._lora_delta_many(
                flats_list, layer_id, "k", hidden_f.view(R, B, -1)
            )
            if k_delta is not None:
                k = k + k_delta.reshape(n, -1)
            v_delta = self._lora_delta_many(
                flats_list, layer_id, "v", hidden_f.view(R, B, -1)
            )
            if v_delta is not None:
                v = v + v_delta.reshape(n, -1)
            head_dim = attn.head_dim
            q = attn.q_norm.forward_native(q.reshape(-1, head_dim)).view(
                n, attn.q_size
            )
            k = attn.k_norm.forward_native(k.reshape(-1, head_dim)).view(
                n, attn.kv_size
            )
            q, k = attn.rotary_emb.forward_native(pos_flat, q, k)

            window = self._layer_prefix_windows[layer_id]
            q_r = q.view(R, B, attn.num_heads, head_dim)
            k_r = k.view(R, B, attn.num_kv_heads, head_dim)
            v_r = v.view(R, B, attn.num_kv_heads, head_dim)
            # Bucket by prefix length: requests with equal prefix length share
            # one batched SDPA call, which only adds a batch axis. The mask
            # and the window geometry depend only on n_tail, so every bucket
            # is homogeneous. Prefix lengths are the same on every rank, and
            # so is the bucket structure. Callers normally pass the buckets
            # precomputed by _prefix_buckets; the fallback below builds the
            # same grouping in the same order.
            if buckets_per_layer is not None:
                layer_buckets = buckets_per_layer[layer_id]
            else:
                buckets: Dict[int, List[int]] = {}
                for r in range(R):
                    buckets.setdefault(
                        int(prefix_kvs[r][layer_id][0].shape[0]), []
                    ).append(r)
                layer_buckets = list(buckets.values())
            attn_out_rows: List[Optional[torch.Tensor]] = [None] * R
            for idxs in layer_buckets:
                k_pref = torch.stack(
                    [prefix_kvs[r][layer_id][0] for r in idxs]
                ).to(k.dtype)  # [G, n_tail, kv, D]
                v_pref = torch.stack(
                    [prefix_kvs[r][layer_id][1] for r in idxs]
                ).to(v.dtype)
                # Every request of a bucket has this prefix (tail) length.
                n_tail = k_pref.shape[1]
                k_g = torch.cat([k_pref, k_r[idxs]], dim=1)  # [G, L, kv, D]
                v_g = torch.cat([v_pref, v_r[idxs]], dim=1)
                attn_mask = None
                if window is not None and n_tail + (B - 1) > window:
                    # Per-query-row tail mask, same as _replay_block_hidden;
                    # [B, L] broadcasts over the bucket and head dims.
                    q_rows = torch.arange(B, device=hidden.device).unsqueeze(1)
                    k_idx = torch.arange(
                        n_tail, device=hidden.device
                    ).unsqueeze(0)
                    attn_mask = torch.cat(
                        [
                            k_idx >= (n_tail + q_rows - window),
                            torch.ones(
                                B, B, dtype=torch.bool, device=hidden.device
                            ),
                        ],
                        dim=1,
                    )
                q_s = q_r[idxs].transpose(1, 2)  # [G, H, B, D]
                if attn_mask is None:
                    k_s = k_g.transpose(1, 2)  # [G, kv, L, D]
                    v_s = v_g.transpose(1, 2)
                    enable_gqa = attn.num_heads != attn.num_kv_heads
                else:
                    n_rep = attn.num_heads // attn.num_kv_heads
                    k_s = k_g.repeat_interleave(n_rep, dim=2).transpose(1, 2)
                    v_s = v_g.repeat_interleave(n_rep, dim=2).transpose(1, 2)
                    enable_gqa = False
                out_g = F.scaled_dot_product_attention(
                    q_s,
                    k_s,
                    v_s,
                    attn_mask=attn_mask,
                    scale=attn.scaling,
                    is_causal=False,
                    enable_gqa=enable_gqa,
                )  # [G, H, B, D]
                out_g = out_g.transpose(1, 2).reshape(-1, B, attn.q_size)
                for j, r in enumerate(idxs):
                    attn_out_rows[r] = out_g[j]
            attn_out = torch.cat(attn_out_rows, dim=0)  # [R*B, q_size]

            o = F.linear(attn_out, attn.o_proj.weight)
            if tp:
                o = _AllReduceTP.apply(o)
            if attn.o_proj.bias is not None:
                o = o + attn.o_proj.bias
            o_delta = self._lora_delta_many(
                flats_list, layer_id, "o", attn_out.view(R, B, -1)
            )
            if o_delta is not None:
                o = o + o_delta.reshape(n, -1)

            hidden, residual = layer.post_attention_layernorm.forward_native(
                o, residual
            )

            mlp = layer.mlp
            hidden_f2 = _CopyToTP.apply(hidden) if tp else hidden
            gate_up = F.linear(hidden_f2, mlp.gate_up_proj.weight)
            gu_delta = self._lora_delta_many(
                flats_list, layer_id, "gate_up", hidden_f2.view(R, B, -1)
            )
            if gu_delta is not None:
                gate_up = gate_up + gu_delta.reshape(n, -1)
            inter = gate_up.shape[-1] // 2
            act = F.silu(gate_up[..., :inter]) * gate_up[..., inter:]
            down = F.linear(act, mlp.down_proj.weight)
            if tp:
                down = _AllReduceTP.apply(down)
            d_delta = self._lora_delta_many(
                flats_list, layer_id, "down", act.view(R, B, -1)
            )
            if d_delta is not None:
                down = down + d_delta.reshape(n, -1)
            hidden = down

        hidden, _ = self.draft_model.norm.forward_native(hidden, residual)
        return hidden.view(R, B, -1)

    def _compute_loss_many(
        self,
        draft_logits: torch.Tensor,  # [R*(B-1), vocab], grad
        p_t: torch.Tensor,  # [R, B-1, vocab], constant
        ent: Optional[torch.Tensor],  # [R, B-1], constant; "kl" loss only
    ) -> torch.Tensor:  # [R] per-request losses
        R, num_pos = p_t.shape[0], p_t.shape[1]
        draft_logits = draft_logits.float()
        if self.cfg.logit_softcap:
            cap = float(self.cfg.logit_softcap)
            draft_logits = cap * torch.tanh(draft_logits / cap)
        log_p_d = F.log_softmax(draft_logits, dim=-1)
        pt = p_t.reshape(R * num_pos, -1)
        if self.cfg.tts_loss == "lk":
            per_pos_lk = lk_per_position(pt, log_p_d).view(R, num_pos)
            return (per_pos_lk * self._pos_weights).sum(dim=1)  # [R]

        # KL split, same as _compute_loss (ent constant, cross term
        # needs no mask).
        per_pos_kl = ent - (pt * log_p_d).sum(dim=-1).view(R, num_pos)
        with torch.no_grad():
            self_targets = draft_logits.argmax(dim=-1)
        per_pos_ce = (
            -log_p_d.gather(1, self_targets.unsqueeze(1))
            .squeeze(1)
            .view(R, num_pos)
        )
        kl_term = (per_pos_kl * self._pos_weights).sum(dim=1)  # [R]
        ce_term = (per_pos_ce * self._pos_weights).sum(dim=1)  # [R]
        lam = self.cfg.tts_lambda
        return lam * kl_term + (1.0 - lam) * ce_term

    # ---- Forward and loss of one inner step ------------------------------
    #
    # _forward_loss_one / _forward_loss_many run the block replay, the lm_head
    # and the loss. They do not modify any state of `self`.

    def _prefix_buckets(
        self, prefix_kvs: List[List[Tuple[torch.Tensor, torch.Tensor]]]
    ) -> List[List[List[int]]]:
        """Per-layer buckets of request indices with equal prefix (tail)
        length: layer -> bucket -> request indices. Buckets are ordered by
        first occurrence, and the indices within a bucket are ascending."""
        R = len(prefix_kvs)
        out: List[List[List[int]]] = []
        for layer_id in range(len(self.layers)):
            buckets: Dict[int, List[int]] = {}
            for r in range(R):
                buckets.setdefault(
                    int(prefix_kvs[r][layer_id][0].shape[0]), []
                ).append(r)
            out.append(list(buckets.values()))
        return out

    def _forward_loss_one(
        self,
        flats: Dict[str, torch.Tensor],
        input_embeds: torch.Tensor,  # [B, hidden]
        positions: torch.Tensor,  # [B] int64
        prefix_kv: List[Tuple[torch.Tensor, torch.Tensor]],
        lm_head_weight: torch.Tensor,
        p_t: torch.Tensor,  # [B-1, vocab] constant
        ent: Optional[torch.Tensor],  # [B-1] constant; "kl" loss only
    ) -> torch.Tensor:  # scalar loss
        """Forward and loss of one inner step for a single capture."""
        block = self.block_size
        hidden = self._replay_block_hidden(
            flats, input_embeds, positions, prefix_kv
        )
        # The block is non-causal: hidden[k] predicts block position k.
        # Position 0 holds the already-verified bonus token, so rows 1..B-1
        # are scored against the causal target's verify rows 0..B-2. With
        # tp_size > 1 the lm_head is vocab-sharded: apply "f" to the
        # replicated head input (dL/dhidden must sum the per-shard
        # contributions, otherwise every upstream gradient is partial and the
        # replicated leaves diverge across ranks), compute the local shard's
        # logits, and gather them differentiably (the shards are contiguous
        # vocab ranges in rank order).
        head_in = hidden[1:block]
        if self.tp_size > 1:
            head_in = _CopyToTP.apply(head_in)
        draft_logits = F.linear(head_in, lm_head_weight)
        if self.tp_size > 1:
            draft_logits = _VocabAllGather.apply(draft_logits)
        if draft_logits.shape[-1] != p_t.shape[-1]:
            raise RuntimeError(
                f"draft logits vocab ({draft_logits.shape[-1]}) != target "
                f"vocab ({p_t.shape[-1]}) after gather — padded "
                "or unequal lm_head shards are not supported."
            )
        return self._compute_loss(draft_logits, p_t, ent)

    def _forward_loss_many(
        self,
        flats_list: List[Dict[str, torch.Tensor]],
        embeds: torch.Tensor,  # [R, B, hidden]
        positions: torch.Tensor,  # [R, B] int64
        prefix_kvs: List[List[Tuple[torch.Tensor, torch.Tensor]]],
        buckets_per_layer: List[List[List[int]]],
        lm_head_weight: torch.Tensor,
        p_t: torch.Tensor,  # [R, B-1, vocab] constant
        ent: Optional[torch.Tensor],  # [R, B-1] constant; "kl" loss only
    ) -> torch.Tensor:  # [R] per-request losses
        """Forward and per-request losses of one inner step for a chunk."""
        R = embeds.shape[0]
        block = self.block_size  # captures are always full [block, hidden] rows
        num_pos = block - 1
        hidden = self._replay_block_hidden_many(
            flats_list,
            embeds,
            positions,
            prefix_kvs,
            buckets_per_layer=buckets_per_layer,
        )
        head_in = hidden[:, 1:block].reshape(R * num_pos, -1)
        if self.tp_size > 1:
            head_in = _CopyToTP.apply(head_in)
        draft_logits = F.linear(head_in, lm_head_weight)
        if self.tp_size > 1:
            draft_logits = _VocabAllGather.apply(draft_logits)
        if draft_logits.shape[-1] != p_t.shape[-1]:
            raise RuntimeError(
                f"draft logits vocab ({draft_logits.shape[-1]}) != "
                f"target vocab ({p_t.shape[-1]}) after gather — padded or "
                "unequal lm_head shards are not supported."
            )
        return self._compute_loss_many(draft_logits, p_t, ent)

    def _step_many(
        self,
        chunk: List[
            Tuple[
                _ReplayCapture,
                torch.Tensor,
                List[Tuple[torch.Tensor, torch.Tensor]],
            ]
        ],
        lm_head_weight: torch.Tensor,
    ) -> None:
        """Batched replay and per-request Adam for a chunk of captures.

        Equivalent to updating the requests one at a time, up to the
        floating-point reassociation of batched GEMMs: the total loss is the
        sum of the per-request losses, and request i's loss depends only on
        request i's adapter leaves, so the batched backward yields the
        per-request gradients with no cross terms. Clipping, completing the
        replicated-A gradients under TP, and Adam stay per request, in
        capture order. That order is deterministic and the same on every
        rank, so the TP collectives are enqueued in the same order
        everywhere.
        """
        caps = [cap for cap, _t, _kv in chunk]
        rs_list = [self._reqs[cap.req_pool_idx] for cap in caps]
        R = len(caps)
        prefix_kvs = [kv for _c, _t, kv in chunk]
        embeds = torch.stack([cap.input_embeds for cap in caps])  # [R,B,h]
        positions = torch.stack([cap.positions for cap in caps])  # [R,B]

        # Target-side constants, computed once per chunk. Only p_t [R,B-1,V]
        # and ent [R,B-1] are held across inner steps. The stacked fp32
        # logits and log_p_t are freed when this returns; they are allocated
        # and used on the same stream, so their memory can be reused safely.
        p_t, ent = self._target_constants(
            torch.stack([t for _c, t, _kv in chunk]).float()  # [R,B-1,V]
        )

        # The bucket structure is constant across inner steps; compute it
        # once per chunk (see _prefix_buckets).
        buckets_per_layer = self._prefix_buckets(prefix_kvs)

        leaves_list = [self._materialize_leaves(rs) for rs in rs_list]
        ws_list: List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = [
            {} for _ in rs_list
        ]
        # The writeback and the refresh must run even after a partial failure
        # that follows an Adam step (see _step_one).
        try:
            for _ in range(max(1, self.cfg.steps)):
                loss_per_req = self._forward_loss_many(
                    leaves_list,
                    embeds,
                    positions,
                    prefix_kvs,
                    buckets_per_layer,
                    lm_head_weight,
                    p_t,
                    ent,
                )

                for lv in leaves_list:
                    for p in lv.values():
                        p.grad = None
                stepping = list(range(R))
                if not self._use_async:
                    # One host sync for the whole chunk. Requests with a
                    # non-finite loss are excluded from the summed loss and
                    # their Adam step is skipped.
                    finite = torch.isfinite(loss_per_req)
                    if self.tp_size > 1:
                        # The mask decides which requests join the collectives
                        # after backward, so it must be the same on every rank
                        # by construction, without relying on the loss being
                        # bitwise equal across ranks. Combine the per-rank
                        # masks with a logical AND, computed as a summed 0/1
                        # all-reduce. The reduce uses float32 because the
                        # custom all-reduce kernel supports only
                        # floating-point dtypes; sums of 0.0 / 1.0 are exact.
                        finite = (
                            _tts_all_reduce(
                                finite.to(torch.float32).contiguous()
                            )
                            >= self.tp_size - 0.5
                        )
                    if not bool(finite.all()):
                        if not self._warned_nonfinite:
                            logger.warning(
                                "DFlash LoRA-TTS non-finite loss; skipping "
                                "affected request steps (warned once)"
                            )
                            self._warned_nonfinite = True
                        stepping = [i for i in range(R) if bool(finite[i])]
                        if not stepping:
                            return
                        loss = loss_per_req[finite].sum()
                    else:
                        loss = loss_per_req.sum()
                else:
                    # Async mode skips the isfinite check (reading it would
                    # synchronize the side stream); nan_to_num_ below is the
                    # safeguard. Non-finite losses are counted on the GPU
                    # (see _nonfinite_ct_gpu).
                    self._nonfinite_ct_gpu += (
                        (~torch.isfinite(loss_per_req)).sum()
                    )
                    loss = loss_per_req.sum()
                loss.backward()

                # Phase 1: zero the non-finite gradient elements before any
                # collective or clipping (see _run_inner_steps).
                for i in stepping:
                    for p in leaves_list[i].values():
                        if p.grad is not None:
                            torch.nan_to_num_(
                                p.grad, nan=0.0, posinf=0.0, neginf=0.0
                            )
                if self.tp_size > 1:
                    # Phase 2: complete the replicated-A gradients with one
                    # stacked all-reduce for the whole chunk. The elementwise
                    # sum keeps each request's slice separate, so the result
                    # equals per-request reduces with fewer NCCL calls.
                    # `stepping` is the same on every rank: in async mode it
                    # is range(R), and in sync mode it comes from the
                    # all-reduced finite mask above. Whether a gradient is
                    # None depends only on the model structure.
                    idxs = [
                        i
                        for i in stepping
                        if leaves_list[i]["a_repl"].grad is not None
                    ]
                    if idxs and self._flat_sizes["a_repl"] > 0:
                        reduced = _tts_all_reduce(
                            torch.stack(
                                [
                                    leaves_list[i]["a_repl"].grad
                                    for i in idxs
                                ]
                            ).contiguous()
                        )
                        for j, i in enumerate(idxs):
                            leaves_list[i]["a_repl"].grad = reduced[j]
                    if self.cfg.grad_clip > 0:
                        self._clip_grads_tp_many(
                            [leaves_list[i] for i in stepping]
                        )
                elif self.cfg.grad_clip > 0:
                    for i in stepping:
                        torch.nn.utils.clip_grad_norm_(
                            list(leaves_list[i].values()),
                            self.cfg.grad_clip,
                        )
                # Phase 3: per-request Adam. Steps are never merged across
                # requests, whose step counts and bias corrections differ.
                for i in stepping:
                    self._adam_step(rs_list[i], leaves_list[i], ws_list[i])
                    rs_list[i].step_ct += 1
        finally:
            for rs, lv, ws in zip(rs_list, leaves_list, ws_list):
                # Contain failures per request: a failure in one request's
                # writeback / refresh must not skip those of the others. (A
                # skipped refresh after an Adam step would leave serving on
                # the old adapters while later replays use the new masters.)
                try:
                    self._writeback_state(rs, lv, ws)
                    self._refresh_after_fire(rs, lv)
                except Exception as e:  # noqa: BLE001
                    if self.tp_size > 1:
                        raise
                    if not self._warned_step_failed:
                        logger.warning(
                            "DFlash LoRA-TTS writeback/refresh failed for "
                            "one request (warned once): %s",
                            e,
                        )
                        self._warned_step_failed = True

    def _run_inner_steps(
        self,
        rs: _ReqLoRAState,
        leaves: Dict[str, torch.Tensor],
        adam_ws: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        cap: _ReplayCapture,
        prefix_kv: List[Tuple[torch.Tensor, torch.Tensor]],
        p_t: torch.Tensor,
        ent: Optional[torch.Tensor],
        lm_head_weight: torch.Tensor,
    ) -> None:
        for _ in range(max(1, self.cfg.steps)):
            # Forward and loss. See _forward_loss_one for the head and loss
            # details, including where the TP collectives are placed.
            loss = self._forward_loss_one(
                leaves,
                cap.input_embeds,
                cap.positions,
                prefix_kv,
                lm_head_weight,
                p_t,
                ent,
            )

            for p in leaves.values():
                p.grad = None
            loss.backward()
            # A non-finite loss would corrupt the Adam moments, and with
            # momentum even a zeroed gradient is not a no-op, so sync mode
            # skips the step entirely. The check comes after backward:
            # reading isfinite synchronizes with the device, and gradients
            # left by a skipped step are cleared before the next backward.
            # Async mode omits the check, because the read would synchronize
            # the side stream and remove the overlap with serving. There,
            # nan_to_num_ below maps inf / NaN gradients to 0, so a
            # non-finite loss becomes a near-no-op step.
            if not self._use_async:
                loss_finite = torch.isfinite(loss)
                if self.tp_size > 1:
                    # Decide collectively (see _step_many). The early return
                    # below comes before the a_repl all-reduce, so ranks that
                    # disagreed on finiteness would deadlock. float32 because
                    # the custom all-reduce supports only floating-point
                    # dtypes.
                    loss_finite = (
                        float(
                            _tts_all_reduce(
                                loss_finite.to(torch.float32)
                                .reshape(1)
                                .contiguous()
                            ).item()
                        )
                        >= self.tp_size - 0.5
                    )
                if not loss_finite:
                    if not self._warned_nonfinite:
                        logger.warning(
                            "DFlash LoRA-TTS non-finite loss; skipping step "
                            "(warned once)"
                        )
                        self._warned_nonfinite = True
                    return
            if self._use_async:
                # Counted on the GPU; no host read (see _nonfinite_ct_gpu).
                self._nonfinite_ct_gpu += (~torch.isfinite(loss)).to(
                    torch.int64
                )
            trainable = list(leaves.values())
            # Zero the non-finite gradient elements before clipping: a single
            # NaN makes clip_grad_norm_'s total norm NaN, and its scale would
            # then turn every gradient into NaN.
            for p in trainable:
                if p.grad is not None:
                    torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
            if self.tp_size > 1:
                # A of column-parallel targets is replicated, but its backward
                # saw only the contribution of the local B shard. One
                # all-reduce over the 'a_repl' flat leaf completes all of
                # these gradients. (B of row-parallel targets is replicated
                # too, but its gradient is already complete and the same on
                # every rank, because both of its inputs, the reduced z and
                # the replicated upstream gradient, are complete.)
                g = leaves["a_repl"].grad
                if g is not None and self._flat_sizes["a_repl"] > 0:
                    leaves["a_repl"].grad = _tts_all_reduce(
                        g.contiguous()
                    )
                if self.cfg.grad_clip > 0:
                    self._clip_grads_tp(leaves)
            elif self.cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, self.cfg.grad_clip)
            self._adam_step(rs, leaves, adam_ws)
            rs.step_ct += 1

    # Disable TTS after this many consecutive rounds whose update raised (even
    # if some requests of the round were updated before the failure). A
    # persistent failure, such as a long-context replay that runs out of
    # memory every time, would otherwise be retried on every round.
    _MAX_STEP_FAILURE_STREAK = 3
    # Maximum number of pinned-CPU state blocks kept for reuse (see
    # _recycle_cpu_state). A block holds one request's masters and both Adam
    # moments, i.e. 12 bytes per adapter parameter (6 with the bfloat16
    # optimizer dtype).
    _CPU_CACHE_MAX = 32

    def note_round_failure(self, reason: str) -> None:
        """Record a failed TTS round (update, logits snapshot, capture /
        publish, or shape mismatch). After _MAX_STEP_FAILURE_STREAK
        consecutive failures TTS is disabled."""
        self._step_failure_streak += 1
        if self._step_failure_streak >= self._MAX_STEP_FAILURE_STREAK:
            self.disable(
                f"{self._step_failure_streak} consecutive failed rounds "
                f"(last: {reason})"
            )

    def _clip_grads_tp(self, leaves: Dict[str, torch.Tensor]) -> None:
        """Global grad-norm clip across a TP-sharded adapter set.

        Sharded leaves' squared norms are partial per rank and summed via
        all-reduce; replicated leaves (identical grads on every rank) are
        counted exactly once outside the reduce. The resulting scale is
        rank-identical, keeping replicated leaves in sync.
        """
        sharded_sq = torch.zeros((), dtype=torch.float32, device=self.device)
        replicated_sq = torch.zeros((), dtype=torch.float32, device=self.device)
        # B shards opposite to A: col-parallel B ('b_col') is sharded,
        # row-parallel B ('b_repl') is replicated.
        for name, p in leaves.items():
            if p.grad is None:
                continue
            sq = p.grad.float().pow(2).sum()
            (
                sharded_sq if name in ("b_col", "a_row") else replicated_sq
            ).add_(sq)
        total_sq = _tts_all_reduce(sharded_sq) + replicated_sq
        total_norm = total_sq.sqrt()
        clip = self.cfg.grad_clip
        scale = (clip / (total_norm + 1e-6)).clamp(max=1.0)
        for p in leaves.values():
            if p.grad is not None:
                p.grad.mul_(scale)

    def _clip_grads_tp_many(
        self, leaves_sub: List[Dict[str, torch.Tensor]]
    ) -> None:
        """Per-request global-norm clipping with one all-reduce for the whole
        chunk. The per-request squared norms of the sharded leaves travel in
        a single [R] vector (the elementwise sum keeps the requests
        separate), so every request gets the same norm and scale as from
        _clip_grads_tp."""
        sharded, replicated = [], []
        for lv in leaves_sub:
            s = torch.zeros((), dtype=torch.float32, device=self.device)
            rp = torch.zeros((), dtype=torch.float32, device=self.device)
            for name, p in lv.items():
                if p.grad is None:
                    continue
                sq = p.grad.float().pow(2).sum()
                (s if name in ("b_col", "a_row") else rp).add_(sq)
            sharded.append(s)
            replicated.append(rp)
        total_sq = _tts_all_reduce(
            torch.stack(sharded).contiguous()
        ) + torch.stack(replicated)
        total_norm = total_sq.sqrt()
        clip = self.cfg.grad_clip
        scales = (clip / (total_norm + 1e-6)).clamp(max=1.0)
        for i, lv in enumerate(leaves_sub):
            for p in lv.values():
                if p.grad is not None:
                    p.grad.mul_(scales[i])

    def run_tts_steps(
        self,
        *,
        target_logits: torch.Tensor,  # [n_captures, block_size - 1, vocab]
        lm_head_weight: torch.Tensor,
        skip_req_pool_indices: Optional[Set[int]] = None,
    ) -> None:
        """Replay and Adam step for every captured request.

        `target_logits[i]` holds the target's verify logits, taken before
        verify() modifies them, for the batch row of `self._captures[i]`.

        Failure policy: with tp_size == 1 this never raises; a failed update
        only loses adaptation, never correctness. With tp_size > 1 exceptions
        propagate. A failure on one rank in the middle of a sequence of
        collectives leaves the ranks' NCCL call order inconsistent, which is
        much harder to diagnose than a crashed scheduler."""
        if not self._captures:
            return
        skip = skip_req_pool_indices or set()

        num_pos = self.block_size - 1
        if target_logits.shape[:2] != (len(self._captures), num_pos):
            # Count this as a failed round instead of raising. A systematic
            # mismatch then disables TTS after a few rounds instead of
            # warning forever, and the caller's advance() / drop()
            # bookkeeping still runs. The shapes derive from state that is
            # the same on every rank, so with tp_size > 1 all ranks take this
            # branch together, before any collective has been enqueued.
            msg = (
                f"target logits shape {tuple(target_logits.shape)} does "
                f"not match {len(self._captures)} captures x {num_pos} "
                f"scoreable positions."
            )
            self._captures = []
            self.note_round_failure(f"shape: {msg}")
            if not self._warned_step_failed:
                logger.warning(
                    "DFlash LoRA-TTS step skipped (warned once): %s", msg
                )
                self._warned_step_failed = True
            return
        # The list of requests to update is built, and their prefix K/V
        # gathered, here on the main stream: the side stream must never read
        # the KV pool. release_memory_occupation can unmap KV memory before
        # its internal flush_cache reaches the TTS fence, so an update that is
        # still running at that point (e.g. all requests aborted between
        # rounds, followed by an immediate release) would read unmapped
        # memory if it touched the pool. The gathered values equal what a
        # later side-stream read would see: nothing writes the captured
        # locations between the enqueue and the next round's fence.
        caps_live = [
            (i, cap)
            for i, cap in enumerate(self._captures)
            if cap.req_pool_idx not in skip
            and self._reqs.get(cap.req_pool_idx) is not None
        ]
        # Chunk here, so that the prefix-KV gather is done per chunk and not
        # for all captures up front. The gather grows with the context length
        # and can be large for long contexts; gathering per chunk keeps about
        # two chunks' worth alive (the side stream consumes chunk k while the
        # main stream gathers chunk k+1). The gather is also the allocation
        # most likely to fail, so it runs inside the same error handling as
        # the update itself.
        # inference_mode(False): the gathered constants are saved for
        # backward and must not be inference tensors.
        chunk_lists = [
            caps_live[s : s + self._replay_chunk]
            for s in range(0, len(caps_live), self._replay_chunk)
        ]

        def _chunk_active(ch):
            with torch.inference_mode(False), torch.no_grad():
                pkvs = self._gather_prefix_kv_many([c for _, c in ch])
            # Causal target verify row k predicts block position k+1; the
            # snapshot holds exactly rows 0..B-2.
            return [
                (cap, target_logits[i].detach(), pkv)
                for (i, cap), pkv in zip(ch, pkvs)
            ]

        if not caps_live:
            # Nothing to update this round (all captured requests finished).
            # The failure streak is deliberately not reset here. A round
            # without an update is no evidence of recovery, and resetting
            # would let a failure that recurs on every update, interleaved
            # with empty rounds, never reach the disable threshold.
            self._captures = []
            # Non-blocking drain: if earlier updates have already finished,
            # release their parked state now instead of one round later.
            # Skipped with tp_size > 1 in delayed mode: popping advances
            # _synced_seq, which feeds the generation selection, and the
            # result of query() depends on device timing (see
            # fence_for_publish).
            if self.tp_size == 1 or not self._delayed:
                while self._step_events and self._step_events[0][1].query():
                    seq, _ev = self._step_events.pop(0)
                    if seq > self._synced_seq:
                        self._synced_seq = seq
                self._drain_deferred()
            return

        if not self._use_async:
            try:
                for ch in chunk_lists:
                    active = _chunk_active(ch)
                    with torch.inference_mode(False), torch.enable_grad():
                        self._run_chunk(active, lm_head_weight)
                self._step_failure_streak = 0
            except Exception as e:  # noqa: BLE001
                self._handle_step_exception(e)
            finally:
                self._captures = []
            return

        # ---- async: enqueue the round's updates on the side stream ----
        # The previous in-flight update wrote the masters / pool rows that
        # this one reads, and the side stream must see the main stream's
        # writes to the captures, the logits snapshot and the gathered K/V
        # before it uses them. The setup is inside the try block so that a
        # host-side failure before anything is enqueued is counted like any
        # other; recording the event in the finally block is harmless on an
        # idle stream.
        step_exception: Optional[Exception] = None
        self._enqueue_seq += 1
        try:
            if self._delayed:
                # Lag-1 here as well. The previous round's update may still
                # be running, and this enqueue may read what it wrote (for
                # example the same request's pinned masters at freq == 1).
                # Both run on the same side stream, so stream order puts the
                # earlier writeback before this enqueue's reads without a
                # host wait. Only older events are synchronized here, which
                # bounds the event queue and drains fenced deferred frees.
                self._sync_events(1)
            else:
                self.wait_for_pending()
            # record_stream: these tensors were allocated on the main stream
            # and their Python references are dropped when this call returns.
            # Without it the caching allocator could reuse their memory for
            # main-stream allocations while the side stream still reads them.
            # lm_head_weight is a view of a persistent Parameter and needs
            # none. The gathered prefix K/V are views; record_stream marks
            # their shared base storage, and repeated calls on the same
            # storage are cheap.
            target_logits.record_stream(self._tts_stream)
            for ch in chunk_lists:
                active = _chunk_active(ch)  # gathered on the main stream
                for cap, _t, pkv in active:
                    cap.input_embeds.record_stream(self._tts_stream)
                    cap.positions.record_stream(self._tts_stream)
                    for k_p, v_p in pkv:
                        k_p.record_stream(self._tts_stream)
                        v_p.record_stream(self._tts_stream)
                # The side stream must see this chunk's gather (and, for the
                # first chunk, the round's capture and snapshot writes) before
                # it uses them.
                self._tts_stream.wait_stream(
                    torch.cuda.current_stream(self.device)
                )
                with torch.cuda.stream(self._tts_stream):
                    with torch.inference_mode(False), torch.enable_grad():
                        self._run_chunk(active, lm_head_weight)
        except Exception as e:  # noqa: BLE001
            step_exception = e
        finally:
            # Record the event in a finally block, before any exception
            # handling. Without a recorded event, later fences would be no-ops
            # while kernels of a partial enqueue still write adapter state,
            # and a failure path that reaches disable() would free tensors
            # that the side stream is writing. The event covers everything
            # enqueued on the stream so far, including partial work.
            step_event = torch.cuda.Event()
            step_event.record(self._tts_stream)
            self._step_events.append((self._enqueue_seq, step_event))
            self._captures = []
        if step_exception is None:
            # Success here means that the enqueue succeeded. Device-side
            # failures of the async work surface at a later fence.
            self._step_failure_streak = 0
        else:
            self._handle_step_exception(step_exception)

    def _run_chunk(
        self,
        chunk: List[
            Tuple[
                _ReplayCapture,
                torch.Tensor,
                List[Tuple[torch.Tensor, torch.Tensor]],
            ]
        ],
        lm_head_weight: torch.Tensor,
    ) -> None:
        """Run one replay chunk (at most _replay_chunk captures)."""
        lm_head_weight = lm_head_weight.detach()
        if len(chunk) == 1:
            self._step_one(
                chunk[0][0], chunk[0][1], chunk[0][2], lm_head_weight
            )
        else:
            self._step_many(chunk, lm_head_weight)

    def _handle_step_exception(self, e: Exception) -> None:
        if self.tp_size > 1:
            # `raise e` rather than a bare `raise`: on the async path this
            # runs after the except block has exited, where a bare raise has
            # no active exception.
            raise e
        if not self._warned_step_failed:
            logger.warning(
                "DFlash LoRA-TTS step failed (skipping; warned once): %s", e
            )
            self._warned_step_failed = True
        self.note_round_failure(f"step: {e}")
