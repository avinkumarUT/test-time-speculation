"""Full-weight test-time speculation (TTS) for the DFlash speculator.

In speculative decoding, the target model's verification pass yields the
target's next-token distribution at every drafted position. TTS uses those
distributions to update the DFlash draft model (the "speculator") online,
while the response is being generated, so that later blocks are accepted more
often.

This module implements the full-weight mode: every draft weight that the block
forward uses is updated, and the adapted weights belong to the one request
that is being generated. It serves a single in-flight request.
`dflash_lora_tts.py` implements the per-request LoRA mode for batched serving.
DFLASH_TTS_DESIGN.md in this directory describes the design.

Main pieces
-----------
`DFlashTTSConfig`
    Options, read from the `--speculative-dflash-tts-*` server arguments.
`DFlashTTSState`
    Per-worker state: the update stride, the loss, the optimizer and the
    state selected by `optimizer_dtype`, the asynchronous step, and the weight
    reset at the start of each request.
`DFlashBlockForward`
    The draft block forward written with autograd-capable operations. It is
    the forward of every round in which an update fires; all other rounds
    run the serving forward and its CUDA graphs.
`resolve_lm_head_rows`
    Startup check of the target's vocabulary-parallel head, which the loss
    reads.

Objective
---------
The draft block is non-causal: hidden state k of a block predicts the token at
block position k. Position 0 holds the bonus token that the target has already
emitted, so positions 1..B-1 are the drafted ones. The target verifies the
block causally: its logits at block row k are the distribution of position
k + 1. For each drafted position, with draft distribution q and target
distribution p:

    lk:  loss = -log(sum_x min(p(x), q(x)))
    kl:  loss = lambda * KL(p || q) + (1 - lambda) * CE(q, argmax q)

The positions are averaged with weights exp(-k / gamma), k = 0..B-2, so that
early positions, which gate the acceptance of everything behind them, count
most. The self cross-entropy term keeps the draft's own predictions sharp.

Schedule
--------
An update fires on the rounds whose index is a multiple of `freq` (the update
stride), counted from the request's first decode round, and only while a
single request is in the batch. On a firing round the block forward runs under
autograd and its hidden states are kept until the target has verified the
block; the update is then one Adam step on the loss above. With `async_step`,
the optimizer step, including the store of its results into the draft's
parameters, runs on a side CUDA stream, and the host waits on a CUDA event
before the next draft forward, the first point at which the updated weights
are read. At every prefill the weights are restored from a snapshot
taken at startup and the optimizer state is cleared, so every request starts
from the pretrained draft. Draft K/V already cached for generated tokens are
not rewritten; with the radix cache on, a later request that shares such a
prefix reuses them.

Precision
---------
The draft's parameters are served in the dtype they were loaded in (bfloat16
for the published drafts) and never change dtype. The serving forward and the
draft CUDA graphs read them in place. `optimizer_dtype` selects the precision
in which the optimizer state is stored. The arithmetic of a step is float32 in
both modes:

"float32" (default)
    Every parameter has a float32 master copy, and Adam steps on the masters
    with float32 moments. An update is: backward through the served
    parameters, gradients cast to float32 and handed to the masters, Adam
    step on the masters, masters copied back into the served parameters
    (which rounds them to the served dtype). An update smaller than the
    spacing of a served weight stays in the master and takes effect once the
    accumulated change crosses a rounding boundary. A draft whose parameters
    are already float32 has no masters: Adam steps on the parameters in
    place.
"bfloat16"
    The state is stored in bfloat16: the parameters are their own master and
    the moments are bfloat16 tensors. An update is: backward through the
    served parameters, gradients cast to float32, then for one tensor at a
    time the weight and the moments widened to float32, an Adam step, and the
    results rounded once when they are stored (`Bf16StateAdam`). No float32
    tensor outlives a step. A bfloat16 value has 8 significant bits, so an
    update smaller than half the spacing of the weight it is added to (0.2%
    to 0.4% of the weight's magnitude) is rounded away, and the second moment
    stops decaying for the same reason.

In both modes `DFlashBlockForward` runs only on update rounds; every other
round runs the serving forward, including its CUDA graphs.

Invariants
----------
* The target model is never written. The draft owns no embedding and no head:
  the worker feeds it the target's embeddings and scores its hidden states
  with the target's `lm_head.weight`, which enters the loss detached. The
  state refuses to start if a trainable tensor shares storage with a target
  parameter.
* The cached context K/V are constants of the update. They are read back from
  the draft KV cache; each cached token keeps the projection weights that were
  current when it was appended.
* With tp_size > 1 the forward uses differentiable collectives, gradients of
  replicated parameters are completed across ranks, and the clipping norm is
  the global one. Every rank therefore applies the update a single-GPU run
  would apply to its shard, and replicated parameters stay identical across
  ranks. All collectives of an update are issued on the main stream.
* Failures in an update raise with tp_size > 1, because a rank that skipped
  its part would leave the ranks' weights and collectives out of step. With
  tp_size == 1 they are contained: a round whose differentiable forward
  failed is drafted by the serving forward, decoding continues, and TTS
  disables itself after repeated failures.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.speculative.dflash_lora_tts import (
    _AllReduceTP,
    _CopyToTP,
    _tts_all_reduce,
    _VocabAllGather,
)
from sglang.srt.speculative.tts_loss import lk_per_position
from sglang.srt.speculative.tts_optim import Bf16StateAdam

logger = logging.getLogger(__name__)

# Accepted values of `--speculative-dflash-tts-optimizer-dtype`.
OPTIMIZER_DTYPES: Dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

# Unquantized weight dtypes. The block forward applies weights with plain
# F.linear, which would be wrong for a quantized weight.
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# Parameters that only project the target's hidden states into the draft KV
# cache (`project_target_hidden`). The block forward does not use them, so
# the loss has no gradient path to them and they are never updated.
_CONTEXT_ONLY_PREFIXES = ("fc.", "hidden_norm.")


@dataclass
class DFlashTTSConfig:
    enable: bool = False
    lr: float = 1e-5
    # "lk" -> -log(sum_x min(p_target, q_draft)), see tts_loss.py
    # "kl" -> tts_lambda * KL(target || draft) + (1 - tts_lambda) * self-CE
    tts_loss: str = "lk"
    tts_lambda: float = 0.9
    freq: int = 10  # update stride: fire when round_idx % freq == 0
    pos_weight_gamma: float = 7.0  # w_k = exp(-k / gamma)
    # Skip the update when the divergence term (KL, or the lk term) is below
    # this threshold (0 disables).
    kl_threshold: float = 0.0
    grad_clip: float = 1.0  # global gradient-norm limit (<= 0 disables)
    # Run the optimizer step on a side CUDA stream, overlapping the host-side
    # work between two rounds.
    async_step: bool = True
    # Precision in which the optimizer state is stored (the moments and, for
    # float32, the master weights); a key of OPTIMIZER_DTYPES. See the module
    # docstring.
    optimizer_dtype: str = "float32"

    @classmethod
    def from_server_args(cls, server_args) -> "DFlashTTSConfig":
        def _get(name, default):
            value = getattr(server_args, f"speculative_dflash_tts_{name}", None)
            return default if value is None else value

        return cls(
            enable=bool(_get("enable", False)),
            lr=float(_get("lr", 1e-5)),
            tts_loss=str(getattr(server_args, "speculative_tts_loss", "lk")),
            tts_lambda=float(getattr(server_args, "speculative_tts_lambda", 0.9)),
            freq=int(_get("freq", 10)),
            pos_weight_gamma=float(_get("pos_weight_gamma", 7.0)),
            kl_threshold=float(_get("kl_threshold", 0.0)),
            grad_clip=float(_get("grad_clip", 1.0)),
            async_step=bool(_get("async", True)),
            optimizer_dtype=str(_get("optimizer_dtype", "float32")),
        )


def resolve_lm_head_rows(lm_head, *, vocab_size: int, tp_size: int) -> int:
    """Check the target's vocabulary-parallel head and return its row count.

    The loss computes draft logits as `hidden @ lm_head.weight[:rows].T` on
    every rank and, with tp_size > 1, gathers the shards into the full
    vocabulary. That matches the target's verify logits only if this rank's
    first `rows` rows are exactly its contiguous share of the base
    vocabulary. Raises ValueError for a layout where they are not.
    """
    weight = getattr(lm_head, "weight", None)
    if weight is None:
        raise ValueError("DFlash TTS: the target model exposes no lm_head.weight.")
    if weight.dtype not in _FLOAT_DTYPES:
        raise ValueError(
            "DFlash TTS requires an unquantized (fp16 / bf16 / fp32) target "
            f"lm_head, got dtype {weight.dtype}."
        )
    rows = int(weight.shape[0])
    shard = getattr(lm_head, "shard_indices", None)
    if shard is not None:
        if int(shard.num_added_elements) > 0:
            raise ValueError(
                "DFlash TTS does not support a target lm_head with added vocabulary."
            )
        rows = int(shard.num_org_elements)
        if tp_size > 1 and rows != int(shard.num_org_elements_padded):
            raise ValueError(
                "DFlash TTS with tp_size > 1 requires unpadded lm_head shards: "
                f"this rank holds {rows} vocabulary rows padded to "
                f"{int(shard.num_org_elements_padded)}."
            )
    if vocab_size % tp_size != 0 or rows != vocab_size // tp_size:
        raise ValueError(
            f"DFlash TTS: lm_head holds {rows} vocabulary rows on this rank, "
            f"expected {vocab_size} / {tp_size} for the target's vocabulary."
        )
    return rows


def _tp_param_roles(draft_model: nn.Module) -> Tuple[Set[int], List[nn.Parameter]]:
    """Classify the draft's parameters by their role under tensor parallelism.

    Returns `(sharded_ids, partial_grad_params)`.

    `sharded_ids` holds the ids of the parameters of which each rank owns a
    different shard (the column- and row-parallel projections). All other
    parameters are replicated.

    `partial_grad_params` are replicated parameters that act inside a sharded
    region: `q_norm` and `k_norm` normalize the heads a rank owns, so each
    rank's backward yields only its heads' contribution to their gradient.
    """
    sharded_ids: Set[int] = set()
    partial_grad_params: List[nn.Parameter] = []
    for layer in getattr(draft_model, "layers", []):
        attn, mlp = layer.self_attn, layer.mlp
        for param in (
            attn.qkv_proj.weight,
            attn.qkv_proj.bias,
            attn.o_proj.weight,
            mlp.gate_up_proj.weight,
            mlp.down_proj.weight,
        ):
            if param is not None:
                sharded_ids.add(id(param))
        partial_grad_params.extend([attn.q_norm.weight, attn.k_norm.weight])
    return sharded_ids, partial_grad_params


class DFlashBlockForward:
    """The draft block forward, written with autograd-capable operations.

    The serving forward of the draft model cannot be differentiated: its
    attention backends, fused RMSNorm, fused SwiGLU and fused RoPE kernels
    have no backward, the model's `forward` is decorated with
    `torch.no_grad()`, the tensor-parallel collectives sever or zero the
    gradient, and draft CUDA graphs replay recorded kernels without building
    an autograd graph. None of these raises; each silently leaves parameters
    without a gradient.

    This class computes the same function as `DFlashDraftModel.forward` from
    the model's own weights, using the pure-PyTorch implementation of every
    layer (`forward_native`), `F.scaled_dot_product_attention` over
    [cached prefix K/V | the block's own K/V], and conjugate pairs of
    collectives under tensor parallelism. The prefix K/V are read from the
    draft KV cache as constants; the block's K/V stay on the autograd tape,
    so the K and V projections receive gradients as well.
    """

    def __init__(
        self,
        *,
        draft_model: nn.Module,
        token_to_kv_pool,
        block_size: int,
        tp_size: int = 1,
    ):
        self.draft_model = draft_model
        self.token_to_kv_pool = token_to_kv_pool
        self.block_size = int(block_size)
        self.tp_size = int(tp_size)
        self.layers = list(draft_model.layers)

        if self.block_size < 2:
            # With block_size 1 there is no drafted position to score.
            raise RuntimeError(
                f"DFlash TTS requires block_size >= 2, got {self.block_size}."
            )

        # The prefix K/V are gathered with the full-pool locations stored in
        # the draft's req_to_token table, which requires a plain
        # [num_tokens, kv_heads, head_dim] buffer per layer. Other pools
        # (sliding-window pools with translated indices, MLA) would return
        # plausible but wrong rows.
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        if not isinstance(token_to_kv_pool, MHATokenToKVPool):
            raise RuntimeError(
                f"DFlash TTS: the draft KV pool is {type(token_to_kv_pool).__name__}; "
                "the differentiable block forward requires MHATokenToKVPool."
            )

        # Per-layer prefix window, read from the serving RadixAttention
        # (SGLang's exclusive convention; -1 or absent means full attention).
        # A windowed layer attends to the last `window` prefix keys, masked
        # per query row exactly like the serving kernel (see `block_hidden`),
        # so the differentiable forward and the fused forward agree.
        self._layer_prefix_windows: List[Optional[int]] = []
        for layer in self.layers:
            attn = layer.self_attn
            window = int(getattr(attn.attn, "sliding_window_size", -1) or -1)
            if 0 < window < self.block_size - 1:
                # Block-to-block attention is left unmasked; a window shorter
                # than the block would need masking inside the block too.
                raise RuntimeError(
                    f"DFlash TTS requires sliding window ({window}) >= "
                    f"block_size - 1 ({self.block_size - 1})."
                )
            self._layer_prefix_windows.append(window if window > 0 else None)

            for weight in (
                attn.qkv_proj.weight,
                attn.o_proj.weight,
                layer.mlp.gate_up_proj.weight,
                layer.mlp.down_proj.weight,
            ):
                if weight.dtype not in _FLOAT_DTYPES:
                    raise RuntimeError(
                        "DFlash TTS requires unquantized fp16/bf16/fp32 draft "
                        f"weights, got dtype {weight.dtype}."
                    )
            k_buffer = token_to_kv_pool.get_key_buffer(attn.attn.layer_id)
            if k_buffer.dim() != 3 or k_buffer.shape[1] != attn.num_kv_heads:
                raise RuntimeError(
                    f"DFlash TTS: draft KV buffer has shape {tuple(k_buffer.shape)}, "
                    f"expected [num_tokens, {attn.num_kv_heads}, head_dim]."
                )
            if self.tp_size > 1 and attn.total_num_kv_heads < self.tp_size:
                # The K/V projections are then replicated across ranks rather
                # than sharded, and each replica would receive only the
                # gradient of the query heads on its own rank.
                raise RuntimeError(
                    "DFlash TTS with tp_size > 1 requires at least tp_size "
                    f"key/value heads; the draft has {attn.total_num_kv_heads} "
                    f"and tp_size is {self.tp_size}."
                )

    def gather_prefix_kv(
        self, prefix_locs: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Read one request's context K/V from the draft KV cache.

        `prefix_locs` are the request's cache locations in sequence order.
        A windowed layer gathers only the last `window` of them, which is the
        tail the serving kernel exposes to the block and bounds the size of
        the gather. Advanced indexing copies, so later writes to the cache do
        not affect the returned tensors.
        """
        prefix_kv = []
        for layer, window in zip(self.layers, self._layer_prefix_windows):
            k_buffer, v_buffer = self.token_to_kv_pool.get_kv_buffer(
                layer.self_attn.attn.layer_id
            )
            locs = prefix_locs if window is None else prefix_locs[-window:]
            prefix_kv.append((k_buffer[locs].detach(), v_buffer[locs].detach()))
        return prefix_kv

    def block_hidden(
        self,
        input_embeds: torch.Tensor,  # [B, hidden]
        positions: torch.Tensor,  # [B] int64, absolute positions (RoPE)
        prefix_kv: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Final hidden states [B, hidden] of one request's draft block.

        Follows `DFlashDraftModel.forward` / `DFlashDecoderLayer.forward`
        layer by layer and returns the result in the parameter dtype.
        """
        model = self.draft_model
        # Megatron-style conjugate collectives. "f" (`_CopyToTP`) is the
        # identity on the replicated input of a column-parallel projection and
        # all-reduces its gradient; without it every gradient upstream of the
        # projection would contain only this rank's shard. "g" (`_AllReduceTP`)
        # sums the partial outputs of a row-parallel projection and passes the
        # gradient through unchanged.
        tp = self.tp_size > 1
        hidden = input_embeds
        residual: Optional[torch.Tensor] = None
        n = hidden.shape[0]

        for layer, (k_prefix, v_prefix), window in zip(
            self.layers, prefix_kv, self._layer_prefix_windows
        ):
            attn = layer.self_attn
            head_dim = attn.head_dim
            if residual is None:
                residual = hidden
                hidden = layer.input_layernorm.forward_native(hidden)
            else:
                hidden, residual = layer.input_layernorm.forward_native(
                    hidden, residual
                )

            # Attention, non-causal over [prefix | block].
            if tp and hidden.requires_grad:
                hidden = _CopyToTP.apply(hidden)
            qkv = F.linear(hidden, attn.qkv_proj.weight, attn.qkv_proj.bias)
            q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
            q = attn.q_norm.forward_native(q.reshape(-1, head_dim)).view(n, attn.q_size)
            k = attn.k_norm.forward_native(k.reshape(-1, head_dim)).view(
                n, attn.kv_size
            )
            q, k = attn.rotary_emb.forward_native(positions, q, k)

            q_heads = q.view(n, attn.num_heads, head_dim)
            k_heads = torch.cat(
                [k_prefix.to(k.dtype), k.view(n, attn.num_kv_heads, head_dim)]
            )
            v_heads = torch.cat(
                [v_prefix.to(v.dtype), v.view(n, attn.num_kv_heads, head_dim)]
            )

            # Windowed layer: with T gathered tail keys, block row m (absolute
            # position prefix_len + m) sees tail key j (absolute position
            # prefix_len - T + j) only if their distance T + m - j is at most
            # `window`. Row m therefore masks its first T + m - window tail
            # keys whenever that number is positive. Block keys are always
            # visible.
            num_tail = k_prefix.shape[0]
            attn_mask = None
            if window is not None and num_tail + (n - 1) > window:
                q_rows = torch.arange(n, device=hidden.device).unsqueeze(1)
                tail_idx = torch.arange(num_tail, device=hidden.device).unsqueeze(0)
                attn_mask = torch.cat(
                    [
                        tail_idx >= (num_tail + q_rows - window),
                        torch.ones(n, n, dtype=torch.bool, device=hidden.device),
                    ],
                    dim=1,
                )  # [n, num_tail + n], broadcast over heads

            # SDPA layout: [1, heads, seq, head_dim]. Without a mask,
            # enable_gqa broadcasts the KV heads; with a mask the repeat is
            # materialized, since the combination is not supported by every
            # torch version.
            num_rep = attn.num_heads // attn.num_kv_heads
            if attn_mask is not None and num_rep > 1:
                k_heads = k_heads.repeat_interleave(num_rep, dim=1)
                v_heads = v_heads.repeat_interleave(num_rep, dim=1)
                enable_gqa = False
            else:
                enable_gqa = num_rep > 1
            attn_out = F.scaled_dot_product_attention(
                q_heads.transpose(0, 1).unsqueeze(0),
                k_heads.transpose(0, 1).unsqueeze(0),
                v_heads.transpose(0, 1).unsqueeze(0),
                attn_mask=attn_mask,
                scale=attn.scaling,
                is_causal=False,
                enable_gqa=enable_gqa,
            )
            attn_out = attn_out.squeeze(0).transpose(0, 1).reshape(n, attn.q_size)

            # Row-parallel o_proj: the matmul over this rank's heads is a
            # partial sum. The bias is added once, after the reduce.
            attn_proj = F.linear(attn_out, attn.o_proj.weight)
            if tp:
                attn_proj = _AllReduceTP.apply(attn_proj)
            if attn.o_proj.bias is not None:
                attn_proj = attn_proj + attn.o_proj.bias

            hidden, residual = layer.post_attention_layernorm.forward_native(
                attn_proj, residual
            )

            # MLP (SwiGLU): gate_up is column-parallel, down is row-parallel.
            mlp = layer.mlp
            if tp and hidden.requires_grad:
                hidden = _CopyToTP.apply(hidden)
            gate_up = F.linear(hidden, mlp.gate_up_proj.weight)
            hidden = F.linear(mlp.act_fn.forward_native(gate_up), mlp.down_proj.weight)
            if tp:
                hidden = _AllReduceTP.apply(hidden)

        if residual is None:
            return model.norm.forward_native(hidden)
        hidden, _ = model.norm.forward_native(hidden, residual)
        return hidden

    def __call__(
        self,
        *,
        input_embeds: torch.Tensor,  # [bs, B, hidden]
        positions: torch.Tensor,  # [bs, B] int64
        prefix_lens: Sequence[int],  # [bs] draft-cache prefix lengths (host)
        req_pool_indices: Sequence[int],  # [bs] request slots (host)
        req_to_token: torch.Tensor,  # the draft's request-to-token table
    ) -> torch.Tensor:
        """Run the block forward for every request in the batch.

        Returns [bs, B, hidden] in the parameter dtype, carrying the autograd
        graph of the forward.

        Every request is run on its own, because each attends to a different
        prefix. The forward leaves inference mode explicitly: tensors created
        under inference mode cannot be saved for backward, and an enclosing
        inference mode would otherwise turn every result into one.
        """
        with torch.inference_mode(False), torch.enable_grad():
            hidden_rows = []
            for row in range(input_embeds.shape[0]):
                with torch.no_grad():
                    prefix_locs = req_to_token[
                        int(req_pool_indices[row]), : int(prefix_lens[row])
                    ].to(torch.int64)
                    prefix_kv = self.gather_prefix_kv(prefix_locs)
                embeds = input_embeds[row]
                if embeds.is_inference():
                    embeds = embeds.clone()
                # The positions are a view of a buffer that the worker
                # rewrites in place every round; the tape must hold its own
                # copy.
                hidden_rows.append(
                    self.block_hidden(embeds, positions[row].clone(), prefix_kv)
                )
            return torch.stack(hidden_rows)


class DFlashTTSState:
    """Full-weight TTS state of one DFlash draft model on one worker.

    Owns the snapshot for the per-request reset, the optimizer and the state
    selected by `optimizer_dtype`, the update stride (`should_fire` /
    `advance`), the loss and the update (`step`), and the asynchronous
    hand-off (`wait_for_pending`).

    The state is for a single in-flight request: `reset()` restores the
    pretrained weights at every prefill.
    """

    # Consecutive failed updates after which TTS disables itself (tp_size 1).
    _MAX_FAILURE_STREAK = 3

    def __init__(
        self,
        *,
        draft_model: nn.Module,
        config: DFlashTTSConfig,
        device: torch.device | str,
        target_model: Optional[nn.Module] = None,
        tp_size: int = 1,
        tp_rank: int = 0,
    ):
        if config.optimizer_dtype not in OPTIMIZER_DTYPES:
            raise ValueError(
                f"DFlash TTS: unknown optimizer dtype {config.optimizer_dtype!r}; "
                f"expected one of {sorted(OPTIMIZER_DTYPES)}."
            )
        self.cfg = config
        self.draft_model = draft_model
        self.device = torch.device(device)
        self.tp_size = int(tp_size)
        self.tp_rank = int(tp_rank)
        self.optimizer_dtype = OPTIMIZER_DTYPES[config.optimizer_dtype]

        named = list(draft_model.named_parameters())
        if not named:
            raise ValueError("DFlash TTS: the draft model has no parameters.")
        self._param_names = [name for name, _ in named]
        self._params = [param for _, param in named]

        self._check_trainable(named, target_model)

        # SGLang loads weights with requires_grad=False.
        for param in self._params:
            param.requires_grad_(True)

        # Snapshot of the pretrained weights for the per-request reset, in the
        # parameters' own dtype and on the device: a restore happens once per
        # request, and a host round trip would stall the prefill.
        self._snapshot = [param.detach().clone() for param in self._params]

        # float32 optimizer dtype: Adam steps on a float32 master copy of
        # every parameter, and each update copies the masters back into the
        # parameters, which keep their dtype and their storage (the serving
        # forward and the draft CUDA graphs read them in place). The cast to
        # float32 is exact, so a master starts at the value of its parameter.
        # A draft that is float32 throughout needs no masters.
        self._masters: Optional[List[torch.Tensor]] = None
        if self.optimizer_dtype == torch.float32 and any(
            param.dtype != torch.float32 for param in self._params
        ):
            self._masters = [
                param.detach().to(torch.float32, copy=True) for param in self._params
            ]
        # bfloat16 optimizer dtype: the parameters are their own master. A
        # float32 gradient can only be assigned to a float32 tensor, so each
        # parameter has a holder: a scalar expanded to the parameter's shape,
        # which takes no memory of that size.
        self._grad_holders: Optional[List[torch.Tensor]] = None
        if self.optimizer_dtype == torch.bfloat16:
            self._grad_holders = [
                param.new_zeros((), dtype=torch.float32).expand(param.shape)
                for param in self._params
            ]
        # The tensors that carry the gradients of an update from the end of
        # backward to the step, and that Adam steps on unless they are holders.
        self._step_params: List[torch.Tensor] = (
            self._grad_holders or self._masters or self._params
        )

        # Tensor-parallel roles, expressed for the tensors holding the gradients.
        self._tp_sharded_ids: Set[int] = set()
        self._tp_partial_grad_params: List[torch.Tensor] = []
        if self.tp_size > 1:
            sharded_ids, partial_grad_params = _tp_param_roles(draft_model)
            step_param_of = {
                id(param): step_param
                for param, step_param in zip(self._params, self._step_params)
            }
            self._tp_sharded_ids = {id(step_param_of[i]) for i in sharded_ids}
            self._tp_partial_grad_params = [
                step_param_of[id(param)] for param in partial_grad_params
            ]

        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._build_optimizer()

        self._use_async = (
            bool(config.async_step)
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        )
        self._tts_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(device=self.device) if self._use_async else None
        )
        self._step_event: Optional[torch.cuda.Event] = None
        # Device-side flag of the last update: whether its loss and gradient
        # norm were finite. Read on the host at the next fence.
        self._pending_finite: Optional[torch.Tensor] = None

        # Index of the current decode round of the request (drives the
        # stride) and whether the weights or the masters may differ from the
        # snapshot.
        self.round_idx: int = 0
        self._dirty: bool = False
        # Per-request counters, reported by log_request_summary().
        self._request_update_ct: int = 0
        self._request_kl_skip_ct: int = 0
        # Lifetime counters.
        self.update_ct: int = 0
        self.kl_skip_ct: int = 0
        self.nonfinite_ct: int = 0
        self._failure_streak: int = 0
        self._warned_step_failed: bool = False
        self._checked_grad_coverage: bool = False
        self._disabled: bool = False

        if self.tp_rank == 0:
            logger.info(
                "DFlash TTS enabled (full-weight). loss=%s optimizer_dtype=%s "
                "fp32_master=%s lr=%.2e "
                "lambda=%.3f freq=%d gamma=%.2f kl_threshold=%.3f grad_clip=%.2f "
                "async=%s tp_size=%d trainable_params=%d",
                config.tts_loss,
                config.optimizer_dtype,
                self._masters is not None,
                config.lr,
                config.tts_lambda,
                config.freq,
                config.pos_weight_gamma,
                config.kl_threshold,
                config.grad_clip,
                self._use_async,
                self.tp_size,
                sum(param.numel() for param in self._params),
            )

    # ------------------------------------------------------------------ setup

    def _check_trainable(self, named, target_model: Optional[nn.Module]) -> None:
        """Raise unless every draft parameter can be trained in place."""
        inference = [name for name, param in named if param.is_inference()]
        if inference:
            raise RuntimeError(
                "DFlash TTS: these draft parameters are inference tensors and "
                f"cannot be trained: {inference}."
            )
        quantized = sorted(
            {str(param.dtype) for _, param in named if param.dtype not in _FLOAT_DTYPES}
        )
        if quantized:
            raise RuntimeError(
                "DFlash TTS requires unquantized fp16/bf16/fp32 draft weights, "
                f"got {quantized}."
            )
        if self.optimizer_dtype == torch.bfloat16:
            other = sorted(
                {
                    str(param.dtype)
                    for _, param in named
                    if param.dtype != torch.bfloat16
                }
            )
            if other:
                raise ValueError(
                    "--speculative-dflash-tts-optimizer-dtype bfloat16 keeps the "
                    "optimizer state in the dtype of the draft weights and is "
                    "only supported for a bfloat16 draft model; this draft has "
                    f"{other} parameters. Use float32."
                )
        if target_model is not None:
            # An optimizer step on a tensor shared with the target would
            # change the target's output and make decoding lossy.
            target_ptrs = {param.data_ptr() for param in target_model.parameters()}
            shared = [name for name, param in named if param.data_ptr() in target_ptrs]
            if shared:
                raise RuntimeError(
                    "DFlash TTS: cannot start: these draft parameters "
                    "share storage with the target model and an optimizer "
                    f"step would modify it: {shared}."
                )

    def _build_optimizer(self) -> None:
        # Adam over all draft parameters, or over their float32 masters.
        # Parameters the loss cannot reach (see _CONTEXT_ONLY_PREFIXES) never
        # receive a gradient and are skipped by the optimizer.
        if self._grad_holders is not None:
            self._optimizer = Bf16StateAdam(self._params, lr=self.cfg.lr)
        else:
            self._optimizer = torch.optim.Adam(self._step_params, lr=self.cfg.lr)

    # ---------------------------------------------------------- stride / fence

    def should_fire(self) -> bool:
        if self._disabled:
            return False
        return (self.round_idx % max(1, self.cfg.freq)) == 0

    def advance(self) -> None:
        self.round_idx += 1

    def is_disabled(self) -> bool:
        return self._disabled

    def wait_for_pending(self) -> None:
        """Block until the pending optimizer step has completed.

        Called immediately before the draft weights are read again. This is a
        host-side `event.synchronize()` rather than a `wait_event` on the main
        stream: the next draft forward may be replayed from a CUDA graph on
        PyTorch's internal stream, and a wait recorded on the host's current
        stream does not propagate into a graph replay. The replay would then
        read partially written weights without any error.

        Also checks the completed update: if its loss or gradient norm was
        not finite, the weights are restored from the snapshot.
        """
        self._drain()
        self._check_last_update()

    def _drain(self) -> None:
        if self._step_event is not None:
            self._step_event.synchronize()
            self._step_event = None
            self._free_grads()

    def _free_grads(self) -> None:
        """Release the gradients of a completed step.

        They are as large as the tensors they belong to and are not needed
        between updates.
        """
        for param in self._params:
            param.grad = None
        if self._step_params is not self._params:
            for step_param in self._step_params:
                step_param.grad = None

    def _check_last_update(self) -> None:
        finite = self._pending_finite
        if finite is None:
            return
        self._pending_finite = None
        if bool(finite):
            return
        # A non-finite update leaves non-finite weights and Adam moments that
        # would persist until the next prefill. Restoring the snapshot costs
        # the request's adaptation so far and keeps drafting meaningful. The
        # flag derives from the loss and the global gradient norm, which are
        # identical on every rank, so all ranks restore together.
        self.nonfinite_ct += 1
        if self.nonfinite_ct <= 3 and self.tp_rank == 0:
            logger.warning(
                "DFlash TTS: an update at round %d had a non-finite loss or "
                "gradient norm; restoring the pretrained draft weights "
                "(%d so far).",
                self.round_idx,
                self.nonfinite_ct,
            )
        self._restore()

    def _restore(self) -> None:
        """Restore the pretrained weights and clear the optimizer state.

        The masters are set to the float32 value of the snapshot, which is
        what they hold at startup. Rebuilding the optimizer releases its
        moments.
        """
        with torch.no_grad():
            for param, snapshot in zip(self._params, self._snapshot):
                param.copy_(snapshot)
            if self._masters is not None:
                for master, snapshot in zip(self._masters, self._snapshot):
                    master.copy_(snapshot)
        self._build_optimizer()
        self._dirty = False

    def reset(self) -> None:
        """Start a new request: pretrained weights, fresh optimizer, round 0."""
        self._drain()
        self._pending_finite = None
        self.log_request_summary()
        if self._dirty:
            self._restore()
        self.round_idx = 0
        # Failures are counted per request: a transient failure in one request
        # does not add up with failures of later requests.
        self._failure_streak = 0

    def disable(self, reason: str) -> None:
        """Stop updating and return to the pretrained weights."""
        if self._disabled:
            return
        self._disabled = True
        self._drain()
        self._pending_finite = None
        if self._dirty:
            self._restore()
        # No further update runs: release the gradients of the last backward
        # and the optimizer state.
        self._free_grads()
        self._masters = None
        self._grad_holders = None
        self._step_params = self._params
        self._optimizer = None
        logger.error(
            "DFlash TTS disabled: %s. Drafting continues with the pretrained "
            "draft weights.",
            reason,
        )

    def note_step_failure(self, error: Exception) -> None:
        """Record a failed update (tp_size 1); repeated failures disable TTS."""
        self._failure_streak += 1
        if not self._warned_step_failed:
            self._warned_step_failed = True
            logger.warning(
                "DFlash TTS update failed (skipping; warned once): %s", error
            )
        if self._failure_streak >= self._MAX_FAILURE_STREAK:
            self.disable(
                f"{self._failure_streak} consecutive failed updates (last: {error})"
            )

    def log_request_summary(self) -> None:
        """Log what TTS did for the request that is ending (rank 0, once).

        Reads the largest weight change on the host, so it must only be
        called where a host synchronization is acceptable: when a request
        finishes and at a prefill.
        """
        update_ct, kl_skip_ct = self._request_update_ct, self._request_kl_skip_ct
        self._request_update_ct = self._request_kl_skip_ct = 0
        if self.tp_rank != 0 or (update_ct == 0 and kl_skip_ct == 0):
            return
        self._drain()
        with torch.no_grad():
            max_delta = float(
                torch.stack(
                    [
                        (param - snapshot).abs().max().float()
                        for param, snapshot in zip(self._params, self._snapshot)
                    ]
                ).max()
            )
        logger.info(
            "DFlash TTS request summary: rounds=%d updates=%d kl_skipped=%d "
            "max_abs_weight_change=%.3e",
            self.round_idx,
            update_ct,
            kl_skip_ct,
            max_delta,
        )
        if update_ct > 0 and max_delta == 0.0 and not self._disabled:
            if self._masters is not None:
                logger.warning(
                    "DFlash TTS: %d updates changed no draft weight. Their "
                    "sum, held in the float32 masters, is still below the "
                    "spacing of every weight in the draft's dtype; use a "
                    "larger learning rate.",
                    update_ct,
                )
            else:
                logger.warning(
                    "DFlash TTS: %d updates changed no draft weight. With "
                    "optimizer dtype bfloat16 this means every update was below "
                    "the bfloat16 spacing of the weights; use a larger learning "
                    "rate or optimizer dtype float32.",
                    update_ct,
                )

    # ------------------------------------------------------------------- loss

    def compute_loss(
        self,
        *,
        draft_hidden: torch.Tensor,  # [bs, B, hidden], with grad
        target_logits: torch.Tensor,  # [bs * B, vocab] target verify logits
        lm_head_weight: torch.Tensor,  # [vocab / tp_size, hidden], this rank's rows
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """The position-weighted terms of the loss.

        Returns `(divergence_term, ce_term)`. With the "lk" loss the first is
        the loss and the second is None. With "kl" they are the KL and the self
        cross-entropy, and the loss is
        `lambda * divergence_term + (1 - lambda) * ce_term`.
        """
        bs, block_size, hidden_size = draft_hidden.shape
        num_positions = block_size - 1
        if num_positions <= 0:
            raise ValueError("DFlash TTS: a block of size 1 has no drafted position.")

        # Draft logits for block positions 1..B-1, computed with the target's
        # head in the head's dtype, as the worker does when it samples the
        # draft tokens. The head is a constant of the loss.
        head_input = (
            draft_hidden[:, 1:block_size, :]
            .to(lm_head_weight.dtype)
            .reshape(-1, hidden_size)
        )
        if self.tp_size > 1:
            # The head is vocabulary-sharded: compute this rank's shard of
            # the logits and gather the shards into the full vocabulary. "f"
            # on the replicated head input all-reduces its gradient, which
            # each rank otherwise receives only from its own shard.
            head_input = _CopyToTP.apply(head_input)
        draft_logits = torch.matmul(head_input, lm_head_weight.detach().T)
        if self.tp_size > 1:
            draft_logits = _VocabAllGather.apply(draft_logits)
        vocab = draft_logits.shape[-1]
        if target_logits.numel() != bs * block_size * vocab:
            raise RuntimeError(
                f"DFlash TTS: draft logits have vocabulary {vocab} but the "
                f"target logits have shape {tuple(target_logits.shape)} for "
                f"bs={bs}, block_size={block_size}."
            )
        draft_logits = draft_logits.view(bs, num_positions, vocab)

        # The target verifies causally: row k predicts block position k + 1,
        # so rows 0..B-2 pair with the draft's positions 1..B-1.
        target_logits = target_logits.view(bs, block_size, vocab)[
            :, :num_positions, :
        ].detach()

        draft_logits_f = draft_logits.float()
        log_p_draft = F.log_softmax(draft_logits_f, dim=-1)
        log_p_target = F.log_softmax(target_logits.float(), dim=-1)
        p_target = log_p_target.exp()
        pos_weights = self._position_weights(num_positions, draft_logits.device)
        weight_sum = pos_weights.sum().clamp_min(1e-6)

        if self.cfg.tts_loss == "lk":
            per_pos_lk = lk_per_position(p_target, log_p_draft)
            return (per_pos_lk * pos_weights).sum() / (bs * weight_sum), None

        # KL(target || draft) per position, in fp32.
        # A token the target gives zero probability (a -inf logit) contributes
        # nothing; without the mask its term would be 0 * inf.
        per_pos_kl = (
            (p_target * (log_p_target - log_p_draft))
            .masked_fill(p_target == 0, 0.0)
            .sum(dim=-1)
        )

        # Self cross-entropy against the draft's own argmax.
        with torch.no_grad():
            self_targets = draft_logits.argmax(dim=-1)
        per_pos_ce = F.cross_entropy(
            draft_logits_f.reshape(-1, vocab),
            self_targets.reshape(-1),
            reduction="none",
        ).reshape(bs, num_positions)

        kl_term = (per_pos_kl * pos_weights).sum() / (bs * weight_sum)
        ce_term = (per_pos_ce * pos_weights).sum() / (bs * weight_sum)
        return kl_term, ce_term

    def _position_weights(self, num_positions: int, device) -> torch.Tensor:
        """w_k = exp(-k / gamma) for the k-th drafted position, k = 0..B-2."""
        gamma = max(self.cfg.pos_weight_gamma, 1e-6)
        ks = torch.arange(num_positions, device=device, dtype=torch.float32)
        return torch.exp(-ks / gamma)

    # ----------------------------------------------------------------- update

    def step(
        self,
        *,
        draft_hidden: torch.Tensor,
        target_logits: torch.Tensor,
        lm_head_weight: torch.Tensor,
    ) -> Optional[float]:
        """One TTS update from a verified block.

        Args:
            draft_hidden: [bs, B, hidden] hidden states of the draft block,
                carrying the autograd graph of the block forward.
            target_logits: [bs * B, vocab] the target's verify logits for the
                same block, taken before verification modifies them.
            lm_head_weight: this rank's base-vocabulary rows of the target's
                `lm_head.weight`.

        Returns the divergence term when `kl_threshold` is enabled (the update is
        skipped if it is below the threshold), else None.
        """
        if self._disabled:
            return None
        if draft_hidden.grad_fn is None:
            # Something on the draft forward was not differentiable, or the
            # forward ran with autograd disabled. Stepping is impossible, and
            # skipping silently would leave TTS inactive without notice.
            raise RuntimeError(
                "DFlash TTS: the draft hidden states carry no autograd graph."
            )

        # The scheduler loop runs with autograd disabled (and optionally in
        # inference mode); the loss, backward and the optimizer step need
        # both restored.
        with torch.inference_mode(False), torch.enable_grad():
            divergence_term, ce_term = self.compute_loss(
                draft_hidden=draft_hidden,
                target_logits=target_logits,
                lm_head_weight=lm_head_weight,
            )

            kl_value: Optional[float] = None
            if self.cfg.kl_threshold > 0:
                kl = divergence_term.detach()
                if self.tp_size > 1:
                    # Every rank must take the same branch below. The ranks
                    # compute the same value, and averaging it through an
                    # all-reduce makes the value they compare bit-identical.
                    kl = _tts_all_reduce(kl.reshape(1).clone()) / self.tp_size
                # .item() synchronizes with the device; this cost is only
                # paid when the gate is enabled.
                kl_value = float(kl.item())
                if kl_value < self.cfg.kl_threshold:
                    self.kl_skip_ct += 1
                    self._request_kl_skip_ct += 1
                    return kl_value

            if ce_term is None:
                loss = divergence_term
            else:
                lam = self.cfg.tts_lambda
                loss = lam * divergence_term + (1.0 - lam) * ce_term
            # Set before the step: if the optimizer raises part-way, some
            # weights or masters may already be written and the next reset
            # must restore them.
            self._dirty = True
            self._backward_and_step(loss)

        self.update_ct += 1
        self._request_update_ct += 1
        self._failure_streak = 0
        return kl_value

    def _backward_and_step(self, loss: torch.Tensor) -> None:
        """Backward, gradient clipping and one Adam step.

        Backward runs through the draft's parameters. Unless the draft is
        stepped in place, the gradients are then cast to float32 and handed to
        the masters or the holders; everything after that (completion across
        ranks, the norm, clipping) works on those. The step ends with the
        results in the parameters: the masters are copied back, and with
        bfloat16 state each result is stored as it is computed.

        Backward runs its kernels on the stream of the forward (the main
        stream) whatever stream is current, and with tp_size > 1 it contains
        collectives; the cross-rank gradient completion is a collective too.
        All of these are issued on the main stream. Only the rank-local
        remainder, scaling the gradients, the Adam step and the store of its
        results, moves to the side stream in asynchronous mode, after that
        stream has been made to wait for the main stream.
        """
        self._free_grads()
        loss.backward()
        self._check_grad_coverage()
        if self._step_params is not self._params:
            self._move_grads_to_step_params()

        # The global gradient norm is computed whether or not clipping is
        # enabled: it is also what detects a non-finite gradient.
        clip = self.cfg.grad_clip > 0
        finite = torch.isfinite(loss.detach())
        clip_coef: Optional[torch.Tensor] = None
        if self.tp_size > 1:
            total_norm = self._complete_tp_grads()
            finite = finite & torch.isfinite(total_norm)
            if clip:
                clip_coef = (self.cfg.grad_clip / (total_norm + 1e-6)).clamp(max=1.0)

        step_ctx = contextlib.nullcontext()
        if self._use_async:
            self._tts_stream.wait_stream(torch.cuda.current_stream(self.device))
            # These tensors are allocated on the main stream and consumed on
            # the side stream. record_stream keeps the allocator from reusing
            # their memory while the side stream still reads it.
            for step_param in self._step_params:
                if step_param.grad is not None:
                    step_param.grad.record_stream(self._tts_stream)
            if clip_coef is not None:
                clip_coef.record_stream(self._tts_stream)
            finite.record_stream(self._tts_stream)
            step_ctx = torch.cuda.stream(self._tts_stream)

        try:
            with step_ctx:
                if self.tp_size > 1:
                    if clip_coef is not None:
                        with torch.no_grad():
                            for step_param in self._step_params:
                                if step_param.grad is not None:
                                    step_param.grad.mul_(clip_coef)
                else:
                    # An infinite bound leaves the gradients unscaled and
                    # still returns their norm.
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        self._step_params,
                        self.cfg.grad_clip if clip else float("inf"),
                    )
                    finite = finite & torch.isfinite(total_norm)
                if self._grad_holders is not None:
                    self._step_bf16_state()
                else:
                    self._optimizer.step()
                    if self._masters is not None:
                        self._write_back_masters()
        finally:
            if self._use_async:
                # Recorded after the results are stored, and even if the step
                # raised part-way: kernels that were already enqueued still
                # write the weights, and the next fence must wait for them.
                if self._step_event is None:
                    self._step_event = torch.cuda.Event()
                self._step_event.record(self._tts_stream)
            else:
                self._free_grads()
        self._pending_finite = finite

    def _move_grads_to_step_params(self) -> None:
        """Hand each parameter's gradient to its master or holder, in float32.

        The gradient in the parameter's dtype is released as soon as it has
        been cast, so the two copies of a gradient coexist for one tensor at
        a time. Runs on the stream of backward.
        """
        try:
            for param, step_param in zip(self._params, self._step_params):
                if param.grad is not None:
                    step_param.grad = param.grad.to(torch.float32)
                    param.grad = None
        except BaseException:
            # A partial hand-over would hold gradients on both sides until
            # the next update.
            self._free_grads()
            raise

    def _step_bf16_state(self) -> None:
        """Adam on the widened state, one tensor at a time, rounded when stored.

        A tensor is widened where the step runs, so in asynchronous mode the
        float32 weight and moments are allocated and consumed on the side
        stream alone, and the allocator reuses their memory only there.
        """
        for i, holder in enumerate(self._grad_holders):
            if holder.grad is not None:
                self._optimizer.step_param(i, holder.grad)

    def _write_back_masters(self) -> None:
        """Copy the stepped masters into the draft's parameters, in place.

        The copy rounds to the parameter's dtype and keeps its storage, which
        the serving forward and the draft CUDA graphs read. A master without
        a gradient was not stepped and still equals what its parameter holds.
        """
        with torch.no_grad():
            for param, master in zip(self._params, self._masters):
                if master.grad is not None:
                    param.copy_(master)

    def _complete_tp_grads(self) -> torch.Tensor:
        """Make the gradients exact under tensor parallelism (tp_size > 1).

        Works on the tensors that hold the gradients (the parameters, their
        float32 masters or the holders). Completes the gradients of which each
        rank holds a partial sum (see `_tp_param_roles`) and returns the global
        gradient norm:
        sharded parameters contribute the sum of all ranks' shards,
        replicated parameters (identical on every rank) are counted once.
        Both travel in a single all-reduce.
        """
        partial = [p for p in self._tp_partial_grad_params if p.grad is not None]
        pieces = [param.grad.detach().reshape(-1).float() for param in partial]
        sharded_sq = torch.zeros((), dtype=torch.float32, device=self.device)
        for param in self._step_params:
            if param.grad is not None and id(param) in self._tp_sharded_ids:
                sharded_sq += torch.linalg.vector_norm(
                    param.grad.detach().float()
                ).square()
        pieces.append(sharded_sq.reshape(1))

        packed = _tts_all_reduce(torch.cat(pieces))
        offset = 0
        with torch.no_grad():
            for param in partial:
                num = param.grad.numel()
                param.grad.copy_(packed[offset : offset + num].view_as(param.grad))
                offset += num

        replicated_sq = torch.zeros((), dtype=torch.float32, device=self.device)
        for param in self._step_params:
            if param.grad is not None and id(param) not in self._tp_sharded_ids:
                replicated_sq += torch.linalg.vector_norm(
                    param.grad.detach().float()
                ).square()
        return (packed[offset] + replicated_sq).sqrt()

    def _check_grad_coverage(self) -> None:
        """Verify, once, that backward reached every weight of the block forward.

        An operation without a backward ends the autograd tape without
        raising: the step still runs, but the parameters upstream of the
        break never change. Whether a parameter received a gradient at all
        is known on the host, so this check costs nothing.
        """
        if self._checked_grad_coverage:
            return
        self._checked_grad_coverage = True
        unreached = [
            name
            for name, param in zip(self._param_names, self._params)
            if param.grad is None
        ]
        unexpected = [
            name for name in unreached if not name.startswith(_CONTEXT_ONLY_PREFIXES)
        ]
        if unexpected:
            reason = (
                "backward did not reach these draft parameters, so the draft "
                f"forward is not differentiable up to them: {unexpected}"
            )
            self.disable(reason)
            raise RuntimeError(f"DFlash TTS: {reason}.")
        if self.tp_rank == 0:
            logger.info(
                "DFlash TTS: the first backward reached %d of %d draft parameter "
                "tensors; not reached (they only project the target's hidden "
                "states into the draft KV cache): %s",
                len(self._params) - len(unreached),
                len(self._params),
                unreached,
            )
