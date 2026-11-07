# Adapted from the DFlash reference implementation (HF) but implemented with
# SGLang primitives (RadixAttention + SGLang KV cache). This model intentionally
# does not include token embeddings or an LM head; DFlash uses the target model's
# embedding/lm_head.

from __future__ import annotations

import logging
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.dflash_utils import (
    can_dflash_slice_qkv_weight,
    parse_dflash_draft_config,
)

logger = logging.getLogger(__name__)

# --- DFlash draft sliding-window attention -----------------------------------
#
# DFlash draft models (e.g. z-lab/Qwen3.6-27B-DFlash) are trained with
# sliding-window attention on the layers whose `layer_types` entry is
# "sliding_attention" (window `config.sliding_window`). With TTS enabled
# (per-request LoRA or full-weight), these layers are served with that window.
# The TTS forward that computes an update reads the resulting per-layer
# RadixAttention.sliding_window_size, so serving and update use the same
# window, and the prefix that an update reads is bounded by the window instead
# of growing with the response.
def draft_window_enabled(server_args=None) -> bool:
    """Whether windowed draft layers are served with their trained window."""
    if server_args is None:
        try:
            server_args = get_global_server_args()
        except ValueError:
            # No server is being set up (e.g. a model built in a unit test).
            return False
    if getattr(server_args, "speculative_dflash_tts_enable", False):
        # Full-weight TTS (speculative/dflash_tts.py) serves and trains the
        # windowed layers with their trained window as well.
        return True
    return bool(getattr(server_args, "speculative_dflash_lora_tts_enable", False))

# --- DFlash LoRA-TTS serving hook (CUDA-graph-safe) --------------------------
#
# When LoRA-TTS can update adapters (enabled and freq > 0), the worker
# installs one _StaticLoRABuffers at init, before the draft model is
# constructed. Draft CUDA-graph capture then records the delta bmms reading
# these fixed tensors, and each round only their contents are rewritten
# (dflash_lora_tts.publish_static: buffer row r holds batch position r and is
# refreshed before every draft forward, behind the per-round fence). Rows
# beyond the current batch are kept at zero, and B is zero until a request's
# first update, so the delta is exactly zero wherever no adapter has been
# trained. That includes the padding rows of a batch that runs in a graph
# captured for a larger batch size.
#
# When None, the delta path does not exist: it is neither captured in the
# graphs nor run eagerly, and the draft behaves as it does without TTS. This
# is the case for freq <= 0.
_LORA_STATIC = None


class _StaticLoRABuffers:
    """Persistent serving-side adapter buffers, indexed by batch row.

    a[(layer_id, kind)]: [max_bs, r, in_local]   (serving dtype)
    b[(layer_id, kind)]: [max_bs, out_local, r]  (serving dtype)
    """

    __slots__ = ("block_size", "scaling", "max_bs", "a", "b")

    def __init__(self, block_size, scaling, max_bs, a, b):
        self.block_size = int(block_size)
        self.scaling = float(scaling)
        self.max_bs = int(max_bs)
        self.a = a
        self.b = b


def _set_lora_static(buffers) -> None:
    global _LORA_STATIC
    _LORA_STATIC = buffers


def _lora_block_delta(
    kind: str, layer_id: int, x: torch.Tensor
) -> Optional[torch.Tensor]:
    """Per-row LoRA delta for one targeted projection, computed as two bmms
    over the static buffers. Safe to capture in a CUDA graph: nothing varies
    from round to round except the buffer contents. The delta is applied only
    when the token count is a whole number of blocks and the batch fits the
    buffers, which keeps other forwards through these modules (e.g.
    kv_proj_only) from picking up the adapters. During graph capture this
    check is evaluated with the static shapes of the captured batch size, so
    the delta is recorded only in draft-block graphs.
    """
    buf = _LORA_STATIC
    if buf is None:
        return None
    n = x.shape[0]
    bs = n // buf.block_size
    if bs * buf.block_size != n or bs > buf.max_bs or bs == 0:
        return None
    pair_a = buf.a.get((layer_id, kind))
    if pair_a is None:
        return None
    pair_b = buf.b[(layer_id, kind)]
    x3 = x.reshape(bs, buf.block_size, -1)
    if x3.dtype != pair_a.dtype:
        x3 = x3.to(pair_a.dtype)
    z = torch.bmm(x3, pair_a[:bs].transpose(1, 2))  # [bs, block, r]
    if kind in ("o", "down") and get_tensor_model_parallel_world_size() > 1:
        # Row-parallel target: A is sharded along its input, so z is a partial
        # sum over ranks. Reduce it at the rank-r bottleneck, where it is
        # small, before applying the replicated B. (All-reduces can be
        # captured in CUDA graphs; the target's TP graphs already do so.)
        z = tensor_model_parallel_all_reduce(z)
    delta = torch.bmm(z, pair_b[:bs].transpose(1, 2)) * buf.scaling
    return delta.reshape(n, -1).to(x.dtype)


class DFlashAttention(nn.Module):
    def __init__(self, config, layer_id: int) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        tp_size = int(get_tensor_model_parallel_world_size())
        total_num_heads = int(config.num_attention_heads)
        total_num_kv_heads = int(
            getattr(config, "num_key_value_heads", total_num_heads)
        )
        head_dim = int(getattr(config, "head_dim", hidden_size // total_num_heads))

        self.hidden_size = hidden_size
        self.layer_id = int(layer_id)
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        assert self.total_num_heads % tp_size == 0, (
            f"DFlashAttention requires total_num_heads divisible by tp_size. "
            f"total_num_heads={self.total_num_heads}, tp_size={tp_size}."
        )
        self.num_heads = self.total_num_heads // tp_size
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0, (
                f"DFlashAttention requires total_num_kv_heads divisible by tp_size when >= tp_size. "
                f"total_num_kv_heads={self.total_num_kv_heads}, tp_size={tp_size}."
            )
        else:
            assert tp_size % self.total_num_kv_heads == 0, (
                f"DFlashAttention requires tp_size divisible by total_num_kv_heads when total_num_kv_heads < tp_size. "
                f"total_num_kv_heads={self.total_num_kv_heads}, tp_size={tp_size}."
            )
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * head_dim
        self.kv_size = self.num_kv_heads * head_dim

        attention_bias = bool(getattr(config, "attention_bias", False))
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=attention_bias,
            prefix="qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * head_dim,
            hidden_size,
            bias=attention_bias,
            prefix="o_proj",
        )

        # Per-head Q/K RMSNorm, matching HF Qwen3.
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

        rope_theta = float(getattr(config, "rope_theta", 1000000))
        rope_scaling = getattr(config, "rope_scaling", None)
        rope_is_neox_style = bool(
            getattr(
                config, "rope_is_neox_style", getattr(config, "is_neox_style", True)
            )
        )
        max_position_embeddings = int(getattr(config, "max_position_embeddings", 32768))
        self.rotary_emb = get_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=rope_is_neox_style,
        )

        self.scaling = head_dim**-0.5

        # Draft models may mix full-attention and sliding-window layers through
        # config.layer_types (e.g. z-lab/Qwen3.6-27B-DFlash uses a 2048-token
        # window on 4 of its 5 layers). The window is applied when TTS is
        # enabled (see the module-level comment); otherwise every layer uses
        # full attention.
        layer_types = getattr(config, "layer_types", None)
        if layer_types and layer_id < len(layer_types):
            self._is_sliding = layer_types[layer_id] == "sliding_attention"
        else:
            self._is_sliding = bool(getattr(config, "use_sliding_window", False))
        self._sliding_window = int(getattr(config, "sliding_window", 0) or 0)
        if not self._sliding_window:
            self._is_sliding = False
        self._serve_windowed = self._is_sliding and draft_window_enabled()

        # DFlash uses non-causal attention over the draft block.
        # sliding_window_size uses SGLang's exclusive convention (window - 1);
        # the TTS forwards mask with the HF-inclusive
        # `q_pos - k_pos < window`, which is the same set of keys.
        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            attn_type=AttentionType.ENCODER_ONLY,
            sliding_window_size=(
                self._sliding_window - 1 if self._serve_windowed else -1
            ),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if _LORA_STATIC is not None:
            q_delta = _lora_block_delta("q", self.layer_id, hidden_states)
            if q_delta is not None:
                q = q + q_delta
            # k / v deltas when those kinds are targeted (None otherwise). The
            # k delta is applied before k_norm, like the q delta before q_norm.
            k_delta = _lora_block_delta("k", self.layer_id, hidden_states)
            if k_delta is not None:
                k = k + k_delta
            v_delta = _lora_block_delta("v", self.layer_id, hidden_states)
            if v_delta is not None:
                v = v + v_delta
        q, k = apply_qk_norm(q, k, self.q_norm, self.k_norm, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        if _LORA_STATIC is not None:
            o_delta = _lora_block_delta("o", self.layer_id, attn_output)
            if o_delta is not None:
                output = output + o_delta
        return output

    def kv_proj_only(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project hidden_states to K/V only (skip Q).

        This is used by DFlash to materialize ctx tokens into the draft KV cache:
        we only need K/V for the cached tokens; Q is never consumed.
        """
        # Fast path for unquantized weights: slice the fused QKV weight and run one GEMM.
        can_slice_qkv_weight, _ = can_dflash_slice_qkv_weight(self.qkv_proj)
        if can_slice_qkv_weight:
            kv_slice = slice(self.q_size, self.q_size + 2 * self.kv_size)
            weight = self.qkv_proj.weight[kv_slice]
            bias = (
                self.qkv_proj.bias[kv_slice] if self.qkv_proj.bias is not None else None
            )
            kv = F.linear(hidden_states, weight, bias)
            k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
            return k, v

        # Fallback: compute full QKV and discard Q (keeps compatibility with quantized weights).
        qkv, _ = self.qkv_proj(hidden_states)
        _, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        return k, v

    def apply_k_norm(self, k: torch.Tensor) -> torch.Tensor:
        k_by_head = k.reshape(-1, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        return k_by_head.view_as(k)

    def apply_k_rope(self, positions: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        # Match K shape so RoPE kernel head-count check passes on all backends.
        dummy_q = k.new_empty(k.shape)
        _, k = self.rotary_emb(positions, dummy_q, k)
        return k


class DFlashMLP(nn.Module):
    def __init__(
        self, config, layer_id: int, quant_config=None, prefix: str = ""
    ) -> None:
        super().__init__()
        self.layer_id = int(layer_id)
        hidden_size = int(config.hidden_size)
        intermediate_size = int(getattr(config, "intermediate_size", 0))
        if intermediate_size <= 0:
            raise ValueError(
                f"Invalid intermediate_size={intermediate_size} for DFlash MLP."
            )

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix="gate_up_proj" if not prefix else f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix="down_proj" if not prefix else f"{prefix}.down_proj",
        )
        hidden_act = getattr(config, "hidden_act", "silu")
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported DFlash activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        if _LORA_STATIC is not None:
            gu_delta = _lora_block_delta("gate_up", self.layer_id, x)
            if gu_delta is not None:
                gate_up = gate_up + gu_delta
        act = self.act_fn(gate_up)
        out, _ = self.down_proj(act)
        if _LORA_STATIC is not None:
            down_delta = _lora_block_delta("down", self.layer_id, act)
            if down_delta is not None:
                out = out + down_delta
        return out


class DFlashDecoderLayer(nn.Module):
    def __init__(self, config, layer_id: int) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = DFlashAttention(config=config, layer_id=layer_id)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = DFlashMLP(config=config, layer_id=layer_id)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.numel() == 0:
            # Keep return types consistent for upstream callers.
            if residual is None:
                residual = hidden_states
            return hidden_states, residual

        # Pre-norm attention with fused residual+norm when possible (Qwen3-style).
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        attn_out = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states, residual = self.post_attention_layernorm(attn_out, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class DFlashDraftModel(nn.Module):
    """SGLang DFlash draft model (no embedding / lm_head weights).

    The checkpoint provides:
      - transformer weights for `layers.*`
      - `fc.weight`, `hidden_norm.weight` for projecting target context features
      - `norm.weight` for final normalization
    """

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        self.config = config

        hidden_size = int(config.hidden_size)
        num_layers = int(config.num_hidden_layers)
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.layers = nn.ModuleList(
            [DFlashDecoderLayer(config=config, layer_id=i) for i in range(num_layers)]
        )
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # Project per-token target context features:
        # concat(K * hidden_size) -> hidden_size, where K is the number of target-layer
        # feature tensors concatenated per token (not necessarily equal to num_layers).
        draft_config = parse_dflash_draft_config(draft_hf_config=config)
        target_num_layers = (
            int(draft_config.num_target_layers)
            if draft_config.num_target_layers is not None
            else num_layers
        )
        target_layer_ids = draft_config.resolve_target_layer_ids(
            target_num_layers=target_num_layers, draft_num_layers=num_layers
        )
        num_context_features = len(target_layer_ids)

        self.num_context_features = int(num_context_features)
        self.fc = nn.Linear(
            self.num_context_features * hidden_size, hidden_size, bias=False
        )
        self.hidden_norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        self.block_size = draft_config.resolve_block_size(default=16)
        self._serves_windowed = any(
            layer.self_attn._serve_windowed for layer in self.layers
        )

    def get_attention_sliding_window_size(self):
        """SGLang exclusive convention (window - 1); None = no sliding layers.

        Read by ModelRunner so that the draft's attention backend allocates
        the window KV index buffers, which the fused kernels require once
        layers set a per-layer sliding_window_size. Returns None unless the
        draft is served with its trained window (TTS enabled), so
        otherwise the draft runner allocates no window buffers and uses full
        attention.
        """
        if not self._serves_windowed:
            return None
        return int(self.config.sliding_window) - 1

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        """Project concatenated target-layer hidden states into draft hidden_size."""
        expected = int(self.fc.in_features)
        if target_hidden.ndim != 2 or int(target_hidden.shape[-1]) != expected:
            raise ValueError(
                "DFLASH target_hidden feature dim mismatch. "
                f"Expected shape [N, {expected}] "
                f"(num_context_features={self.num_context_features}, hidden_size={int(self.config.hidden_size)}), "
                f"but got shape={tuple(target_hidden.shape)}. "
                "This usually means the target model is capturing a different number of layer features than "
                "the draft checkpoint/config expects."
            )
        return self.hidden_norm(self.fc(target_hidden))

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ) -> LogitsProcessorOutput:
        if input_embeds is None:
            raise ValueError(
                "DFlashDraftModel requires `input_embeds` (use the target embedding)."
            )
        hidden_states = input_embeds
        residual: Optional[torch.Tensor] = None

        for layer in self.layers:
            hidden_states, residual = layer(
                positions, hidden_states, forward_batch, residual
            )

        if hidden_states.numel() != 0:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)

        return LogitsProcessorOutput(
            next_token_logits=None,
            hidden_states=hidden_states,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, weight_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())

        def resolve_param_name(name: str) -> Optional[str]:
            if name in params_dict:
                return name
            if name.startswith("model."):
                stripped_name = name[len("model.") :]
                if stripped_name in params_dict:
                    return stripped_name
            else:
                prefixed_name = f"model.{name}"
                if prefixed_name in params_dict:
                    return prefixed_name
            return None

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if f".{weight_name}." not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                resolved_name = resolve_param_name(mapped_name)
                if resolved_name is None:
                    continue
                param = params_dict[resolved_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                resolved_name = resolve_param_name(name)
                if resolved_name is None:
                    # Ignore unexpected weights (e.g., HF rotary caches).
                    continue
                param = params_dict[resolved_name]
                if resolved_name.endswith("fc.weight") and tuple(
                    loaded_weight.shape
                ) != tuple(param.shape):
                    raise ValueError(
                        "DFLASH fc.weight shape mismatch. This usually means the draft checkpoint's "
                        "number of context features (K) does not match this config. "
                        f"Expected fc.weight.shape={tuple(param.shape)} "
                        f"(num_context_features={self.num_context_features}, hidden_size={int(self.config.hidden_size)}), "
                        f"but got {tuple(loaded_weight.shape)} for weight '{name}'."
                    )
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = DFlashDraftModel
