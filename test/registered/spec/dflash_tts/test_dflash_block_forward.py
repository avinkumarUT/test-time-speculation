"""Tests for DFlashBlockForward, the differentiable DFlash draft block forward.

Full-weight DFlash TTS drafts the block of an update round with
`dflash_tts.DFlashBlockForward` instead of the serving forward, whose fused
kernels have no backward. These tests build a small `DFlashDraftModel` with
the real SGLang layers and a real `MHATokenToKVPool` and check that:

  1. the forward equals an independent implementation of the DFlash block
     (pre-norm decoder layers, per-head Q/K RMSNorm, RoPE, non-causal attention
     over [cached prefix | block], grouped-query attention, sliding window),
     for full-attention and windowed layers;
  2. its gradients equal those of the independent implementation, reach every
     weight of the block forward, and nothing else;
  3. it stays differentiable inside the scheduler's no-grad / inference-mode
     context and with `torch.bmm` replaced by a function without autograd, as
     deterministic inference does;
  4. a batch is the per-request results, each with its own prefix;
  5. with float32 masters a bfloat16 draft keeps bfloat16 weights, an update
     through `DFlashTTSState` equals a float32 Adam step on the float32
     weights with the gradient of the bfloat16 block forward, rounded, and
     updates lower the loss;
  6. with the bfloat16 optimizer dtype, several updates through
     `DFlashTTSState` equal float32 Adam with the same clip on the widened
     state, rounded once per update, and updates lower the loss;
  7. the DFlash worker's hooks route a round correctly in both optimizer
     dtypes: the block forward runs on update rounds only, the serving
     forward drafts every other round and a round whose block forward fails,
     and an update is applied, skipped or contained as it should be;
  8. it agrees with the replay forward that the LoRA mode trains with;
  9. the startup validation rejects each fault with its own message;
 10. the real worker methods wire TTS into a round correctly (reset at a
     prefill, fence before the draft weights are read, target logits copied
     before verification and only on update rounds, post-verify hook called,
     request slots, detached drafting input, lm_head rows, stale blocks,
     fused KV materialization off), run against a stubbed target.

Always runs on CPU; needs no GPU, checkpoint or server.

  python test/registered/spec/dflash_tts/test_dflash_block_forward.py
"""

import math
import os
import sys
from types import SimpleNamespace

os.environ.setdefault("SGLANG_USE_CPU_ENGINE", "1")

import torch
import torch.nn.functional as F
from bf16_reference import (
    Bf16StateReference,
    optimizer_is_fresh,
    optimizer_moments,
    same_bits,
)

import sglang.srt.distributed.parallel_state as parallel_state
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

HIDDEN, HEADS, KV_HEADS, HEAD_DIM, INTER = 32, 4, 2, 8, 48
LAYERS, BLOCK, VOCAB, WINDOW = 3, 4, 64, 6  # WINDOW: HF convention (q - k < W)
ROPE_THETA, EPS = 10000.0, 1e-6
FAILS = []


def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' — ' + extra) if extra else ''}")
    if not cond:
        FAILS.append(name)


# The draft model reads the TP layout and the global server arguments while it
# is constructed. A single-rank stand-in is enough for both; the server
# arguments enable full-weight TTS so that windowed layers get their window.
parallel_state._TP = SimpleNamespace(world_size=1, rank_in_group=0)
_server_args = ServerArgs(model_path="dummy", device="cpu")
_server_args.speculative_dflash_tts_enable = True
set_global_server_args_for_scheduler(_server_args)

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool  # noqa: E402
from sglang.srt.models import dflash as dflash_model  # noqa: E402
from sglang.srt.speculative import dflash_tts  # noqa: E402
from sglang.srt.speculative.dflash_tts import (  # noqa: E402
    DFlashBlockForward,
    DFlashTTSConfig,
    DFlashTTSState,
)
from sglang.srt.utils import DynamicGradMode  # noqa: E402


def make_model(seed=0, dtype=torch.float32):
    config = SimpleNamespace(
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        intermediate_size=INTER,
        num_hidden_layers=LAYERS,
        rms_norm_eps=EPS,
        attention_bias=False,
        rope_theta=ROPE_THETA,
        max_position_embeddings=256,
        hidden_act="silu",
        # Two windowed layers and one full-attention layer, like the
        # published DFlash drafts.
        layer_types=["sliding_attention", "full_attention", "sliding_attention"],
        sliding_window=WINDOW,
        use_sliding_window=True,
        block_size=BLOCK,
        dflash_config={"mask_token_id": 3, "target_layer_ids": [0, 1]},
        num_target_layers=4,
    )
    model = dflash_model.DFlashDraftModel(config)
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("norm.weight") or "layernorm" in name:
                param.copy_(1.0 + 0.2 * torch.randn(param.shape, generator=gen))
            else:
                param.copy_(
                    torch.randn(param.shape, generator=gen) / math.sqrt(param.shape[-1])
                )
            # Convert the parameters only. The RoPE cos/sin cache is a buffer
            # that serving keeps in fp32 on CUDA.
            param.data = param.data.to(dtype)
    return model


def make_cache(model, prefix_lens, seed=0, dtype=torch.float32):
    """A KV pool holding random prefix K/V for one request per prefix length.

    The prefixes are stored at shuffled pool locations so that the forward
    has to follow the request-to-token table.
    """
    gen = torch.Generator().manual_seed(100 + seed)
    size = sum(prefix_lens) + 16
    pool = MHATokenToKVPool(
        size=size,
        page_size=1,
        dtype=dtype,
        head_num=KV_HEADS,
        head_dim=HEAD_DIM,
        layer_num=LAYERS,
        device="cpu",
        enable_memory_saver=False,
    )
    perm = torch.randperm(size, generator=gen)
    req_to_token = torch.zeros(
        len(prefix_lens), max(prefix_lens) + BLOCK, dtype=torch.int32
    )
    prefix_k, prefix_v, offset = [], [], 0
    for row, prefix_len in enumerate(prefix_lens):
        locs = perm[offset : offset + prefix_len]
        offset += prefix_len
        req_to_token[row, :prefix_len] = locs.to(torch.int32)
        k = torch.randn(LAYERS, prefix_len, KV_HEADS, HEAD_DIM, generator=gen)
        v = torch.randn(LAYERS, prefix_len, KV_HEADS, HEAD_DIM, generator=gen)
        for layer_id in range(LAYERS):
            k_buffer, v_buffer = pool.get_kv_buffer(layer_id)
            k_buffer[locs] = k[layer_id].to(dtype)
            v_buffer[locs] = v[layer_id].to(dtype)
        prefix_k.append(k.to(dtype))
        prefix_v.append(v.to(dtype))
    return pool, req_to_token, prefix_k, prefix_v


def make_inputs(prefix_lens, seed=0, dtype=torch.float32):
    gen = torch.Generator().manual_seed(200 + seed)
    embeds = torch.randn(len(prefix_lens), BLOCK, HIDDEN, generator=gen).to(dtype)
    positions = torch.stack(
        [torch.arange(p, p + BLOCK, dtype=torch.int64) for p in prefix_lens]
    )
    return embeds, positions


# --- Independent implementation ------------------------------------------------
# Written from the model definition (HF-style formulation: explicit residual
# stream, explicit softmax attention on absolute positions over the whole
# prefix), sharing no code with DFlashBlockForward. `weights` maps parameter
# names to float64 tensors, so it also serves as the reference for gradients.


def _rms_norm(x, weight):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * weight


def _rope(x, positions):
    """Neox-style RoPE on [n, heads, head_dim] at absolute `positions`."""
    half = HEAD_DIM // 2
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(half, dtype=torch.float64) / half))
    angles = positions.double()[:, None] * inv_freq[None, :]
    cos, sin = angles.cos()[:, None, :], angles.sin()[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def reference_forward(weights, embeds, positions, prefix_k, prefix_v, windowed_layers):
    """Hidden states [B, hidden] of one request's block, in float64."""
    n = embeds.shape[0]
    prefix_len = prefix_k.shape[1]
    num_rep = HEADS // KV_HEADS
    h = embeds.double()
    for layer_id in range(LAYERS):
        pre = f"layers.{layer_id}."
        x = _rms_norm(h, weights[pre + "input_layernorm.weight"])
        qkv = x @ weights[pre + "self_attn.qkv_proj.weight"].T
        q, k, v = qkv.split(
            [HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM], dim=-1
        )
        q = _rms_norm(
            q.view(n, HEADS, HEAD_DIM), weights[pre + "self_attn.q_norm.weight"]
        )
        k = _rms_norm(
            k.view(n, KV_HEADS, HEAD_DIM), weights[pre + "self_attn.k_norm.weight"]
        )
        q, k = _rope(q, positions), _rope(k, positions)
        v = v.view(n, KV_HEADS, HEAD_DIM)

        keys = torch.cat([prefix_k[layer_id].double(), k])  # [P + n, kv_heads, d]
        values = torch.cat([prefix_v[layer_id].double(), v])
        key_pos = torch.cat([torch.arange(prefix_len), positions])
        allowed = torch.ones(n, prefix_len + n, dtype=torch.bool)
        if layer_id in windowed_layers:
            # A prefix key is visible within the window; block keys always are.
            in_window = (positions[:, None] - key_pos[None, :]) < WINDOW
            allowed = in_window | (torch.arange(prefix_len + n)[None, :] >= prefix_len)
        out = torch.zeros(n, HEADS, HEAD_DIM, dtype=torch.float64)
        for head in range(HEADS):
            kv_head = head // num_rep
            scores = q[:, head] @ keys[:, kv_head].T / math.sqrt(HEAD_DIM)
            scores = scores.masked_fill(~allowed, float("-inf"))
            out[:, head] = torch.softmax(scores, dim=-1) @ values[:, kv_head]
        h = (
            h
            + out.reshape(n, HEADS * HEAD_DIM)
            @ weights[pre + "self_attn.o_proj.weight"].T
        )

        x = _rms_norm(h, weights[pre + "post_attention_layernorm.weight"])
        gate, up = (x @ weights[pre + "mlp.gate_up_proj.weight"].T).chunk(2, dim=-1)
        h = h + (F.silu(gate) * up) @ weights[pre + "mlp.down_proj.weight"].T
    return _rms_norm(h, weights["norm.weight"])


def double_weights(model, requires_grad=False):
    return {
        name: param.detach().double().requires_grad_(requires_grad)
        for name, param in model.named_parameters()
    }


WINDOWED = {0, 2}

print("\n[1] forward equals the independent implementation")
model = make_model()
check(
    "windowed layers carry their serving window",
    [layer.self_attn.attn.sliding_window_size for layer in model.layers]
    == [WINDOW - 1, -1, WINDOW - 1],
)
for prefix_len in (2, WINDOW - 1, WINDOW, WINDOW + 3, 20):
    pool, req_to_token, prefix_k, prefix_v = make_cache(model, [prefix_len])
    forward = DFlashBlockForward(
        draft_model=model, token_to_kv_pool=pool, block_size=BLOCK
    )
    embeds, positions = make_inputs([prefix_len])
    out = forward(
        input_embeds=embeds,
        positions=positions,
        prefix_lens=[prefix_len],
        req_pool_indices=[0],
        req_to_token=req_to_token,
    )
    ref = reference_forward(
        double_weights(model),
        embeds[0],
        positions[0],
        prefix_k[0],
        prefix_v[0],
        WINDOWED,
    )
    err = float((out[0].double() - ref).abs().max())
    check(
        f"prefix {prefix_len:2d}: matches (window {WINDOW})",
        out.shape == (1, BLOCK, HIDDEN) and err < 1e-4,
        f"max|err|={err:.2e}",
    )
# Negative control: without the window mask the long-prefix case must differ,
# so the comparison above is sensitive to the mask.
ref_unmasked = reference_forward(
    double_weights(model), embeds[0], positions[0], prefix_k[0], prefix_v[0], set()
)
check(
    "the window mask matters for a long prefix",
    float((out[0].double() - ref_unmasked).abs().max()) > 1e-3,
)

print("\n[2] gradients equal the independent implementation")
prefix_len = WINDOW + 3
pool, req_to_token, prefix_k, prefix_v = make_cache(model, [prefix_len])
forward = DFlashBlockForward(draft_model=model, token_to_kv_pool=pool, block_size=BLOCK)
embeds, positions = make_inputs([prefix_len])
probe = torch.randn(BLOCK, HIDDEN, generator=torch.Generator().manual_seed(7))
for param in model.parameters():
    param.requires_grad_(True)
    param.grad = None
out = forward(
    input_embeds=embeds,
    positions=positions,
    prefix_lens=[prefix_len],
    req_pool_indices=[0],
    req_to_token=req_to_token,
)
check("the forward returns a graph", out.grad_fn is not None)
(out[0] * probe).sum().backward()
ref_weights = double_weights(model, requires_grad=True)
ref = reference_forward(
    ref_weights, embeds[0], positions[0], prefix_k[0], prefix_v[0], WINDOWED
)
(ref * probe.double()).sum().backward()
worst, unreached = 0.0, []
for name, param in model.named_parameters():
    ref_grad = ref_weights[name].grad
    if param.grad is None:
        unreached.append(name)
        if ref_grad is not None and float(ref_grad.abs().max()) > 0:
            worst = float("inf")
        continue
    scale = float(ref_grad.abs().max())
    worst = max(
        worst, float((param.grad.double() - ref_grad).abs().max()) / max(scale, 1e-12)
    )
check("every gradient matches", worst < 1e-3, f"max relative error={worst:.2e}")
check(
    "only the context projection is unreached",
    unreached == ["fc.weight", "hidden_norm.weight"],
    str(unreached),
)
check(
    "all reached gradients are finite and non-zero",
    all(
        bool(torch.isfinite(p.grad).all()) and float(p.grad.abs().sum()) > 0
        for p in model.parameters()
        if p.grad is not None
    ),
)

print("\n[3] differentiable inside the serving stack")
with DynamicGradMode():  # the scheduler loop: autograd disabled
    embeds_ng, positions_ng = make_inputs([prefix_len])
    out_ng = forward(
        input_embeds=embeds_ng,
        positions=positions_ng,
        prefix_lens=[prefix_len],
        req_pool_indices=[0],
        req_to_token=req_to_token,
    )
check("graph survives an enclosing no-grad context", out_ng.grad_fn is not None)
with torch.inference_mode():  # SGLANG_ENABLE_TORCH_INFERENCE_MODE=1
    embeds_inf, positions_inf = make_inputs([prefix_len])
    embeds_inf = embeds_inf + 0  # an inference tensor
    out_inf = forward(
        input_embeds=embeds_inf,
        positions=positions_inf,
        prefix_lens=[prefix_len],
        req_pool_indices=[0],
        req_to_token=req_to_token,
    )
    for param in model.parameters():
        param.grad = None
    with torch.inference_mode(False), torch.enable_grad():
        out_inf.sum().backward()
check(
    "graph survives an enclosing inference mode",
    out_inf.grad_fn is not None
    and model.layers[0].self_attn.qkv_proj.weight.grad is not None,
)
check(
    "results agree",
    torch.allclose(out_ng, out, atol=1e-6) and torch.allclose(out_inf, out, atol=1e-6),
)

# Deterministic inference replaces torch.bmm with a Python-level function that
# bypasses autograd. The block forward must not depend on it.
real_bmm = torch.bmm


def bmm_without_autograd(a, b, *, out=None):
    with torch.no_grad():
        return real_bmm(a.detach(), b.detach())


torch.bmm = bmm_without_autograd
try:
    probe_a = torch.randn(2, 3, 4, requires_grad=True)
    premise = torch.bmm(probe_a, torch.randn(2, 4, 5)).grad_fn is None
    for param in model.parameters():
        param.grad = None
    out_det = forward(
        input_embeds=embeds,
        positions=positions,
        prefix_lens=[prefix_len],
        req_pool_indices=[0],
        req_to_token=req_to_token,
    )
    (out_det[0] * probe).sum().backward()
    reached = [name for name, p in model.named_parameters() if p.grad is not None]
finally:
    torch.bmm = real_bmm
check("premise: the replaced torch.bmm has no autograd", premise)
check(
    "forward keeps its graph and reaches every block weight",
    out_det.grad_fn is not None
    and len(reached) == len(list(model.parameters())) - 2
    and torch.allclose(out_det, out, atol=1e-6),
)
with open(dflash_tts.__file__) as source:
    check(
        "dflash_tts.py contains no direct torch.bmm call",
        "torch.bmm(" not in source.read(),
    )

print("\n[4] a batch is the per-request results")
prefix_lens = [3, WINDOW + 5, 11]
pool, req_to_token, prefix_k, prefix_v = make_cache(model, prefix_lens, seed=1)
forward = DFlashBlockForward(draft_model=model, token_to_kv_pool=pool, block_size=BLOCK)
embeds, positions = make_inputs(prefix_lens, seed=1)
out = forward(
    input_embeds=embeds,
    positions=positions,
    prefix_lens=prefix_lens,
    req_pool_indices=[0, 1, 2],
    req_to_token=req_to_token,
)
errs = []
for row in range(3):
    ref = reference_forward(
        double_weights(model),
        embeds[row],
        positions[row],
        prefix_k[row],
        prefix_v[row],
        WINDOWED,
    )
    errs.append(float((out[row].double() - ref).abs().max()))
check(
    "each row uses its own prefix",
    out.shape == (3, BLOCK, HIDDEN) and max(errs) < 1e-4,
    str(errs),
)
# The request slot, not the batch row, selects the prefix.
out_swapped = forward(
    input_embeds=embeds[[1, 0]],
    positions=positions[[1, 0]],
    prefix_lens=[prefix_lens[1], prefix_lens[0]],
    req_pool_indices=[1, 0],
    req_to_token=req_to_token,
)
check(
    "request slots select the prefix",
    torch.allclose(out_swapped[0], out[1], atol=1e-6)
    and torch.allclose(out_swapped[1], out[0], atol=1e-6),
)

print("\n[5] float32 masters over bfloat16 draft weights")
prefix_len = WINDOW + 3
target = torch.nn.Linear(HIDDEN, VOCAB, bias=False).to(torch.bfloat16)
target_logits = 3.0 * torch.randn(
    BLOCK, VOCAB, generator=torch.Generator().manual_seed(5)
)
target_logits = target_logits.to(torch.bfloat16)
embeds, positions = make_inputs([prefix_len], dtype=torch.bfloat16)
BLOCK_WEIGHTS_UNREACHED = ["fc.weight", "hidden_norm.weight"]


def make_tts(optimizer_dtype, lr, **cfg_kwargs):
    """A bfloat16 draft on a bfloat16 KV cache, its block forward and state."""
    cfg_kwargs.setdefault("tts_loss", "kl")
    model = make_model(dtype=torch.bfloat16)
    pool, req_to_token, prefix_k, prefix_v = make_cache(
        model, [prefix_len], dtype=torch.bfloat16
    )
    forward = DFlashBlockForward(
        draft_model=model, token_to_kv_pool=pool, block_size=BLOCK
    )
    state = DFlashTTSState(
        draft_model=model,
        config=DFlashTTSConfig(
            enable=True,
            lr=lr,
            async_step=False,
            optimizer_dtype=optimizer_dtype,
            **cfg_kwargs,
        ),
        device="cpu",
        target_model=target,
    )
    call = dict(
        input_embeds=embeds,
        positions=positions,
        prefix_lens=[prefix_len],
        req_pool_indices=[0],
        req_to_token=req_to_token,
    )
    return SimpleNamespace(
        model=model,
        forward=forward,
        state=state,
        call=call,
        prefix_k=prefix_k,
        prefix_v=prefix_v,
    )


def tts_loss(tts):
    with torch.no_grad():
        kl, ce = tts.state.compute_loss(
            draft_hidden=tts.forward(**tts.call),
            target_logits=target_logits,
            lm_head_weight=target.weight,
        )
    return float(tts.state.cfg.tts_lambda * kl + (1 - tts.state.cfg.tts_lambda) * ce)


def update(tts):
    """One update round, as the worker runs it inside the scheduler loop."""
    with DynamicGradMode():
        hidden = tts.forward(**tts.call)
        tts.state.step(
            draft_hidden=hidden,
            target_logits=target_logits,
            lm_head_weight=target.weight,
        )
        tts.state.advance()


tts = make_tts("float32", lr=1e-3)
model, state = tts.model, tts.state
names = [n for n, _ in model.named_parameters()]
before = {n: p.detach().clone() for n, p in model.named_parameters()}
storage = {n: p.data_ptr() for n, p in model.named_parameters()}
check(
    "the state holds a float32 master of every bfloat16 parameter",
    state._masters is not None
    and len(state._masters) == len(names)
    and all(p.dtype == torch.bfloat16 for p in model.parameters())
    and all(
        m.dtype == torch.float32 and torch.equal(m, before[n].float())
        for n, m in zip(names, state._masters)
    ),
)
out = tts.forward(**tts.call)
ref = reference_forward(
    double_weights(model),
    embeds[0],
    positions[0],
    tts.prefix_k[0],
    tts.prefix_v[0],
    WINDOWED,
)
cos = float(F.cosine_similarity(out[0].double().flatten(), ref.flatten(), dim=0))
check(
    "forward is bfloat16 and close to the reference",
    out.dtype == torch.bfloat16 and cos > 0.999,
    f"cos={cos:.6f}",
)

# One update against a reference: the gradient of the bfloat16 block forward
# on a twin model, then clipping and Adam in float32 on the float32 copy of
# the weights, then rounding to bfloat16. The twin's state only evaluates the
# loss; it never steps.
twin = make_tts("bfloat16", lr=1e-3)
twin_params = dict(twin.model.named_parameters())
with DynamicGradMode():
    hidden = twin.forward(**twin.call)
    with torch.inference_mode(False), torch.enable_grad():
        kl, ce = twin.state.compute_loss(
            draft_hidden=hidden,
            target_logits=target_logits,
            lm_head_weight=target.weight,
        )
        lam = twin.state.cfg.tts_lambda
        reached = [n for n in names if n not in BLOCK_WEIGHTS_UNREACHED]
        grads = torch.autograd.grad(
            lam * kl + (1.0 - lam) * ce, [twin_params[n] for n in reached]
        )
ref_masters = {n: before[n].float().clone() for n in reached}
for n, grad in zip(reached, grads):
    ref_masters[n].grad = grad.float()
ref_norm = float(
    torch.nn.utils.clip_grad_norm_(list(ref_masters.values()), state.cfg.grad_clip)
)
torch.optim.Adam(list(ref_masters.values()), lr=1e-3).step()
update(tts)
masters = dict(zip(names, state._masters))
check(
    "one update: masters equal float32 Adam on the float32 weights, bit for bit",
    all(g.dtype == torch.bfloat16 for g in grads)
    and ref_norm > state.cfg.grad_clip
    and all(
        torch.equal(masters[n], ref_masters[n])
        and not torch.equal(masters[n], before[n].float())
        for n in reached
    ),
    f"gradient norm before clipping={ref_norm:.3f}",
)
check(
    "one update: weights are the reference rounded to bfloat16, in place",
    all(
        torch.equal(p, ref_masters[n].to(torch.bfloat16))
        for n, p in model.named_parameters()
        if n in reached
    )
    and all(p.data_ptr() == storage[n] for n, p in model.named_parameters())
    and all(p.dtype == torch.bfloat16 for p in model.parameters()),
)
check(
    "one update: fc and hidden_norm keep their weights and masters",
    all(
        torch.equal(dict(model.named_parameters())[n], before[n])
        and torch.equal(masters[n], before[n].float())
        for n in BLOCK_WEIGHTS_UNREACHED
    ),
)

loss_before = tts_loss(tts)
target_before = target.weight.detach().clone()
for _ in range(20):
    update(tts)
loss_after = tts_loss(tts)
changed = [n for n, p in model.named_parameters() if not torch.equal(p, before[n])]
check(
    "updates lower the loss",
    loss_after < loss_before,
    f"{loss_before:.4f} -> {loss_after:.4f}",
)
check(
    "every block weight changed; fc and hidden_norm did not",
    sorted(set(before) - set(changed)) == BLOCK_WEIGHTS_UNREACHED,
    str(sorted(set(before) - set(changed))),
)
check(
    "the weights are the masters rounded to bfloat16",
    all(
        torch.equal(p, m.to(torch.bfloat16))
        for p, m in zip(model.parameters(), state._masters)
    ),
)
check("the target head is untouched", torch.equal(target.weight, target_before))
state.reset()
check(
    "reset restores every weight and every master",
    all(torch.equal(p, before[n]) for n, p in model.named_parameters())
    and all(torch.equal(m, before[n].float()) for n, m in zip(names, state._masters)),
)

print("\n[6] bfloat16 optimizer state over bfloat16 draft weights")
tts = make_tts("bfloat16", lr=1e-2, grad_clip=0.1)
check("no masters with the bfloat16 optimizer dtype", tts.state._masters is None)
# Several updates against a reference: the gradient of the bfloat16 block
# forward on a twin model, then the same clip and float32 Adam on the widened
# state, then rounding to bfloat16 once per update. The twin's state only
# evaluates the loss.
twin = make_tts("float32", lr=1e-2)
names = [n for n, _ in tts.model.named_parameters()]
twin_params = dict(twin.model.named_parameters())
reached = [n for n in names if n not in BLOCK_WEIGHTS_UNREACHED]
before = {n: p.detach().clone() for n, p in tts.model.named_parameters()}
storage = {n: p.data_ptr() for n, p in tts.model.named_parameters()}
reference = Bf16StateReference(
    [twin_params[n] for n in reached], 1e-2, tts.state.cfg.grad_clip
)
state_index = {n: names.index(n) for n in reached}
weights_equal, moments_equal, norms = [], [], []
for _ in range(6):
    with DynamicGradMode():
        hidden = twin.forward(**twin.call)
        with torch.inference_mode(False), torch.enable_grad():
            kl, ce = twin.state.compute_loss(
                draft_hidden=hidden,
                target_logits=target_logits,
                lm_head_weight=target.weight,
            )
            lam = twin.state.cfg.tts_lambda
            grads = torch.autograd.grad(
                lam * kl + (1.0 - lam) * ce, [twin_params[n] for n in reached]
            )
    norms.append(reference.update(grads))
    update(tts)
    weights_equal.append(
        all(
            same_bits(p, twin_params[n])
            for n, p in tts.model.named_parameters()
            if n in reached
        )
    )
    optimizer = tts.state._optimizer
    moments_equal.append(
        all(
            same_bits(optimizer.m[state_index[n]], reference.m[j])
            and same_bits(optimizer.v[state_index[n]], reference.v[j])
            for j, n in enumerate(reached)
        )
    )
check(
    "weights equal the reference after every update, bit for bit",
    all(weights_equal) and all(n > tts.state.cfg.grad_clip for n in norms),
    f"{weights_equal}; gradient norms before clipping {norms}",
)
check(
    "moments equal the reference after every update, bit for bit",
    all(moments_equal),
    str(moments_equal),
)
check(
    "the state is bfloat16 only, in the parameters' own storage",
    all(p.dtype == torch.bfloat16 for p in tts.model.parameters())
    and all(p.data_ptr() == storage[n] for n, p in tts.model.named_parameters())
    and all(m.dtype == torch.bfloat16 for m in optimizer_moments(tts.state))
    and sum(m.untyped_storage().nbytes() for m in optimizer_moments(tts.state))
    == 4 * sum(twin_params[n].numel() for n in reached),
)
check(
    "fc and hidden_norm keep their weights and have no moments",
    all(
        same_bits(dict(tts.model.named_parameters())[n], before[n])
        and tts.state._optimizer.m[names.index(n)] is None
        for n in BLOCK_WEIGHTS_UNREACHED
    ),
)
tts.state.reset()
check(
    "reset restores every weight and clears the moments",
    all(same_bits(p, before[n]) for n, p in tts.model.named_parameters())
    and optimizer_is_fresh(tts.state),
)
loss_before = tts_loss(tts)
for _ in range(20):
    update(tts)
loss_after = tts_loss(tts)
check(
    "weights stay bfloat16 and the loss decreases",
    all(p.dtype == torch.bfloat16 for p in tts.model.parameters())
    and loss_after < loss_before,
    f"{loss_before:.4f} -> {loss_after:.4f}",
)

print("\n[7] worker hooks")
# Fixture of the worker sections. Defined here so that they do not depend on
# names that earlier sections happen to leave behind.
W_PREFIX = WINDOW + 3
W_HEAD = torch.nn.Linear(HIDDEN, VOCAB, bias=False).to(torch.bfloat16)
W_LOGITS = (
    3.0 * torch.randn(BLOCK, VOCAB, generator=torch.Generator().manual_seed(5))
).to(torch.bfloat16)

from sglang.srt.speculative.dflash_worker import DFlashWorker  # noqa: E402


class Req:
    def __init__(self, req_pool_idx, finished=False):
        self.req_pool_idx = req_pool_idx
        self._finished = finished

    def finished(self):
        return self._finished


def make_batch(*reqs):
    return SimpleNamespace(reqs=list(reqs), batch_size=lambda: len(reqs))


class CountingForward:
    """Wraps the block forward: counts its calls and can make it fail."""

    def __init__(self, forward):
        self.forward = forward
        self.calls = []  # one entry per call
        self.slots = []  # the `req_pool_indices` argument of every call
        self.error = None

    def __call__(self, **kwargs):
        self.calls.append(True)
        self.slots.append(list(kwargs["req_pool_indices"]))
        if self.error is not None:
            raise self.error
        return self.forward(**kwargs)


def make_worker(optimizer_dtype, freq=2, lm_head=None):
    """A DFlashWorker with only the attributes its TTS hooks read."""
    model = make_model(dtype=torch.bfloat16)
    pool, req_to_token, _, _ = make_cache(
        model, [W_PREFIX, W_PREFIX], dtype=torch.bfloat16
    )
    worker = object.__new__(DFlashWorker)
    worker.__dict__.update(
        target_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                model=SimpleNamespace(lm_head=W_HEAD if lm_head is None else lm_head)
            )
        ),
        draft_model_runner=SimpleNamespace(
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token)
        ),
        block_size=BLOCK,
        _tts_lm_head_rows=VOCAB,
        _tts_pending_hidden=None,
        _tts_forward=CountingForward(
            DFlashBlockForward(
                draft_model=model, token_to_kv_pool=pool, block_size=BLOCK
            )
        ),
        _tts_state=DFlashTTSState(
            draft_model=model,
            config=DFlashTTSConfig(
                enable=True,
                lr=1e-2,
                freq=freq,
                async_step=False,
                optimizer_dtype=optimizer_dtype,
            ),
            device="cpu",
            target_model=W_HEAD,
        ),
    )
    return worker, model


def draft(worker, batch):
    """The draft hook of one round. None means: run the serving forward."""
    bs = batch.batch_size()
    block_embeds, block_positions = make_inputs([W_PREFIX] * bs, dtype=torch.bfloat16)
    with DynamicGradMode():
        return worker._tts_draft_forward(
            batch=batch,
            block_input_embeds=block_embeds,
            positions_2d=block_positions,
            draft_prefix_lens_cpu=torch.tensor([W_PREFIX] * bs, dtype=torch.int32),
        )


def weights_of(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def unchanged(model, weights):
    return all(torch.equal(p, weights[n]) for n, p in model.named_parameters())


def masters_match(state, weights):
    """Whether the masters, if the mode has them, are the float32 `weights`."""
    if state._masters is None:
        return True
    return all(
        torch.equal(m, w.float()) for m, w in zip(state._masters, weights.values())
    )


one = make_batch(Req(0))
bad_logits = W_LOGITS[:, : VOCAB // 2]
for dtype in ("float32", "bfloat16"):
    worker, model = make_worker(dtype)
    state, block_forward = worker._tts_state, worker._tts_forward
    pretrained = weights_of(model)
    check(
        f"{dtype}: masters {'exist' if dtype == 'float32' else 'do not exist'}",
        (state._masters is not None) == (dtype == "float32"),
    )
    out = draft(worker, one)
    check(
        f"{dtype}: update round: drafted by the block forward, graph kept aside",
        out is not None
        and block_forward.calls == [True]
        and out.hidden_states.shape == (BLOCK, HIDDEN)
        and out.hidden_states.dtype == torch.bfloat16
        and out.hidden_states.grad_fn is None
        and worker._tts_pending_hidden is not None
        and worker._tts_pending_hidden.grad_fn is not None,
    )
    with DynamicGradMode():
        worker._run_tts_post_verify(one, W_LOGITS)
    check(
        f"{dtype}: update applied after verification, stride advanced",
        state.update_ct == 1
        and state.round_idx == 1
        and worker._tts_pending_hidden is None
        and not torch.equal(model.norm.weight, pretrained["norm.weight"])
        and all(p.dtype == torch.bfloat16 for p in model.parameters())
        and (state._masters is None or not masters_match(state, pretrained)),
    )
    out = draft(worker, one)
    check(
        f"{dtype}: other rounds: the serving forward drafts, no block forward",
        not state.should_fire()
        and out is None
        and worker._tts_pending_hidden is None
        and block_forward.calls == [True],
    )
    worker._run_tts_post_verify(one, None)
    check(
        f"{dtype}: no update without a pending block",
        state.update_ct == 1 and state.round_idx == 2,
    )
    out = draft(worker, make_batch(Req(0), Req(1)))
    check(
        f"{dtype}: two requests in the batch: serving forward, no update",
        state.should_fire()
        and out is None
        and worker._tts_pending_hidden is None
        and block_forward.calls == [True],
    )
    out = draft(worker, one)
    weights_now = weights_of(model)
    worker._run_tts_post_verify(make_batch(Req(0, finished=True)), W_LOGITS)
    check(
        f"{dtype}: finished request: the update is skipped",
        out is not None
        and block_forward.calls == [True, True]
        and state.update_ct == 1
        and state.round_idx == 3
        and unchanged(model, weights_now),
    )
    state.advance()
    out = draft(worker, one)
    worker._run_tts_post_verify(one, bad_logits)
    check(
        f"{dtype}: tp_size 1: a failed update is contained",
        out is not None
        and state.update_ct == 1
        and state._failure_streak == 1
        and not state.is_disabled(),
    )
    state.advance()
    out = draft(worker, one)
    # A block without an autograd graph fails before any collective is issued.
    worker._tts_pending_hidden = worker._tts_pending_hidden.detach()
    state.tp_size = 2
    try:
        worker._run_tts_post_verify(one, W_LOGITS)
        raised = False
    except RuntimeError:
        raised = True
    state.tp_size = 1
    check(f"{dtype}: tp_size > 1: a failed update raises", raised)

    # The block forward itself fails on an update round (for example, out of
    # memory under autograd): the serving forward drafts the round.
    worker, model = make_worker(dtype)
    state, block_forward = worker._tts_state, worker._tts_forward
    pretrained = weights_of(model)
    block_forward.error = RuntimeError("CUDA out of memory")
    out = draft(worker, one)
    check(
        f"{dtype}: block forward fails: the round falls back to the serving forward",
        state.round_idx == 0
        and block_forward.calls == [True]
        and out is None
        and worker._tts_pending_hidden is None
        and state._failure_streak == 1
        and not state.is_disabled(),
    )
    worker._run_tts_post_verify(one, W_LOGITS)
    check(
        f"{dtype}: the failed round applies no update and advances the stride",
        state.update_ct == 0
        and state.round_idx == 1
        and unchanged(model, pretrained)
        and masters_match(state, pretrained),
    )
    block_forward.error = None
    state.advance()
    out = draft(worker, one)
    with DynamicGradMode():
        worker._run_tts_post_verify(one, W_LOGITS)
    check(
        f"{dtype}: the next update round succeeds and clears the failure streak",
        out is not None
        and state.update_ct == 1
        and state._failure_streak == 0
        and not unchanged(model, pretrained),
    )
    block_forward.error = RuntimeError("CUDA out of memory")
    state.advance()
    state.tp_size = 2
    try:
        draft(worker, one)
        raised = False
    except RuntimeError:
        raised = True
    state.tp_size = 1
    check(f"{dtype}: tp_size > 1: a failing block forward raises", raised)
    calls_before = len(block_forward.calls)
    for _ in range(DFlashTTSState._MAX_FAILURE_STREAK):
        check(
            f"{dtype}: update round {state.round_idx} still attempts the block forward",
            state.should_fire() and not state.is_disabled(),
        )
        out = draft(worker, one)
        worker._run_tts_post_verify(one, W_LOGITS)
        state.advance()
    check(
        f"{dtype}: repeated failures disable TTS and restore the pretrained weights",
        out is None
        and len(block_forward.calls)
        == calls_before + DFlashTTSState._MAX_FAILURE_STREAK
        and state.is_disabled()
        and unchanged(model, pretrained)
        and masters_match(state, pretrained),
    )
    out = draft(worker, one)
    check(
        f"{dtype}: a disabled state leaves every round to the serving forward",
        out is None
        and len(block_forward.calls)
        == calls_before + DFlashTTSState._MAX_FAILURE_STREAK,
    )

print("\n[8] agrees with the replay forward of the LoRA mode")
# dflash_lora_tts replays the same block forward to train its adapters. With
# freshly initialized adapters (B = 0) its delta is exactly zero, so the two
# forwards must agree, including the windowed prefix gather.
from sglang.srt.speculative.dflash_lora_tts import (  # noqa: E402
    DFlashLoRATTSConfig,
    DFlashLoRATTSState,
    _ReplayCapture,
)

model = make_model(seed=3)
for prefix_len in (3, WINDOW + 3, 20):
    pool, req_to_token, _, _ = make_cache(model, [prefix_len], seed=3)
    embeds, positions = make_inputs([prefix_len], seed=3)
    out = DFlashBlockForward(
        draft_model=model, token_to_kv_pool=pool, block_size=BLOCK
    )(
        input_embeds=embeds,
        positions=positions,
        prefix_lens=[prefix_len],
        req_pool_indices=[0],
        req_to_token=req_to_token,
    )
    lora_state = DFlashLoRATTSState(
        config=DFlashLoRATTSConfig(async_step=False, offload_state=False),
        draft_model=model,
        token_to_kv_pool=pool,
        block_size=BLOCK,
        device=torch.device("cpu"),
    )
    capture = _ReplayCapture(
        req_pool_idx=0,
        row=0,
        input_embeds=embeds[0],
        positions=positions[0],
        prefix_locs=req_to_token[0, :prefix_len].to(torch.int64),
    )
    with torch.no_grad():
        replay = lora_state._replay_block_hidden(
            lora_state.get(0).flats,
            embeds[0],
            positions[0],
            lora_state._gather_prefix_kv_many([capture])[0],
        )
    err = float((out[0] - replay).abs().max())
    check(
        f"prefix {prefix_len:2d}: same hidden states", err < 1e-6, f"max|err|={err:.2e}"
    )

print("\n[9] startup validation")


def raises(fn):
    try:
        fn()
    except (RuntimeError, ValueError) as e:
        return str(e)
    return None


model9 = make_model(seed=3)
pool9, _, _, _ = make_cache(model9, [3], seed=3)


def rejected(name, expected, **kwargs):
    kwargs = {"draft_model": model9, "token_to_kv_pool": pool9, "block_size": BLOCK, **kwargs}
    msg = raises(lambda: DFlashBlockForward(**kwargs))
    check(name, msg is not None and expected in msg, (msg or "no error")[:70])


rejected(
    "a KV pool that is not MHATokenToKVPool is rejected",
    "requires MHATokenToKVPool",
    token_to_kv_pool=object(),
)
rejected("block_size 1 is rejected", "requires block_size >= 2", block_size=1)
rejected(
    "a window shorter than the block is rejected",
    f"sliding window ({WINDOW - 1}) >= block_size - 1 ({WINDOW + 1})",
    block_size=WINDOW + 2,
)
rejected(
    "replicated KV heads under TP are rejected",
    "requires at least tp_size key/value heads",
    tp_size=4,
)
check(
    "the same arguments without the fault are accepted",
    raises(lambda: DFlashBlockForward(draft_model=model9, token_to_kv_pool=pool9, block_size=BLOCK))
    is None,
)

print("\n[10] worker wiring: the real worker methods run against a stubbed target")
# `forward_batch_generation`, `_prepare_for_speculative_decoding`,
# `_tts_draft_forward`, `_run_tts_post_verify` and `_init_full_weight_tts` are
# the real methods. Stubbed: the target forward, the verification (a subclass
# of DFlashVerifyInput that biases the logits in place, as logit bias does),
# the draft KV append and, for the rounds that only need the post-verify
# path, the block-building part of `_prepare_for_speculative_decoding` (it
# stands in with the real `_tts_draft_forward` call).
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.speculative import dflash_worker as dflash_worker_module  # noqa: E402
from sglang.srt.speculative.dflash_info import (  # noqa: E402
    DFlashDraftInput,
    DFlashVerifyInput,
)

VERIFY_BIAS = -7.0


class StopRound(Exception):
    """Raised by a stub to end a round at a chosen point."""


class FakeVerify(DFlashVerifyInput):
    def verify(self, *, batch, logits_output, page_size):
        logits_output.next_token_logits.fill_(VERIFY_BIAS)  # in place
        return (
            torch.zeros(1, dtype=torch.int64),
            torch.full((1,), BLOCK, dtype=torch.int32),
            torch.zeros(BLOCK, HIDDEN),
            [BLOCK - 1],
        )


def make_round_worker(optimizer_dtype, freq=2, lm_head=None):
    """A worker that can run `forward_batch_generation` on a stubbed target."""
    worker, model = make_worker(optimizer_dtype, freq=freq, lm_head=lm_head)
    live = {"logits": None}

    def target_forward(model_worker_batch, **kwargs):
        if hasattr(model_worker_batch, "extend_seq_lens"):  # prefill
            return SimpleNamespace(
                logits_output=SimpleNamespace(hidden_states=torch.zeros(5, HIDDEN)),
                next_token_ids=torch.tensor([3]),
                can_run_cuda_graph=False,
            )
        return SimpleNamespace(
            logits_output=SimpleNamespace(next_token_logits=live["logits"]),
            can_run_cuda_graph=False,
        )

    worker.target_worker.forward_batch_generation = target_forward
    worker.target_worker.model_runner.attn_backend = object()
    worker.__dict__.update(
        tp_rank=0,
        page_size=1,
        device="cpu",
        use_compact_draft_cache=False,
        _lora_tts_state=None,
        _logged_first_verify=True,
        _accept_log=None,
        _append_target_hidden_to_draft_kv=lambda *args, **kwargs: None,
        _live=live,
    )
    return worker, model


def decode_round(worker, reqs, logits):
    """One decode round through the real `forward_batch_generation`.

    The target writes a fresh copy of `logits` as its verify output, and the
    stubbed verification then overwrites that tensor in place. Returns the
    target's output tensor (after verification).
    """
    live = logits.clone()
    worker._live["logits"] = live
    verify = FakeVerify(
        draft_token=torch.zeros(BLOCK, dtype=torch.int64),
        positions=torch.zeros(BLOCK, dtype=torch.int64),
        draft_token_num=BLOCK,
    )
    model_worker_batch = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY, spec_info=verify
    )
    batch = SimpleNamespace(
        reqs=list(reqs),
        batch_size=lambda: len(reqs),
        forward_mode=ForwardMode.DECODE,
        is_extend_in_batch=False,
        spec_info=DFlashDraftInput(
            bonus_tokens=torch.zeros(1, dtype=torch.int64),
            target_hidden=torch.zeros(1, HIDDEN),
            ctx_lens=torch.ones(1, dtype=torch.int32),
            draft_seq_lens=torch.zeros(1, dtype=torch.int32),
        ),
        get_model_worker_batch=lambda: model_worker_batch,
    )

    def prepare(round_batch, draft_input):
        draft(worker, round_batch)  # the real `_tts_draft_forward` call
        round_batch.forward_mode = ForwardMode.TARGET_VERIFY

    worker._prepare_for_speculative_decoding = prepare
    with DynamicGradMode():
        worker.forward_batch_generation(batch)
    return live


def prefill(worker, reqs):
    """A prefill (extend) batch through the real `forward_batch_generation`."""
    model_worker_batch = SimpleNamespace(
        capture_hidden_mode=None,
        extend_seq_lens=torch.tensor([5], dtype=torch.int32),
        extend_prefix_lens=torch.tensor([0], dtype=torch.int32),
    )
    batch = SimpleNamespace(
        reqs=list(reqs),
        forward_mode=ForwardMode.EXTEND,
        is_extend_in_batch=False,
        req_pool_indices=torch.tensor([r.req_pool_idx for r in reqs]),
        get_model_worker_batch=lambda: model_worker_batch,
        spec_info=None,
    )
    with DynamicGradMode():
        worker.forward_batch_generation(batch)
    return batch


def spy(obj, name):
    """Record the arguments of every call to obj.<name>, then call through."""
    original, calls = getattr(obj, name), []

    def wrapper(*args, **kwargs):
        calls.append(
            (
                [a.detach().clone() if torch.is_tensor(a) else a for a in args],
                {
                    k: v.detach().clone() if torch.is_tensor(v) else v
                    for k, v in kwargs.items()
                },
            )
        )
        return original(*args, **kwargs)

    setattr(obj, name, wrapper)
    return calls


for dtype in ("float32", "bfloat16"):
    # --- decode round: snapshot, post-verify call, stride ------------------
    worker, model = make_round_worker(dtype)
    state = worker._tts_state
    pretrained = weights_of(model)
    step_calls = spy(state, "step")
    post_calls = spy(worker, "_run_tts_post_verify")
    live0 = decode_round(worker, [Req(0)], W_LOGITS)
    check(
        f"{dtype}: decode round: the verification biased the target logits in place",
        bool((live0 == VERIFY_BIAS).all()),
    )
    check(
        f"{dtype}: update round: the post-verify hook ran and applied the update",
        len(post_calls) == 1
        and state.update_ct == 1
        and state.round_idx == 1
        and worker._tts_pending_hidden is None
        and not unchanged(model, pretrained),
    )
    check(
        f"{dtype}: update round: the update gets the logits as they were before verification",
        len(step_calls) == 1
        and torch.equal(step_calls[0][1]["target_logits"], W_LOGITS)
        and not bool((step_calls[0][1]["target_logits"] == VERIFY_BIAS).any()),
    )
    check(
        f"{dtype}: update round: the logits kept for the update are a copy of the target's",
        post_calls[0][0][1] is not None
        and post_calls[0][0][1].data_ptr() != live0.data_ptr(),
    )
    weights_after_update = weights_of(model)
    decode_round(worker, [Req(0)], W_LOGITS)
    check(
        f"{dtype}: other round: no logits are kept, no update, the stride advances",
        len(post_calls) == 2
        and post_calls[1][0][1] is None
        and len(step_calls) == 1
        and state.update_ct == 1
        and state.round_idx == 2
        and unchanged(model, weights_after_update),
    )
    decode_round(worker, [Req(0)], W_LOGITS)
    check(
        f"{dtype}: next update round: logits kept again, second update applied",
        len(post_calls) == 3
        and post_calls[2][0][1] is not None
        and state.update_ct == 2
        and state.round_idx == 3,
    )

    # --- prefill: reset ----------------------------------------------------
    worker._tts_pending_hidden = torch.zeros(1)  # a stale block
    batch = prefill(worker, [Req(0)])
    check(
        f"{dtype}: prefill restores the pretrained draft weights (and masters)",
        unchanged(model, pretrained) and masters_match(state, pretrained),
    )
    check(
        f"{dtype}: prefill clears the optimizer state and restarts the stride",
        optimizer_is_fresh(state)
        and state.round_idx == 0
        and not state._dirty,
    )
    check(
        f"{dtype}: prefill drops a stale pending block",
        worker._tts_pending_hidden is None,
    )
    check(
        f"{dtype}: prefill still writes the draft input for the new request",
        isinstance(batch.spec_info, DFlashDraftInput),
    )

    # --- fence before the draft weights are read ---------------------------
    worker, model = make_round_worker(dtype)
    state = worker._tts_state
    pretrained = weights_of(model)
    bad_target_logits = W_LOGITS.clone()
    bad_target_logits[0, 0] = float("nan")
    draft(worker, one)
    with DynamicGradMode():
        worker._run_tts_post_verify(one, bad_target_logits)
    poisoned = not unchanged(model, pretrained)
    seen = {}

    def stop_at_append(*args, **kwargs):
        seen["nonfinite_ct"] = state.nonfinite_ct
        seen["restored"] = unchanged(model, pretrained) and masters_match(
            state, pretrained
        )
        raise StopRound

    worker._append_target_hidden_to_draft_kv = stop_at_append
    decode_batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        has_grammar=False,
        sampling_info=None,
        batch_size=lambda: 1,
        reqs=[Req(0)],
    )
    try:
        worker._prepare_for_speculative_decoding(decode_batch, None)
        stopped = False
    except StopRound:
        stopped = True
    check(
        f"{dtype}: a decode round waits for the previous update before it reads the draft weights",
        poisoned
        and stopped
        and seen.get("nonfinite_ct") == 1
        and seen.get("restored") is True,
        str(seen),
    )

    # --- request slots, detach, lm_head slice -------------------------------
    worker, model = make_round_worker(dtype)
    block_forward = worker._tts_forward
    out = draft(worker, make_batch(Req(1)))
    r2t = worker.draft_model_runner.req_to_token_pool.req_to_token
    embeds10, positions10 = make_inputs([W_PREFIX], dtype=torch.bfloat16)
    by_slot = {
        slot: block_forward.forward(
            input_embeds=embeds10,
            positions=positions10,
            prefix_lens=[W_PREFIX],
            req_pool_indices=[slot],
            req_to_token=r2t,
        )
        for slot in (0, 1)
    }
    check(
        f"{dtype}: fixture: the two request slots hold different prefixes",
        not torch.allclose(by_slot[0], by_slot[1], atol=1e-3),
    )
    check(
        f"{dtype}: the block forward gets the request's slot, not its row in the batch",
        block_forward.slots == [[1]]
        and torch.allclose(worker._tts_pending_hidden.detach(), by_slot[1], atol=1e-6)
        and not torch.allclose(worker._tts_pending_hidden.detach(), by_slot[0], atol=1e-3),
        str(block_forward.slots),
    )
    worker, model = make_round_worker(dtype)
    block_embeds, block_positions = make_inputs([W_PREFIX], dtype=torch.bfloat16)
    with torch.enable_grad():  # a caller that has autograd on
        out = worker._tts_draft_forward(
            batch=one,
            block_input_embeds=block_embeds,
            positions_2d=block_positions,
            draft_prefix_lens_cpu=torch.tensor([W_PREFIX], dtype=torch.int32),
        )
    check(
        f"{dtype}: the hidden states handed on to drafting are detached from the update's graph",
        out is not None
        and out.hidden_states.grad_fn is None
        and not out.hidden_states.requires_grad
        and worker._tts_pending_hidden.requires_grad,
    )

    padded_head = SimpleNamespace(
        weight=torch.cat(
            [
                W_HEAD.weight.detach(),
                torch.randn(5, HIDDEN, generator=torch.Generator().manual_seed(9)).to(
                    torch.bfloat16
                ),
            ]
        )
    )
    worker, model = make_round_worker(dtype, lm_head=padded_head)
    state = worker._tts_state
    pretrained = weights_of(model)
    step_calls = spy(state, "step")
    decode_round(worker, [Req(0)], W_LOGITS)
    check(
        f"{dtype}: padded lm_head: the update scores only the vocabulary rows",
        len(step_calls) == 1
        and step_calls[0][1]["lm_head_weight"].shape == (VOCAB, HIDDEN)
        and state.update_ct == 1
        and state._failure_streak == 0
        and not unchanged(model, pretrained),
    )

    # --- stale pending blocks ----------------------------------------------
    stale = torch.zeros(1)
    worker, model = make_round_worker(dtype)
    worker._tts_state.advance()  # round 1: no update fires
    worker._tts_pending_hidden = stale
    out = draft(worker, one)
    check(
        f"{dtype}: a round without an update drops a stale pending block",
        out is None and worker._tts_pending_hidden is None,
    )
    worker, model = make_round_worker(dtype)
    worker._tts_pending_hidden = stale
    out = draft(worker, make_batch(Req(0), Req(1)))
    check(
        f"{dtype}: a two-request round drops a stale pending block",
        out is None and worker._tts_pending_hidden is None,
    )
    worker, model = make_round_worker(dtype)
    draft(worker, one)
    worker._run_tts_post_verify(make_batch(Req(0, finished=True)), W_LOGITS)
    check(
        f"{dtype}: post-verify of a finished request consumes the pending block",
        worker._tts_pending_hidden is None,
    )
    worker, model = make_round_worker(dtype)
    worker._tts_pending_hidden = stale
    worker._run_tts_post_verify(one, None)
    check(
        f"{dtype}: post-verify without target logits consumes the pending block",
        worker._tts_pending_hidden is None,
    )

    # --- several requests: the update is skipped only when all have finished
    # A batch larger than one cannot reach the update through the validated
    # configuration (max_running_requests is 1); this pins the hook's own rule.
    worker, model = make_round_worker(dtype)
    state = worker._tts_state
    summaries = spy(state, "log_request_summary")
    draft(worker, one)
    with DynamicGradMode():
        worker._run_tts_post_verify(
            make_batch(Req(0, finished=True), Req(1, finished=False)), W_LOGITS
        )
    check(
        f"{dtype}: one of two requests still running: the update is applied, no summary yet",
        state.update_ct == 1 and len(summaries) == 0,
    )
    state.round_idx = 0  # an update round again
    draft(worker, one)
    worker._run_tts_post_verify(
        make_batch(Req(0, finished=True), Req(1, finished=True)), W_LOGITS
    )
    check(
        f"{dtype}: all requests finished: the update is skipped and the summary is logged",
        state.update_ct == 1 and len(summaries) == 1,
    )

# --- _init_full_weight_tts -------------------------------------------------
class TargetStub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = torch.nn.Linear(HIDDEN, VOCAB, bias=False).to(torch.bfloat16)


model10 = make_model(dtype=torch.bfloat16)
pool10, r2t10, _, _ = make_cache(model10, [W_PREFIX], dtype=torch.bfloat16)
init_worker = object.__new__(DFlashWorker)
init_worker.__dict__.update(
    target_worker=SimpleNamespace(
        model_runner=SimpleNamespace(
            model=TargetStub(), model_config=SimpleNamespace(vocab_size=VOCAB)
        )
    ),
    draft_model=model10,
    draft_model_runner=SimpleNamespace(
        token_to_kv_pool=pool10, graph_runner=None, req_to_token_pool=None
    ),
    block_size=BLOCK,
    device="cpu",
    tp_rank=0,
    # What a CUDA worker has built before full-weight TTS is initialized.
    _use_fused_kv_materialize=True,
    _fused_kv_helper=object(),
)
real_get_tp_group = dflash_worker_module.get_tp_group
dflash_worker_module.get_tp_group = lambda: SimpleNamespace(world_size=1)
try:
    init_worker._init_full_weight_tts(DFlashTTSConfig(enable=True, async_step=False))
finally:
    dflash_worker_module.get_tp_group = real_get_tp_group
check(
    "init: the fused KV materialization is switched off (it stacks stale copies of K/V weights)",
    init_worker._use_fused_kv_materialize is False
    and init_worker._fused_kv_helper is None,
)
check(
    "init: the block forward, the state and the lm_head rows are set up",
    isinstance(init_worker._tts_forward, DFlashBlockForward)
    and isinstance(init_worker._tts_state, DFlashTTSState)
    and init_worker._tts_lm_head_rows == VOCAB,
)

print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
