"""LoRA-mode TTS adapters on a real EAGLE-3 draft.

Properties checked:

  A. identity at init  — B=0, so every delta is exactly zero: with TTS enabled
                         and no update applied, the draft is bit-identical to
                         the pretrained one.
  B. base frozen       — apart from post_attention_layernorm and the final
                         norm, trained in unpooled mode, no base weight requires grad or changes
                         across several optimizer steps. This is what keeps the
                         target model unmodified without cloning its aliased
                         tensors.
  C. gradients         — every adapter tensor gets a finite gradient and
                         changes.
  D. reset             — restores the pretrained function exactly (A back to
                         its template, B back to zero), with no weight
                         snapshot.
  E. attention gating  — a target set confined to at/above the attention output
                         reports that it needs no differentiable attention;
                         adding `q`, `k`, `v` or `fc` changes that.
  F. wiring            — the fused qkv_proj is wrapped once even when several
                         of q/k/v are selected.

Needs a GPU and AngelSlim/Qwen3-8B_eagle3 in the Hugging Face cache:

  HF_HUB_OFFLINE=1 python test/registered/spec/eagle_tts/test_lora_adapters.py
"""
import glob
import os
import sys

import torch

_DEV = os.environ.get("TTS_TEST_DEV", "cuda")
HUB = os.path.expanduser("~/.cache/huggingface/hub")
DRAFT = "AngelSlim--Qwen3-8B_eagle3"
FAILS = []

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29594")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")

from sglang.srt.distributed import (  # noqa: E402
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.server_args import (  # noqa: E402
    ServerArgs,
    set_global_server_args_for_scheduler,
)


def check(name, cond, extra=""):
    print(f"    {'PASS' if cond else 'FAIL'}  {name}{(' — ' + extra) if extra else ''}")
    if not cond:
        FAILS.append(name)


def build(path):
    from sglang.srt.models.llama_eagle3 import LlamaForCausalLMEagle3
    from sglang.srt.utils.hf_transformers.config import get_config

    cfg = get_config(path, trust_remote_code=False)
    with torch.device(_DEV):
        model = LlamaForCausalLMEagle3(cfg)
    ckpt = torch.load(
        os.path.join(path, "pytorch_model.bin"), map_location="cpu", weights_only=True
    )
    model.load_weights(list(ckpt.items()))
    model = model.to(torch.bfloat16)
    # The draft ships no embed_tokens; the worker rebinds it to the
    # target's at runtime. Seed whatever the checkpoint left unset.
    #
    # "Unset" is tested with isfinite as well as `abs().sum() == 0`: under
    # torch.use_deterministic_algorithms, torch.empty returns NaN rather than
    # zeros, and `NaN == 0` is False.
    torch.manual_seed(1234)
    with torch.no_grad():
        for mod in (model.model.embed_tokens, model.lm_head):
            w = mod.weight
            if not torch.isfinite(w).all() or w.abs().sum() == 0:
                w.normal_(0, 0.02)
    unset = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    assert not unset, f"uninitialised draft weights: {unset}"
    return model, cfg


def surrogate(model, cfg, n_pos=6):
    """Forward through the draft's real module order, without a ForwardBatch.

    Attention is a plain GQA-style mix rather than a real attention op, so a
    gradient still reaches q/k/v and o_proj. That is enough to exercise every
    adapter; the differentiable attention path itself is not covered here.
    """
    m = model.model
    layer = m.layers[0]
    n_aux = m.num_aux_hidden_states
    dt = model.lm_head.weight.dtype
    dev = model.lm_head.weight.device

    unwrap = lambda r: r[0] if isinstance(r, tuple) else r

    ids = torch.randint(0, cfg.vocab_size, (n_pos,), device=dev)
    aux = torch.randn(n_pos, m.hidden_size_in * n_aux, device=dev, dtype=dt)
    embeds = m.embed_tokens(ids)

    h = aux
    if m.fc_norm is not None:
        chunks = h.chunk(m.num_aux_hidden_states, dim=-1)
        h = torch.cat([n(c) for n, c in zip(m.fc_norm, chunks)], dim=-1)
    h = m.fc(h)

    residual = h
    h = layer.hidden_norm(h)
    e = layer.input_layernorm(embeds)
    h = torch.cat([e, h], dim=-1)

    qkv = unwrap(layer.self_attn.qkv_proj(h))
    q_size, kv_size = layer.self_attn.q_size, layer.self_attn.kv_size
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    groups = q_size // kv_size
    mix = q + k.repeat(1, groups) + v.repeat(1, groups)
    attn_out = unwrap(layer.self_attn.o_proj(mix))

    h, residual = layer.post_attention_layernorm(attn_out, residual)
    h = layer.mlp(h)
    h = unwrap(m.norm(h, residual))

    # The same dispatch LogitsProcessor._compute_lm_head performs: the
    # set_lora/apply_lora sentinels route an adapted head through its forward,
    # and a bare head through the matmul. Calling the matmul unconditionally
    # would bypass the lm_head adapter and report it as untrained.
    head = model.lm_head
    if hasattr(head, "set_lora") and hasattr(head, "apply_lora"):
        return head(h).float()
    return torch.matmul(h.to(head.weight.dtype), head.weight.T).float()


def uninstall_adapters(draft_model):
    """Restore original forwards on every adapted module. Returns the count.

    The wrappers are instance-level overrides, so removing them from the
    module's `__dict__` falls back to the class method.
    """
    n = 0
    for module in draft_model.modules():
        if getattr(module, "_tts_lora", None) is None:
            continue
        for attr in ("forward", "_tts_lora", "set_lora", "apply_lora"):
            module.__dict__.pop(attr, None)
        n += 1
    return n


def run_one(targets):
    from sglang.srt.speculative.eagle_lora_tts import (
        EagleLoRAAdapterSet,
    )
    from sglang.srt.speculative.eagle_tts import (
        EagleTTSConfig,
        EagleTTSState,
        tts_grad_mode,
    )

    print(f"\n=== targets={','.join(targets)} ===")
    model, cfg = build(glob.glob(f"{HUB}/models--{DRAFT}/snapshots/*")[0])

    torch.manual_seed(0)
    base_logits = surrogate(model, cfg).detach().clone()
    base_before = {n: p.detach().clone() for n, p in model.named_parameters()}

    aset = EagleLoRAAdapterSet(
        draft_model=model, targets=targets, rank=16, alpha=32.0, tp_size=1, tp_rank=0
    )

    # --- A. identity at init -------------------------------------------------
    torch.manual_seed(0)
    lora_logits = surrogate(model, cfg).detach()
    check(
        "delta is exactly zero at init (B=0)",
        torch.equal(base_logits, lora_logits),
        f"max|diff|={(base_logits - lora_logits).abs().max().item():.3e}",
    )
    check(
        "every requested target resolved",
        len(aset.adapters) == len(targets),
        f"{sorted(aset.adapters)}",
    )
    check(
        "fused qkv_proj wrapped once",
        len(getattr(model.model.layers[0].self_attn.qkv_proj, "_tts_lora", []))
        == len([t for t in targets if t in ("q", "k", "v")]),
    )

    # --- B/C. train ----------------------------------------------------------
    params = aset.parameters()
    check("no base weight requires grad", not any(p.requires_grad for p in base_before_params(model, aset)))

    state = EagleTTSState(
        draft_model=model,
        config=EagleTTSConfig(enable=True, lr=1e-2, freq=1, async_step=False),
        device=_DEV,
        adapters=aset,
    )
    check("state trains only adapter tensors", len(state._params) == len(params))
    check("no full-weight snapshot was taken", state._snapshot == {})
    check("nothing was privatized", state._privatized == [])

    before = [p.detach().clone() for p in params]
    for _ in range(3):
        torch.manual_seed(1)
        # The forward must run inside tts_grad_mode: sglang's fused RMSNorm and
        # SiluAndMul have no autograd registration, so outside it the tape ends
        # at the first one and only adapters downstream of the last fused op
        # would receive a gradient. No model argument means no attention patch,
        # which is all this surrogate needs (it has no real attention op).
        with tts_grad_mode(), torch.enable_grad():
            dl = surrogate(model, cfg)
            tl = torch.randn_like(dl)
            state.step(draft_logits=dl, target_logits=tl)

    no_grad = [n for n, p in zip(aset.parameter_names(), params) if p.grad is None]
    check("every adapter tensor got a gradient", not no_grad, f"none for={no_grad}")
    bad = [
        n
        for n, p in zip(aset.parameter_names(), params)
        if p.grad is not None and not torch.isfinite(p.grad).all()
    ]
    check("gradients are finite", not bad, f"bad={bad}")
    moved = [
        n
        for n, p, b in zip(aset.parameter_names(), params, before)
        if not torch.equal(p, b)
    ]
    check(
        "every adapter tensor moved",
        len(moved) == len(params),
        f"{len(moved)}/{len(params)}",
    )
    check(
        "the draft function actually changed",
        not torch.equal(base_logits, surrogate_seeded(model, cfg)),
    )

    # --- B. base frozen ------------------------------------------------------
    drifted = [
        n
        for n, p in model.named_parameters()
        if n in base_before and not torch.equal(p, base_before[n])
    ]
    # The two post-attention norms are updated in full (unpooled mode).
    from sglang.srt.speculative.eagle_lora_tts import POST_ATTENTION_NORMS

    unexpected = [n for n in drifted if n not in POST_ATTENTION_NORMS]
    check(
        "every base weight is bit-identical after training",
        not unexpected,
        f"drifted={unexpected}",
    )
    check(
        "the post-attention norms did train",
        sorted(drifted) == sorted(n for n in POST_ATTENTION_NORMS if n in base_before),
        f"{drifted}",
    )

    # --- D. reset ------------------------------------------------------------
    state.reset()
    check(
        "reset restores the pretrained function exactly",
        torch.equal(base_logits, surrogate_seeded(model, cfg)),
    )
    zeroed = all(p.abs().sum().item() == 0 for n, p in zip(aset.parameter_names(), aset.parameters()) if n.endswith("lora_b"))
    check("reset zeroed every B", zeroed)

    print(f"    ({len(aset.adapters)} adapters, {aset.num_parameters()/1e6:.3f}M params)")
    uninstall_adapters(model)
    check("uninstall restores the base forward", _forward_is_clean(model))

    # Free the draft before the next target set: three sets run back to back,
    # each with its own copy of the draft.
    del model, aset, state, base_before, base_logits
    if _DEV == "cuda":
        torch.cuda.empty_cache()


def base_before_params(model, aset):
    adapter_ids = {id(p) for p in aset.parameters()}
    return [p for p in model.parameters() if id(p) not in adapter_ids]


def surrogate_seeded(model, cfg):
    torch.manual_seed(0)
    return surrogate(model, cfg).detach()


def _forward_is_clean(model):
    return not any(getattr(m, "_tts_lora", None) is not None for m in model.modules())


def main():
    set_global_server_args_for_scheduler(
        ServerArgs(model_path=glob.glob(f"{HUB}/models--{DRAFT}/snapshots/*")[0])
    )
    if _DEV == "cuda":
        torch.cuda.set_device(0)
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="tcp://127.0.0.1:29594",
        backend="nccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=1)

    from sglang.srt.speculative.eagle_lora_tts import (
        DEFAULT_LORA_TARGETS,
        lora_needs_differentiable_attention,
        parse_lora_targets,
    )

    print("=== E. attention gating ===")
    check(
        "default set (with q) needs differentiable attention",
        lora_needs_differentiable_attention(parse_lora_targets(DEFAULT_LORA_TARGETS)),
    )
    check(
        "o,gate_up,down,lm_head does NOT",
        not lora_needs_differentiable_attention(("o", "gate_up", "down", "lm_head")),
    )
    check(
        "adding k or v or fc flips it",
        all(
            lora_needs_differentiable_attention(("o", t))
            for t in ("k", "v", "fc")
        ),
    )

    run_one(parse_lora_targets(DEFAULT_LORA_TARGETS))
    run_one(("o", "gate_up", "down", "lm_head"))
    run_one(("q", "k", "v", "o"))  # fused-module sharing

    print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
