"""Checks that full-mode TTS updates every parameter of a real EAGLE-3 draft.

On AngelSlim/Qwen3-8B_eagle3 from the Hugging Face cache this checks four
things:

  A. coverage   — the TTS optimizer and snapshot include every named parameter
  B. gradients  — after a surrogate draft forward, every parameter has a
                  finite, non-zero gradient
  C. update     — after one TTS step, every parameter's value has changed
  D. reset      — reset() restores every parameter exactly

The surrogate forward walks the draft's module order (fc_norm -> fc ->
decoder layer -> norm -> lm_head) rather than driving a ForwardBatch, so it
runs without the target model. Attention is replaced by a simple mix of q, k
and v, which still routes gradients through q/k/v and o_proj. It includes `fc`,
which a real update does not reach, so that the optimizer's handling of every
parameter is covered.

A CUDA device must be visible, because `ServerArgs` requires an accelerator;
`TTS_TEST_DEV=cpu` keeps the tensors on CPU.

  python test/registered/spec/eagle_tts/test_tts_param_coverage.py
"""
import glob
import os
import sys

import torch

_DEV = os.environ.get("TTS_TEST_DEV", "cuda")
HUB = os.path.expanduser("~/.cache/huggingface/hub")
DRAFT = "AngelSlim--Qwen3-8B_eagle3"

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29593")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")

from sglang.srt.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.speculative.eagle_tts import (
    EagleTTSConfig,
    EagleTTSState,
    tts_grad_mode,
)
from sglang.srt.utils.hf_transformers_utils import get_config

FAILS = []


def check(name, cond, extra=""):
    print(f"    {'PASS' if cond else 'FAIL'}  {name}{('  — ' + extra) if extra else ''}")
    if not cond:
        FAILS.append(name)


def unwrap(x):
    return x[0] if isinstance(x, tuple) else x


def surrogate_forward(model, cfg, n_pos=8):
    """Walk the draft's modules in their real order, touching every parameter."""
    m = model.model
    dev = model.lm_head.weight.device
    dt = model.lm_head.weight.dtype
    hidden = cfg.hidden_size
    n_aux = getattr(m, "num_aux_hidden_states", 3)

    input_ids = torch.randint(0, cfg.vocab_size, (n_pos,), device=dev)
    aux = torch.randn(n_pos, m.hidden_size_in * n_aux, device=dev, dtype=dt)

    embeds = m.embed_tokens(input_ids)
    h = aux
    if m.fc_norm is not None:
        chunks = h.chunk(n_aux, dim=-1)
        h = torch.cat([n(c) for n, c in zip(m.fc_norm, chunks)], dim=-1)
    h = m.fc(h)

    layer = m.layers[0]
    # is_input_layer branch, honouring norm_before_residual
    if getattr(layer, "norm_before_residual", False):
        h = layer.hidden_norm(h)
        residual = h
    else:
        residual = h
        h = layer.hidden_norm(h)
    e = layer.input_layernorm(embeds)
    h = torch.cat([e, h], dim=-1)

    # Attention without the kernel: qkv -> toy mix -> o_proj.
    qkv = unwrap(layer.self_attn.qkv_proj(h))
    q_size = layer.self_attn.q_size
    kv_size = layer.self_attn.kv_size
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    # Route grad through k and v as well as q, without a real attention op.
    groups = q_size // kv_size
    mix = q + k.repeat(1, groups) + v.repeat(1, groups)
    attn_out = unwrap(layer.self_attn.o_proj(mix))

    h, residual = layer.post_attention_layernorm(attn_out, residual)
    h = layer.mlp(h)

    h = unwrap(m.norm(h, residual))
    # ParallelLMHead refuses to be called directly; the logits processor uses its
    # weight, so do the same matmul here.
    return torch.matmul(h, model.lm_head.weight.T).float()


def run_one():
    path = glob.glob(f"{HUB}/models--{DRAFT}/snapshots/*")[0]

    cfg = get_config(path, trust_remote_code=False)
    from sglang.srt.models.llama_eagle3 import LlamaForCausalLMEagle3

    with torch.device(_DEV):
        model = LlamaForCausalLMEagle3(cfg)
    ckpt = torch.load(
        os.path.join(path, "pytorch_model.bin"), map_location="cpu", weights_only=True
    )
    model.load_weights(list(ckpt.items()))
    model = model.to(torch.bfloat16)

    # Some drafts do not ship embed_tokens (or lm_head); at runtime the worker
    # binds them to the target's (EagleDraftWorker.init_lm_head -> set_embed).
    # Seed whatever the checkpoint left all-zero. Otherwise input_layernorm
    # would see an all-zero input and its weight gradient would be exactly
    # zero, which is an artifact of this test rather than a TTS failure.
    with torch.no_grad():
        for attr, mod in (("embed_tokens", model.model.embed_tokens),
                          ("lm_head", model.lm_head)):
            w = mod.weight
            if not any(attr in k for k in ckpt) and (
                not torch.isfinite(w).all() or w.abs().sum() == 0
            ):
                mod.weight.normal_(0, 0.02)
                print(f"    (seeded {attr} — supplied by the target at runtime)")

    tts_cfg = EagleTTSConfig(enable=True, lr=1e-2, freq=1, async_step=False)
    state = EagleTTSState(draft_model=model, config=tts_cfg, device=_DEV)

    all_named = dict(model.named_parameters())
    n_all = len(all_named)

    # ---- A. coverage
    tracked = {id(p) for p in state._params}
    missing = [n for n, p in all_named.items() if id(p) not in tracked]
    check(f"optimizer covers all {n_all} params", not missing, f"missing={missing}")
    snap_missing = [n for n in all_named if n not in state._snapshot]
    check("snapshot covers all params", not snap_missing, f"missing={snap_missing}")
    if state._master is not None:
        check("fp32 master has one entry per param",
              len(state._master) == len(state._params))

    # ---- B/C. gradients + update
    before = {n: p.detach().clone() for n, p in all_named.items()}
    with tts_grad_mode():
        draft_logits = surrogate_forward(model, cfg)
    target_logits = torch.randn(
        draft_logits.shape[0], cfg.vocab_size, device=_DEV, dtype=torch.float32
    )
    hot = getattr(model, "hot_token_id", None)
    if hot is not None:
        hot = hot.to(_DEV)

    check("draft logits carry grad", draft_logits.requires_grad)
    state.step(draft_logits=draft_logits, target_logits=target_logits, hot_token_id=hot)

    no_grad, bad_grad, unmoved = [], [], []
    for n, p in model.named_parameters():
        g = p.grad
        if g is None:
            no_grad.append(n)
        elif not torch.isfinite(g).all() or float(g.abs().sum()) == 0.0:
            bad_grad.append(n)
        if torch.equal(before[n], p.detach()):
            unmoved.append(n)

    check("every param received a gradient", not no_grad, f"none for={no_grad}")
    check("every gradient is finite and non-zero", not bad_grad, f"bad={bad_grad}")
    check("every param value changed", not unmoved, f"unchanged={unmoved}")
    print(f"    ({n_all} params, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M elements)")

    # ---- D. reset
    state.reset()
    not_restored = [
        n for n, p in model.named_parameters()
        if not torch.equal(before[n], p.detach())
    ]
    check("reset restores every param", not not_restored, f"drifted={not_restored}")

    del model, state
    (torch.cuda.empty_cache() if _DEV=="cuda" else None)


def main():
    set_global_server_args_for_scheduler(
        ServerArgs(model_path=glob.glob(f"{HUB}/models--{DRAFT}/snapshots/*")[0])
    )
    (torch.cuda.set_device(0) if _DEV=="cuda" else None)
    init_distributed_environment(
        world_size=1, rank=0, local_rank=0,
        distributed_init_method="tcp://127.0.0.1:29593", backend="nccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=1)

    run_one()

    print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + '; '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
