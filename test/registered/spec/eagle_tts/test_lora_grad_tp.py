"""Checks that EAGLE-3 LoRA-mode TTS gradients are correct under tensor parallelism.

sglang is inference-only, so its tensor-parallel collectives have no backward.
TTS adds one, in two parts:

  g (all-reduce -> identity, all-gather -> slice):
    eagle_tts._DifferentiableAllReduce / _DifferentiableAllGather
  f (column-parallel input: identity forward, all-reduce backward):
    eagle_tts._patch_column_parallel_input_grad + all_reduce_input_grad

Without `f`, the `dL/d(hidden)` coming out of the vocab-parallel lm_head is
only this rank's vocab shard's contribution and is never summed, so every
gradient upstream is a partial sum and each rank optimizes a different
objective. Separately, `dL/dA` for a column-parallel adapter depends on the
local `B` shard and is per-rank by construction;
`EagleTTSState._reduce_partial_grads` sums those.

The property asserted is that every tensor that is replicated across ranks
holds an identical gradient, and an identical value after two optimizer steps,
on every rank.

Needs one GPU per rank and AngelSlim/Qwen3-8B_eagle3 in the Hugging Face
cache; each rank loads the draft. Run with torchrun:

  HF_HUB_OFFLINE=1 torchrun --nproc_per_node=2 \
      test/registered/spec/eagle_tts/test_lora_grad_tp.py

Options:
  --negative-control    disable the TP gradient corrections and pass only if
                        the ranks are observed to diverge
  --save-ref PATH       (with --nproc_per_node=1) save the tp=1 gradients
  --compare-ref PATH    compare gradient magnitudes against a saved tp=1 run

At tp=1 the test reduces to a plain gradient-flow check.
"""
import glob
import os
import sys

import torch
import torch.distributed as dist

HUB = os.path.expanduser("~/.cache/huggingface/hub")
DRAFT = "AngelSlim--Qwen3-8B_eagle3"
FAILS = []

# With --negative-control the two TP gradient corrections are stubbed out and
# the verdict is inverted: the run passes only if divergence is observed. This
# shows that the test is able to detect the failure it checks for.
NEGATIVE_CONTROL = "--negative-control" in sys.argv

# Rank consistency does not imply correctness: an error that scales every
# gradient by the same factor on every rank (for example, applying the `f`
# conjugate twice at the lm_head) leaves the ranks in agreement while all the
# magnitudes are wrong. So the gradients are also compared against a tp=1 run.
#   step 1:  torchrun --nproc_per_node=1 ... --save-ref ref.pt
#   step 2:  torchrun --nproc_per_node=2 ... --compare-ref ref.pt
def _arg(flag):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else None


SAVE_REF = _arg("--save-ref")
COMPARE_REF = _arg("--compare-ref")


def check(name, cond, extra=""):
    if dist.get_rank() == 0:
        print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' — ' + extra) if extra else ''}")
    if not cond:
        FAILS.append(name)


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)

    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    path = glob.glob(f"{HUB}/models--{DRAFT}/snapshots/*")[0]
    server_args = ServerArgs(model_path=path, tp_size=world)
    set_global_server_args_for_scheduler(server_args)
    init_distributed_environment(
        world_size=world, rank=rank, local_rank=rank,
        distributed_init_method="env://", backend="nccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=world)
    # LogitsProcessor.__init__ reads the dp-attention globals even when dp
    # attention is off, so they have to be initialized.
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.layers.dp_attention import initialize_dp_attention

    initialize_dp_attention(server_args, ModelConfig.from_server_args(server_args))

    from sglang.srt.models.llama_eagle3 import LlamaForCausalLMEagle3
    from sglang.srt.speculative.eagle_lora_tts import (
        DEFAULT_LORA_TARGETS,
        EagleLoRAAdapterSet,
        parse_lora_targets,
    )
    from sglang.srt.speculative.eagle_tts import (
        EagleTTSConfig,
        EagleTTSState,
        tts_grad_mode,
    )
    from sglang.srt.utils.hf_transformers.config import get_config

    if NEGATIVE_CONTROL:
        # Disable both corrections: the `f` conjugate (so dL/d(hidden) out of
        # the vocab-parallel head stays a per-rank partial sum) and the
        # partial-gradient reduction for column-parallel A.
        import sglang.srt.speculative.eagle_tts as _tts

        _tts.all_reduce_input_grad = lambda x: x
        _tts._patch_column_parallel_input_grad = lambda stack: None
        EagleTTSState._reduce_partial_grads = lambda self: None
        if rank == 0:
            print("  [negative control] TP gradient fixes disabled")

    cfg = get_config(path, trust_remote_code=False)
    with torch.device("cuda"):
        model = LlamaForCausalLMEagle3(cfg)
    ckpt = torch.load(
        os.path.join(path, "pytorch_model.bin"), map_location="cpu", weights_only=True
    )
    model.load_weights(list(ckpt.items()))
    model = model.to(torch.bfloat16)
    # The draft ships no embed_tokens (the worker rebinds it to the target's at
    # runtime via set_embed) but does ship lm_head. Seed whatever
    # the checkpoint left unset, identically on every rank.
    #
    # "Unset" is tested with isfinite as well as `abs().sum() != 0`: under
    # torch.use_deterministic_algorithms, torch.empty returns NaN rather than
    # zeros, and `NaN == 0` is False.
    #
    # The seeding must be TP-consistent: generate the full tensor from a fixed
    # seed and take this rank's slice. Calling `w.normal_()` on the local shard
    # would draw the same numbers into a different slice on every rank, so the
    # tp=2 model would not be the tp=1 model and a cross-tp magnitude
    # comparison would be meaningless. `_init_a_template` initializes the
    # adapters the same way.
    with torch.no_grad():
        for mod in (model.model.embed_tokens, model.lm_head):
            w = mod.weight
            if torch.isfinite(w).all() and w.abs().sum() != 0:
                continue  # supplied by the checkpoint
            g = torch.Generator(device="cpu").manual_seed(1234)
            full_rows = w.shape[0] * world
            full = torch.empty(full_rows, w.shape[1], dtype=torch.float32)
            full.normal_(0, 0.02, generator=g)
            lo = rank * w.shape[0]
            w.copy_(full[lo : lo + w.shape[0]].to(w.dtype).to(w.device))
    unset = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    assert not unset, f"uninitialised draft weights: {unset}"

    aset = EagleLoRAAdapterSet(
        draft_model=model, targets=parse_lora_targets(DEFAULT_LORA_TARGETS),
        rank=16, alpha=32.0, tp_size=world, tp_rank=rank,
    )
    state = EagleTTSState(
        draft_model=model,
        config=EagleTTSConfig(enable=True, lr=1e-3, freq=1, async_step=False),
        device="cuda", tp_size=world, adapters=aset,
    )

    if rank == 0:
        print(f"\n=== tp={world} | {len(aset.adapters)} adapters, "
              f"{aset.num_parameters()/1e6:.3f}M params/rank ===")

    # Two steps: the first moves B off zero (dL/dA is identically zero until
    # then), the second produces meaningful gradients for A.
    for step in range(2):
        torch.manual_seed(7)  # identical inputs on every rank
        with tts_grad_mode(), torch.enable_grad():
            dl = surrogate(model, cfg)
            # Assert here so that a bad test input fails as a test error
            # instead of being skipped by the non-finite-loss guard in `step`.
            assert torch.isfinite(dl).all(), (
                "surrogate produced non-finite logits -- bad test input, not a "
                "TTS bug"
            )
            # A target close to the draft. Both sides are cast to fp32 and
            # log_softmax subtracts the max, so the loss stays finite either
            # way; a pure randn target would only give a larger KL.
            tl = dl.detach() + 0.5 * torch.randn_like(dl)
            state.step(draft_logits=dl, target_logits=tl)

    names = aset.parameter_names()
    params = aset.parameters()

    # --- the optimizer actually ran -----------------------------------------
    # Checked first: the "finite" / "non-zero" checks below skip None grads, so
    # if no update had run they would both pass vacuously.
    check(
        "optimizer stepped on every firing round",
        state.step_ct == 2 and state.skip_ct == 0,
        f"step_ct={state.step_ct} skip_ct={state.skip_ct} "
        f"nonfinite_ct={state.nonfinite_ct}",
    )
    moved = sum(1 for n, p in zip(names, params)
                if n.endswith("lora_b") and p.abs().sum().item() > 0)
    check("B lifted off zero (i.e. training happened)", moved > 0,
          f"{moved} non-zero B tensors")

    # --- gradients exist and are usable -------------------------------------
    missing = [n for n, p in zip(names, params) if p.grad is None]
    check("every trainable has a gradient", not missing, f"missing={missing}")
    nonfinite = [n for n, p in zip(names, params)
                 if p.grad is not None and not torch.isfinite(p.grad).all()]
    check("all gradients finite", not nonfinite, f"bad={nonfinite}")
    zero = [n for n, p in zip(names, params)
            if p.grad is not None and p.grad.abs().sum().item() == 0.0]
    check("no gradient is identically zero", not zero, f"zero={zero}")

    # --- rank consistency of replicated tensors ------------------------------
    # Replicated tensors must be equal across ranks. Sharded tensors differ
    # per rank by design and are skipped.
    replicated = replicated_names(aset)
    g_worst, g_name = max_rank_disagreement(
        [(n, p.grad) for n, p in zip(names, params) if n in replicated], names
    )
    check(
        "replicated gradients identical across ranks",
        g_worst < 1e-6,
        f"worst rel diff {g_worst:.3e} on {g_name} "
        f"(over {len(replicated)} replicated tensors)",
    )

    p_worst, p_name = max_rank_disagreement(
        [(n, p.detach()) for n, p in zip(names, params) if n in replicated], names
    )
    check(
        "replicated PARAMETERS identical after 2 optimizer steps",
        p_worst < 1e-6,
        f"worst rel diff {p_worst:.3e} on {p_name}",
    )

    # --- magnitude correctness vs a tp=1 run ---------------------------------
    if SAVE_REF and rank == 0:
        torch.save({n: (p.grad.detach().float().cpu() if p.grad is not None else None)
                    for n, p in zip(names, params)}, SAVE_REF)
        print(f"  saved tp=1 reference gradients -> {SAVE_REF}")
    elif COMPARE_REF:
        ref = torch.load(COMPARE_REF, map_location="cpu")
        worst, worst_n = 0.0, None
        for n, p in zip(names, params):
            r = ref.get(n)
            if r is None or p.grad is None:
                continue
            mine = gather_full(n, p.grad.detach().float(), aset)
            if mine.shape != r.shape:
                check(f"shape matches tp=1 for {n}", False,
                      f"{tuple(mine.shape)} vs {tuple(r.shape)}")
                continue
            d = (mine.cpu() - r).abs().max().item() / (r.abs().max().item() or 1.0)
            if d > worst:
                worst, worst_n = d, n
        # bf16 activations rule out exact equality. A scale error, which is
        # what this comparison targets, is O(tp_size), far above the threshold.
        check("gradient MAGNITUDES match the tp=1 reference", worst < 0.05,
              f"worst rel diff {worst:.3e} on {worst_n}")

    ok = not FAILS
    if NEGATIVE_CONTROL:
        # Inverted: with the corrections stubbed out the replicated tensors
        # must diverge. If they do not, this test cannot detect the failure it
        # checks for.
        ok = bool(FAILS)
        if rank == 0:
            print(
                f"\nNEGATIVE CONTROL: {'PASS' if ok else 'FAIL'} — "
                + ("divergence correctly detected" if ok else
                   "NO divergence detected with the fix disabled; this test "
                   "does not actually check what it claims")
            )
    elif rank == 0:
        print(f"\n{'ALL PASS' if ok else 'FAILURES: ' + ', '.join(FAILS)}")

    dist.barrier()
    dist.destroy_process_group()
    sys.exit(0 if ok else 1)


def gather_full(name, t, aset):
    """Reassemble a trainable's full (unsharded) tensor for comparison with tp=1.

    Replicated tensors are already full. Sharded ones are concatenated along the
    axis their layout shards: column-parallel/vocab B is [out_local, rank] (dim
    0); row-parallel A is [rank, in_local] (dim 1).
    """
    world = dist.get_world_size()
    if world == 1:
        return t
    replicated = {id(p) for p in aset.replicated_params()}
    lookup = dict(zip(aset.parameter_names(), aset.parameters()))
    if id(lookup[name]) in replicated:
        return t
    base = name.rsplit(".", 1)[0]
    adapter = aset.adapters.get(base)
    # Adapter tensors carry a leading pool axis: A is [P, rank, in], B is
    # [P, out, rank]. So the sharded axis is 2 for row-parallel A and 1 for
    # column-parallel B, one further right than without the pool axis.
    # Concatenating along dim 0 would join along the slot axis instead.
    pooled = adapter is not None and adapter.a.dim() == 3
    if adapter is not None and adapter.layout == "row":
        dim = 2 if pooled else 1
    else:
        dim = 1 if pooled else 0

    def gather(x, d):
        parts = [torch.empty_like(x) for _ in range(world)]
        dist.all_gather(parts, x.contiguous())
        return torch.cat(parts, dim=d)

    # `gate_up_proj` is a MergedColumnParallelLinear: each rank stores
    # [gate_shard | up_shard], so a naive rank-concat gives
    # [gate0, up0, gate1, up1] while tp=1 is [gate0, gate1, up0, up1]. Gather
    # the two halves separately and re-join. (`q` needs no such handling: its B
    # covers only the q slice, which is contiguous within each rank.)
    if base.split(".", 1)[0] == "gate_up" and name.endswith("lora_b"):
        half = t.shape[dim] // 2
        lo = t.narrow(dim, 0, half)
        hi = t.narrow(dim, half, t.shape[dim] - half)
        return torch.cat([gather(lo, dim), gather(hi, dim)], dim=dim)
    return gather(t, dim)


def max_rank_disagreement(items, _names):
    """Largest relative disagreement between ranks, as seen by every rank.

    Comparing each rank against a broadcast from rank 0 is not enough on its
    own: rank 0 then compares against itself and always reports zero, and only
    rank 0 prints. The per-tensor differences are therefore reduced with MAX
    over ranks, so all ranks reach the same verdict.
    """
    names = [n for n, t in items if t is not None]
    diffs = torch.zeros(len(names), device="cuda")
    for i, (name, t) in enumerate([(n, t) for n, t in items if t is not None]):
        v = t.float()
        ref = v.clone()
        dist.broadcast(ref, src=0)
        scale = v.abs().max().item() or 1.0
        diffs[i] = (v - ref).abs().max().item() / scale
    # MAX over ranks per tensor, so rank 0 (which measures itself as 0 against
    # its own broadcast) still sees what the other ranks see.
    dist.all_reduce(diffs, op=dist.ReduceOp.MAX)
    order = torch.argsort(diffs, descending=True)
    if dist.get_rank() == 0:
        for i in order[:6].tolist():
            if diffs[i].item() > 1e-9:
                print(f"       {names[i]:34s} rel diff {diffs[i].item():.3e}")
    top = order[0].item() if len(names) else 0
    return (diffs[top].item() if len(names) else 0.0,
            names[top] if len(names) else None)


def replicated_names(aset):
    """Names whose value/gradient must match on every rank.

    Column-parallel targets keep A replicated (B is sharded); row-parallel keep
    B replicated (A is sharded); `fc` is a plain replicated Linear so both are;
    the trained RMSNorm weights are 1-D and replicated.
    """
    col = {"q", "k", "v", "gate_up", "lm_head"}
    row = {"o", "down"}
    out = set()
    for name in aset.adapters:
        base = name.split(".", 1)[0]
        if base in col:
            out.add(f"{name}.lora_a")
        elif base in row:
            out.add(f"{name}.lora_b")
        else:  # fc: replicated both ways
            out.add(f"{name}.lora_a")
            out.add(f"{name}.lora_b")
    out.update(n for n, _ in aset._norm_params)
    return out


def surrogate(model, cfg, n_pos=6):
    """Forward through the draft's real module order, without a ForwardBatch.

    Attention is a plain GQA-style mix rather than a real attention op, so a
    gradient still reaches q/k/v and o_proj. Must run inside tts_grad_mode:
    the fused RMSNorm/SiluAndMul have no autograd registration and would end
    the tape.
    """
    m, layer = model.model, model.model.layers[0]
    dt, dev = model.lm_head.weight.dtype, model.lm_head.weight.device
    unwrap = lambda r: r[0] if isinstance(r, tuple) else r

    ids = torch.randint(0, cfg.vocab_size, (n_pos,), device=dev)
    aux = torch.randn(n_pos, m.hidden_size_in * m.num_aux_hidden_states,
                      device=dev, dtype=dt)
    embeds = m.embed_tokens(ids)

    h = aux
    if m.fc_norm is not None:
        chunks = h.chunk(m.num_aux_hidden_states, dim=-1)
        h = torch.cat([n(c) for n, c in zip(m.fc_norm, chunks)], dim=-1)
    h = m.fc(h)
    residual = h
    h = layer.hidden_norm(h)
    h = torch.cat([layer.input_layernorm(embeds), h], dim=-1)

    qkv = unwrap(layer.self_attn.qkv_proj(h))
    attn = layer.self_attn
    q_size, kv_size = attn.q_size, attn.kv_size
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

    # GQA-style head expansion rather than `k.repeat(1, groups)`. `repeat`
    # tiles the flat vector, so element j pairs with k[j % kv_size], and
    # kv_size is per-rank: the surrogate's forward would differ between tp=1
    # and tp=2 and a cross-tp gradient comparison would be meaningless.
    # Expanding along the head axis is TP-equivariant, because q heads and kv
    # heads are sharded by the same factor.
    hd = attn.head_dim
    nq, nkv = q_size // hd, kv_size // hd
    n = q.shape[0]
    kx = k.view(n, nkv, hd).repeat_interleave(nq // nkv, dim=1).reshape(n, q_size)
    vx = v.view(n, nkv, hd).repeat_interleave(nq // nkv, dim=1).reshape(n, q_size)
    attn_out = unwrap(attn.o_proj(q + kx + vx))

    h, residual = layer.post_attention_layernorm(attn_out, residual)
    h = layer.mlp(h)
    h = unwrap(m.norm(h, residual))

    # The serving path reaches the head through
    # LogitsProcessor._compute_lm_head, which tts_grad_mode patches to apply
    # the `f` conjugate. Calling head(h) directly would skip that, so apply it
    # here exactly once, as the patch does.
    from sglang.srt.speculative.eagle_tts import all_reduce_input_grad

    h = all_reduce_input_grad(h)
    head = model.lm_head
    if hasattr(head, "set_lora") and hasattr(head, "apply_lora"):
        logits = head(h)
    else:
        logits = torch.matmul(h.to(head.weight.dtype), head.weight.T)

    # All-gather the logits, as LogitsProcessor._get_logits does. The head is
    # vocab-parallel, so without the gather each rank's loss would be computed
    # over a different slice of the vocabulary: a different objective per
    # rank, which no backward conjugate can reconcile.
    from sglang.srt.layers import logits_processor as _lp

    if dist.get_world_size() > 1:
        logits = _lp.tensor_model_parallel_all_gather(logits)
    return logits.float()


if __name__ == "__main__":
    main()
