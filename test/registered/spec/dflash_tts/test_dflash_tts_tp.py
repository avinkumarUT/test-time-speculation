"""Tensor-parallel exactness of full-weight DFlash TTS, emulated in one process.

With tp_size > 1 each rank holds a shard of the draft's projections and of the
target's `lm_head`, and a copy of the replicated parameters (the norms). A TTS
update is correct only if every rank applies exactly the update a single-GPU
run would apply to its shard, and identical updates to the replicated
parameters. Three things can break that without raising:

  * a collective without a backward severs or zeroes the gradient upstream of
    a row-parallel projection;
  * without the conjugate reduction on the input of a column-parallel
    projection, upstream gradients contain only one rank's shard;
  * `q_norm` / `k_norm` act on the heads a rank owns, so their gradient is a
    per-rank partial sum; and a per-rank clipping norm scales replicated
    parameters differently on every rank.

This test runs a tp_size 2 update with one thread per rank. The TP collectives
are replaced by in-process exchanges between the threads, and the rank models
are the shards of a tp_size 1 model, sliced the way SGLang's parallel layers
shard a checkpoint. It then checks, against the tp_size 1 run:

  1. the block hidden states are the same on every rank;
  2. every gradient equals the corresponding shard (or, for replicated
     parameters, the whole) of the single-GPU gradient;
  3. the clipped update, and therefore the clipping norm, is the global one;
  4. the KL gate sees the same value on every rank;
  5. negative controls: dropping the input-gradient reduction or the
     `q_norm` / `k_norm` completion is detected;
  6. the same with float32 masters over bfloat16 weights, including the
     negative controls;
  7. the same with the bfloat16 optimizer dtype, over several updates.

Sections 1 to 5 run a float32 draft, whose update is a float32 step in place,
so the comparison with the single-GPU run is tight. Sections 6 and 7 run a
bfloat16 draft, with float32 masters and with bfloat16 state. A bfloat16
forward rounds after every operation and a tensor-parallel forward sums in a
different order, so there the comparison with the single-GPU run has a
bfloat16 tolerance, and the exact checks are the ones that do not depend on
it: the step against a reference computed from the rank's own gradient, the
global clipping norm, and bit-identical replicated parameters across ranks.

Always runs on CPU; needs no GPU, checkpoint or server.

  python test/registered/spec/dflash_tts/test_dflash_tts_tp.py
"""

import math
import os
import sys
import threading
from types import SimpleNamespace

os.environ.setdefault("SGLANG_USE_CPU_ENGINE", "1")

import torch
from bf16_reference import Bf16StateReference, optimizer_moments, same_bits

import sglang.srt.distributed.parallel_state as parallel_state
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

TP = 2
HIDDEN, HEADS, KV_HEADS, HEAD_DIM, INTER = 32, 4, 2, 8, 48
LAYERS, BLOCK, VOCAB, WINDOW, PREFIX = 2, 4, 64, 6, 9
Q_SIZE, KV_SIZE = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
FAILS = []


def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' — ' + extra) if extra else ''}")
    if not cond:
        FAILS.append(name)


_server_args = ServerArgs(model_path="dummy", device="cpu")
_server_args.speculative_dflash_tts_enable = True
set_global_server_args_for_scheduler(_server_args)
parallel_state._TP = SimpleNamespace(world_size=1, rank_in_group=0)

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool  # noqa: E402
from sglang.srt.models.dflash import DFlashDraftModel  # noqa: E402
from sglang.srt.speculative import dflash_lora_tts, dflash_tts  # noqa: E402
from sglang.srt.speculative.dflash_tts import (  # noqa: E402
    DFlashBlockForward,
    DFlashTTSConfig,
    DFlashTTSState,
)


class InProcessTP:
    """Stands in for the TP collectives: one thread per rank.

    Every rank deposits its tensor, all ranks wait at a barrier, read every
    deposit, and wait again before the slots are reused. Ranks issue the same
    sequence of collectives, as they must on real hardware; a mismatch breaks
    the barrier instead of hanging.
    """

    def __init__(self, world_size):
        self.world_size = world_size
        self._local = threading.local()
        self._slots = [None] * world_size
        self._barrier = threading.Barrier(world_size, timeout=60)
        self.all_reduce_ct = 0

    def set_rank(self, rank):
        self._local.rank = rank

    def rank(self):
        return self._local.rank

    def abort(self):
        self._barrier.abort()

    def _exchange(self, x):
        self._slots[self.rank()] = x.detach().clone()
        self._barrier.wait()
        parts = list(self._slots)
        self._barrier.wait()
        return parts

    def all_reduce(self, x):
        if self.rank() == 0:
            self.all_reduce_ct += 1
        return torch.stack(self._exchange(x)).sum(0)

    def all_gather(self, x, dim=-1):
        return torch.cat(self._exchange(x), dim=dim)


def make_config():
    return SimpleNamespace(
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        intermediate_size=INTER,
        num_hidden_layers=LAYERS,
        rms_norm_eps=1e-6,
        attention_bias=False,
        rope_theta=10000.0,
        max_position_embeddings=256,
        hidden_act="silu",
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=WINDOW,
        use_sliding_window=True,
        block_size=BLOCK,
        dflash_config={"mask_token_id": 3, "target_layer_ids": [0, 1]},
        num_target_layers=4,
    )


def build_model(world_size, rank):
    parallel_state._TP = SimpleNamespace(world_size=world_size, rank_in_group=rank)
    try:
        return DFlashDraftModel(make_config())
    finally:
        parallel_state._TP = SimpleNamespace(world_size=1, rank_in_group=0)


def shard_slices(name, rank):
    """How rank `rank` of TP ranks slices the single-GPU parameter `name`.

    Returns `(dim, index)` or None for a replicated parameter. Column-parallel
    projections are sharded along their output rows (q, k, v and gate, up
    separately), row-parallel projections along their input columns.
    """

    def part(start, size):
        local = size // TP
        return torch.arange(start + rank * local, start + (rank + 1) * local)

    if name.endswith("qkv_proj.weight"):
        return 0, torch.cat(
            [part(0, Q_SIZE), part(Q_SIZE, KV_SIZE), part(Q_SIZE + KV_SIZE, KV_SIZE)]
        )
    if name.endswith("o_proj.weight"):
        return 1, part(0, Q_SIZE)
    if name.endswith("gate_up_proj.weight"):
        return 0, torch.cat([part(0, INTER), part(INTER, INTER)])
    if name.endswith("down_proj.weight"):
        return 1, part(0, INTER)
    return None


def shard_of(name, tensor, rank):
    sl = shard_slices(name, rank)
    return tensor if sl is None else tensor.index_select(sl[0], sl[1])


# --- Single-GPU model, its rank shards and the inputs ----------------------------
# One setup per draft dtype, holding the same values (rounded to the dtype):
# the single-GPU model, the rank models, block embeddings, cached prefix K/V,
# target logits and the target's head.
gen = torch.Generator().manual_seed(0)
_full_model = build_model(1, 0)
_values = {}
for name, param in _full_model.named_parameters():
    if name.endswith("norm.weight") or "layernorm" in name:
        _values[name] = 1.0 + 0.2 * torch.randn(param.shape, generator=gen)
    else:
        _values[name] = torch.randn(param.shape, generator=gen) / math.sqrt(
            param.shape[-1]
        )
_embeds = torch.randn(1, BLOCK, HIDDEN, generator=gen)
_prefix_k = torch.randn(LAYERS, PREFIX, KV_HEADS, HEAD_DIM, generator=gen)
_prefix_v = torch.randn(LAYERS, PREFIX, KV_HEADS, HEAD_DIM, generator=gen)
_target_logits = 3.0 * torch.randn(BLOCK, VOCAB, generator=gen)
_lm_head = 0.5 * torch.randn(VOCAB, HIDDEN, generator=gen)
positions = torch.arange(PREFIX, PREFIX + BLOCK, dtype=torch.int64).unsqueeze(0)
req_to_token = torch.zeros(1, PREFIX + BLOCK, dtype=torch.int32)
req_to_token[0, :PREFIX] = torch.arange(PREFIX, dtype=torch.int32)


def load(model, dtype, rank=None):
    """Give `model` the values of the checkpoint (rank's shard) in `dtype`."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            value = _values[name]
            if rank is not None:
                value = shard_of(name, value, rank)
            # Only the parameters are converted; the RoPE cos/sin cache is a
            # buffer that serving keeps in fp32.
            param.data = value.to(dtype, copy=True)
    return model


def make_setup(dtype):
    return SimpleNamespace(
        dtype=dtype,
        full_model=load(build_model(1, 0), dtype),
        rank_models=[load(build_model(TP, rank), dtype, rank) for rank in range(TP)],
        pretrained={name: value.to(dtype) for name, value in _values.items()},
        embeds=_embeds.to(dtype),
        prefix_k=_prefix_k.to(dtype),
        prefix_v=_prefix_v.to(dtype),
        target_logits=_target_logits.to(dtype),
        lm_head=_lm_head.to(dtype),
    )


FP32 = make_setup(torch.float32)
BF16 = make_setup(torch.bfloat16)
pretrained = FP32.pretrained
check(
    "rank models hold shards of the projections",
    tuple(FP32.rank_models[0].layers[0].self_attn.qkv_proj.weight.shape)
    == ((Q_SIZE + 2 * KV_SIZE) // TP, HIDDEN)
    and tuple(FP32.rank_models[0].layers[0].mlp.down_proj.weight.shape)
    == (HIDDEN, INTER // TP),
)


def make_pool(setup, kv_heads, head_start):
    pool = MHATokenToKVPool(
        size=PREFIX + 8,
        page_size=1,
        dtype=setup.dtype,
        head_num=kv_heads,
        head_dim=HEAD_DIM,
        layer_num=LAYERS,
        device="cpu",
        enable_memory_saver=False,
    )
    for layer_id in range(LAYERS):
        k_buffer, v_buffer = pool.get_kv_buffer(layer_id)
        k_buffer[:PREFIX] = setup.prefix_k[layer_id][
            :, head_start : head_start + kv_heads
        ]
        v_buffer[:PREFIX] = setup.prefix_v[layer_id][
            :, head_start : head_start + kv_heads
        ]
    return pool


def run_update(setup, model, pool, tp_size, tp_rank, lm_head_rows, cfg_kwargs, sgd):
    """One update on one rank; several with `num_updates` in `cfg_kwargs`.

    Returns the hidden states, the gradients the optimizer stepped on, the
    weights and the masters after the update, and the KL. With several updates,
    `history` has the gradients, the weights and the moments of each.
    """
    cfg_kwargs = {"optimizer_dtype": "float32", **cfg_kwargs}
    num_updates = cfg_kwargs.pop("num_updates", 1)
    forward = DFlashBlockForward(
        draft_model=model, token_to_kv_pool=pool, block_size=BLOCK, tp_size=tp_size
    )
    state = DFlashTTSState(
        draft_model=model,
        config=DFlashTTSConfig(enable=True, async_step=False, **cfg_kwargs),
        device="cpu",
        tp_size=tp_size,
        tp_rank=tp_rank,
    )
    if cfg_kwargs["optimizer_dtype"] == "bfloat16":
        return run_bf16_updates(
            setup, model, forward, state, lm_head_rows, num_updates
        )
    if sgd:
        # With plain SGD at lr 1 the applied update is the clipped gradient
        # itself, which makes the clipping coefficient observable.
        state._optimizer = torch.optim.SGD(state._step_params, lr=1.0)
    hidden = forward(
        input_embeds=setup.embeds,
        positions=positions,
        prefix_lens=[PREFIX],
        req_pool_indices=[0],
        req_to_token=req_to_token,
    )
    # The gradients are captured where the optimizer reads them: on the
    # stepped tensors (the parameters, or their float32 masters), completed
    # across ranks and clipped. The state releases them after the step.
    names = [name for name, _ in model.named_parameters()]
    grads = {}
    optimizer_step = state._optimizer.step

    def capture_then_step():
        for name, tensor in zip(names, state._step_params):
            grads[name] = None if tensor.grad is None else tensor.grad.detach().clone()
        return optimizer_step()

    state._optimizer.step = capture_then_step
    kl_value = state.step(
        draft_hidden=hidden,
        target_logits=setup.target_logits,
        lm_head_weight=lm_head_rows,
    )
    stepped = bool(grads)
    if not stepped:  # the update was skipped (KL gate)
        grads = {name: None for name in names}
    weights = {name: p.detach().clone() for name, p in model.named_parameters()}
    masters = None
    if state._masters is not None:
        masters = {name: m.detach().clone() for name, m in zip(names, state._masters)}
    return SimpleNamespace(
        hidden=hidden.detach().clone(),
        grads=grads,
        weights=weights,
        masters=masters,
        kl=kl_value,
        state=state,
        stepped=stepped,
    )


def run_bf16_updates(setup, model, forward, state, lm_head_rows, num_updates):
    """Updates with bfloat16 state, capturing what `step_param` receives."""
    names = [name for name, _ in model.named_parameters()]
    optimizer_step_param = state._optimizer.step_param
    grads = {}

    def capture_then_step(i, grad):
        grads[names[i]] = grad.detach().clone()
        optimizer_step_param(i, grad)

    state._optimizer.step_param = capture_then_step
    history = []
    for _ in range(num_updates):
        grads.clear()
        hidden = forward(
            input_embeds=setup.embeds,
            positions=positions,
            prefix_lens=[PREFIX],
            req_pool_indices=[0],
            req_to_token=req_to_token,
        )
        state.step(
            draft_hidden=hidden,
            target_logits=setup.target_logits,
            lm_head_weight=lm_head_rows,
        )
        state.advance()
        history.append(
            SimpleNamespace(
                grads={name: grads.get(name) for name in names},
                weights={n: p.detach().clone() for n, p in model.named_parameters()},
                m=[None if t is None else t.clone() for t in state._optimizer.m],
                v=[None if t is None else t.clone() for t in state._optimizer.v],
            )
        )
    return SimpleNamespace(
        hidden=hidden.detach().clone(),
        grads=history[-1].grads,
        weights=history[-1].weights,
        masters=None,
        kl=None,
        state=state,
        stepped=bool(grads),
        history=history,
    )


def run_single(cfg_kwargs, sgd=True, setup=FP32):
    load(setup.full_model, setup.dtype)
    return run_update(
        setup,
        setup.full_model,
        make_pool(setup, KV_HEADS, 0),
        1,
        0,
        setup.lm_head,
        cfg_kwargs,
        sgd,
    )


def run_tp(cfg_kwargs, sgd=True, patch=None, setup=FP32):
    """The same update with tp_size 2, one thread per rank."""
    tp = InProcessTP(TP)
    saved = (
        dflash_lora_tts.tensor_model_parallel_all_reduce,
        dflash_lora_tts.tensor_model_parallel_all_gather,
        dflash_lora_tts.get_tensor_model_parallel_rank,
    )
    dflash_lora_tts.tensor_model_parallel_all_reduce = tp.all_reduce
    dflash_lora_tts.tensor_model_parallel_all_gather = tp.all_gather
    dflash_lora_tts.get_tensor_model_parallel_rank = tp.rank
    results, errors = [None] * TP, [None] * TP
    kv_local = KV_HEADS // TP
    vocab_local = VOCAB // TP

    def rank_main(rank):
        tp.set_rank(rank)
        try:
            load(setup.rank_models[rank], setup.dtype, rank)
            results[rank] = run_update(
                setup,
                setup.rank_models[rank],
                make_pool(setup, kv_local, rank * kv_local),
                TP,
                rank,
                setup.lm_head[rank * vocab_local : (rank + 1) * vocab_local],
                cfg_kwargs,
                sgd,
            )
        except BaseException as e:  # noqa: BLE001
            errors[rank] = e
            tp.abort()

    undo_patch = patch() if patch is not None else None
    try:
        threads = [threading.Thread(target=rank_main, args=(r,)) for r in range(TP)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        (
            dflash_lora_tts.tensor_model_parallel_all_reduce,
            dflash_lora_tts.tensor_model_parallel_all_gather,
            dflash_lora_tts.get_tensor_model_parallel_rank,
        ) = saved
        if undo_patch is not None:
            undo_patch()
    real_errors = [
        e
        for e in errors
        if e is not None and not isinstance(e, threading.BrokenBarrierError)
    ]
    if real_errors or any(errors):
        raise (real_errors or [e for e in errors if e is not None])[0]
    return results, tp


def max_rel_err(single, ranks, field):
    """Largest error of the rank tensors against the single-GPU shards."""
    worst, worst_name = 0.0, ""
    for name in pretrained:
        ref_full = getattr(single, field)[name]
        for rank in range(TP):
            got = getattr(ranks[rank], field)[name]
            if ref_full is None or got is None:
                if (ref_full is None) != (got is None):
                    return float("inf"), name
                continue
            ref = shard_of(name, ref_full, rank)
            if field == "weights":
                ref, got = ref - shard_of(name, pretrained[name], rank), got - shard_of(
                    name, pretrained[name], rank
                )
            err = float((got - ref).abs().max()) / max(float(ref.abs().max()), 1e-12)
            if err > worst:
                worst, worst_name = err, name
    return worst, worst_name


print("\n[1] tp_size 2 reproduces the single-GPU update (clipping active)")
cfg = dict(lr=1.0, grad_clip=1e-2)
single = run_single(cfg)
ranks, tp = run_tp(cfg)
err = max(float((r.hidden - single.hidden).abs().max()) for r in ranks)
check("block hidden states agree on every rank", err < 1e-5, f"max|err|={err:.2e}")
check(
    "replicated hidden states are identical across ranks",
    torch.equal(ranks[0].hidden, ranks[1].hidden),
)
check(
    "gradients were captured for the comparison",
    single.stepped
    and all(r.stepped for r in ranks)
    and sum(g is not None for g in single.grads.values()) == len(pretrained) - 2
    and all(
        sum(g is not None for g in r.grads.values()) == len(pretrained) - 2
        for r in ranks
    ),
)
check(
    "a float32 draft is stepped in place, without masters",
    single.masters is None and all(r.masters is None for r in ranks),
)
check(
    "gradients are released after the step",
    all(p.grad is None for r in ranks for p in r.state._params),
)
err, name = max_rel_err(single, ranks, "grads")
check(
    "every gradient is the single-GPU gradient of its shard",
    err < 1e-4,
    f"{err:.2e} ({name})",
)
err, name = max_rel_err(single, ranks, "weights")
check(
    "the clipped update is the single-GPU update (global norm)",
    err < 1e-3,
    f"{err:.2e} ({name})",
)
total_sq = sum(
    float((single.weights[n] - pretrained[n]).pow(2).sum()) for n in pretrained
)
check("the single-GPU update has the clip norm", abs(math.sqrt(total_sq) - 1e-2) < 1e-5)
replicated = [n for n in pretrained if shard_slices(n, 0) is None]
check(
    "replicated parameters stay identical across ranks",
    all(torch.equal(ranks[0].weights[n], ranks[1].weights[n]) for n in replicated),
)
check(
    "sharded parameters differ between ranks (the shards are distinct)",
    not torch.equal(
        ranks[0].weights["layers.0.mlp.down_proj.weight"],
        ranks[1].weights["layers.0.mlp.down_proj.weight"],
    ),
)

print("\n[2] clipping disabled")
cfg = dict(lr=1.0, grad_clip=0.0)
single = run_single(cfg)
ranks, _ = run_tp(cfg)
err, name = max_rel_err(single, ranks, "weights")
check("unclipped update matches", err < 1e-3, f"{err:.2e} ({name})")

print("\n[3] Adam update")
cfg = dict(lr=1e-3, grad_clip=1.0)
single = run_single(cfg, sgd=False)
ranks, _ = run_tp(cfg, sgd=False)
# Adam's first step moves every weight by lr * sign(gradient); compare the
# moved fraction instead of magnitudes that are at rounding level.
agree = total = 0
for name in pretrained:
    for rank in range(TP):
        ref = shard_of(name, single.weights[name], rank)
        agree += int(
            torch.isclose(ranks[rank].weights[name], ref, atol=1e-6, rtol=0).sum()
        )
        total += ref.numel()
check(
    "Adam step matches on (almost) every weight",
    agree / total > 0.999,
    f"{agree}/{total}",
)

print("\n[3b] both losses give the single-GPU gradients under tensor parallelism")
for loss in ("lk", "kl"):
    cfg = dict(lr=1.0, grad_clip=0.0, tts_loss=loss)
    single = run_single(cfg)
    ranks, _ = run_tp(cfg)
    err, name = max_rel_err(single, ranks, "grads")
    check(f"{loss}: every gradient matches its shard", err < 1e-4, f"{err:.2e} ({name})")
    check(
        f"{loss}: replicated weights are identical across ranks",
        all(
            torch.equal(ranks[0].weights[n], ranks[1].weights[n])
            for n in pretrained
            if shard_slices(n, 0) is None
        ),
    )

print("\n[4] the gate sees the same value on every rank")
cfg = dict(lr=1.0, grad_clip=1e-2, kl_threshold=1e-9)
single = run_single(cfg)
ranks, _ = run_tp(cfg)
check(
    "KL is identical across ranks and matches the single-GPU value",
    ranks[0].kl == ranks[1].kl
    and abs(ranks[0].kl - single.kl) < 1e-5 * max(1.0, abs(single.kl)),
    f"{ranks[0].kl:.6f} vs {single.kl:.6f}",
)
cfg = dict(lr=1.0, grad_clip=1e-2, kl_threshold=1e9)
ranks, _ = run_tp(cfg)
check(
    "a gated update changes no rank",
    all(
        torch.equal(ranks[r].weights[n], shard_of(n, pretrained[n], r))
        for r in range(TP)
        for n in pretrained
    )
    and all(r.state.kl_skip_ct == 1 and not r.stepped for r in ranks),
)

print("\n[5] negative controls")
cfg = dict(lr=1.0, grad_clip=0.0)
single = run_single(cfg)


def without_input_grad_reduce():
    saved = dflash_tts._CopyToTP
    dflash_tts._CopyToTP = SimpleNamespace(apply=lambda x: x)

    def undo():
        dflash_tts._CopyToTP = saved

    return undo


ranks, _ = run_tp(cfg, patch=without_input_grad_reduce)
err, name = max_rel_err(single, ranks, "grads")
check(
    "dropping the input-gradient reduction is detected",
    err > 1e-2,
    f"{err:.2e} ({name})",
)


def without_partial_grad_completion():
    saved = dflash_tts._tp_param_roles

    def roles(model):
        sharded_ids, _ = saved(model)
        return sharded_ids, []

    dflash_tts._tp_param_roles = roles

    def undo():
        dflash_tts._tp_param_roles = saved

    return undo


ranks, _ = run_tp(cfg, patch=without_partial_grad_completion)
err, name = max_rel_err(single, ranks, "grads")
check(
    "dropping the q_norm / k_norm completion is detected",
    err > 1e-2 and ("q_norm" in name or "k_norm" in name),
    f"{err:.2e} ({name})",
)

print("\n[6] float32 masters over bfloat16 weights")
REPLICATED = [n for n in pretrained if shard_slices(n, 0) is None]
SHARDED = [n for n in pretrained if shard_slices(n, 0) is not None]


def bf16_shard(name, rank):
    return shard_of(name, BF16.pretrained[name], rank)


def master_err(single, ranks, field):
    """Relative error of the ranks' tensors against the single-GPU shards.

    Per parameter, the norm of the difference over the norm of the
    reference; `field` is "grads" or "masters" (the latter compares the
    applied updates). Returns the largest value and its parameter.
    """
    worst, worst_name = 0.0, ""
    for name in pretrained:
        for rank in range(TP):
            ref, got = getattr(single, field)[name], getattr(ranks[rank], field)[name]
            if ref is None or got is None:
                if (ref is None) != (got is None):
                    return float("inf"), name
                continue
            ref = shard_of(name, ref, rank)
            if field == "masters":
                start = bf16_shard(name, rank).float()
                ref, got = ref - start, got - start
            err = float((got - ref).norm()) / max(float(ref.norm()), 1e-12)
            if err > worst:
                worst, worst_name = err, name
    return worst, worst_name


def replicated_mismatch(ranks):
    """Replicated parameters whose masters or weights differ between ranks."""
    return [
        n
        for n in REPLICATED
        if not torch.equal(ranks[0].masters[n], ranks[1].masters[n])
        or not torch.equal(ranks[0].weights[n], ranks[1].weights[n])
    ]


# With clipping at 1e-2 the SGD update is far below the bfloat16 spacing of
# most weights, so it is visible in the masters and mostly not in the weights.
cfg = dict(lr=1.0, grad_clip=1e-2)
single = run_single(cfg, setup=BF16)
ranks, _ = run_tp(cfg, setup=BF16)
runs = [single] + ranks
check(
    "every rank holds float32 masters and bfloat16 weights",
    all(r.stepped and r.masters is not None for r in runs)
    and all(m.dtype == torch.float32 for r in runs for m in r.masters.values())
    and all(w.dtype == torch.bfloat16 for r in runs for w in r.weights.values()),
)
check(
    "gradients reach the masters in float32 and are released after the step",
    all(
        sum(g is not None for g in r.grads.values()) == len(pretrained) - 2
        and all(g.dtype == torch.float32 for g in r.grads.values() if g is not None)
        and all(p.grad is None for p in r.state._params)
        and all(m.grad is None for m in r.state._masters)
        for r in runs
    ),
)
err = max(
    float((r.hidden.float() - single.hidden.float()).norm())
    / float(single.hidden.float().norm())
    for r in ranks
)
check(
    "block hidden states agree within bfloat16 rounding",
    err < 5e-2 and torch.equal(ranks[0].hidden, ranks[1].hidden),
    f"relative error={err:.2e}",
)
# Exact, from each rank's own gradient: the SGD step at lr 1 on the float32
# value of the pretrained shard, and its rounding into the weights.
check(
    "masters are the pretrained shard minus the clipped gradient, bit for bit",
    all(
        torch.equal(
            ranks[rank].masters[n],
            bf16_shard(n, rank).float()
            - (0 if ranks[rank].grads[n] is None else ranks[rank].grads[n]),
        )
        for rank in range(TP)
        for n in pretrained
    ),
)
check(
    "weights are the masters rounded to bfloat16",
    all(
        torch.equal(r.weights[n], r.masters[n].to(torch.bfloat16))
        for r in ranks
        for n in pretrained
    ),
)
kept = sum(
    int((ranks[rank].weights[n] == bf16_shard(n, rank)).sum())
    for rank in range(TP)
    for n in pretrained
    if ranks[rank].grads[n] is not None
)
moved = sum(
    int((ranks[rank].masters[n] != bf16_shard(n, rank).float()).sum())
    for rank in range(TP)
    for n in pretrained
)
total = sum(
    ranks[rank].grads[n].numel()
    for rank in range(TP)
    for n in pretrained
    if ranks[rank].grads[n] is not None
)
check(
    "the masters hold an update that most weights round away",
    moved > 0.99 * total and kept > 0.5 * total,
    f"masters moved {moved}/{total}, weights unchanged {kept}/{total}",
)
# The clipping norm is the global one: shards summed over ranks, replicated
# parameters counted once.
total_sq = sum(
    float((ranks[rank].masters[n] - bf16_shard(n, rank).float()).pow(2).sum())
    for rank in range(TP)
    for n in SHARDED
) + sum(
    float((ranks[0].masters[n] - BF16.pretrained[n].float()).pow(2).sum())
    for n in REPLICATED
)
check(
    "the update of the masters has the global clip norm",
    abs(math.sqrt(total_sq) - 1e-2) < 1e-5,
    f"norm={math.sqrt(total_sq):.6e}",
)
check(
    "replicated masters and weights are identical across ranks",
    replicated_mismatch(ranks) == [],
    str(replicated_mismatch(ranks)),
)
err, name = master_err(single, ranks, "grads")
check(
    "every gradient is the single-GPU gradient of its shard (bfloat16 tolerance)",
    err < 5e-2,
    f"{err:.2e} ({name})",
)
err, name = master_err(single, ranks, "masters")
check(
    "the clipped update is the single-GPU update (bfloat16 tolerance)",
    err < 5e-2,
    f"{err:.2e} ({name})",
)

# Unclipped, the update is large enough to move the weights themselves.
cfg = dict(lr=1.0, grad_clip=0.0)
single = run_single(cfg, setup=BF16)
ranks, _ = run_tp(cfg, setup=BF16)
check(
    "unclipped: every stepped weight tensor moves, as its master rounded",
    all(
        (ranks[rank].grads[n] is None)
        == torch.equal(ranks[rank].weights[n], bf16_shard(n, rank))
        and torch.equal(
            ranks[rank].weights[n], ranks[rank].masters[n].to(torch.bfloat16)
        )
        for rank in range(TP)
        for n in pretrained
    ),
)
err, name = master_err(single, ranks, "masters")
check(
    "unclipped update matches the single-GPU update (bfloat16 tolerance)",
    err < 5e-2 and replicated_mismatch(ranks) == [],
    f"{err:.2e} ({name})",
)

# Adam: each rank's masters are a float32 Adam step on the float32 value of
# its pretrained shard with the gradient it stepped on.
cfg = dict(lr=1e-3, grad_clip=1.0)
adam_ranks, _ = run_tp(cfg, sgd=False, setup=BF16)
exact = True
for rank in range(TP):
    stepped_names = [n for n in pretrained if adam_ranks[rank].grads[n] is not None]
    ref = [bf16_shard(n, rank).float().clone() for n in stepped_names]
    for tensor, n in zip(ref, stepped_names):
        tensor.grad = adam_ranks[rank].grads[n].clone()
    torch.optim.Adam(ref, lr=1e-3).step()
    for tensor, n in zip(ref, stepped_names):
        exact = (
            exact
            and not torch.equal(tensor, bf16_shard(n, rank).float())
            and torch.equal(adam_ranks[rank].masters[n], tensor)
            and torch.equal(adam_ranks[rank].weights[n], tensor.to(torch.bfloat16))
        )
check(
    "Adam: masters equal a float32 Adam step, weights its rounding",
    exact and all(r.stepped and r.masters is not None for r in adam_ranks),
)
check(
    "Adam: replicated masters and weights are identical across ranks",
    replicated_mismatch(adam_ranks) == [],
)

# Negative controls with masters. A gradient that is not completed across
# ranks differs from rank to rank, so besides departing from the single-GPU
# gradient it makes replicated parameters diverge.
cfg = dict(lr=1.0, grad_clip=0.0)
broken, _ = run_tp(cfg, patch=without_input_grad_reduce, setup=BF16)
err, name = master_err(single, broken, "grads")
mismatch = replicated_mismatch(broken)
check(
    "masters: dropping the input-gradient reduction is detected",
    err > 0.5 and any("layernorm" in n for n in mismatch),
    f"{err:.2e} ({name}); diverged: {len(mismatch)} replicated parameters",
)
broken, _ = run_tp(cfg, patch=without_partial_grad_completion, setup=BF16)
err, name = master_err(single, broken, "grads")
mismatch = replicated_mismatch(broken)
check(
    "masters: dropping the q_norm / k_norm completion is detected",
    err > 0.5
    and ("q_norm" in name or "k_norm" in name)
    and len(mismatch) == 2 * LAYERS
    and all("q_norm" in n or "k_norm" in n for n in mismatch),
    f"{err:.2e} ({name}); diverged: {mismatch}",
)

print("\n[7] bfloat16 optimizer state over bfloat16 weights")
BF16_UPDATES = 4
BF16_LR, BF16_CLIP = 1e-3, 1e-1


def bf16_cfg(grad_clip=BF16_CLIP, num_updates=BF16_UPDATES):
    return dict(
        lr=BF16_LR,
        grad_clip=grad_clip,
        optimizer_dtype="bfloat16",
        num_updates=num_updates,
    )


def global_norm(rank_grads):
    """The norm of one gradient set: shards summed over ranks, replicated once."""
    sharded = sum(
        float(rank_grads[rank][n].double().pow(2).sum())
        for rank in range(TP)
        for n in SHARDED
        if rank_grads[rank][n] is not None
    )
    replicated = sum(
        float(rank_grads[0][n].double().pow(2).sum())
        for n in REPLICATED
        if rank_grads[0][n] is not None
    )
    return math.sqrt(sharded + replicated)


def replicated_state_mismatch(ranks):
    """Replicated parameters whose weight or moments differ between ranks."""
    state_names = [n for n, _ in ranks[0].state.draft_model.named_parameters()]
    mismatch = []
    for i, n in enumerate(state_names):
        if n not in REPLICATED:
            continue
        for a, b in zip(ranks[0].history, ranks[1].history):
            moments_equal = (a.m[i] is None and b.m[i] is None) or (
                same_bits(a.m[i], b.m[i]) and same_bits(a.v[i], b.v[i])
            )
            if not same_bits(a.weights[n], b.weights[n]) or not moments_equal:
                mismatch.append(n)
                break
    return mismatch


ranks, _ = run_tp(bf16_cfg(), sgd=False, setup=BF16)
check(
    "every rank updated, with bfloat16 state and no masters",
    all(r.stepped and r.masters is None for r in ranks)
    and all(r.state.update_ct == BF16_UPDATES for r in ranks)
    and all(w.dtype == torch.bfloat16 for r in ranks for w in r.weights.values())
    and all(
        m.dtype == torch.bfloat16 for r in ranks for m in optimizer_moments(r.state)
    )
    and all(p.grad is None for r in ranks for p in r.state._params)
    and all(h.grad is None for r in ranks for h in r.state._step_params),
)
check(
    "the gradients of a step are float32, on every parameter the block forward uses",
    all(
        all(g.dtype == torch.float32 for g in u.grads.values() if g is not None)
        and sum(g is not None for g in u.grads.values()) == len(pretrained) - 2
        for r in ranks
        for u in r.history
    ),
)
# Exact, from each rank's own completed and clipped gradient: float32 Adam on
# the widened state of the rank's shard, rounded once per update.
step_exact = True
for rank in range(TP):
    weights = [bf16_shard(n, rank).clone() for n in pretrained]
    reference = Bf16StateReference(weights, BF16_LR)
    for update in ranks[rank].history:
        reference.update([update.grads[n] for n in pretrained])
        step_exact = (
            step_exact
            and all(
                same_bits(update.weights[n], w) for n, w in zip(pretrained, weights)
            )
            and all(
                (a is None and b is None) or same_bits(a, b)
                for a, b in zip(update.m, reference.m)
            )
            and all(
                (a is None and b is None) or same_bits(a, b)
                for a, b in zip(update.v, reference.v)
            )
        )
check("each rank's step equals the reference, bit for bit", step_exact)
check(
    "the replicated tensors of the ranks are identical after every update",
    replicated_state_mismatch(ranks) == [],
    str(replicated_state_mismatch(ranks)),
)
check(
    "the shards of the ranks differ",
    not same_bits(
        ranks[0].weights["layers.0.mlp.down_proj.weight"],
        ranks[1].weights["layers.0.mlp.down_proj.weight"],
    ),
)
# The clipping norm is the global one, for a clip that is active on every update.
norms = [
    global_norm([ranks[rank].history[i].grads for rank in range(TP)])
    for i in range(BF16_UPDATES)
]
check(
    "the gradients of every update have the global clip norm",
    all(abs(norm - BF16_CLIP) < 1e-5 for norm in norms),
    str([f"{norm:.6f}" for norm in norms]),
)
unclipped, _ = run_tp(bf16_cfg(grad_clip=0.0, num_updates=1), sgd=False, setup=BF16)
clipped, _ = run_tp(bf16_cfg(num_updates=1), sgd=False, setup=BF16)
raw = [r.history[0].grads for r in unclipped]
raw_norm = global_norm(raw)
coef = torch.tensor(BF16_CLIP / (raw_norm + 1e-6), dtype=torch.float32)
scale_err = max(
    float((clipped[rank].history[0].grads[n] - coef * raw[rank][n]).abs().max())
    / float(raw[rank][n].abs().max())
    for rank in range(TP)
    for n in pretrained
    if raw[rank][n] is not None
)
local_norms = [
    math.sqrt(sum(float(g.double().pow(2).sum()) for g in r.values() if g is not None))
    for r in raw
]
check(
    "the clip scales the completed gradients by one coefficient on all ranks",
    raw_norm > BF16_CLIP and scale_err < 1e-5,
    f"norm before clipping {raw_norm:.4f}, relative error {scale_err:.2e}",
)
check(
    "premise: the global norm is not what any single rank would compute",
    all(abs(norm - raw_norm) > 1e-3 for norm in local_norms),
    f"global {raw_norm:.4f}, per rank {local_norms}",
)
# Against the single-GPU run of the same update, with a bfloat16 tolerance.
single = run_single(bf16_cfg(grad_clip=0.0, num_updates=1), sgd=False, setup=BF16)
err, name = master_err(single, unclipped, "grads")
check(
    "every gradient is the single-GPU gradient of its shard (bfloat16 tolerance)",
    err < 5e-2,
    f"{err:.2e} ({name})",
)

# Negative controls.
cfg = bf16_cfg(grad_clip=0.0, num_updates=1)
broken, _ = run_tp(cfg, sgd=False, patch=without_input_grad_reduce, setup=BF16)
err, name = master_err(single, broken, "grads")
mismatch = replicated_state_mismatch(broken)
check(
    "bfloat16: dropping the input-gradient reduction is detected",
    err > 0.5 and any("layernorm" in n for n in mismatch),
    f"{err:.2e} ({name}); diverged: {len(mismatch)} replicated parameters",
)
broken, _ = run_tp(cfg, sgd=False, patch=without_partial_grad_completion, setup=BF16)
err, name = master_err(single, broken, "grads")
mismatch = replicated_state_mismatch(broken)
check(
    "bfloat16: dropping the q_norm / k_norm completion is detected",
    err > 0.5
    and ("q_norm" in name or "k_norm" in name)
    and len(mismatch) == 2 * LAYERS
    and all("q_norm" in n or "k_norm" in n for n in mismatch),
    f"{err:.2e} ({name}); diverged: {mismatch}",
)

print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
