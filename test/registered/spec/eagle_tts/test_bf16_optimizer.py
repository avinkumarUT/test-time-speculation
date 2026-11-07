"""The bfloat16 optimizer dtype of EAGLE-3 TTS, on all three optimizer paths.

With `--speculative-eagle-tts-optimizer-dtype bfloat16` (`fp32_master=False`)
the weights and the Adam moments are stored in bf16, a step computes in fp32 and
rounds each result once when it stores it. The paths are per-request LoRA
(`PoolAdam`), shared LoRA and `full` mode (`Bf16StateAdam`). For each of them:

  (a) a step equals an fp32 Adam step on the widened state, rounded;
  (b) the state is bf16 by dtype and byte count, and no fp32 copy is kept;
  (c) updates below the bf16 spacing are lost, and accumulate with float32;
  (d) a slot or a tensor that did not step keeps every bit of its state;
  (e) a reset restores the state exactly and leaves other slots alone;
  (f) the tensor-parallel gradient completion and clip are those of float32,
      and replicated tensors stay bit-identical across ranks.

Also: non-finite gradients skip the step, the second moment does not decay
(the documented limitation), asynchronous steps equal synchronous ones, and
startup checks. Comparisons are of bytes, not of values within a tolerance.

Needs no checkpoint or server. `TTS_TEST_DEV` selects the device (default: CUDA
when available). Run directly or with pytest:

  python test/registered/spec/eagle_tts/test_bf16_optimizer.py
"""
import argparse
import dataclasses
import os
import sys
import threading
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers import linear as linear_layers
from sglang.srt.speculative.eagle_lora_tts import (
    EagleLoRAAdapterSet,
    LoRAAdapter,
    PoolAdam,
    _RowSelector,
)
from sglang.srt.speculative.eagle_tts import (
    Bf16StateAdam,
    EagleTTSConfig,
    EagleTTSState,
)

DEV = os.environ.get("TTS_TEST_DEV", "cuda" if torch.cuda.is_available() else "cpu")
BF16 = torch.bfloat16
POOL, RANK, IN, OUT, LR = 5, 4, 12, 10, 1e-3
B1, B2, EPS = 0.9, 0.999, 1e-8
PATHS = ("pool", "shared", "full")


def bits(t: torch.Tensor) -> bytes:
    """The bytes of a tensor. Unlike `torch.equal` this tells -0.0 from 0.0."""
    return bytes(t.detach().cpu().clone().untyped_storage())


def same(a, b) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and bits(a) == bits(b)


class ToyAdapters:
    """Real `LoRAAdapter`s behind the part of `EagleLoRAAdapterSet` the state uses.

    `column` has a replicated A with partial gradients and a sharded B; `row`
    has a sharded A and a replicated B.
    """

    parameters = EagleLoRAAdapterSet.parameters
    parameter_names = EagleLoRAAdapterSet.parameter_names
    replicated_params = EagleLoRAAdapterSet.replicated_params
    partial_grad_params = EagleLoRAAdapterSet.partial_grad_params
    reset = EagleLoRAAdapterSet.reset

    def __init__(self, pool_size, tp_size=1, dtype=BF16, seed=0, rows=None):
        self.pool_size = pool_size
        self._norm_params = []
        self._norm_templates = {}
        self.adapters = OrderedDict()
        for name, layout, partial in (("column", "column", True), ("row", "row", False)):
            self.adapters[name] = LoRAAdapter(
                name=name,
                in_features_full=IN,
                in_features_local=IN,
                a_shard_offset=0,
                out_features_local=OUT,
                rank=RANK,
                alpha=8.0,
                dtype=dtype,
                device=torch.device(DEV),
                seed=seed,
                tp_size=tp_size,
                a_grad_is_partial=partial,
                layout=layout,
                pool_size=pool_size,
                rows=rows,
            )


class ToyDraft(nn.Module):
    def __init__(self, dtype=BF16):
        super().__init__()
        torch.manual_seed(1)
        self.body = nn.Linear(IN, OUT, bias=False)
        self.lm_head = nn.Linear(OUT, IN, bias=False)
        self.to(DEV).to(dtype)

    def forward(self, x):
        return self.lm_head(torch.relu(self.body(x)))


def make(path, fp32_master, *, lr=LR, grad_clip=0.0, tp_size=1, async_step=False, dtype=BF16):
    cfg = EagleTTSConfig(
        enable=True,
        lr=lr,
        grad_clip=grad_clip,
        async_step=async_step,
        fp32_master=fp32_master,
        mode="full" if path == "full" else "lora",
    )
    if path == "full":
        return EagleTTSState(
            draft_model=ToyDraft(dtype), config=cfg, device=DEV, tp_size=tp_size
        )
    adapters = ToyAdapters(POOL if path == "pool" else 1, tp_size=tp_size, dtype=dtype)
    return EagleTTSState(
        draft_model=nn.Linear(1, 1), config=cfg, device=DEV, tp_size=tp_size,
        adapters=adapters,
    )


def random_grads(state, seed, scale=1.0, slots=None):
    gen = torch.Generator().manual_seed(seed)
    out = []
    for p in state._params:
        g = torch.randn(p.shape, generator=gen, dtype=torch.float32) * scale
        g = g.to(DEV).to(BF16)
        if slots is not None:
            keep = torch.zeros(p.shape[0], dtype=torch.bool, device=DEV)
            keep[slots] = True
            g = g * keep.view((-1,) + (1,) * (g.dim() - 1)).to(g.dtype)
        out.append(g)
    return out


def apply_grads(state, grads, slots=None):
    """Hand-set gradients, then everything after backward: `_finish_step`."""
    for p, g in zip(state._params, grads):
        p.grad = None if g is None else g.clone()
    if state.is_pooled:
        state.active_slots = slots
    return state._finish_step()


def moments(state, i):
    """The stored moments of trained tensor i; None before its first step."""
    opt = state._optimizer
    if isinstance(opt, (PoolAdam, Bf16StateAdam)):
        return opt.m[i], opt.v[i]
    st = opt.state.get(state._master[i])
    return (st["exp_avg"], st["exp_avg_sq"]) if st else (None, None)


class TorchAdamReference:
    """fp32 `torch.optim.Adam` on the widened state, rounded after every step."""

    def __init__(self, params, lr, grad_clip=0.0):
        self.w = [p.detach().clone() for p in params]
        self.m = [None] * len(self.w)
        self.v = [None] * len(self.w)
        self.t = [0] * len(self.w)
        self.lr, self.grad_clip = lr, grad_clip

    def step(self, grads):
        leaves = [nn.Parameter(w.float()) for w in self.w]
        opt = torch.optim.Adam(leaves, lr=self.lr)
        for i, (leaf, g) in enumerate(zip(leaves, grads)):
            leaf.grad = None if g is None else g.float()
            if g is not None and self.t[i] > 0:
                opt.state[leaf] = {
                    "step": torch.tensor(float(self.t[i])),
                    "exp_avg": self.m[i].float(),
                    "exp_avg_sq": self.v[i].float(),
                }
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(leaves, self.grad_clip)
        opt.step()
        for i, leaf in enumerate(leaves):
            self.w[i] = leaf.detach().to(BF16)
            if leaf.grad is not None:
                self.m[i] = opt.state[leaf]["exp_avg"].to(BF16)
                self.v[i] = opt.state[leaf]["exp_avg_sq"].to(BF16)
                self.t[i] += 1


def pool_reference_step(w, m, v, t, g, lr):
    """The PoolAdam formula on one row of widened state, rounded."""
    w, m, v, g = w.float(), m.float(), v.float(), g.float()
    t = torch.tensor(float(t), dtype=torch.float32, device=w.device)
    m = B1 * m + (1 - B1) * g
    v = B2 * v + (1 - B2) * g * g
    upd = (m / (1.0 - B1**t)) / ((v / (1.0 - B2**t)).sqrt() + EPS)
    return (w - lr * upd).to(BF16), m.to(BF16), v.to(BF16)


# ------------------------------------------------------------------- (a) one step


def test_a_unpooled_steps_equal_rounded_fp32_adam():
    for path in ("shared", "full"):
        for grad_clip in (0.0, 0.05):
            state = make(path, False, grad_clip=grad_clip)
            ref = TorchAdamReference(state._params, LR, grad_clip)
            for seed in range(6):
                grads = random_grads(state, seed, scale=0.1)
                apply_grads(state, grads)
                ref.step(grads)
                for i, p in enumerate(state._params):
                    m, v = moments(state, i)
                    assert same(p, ref.w[i]), f"{path} clip={grad_clip} weight {i} step {seed}"
                    assert same(m, ref.m[i]), f"{path} clip={grad_clip} m {i} step {seed}"
                    assert same(v, ref.v[i]), f"{path} clip={grad_clip} v {i} step {seed}"


def test_a_direct_bf16_adam_is_a_different_update():
    """Negative control: rounding every operation is measurably not the reference."""
    w0 = (torch.randn(4096, device=DEV) * 0.02).to(BF16)
    direct = nn.Parameter(w0.clone())
    opt = torch.optim.Adam([direct], lr=LR)
    ref = TorchAdamReference([w0], LR)
    differs = False
    for seed in range(3):
        gen = torch.Generator().manual_seed(seed)
        g = (torch.randn(4096, generator=gen) * 0.1).to(DEV).to(BF16)
        direct.grad = g.clone()
        opt.step()
        ref.step([g])
        differs = differs or not same(direct.detach(), ref.w[0])
    assert differs, "direct bf16 Adam matched the reference; the control lost its point"


def test_a_pool_steps_equal_rounded_fp32_adam():
    state = make("pool", False)
    opt = state._optimizer
    w = [p.detach().clone() for p in state._params]
    m = [torch.zeros_like(x) for x in w]
    v = [torch.zeros_like(x) for x in w]
    t = [0] * POOL
    fired_rows = [[1], [1, 3], [3], [1, 3, 4], [4]]
    for seed, rows in enumerate(fired_rows):
        slots = torch.tensor(rows, device=DEV)
        grads = random_grads(state, seed, scale=0.1, slots=slots)
        apply_grads(state, grads, slots)
        for r in rows:
            t[r] += 1
            for i in range(len(w)):
                w[i][r], m[i][r], v[i][r] = pool_reference_step(
                    w[i][r], m[i][r], v[i][r], t[r], grads[i][r], LR
                )
        for i, p in enumerate(state._params):
            assert same(p, w[i]), f"weight {i} after step {seed}"
            assert same(opt.m[i], m[i]), f"m {i} after step {seed}"
            assert same(opt.v[i], v[i]), f"v {i} after step {seed}"
    assert opt.step_ct.tolist() == t


def test_a_pool_float32_rounded_is_the_first_bf16_step():
    """From the same state, float32 mode's first step rounds to the bf16 mode's."""
    bf, fp = make("pool", False), make("pool", True)
    slots = torch.tensor([0, 2], device=DEV)
    grads = random_grads(bf, 3, scale=0.1, slots=slots)
    apply_grads(bf, grads, slots)
    apply_grads(fp, grads, slots)
    for i in range(len(bf._params)):
        assert same(bf._params[i], fp._params[i])
        assert same(bf._optimizer.m[i], fp._optimizer.m[i].to(BF16))
        assert same(bf._optimizer.v[i], fp._optimizer.v[i].to(BF16))


# ------------------------------------------------------------------- (b) storage


def _state_tensors(state):
    opt = state._optimizer
    out = list(opt.m) + list(opt.v)
    if isinstance(opt, PoolAdam):
        out += [] if opt.master is opt.params else list(opt.master)
    return [t for t in out if t is not None]


def test_b_state_is_bf16_with_the_expected_bytes():
    for path in PATHS:
        state = make(path, False)
        slots = torch.arange(POOL, device=DEV) if state.is_pooled else None
        apply_grads(state, random_grads(state, 0), slots)
        elements = sum(p.numel() for p in state._params)
        assert state._master is None, path
        tensors = _state_tensors(state)
        assert tensors and all(t.dtype == BF16 for t in tensors), path
        assert sum(t.nbytes for t in tensors) == elements * 2 * 2, path
        assert all(p.dtype == BF16 for p in state._params), path
        if state.is_pooled:
            assert state._optimizer.master is state._optimizer.params
            assert all(
                a is b for a, b in zip(state._optimizer.master, state._params)
            ), "the master must not be a second bf16 copy"


def test_b_float32_state_stays_fp32():
    state = make("pool", True)
    opt = state._optimizer
    apply_grads(state, random_grads(state, 0), torch.arange(POOL, device=DEV))
    elements = sum(p.numel() for p in state._params)
    assert all(t.dtype == torch.float32 for t in list(opt.master) + opt.m + opt.v)
    assert sum(t.nbytes for t in list(opt.master) + opt.m + opt.v) == elements * 12
    for path in ("shared", "full"):
        state = make(path, True)
        apply_grads(state, random_grads(state, 0))
        assert all(m.dtype == torch.float32 for m in state._master), path
        assert isinstance(state._optimizer, torch.optim.Adam), path


# ------------------------------------------------------------- (c) small updates


def _set_all(state, value):
    with torch.no_grad():
        for p in state._params:
            p.fill_(value)
        if state._master is not None:
            for m in state._master:
                m.fill_(value)
        if isinstance(state._optimizer, PoolAdam) and state._optimizer.fp32_state:
            for m in state._optimizer.master:
                m.fill_(value)


def _constant_grad_steps(state, n):
    slots = torch.arange(POOL, device=DEV) if state.is_pooled else None
    grads = [torch.full_like(p, 0.5) for p in state._params]
    for _ in range(n):
        apply_grads(state, grads, slots)


def test_c_small_updates_are_lost_in_bf16_and_accumulate_in_float32():
    # At 1.0 half the spacing below is 2^-9 = 1.95e-3. Each Adam step moves a
    # weight by about the learning rate, 1e-4, so one step rounds away, and
    # 100 steps add up to about 1e-2, more than one bf16 spacing.
    for path in PATHS:
        bf = make(path, False, lr=1e-4)
        _set_all(bf, 1.0)
        _constant_grad_steps(bf, 100)
        for i, p in enumerate(bf._params):
            assert bool((p == 1.0).all()), f"{path}: a bf16 weight moved"
            m, v = moments(bf, i)
            assert bool((m != 0).all()) and bool((v != 0).all()), f"{path}: no moments"

        fp = make(path, True, lr=1e-4)
        _set_all(fp, 1.0)
        _constant_grad_steps(fp, 100)
        masters = fp._optimizer.master if path == "pool" else fp._master
        for p, master in zip(fp._params, masters):
            moved = (1.0 - master).abs().min().item()
            assert 5e-3 < moved < 2e-2, f"{path}: master moved by {moved}"
            assert bool((p < 1.0).all()), f"{path}: the served weight did not follow"
            assert same(p, master.to(BF16)), f"{path}: served != rounded master"


def test_c_second_moment_does_not_decay_in_bf16():
    # The documented limitation: 0.999 * v rounds back to v.
    for path in PATHS:
        results = {}
        for fp32_master in (True, False):
            state = make(path, fp32_master)
            _set_all(state, 0.0)
            slots = torch.arange(POOL, device=DEV) if state.is_pooled else None
            apply_grads(state, [torch.full_like(p, 1.0) for p in state._params], slots)
            small = [torch.full_like(p, 1e-6) for p in state._params]
            first = moments(state, 0)[1].clone()
            for _ in range(600):
                apply_grads(state, small, slots)
            results[fp32_master] = (first, moments(state, 0)[1])
        first, last = results[False]
        assert same(first, last), f"{path}: the bf16 second moment changed"
        first, last = results[True]
        ratio = (last / first).max().item()
        assert ratio < 0.56, f"{path}: the fp32 second moment kept {ratio}"


# ----------------------------------------------------------------- (d) isolation


def _snapshot_slot(state, r):
    opt = state._optimizer
    return (
        [bits(p[r]) for p in state._params],
        [bits(m[r]) for m in opt.m],
        [bits(v[r]) for v in opt.v],
        int(opt.step_ct[r]),
    )


def test_d_a_slot_that_did_not_fire_keeps_every_bit():
    for fp32_master in (True, False):
        state = make("pool", fp32_master)
        all_slots = torch.tensor([1, 2, 3], device=DEV)
        apply_grads(state, random_grads(state, 0, slots=all_slots), all_slots)
        before_1 = _snapshot_slot(state, 1)
        before_3 = _snapshot_slot(state, 3)
        before_idle = _snapshot_slot(state, 4)
        # Slot 2 fires. Slot 1 is in the batch with a zero gradient (a request
        # with no rows in the loss), slot 3 is not in the batch.
        slots = torch.tensor([1, 2], device=DEV)
        apply_grads(state, random_grads(state, 1, slots=torch.tensor([2], device=DEV)), slots)
        assert _snapshot_slot(state, 1) == before_1, f"{fp32_master}: zero-grad slot"
        assert _snapshot_slot(state, 3) == before_3, f"{fp32_master}: absent slot"
        assert _snapshot_slot(state, 4) == before_idle, f"{fp32_master}: idle slot"
        assert int(state._optimizer.step_ct[2]) == 2


def test_d_slot_results_do_not_depend_on_other_slots():
    """A slot's trajectory is the same alone and next to other firing slots."""
    alone, crowd = make("pool", False), make("pool", False)
    for seed in range(4):
        grads = random_grads(alone, seed, scale=0.1)
        apply_grads(alone, grads, torch.tensor([2], device=DEV))
        apply_grads(crowd, grads, torch.tensor([0, 2, 4], device=DEV))
    for a, c, ma, mc in zip(
        alone._params, crowd._params, alone._optimizer.m, crowd._optimizer.m
    ):
        assert same(a[2], c[2]) and same(ma[2], mc[2])


def test_d_a_tensor_without_gradient_keeps_every_bit():
    for path in ("shared", "full"):
        state = make(path, False)
        grads = random_grads(state, 0)
        grads[0] = None
        before = state._params[0].detach().clone()
        apply_grads(state, grads)
        assert same(state._params[0], before), path
        assert moments(state, 0) == (None, None), path
        assert state._optimizer.step_ct[0] == 0 and state._optimizer.step_ct[1] == 1


# --------------------------------------------------------------------- (e) reset


def test_e_pool_reset_restores_the_slot_and_leaves_others_alone():
    state = make("pool", False)
    fresh_a = [p.detach().clone() for p in state._params]
    slots = torch.tensor([1, 2], device=DEV)
    for seed in range(3):
        apply_grads(state, random_grads(state, seed, slots=slots), slots)
    other_before = _snapshot_slot(state, 2)
    state.reset(torch.tensor([1], device=DEV))

    assert _snapshot_slot(state, 2) == other_before, "the other slot changed"
    opt = state._optimizer
    assert int(opt.step_ct[1]) == 0
    for i, p in enumerate(state._params):
        assert same(p[1], fresh_a[i][1]), "the reset slot's weight was not restored"
        assert bool((opt.m[i][1] == 0).all()) and bool((opt.v[i][1] == 0).all())

    # A recycled slot now behaves exactly like one that never stepped.
    twin = make("pool", False)
    one = torch.tensor([1], device=DEV)
    grads = random_grads(state, 9, slots=one)
    apply_grads(state, grads, one)
    apply_grads(twin, grads, one)
    for a, b in zip(state._params, twin._params):
        assert same(a[1], b[1])


def test_e_unpooled_reset_restores_exactly_and_clears_the_optimizer():
    for path in ("shared", "full"):
        state = make(path, False)
        fresh = [p.detach().clone() for p in state._params]
        for seed in range(3):
            apply_grads(state, random_grads(state, seed))
        assert not same(state._params[-1], fresh[-1]), path
        state.reset()
        for p, f in zip(state._params, fresh):
            assert same(p, f), path
        assert state._optimizer.step_ct == [0] * len(fresh), path
        assert all(m is None for m in state._optimizer.m), path

        twin = make(path, False)
        grads = random_grads(state, 9)
        apply_grads(state, grads)
        apply_grads(twin, grads)
        for a, b in zip(state._params, twin._params):
            assert same(a, b), path


# ---------------------------------------------------------- non-finite gradients


def test_non_finite_gradient_skips_the_step():
    for path in PATHS:
        state = make(path, False)
        slots = torch.arange(POOL, device=DEV) if state.is_pooled else None
        apply_grads(state, random_grads(state, 0), slots)
        before = [bits(t) for t in state._params + _state_tensors(state)]
        grads = random_grads(state, 1)
        grads[0][..., 0] = float("nan")
        assert apply_grads(state, grads, slots) is None
        assert state.nonfinite_ct == 1, path
        assert before == [bits(t) for t in state._params + _state_tensors(state)], path


# --------------------------------------------------------------------- (f) TP


class InProcessTP:
    """All-reduce between one thread per rank."""

    def __init__(self, world_size):
        self.slots = [None] * world_size
        self.barrier = threading.Barrier(world_size, timeout=60)
        self.local = threading.local()

    def all_reduce(self, x):
        self.slots[self.local.rank] = x.detach().clone()
        self.barrier.wait()
        total = sum(self.slots)
        self.barrier.wait()
        return total.to(x.dtype)


def _run_ranks(tp, fn):
    errors = []

    def body(rank):
        tp.local.rank = rank
        try:
            fn(rank)
        except Exception as e:  # noqa: BLE001
            errors.append(e)
            tp.barrier.abort()

    original = linear_layers.tensor_model_parallel_all_reduce
    linear_layers.tensor_model_parallel_all_reduce = tp.all_reduce
    try:
        threads = [threading.Thread(target=body, args=(r,)) for r in range(len(tp.slots))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        linear_layers.tensor_model_parallel_all_reduce = original
    if errors:
        raise errors[0]


def _tp_grads(state, rank, seed):
    """Replicated tensors get one gradient on all ranks; partial and sharded
    tensors get a per-rank one."""
    replicated = {id(p) for p in state.adapters.replicated_params()} if state.adapters else set()
    partial = {id(p) for p in state.adapters.partial_grad_params()} if state.adapters else set()
    out = []
    for i, p in enumerate(state._params):
        shared = id(p) in replicated and id(p) not in partial
        gen = torch.Generator().manual_seed(seed * 100 + i + (0 if shared else rank + 1))
        out.append((torch.randn(p.shape, generator=gen) * 0.2).to(DEV).to(BF16))
    return out


def _run_tp(path, fp32_master, grad_clip, updates=3):
    tp = InProcessTP(2)
    states = [make(path, fp32_master, grad_clip=grad_clip, tp_size=2) for _ in range(2)]
    norms = [None, None]

    def body(rank):
        state = states[rank]
        slots = torch.tensor([1, 3], device=DEV) if state.is_pooled else None
        for seed in range(updates):
            norms[rank] = apply_grads(state, _tp_grads(state, rank, seed), slots)

    _run_ranks(tp, body)
    return states, norms


def test_f_replicated_tensors_stay_identical_across_ranks():
    for path in ("pool", "shared"):
        states, _ = _run_tp(path, False, grad_clip=0.05)
        rank0, rank1 = states
        ids0 = {id(p) for p in rank0.adapters.replicated_params()}
        for i, (a, b) in enumerate(zip(rank0._params, rank1._params)):
            if id(a) in ids0:
                assert same(a, b), f"{path}: replicated tensor {i} diverged"
                assert same(moments(rank0, i)[1], moments(rank1, i)[1]), path
            else:
                assert not same(a, b), f"{path}: sharded tensor {i} did not differ"


def test_f_gradients_and_clip_equal_those_of_float32():
    """Same inputs on both precisions: bf16 is float32's first step, rounded.

    A later step differs legitimately, because the two precisions start it from
    different states, so this compares the first update only, with a clip that
    engages. On the first update the gradient completion and the coefficient
    are the only inputs that could differ.
    """
    for path in ("pool", "shared"):
        out = {m: _run_tp(path, m, 0.05, updates=1) for m in (True, False)}
        for rank in range(2):
            fp_state, bf_state = out[True][0][rank], out[False][0][rank]
            assert same(out[True][1][rank], out[False][1][rank]), f"{path}: clip norm"
            for a, b in zip(fp_state._params, bf_state._params):
                assert same(a, b), f"{path}: first update differs on rank {rank}"
        # The norm is one value on every rank.
        assert same(out[False][1][0], out[False][1][1]), path


def test_f_non_finite_on_one_rank_skips_on_all_ranks():
    for path in ("pool", "shared"):
        tp = InProcessTP(2)
        states = [make(path, False, grad_clip=0.05, tp_size=2) for _ in range(2)]

        def body(rank):
            state = states[rank]
            slots = torch.tensor([1, 3], device=DEV) if state.is_pooled else None
            for seed in range(2):
                apply_grads(state, _tp_grads(state, rank, seed), slots)
            before = [bits(t) for t in state._params]
            grads = _tp_grads(state, rank, 2)
            if rank == 1:
                grads[1][..., 0] = float("nan")
            assert apply_grads(state, grads, slots) is None
            assert before == [bits(t) for t in state._params]

        _run_ranks(tp, body)


# ----------------------------------------------------------------- real backward


def test_full_mode_end_to_end_through_a_loss():
    """`EagleTTSState.step` from logits: weights move, stay bf16, reset restores."""
    state = make("full", False, lr=1e-2, grad_clip=1.0)
    draft = state.draft_model
    fresh = [p.detach().clone() for p in state._params]
    gen = torch.Generator().manual_seed(0)
    for _ in range(3):
        x = torch.randn(4, IN, generator=gen).to(DEV).to(BF16)
        target = torch.randn(4, IN, generator=gen).to(DEV).to(BF16)
        state.step(draft_logits=draft(x), target_logits=target)
    assert state.step_ct == 3
    assert all(p.dtype == BF16 for p in state._params)
    assert any(not same(p, f) for p, f in zip(state._params, fresh))
    state.reset()
    assert all(same(p, f) for p, f in zip(state._params, fresh))


def test_pool_end_to_end_through_a_loss():
    """`EagleTTSState.step` through real per-row adapter deltas and a clip."""
    # `_RowSelector.rows_for` asks whether a graph is being captured, which
    # raises when no CUDA device is visible.
    with patch.object(
        torch.cuda,
        "is_current_stream_capturing",
        torch.cuda.is_current_stream_capturing
        if torch.cuda.is_available()
        else (lambda: False),
    ):
        results = _pool_trajectories()
    bf, bf_traj = results[False]
    fp, fp_traj = results[True]
    assert bf_traj[0] == fp_traj[0], "the first update is not float32's, rounded"
    assert bf_traj[0] != bf_traj[2], "the adapters did not move"
    assert all(t.dtype == BF16 for t in _state_tensors(bf))
    # B moved in the slots that fired and nowhere else.
    for b in (bf.adapters.adapters["column"].b, bf.adapters.adapters["row"].b):
        assert bool(b[1].abs().sum() > 0) and bool(b[3].abs().sum() > 0)
        assert bool(b[0].abs().sum() == 0) and bool(b[2].abs().sum() == 0)
    assert bf._optimizer.step_ct.tolist() == [0, 3, 0, 3, 0]


def _pool_trajectories():
    results = {}
    for fp32_master in (True, False):
        selector = _RowSelector(8, torch.device(DEV), active=True)
        adapters = ToyAdapters(POOL, rows=selector)
        state = EagleTTSState(
            draft_model=nn.Linear(1, 1),
            config=EagleTTSConfig(
                enable=True,
                lr=1e-2,
                grad_clip=0.5,
                async_step=False,
                fp32_master=fp32_master,
                mode="lora",
                per_request=True,
            ),
            device=DEV,
            adapters=adapters,
        )
        slots = torch.tensor([1, 3], device=DEV)
        selector.set_rows(torch.tensor([1, 1, 3, 3], device=DEV))
        state.active_slots = slots
        gen = torch.Generator().manual_seed(0)
        target = torch.randn(4, OUT, generator=gen).to(DEV).to(BF16)
        trajectory = []
        for update in range(3):
            x = torch.randn(4, IN, generator=gen).to(DEV).to(BF16)
            base = torch.randn(4, OUT, generator=gen).to(DEV).to(BF16)
            logits = base + sum(a.delta(x) for a in adapters.adapters.values())
            state.step(
                draft_logits=logits,
                target_logits=target,
                row_groups=torch.tensor([0, 0, 1, 1], device=DEV),
                num_groups=2,
            )
            trajectory.append([bits(p) for p in state._params])
        results[fp32_master] = (state, trajectory)
    return results


def test_async_step_equals_sync_step():
    if DEV != "cuda":
        print("  skipped: the asynchronous path needs a CUDA stream")
        return
    states = {}
    for async_step in (False, True):
        state = make("full", False, lr=1e-2, grad_clip=1.0, async_step=async_step)
        assert state._use_async is async_step
        gen = torch.Generator().manual_seed(0)
        for _ in range(3):
            x = torch.randn(4, IN, generator=gen).to(DEV).to(BF16)
            target = torch.randn(4, IN, generator=gen).to(DEV).to(BF16)
            state.step(draft_logits=state.draft_model(x), target_logits=target)
            state.wait_for_pending()
        states[async_step] = state
    torch.cuda.synchronize()
    for i, (a, b) in enumerate(zip(states[False]._params, states[True]._params)):
        assert same(a, b), f"weight {i}"
        assert same(moments(states[False], i)[1], moments(states[True], i)[1]), f"v {i}"

    pool_states = {}
    for async_step in (False, True):
        state = make("pool", False, async_step=async_step)
        slots = torch.tensor([1, 3], device=DEV)
        state.active_slots = slots
        for seed in range(3):
            grads = random_grads(state, seed, slots=slots)
            for p, g in zip(state._params, grads):
                p.grad = g
            if async_step:
                with torch.cuda.stream(state._tts_stream):
                    state._finish_step()
                torch.cuda.synchronize()
            else:
                state._finish_step()
        pool_states[async_step] = state
    for a, b in zip(pool_states[False]._params, pool_states[True]._params):
        assert same(a, b)


# ------------------------------------------------------------------ configuration


def _server_args(fields):
    """A stand-in for ServerArgs with the given fields and the shared validator."""
    from sglang.srt.server_args import ServerArgs

    args = SimpleNamespace(**fields)
    args._validate_tts_loss_args = ServerArgs._validate_tts_loss_args.__get__(args)
    return args


def test_startup_checks_and_flags():
    from sglang.srt.server_args import ServerArgs

    for path in PATHS:
        for dtype in (torch.float32, torch.float16):
            try:
                make(path, False, dtype=dtype)
            except ValueError as e:
                assert "bfloat16 draft model" in str(e) and "float32" in str(e)
            else:
                raise AssertionError(f"{path}: a {dtype} draft was accepted")
        make(path, True, dtype=torch.float32)

    assert EagleTTSConfig.from_server_args(SimpleNamespace()).fp32_master is True
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    base = ["--model-path", "x"]
    args = parser.parse_args(base)
    assert args.speculative_eagle_tts_optimizer_dtype == "float32"
    assert EagleTTSConfig.from_server_args(args).fp32_master is True
    args = parser.parse_args(base + ["--speculative-eagle-tts-optimizer-dtype", "bfloat16"])
    assert EagleTTSConfig.from_server_args(args).fp32_master is False

    # Validation for values that do not come through the parser's `choices`.
    fields = {f.name: f.default for f in dataclasses.fields(ServerArgs)}
    fields.update(speculative_num_steps=3, speculative_eagle_tts_optimizer_dtype="bfloat16")
    ServerArgs._validate_eagle_tts(_server_args(fields))
    fields["speculative_eagle_tts_optimizer_dtype"] = "float16"
    try:
        ServerArgs._validate_eagle_tts(_server_args(fields))
    except ValueError as e:
        assert "float16" in str(e)
    else:
        raise AssertionError("an unknown optimizer dtype passed validation")


def _run_all():
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print("ALL PASS" if not failed else f"{failed} FAILED")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
