"""Unit tests for DFlashTTSState (full-weight DFlash TTS).

Covers the loss, the positional weights, the update stride, the KL gate
(including a KL exactly at the threshold), the per-request reset, the
synchronous and asynchronous update, the two optimizer dtypes (float32 masters
over the served weights; bfloat16 state with float32 arithmetic, rounded once
per update), target safety and the failure handling. Runs on a toy draft
module; needs no checkpoint and no running server.

  python test/registered/spec/dflash_tts/test_dflash_tts.py

`TTS_TEST_DEV` selects the device (default `cuda`). With `TTS_TEST_DEV=cpu` the
async-vs-sync comparison is skipped, since it needs a CUDA stream, and the
stream order of an update is checked with stand-ins for the CUDA stream API.
"""

import contextlib
import gc
import math
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch
from bf16_reference import (
    Bf16StateReference,
    optimizer_is_fresh,
    optimizer_moments,
    same_bits,
)
from torch import nn

from sglang.srt.speculative import dflash_tts
from sglang.srt.speculative.dflash_tts import (
    DFlashTTSConfig,
    DFlashTTSState,
    resolve_lm_head_rows,
)
from sglang.srt.speculative.tts_optim import Bf16StateAdam

IMPORTS_EAGLE_TTS = "sglang.srt.speculative.eagle_tts" in sys.modules

DEV = os.environ.get("TTS_TEST_DEV", "cuda")
HID, VOCAB, BLOCK = 32, 96, 8
FAILS = []


def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' — ' + extra) if extra else ''}")
    if not cond:
        FAILS.append(name)


def raises(exc_type, fn):
    try:
        fn()
    except exc_type as e:
        return str(e) or type(e).__name__
    return None


class ToyDraft(nn.Module):
    """Maps block inputs [bs, B, HID] to hidden states [bs, B, HID]."""

    def __init__(self, dtype=torch.bfloat16):
        super().__init__()
        self.body = nn.Linear(HID, HID, bias=False)
        self.out = nn.Linear(HID, HID, bias=False)
        self.to(dtype)

    def forward(self, x):
        return self.out(torch.tanh(self.body(x.to(self.body.weight.dtype))))


def make(draft_dtype=torch.bfloat16, target_model=None, init=None, **cfg_kwargs):
    torch.manual_seed(0)
    draft = ToyDraft(draft_dtype).to(DEV)
    if init is not None:
        init(draft)
    cfg_kwargs.setdefault("async_step", False)
    cfg_kwargs.setdefault("tts_loss", "kl")
    state = DFlashTTSState(
        draft_model=draft,
        config=DFlashTTSConfig(enable=True, **cfg_kwargs),
        device=DEV,
        target_model=target_model,
    )
    return draft, state


def make_data(seed=1, bs=1):
    """Block inputs, target verify logits and the target's head weight."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(bs, BLOCK, HID, generator=gen).to(DEV, torch.bfloat16)
    target_logits = (3.0 * torch.randn(bs * BLOCK, VOCAB, generator=gen)).to(
        DEV, torch.bfloat16
    )
    lm_head_weight = (0.5 * torch.randn(VOCAB, HID, generator=gen)).to(
        DEV, torch.bfloat16
    )
    return x, target_logits, lm_head_weight


def clones(tensors):
    return [tensor.detach().clone() for tensor in tensors]


def same(tensors_a, tensors_b):
    """Bit-identical, element for element; never true for empty lists."""
    tensors_a, tensors_b = list(tensors_a), list(tensors_b)
    return (
        len(tensors_a) > 0
        and len(tensors_a) == len(tensors_b)
        and all(same_bits(a, b) for a, b in zip(tensors_a, tensors_b))
    )


def max_abs_diff(tensors_a, tensors_b):
    return max(
        float((a.float() - b.float()).abs().max()) for a, b in zip(tensors_a, tensors_b)
    )


adam_moments = optimizer_moments


def live_float32_bytes():
    """Bytes of all live float32 tensors, each storage counted once."""
    gc.collect()
    storages = {}
    for obj in gc.get_objects():
        if isinstance(obj, torch.Tensor) and obj.dtype == torch.float32:
            storage = obj.untyped_storage()
            storages[storage._cdata] = storage.nbytes()
    return sum(storages.values())


def loss_value(state, draft, x, target_logits, lm_head_weight):
    kl, ce = state.compute_loss(
        draft_hidden=draft(x),
        target_logits=target_logits,
        lm_head_weight=lm_head_weight,
    )
    lam = state.cfg.tts_lambda
    return float(lam * kl + (1.0 - lam) * ce), float(kl), float(ce)


def reference_loss(hidden, target_logits, lm_head_weight, gamma, lam):
    """The loss written out position by position, in float64.

    Draft position k (k = 1..B-1) is scored against target row k - 1 with
    weight exp(-(k - 1) / gamma); the weights are normalized by their sum and
    the result is averaged over the batch.
    """
    bs, block, _ = hidden.shape
    target = target_logits.double().view(bs, block, -1)
    weights = [math.exp(-k / gamma) for k in range(block - 1)]
    kl_total = ce_total = 0.0
    for b in range(bs):
        for k in range(1, block):
            # The draft logits are formed in the head's dtype, then scored
            # in fp32 (here: float64 on the same bf16 logits).
            logits = (hidden[b, k].to(lm_head_weight.dtype) @ lm_head_weight.T).double()
            log_q = torch.log_softmax(logits, dim=-1)
            log_p = torch.log_softmax(target[b, k - 1], dim=-1)
            kl = float((log_p.exp() * (log_p - log_q)).sum())
            ce = float(-log_q[int(logits.argmax())])
            kl_total += weights[k - 1] * kl
            ce_total += weights[k - 1] * ce
    norm = bs * sum(weights)
    kl_term, ce_term = kl_total / norm, ce_total / norm
    return lam * kl_term + (1.0 - lam) * ce_term, kl_term, ce_term


print("\n[1] loss matches an independent computation")
for bs in (1, 3):
    draft, st = make(tts_lambda=0.7, pos_weight_gamma=3.0)
    x, t, w = make_data(bs=bs)
    with torch.no_grad():
        hidden = draft(x)
        kl, ce = st.compute_loss(draft_hidden=hidden, target_logits=t, lm_head_weight=w)
    ref_loss, ref_kl, ref_ce = reference_loss(hidden, t, w, gamma=3.0, lam=0.7)
    got = 0.7 * float(kl) + 0.3 * float(ce)
    check(
        f"bs={bs}: KL, CE and total agree",
        abs(float(kl) - ref_kl) < 1e-4 * max(1.0, abs(ref_kl))
        and abs(float(ce) - ref_ce) < 1e-4 * max(1.0, abs(ref_ce))
        and abs(got - ref_loss) < 1e-4 * max(1.0, abs(ref_loss)),
        f"kl {float(kl):.6f}/{ref_kl:.6f} ce {float(ce):.6f}/{ref_ce:.6f}",
    )
draft, st = make()
x, t, w = make_data()
with torch.no_grad():
    hidden = draft(x)
    kl_a, _ = st.compute_loss(draft_hidden=hidden, target_logits=t, lm_head_weight=w)
    # Target row B-1 (the bonus-token row) and draft position 0 are not scored.
    t2 = t.clone()
    t2[BLOCK - 1] += 5.0
    hidden2 = hidden.clone()
    hidden2[:, 0] += 1.0
    kl_b, _ = st.compute_loss(draft_hidden=hidden2, target_logits=t2, lm_head_weight=w)
check(
    "draft position 0 and the last target row are not scored",
    float(kl_a) == float(kl_b),
)

print("\n[2] positional weights are exp(-k / gamma)")
draft, st = make(pos_weight_gamma=7.0)
pw = st._position_weights(BLOCK - 1, DEV).cpu()
expected = torch.tensor([math.exp(-k / 7.0) for k in range(BLOCK - 1)])
check(
    "w_k = exp(-k / 7)", torch.allclose(pw, expected, atol=1e-6), str(pw.tolist()[:3])
)
check("first drafted position has weight 1", abs(float(pw[0]) - 1.0) < 1e-7)
# A uniform per-position KL must come out unchanged: the weights are
# normalized by their sum.
draft, st = make(pos_weight_gamma=0.5)
x, t, w = make_data()
with torch.no_grad():
    hidden = draft(x)
    hidden[:, 1:] = hidden[:, 1:2]  # same draft distribution at every position
    t_same = t.view(1, BLOCK, VOCAB).clone()
    t_same[:] = t_same[:, :1]  # same target distribution at every row
    kl_uniform, _ = st.compute_loss(
        draft_hidden=hidden, target_logits=t_same.view(BLOCK, VOCAB), lm_head_weight=w
    )
    st.cfg.pos_weight_gamma = 50.0
    kl_uniform2, _ = st.compute_loss(
        draft_hidden=hidden, target_logits=t_same.view(BLOCK, VOCAB), lm_head_weight=w
    )
check(
    "weights are normalized (uniform KL is independent of gamma)",
    abs(float(kl_uniform) - float(kl_uniform2)) < 1e-5,
    f"{float(kl_uniform):.6f} vs {float(kl_uniform2):.6f}",
)

print("\n[3] stride: an update fires every `freq` rounds, from round 0")
draft, st = make(freq=4)
pattern = []
for _ in range(10):
    pattern.append(st.should_fire())
    st.advance()
check(
    "fires at rounds 0, 4, 8", pattern == [i % 4 == 0 for i in range(10)], str(pattern)
)
st.reset()
check("reset restarts the stride", st.round_idx == 0 and st.should_fire())
draft, st = make(freq=1)
pattern = []
for _ in range(3):
    pattern.append(st.should_fire())
    st.advance()
check("freq 1 fires every round", pattern == [True, True, True])

print("\n[4] KL gate skips the update below the threshold")
draft, st = make(lr=1e-2, kl_threshold=1e9)
x, t, w = make_data()
before = draft.body.weight.detach().clone()
kl_value = st.step(draft_hidden=draft(x), target_logits=t, lm_head_weight=w)
check(
    "update skipped, KL returned",
    isinstance(kl_value, float)
    and torch.equal(draft.body.weight, before)
    and st.kl_skip_ct == 1
    and st.update_ct == 0,
    f"kl={kl_value}",
)
draft, st = make(lr=1e-2, kl_threshold=1e-9)
kl_value = st.step(draft_hidden=draft(x), target_logits=t, lm_head_weight=w)
check(
    "update applied above the threshold",
    isinstance(kl_value, float)
    and not torch.equal(draft.body.weight, before)
    and st.update_ct == 1,
)
draft, st = make(lr=1e-2)
check(
    "gate disabled: no KL is materialized",
    st.step(draft_hidden=draft(x), target_logits=t, lm_head_weight=w) is None,
)

print("\n[5] synchronous updates move the weights and lower the loss")
draft, st = make(lr=1e-3)
x, t, w = make_data()
with torch.no_grad():
    loss_before, kl_before, _ = loss_value(st, draft, x, t, w)
before = [p.detach().clone() for p in draft.parameters()]
for _ in range(40):
    st.step(draft_hidden=draft(x), target_logits=t, lm_head_weight=w)
with torch.no_grad():
    loss_after, kl_after, _ = loss_value(st, draft, x, t, w)
moved = max(float((p - b).abs().max()) for p, b in zip(draft.parameters(), before))
check("weights moved", moved > 0, f"max|delta|={moved:.3e}")
check(
    "loss decreased",
    loss_after < loss_before and kl_after < kl_before,
    f"loss {loss_before:.4f} -> {loss_after:.4f}, KL {kl_before:.4f} -> {kl_after:.4f}",
)
check("updates counted", st.update_ct == 40)
draft, st = make(lr=0.0)
before = [p.detach().clone() for p in draft.parameters()]
for _ in range(3):
    st.step(draft_hidden=draft(x), target_logits=t, lm_head_weight=w)
check(
    "lr 0 leaves the weights bit-identical",
    all(torch.equal(p, b) for p, b in zip(draft.parameters(), before)),
)

print("\n[6] reset restores the weights, the masters and the optimizer exactly")
for dtype in ("float32", "bfloat16"):
    draft, st = make(lr=1e-3, optimizer_dtype=dtype)
    x, t, w = make_data()
    params = list(draft.parameters())
    pretrained = clones(params)
    with_masters = dtype == "float32"
    check(
        f"{dtype}: masters {'exist' if with_masters else 'do not exist'}",
        (st._masters is not None) == with_masters,
    )

    def run_request(num_rounds=6):
        """A request of `num_rounds` update rounds: the weights (and the
        masters) after every update."""
        trajectory = []
        for _ in range(num_rounds):
            st.step(draft_hidden=draft(x), target_logits=t, lm_head_weight=w)
            st.advance()
            trajectory.append(
                (clones(params), clones(st._masters) if with_masters else None)
            )
        return trajectory

    first = run_request()
    moved = max_abs_diff(params, pretrained)
    if with_masters:
        masters_moved = max_abs_diff(st._masters, pretrained)
    st.reset()
    check(
        f"{dtype}: weights restored bit-exactly",
        moved > 0 and same(params, pretrained),
        f"moved={moved:.3e}",
    )
    if with_masters:
        check(
            f"{dtype}: masters restored to the float32 value of the snapshot",
            masters_moved > 0 and same(st._masters, [b.float() for b in pretrained]),
            f"moved={masters_moved:.3e}",
        )
    check(
        f"{dtype}: optimizer state cleared",
        optimizer_is_fresh(st) and adam_moments(st) == [],
    )
    check(f"{dtype}: round counter reset", st.round_idx == 0)
    # With the weights, the masters, Adam's moments and its step count back at
    # their initial values, the next request repeats the first bit for bit.
    second = run_request()
    check(
        f"{dtype}: the next request reproduces the first, update by update",
        len(second) == len(first)
        and all(same(a[0], b[0]) for a, b in zip(first, second))
        and (not with_masters or all(same(a[1], b[1]) for a, b in zip(first, second))),
    )
    check(
        f"{dtype}: snapshot keeps the weights' dtype and is not modified",
        same(st._snapshot, pretrained),
    )

print("\n[7] optimizer dtype: float32 masters vs bfloat16 state")
x, t, w = make_data()
draft, st = make(optimizer_dtype="float32", lr=1e-3)
params = list(draft.parameters())
storage = [p.data_ptr() for p in params]
check(
    "float32: a master starts at the float32 value of its weight",
    st._masters is not None and same(st._masters, [p.detach().float() for p in params]),
)
st.step(draft_hidden=draft(x), target_logits=t, lm_head_weight=w)
moments = adam_moments(st)
check(
    "float32: weights and snapshot stay bfloat16; masters and moments are float32",
    all(p.dtype == torch.bfloat16 for p in params)
    and all(s.dtype == torch.bfloat16 for s in st._snapshot)
    and all(m.dtype == torch.float32 for m in st._masters)
    and len(moments) == 2 * len(params)
    and all(m.dtype == torch.float32 for m in moments),
)
check(
    "float32: the weights are updated in place (same storage)",
    [p.data_ptr() for p in params] == storage and not same(params, st._snapshot),
)
check(
    "float32: the weights are the masters rounded to bfloat16",
    same(params, [m.to(torch.bfloat16) for m in st._masters])
    and not same(st._masters, [p.detach().float() for p in params]),
)
check(
    "float32: gradients are released after the step",
    all(p.grad is None for p in params) and all(m.grad is None for m in st._masters),
)
draft, st = make(optimizer_dtype="bfloat16", lr=1e-3)
params = list(draft.parameters())
storage = [p.data_ptr() for p in params]
num_elements = sum(p.numel() for p in params)
moments = None
float32_before = live_float32_bytes()
for _ in range(3):
    st.step(draft_hidden=draft(x), target_logits=t, lm_head_weight=w)
moments = adam_moments(st)
check(
    "bfloat16: weights, snapshot and Adam moments stay bfloat16; no masters",
    all(p.dtype == torch.bfloat16 for p in draft.parameters())
    and all(s.dtype == torch.bfloat16 for s in st._snapshot)
    and len(moments) == 2 * len(st._params)
    and all(m.dtype == torch.bfloat16 for m in moments)
    and st._masters is None
    and isinstance(st._optimizer, Bf16StateAdam),
)
check(
    "bfloat16: 2N bytes of weights, 2N of snapshot and 4N of moments",
    sum(p.numel() * p.element_size() for p in params) == 2 * num_elements
    and sum(s.numel() * s.element_size() for s in st._snapshot) == 2 * num_elements
    and sum(m.untyped_storage().nbytes() for m in moments) == 4 * num_elements,
)
check(
    "bfloat16: no float32 tensor outlives a step",
    live_float32_bytes() == float32_before,
    f"{live_float32_bytes() - float32_before} bytes more than before the steps",
)
check(
    "bfloat16: a gradient holder takes the memory of a scalar",
    all(h.untyped_storage().nbytes() == 4 for h in st._step_params),
)
check(
    "bfloat16: the weights are updated in place (same storage)",
    [p.data_ptr() for p in params] == storage and not same(params, st._snapshot),
)
check(
    "bfloat16: gradients are released after the step",
    all(p.grad is None for p in params)
    and all(h.grad is None for h in st._step_params),
)
draft, st = make(optimizer_dtype="float32", lr=1e-3)
float32_before = live_float32_bytes()
st.step(draft_hidden=draft(x), target_logits=t, lm_head_weight=w)
# Control of the live-tensor count: the two moments, and a step scalar per tensor.
check(
    "float32 keeps its moments in float32",
    live_float32_bytes() - float32_before
    == 2 * 4 * sum(p.numel() for p in draft.parameters()) + 4 * len(st._params),
    f"{live_float32_bytes() - float32_before} bytes",
)
msg = raises(
    ValueError, lambda: make(draft_dtype=torch.float32, optimizer_dtype="bfloat16")
)
check(
    "bfloat16 is rejected for a float32 draft",
    msg is not None and "bfloat16" in msg,
    msg or "",
)
msg = raises(ValueError, lambda: make(optimizer_dtype="float16"))
check("unknown optimizer dtype is rejected", msg is not None, msg or "")
draft, st = make(draft_dtype=torch.float32, optimizer_dtype="float32", lr=1e-3)
before = clones(draft.parameters())
st.step(draft_hidden=draft(x), target_logits=t, lm_head_weight=w)
check(
    "a float32 draft has no masters and is stepped in place",
    st._masters is None
    and st.update_ct == 1
    and all(p.dtype == torch.float32 for p in draft.parameters())
    and max_abs_diff(draft.parameters(), before) > 0,
)
draft, st = make(draft_dtype=torch.float16, optimizer_dtype="float32", lr=1e-3)
before = clones(draft.parameters())
st.step(draft_hidden=draft(x), target_logits=t, lm_head_weight=w)
check(
    "a float16 draft keeps float16 weights and gets float32 masters",
    st._masters is not None
    and all(p.dtype == torch.float16 for p in draft.parameters())
    and same(draft.parameters(), [m.to(torch.float16) for m in st._masters])
    and max_abs_diff(draft.parameters(), before) > 0,
)

print("\n[8] a float32-master update equals an independent reference")
# Reference: the gradient of the bfloat16 forward, taken on a twin draft;
# clipping and Adam in float32 on the float32 copy of the weights; the result
# rounded to bfloat16. The twin's state only evaluates the loss (checked
# against an independent computation in [1]); it never steps.
LR, CLIP, LAM = 1e-3, 0.5, 0.9
draft, st = make(optimizer_dtype="float32", lr=LR, grad_clip=CLIP, tts_lambda=LAM)
twin, twin_st = make(optimizer_dtype="bfloat16", tts_lambda=LAM)
in_place, in_place_st = make(
    optimizer_dtype="bfloat16", lr=LR, grad_clip=CLIP, tts_lambda=LAM
)
twin_params = list(twin.parameters())
ref_masters = [p.detach().float().clone() for p in twin_params]
ref_optimizer = torch.optim.Adam(ref_masters, lr=LR)
steps_equal, first_update_err, clipped = [], None, []
for i in range(4):
    xi, ti, wi = make_data(seed=20 + i)
    st.step(draft_hidden=draft(xi), target_logits=ti, lm_head_weight=wi)
    in_place_st.step(draft_hidden=in_place(xi), target_logits=ti, lm_head_weight=wi)

    kl, ce = twin_st.compute_loss(
        draft_hidden=twin(xi), target_logits=ti, lm_head_weight=wi
    )
    grads = torch.autograd.grad(LAM * kl + (1.0 - LAM) * ce, twin_params)
    for master, grad in zip(ref_masters, grads):
        master.grad = grad.float()
    norm = float(torch.nn.utils.clip_grad_norm_(ref_masters, CLIP))
    clipped.append(norm > CLIP)
    if i == 0:
        # Adam's first step, written out: w - lr * g / (|g| + eps).
        first_update_err = max_abs_diff(
            st._masters,
            [m.detach() - LR * m.grad / (m.grad.abs() + 1e-8) for m in ref_masters],
        )
    ref_optimizer.step()
    with torch.no_grad():
        for param, master in zip(twin_params, ref_masters):
            param.copy_(master)  # rounds to bfloat16
    steps_equal.append(
        same(st._masters, ref_masters) and same(draft.parameters(), twin_params)
    )
check(
    "masters and weights equal the reference after every update, bit for bit",
    len(steps_equal) == 4 and all(steps_equal),
    str(steps_equal),
)
check(
    "the first update is w - lr * g / (|g| + eps) on the float32 weights",
    first_update_err is not None and first_update_err < 1e-7,
    f"max|err|={first_update_err:.2e}",
)
check("clipping was active in the reference", all(clipped), str(clipped))
check(
    "the reference is not what the bfloat16 optimizer dtype produces",
    not same(in_place.parameters(), twin_params),
    f"max|diff|={max_abs_diff(in_place.parameters(), twin_params):.3e}",
)

print("\n[8b] a bfloat16 update equals an independent reference")
# Reference: float32 torch.optim.Adam, with the same clip, on the widened state;
# weights and moments rounded once per update. The gradient comes from a twin
# draft that the reference itself updates.
for clip, updates in ((CLIP, 8), (0.0, 8), (1e3, 3)):
    draft, st = make(optimizer_dtype="bfloat16", lr=LR, grad_clip=clip, tts_lambda=LAM)
    twin, twin_st = make(optimizer_dtype="float32", tts_lambda=LAM)
    twin_params = list(twin.parameters())
    reference = Bf16StateReference(twin_params, LR, clip)
    direct_draft, direct_st = make(optimizer_dtype="float32", tts_lambda=LAM)
    direct_params = list(direct_draft.parameters())
    direct_optimizer = torch.optim.Adam(direct_params, lr=LR)
    float32_draft, float32_st = make(
        optimizer_dtype="float32", lr=LR, grad_clip=clip, tts_lambda=LAM
    )
    weights_equal, moments_equal, weights_direct, moments_direct = [], [], [], []
    norms = []
    for i in range(updates):
        xi, ti, wi = make_data(seed=20 + i)
        st.step(draft_hidden=draft(xi), target_logits=ti, lm_head_weight=wi)
        kl, ce = twin_st.compute_loss(
            draft_hidden=twin(xi), target_logits=ti, lm_head_weight=wi
        )
        norms.append(
            reference.update(
                torch.autograd.grad(LAM * kl + (1.0 - LAM) * ce, twin_params)
            )
        )
        weights_equal.append(same(draft.parameters(), twin_params))
        moments_equal.append(
            same(st._optimizer.m, reference.m) and same(st._optimizer.v, reference.v)
        )
        if i == 0:
            float32_st.step(
                draft_hidden=float32_draft(xi), target_logits=ti, lm_head_weight=wi
            )
            first_is_float32_rounded = same(
                draft.parameters(), float32_draft.parameters()
            )
        # Control: torch.optim.Adam stepping directly on the bfloat16 tensors,
        # with the clipping norm computed in bfloat16.
        kl, ce = direct_st.compute_loss(
            draft_hidden=direct_draft(xi), target_logits=ti, lm_head_weight=wi
        )
        direct_optimizer.zero_grad()
        (LAM * kl + (1.0 - LAM) * ce).backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(direct_params, clip)
        direct_optimizer.step()
        weights_direct.append(same(direct_params, twin_params))
        moments_direct.append(
            same(
                [direct_optimizer.state[p]["exp_avg"] for p in direct_params],
                reference.m,
            )
            and same(
                [direct_optimizer.state[p]["exp_avg_sq"] for p in direct_params],
                reference.v,
            )
        )
    label = f"clip {clip}, {updates} updates"
    check(
        f"{label}: weights equal the reference after every update, bit for bit",
        all(weights_equal),
        str(weights_equal),
    )
    check(
        f"{label}: moments equal the reference after every update, bit for bit",
        all(moments_equal),
        str(moments_equal),
    )
    check(
        f"{label}: the first update is the float32 update, rounded",
        first_is_float32_rounded,
    )
    check(
        f"{label}: torch.optim.Adam directly on the bfloat16 tensors differs",
        not all(weights_direct) or not all(moments_direct),
        f"weights equal {sum(weights_direct)}/{updates}, "
        f"moments equal {sum(moments_direct)}/{updates}",
    )
    if clip == CLIP:
        check("clipping was active in the reference", all(n > CLIP for n in norms))

# A parameter without a gradient keeps every bit and allocates no moments.
torch.manual_seed(0)


class ContextOnlyDraft(ToyDraft):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(HID, HID, bias=False).to(torch.bfloat16)


context_only = ContextOnlyDraft().to(DEV)
st = DFlashTTSState(
    draft_model=context_only,
    config=DFlashTTSConfig(
        enable=True, async_step=False, lr=1e-2, optimizer_dtype="bfloat16"
    ),
    device=DEV,
)
fc_before = context_only.fc.weight.detach().clone()
xi, ti, wi = make_data(seed=30)
st.step(draft_hidden=context_only(xi), target_logits=ti, lm_head_weight=wi)
fc_index = st._param_names.index("fc.weight")
check(
    "bfloat16: a parameter without gradient keeps its bits and has no moments",
    same_bits(context_only.fc.weight, fc_before)
    and st._optimizer.m[fc_index] is None
    and st._optimizer.step_ct[fc_index] == 0
    and len(adam_moments(st)) == 2 * (len(st._params) - 1),
)

print("\n[9] updates below the bfloat16 spacing accumulate in the masters only")
SPACING = 2.0**-9  # of a bfloat16 value with magnitude in [0.25, 0.5)
NUM_UPDATES, SMALL_LR = 300, 2e-5


def coarse_weights(draft):
    """Give every weight a magnitude in [0.25, 0.5), where bfloat16 values
    are 2**-9 apart."""
    gen = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for param in draft.parameters():
            magnitude = 0.27 + 0.2 * torch.rand(param.shape, generator=gen)
            sign = 1.0 - 2.0 * (torch.rand(param.shape, generator=gen) < 0.5)
            param.copy_((sign * magnitude).to(param.device, param.dtype))


x, t, w = make_data()
changed, largest_update = {}, 0.0
for dtype in ("float32", "bfloat16"):
    draft, st = make(optimizer_dtype=dtype, lr=SMALL_LR, init=coarse_weights)
    params = list(draft.parameters())
    before = clones(params)
    for _ in range(NUM_UPDATES):
        if st._masters is not None:
            masters_before = clones(st._masters)
        st.step(draft_hidden=draft(x), target_logits=t, lm_head_weight=w)
        if st._masters is not None:
            largest_update = max(
                largest_update, max_abs_diff(st._masters, masters_before)
            )
    total = sum(p.numel() for p in params)
    changed[dtype] = sum(int((p != b).sum()) for p, b in zip(params, before)) / total
    check(
        f"{dtype}: {NUM_UPDATES} updates were applied by the optimizer",
        st.update_ct == NUM_UPDATES
        and len(adam_moments(st)) == 2 * len(params)
        and all(float(m.abs().max()) > 0 for m in adam_moments(st)),
    )
    if dtype == "float32":
        drift = max_abs_diff(st._masters, before)
check(
    "premise: every weight has the spacing 2**-9",
    all(
        float(b.float().abs().min()) >= 0.25 and float(b.float().abs().max()) < 0.5
        for b in before
    ),
)
check(
    "premise: every single update is below half the bfloat16 spacing",
    0 < largest_update < SPACING / 2,
    f"largest update {largest_update:.3e}, half spacing {SPACING / 2:.3e}",
)
check(
    "float32 masters: the accumulated updates move the served weights",
    changed["float32"] > 0.5 and drift > SPACING,
    f"changed fraction {changed['float32']:.3f}, largest master drift {drift:.3e}",
)
check(
    "bfloat16 state: the same updates change no weight",
    changed["bfloat16"] == 0.0,
    f"changed fraction {changed['bfloat16']:.3f}",
)

print("\n[10] asynchronous step")
if DEV != "cuda":
    draft, st = make(async_step=True)
    check(
        "async is guarded to CUDA (CPU runs synchronously)",
        st._use_async is False and st._tts_stream is None,
    )
    st.step(draft_hidden=draft(x), target_logits=t, lm_head_weight=w)
    st.wait_for_pending()
    check(
        "the synchronous fallback updates", st.update_ct == 1 and st._step_event is None
    )
    print("  SKIP  async == sync comparison requires CUDA (TTS_TEST_DEV=cpu)")
else:
    for dtype in ("float32", "bfloat16"):
        draft_a, st_a = make(optimizer_dtype=dtype, lr=1e-3, async_step=True)
        draft_s, st_s = make(optimizer_dtype=dtype, lr=1e-3, async_step=False)
        with_masters = dtype == "float32"
        check(
            f"{dtype}: async enabled",
            st_a._use_async is True and (st_a._masters is not None) == with_masters,
        )
        for i in range(4):
            xi, ti, wi = make_data(seed=10 + i)
            st_a.step(draft_hidden=draft_a(xi), target_logits=ti, lm_head_weight=wi)
            check(
                f"{dtype}: step {i} leaves an event to wait on",
                st_a._step_event is not None,
            )
            st_a.wait_for_pending()
            st_s.step(draft_hidden=draft_s(xi), target_logits=ti, lm_head_weight=wi)
            st_s.wait_for_pending()
            check(
                f"{dtype}: step {i} released the gradients at the fence",
                all(p.grad is None for p in st_a._params)
                and all(p.grad is None for p in st_a._step_params),
            )
        torch.cuda.synchronize()
        check(
            f"{dtype}: async == sync weights",
            same(draft_a.parameters(), draft_s.parameters())
            and not same(draft_a.parameters(), st_a._snapshot),
            f"max|diff|={max_abs_diff(draft_a.parameters(), draft_s.parameters()):.3e}",
        )
        if with_masters:
            check(
                f"{dtype}: async == sync masters, and the weights are their rounding",
                same(st_a._masters, st_s._masters)
                and same(
                    draft_a.parameters(), [m.to(torch.bfloat16) for m in st_a._masters]
                ),
            )
        else:
            check(
                f"{dtype}: async == sync moments, stored in bfloat16",
                same(adam_moments(st_a), adam_moments(st_s))
                and all(m.dtype == torch.bfloat16 for m in adam_moments(st_a)),
            )
        st_a.step(draft_hidden=draft_a(xi), target_logits=ti, lm_head_weight=wi)
        st_a.reset()  # must drain the in-flight step before restoring
        check(
            f"{dtype}: reset drains the in-flight step",
            st_a._step_event is None
            and same(draft_a.parameters(), st_a._snapshot)
            and (
                not with_masters
                or same(st_a._masters, [s.float() for s in st_a._snapshot])
            ),
        )

if DEV != "cuda":
    print("\n[10b] stream order of an asynchronous update, with stand-ins")

    class FakeStream:
        def __init__(self, trace):
            self.trace = trace

        def wait_stream(self, stream):
            self.trace.append("wait_stream")

    class FakeEvent:
        trace = None

        def record(self, stream):
            FakeEvent.trace.append("event")

        def synchronize(self):
            FakeEvent.trace.append("synchronize")

    def run_async_update(dtype):
        """One update through the asynchronous branch; returns the trace."""
        trace, recorded, side = [], [], [False]
        FakeEvent.trace = trace

        @contextlib.contextmanager
        def fake_stream_ctx(stream):
            side[0] = True
            try:
                yield
            finally:
                side[0] = False

        def record_stream(tensor, stream):
            recorded.append(tensor)

        def on(name, fn):
            def wrapper(*args, **kwargs):
                trace.append(f"{name}:{'side' if side[0] else 'main'}")
                return fn(*args, **kwargs)

            return wrapper

        draft, st = make(optimizer_dtype=dtype, lr=1e-3)
        st._use_async = True
        st._tts_stream = FakeStream(trace)
        st._move_grads_to_step_params = on(
            "move_grads", st._move_grads_to_step_params
        )
        if dtype == "bfloat16":
            st._optimizer.step_param = on("step_param", st._optimizer.step_param)
        else:
            st._optimizer.step = on("adam_step", st._optimizer.step)
            st._write_back_masters = on("write_back", st._write_back_masters)
        x, t, w = make_data()
        hidden = draft(x)
        with (
            patch.object(torch.cuda, "current_stream", lambda device=None: "main"),
            patch.object(torch.cuda, "stream", fake_stream_ctx),
            patch.object(torch.cuda, "Event", FakeEvent),
            patch.object(torch.Tensor, "record_stream", record_stream),
            patch.object(
                torch.nn.utils,
                "clip_grad_norm_",
                on("clip", torch.nn.utils.clip_grad_norm_),
            ),
        ):
            st.step(draft_hidden=hidden, target_logits=t, lm_head_weight=w)
            held = [h.grad for h in st._step_params]
            event_pending = st._step_event is not None
            trace_before_fence = list(trace)
            st.wait_for_pending()
        return SimpleNamespace(
            trace=trace_before_fence,
            full_trace=trace,
            recorded=recorded,
            held=held,
            event_pending=event_pending,
            state=st,
            draft=draft,
        )

    for dtype in ("float32", "bfloat16"):
        run = run_async_update(dtype)
        stepper = "step_param" if dtype == "bfloat16" else "adam_step"
        order = run.trace
        check(
            f"{dtype}: gradients to float32 on the main stream, before the wait",
            order[0] == "move_grads:main" and order[1] == "wait_stream",
            str(order[:3]),
        )
        check(
            f"{dtype}: clip, step and store on the side stream, event last",
            order[2] == "clip:side"
            and order[3] == f"{stepper}:side"
            and not any(entry.endswith(":main") for entry in order[1:])
            and order[-1] == "event",
            str(order[2:4]) + " ... " + order[-1],
        )
        recorded_ids = {id(tensor) for tensor in run.recorded}
        check(
            f"{dtype}: every float32 gradient is marked for the side stream",
            all(g is not None and id(g) in recorded_ids for g in run.held),
        )
        check(
            f"{dtype}: gradients stay held until the fence, then are released",
            run.event_pending
            and run.full_trace[-1] == "synchronize"
            and all(p.grad is None for p in run.state._params)
            and all(h.grad is None for h in run.state._step_params)
            and run.state._step_event is None,
        )
    run = run_async_update("bfloat16")
    reference_draft, reference_st = make(optimizer_dtype="bfloat16", lr=1e-3)
    x, t, w = make_data()
    reference_st.step(
        draft_hidden=reference_draft(x), target_logits=t, lm_head_weight=w
    )
    check(
        "bfloat16: the asynchronous branch stores what the synchronous one does",
        same(run.draft.parameters(), reference_draft.parameters())
        and same(adam_moments(run.state), adam_moments(reference_st)),
    )

print("\n[11] the target model is never modified")
torch.manual_seed(3)
target = nn.Linear(HID, VOCAB, bias=False).to(DEV, torch.bfloat16)
target_before = target.weight.detach().clone()
draft, st = make(lr=1e-2, target_model=target)
x, t, _ = make_data()
for _ in range(5):
    st.step(draft_hidden=draft(x), target_logits=t, lm_head_weight=target.weight)
st.reset()
check(
    "target lm_head is bit-identical after updates and reset",
    torch.equal(target.weight, target_before) and target.weight.grad is None,
)
torch.manual_seed(0)
aliased = ToyDraft().to(DEV)
alias_target = nn.Linear(HID, HID, bias=False).to(DEV, torch.bfloat16)
aliased.out.weight = alias_target.weight  # the draft shares a target tensor
msg = raises(
    RuntimeError,
    lambda: DFlashTTSState(
        draft_model=aliased,
        config=DFlashTTSConfig(enable=True, async_step=False),
        device=DEV,
        target_model=alias_target,
    ),
)
check(
    "refuses a draft that shares storage with the target",
    msg is not None,
    (msg or "")[:70],
)

print("\n[12] a forward without an autograd graph fails loudly")
draft, st = make()
x, t, w = make_data()
with torch.no_grad():
    hidden = draft(x)
msg = raises(
    RuntimeError,
    lambda: st.step(draft_hidden=hidden, target_logits=t, lm_head_weight=w),
)
check(
    "hidden states without grad_fn raise",
    msg is not None and "autograd" in msg,
    msg or "",
)


class PartialDraft(ToyDraft):
    """A draft with a parameter the forward does not reach."""

    def __init__(self, unused_name):
        super().__init__()
        setattr(self, unused_name, nn.Linear(HID, HID, bias=False).to(torch.bfloat16))


torch.manual_seed(0)
broken = PartialDraft("severed").to(DEV)
st = DFlashTTSState(
    draft_model=broken,
    config=DFlashTTSConfig(enable=True, async_step=False),
    device=DEV,
)
before = [p.detach().clone() for p in broken.parameters()]
msg = raises(
    RuntimeError,
    lambda: st.step(draft_hidden=broken(x), target_logits=t, lm_head_weight=w),
)
check(
    "an unreached parameter raises and disables TTS",
    msg is not None
    and "severed.weight" in msg
    and st.is_disabled()
    and not st.should_fire(),
    (msg or "")[:80],
)
check(
    "no weight was changed by the refused update",
    all(torch.equal(p, b) for p, b in zip(broken.parameters(), before)),
)
torch.manual_seed(0)
context_only = PartialDraft("fc").to(DEV)
st = DFlashTTSState(
    draft_model=context_only,
    config=DFlashTTSConfig(enable=True, async_step=False, lr=1e-2),
    device=DEV,
)
fc_before = context_only.fc.weight.detach().clone()
st.step(draft_hidden=context_only(x), target_logits=t, lm_head_weight=w)
check(
    "context-only parameters (fc, hidden_norm) may stay without gradient",
    st.update_ct == 1 and torch.equal(context_only.fc.weight, fc_before),
)

print("\n[13] repeated failures disable TTS and restore the weights")
for dtype in ("float32", "bfloat16"):
    draft, st = make(lr=1e-2, optimizer_dtype=dtype)
    pretrained = clones(draft.parameters())
    st.step(draft_hidden=draft(x), target_logits=t, lm_head_weight=w)
    moved = max_abs_diff(draft.parameters(), pretrained)
    for i in range(DFlashTTSState._MAX_FAILURE_STREAK):
        check(f"{dtype}: still enabled before failure {i + 1}", not st.is_disabled())
        st.note_step_failure(RuntimeError("out of memory"))
    check(
        f"{dtype}: disabled after the failure streak, weights restored",
        moved > 0
        and st.is_disabled()
        and not st.should_fire()
        and same(draft.parameters(), pretrained)
        and (st._masters is None or same(st._masters, [b.float() for b in pretrained])),
    )
    check(
        f"{dtype}: a disabled state holds no gradients and no optimizer state",
        all(p.grad is None for p in draft.parameters())
        and st._masters is None
        and st._grad_holders is None
        and st._step_params is st._params
        and st._optimizer is None,
    )
    check(
        f"{dtype}: a disabled state ignores further steps",
        st.step(draft_hidden=draft(x), target_logits=t, lm_head_weight=w) is None
        and same(draft.parameters(), pretrained),
    )
    st.reset()
    check(
        f"{dtype}: a disabled state survives a reset",
        st.is_disabled() and same(draft.parameters(), pretrained),
    )
draft, st = make(lr=1e-2)
st.note_step_failure(RuntimeError("transient"))
st.step(draft_hidden=draft(x), target_logits=t, lm_head_weight=w)
check("a successful update clears the failure streak", st._failure_streak == 0)
draft, st = make(lr=1e-2)
for _ in range(DFlashTTSState._MAX_FAILURE_STREAK - 1):
    st.note_step_failure(RuntimeError("transient"))
st.reset()
st.note_step_failure(RuntimeError("transient"))
check(
    "the failure streak is counted per request",
    st._failure_streak == 1 and not st.is_disabled(),
)

print("\n[14] a non-finite update is undone at the next fence")
x, t, w = make_data()
t_bad = t.clone()
t_bad[0, 0] = float("nan")
for dtype in ("float32", "bfloat16"):
    draft, st = make(lr=1e-2, optimizer_dtype=dtype)
    pretrained = clones(draft.parameters())
    st.step(draft_hidden=draft(x), target_logits=t, lm_head_weight=w)
    st.wait_for_pending()
    check(
        f"{dtype}: a finite update is kept",
        st.nonfinite_ct == 0
        and st._dirty
        and max_abs_diff(draft.parameters(), pretrained) > 0,
    )
    st.step(draft_hidden=draft(x), target_logits=t_bad, lm_head_weight=w)
    # The update wrote non-finite values into the weights and, in the masters
    # mode, into the masters as well.
    stored = list(st._params) + list(st._masters or [])
    poisoned = not all(bool(torch.isfinite(tensor).all()) for tensor in stored)
    st.wait_for_pending()
    check(
        f"{dtype}: non-finite update: pretrained weights restored, optimizer cleared",
        poisoned
        and st.nonfinite_ct == 1
        and same(draft.parameters(), pretrained)
        and (st._masters is None or same(st._masters, [b.float() for b in pretrained]))
        and optimizer_is_fresh(st)
        and not st._dirty,
    )
    fresh_draft, fresh_st = make(lr=1e-2, optimizer_dtype=dtype)
    st.step(draft_hidden=draft(x), target_logits=t, lm_head_weight=w)
    fresh_st.step(draft_hidden=fresh_draft(x), target_logits=t, lm_head_weight=w)
    check(
        f"{dtype}: the update after a restore is that of a fresh state",
        same(draft.parameters(), fresh_draft.parameters())
        and same(adam_moments(st), adam_moments(fresh_st)),
    )

draft, st = make(lr=1e-2)
pretrained = clones(draft.parameters())
t_masked = t.clone()
t_masked[..., ::7] = float("-inf")  # tokens the target rules out
kl_masked = float(
    st.compute_loss(draft_hidden=draft(x), target_logits=t_masked, lm_head_weight=w)[0]
)
st.step(draft_hidden=draft(x), target_logits=t_masked, lm_head_weight=w)
st.wait_for_pending()
check(
    "-inf target logits give a finite loss and a kept update",
    math.isfinite(kl_masked)
    and st.nonfinite_ct == 0
    and max_abs_diff(draft.parameters(), pretrained) > 0,
    f"kl={kl_masked:.4f}",
)

print("\n[15] configuration from server arguments")
args = SimpleNamespace(
    speculative_dflash_tts_enable=True,
    speculative_dflash_tts_lr=3e-5,
    speculative_tts_loss="kl",
    speculative_tts_lambda=0.5,
    speculative_dflash_tts_freq=7,
    speculative_dflash_tts_pos_weight_gamma=4.0,
    speculative_dflash_tts_kl_threshold=0.25,
    speculative_dflash_tts_grad_clip=2.0,
    speculative_dflash_tts_async=False,
    speculative_dflash_tts_optimizer_dtype="bfloat16",
)
cfg = DFlashTTSConfig.from_server_args(args)
check(
    "every flag is mapped",
    cfg
    == DFlashTTSConfig(
        enable=True,
        lr=3e-5,
        tts_loss="kl",
        tts_lambda=0.5,
        freq=7,
        pos_weight_gamma=4.0,
        kl_threshold=0.25,
        grad_clip=2.0,
        async_step=False,
        optimizer_dtype="bfloat16",
    ),
    str(cfg),
)
check(
    "defaults",
    DFlashTTSConfig.from_server_args(SimpleNamespace()) == DFlashTTSConfig(),
    str(DFlashTTSConfig()),
)

print("\n[16] gradient clipping bounds the update")
# The gradients are clipped in float32 on the masters. With plain SGD at lr 1
# the update of the masters is the clipped gradient itself.
x, t, w = make_data()
draft, st = make(lr=1.0, grad_clip=1e-3, optimizer_dtype="float32")
st._optimizer = torch.optim.SGD(st._step_params, lr=1.0)
before = [m.detach().clone() for m in st._masters]
st.step(draft_hidden=draft(x), target_logits=t, lm_head_weight=w)
norm = math.sqrt(sum(float((m - b).pow(2).sum()) for m, b in zip(st._masters, before)))
check("update norm equals the clip value", abs(norm - 1e-3) < 1e-6, f"norm={norm:.6e}")
draft, st = make(lr=1.0, grad_clip=0.0, optimizer_dtype="float32")
st._optimizer = torch.optim.SGD(st._step_params, lr=1.0)
before = [m.detach().clone() for m in st._masters]
st.step(draft_hidden=draft(x), target_logits=t, lm_head_weight=w)
norm = math.sqrt(sum(float((m - b).pow(2).sum()) for m, b in zip(st._masters, before)))
check("grad_clip <= 0 disables clipping", norm > 1e-2, f"norm={norm:.6e}")


def bf16_stepped_grads(**cfg_kwargs):
    """The gradients a `bfloat16` step receives, as float32 tensors."""
    draft, st = make(optimizer_dtype="bfloat16", lr=1e-3, **cfg_kwargs)
    grads = []
    step_param = st._optimizer.step_param

    def capture(i, grad):
        grads.append(grad.clone())
        step_param(i, grad)

    st._optimizer.step_param = capture
    st.step(draft_hidden=draft(x), target_logits=t, lm_head_weight=w)
    return grads


clipped = bf16_stepped_grads(grad_clip=1e-3)
norm = math.sqrt(sum(float(g.pow(2).sum()) for g in clipped))
check(
    "bfloat16: the clip applies to the float32 gradients of the step",
    abs(norm - 1e-3) < 1e-6 and all(g.dtype == torch.float32 for g in clipped),
    f"norm={norm:.6e}",
)
unclipped = bf16_stepped_grads(grad_clip=0.0)
norm = math.sqrt(sum(float(g.pow(2).sum()) for g in unclipped))
check("bfloat16: grad_clip <= 0 disables clipping", norm > 1e-2, f"norm={norm:.6e}")

print("\n[17] lm_head layout check")


def head(rows, num_org, num_org_padded, num_added=0, dtype=torch.bfloat16):
    return SimpleNamespace(
        weight=torch.zeros(rows, 4, dtype=dtype),
        shard_indices=SimpleNamespace(
            num_org_elements=num_org,
            num_org_elements_padded=num_org_padded,
            num_added_elements=num_added,
        ),
    )


check(
    "tp=1: padded head resolves to the vocabulary rows",
    resolve_lm_head_rows(head(128, 100, 128), vocab_size=100, tp_size=1) == 100,
)
check(
    "tp=4: unpadded shard resolves to vocab / tp rows",
    resolve_lm_head_rows(head(64, 64, 64), vocab_size=256, tp_size=4) == 64,
)
def rejected_with(name, expected, fn):
    msg = raises(ValueError, fn)
    check(name, msg is not None and expected in msg, (msg or "no error")[:80])


rejected_with(
    "tp>1: padded shard is rejected",
    "requires unpadded lm_head shards: this rank holds 61 vocabulary rows padded to 64",
    lambda: resolve_lm_head_rows(head(64, 61, 64), vocab_size=253, tp_size=4),
)
rejected_with(
    "added vocabulary is rejected",
    "does not support a target lm_head with added vocabulary",
    lambda: resolve_lm_head_rows(head(128, 100, 100, 4), vocab_size=100, tp_size=1),
)
rejected_with(
    "quantized head is rejected",
    "requires an unquantized (fp16 / bf16 / fp32) target lm_head, got dtype torch.int8",
    lambda: resolve_lm_head_rows(
        head(100, 100, 100, dtype=torch.int8), vocab_size=100, tp_size=1
    ),
)
rejected_with(
    "row count must match the vocabulary",
    "lm_head holds 100 vocabulary rows on this rank, expected 120 / 1",
    lambda: resolve_lm_head_rows(head(100, 100, 100), vocab_size=120, tp_size=1),
)
rejected_with(
    "a head without weights is rejected",
    "exposes no lm_head.weight",
    lambda: resolve_lm_head_rows(SimpleNamespace(), vocab_size=100, tp_size=1),
)
rejected_with(
    "a vocabulary that tp_size does not divide is rejected",
    "expected 101 / 4",
    lambda: resolve_lm_head_rows(head(25, 25, 25), vocab_size=101, tp_size=4),
)

print("\n[18] KL gate: a KL exactly at the threshold is not skipped")
# The gate skips an update when KL < threshold. The threshold is set to the
# very KL value that an ungated step on identical inputs returns.
x, t, w = make_data()
draft, st = make(lr=1e-2, kl_threshold=1e-30)
kl_at = st.step(draft_hidden=draft(x), target_logits=t, lm_head_weight=w)
check("fixture: the KL is positive and finite", kl_at is not None and 0 < kl_at < 1e9)
draft, st = make(lr=1e-2, kl_threshold=kl_at)
before = clones(draft.parameters())
kl_value = st.step(draft_hidden=draft(x), target_logits=t, lm_head_weight=w)
check(
    "KL == threshold: the update is applied",
    kl_value == kl_at
    and st.update_ct == 1
    and st.kl_skip_ct == 0
    and not same(draft.parameters(), before),
    f"kl={kl_value!r} threshold={kl_at!r}",
)
draft, st = make(lr=1e-2, kl_threshold=math.nextafter(kl_at, math.inf))
before = clones(draft.parameters())
kl_value = st.step(draft_hidden=draft(x), target_logits=t, lm_head_weight=w)
check(
    "KL just below the threshold: the update is skipped",
    st.update_ct == 0 and st.kl_skip_ct == 1 and same(draft.parameters(), before),
)

print("\n[18b] the lk loss")
GAMMA_LK = 3.0
draft, st = make(tts_loss="lk", pos_weight_gamma=GAMMA_LK, lr=1e-2)
x, t, w = make_data(bs=2)
with torch.no_grad():
    hidden = draft(x)
    term, ce = st.compute_loss(draft_hidden=hidden, target_logits=t, lm_head_weight=w)
target_rows = t.double().view(2, BLOCK, -1)
pos_weights = [math.exp(-k / GAMMA_LK) for k in range(BLOCK - 1)]
expected = 0.0
for b in range(2):
    for k in range(1, BLOCK):
        logits = (hidden[b, k].to(w.dtype) @ w.T).double()
        q = torch.softmax(logits, dim=-1)
        p = torch.softmax(target_rows[b, k - 1], dim=-1)
        expected += pos_weights[k - 1] * -math.log(float(torch.min(p, q).sum()))
expected /= 2 * sum(pos_weights)
check(
    "the term matches an independent computation, with no CE term",
    ce is None and abs(float(term) - expected) < 1e-4 * max(1.0, expected),
    f"{float(term):.6f}/{expected:.6f}",
)
before = float(term)
pretrained = clones(draft.parameters())
for _ in range(5):
    st.step(draft_hidden=draft(x), target_logits=t, lm_head_weight=w)
with torch.no_grad():
    after, _ = st.compute_loss(draft_hidden=draft(x), target_logits=t, lm_head_weight=w)
check(
    "updates change the weights and lower the loss",
    max_abs_diff(draft.parameters(), pretrained) > 0 and float(after) < before,
    f"{before:.4f} -> {float(after):.4f}",
)
draft, st = make(tts_loss="lk", kl_threshold=1e9)
pretrained = clones(draft.parameters())
gate = st.step(draft_hidden=draft(x), target_logits=t, lm_head_weight=w)
check(
    "the threshold gate reads the lk term",
    gate is not None
    and st.kl_skip_ct == 1
    and max_abs_diff(draft.parameters(), pretrained) == 0,
)
check("the default is lk", DFlashTTSConfig().tts_loss == "lk")

print("\n[19] a non-finite loss or a non-finite gradient is detected on its own")
# Finite loss, non-finite gradient: a hook on a parameter turns its gradient
# into inf / nan during backward. Non-finite loss, finite gradient: a NaN
# constant is added to the KL term, which passes the gradient through.
x, t, w = make_data()
for dtype in ("float32", "bfloat16"):
    # Control: the loss is finite in the finite-loss case.
    draft, st = make(lr=1e-2, optimizer_dtype=dtype)
    loss, _, _ = loss_value(st, draft, x, t, w)
    check(f"{dtype}: fixture: the loss is finite", math.isfinite(loss))

    draft, st = make(lr=1e-2, optimizer_dtype=dtype)
    pretrained = clones(draft.parameters())
    handle = draft.out.weight.register_hook(lambda g: g * float("inf"))
    st.step(draft_hidden=draft(x), target_logits=t, lm_head_weight=w)
    handle.remove()
    changed = not same(draft.parameters(), pretrained)
    count_before_fence = st.nonfinite_ct
    st.wait_for_pending()
    check(
        f"{dtype}: finite loss, non-finite gradient: counted and the weights restored",
        changed
        and count_before_fence == 0
        and st.nonfinite_ct == 1
        and same(draft.parameters(), pretrained)
        and (st._masters is None or same(st._masters, [b.float() for b in pretrained]))
        and not st._dirty,
    )

    draft, st = make(lr=1e-2, optimizer_dtype=dtype)
    pretrained = clones(draft.parameters())
    real_compute_loss = st.compute_loss

    def nan_kl_loss(**kwargs):
        kl, ce = real_compute_loss(**kwargs)
        return kl + float("nan"), ce

    # Control: this construction really gives a NaN loss with finite grads.
    kl, ce = nan_kl_loss(draft_hidden=draft(x), target_logits=t, lm_head_weight=w)
    probe = st.cfg.tts_lambda * kl + (1.0 - st.cfg.tts_lambda) * ce
    probe.backward()
    check(
        f"{dtype}: fixture: NaN loss with finite gradients",
        not bool(torch.isfinite(probe))
        and all(
            p.grad is not None and bool(torch.isfinite(p.grad).all())
            for p in draft.parameters()
        ),
    )
    for p in draft.parameters():
        p.grad = None
    st.compute_loss = nan_kl_loss
    st.step(draft_hidden=draft(x), target_logits=t, lm_head_weight=w)
    changed = not same(draft.parameters(), pretrained)
    count_before_fence = st.nonfinite_ct
    st.wait_for_pending()
    check(
        f"{dtype}: non-finite loss, finite gradient: counted and the weights restored",
        changed
        and count_before_fence == 0
        and st.nonfinite_ct == 1
        and same(draft.parameters(), pretrained)
        and (st._masters is None or same(st._masters, [b.float() for b in pretrained]))
        and not st._dirty,
    )

print("\n[19] the request summary warns when updates changed no weight")
x, t, w = make_data()
for dtype, text in (("float32", "float32 masters"), ("bfloat16", "bfloat16 spacing")):
    draft, st = make(optimizer_dtype=dtype, lr=1e-9, init=coarse_weights)
    st.step(draft_hidden=draft(x), target_logits=t, lm_head_weight=w)
    with patch.object(dflash_tts.logger, "warning") as warning:
        st.log_request_summary()
    check(
        f"{dtype}: an update below the spacing is reported",
        warning.call_count == 1 and text in warning.call_args.args[0],
        warning.call_args.args[0] if warning.call_args else "no warning",
    )
    draft, st = make(optimizer_dtype=dtype, lr=1e-1, init=coarse_weights)
    st.step(draft_hidden=draft(x), target_logits=t, lm_head_weight=w)
    with patch.object(dflash_tts.logger, "warning") as warning:
        st.log_request_summary()
    check(f"{dtype}: an update that moved weights is not", warning.call_count == 0)

print("\n[20] the optimizer is shared, and importing it has no side effects")
check("dflash_tts does not import eagle_tts", not IMPORTS_EAGLE_TTS)
from sglang.srt.speculative import eagle_tts  # noqa: E402

check(
    "eagle_tts uses the same class", eagle_tts.Bf16StateAdam is dflash_tts.Bf16StateAdam
)
msg = raises(ValueError, lambda: Bf16StateAdam([], lr=1e-3))
check(
    "the error for no parameters names no specific mode",
    msg is not None and "EAGLE" not in msg and "DFlash" not in msg,
    msg or "",
)

print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
