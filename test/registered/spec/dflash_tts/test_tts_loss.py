"""The TTS losses of the per-request DFlash path and the loss shared by all modes.

Checks `lk_per_position` and `DFlashLoRATTSState._compute_loss` and
`_compute_loss_many` against the formulas written out in float64, for both
values of `tts_loss`, and that the batched loss is the per-request loss.

Needs no GPU, checkpoint or server:

  python test/registered/spec/dflash_tts/test_tts_loss.py
"""

import math
import os
import sys
from types import SimpleNamespace

os.environ.setdefault("SGLANG_USE_CPU_ENGINE", "1")

import torch
import torch.nn.functional as F

from sglang.srt.speculative.dflash_lora_tts import (
    DFlashLoRATTSConfig,
    DFlashLoRATTSState,
)
from sglang.srt.speculative.tts_loss import lk_per_position

FAILS = []


def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' — ' + extra) if extra else ''}")
    if not cond:
        FAILS.append(name)


R, POS, VOCAB, GAMMA, LAM = 3, 5, 40, 3.0, 0.7
torch.manual_seed(0)
draft = torch.randn(R, POS, VOCAB) * 3
target = torch.randn(R, POS, VOCAB) * 3
weights = torch.exp(-torch.arange(POS, dtype=torch.float32) / GAMMA)
weights = weights / weights.sum()


def reference(d_logits, t_logits, loss, cap=None):
    """The loss of one request, position by position, in float64."""
    total = 0.0
    for k in range(POS):
        d = d_logits[k].double()
        if cap is not None:
            d = cap * torch.tanh(d / cap)
        log_q = F.log_softmax(d, dim=-1)
        log_p = F.log_softmax(t_logits[k].double(), dim=-1)
        if loss == "lk":
            term = -math.log(float(torch.min(log_p.exp(), log_q.exp()).sum()))
        else:
            kl = float((log_p.exp() * (log_p - log_q)).sum())
            ce = float(-log_q[int(d.argmax())])
            term = LAM * kl + (1.0 - LAM) * ce
        total += float(weights[k]) * term
    return total


def stub(loss, cap=None):
    """The fields of the state that the loss methods read."""
    state = SimpleNamespace(
        cfg=DFlashLoRATTSConfig(tts_loss=loss, tts_lambda=LAM, logit_softcap=cap),
        _pos_weights=weights,
    )
    state._target_constants = lambda t: DFlashLoRATTSState._target_constants(state, t)
    return state


print("\n[1] lk_per_position")
p = F.softmax(target[0], dim=-1)
log_q = F.log_softmax(draft[0], dim=-1)
expected = -torch.log(torch.min(p, log_q.exp()).sum(dim=-1))
check("matches the formula", torch.allclose(lk_per_position(p, log_q), expected))
check(
    "is zero when the distributions are equal",
    float(lk_per_position(p, p.log()).abs().max()) < 1e-5,
)
disjoint = torch.zeros(1, 4)
disjoint[0, 0] = 1.0
log_q = torch.full((1, 4), -1000.0)
log_q[0, 1] = 0.0
log_q.requires_grad_(True)
value = lk_per_position(disjoint, log_q)
value.sum().backward()
check(
    "stays finite on disjoint distributions",
    bool(torch.isfinite(value).all()) and bool(torch.isfinite(log_q.grad).all()),
    f"{float(value):.2f}",
)

print("\n[2] the DFlash per-request loss")
for loss in ("lk", "kl"):
    for cap in (None, 10.0):
        state = stub(loss, cap)
        got = []
        for r in range(R):
            p_t, ent = state._target_constants(target[r])
            got.append(
                float(DFlashLoRATTSState._compute_loss(state, draft[r], p_t, ent))
            )
        ref = [reference(draft[r], target[r], loss, cap) for r in range(R)]
        check(
            f"{loss}, softcap {cap}: one request matches the formula",
            all(abs(a - b) < 1e-4 * max(1.0, abs(b)) for a, b in zip(got, ref)),
            f"{got[0]:.6f}/{ref[0]:.6f}",
        )
        p_t, ent = state._target_constants(target)
        many = DFlashLoRATTSState._compute_loss_many(
            state, draft.reshape(R * POS, VOCAB), p_t, ent
        )
        check(
            f"{loss}, softcap {cap}: the batched loss is the per-request loss",
            many.shape == (R,)
            and all(abs(float(many[r]) - got[r]) < 1e-5 for r in range(R)),
        )

check("lk keeps no entropy constant", stub("lk")._target_constants(target)[1] is None)
check("kl keeps the entropy constant", stub("kl")._target_constants(target)[1] is not None)
check("the default is lk", DFlashLoRATTSConfig().tts_loss == "lk")

print("\n[3] the gradient of lk reaches every position with weight")
state = stub("lk")
leaf = draft[0].clone().requires_grad_(True)
p_t, ent = state._target_constants(target[0])
DFlashLoRATTSState._compute_loss(state, leaf, p_t, ent).backward()
check(
    "each position has a finite, non-zero gradient",
    bool(torch.isfinite(leaf.grad).all())
    and all(float(leaf.grad[k].abs().sum()) > 0 for k in range(POS)),
)

print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
