"""Unit tests for EagleTTSState: stride, async stepping, vocab mapping, reset.

Runs on a toy draft module; needs no checkpoint and no running server.

  python test/registered/spec/eagle_tts/test_eagle_tts.py

`TTS_TEST_DEV` selects the device (default `cuda`). With `TTS_TEST_DEV=cpu` the
async-vs-sync comparison is skipped, since it needs a CUDA stream.
"""
import dataclasses
import os
import sys
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.speculative.eagle_tts import (
    EagleTTSConfig,
    EagleTTSState,
    tts_loss_rows,
)

DEV = os.environ.get("TTS_TEST_DEV", "cuda")
HID, DRAFT_VOCAB, TARGET_VOCAB = 64, 32, 128
FAILS = []


def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' — ' + extra) if extra else ''}")
    if not cond:
        FAILS.append(name)


class ToyDraft(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Linear(HID, HID, bias=False)
        self.lm_head = nn.Linear(HID, DRAFT_VOCAB, bias=False)

    def forward(self, x):
        return self.lm_head(torch.relu(self.body(x)))


def make(cfg_kwargs):
    torch.manual_seed(0)
    m = ToyDraft().to(DEV).to(torch.bfloat16)
    cfg = EagleTTSConfig(enable=True, **cfg_kwargs)
    return m, EagleTTSState(draft_model=m, config=cfg, device=DEV)


def logits_pair(m, n_pos=4, mapped=False):
    x = torch.randn(n_pos, HID, device=DEV, dtype=torch.bfloat16)
    draft_logits = m(x)
    vocab = TARGET_VOCAB if mapped else DRAFT_VOCAB
    target_logits = torch.randn(n_pos, vocab, device=DEV, dtype=torch.bfloat16)
    return draft_logits, target_logits


print("\n[1] stride: fires only every `freq` rounds")
m, st = make(dict(freq=5, async_step=False))
pattern = []
for _ in range(11):
    pattern.append(st.should_fire())
    st.advance()
check("fires at rounds 0,5,10", pattern == [i % 5 == 0 for i in range(11)], str(pattern))

print("\n[2] sync step actually changes weights")
m, st = make(dict(lr=1e-2, async_step=False))
before = m.body.weight.detach().clone()
d, t = logits_pair(m)
st.step(draft_logits=d, target_logits=t)
delta = (m.body.weight - before).abs().max().item()
check("weights moved", delta > 0, f"max|delta|={delta:.3e}")
check("step counted", st.step_ct == 1)

print("\n[3] fp32 master keeps tiny lr from vanishing in bf16")
m, st = make(dict(lr=1e-5, async_step=False, fp32_master=True))
before = m.body.weight.detach().clone()
for _ in range(50):
    d, t = logits_pair(m)
    st.step(draft_logits=d, target_logits=t)
drift_master = (m.body.weight - before).abs().max().item()
m2, st2 = make(dict(lr=1e-5, async_step=False, fp32_master=False))
before2 = m2.body.weight.detach().clone()
for _ in range(50):
    d, t = logits_pair(m2)
    st2.step(draft_logits=d, target_logits=t)
drift_plain = (m2.body.weight - before2).abs().max().item()
check("fp32 master accumulates drift", drift_master > 0,
      f"master={drift_master:.3e} plain={drift_plain:.3e}")

print("\n[4] async step + wait_for_pending produces the same weights as sync")
if DEV != "cuda":
    # The side-stream path needs a real CUDA stream; on CPU `_use_async` is
    # False by construction, so there is nothing to compare against sync.
    print("  SKIP  async path requires CUDA (TTS_TEST_DEV=cpu)")
else:
    torch.manual_seed(0)
    ma, sta = make(dict(lr=1e-2, async_step=True))
    torch.manual_seed(0)
    ms, sts = make(dict(lr=1e-2, async_step=False))
    check("async enabled", sta._use_async is True)
    for i in range(3):
        torch.manual_seed(100 + i)
        xa = torch.randn(4, HID, device=DEV, dtype=torch.bfloat16)
        ta = torch.randn(4, DRAFT_VOCAB, device=DEV, dtype=torch.bfloat16)
        sta.step(draft_logits=ma(xa), target_logits=ta)
        sta.wait_for_pending()
        sts.step(draft_logits=ms(xa), target_logits=ta)
    torch.cuda.synchronize()
    diff = (ma.body.weight - ms.body.weight).abs().max().item()
    check("async == sync weights", diff == 0, f"max|diff|={diff:.3e}")

print("\n[5] hot_token_id maps target vocab down to draft vocab")
m, st = make(dict(lr=1e-2, async_step=False))
d, t = logits_pair(m, mapped=True)
hot = torch.randperm(TARGET_VOCAB, device=DEV)[:DRAFT_VOCAB].contiguous()
before = m.body.weight.detach().clone()
out = st.step(draft_logits=d, target_logits=t, hot_token_id=hot)
check("mapped step ran", (m.body.weight - before).abs().max().item() > 0)

print("\n[6] no-grad input is skipped, not crashed")
m, st = make(dict(async_step=False))
with torch.no_grad():
    d = m(torch.randn(4, HID, device=DEV, dtype=torch.bfloat16))
t = torch.randn(4, DRAFT_VOCAB, device=DEV, dtype=torch.bfloat16)
r = st.step(draft_logits=d, target_logits=t)
check("skipped cleanly", r is None and st.skip_ct == 1)

print("\n[7] reset restores the snapshot")
m, st = make(dict(lr=1e-1, async_step=False))
orig = m.body.weight.detach().clone()
d, t = logits_pair(m)
st.step(draft_logits=d, target_logits=t)
moved = (m.body.weight - orig).abs().max().item()
st.reset()
restored = (m.body.weight - orig).abs().max().item()
check("weights restored", moved > 0 and restored == 0,
      f"moved={moved:.3e} after_reset={restored:.3e}")
check("round counter reset", st.round_idx == 0)

print("\n[8] mismatched position counts raise instead of misaligning")
m, st = make(dict(lr=1e-2, async_step=False))
x = torch.randn(6, HID, device=DEV, dtype=torch.bfloat16)
d = m(x)
t = torch.randn(4, DRAFT_VOCAB, device=DEV, dtype=torch.bfloat16)
before = m.body.weight.detach().clone()
try:
    st.step(draft_logits=d, target_logits=t)
    raised = False
except ValueError:
    raised = True
check("row-count mismatch raises ValueError", raised)
check("weights untouched", (m.body.weight - before).abs().max().item() == 0)

print("\n[9] kl_threshold gates the step")
m, st = make(dict(lr=1e-2, async_step=False, kl_threshold=1e9))
before = m.body.weight.detach().clone()
d, t = logits_pair(m)
st.step(draft_logits=d, target_logits=t)
check("gated out", (m.body.weight - before).abs().max().item() == 0)

print("\n[10] tts_loss_rows pairs draft rollout rows with target verify rows")
# bs=2, n_step=3, nd=4. Draft row = b*n_step + i, target row = b*nd + i + 1.
keep, d_rows, t_rows = tts_loss_rows(2, 3, 4, [3, 5], False)
check(
    "all positions kept",
    keep == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    and d_rows == [0, 1, 2, 3, 4, 5]
    and t_rows == [1, 2, 3, 5, 6, 7],
    f"{d_rows} {t_rows}",
)
# Position i is kept only if i + 2 <= accept_len - 1: request 0 (3 accept
# tokens) keeps i=0; request 1 (5 accept tokens) keeps i=0..2.
keep, d_rows, t_rows = tts_loss_rows(2, 3, 4, [3, 5], True)
check(
    "correct drafts only",
    keep == [(0, 0), (1, 0), (1, 1), (1, 2)]
    and d_rows == [0, 3, 4, 5]
    and t_rows == [1, 5, 6, 7],
    f"{d_rows} {t_rows}",
)
keep, d_rows, t_rows = tts_loss_rows(2, 3, 4, [1, 2], True)
check("no correct drafts keeps nothing", keep == [] and d_rows == [] and t_rows == [])

# End to end on synthetic logits. Node tokens are random; the draft's row for
# node i+2 puts its argmax on that node's token, and the target's verify row
# j puts its argmax on node j+1's token when node j+1 is accepted (a correct
# draft) and on a different token otherwise.
torch.manual_seed(0)
bs, n_step, nd = 3, 4, 6
accept_lens = [1, 3, 5]
node_tok = torch.randint(0, DRAFT_VOCAB, (bs, nd))
draft_logits = torch.zeros(bs * n_step, DRAFT_VOCAB)
target_logits = torch.zeros(bs * nd, DRAFT_VOCAB)
for b in range(bs):
    for i in range(n_step):
        draft_logits[b * n_step + i, node_tok[b, i + 2]] = 1.0
    for j in range(nd - 1):
        tok = node_tok[b, j + 1]
        if j + 1 > accept_lens[b] - 1:
            tok = (tok + 1) % DRAFT_VOCAB
        target_logits[b * nd + j, tok] = 1.0
# The verify row j predicts node j+1, i.e. row (b*nd + i + 1) predicts node i+2.
keep, d_rows, t_rows = tts_loss_rows(bs, n_step, nd, accept_lens, True)
d_arg = draft_logits[d_rows].argmax(-1)
t_arg = target_logits[t_rows].argmax(-1)
check(
    "correct drafts agree 100% with the paired target rows",
    len(keep) == 0 + 1 + 3 and bool((d_arg == t_arg).all()),
    f"{int((d_arg == t_arg).sum())}/{len(keep)}",
)
# Negative controls: a shifted pairing disagrees, and the rows past the
# accepted prefix (kept when correct_drafts_only is off) are the rejected ones.
shifted = target_logits[[r + 1 for r in t_rows]].argmax(-1)
check("shifted pairing is detected", not bool((d_arg == shifted).all()))
keep_all, d_all, t_all = tts_loss_rows(bs, n_step, nd, accept_lens, False)
agree = draft_logits[d_all].argmax(-1) == target_logits[t_all].argmax(-1)
expect = torch.tensor([i + 2 <= accept_lens[b] - 1 for b, i in keep_all])
check("rows past the accepted prefix disagree", bool((agree == expect).all()))

print("\n[11] the loss defaults to lk, and both losses train")
check("the config default is lk", EagleTTSConfig().tts_loss == "lk")
check(
    "server arguments without a loss select lk",
    EagleTTSConfig.from_server_args(SimpleNamespace()).tts_loss == "lk",
)
for loss in ("lk", "kl"):
    m, st = make(dict(lr=1e-2, async_step=False, tts_loss=loss))
    d, t = logits_pair(m)
    d_f, t_f = d.detach().float(), t.detach().float()
    log_q = torch.log_softmax(d_f, dim=-1)
    log_p = torch.log_softmax(t_f, dim=-1)
    if loss == "lk":
        per_pos = -torch.min(log_p.exp(), log_q.exp()).sum(-1).clamp_min(1e-8).log()
        expected = per_pos.mean()
    else:
        per_pos = (log_p.exp() * (log_p - log_q)).sum(-1)
        ce = -log_q.max(dim=-1).values
        expected = 0.9 * per_pos.mean() + 0.1 * ce.mean()
    expected = expected.item()
    d_r = d.clone()
    captured = {}
    original_backward = torch.Tensor.backward

    def recording_backward(self, *args, **kwargs):
        captured["loss"] = self.item()
        return original_backward(self, *args, **kwargs)

    torch.Tensor.backward = recording_backward
    try:
        before = m.body.weight.detach().clone()
        st.step(draft_logits=d_r, target_logits=t)
    finally:
        torch.Tensor.backward = original_backward
    check(
        f"{loss}: the loss equals an independent computation",
        abs(captured["loss"] - expected) < 1e-3 * max(1.0, abs(expected)),
        f"{captured['loss']:.5f} vs {expected:.5f}",
    )
    check(f"{loss}: the update moves the weights", (m.body.weight - before).abs().max().item() > 0)
    m, st = make(dict(lr=1e-2, async_step=False, tts_loss=loss))
    d, t = logits_pair(m)
    t_masked = t.clone()
    t_masked[..., ::7] = float("-inf")
    st.step(draft_logits=d, target_logits=t_masked)
    check(f"{loss}: -inf target logits do not skip the update", st.nonfinite_ct == 0 and st.step_ct == 1)

print("\n[12] the clip norm counts a replicated tensor once under tensor parallelism")
import sglang.srt.layers.linear as _linear

_real_all_reduce = _linear.tensor_model_parallel_all_reduce
_linear.tensor_model_parallel_all_reduce = lambda x: x * 2  # two ranks, same local sum
try:
    for label, fp32_master in (("fp32 masters", True), ("bf16-state leaves", False)):
        m, st = make(dict(async_step=False, grad_clip=1e6, fp32_master=fp32_master))
        st.tp_size = 2
        replicated, sharded = m.lm_head.weight, m.body.weight
        st.adapters = SimpleNamespace(replicated_params=lambda: [replicated])
        for p in (replicated, sharded):
            p.grad = torch.full_like(p, 0.5)
        if fp32_master:
            clipped = st._master
            for live, c in zip(st._params, clipped):
                c.grad = live.grad.float()
        else:
            clipped = st._optimizer.widen()
        norm = st._clip_grads(clipped).item()
        expected = (0.25 * replicated.numel() + 2 * 0.25 * sharded.numel()) ** 0.5
        check(
            f"{label}: replicated counted once, sharded summed over ranks",
            abs(norm - expected) < 1e-3 * expected,
            f"{norm:.4f} vs {expected:.4f}",
        )
finally:
    _linear.tensor_model_parallel_all_reduce = _real_all_reduce

print("\n[13] without depth weights a request's loss is still its own mean")
m, st = make(dict(lr=1e-2, async_step=False))
d, t = logits_pair(m, n_pos=6)
d_f, t_f = d.detach().float(), t.detach().float()
per_pos = -torch.min(
    torch.softmax(t_f, dim=-1), torch.softmax(d_f, dim=-1)
).sum(-1).clamp_min(1e-8).log()
groups = torch.tensor([0, 0, 0, 0, 0, 1], device=DEV)
expected = (per_pos[:5].mean() + per_pos[5]).item()
captured = {}
original_backward = torch.Tensor.backward


def recording_backward(self, *args, **kwargs):
    captured["loss"] = self.item()
    return original_backward(self, *args, **kwargs)


torch.Tensor.backward = recording_backward
try:
    st.step(draft_logits=d, target_logits=t, row_groups=groups, num_groups=2)
finally:
    torch.Tensor.backward = original_backward
check(
    "the loss is the sum of per-request means",
    abs(captured["loss"] - expected) < 1e-3 * abs(expected),
    f"{captured['loss']:.5f} vs {expected:.5f}",
)

print("\n[14] argument validation")
from sglang.srt.server_args import ServerArgs

_eagle_args = {f.name: f.default for f in dataclasses.fields(ServerArgs)}
_eagle_args["speculative_num_steps"] = 3


def _validation_error(**overrides):
    args = SimpleNamespace(**{**_eagle_args, **overrides})
    args._validate_tts_loss_args = ServerArgs._validate_tts_loss_args.__get__(args)
    try:
        ServerArgs._validate_eagle_tts(args)
    except ValueError as e:
        return str(e)
    return None


check(
    "gradient accumulation with per-request slots is rejected",
    "accum-steps" in str(
        _validation_error(
            speculative_eagle_tts_mode="lora",
            speculative_eagle_tts_per_request=True,
            speculative_eagle_tts_accum_steps=2,
        )
    ),
)
check(
    "gradient accumulation without per-request slots is accepted",
    _validation_error(speculative_eagle_tts_accum_steps=2) is None,
)

check(
    "the shared lambda is validated",
    "tts-lambda" in str(_validation_error(speculative_tts_lambda=1.5)),
)
check(
    "the shared loss selects the loss",
    EagleTTSConfig.from_server_args(
        SimpleNamespace(speculative_tts_loss="kl")
    ).tts_loss
    == "kl",
)

print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
