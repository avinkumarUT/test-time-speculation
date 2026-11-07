"""Batched LoRA deltas must stay differentiable under deterministic inference.

`enable_batch_invariant_mode` (batch_invariant_ops.py) replaces `torch.bmm`
with a Python-level function. That function bypasses the dispatcher and
therefore autograd: its output has no grad_fn, so training code that calls
`torch.bmm` silently stops producing gradients. The batched replay delta
(_lora_delta_many) therefore calls `torch.ops.aten.bmm`, which records the
autograd node and then dispatches to the same batch-invariant kernel.

The test runs the real methods on a minimal stand-in for the state object and
checks, with the replacement active, that:
  1. the output of plain torch.bmm has no grad_fn (the premise of the test);
  2. the delta path returns grad-tracked outputs;
  3. backward routes gradients to each request's own flat leaf only;
  4. numerics match the per-request reference (x @ A^T @ B^T * scaling);
  5. the TTS modules contain no direct torch.bmm calls.

On CPU the replacement is simulated by swapping torch.bmm for a no-grad
wrapper, because the real kernel needs triton and CUDA. On CUDA, when it is
available, the real enable_batch_invariant_mode is used.

Run: python test/registered/spec/dflash_tts/test_aten_bmm_autograd.py
"""

import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../python")
)

from sglang.srt.speculative.dflash_lora_tts import DFlashLoRATTSState  # noqa: E402

R, N, RANK, IN_F, OUT_F, SCALING = 3, 5, 4, 16, 8, 2.0
KEY = (0, "q")
GROUP_A, GROUP_B = "a_repl", "b_col"


def make_state():
    """Stand-in state with only the attributes the two real methods read."""
    st = SimpleNamespace()
    st.cfg = SimpleNamespace(rank=RANK, scaling=SCALING)
    st.tp_size = 1
    st._a_sharded = {KEY: False}
    st._shapes = {KEY: (IN_F, OUT_F)}
    st._flat_layout = {KEY: (GROUP_A, 0, GROUP_B, 0)}
    st._leaf_views = DFlashLoRATTSState._leaf_views.__get__(st)
    return st


def make_flats(device, gen):
    return [
        {
            GROUP_A: torch.randn(
                RANK * IN_F, dtype=torch.float32, device=device, generator=gen
            ).requires_grad_(),
            GROUP_B: torch.randn(
                OUT_F * RANK, dtype=torch.float32, device=device, generator=gen
            ).requires_grad_(),
        }
        for _ in range(R)
    ]


def reference_delta(flats_list, x):
    outs = []
    for i, f in enumerate(flats_list):
        a = f[GROUP_A].detach().view(RANK, IN_F)
        b = f[GROUP_B].detach().view(OUT_F, RANK)
        outs.append((x[i] @ a.T) @ b.T * SCALING)
    return torch.stack(outs)


def check_delta_path(name, out, flats_list, ref):
    assert out is not None, f"{name}: returned None for a laid-out key"
    assert out.grad_fn is not None, (
        f"{name}: output has NO grad_fn under the batch-invariant "
        f"monkeypatch -- training would be silently severed"
    )
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)

    # Gradient isolation: a loss on request 1 only must put grad on request
    # 1's leaves and leave requests 0/2 untouched.
    for f in flats_list:
        f[GROUP_A].grad = None
        f[GROUP_B].grad = None
    out[1].sum().backward(retain_graph=True)
    for i, f in enumerate(flats_list):
        for g in (GROUP_A, GROUP_B):
            grad = f[g].grad
            if i == 1:
                assert grad is not None and grad.abs().sum() > 0, (
                    f"{name}: no gradient reached request {i} leaf {g}"
                )
            else:
                assert grad is None or grad.abs().sum() == 0, (
                    f"{name}: cross-request gradient leak into req {i} / {g}"
                )
    print(f"  PASS {name}: grad_fn present, numerics match, grads isolated")


def run_paths(device, gen):
    st = make_state()
    flats_list = make_flats(device, gen)
    x = torch.randn(R, N, IN_F, dtype=torch.float32, device=device, generator=gen)
    ref = reference_delta(flats_list, x)

    # Premise: the Python-level replacement of torch.bmm bypasses autograd.
    a = torch.randn(2, 3, 4, device=device, requires_grad=True, generator=gen)
    b = torch.randn(2, 4, 5, device=device, generator=gen)
    assert torch.bmm(a, b).grad_fn is None, (
        "monkeypatched torch.bmm unexpectedly tracks grad; the test needs a "
        "bmm without autograd"
    )
    assert torch.ops.aten.bmm(a, b).grad_fn is not None, (
        "torch.ops.aten.bmm does not record autograd"
    )

    out = DFlashLoRATTSState._lora_delta_many(st, flats_list, *KEY, x)
    check_delta_path("_lora_delta_many", out, flats_list, ref)


def nograd_bmm_factory(real_bmm):
    def bmm_nograd(a, b, **kw):
        with torch.no_grad():
            return real_bmm(a, b, **kw)

    return bmm_nograd


def test_cpu_simulated():
    print("[cpu] simulated python-level monkeypatch")
    real = torch.bmm
    torch.bmm = nograd_bmm_factory(real)
    try:
        run_paths("cpu", torch.Generator("cpu").manual_seed(0))
    finally:
        torch.bmm = real


def test_cuda_real_mode():
    if not torch.cuda.is_available():
        print("[cuda] skipped: no CUDA")
        return
    from sglang.srt.batch_invariant_ops.batch_invariant_ops import (
        disable_batch_invariant_mode,
        enable_batch_invariant_mode,
    )

    print("[cuda] real enable_batch_invariant_mode")
    enable_batch_invariant_mode()
    try:
        gen = torch.Generator("cuda").manual_seed(0)
        run_paths("cuda", gen)
    finally:
        disable_batch_invariant_mode()


def test_no_torch_bmm_calls_left():
    here = os.path.dirname(os.path.abspath(__file__))
    spec = os.path.join(here, "../../../../python/sglang/srt/speculative")
    for mod in ("dflash_lora_tts.py", "eagle_lora_tts.py", "eagle_tts.py"):
        path = os.path.join(spec, mod)
        assert os.path.exists(path), f"module to scan not found: {path}"
        offenders = [
            f"{mod}:{ln}"
            for ln, line in enumerate(open(path), 1)
            if "torch.bmm(" in line
        ]
        assert not offenders, f"raw torch.bmm call(s) remain: {offenders}"
    print("  PASS source scan: no raw torch.bmm( calls in TTS modules")


if __name__ == "__main__":
    test_no_torch_bmm_calls_left()
    test_cpu_simulated()
    test_cuda_real_mode()
    print("ALL PASS")
