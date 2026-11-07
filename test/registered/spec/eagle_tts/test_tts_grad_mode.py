"""Checks that `tts_grad_mode` restores autograd through a serving forward.

Three things independently disable autograd on the draft forward of the
serving path. None of them raises: decoding keeps working and the TTS update
simply receives no gradient.

  1. `LlamaForCausalLM.forward` is decorated `@torch.no_grad()` (models/llama.py),
     and `LlamaForCausalLMEagle3` inherits it. A `forward_pre_hook` cannot undo
     this: the hook runs first, then the decorator disables grad on entry.
  2. The scheduler event loop is wrapped in `DynamicGradMode` (scheduler.py),
     which by default is `torch.set_grad_enabled(False)` for the whole loop.
  3. Fused kernels (`MultiPlatformOp` subclasses) have no autograd registration,
     so their outputs come back with `grad_fn=None`.

Also checks the explicit backward of the differentiable TP collectives. Runs
on CPU with toy modules; needs no checkpoint.

  python test/registered/spec/eagle_tts/test_tts_grad_mode.py
"""

import sys

import torch
from torch import nn

from sglang.srt.speculative.eagle_tts import tts_grad_mode
from sglang.srt.utils import DynamicGradMode

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}{(' - ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


class Base(nn.Module):
    """Stands in for LlamaForCausalLM: forward carries @torch.no_grad()."""

    @torch.no_grad()
    def forward(self, x):
        return self.body(x)


class Draft(Base):
    """Stands in for LlamaForCausalLMEagle3: inherits the decorated forward."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 8, bias=False)
        self.head = nn.Linear(8, 4, bias=False)

    def body(self, x):
        return self.head(self.lin(x))


def main():
    torch.manual_seed(0)
    model = Draft()
    x = torch.randn(3, 8)
    for p in model.parameters():
        p.requires_grad_(True)

    print("baseline: without tts_grad_mode")
    # The serving path's call sequence, without tts_grad_mode.
    with DynamicGradMode():  # scheduler event loop
        with torch.enable_grad():  # enable_grad at the call site
            out = model(x)
    check(
        "inherited @torch.no_grad() severs the tape",
        out.grad_fn is None,
        "confirms enable_grad() alone is not enough",
    )

    print("with tts_grad_mode")
    with DynamicGradMode():
        with torch.enable_grad(), tts_grad_mode(model):
            out = model(x)
    check("logits carry grad_fn", out.grad_fn is not None)
    check("logits.requires_grad", out.requires_grad)

    if out.requires_grad:
        out.float().pow(2).mean().backward()
        starved = [n for n, p in model.named_parameters() if p.grad is None]
        check(
            "every parameter received a gradient",
            not starved,
            f"starved={starved}" if starved else "2/2",
        )
        check(
            "gradients are finite and non-zero",
            all(
                torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                for p in model.parameters()
            ),
        )

    print("restoration after the context exits")
    check(
        "instance forward override removed",
        "forward" not in model.__dict__,
        "class @torch.no_grad() forward is back in effect",
    )
    with DynamicGradMode():
        with torch.enable_grad():
            out2 = model(x)
    check("serving path is no_grad again", out2.grad_fn is None)

    from sglang.srt.layers.utils.multi_platform import MultiPlatformOp

    check(
        "MultiPlatformOp.forward restored",
        getattr(MultiPlatformOp.forward, "__name__", "") != "forward_native_dispatch",
    )

    check_collectives()

    print("\n" + ("FAILED: " + ", ".join(FAILURES) if FAILURES else "all checks passed"))
    return 1 if FAILURES else 0


def check_collectives():
    """The TP collectives' explicit backwards, checked against autograd.

    `c10d::allreduce_` and `c10d::_allgather_base_` have no autograd kernel, so
    backward through them yields zero, which zeroes the gradient of every
    parameter upstream. `_DifferentiableAllReduce` / `_DifferentiableAllGather`
    supply the backward. They are checked here without a process group, by
    substituting a local function for the collective and comparing against the
    expected gradient.
    """
    from sglang.srt.speculative.eagle_tts import (
        _DifferentiableAllGather,
        _DifferentiableAllReduce,
    )

    print("differentiable TP collectives")

    # all-reduce: y = sum_r x_r, every rank holds the same y and the same
    # dL/dy, so dL/dx_r = dL/dy (identity).
    world, rank = 4, 1
    x = torch.randn(3, 5, requires_grad=True)
    other = torch.randn(3, 5)  # the other ranks' contributions, constants here
    y = _DifferentiableAllReduce.apply(x, lambda t: t + other)
    g = torch.randn(3, 5)
    y.backward(g)
    check("all-reduce backward is identity", torch.equal(x.grad, g))

    # all-gather: y = concat_r(x_r) along dim; dL/dx_r is this rank's slice.
    shard = 5
    x2 = torch.randn(3, shard, requires_grad=True)

    def fake_all_gather(t, dim):
        parts = [torch.zeros_like(t)] * world
        parts = list(parts)
        parts[rank] = t
        return torch.cat(parts, dim=dim)

    y2 = _DifferentiableAllGather.apply(x2, -1, rank, world, fake_all_gather)
    check("all-gather forward shape", tuple(y2.shape) == (3, shard * world))
    g2 = torch.randn(3, shard * world)
    y2.backward(g2)
    expected = g2[:, rank * shard : (rank + 1) * shard]
    check("all-gather backward is this rank's slice", torch.equal(x2.grad, expected))

    # The gradient must also be non-zero: without the explicit backward,
    # autograd yields exactly zero through a collective.
    check(
        "collective gradients are non-zero",
        x.grad.abs().sum() > 0 and x2.grad.abs().sum() > 0,
    )


if __name__ == "__main__":
    sys.exit(main())
