"""Optimizer shared by the TTS modes that store their state in bfloat16.

This module has no side effects at import and depends only on torch.
"""

from __future__ import annotations

from typing import List, Optional

import torch


class Bf16StateAdam:
    """Adam over unpooled bf16 parameters, with fp32 arithmetic.

    The parameters are their own master and the moments are bf16 tensors, so
    no fp32 copy outlives a step. A step works on fp32 leaves whose gradients
    the caller clips like those of an fp32 master, and for one parameter at a
    time widens the moments, runs torch's functional Adam (the function
    `torch.optim.Adam.step` dispatches to) and rounds the weight and the
    moments once when it stores them. The result is a `torch.optim.Adam` step
    on the widened state, rounded.

    The rounding also acts on the moments. A second moment that loses less than
    half its bf16 spacing in a step keeps its value, and with `beta2` = 0.999
    that holds whenever the new squared gradient is small. It therefore stops
    decaying, and when gradients shrink the steps become smaller than those of
    fp32 state.

    Stepping `torch.optim.Adam` on the bf16 parameters directly is not the same
    update: it rounds after every operation of the update, so its moments and
    weights depart from this result, and its clipping norm is a bf16 value.

    There are two ways to step. `widen` and `step` hold the fp32 weights of
    all parameters for the whole step. `step_param` widens the weight of one
    parameter when its turn comes, for a caller that keeps the gradients
    itself.

    A moment tensor is allocated at a parameter's first step, and parameters
    without a gradient are skipped, as in `torch.optim.Adam`. The owner
    rebuilds this object at every reset.
    """

    def __init__(self, params, lr):
        from torch.optim.adam import adam as functional_adam

        self.params = list(params)
        if not self.params:
            raise ValueError("TTS: no trainable parameters for the optimizer.")
        self._functional_adam = functional_adam
        self.lr = float(lr)
        self.m: List[Optional[torch.Tensor]] = [None] * len(self.params)
        self.v: List[Optional[torch.Tensor]] = [None] * len(self.params)
        self.step_ct: List[int] = [0] * len(self.params)

    @torch.no_grad()
    def widen(self) -> List[torch.Tensor]:
        """One fp32 leaf per parameter, carrying the fp32 gradient if it has one."""
        leaves = []
        for p in self.params:
            leaf = p.detach().float()
            leaf.grad = None if p.grad is None else p.grad.detach().float()
            leaves.append(leaf)
        return leaves

    @torch.no_grad()
    def step(self, leaves: List[torch.Tensor]) -> None:
        """Adam on the leaves of `widen`; stores the rounded results."""
        for i, leaf in enumerate(leaves):
            if leaf.grad is not None:
                self._step_leaf(i, leaf)

    @torch.no_grad()
    def step_param(self, i: int, grad: torch.Tensor) -> None:
        """Adam on parameter `i` with the fp32 gradient `grad`; stores the
        rounded results."""
        leaf = self.params[i].detach().float()
        leaf.grad = grad
        self._step_leaf(i, leaf)

    def _step_leaf(self, i: int, leaf: torch.Tensor) -> None:
        p = self.params[i]
        if self.m[i] is None:
            self.m[i] = torch.zeros_like(p)
            self.v[i] = torch.zeros_like(p)
        m = self.m[i].float()
        v = self.v[i].float()
        self._functional_adam(
            [leaf],
            [leaf.grad],
            [m],
            [v],
            [],
            [torch.tensor(float(self.step_ct[i]))],
            # None selects the implementation of a default-constructed
            # torch.optim.Adam, as the fp32 master path uses.
            foreach=None,
            capturable=False,
            differentiable=False,
            fused=None,
            grad_scale=None,
            found_inf=None,
            has_complex=False,
            decoupled_weight_decay=False,
            amsgrad=False,
            beta1=0.9,
            beta2=0.999,
            lr=self.lr,
            weight_decay=0.0,
            eps=1e-8,
            maximize=False,
        )
        self.step_ct[i] += 1
        p.copy_(leaf)
        self.m[i].copy_(m)
        self.v[i].copy_(v)

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Same interface as `torch.optim.Optimizer.zero_grad`."""
        for p in self.params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()
