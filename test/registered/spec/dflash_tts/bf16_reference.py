"""Helpers shared by the DFlash TTS tests of the `bfloat16` optimizer dtype."""

import torch


def bits(tensor):
    """The bytes of a tensor. Unlike `torch.equal` this tells -0.0 from 0.0."""
    return bytes(tensor.detach().cpu().clone().untyped_storage())


def same_bits(tensor_a, tensor_b):
    return (
        tensor_a.dtype == tensor_b.dtype
        and tensor_a.shape == tensor_b.shape
        and bits(tensor_a) == bits(tensor_b)
    )


class Bf16StateReference:
    """An independent `bfloat16` update, written without the code under test.

    `torch.optim.Adam` in float32, with `clip_grad_norm_`, on the state widened
    to float32. The weights and the moments are rounded to bfloat16 once per
    update and kept only in that form. `clip` <= 0 disables clipping.
    """

    def __init__(self, params, lr, clip=0.0):
        self.params = list(params)
        self.lr = lr
        self.clip = clip
        self.m = [None] * len(self.params)
        self.v = [None] * len(self.params)
        self.step_ct = [0] * len(self.params)

    def update(self, grads):
        """One update; `grads` has one tensor or None per parameter.

        Returns the gradient norm before clipping, None without clipping.
        """
        stepped = [i for i, grad in enumerate(grads) if grad is not None]
        leaves = [self.params[i].detach().float() for i in stepped]
        for i, leaf in zip(stepped, leaves):
            leaf.grad = grads[i].detach().float()
        norm = None
        if self.clip > 0:
            norm = float(torch.nn.utils.clip_grad_norm_(leaves, self.clip))
        optimizer = torch.optim.Adam(leaves, lr=self.lr)
        for i, leaf in zip(stepped, leaves):
            if self.step_ct[i] > 0:
                optimizer.state[leaf] = {
                    "step": torch.tensor(float(self.step_ct[i])),
                    "exp_avg": self.m[i].float(),
                    "exp_avg_sq": self.v[i].float(),
                }
        optimizer.step()
        with torch.no_grad():
            for i, leaf in zip(stepped, leaves):
                moments = optimizer.state[leaf]
                self.params[i].copy_(leaf)
                self.m[i] = moments["exp_avg"].to(torch.bfloat16)
                self.v[i] = moments["exp_avg_sq"].to(torch.bfloat16)
                self.step_ct[i] += 1
        return norm


def optimizer_moments(state):
    """The moment tensors a `DFlashTTSState` holds, for either optimizer."""
    optimizer = state._optimizer
    if hasattr(optimizer, "m"):
        return [t for t in list(optimizer.m) + list(optimizer.v) if t is not None]
    return [
        value
        for group in optimizer.state.values()
        for key, value in group.items()
        if key != "step"
    ]


def optimizer_is_fresh(state):
    """Whether the optimizer holds no moment and no step count."""
    optimizer = state._optimizer
    if hasattr(optimizer, "m"):
        return not optimizer_moments(state) and not any(optimizer.step_ct)
    return len(optimizer.state) == 0
