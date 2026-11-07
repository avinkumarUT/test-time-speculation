"""The loss term shared by the TTS modes of every speculator."""

import torch


def lk_per_position(p_target: torch.Tensor, log_p_draft: torch.Tensor) -> torch.Tensor:
    """-log(sum_x min(p(x), q(x))) over the last dimension.

    sum_x min(p, q) equals 1 - TV(p, q) and is the probability that a token
    drafted from q is accepted against p, so this minimizes the quantity that
    acceptance depends on directly.
    """
    alpha = torch.min(p_target, log_p_draft.exp()).sum(dim=-1).clamp_min(1e-8)
    return -torch.log(alpha)
