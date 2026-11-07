"""Per-request adapter pool: slot isolation, recycling, per-slot clipping.

Tests `PoolAdam`, the per-request loss reduction and the per-slot gradient
clip. CPU-only; needs no checkpoint. These properties cannot be observed in a
single-request test, and violating them raises no error: a slot that leaks
into another only degrades the drafts.

The file has no `__main__` block; run it with pytest:

  python -m pytest test/registered/spec/eagle_tts/test_pool_adam.py -q
"""
import torch

from sglang.srt.speculative.eagle_lora_tts import PoolAdam

POOL, ROWS, COLS, LR = 4, 3, 5, 0.1


def _pool_params():
    a = torch.zeros(POOL, ROWS, COLS)
    b = torch.zeros(POOL, COLS)
    return [a, b]


def _set_grads(params, rows, value):
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        p.grad.zero_()
        for r in rows:
            p.grad[r] = value


def test_step_touches_only_the_slots_that_fired():
    params = _pool_params()
    opt = PoolAdam(params, POOL, lr=LR)
    _set_grads(params, [1], 1.0)
    opt.step(torch.tensor([1]))

    for p in params:
        assert p[1].abs().sum() > 0, "the firing slot did not move"
        for other in (0, 2, 3):
            assert p[other].abs().sum() == 0, f"slot {other} moved without firing"
    assert opt.step_ct.tolist() == [0, 1, 0, 0]


def test_slots_are_independent_across_steps():
    """Two slots fed identical gradients on different rounds must end equal.

    If the moments were shared, the slot that stepped second would inherit the
    first one's momentum and land somewhere else.
    """
    params = _pool_params()
    opt = PoolAdam(params, POOL, lr=LR)

    _set_grads(params, [0], 1.0)
    opt.step(torch.tensor([0]))
    _set_grads(params, [0], 1.0)
    opt.step(torch.tensor([0]))          # slot 0 has now taken two steps

    _set_grads(params, [2], 1.0)
    opt.step(torch.tensor([2]))
    _set_grads(params, [2], 1.0)
    opt.step(torch.tensor([2]))          # slot 2 takes the same two steps

    for p in params:
        torch.testing.assert_close(p[0], p[2])


def test_reset_slots_clears_moments_and_step_count():
    params = _pool_params()
    opt = PoolAdam(params, POOL, lr=LR)
    _set_grads(params, [0, 1], 1.0)
    opt.step(torch.tensor([0, 1]))

    with torch.no_grad():                # a prefill restores the weights first
        for p in params:
            p[1] = 0
    opt.reset_slots(torch.tensor([1]), params)

    assert opt.step_ct.tolist() == [1, 0, 0, 0]
    for m, v, mst, p in zip(opt.m, opt.v, opt.master, params):
        assert m[1].abs().sum() == 0 and v[1].abs().sum() == 0
        torch.testing.assert_close(mst[1], p[1].float())
        assert m[0].abs().sum() > 0, "the untouched slot lost its momentum"

    # A recycled slot must now behave exactly like a fresh one.
    fresh = PoolAdam(_pool_params(), POOL, lr=LR)
    for opt_, slot in ((opt, 1), (fresh, 1)):
        _set_grads(opt_.params, [slot], 1.0)
        opt_.step(torch.tensor([slot]))
    for p_recycled, p_fresh in zip(opt.params, fresh.params):
        torch.testing.assert_close(p_recycled[1], p_fresh[1])


def test_duplicate_slots_raise():
    """Two batch rows mapped to one pool row is a mapping error and must raise."""
    params = _pool_params()
    opt = PoolAdam(params, POOL, lr=LR)
    _set_grads(params, [0], 1.0)
    try:
        opt.step(torch.tensor([0, 0]))
    except RuntimeError as exc:
        assert "duplicate slots" in str(exc)
    else:
        raise AssertionError("duplicate slots were accepted")


def test_no_grad_does_not_advance_step_count():
    """A step without gradients must not advance the slot's step count.

    Otherwise the slot's bias correction would be offset for its later updates.
    """
    params = _pool_params()
    opt = PoolAdam(params, POOL, lr=LR)
    opt.step(torch.tensor([0]))          # no grads set at all
    assert opt.step_ct.tolist() == [0, 0, 0, 0]


def test_slot_with_zero_gradient_is_left_alone():
    """A slot whose request contributed no rows this round is not stepped.

    Its gradient is exactly zero, so stepping it would advance its step count
    and apply an update from decayed momentum alone, with no new signal.
    """
    params = _pool_params()
    opt = PoolAdam(params, POOL, lr=LR)

    _set_grads(params, [0, 1], 1.0)
    opt.step(torch.tensor([0, 1]))
    before = [p[1].clone() for p in params]
    m_before = [m[1].clone() for m in opt.m]

    # Round two: slot 0 contributes, slot 1 is in the batch with a zero gradient.
    _set_grads(params, [0], 1.0)          # slot 1's grad stays zero
    opt.step(torch.tensor([0, 1]))

    assert opt.step_ct.tolist() == [2, 1, 0, 0], "a zero-grad slot was stepped"
    for p, was in zip(params, before):
        torch.testing.assert_close(p[1], was)
    for m, was in zip(opt.m, m_before):
        torch.testing.assert_close(m[1], was), "momentum decayed without signal"


def test_per_request_loss_equals_running_alone():
    """Summed per-request means give each request the loss it has when alone.

    One mean over the whole batch would instead divide every request by the
    round's total row weight, so a request's update would depend on how many
    rows the other requests contributed.
    """
    torch.manual_seed(0)
    per_pos, w = torch.rand(10), torch.rand(10)
    groups = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 2])
    n = 3

    zeros = torch.zeros(n)
    wsum = zeros.index_add(0, groups, w).clamp_min(1e-6)
    batched = zeros.index_add(0, groups, per_pos * w) / wsum

    for r in range(n):
        m = groups == r
        alone = (per_pos[m] * w[m]).sum() / w[m].sum()
        torch.testing.assert_close(batched[r], alone)


def test_per_slot_clip_does_not_couple_requests():
    """A huge gradient in one slot must not shrink another slot's update."""
    from sglang.srt.speculative.eagle_tts import EagleTTSConfig, EagleTTSState

    params = _pool_params()

    class _Adapters:                      # minimal stand-in for the adapter set
        pool_size = POOL

        def replicated_params(self):
            return []

    state = EagleTTSState.__new__(EagleTTSState)
    state.cfg = EagleTTSConfig(enable=True, grad_clip=1.0)
    state.adapters = _Adapters()
    state._params = params
    state.tp_size = 1

    _set_grads(params, [0], 1000.0)       # slot 0 way over the clip
    for p in params:
        p.grad[1] = 1e-4                  # slot 1 well under it
    before = [p.grad[1].clone() for p in params]
    state._clip_grads_pooled(torch.tensor([0, 1]))

    for p, was in zip(params, before):
        torch.testing.assert_close(p.grad[1], was)   # untouched
        assert p.grad[0].norm() < 1000.0             # clipped
