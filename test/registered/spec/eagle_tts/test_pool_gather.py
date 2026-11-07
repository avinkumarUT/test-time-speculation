"""Per-request adapter pool: the gather path in `LoRAAdapter.delta`.

With `pool_size=1` an adapter takes the plain `a[0]`/`b[0]` matmul. These
tests cover the `index_select` + `bmm` branch taken when `pool_size > 1`, and
the `_RowSelector` that feeds it. A wrong slot mapping raises no error: it
drafts one request with another request's adapter, which only shows up as
lower acceptance.

Needs no checkpoint and keeps all tensors on CPU. A CUDA device must still be
visible: under grad, `_RowSelector.rows_for` calls
`torch.cuda.is_current_stream_capturing()`, which fails when no device is
available.

  python test/registered/spec/eagle_tts/test_pool_gather.py
"""
import torch

from sglang.srt.speculative.eagle_lora_tts import LoRAAdapter, _RowSelector

POOL, RANK, IN, OUT, ALPHA = 4, 4, 6, 5, 8.0


def _adapter(pool_size, rows=None):
    return LoRAAdapter(
        name="o",
        in_features_full=IN,
        in_features_local=IN,
        a_shard_offset=0,
        out_features_local=OUT,
        rank=RANK,
        alpha=ALPHA,
        dtype=torch.float32,
        device=torch.device("cpu"),
        seed=0,
        tp_size=1,
        pool_size=pool_size,
        rows=rows,
    )


def _randomize_b(adapter, generator):
    """Give each slot its own distinct B.

    B starts at zero, which would make every delta identically zero and the
    tests vacuous.
    """
    with torch.no_grad():
        adapter.b.copy_(torch.randn(adapter.b.shape, generator=generator))


def test_each_row_uses_only_its_own_slot():
    """Row i's output must depend on slot i alone.

    Compared against a value computed directly from that slot's A and B, so
    the test does not repeat the implementation's gather.
    """
    g = torch.Generator().manual_seed(0)
    sel = _RowSelector(POOL, torch.device("cpu"), active=True)
    ad = _adapter(POOL, rows=sel)
    _randomize_b(ad, g)

    slots = torch.tensor([2, 0])              # deliberately not identity order
    sel.set_rows(slots)
    x = torch.randn(2, IN, generator=g)

    out = ad.delta(x)

    for row, slot in enumerate(slots.tolist()):
        want = (x[row] @ ad.a[slot].T) @ ad.b[slot].T * ad.scale
        torch.testing.assert_close(out[row], want)


def test_gradients_do_not_cross_between_slots():
    """One batched backward must give each slot exactly its solo gradient."""
    g = torch.Generator().manual_seed(1)
    sel = _RowSelector(POOL, torch.device("cpu"), active=True)
    ad = _adapter(POOL, rows=sel)
    _randomize_b(ad, g)
    b_init = ad.b.detach().clone()

    slots = torch.tensor([1, 3])
    sel.set_rows(slots)
    x = torch.randn(2, IN, generator=g)

    ad.delta(x).pow(2).sum().backward()
    batched_a = ad.a.grad.detach().clone()
    batched_b = ad.b.grad.detach().clone()

    # Slots that took part in no row must be untouched by the backward.
    for idle in (0, 2):
        assert batched_a[idle].abs().sum() == 0, f"slot {idle} got gradient"
        assert batched_b[idle].abs().sum() == 0, f"slot {idle} got gradient"

    # Each participating slot must match running that row entirely alone.
    for row, slot in enumerate(slots.tolist()):
        solo_sel = _RowSelector(1, torch.device("cpu"), active=True)
        solo = _adapter(1, rows=solo_sel)
        with torch.no_grad():
            solo.a.copy_(ad.a.detach()[slot : slot + 1])
            solo.b.copy_(b_init[slot : slot + 1])
        solo_sel.set_rows(torch.tensor([0]))
        solo.delta(x[row : row + 1]).pow(2).sum().backward()
        torch.testing.assert_close(batched_a[slot], solo.a.grad[0])
        torch.testing.assert_close(batched_b[slot], solo.b.grad[0])


def test_remapping_is_picked_up():
    """The static buffer is rewritten in place, so a later mapping must win.

    This is the property CUDA-graph replay depends on: the captured graph holds
    the buffer's address, and every replay must see whatever the current round
    wrote into it. A stale read would apply one request's adapter to another
    request's rows.
    """
    g = torch.Generator().manual_seed(2)
    sel = _RowSelector(POOL, torch.device("cpu"), active=True)
    ad = _adapter(POOL, rows=sel)
    _randomize_b(ad, g)
    x = torch.randn(1, IN, generator=g)

    ptr = sel.buf.data_ptr()
    sel.set_rows(torch.tensor([0]))
    first = ad.delta(x).clone()
    sel.set_rows(torch.tensor([1]))
    second = ad.delta(x).clone()

    assert sel.buf.data_ptr() == ptr, "buffer was reallocated; a graph would miss it"
    assert not torch.allclose(first, second), "the second mapping was ignored"
    with torch.no_grad():
        torch.testing.assert_close(
            second[0], (x[0] @ ad.a[1].T) @ ad.b[1].T * ad.scale
        )


def test_graph_init_traces_the_gather_before_any_mapping_exists():
    """Graph warmup and capture run the draft forward with nothing published.

    They do not bind a mapping, so `n` is still 0. If `rows_for` raised in
    that state, CUDA graph initialization would fail; if it returned None,
    capture would record the plain-matmul path and every replayed round would
    draft from the unused padding slot.
    """
    sel = _RowSelector(POOL, torch.device("cpu"), active=True)
    ad = _adapter(POOL, rows=sel)
    assert sel.n == 0
    rows = sel.rows_for(torch.zeros(2, IN))
    assert rows is not None, "capture would bake in the non-pooled path"
    assert rows.numel() == 2
    ad.delta(torch.zeros(2, IN))          # must not raise


def test_stale_mapping_is_rejected_once_serving():
    """After a real bind, a width mismatch means the mapping was not refreshed."""
    sel = _RowSelector(POOL, torch.device("cpu"), active=True)
    _adapter(POOL, rows=sel)
    sel.set_rows(torch.tensor([0, 1]))
    try:
        sel.rows_for(torch.zeros(3, IN))
    except RuntimeError as exc:
        assert "not refreshed" in str(exc)
    else:
        raise AssertionError("a stale mapping was accepted")


def test_head_selector_multi_row_blocks():
    """lm_head's selector: per-request blocks of `per_req` rows.

    The caller publishes the slots expanded with `repeat_interleave`; the
    factor is not inferred from tensor widths.
    """
    g = torch.Generator().manual_seed(4)
    per_req = 10                                     # k+1 in the decode extend
    sel = _RowSelector(POOL * per_req, torch.device("cpu"), active=True)
    ad = _adapter(POOL, rows=sel)
    _randomize_b(ad, g)

    slots = torch.tensor([2, 0])
    sel.set_rows(slots.repeat_interleave(per_req))   # what the worker publishes
    x = torch.randn(2 * per_req, IN, generator=g)

    out = ad.delta(x)
    for row in range(x.shape[0]):
        slot = slots[row // per_req].item()          # contiguous per-request
        want = (x[row] @ ad.a[slot].T) @ ad.b[slot].T * ad.scale
        torch.testing.assert_close(out[row], want)

    # Factor 1 on the same selector (prefill and the rollout reuse it).
    sel.set_rows(slots)
    out1 = ad.delta(x[:2])
    for row, slot in enumerate(slots.tolist()):
        want = (x[row] @ ad.a[slot].T) @ ad.b[slot].T * ad.scale
        torch.testing.assert_close(out1[row], want)


def test_head_selector_capture_width_exceeding_pool():
    """The head buffer holds the widest capture forward.

    That forward has capture_bs x per_req rows, more than pool_size. Before any
    mapping is published, `rows_for` must return an index of the full width,
    not a slice clamped to the pool size.
    """
    per_req = 10
    sel = _RowSelector(POOL * per_req, torch.device("cpu"), active=True)
    width = (POOL - 1) * per_req                     # widest legal capture
    rows = sel.rows_for(torch.zeros(width, IN))
    assert rows is not None and rows.numel() == width, (
        "sentinel returned a clamped index; a captured bmm would crash "
        "with 'Batch sizes must match'"
    )
    # A forward wider than the buffer raises instead of being clamped.
    try:
        sel.rows_for(torch.zeros(POOL * per_req + 1, IN))
    except RuntimeError as exc:
        assert "more than the row buffer" in str(exc)
    else:
        raise AssertionError("over-wide forward was silently clamped")


def test_pool_size_one_keeps_the_plain_path():
    """With pool_size 1 the adapter takes the plain-matmul path, with no gather."""
    g = torch.Generator().manual_seed(3)
    sel = _RowSelector(1, torch.device("cpu"), active=False)
    ad = _adapter(1, rows=sel)
    _randomize_b(ad, g)
    x = torch.randn(3, IN, generator=g)

    assert sel.rows_for(x) is None
    torch.testing.assert_close(
        ad.delta(x), (x @ ad.a[0].T) @ ad.b[0].T * ad.scale
    )


if __name__ == "__main__":
    import traceback

    failed = 0
    for name in sorted(n for n in dir() if n.startswith("test_")):
        try:
            globals()[name]()
            print("PASS", name)
        except Exception:
            failed += 1
            print("FAIL", name)
            traceback.print_exc()
    print("---", "ALL PASS" if not failed else f"{failed} FAILED")
    raise SystemExit(1 if failed else 0)
