"""The bfloat16 optimizer dtype of per-request DFlash LoRA-TTS.

With `--speculative-dflash-lora-tts-optimizer-dtype bfloat16` a request's
masters and Adam moments are stored as bf16 blocks. An update materializes fp32
leaves, runs the same fp32 arithmetic and rounds once when it stores the
results. These tests drive the state through `run_tts_steps`, with a linear loss
in place of the replay so that the gradients are known exactly (one test uses
the real replay), and check:

  (a) an update equals an fp32 torch.optim.Adam update on the widened state,
      rounded once, including several inner steps and the batched path;
  (b) the state is bf16 by dtype and byte count, pinned when offloaded;
  (c) updates below the bf16 spacing are lost, and accumulate with float32;
  (d) a request that did not fire keeps every bit of its state, and a leaf
      group without a gradient keeps its bits and its moments;
  (e) a reset restores the slot exactly and leaves other slots alone;
  (f) the tensor-parallel gradient completion and clip are those of float32,
      and replicated leaves stay bit-identical across ranks.

Also: the served pool row and the published buffers hold the stored bits; a
non-finite loss leaves the state untouched; on CUDA, offloaded, resident,
synchronous, asynchronous and delayed-publish runs give identical bits; the
second moment does not decay (the documented limitation); startup checks.

Needs no checkpoint or server. `TTS_TEST_DEV` selects the device (default: CUDA
when available). On CPU, state cannot be pinned and the path without offload
runs; the offloaded path needs CUDA. Run directly or with pytest:

  python test/registered/spec/dflash_tts/test_lora_bf16_optimizer.py
"""

import argparse
import math
import os
import sys
import threading
from types import SimpleNamespace

DEV = os.environ.get("TTS_TEST_DEV") or None

import torch

DEV = DEV or ("cuda" if torch.cuda.is_available() else "cpu")
if DEV == "cpu":
    os.environ.setdefault("SGLANG_USE_CPU_ENGINE", "1")

from torch import nn  # noqa: E402

import sglang.srt.distributed.parallel_state as parallel_state  # noqa: E402
from sglang.srt.server_args import (  # noqa: E402
    ServerArgs,
    set_global_server_args_for_scheduler,
)

parallel_state._TP = SimpleNamespace(world_size=1, rank_in_group=0)
_server_args = ServerArgs(model_path="dummy", device="cpu")
_server_args.speculative_dflash_lora_tts_enable = True
set_global_server_args_for_scheduler(_server_args)

from sglang.srt.speculative import dflash_lora_tts  # noqa: E402
from sglang.srt.speculative.dflash_lora_tts import (  # noqa: E402
    DFlashLoRATTSConfig,
    DFlashLoRATTSState,
    _ReplayCapture,
)

BF16 = torch.bfloat16
DEVICE = torch.device(DEV)
HIDDEN, Q, KV, INTER, BLOCK, VOCAB, RANK = 16, 8, 4, 24, 4, 12, 3
GROUPS = ("a_repl", "b_col", "a_row", "b_repl")
LR = 1e-3


def bits(t: torch.Tensor) -> bytes:
    """The bytes of a tensor. Unlike `torch.equal` this tells -0.0 from 0.0."""
    return bytes(t.detach().cpu().clone().untyped_storage())


def same(a, b) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and bits(a) == bits(b)


class ToyLayer(nn.Module):
    """The attributes of a DFlash decoder layer that the state reads at init."""

    def __init__(self, layer_id, dtype):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.attn = SimpleNamespace(sliding_window_size=-1, layer_id=layer_id)
        self.self_attn.hidden_size = HIDDEN
        self.self_attn.q_size = Q
        self.self_attn.kv_size = KV
        self.self_attn.qkv_proj = nn.Linear(HIDDEN, Q + 2 * KV, bias=False)
        self.self_attn.o_proj = nn.Linear(Q, HIDDEN, bias=False)
        self.mlp = nn.Module()
        self.mlp.gate_up_proj = nn.Linear(HIDDEN, 2 * INTER, bias=False)
        self.mlp.down_proj = nn.Linear(INTER, HIDDEN, bias=False)
        self.to(DEVICE).to(dtype)


class ToyDraft(nn.Module):
    def __init__(self, dtype=BF16):
        super().__init__()
        self.layers = nn.ModuleList([ToyLayer(i, dtype) for i in range(2)])


class ToyPool:
    def __init__(self):
        self.k = torch.zeros(4, KV, device=DEVICE)
        self.v = torch.zeros(4, KV, device=DEVICE)

    def get_kv_buffer(self, _layer_id):
        return self.k, self.v


class Harness:
    """A state whose loss is `sum(coef * leaf)`, so a leaf's gradient is `coef`.

    A capture's request index travels in its input embeds, which is how the
    loss knows which coefficients apply to it.
    """

    def __init__(self, dtype="bfloat16", draft_dtype=BF16, **cfg):
        cfg.setdefault("async_step", False)
        cfg.setdefault("grad_clip", 0.0)
        cfg.setdefault("lr", LR)
        self.state = DFlashLoRATTSState(
            config=DFlashLoRATTSConfig(rank=RANK, optimizer_dtype=dtype, **cfg),
            draft_model=ToyDraft(draft_dtype),
            token_to_kv_pool=ToyPool(),
            block_size=BLOCK,
            device=DEVICE,
        )
        self.grads = {}
        self.nan_loss = set()
        self.state._forward_loss_one = self._loss_one
        self.state._forward_loss_many = self._loss_many

    def _loss_for(self, leaves, req):
        loss = None
        for g, coef in self.grads[req].items():
            term = (coef * leaves[g]).sum()
            loss = term if loss is None else loss + term
        return loss + float("nan") if req in self.nan_loss else loss

    def _loss_one(self, leaves, input_embeds, *_):
        return self._loss_for(leaves, int(input_embeds[0, 0]))

    def _loss_many(self, leaves_list, embeds, *_):
        return torch.stack(
            [
                self._loss_for(lv, int(embeds[i, 0, 0]))
                for i, lv in enumerate(leaves_list)
            ]
        )

    def capture(self, row, req):
        embeds = torch.zeros(BLOCK, HIDDEN, dtype=BF16, device=DEVICE)
        embeds[0, 0] = req
        return _ReplayCapture(
            req_pool_idx=req,
            row=row,
            input_embeds=embeds,
            positions=torch.arange(BLOCK, device=DEVICE),
            prefix_locs=torch.tensor([0, 1], device=DEVICE),
        )

    def fire(self, grads_by_req):
        """One round's update for the given requests, fenced."""
        state = self.state
        self.grads = grads_by_req
        reqs = list(grads_by_req)
        for req in reqs:
            state.get(req)
        state.wait_for_pending()
        state._captures = [self.capture(r, req) for r, req in enumerate(reqs)]
        state.run_tts_steps(
            target_logits=torch.zeros(
                len(reqs), BLOCK - 1, VOCAB, dtype=BF16, device=DEVICE
            ),
            lm_head_weight=torch.zeros(VOCAB, HIDDEN, dtype=BF16, device=DEVICE),
        )
        state.wait_for_pending()
        assert state._step_failure_streak == 0, "the update raised"

    def random_grads(self, seed, scale=0.1):
        gen = torch.Generator().manual_seed(seed)
        return {
            g: (torch.randn(n, generator=gen) * scale).to(DEVICE)
            for g, n in self.state._flat_sizes.items()
        }

    def constant_grads(self, value):
        return {
            g: torch.full((n,), value, device=DEVICE)
            for g, n in self.state._flat_sizes.items()
        }


def read_moments(rs, g):
    """(m, v) of a leaf group, wherever the mode keeps them."""
    if rs.opt is not None:
        st = rs.opt.state[rs.flats[g]]
        return st["exp_avg"], st["exp_avg_sq"]
    return rs.adam_m[g], rs.adam_v[g]


class AdamReference:
    """fp32 `torch.optim.Adam` on the widened state, rounded once per update."""

    def __init__(self, flats, *, lr=LR, grad_clip=0.0, steps=1):
        self.w = {g: t.detach().to(DEVICE).clone() for g, t in flats.items()}
        self.m = {}
        self.v = {}
        self.t = 0
        self.lr, self.grad_clip, self.steps = lr, grad_clip, steps

    def update(self, grads):
        leaves = {g: nn.Parameter(w.float()) for g, w in self.w.items()}
        opt = torch.optim.Adam(list(leaves.values()), lr=self.lr)
        if self.t:
            for g, leaf in leaves.items():
                opt.state[leaf] = {
                    "step": torch.tensor(float(self.t)),
                    "exp_avg": self.m[g].float(),
                    "exp_avg_sq": self.v[g].float(),
                }
        for _ in range(self.steps):
            for g, leaf in leaves.items():
                leaf.grad = grads[g].clone()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(list(leaves.values()), self.grad_clip)
            opt.step()
            self.t += 1
        for g, leaf in leaves.items():
            self.w[g] = leaf.detach().to(BF16)
            self.m[g] = opt.state[leaf]["exp_avg"].to(BF16)
            self.v[g] = opt.state[leaf]["exp_avg_sq"].to(BF16)

    def check(self, rs, label):
        for g in GROUPS:
            assert same(rs.flats[g], self.w[g]), f"{label}: master {g}"
            assert same(rs.adam_m[g], self.m[g]), f"{label}: m {g}"
            assert same(rs.adam_v[g], self.v[g]), f"{label}: v {g}"


def check_pool_row_matches_state(state, rs):
    """Serving reads what the next update widens."""
    gen = state._published_gen(rs)
    row = gen * state._pool_cap + rs.pool_row
    for key in state._shapes:
        a, b = state._leaf_views(rs.flats, key)
        assert same(state._a_pools[key][row], a), f"pool A {key}"
        assert same(state._b_pools[key][row], b), f"pool B {key}"


# ------------------------------------------------------------------- (a) one step


def test_a_updates_equal_rounded_fp32_adam():
    for grad_clip in (0.0, 0.05):
        h = Harness(grad_clip=grad_clip)
        rs = h.state.get(3)
        ref = AdamReference(rs.flats, grad_clip=grad_clip)
        for seed in range(6):
            grads = h.random_grads(seed)
            h.fire({3: grads})
            ref.update(grads)
            ref.check(rs, f"clip={grad_clip} update {seed}")
            check_pool_row_matches_state(h.state, rs)
            assert rs.step_ct == seed + 1


def _emulate_offload(h):
    """Run the offload branches of a CPU state, with unpinned blocks standing in
    for the pinned ones, which need CUDA. On CUDA the real path runs instead."""
    state = h.state
    state._offload = True

    def acquire():
        if state._cpu_state_cache:
            return state._cpu_state_cache.pop()

        def blocks():
            return {
                g: torch.empty(n, dtype=state._state_dtype)
                for g, n in state._flat_sizes.items()
            }

        return blocks(), blocks(), blocks()

    state._acquire_state_blocks = acquire


def test_a_offload_branch_equals_rounded_fp32_adam():
    h = Harness(grad_clip=0.05)
    if DEV == "cpu":
        _emulate_offload(h)
    assert h.state._offload
    rs = h.state.get(3)
    other = h.state.get(4)
    ref = AdamReference(rs.flats, grad_clip=0.05)
    ref_other = AdamReference(other.flats, grad_clip=0.05)
    for seed in range(4):
        grads = h.random_grads(seed)
        h.fire({3: grads})
        ref.update(grads)
        ref.check(rs, f"update {seed}")
        check_pool_row_matches_state(h.state, rs)
        grads = {3: h.random_grads(seed + 10), 4: h.random_grads(seed + 20)}
        h.fire(grads)
        ref.update(grads[3])
        ref_other.update(grads[4])
        ref.check(rs, f"batched update {seed}")
        ref_other.check(other, f"batched update {seed}")
    # A released block returns to the cache, and its next owner does not read
    # the old moments.
    h.state.drop(4)
    h.state.wait_for_pending()
    h.state._drain_deferred()
    assert len(h.state._cpu_state_cache) == 1
    recycled = h.state.get(5)
    assert not recycled.moments_valid and recycled.step_ct == 0
    ref = AdamReference(recycled.flats, grad_clip=0.05)
    grads = h.random_grads(50)
    h.fire({5: grads})
    ref.update(grads)
    ref.check(recycled, "recycled block")


def test_a_inner_steps_round_once_per_update():
    h = Harness(steps=3)
    rs = h.state.get(1)
    ref = AdamReference(rs.flats, steps=3)
    for seed in range(3):
        grads = h.random_grads(seed)
        h.fire({1: grads})
        ref.update(grads)
        ref.check(rs, f"update {seed}")
    assert rs.step_ct == 9


def test_a_batched_updates_equal_their_own_references():
    for grad_clip in (0.0, 0.05):
        h = Harness(grad_clip=grad_clip)
        reqs = [2, 5, 7]
        refs = {
            r: AdamReference(h.state.get(r).flats, grad_clip=grad_clip) for r in reqs
        }
        for seed in range(4):
            grads = {r: h.random_grads(seed * 10 + r) for r in reqs}
            h.fire(grads)
            for r in reqs:
                refs[r].update(grads[r])
                refs[r].check(h.state.get(r), f"request {r} update {seed}")
                check_pool_row_matches_state(h.state, h.state.get(r))


def _check_first_update_is_float32_rounded(rs_bf, rs_fp):
    """B starts at zero in both precisions, so its first update rounds to the
    float32 result. A starts from the fp32 template in float32 and from the
    rounded template in bf16, so it is compared through its moments, which see
    only the gradient."""
    for g in GROUPS:
        for mine, theirs in zip(read_moments(rs_bf, g), read_moments(rs_fp, g)):
            assert same(mine, theirs.detach().to(DEVICE).to(BF16)), f"moment {g}"
    for g in ("b_col", "b_repl"):
        assert same(rs_bf.flats[g], rs_fp.flats[g].detach().to(DEVICE).to(BF16)), g


def test_a_first_update_rounds_to_the_float32_result():
    bf, fp = Harness(), Harness(dtype="float32")
    grads = bf.random_grads(0)
    bf.fire({0: grads})
    fp.fire({0: grads})
    _check_first_update_is_float32_rounded(bf.state.get(0), fp.state.get(0))
    for key in bf.state._shapes:
        assert same(bf.state._b_pools[key], fp.state._b_pools[key]), f"pool {key}"


def test_a_real_replay_update():
    """The real replay and loss: the first update is float32's, rounded."""
    if DEV != "cpu":
        print("  skipped: the toy DFlash model runs on the CPU engine only")
        return
    results = {}
    for dtype in ("float32", "bfloat16"):
        state, _pool, embeds, positions, locs, lm_head, target = _real_model_state(dtype)
        rs = state.get(0)
        before = {g: bits(t.to(BF16)) for g, t in rs.flats.items()}
        state._captures = [
            _ReplayCapture(
                req_pool_idx=0,
                row=0,
                input_embeds=embeds,
                positions=positions,
                prefix_locs=locs,
            )
        ]
        state.run_tts_steps(target_logits=target, lm_head_weight=lm_head)
        assert state._step_failure_streak == 0, "the replay raised"
        assert rs.step_ct == 1
        assert bits(rs.flats["b_col"].to(BF16)) != before["b_col"], "B did not move"
        results[dtype] = rs
    for g in GROUPS:
        assert same(
            results["bfloat16"].flats[g], results["float32"].flats[g].detach().to(BF16)
        ), g
        assert results["bfloat16"].flats[g].dtype == BF16


def _real_model_state(optimizer_dtype):
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.models.dflash import DFlashDraftModel

    head_dim, layers = 4, 2
    config = SimpleNamespace(
        hidden_size=HIDDEN,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=head_dim,
        intermediate_size=INTER,
        num_hidden_layers=layers,
        rms_norm_eps=1e-6,
        attention_bias=False,
        rope_theta=10000.0,
        max_position_embeddings=256,
        hidden_act="silu",
        block_size=BLOCK,
        dflash_config={"mask_token_id": 3, "target_layer_ids": [0, 1]},
        num_target_layers=4,
    )
    model = DFlashDraftModel(config)
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "norm" in name:
                param.copy_(1.0 + 0.2 * torch.randn(param.shape, generator=gen))
            else:
                param.copy_(
                    torch.randn(param.shape, generator=gen) / math.sqrt(param.shape[-1])
                )
            param.data = param.data.to(BF16)
    pool = MHATokenToKVPool(
        size=16,
        page_size=1,
        dtype=BF16,
        head_num=2,
        head_dim=head_dim,
        layer_num=layers,
        device="cpu",
        enable_memory_saver=False,
    )
    locs = torch.tensor([1, 4, 2, 7, 9], dtype=torch.int64)
    for layer_id in range(layers):
        k_buffer, v_buffer = pool.get_kv_buffer(layer_id)
        k_buffer[locs] = torch.randn(5, 2, head_dim, generator=gen).to(BF16)
        v_buffer[locs] = torch.randn(5, 2, head_dim, generator=gen).to(BF16)
    state = DFlashLoRATTSState(
        config=DFlashLoRATTSConfig(
            rank=RANK, async_step=False, optimizer_dtype=optimizer_dtype, lr=1e-2
        ),
        draft_model=model,
        token_to_kv_pool=pool,
        block_size=BLOCK,
        device=torch.device("cpu"),
    )
    embeds = torch.randn(BLOCK, HIDDEN, generator=gen).to(BF16)
    positions = torch.arange(5, 5 + BLOCK, dtype=torch.int64)
    lm_head = (torch.randn(VOCAB, HIDDEN, generator=gen) / 4).to(BF16)
    target = torch.randn(1, BLOCK - 1, VOCAB, generator=gen).to(BF16)
    return state, pool, embeds, positions, locs, lm_head, target


# ------------------------------------------------------------------- (b) storage


def _params_per_req(state):
    return sum(RANK * (i + o) for i, o in state._shapes.values())


def test_b_state_is_bf16_with_the_expected_bytes():
    h = Harness()
    rs = h.state.get(0)
    h.fire({0: h.random_grads(0)})
    p = _params_per_req(h.state)
    assert sum(h.state._flat_sizes.values()) == p
    for blocks in (rs.flats, rs.adam_m, rs.adam_v):
        assert all(t.dtype == BF16 for t in blocks.values())
        assert sum(t.nbytes for t in blocks.values()) == 2 * p
        if DEV == "cuda" and h.state._offload:
            assert all(t.is_pinned() and t.device.type == "cpu" for t in blocks.values())
    assert rs.opt is None


def test_b_float32_state_stays_fp32():
    h = Harness(dtype="float32")
    rs = h.state.get(0)
    h.fire({0: h.random_grads(0)})
    p = _params_per_req(h.state)
    total = 0
    for g in GROUPS:
        for t in (rs.flats[g], *read_moments(rs, g)):
            assert t.dtype == torch.float32
            total += t.nbytes
    assert total == 12 * p


def test_b_no_offload_keeps_bf16_blocks_on_the_device():
    h = Harness(offload_state=False)
    rs = h.state.get(0)
    ref = AdamReference(rs.flats)
    grads = h.random_grads(0)
    h.fire({0: grads})
    ref.update(grads)
    ref.check(rs, "no offload")
    assert not h.state._offload and rs.opt is None
    assert all(t.device.type == DEVICE.type for t in rs.flats.values())
    assert not any(t.is_pinned() for t in rs.flats.values()) if DEV == "cuda" else True


# ------------------------------------------------------------- (c) small updates


def _fill(rs, value):
    with torch.no_grad():
        for t in rs.flats.values():
            t.fill_(value)


def test_c_small_updates_are_lost_in_bf16_and_accumulate_in_float32():
    # At 1.0 half the spacing below is 2^-9 = 1.95e-3. Each Adam step moves a
    # weight by about the learning rate, 1e-4, so one step rounds away, and
    # 100 steps add up to about 1e-2, more than one bf16 spacing.
    h = Harness(lr=1e-4)
    rs = h.state.get(0)
    _fill(rs, 1.0)
    for _ in range(100):
        h.fire({0: h.constant_grads(0.5)})
    for g in GROUPS:
        assert bool((rs.flats[g] == 1.0).all()), f"{g}: a bf16 master moved"
        assert bool((rs.adam_m[g] != 0).all()) and bool((rs.adam_v[g] != 0).all()), g
    assert rs.step_ct == 100

    h = Harness(dtype="float32", lr=1e-4)
    rs = h.state.get(0)
    _fill(rs, 1.0)
    for _ in range(100):
        h.fire({0: h.constant_grads(0.5)})
    for g in GROUPS:
        moved = (1.0 - rs.flats[g]).abs().min().item()
        assert 5e-3 < moved < 2e-2, f"{g}: master moved by {moved}"
    row = rs.pool_row
    for key in h.state._shapes:
        assert bool((h.state._a_pools[key][row] < 1.0).all()), f"{key}: pool stayed"


def test_c_second_moment_does_not_decay_in_bf16():
    # The documented limitation: 0.999 * v rounds back to v.
    out = {}
    for dtype in ("float32", "bfloat16"):
        h = Harness(dtype=dtype)
        rs = h.state.get(0)
        h.fire({0: h.constant_grads(1.0)})
        first = read_moments(rs, "b_col")[1].detach().clone()
        for _ in range(600):
            h.fire({0: h.constant_grads(1e-6)})
        out[dtype] = (first, read_moments(rs, "b_col")[1])
    assert same(*out["bfloat16"]), "the bf16 second moment changed"
    ratio = (out["float32"][1] / out["float32"][0]).max().item()
    assert ratio < 0.56, f"the fp32 second moment kept {ratio}"


# ----------------------------------------------------------------- (d) isolation


def _snapshot(h, req):
    rs = h.state.get(req)
    row = rs.pool_row
    return (
        {g: bits(rs.flats[g]) for g in GROUPS},
        {g: bits(rs.adam_m[g]) for g in GROUPS},
        {g: bits(rs.adam_v[g]) for g in GROUPS},
        rs.step_ct,
        rs.round_ct,
        rs.moments_valid.copy(),
        {k: bits(h.state._a_pools[k][row]) for k in h.state._shapes},
        {k: bits(h.state._b_pools[k][row]) for k in h.state._shapes},
    )


def test_d_a_request_that_did_not_fire_keeps_every_bit():
    h = Harness()
    h.fire({r: h.random_grads(r) for r in (1, 2, 3)})
    idle = _snapshot(h, 2)
    h.fire({1: h.random_grads(10)})                   # the single-request path
    assert _snapshot(h, 2) == idle, "changed by another request's update"
    h.fire({1: h.random_grads(11), 3: h.random_grads(12)})  # the batched path
    assert _snapshot(h, 2) == idle, "changed by a batched update"
    assert h.state.get(1).step_ct == 3 and h.state.get(3).step_ct == 2


def test_d_a_leaf_group_without_a_gradient_keeps_bits_and_moments():
    h = Harness()
    rs = h.state.get(0)
    a_row = bits(rs.flats["a_row"])
    grads = h.random_grads(0)
    del grads["a_row"]
    h.fire({0: grads})
    assert bits(rs.flats["a_row"]) == a_row
    assert "a_row" not in rs.moments_valid
    assert rs.moments_valid == {"a_repl", "b_col", "b_repl"}
    # Its first real step starts from zero moments, not from the block's bytes.
    ref = AdamReference({"a_row": rs.flats["a_row"]})
    full = h.random_grads(1)
    h.fire({0: full})
    ref.update({"a_row": full["a_row"]})
    assert same(rs.adam_m["a_row"], ref.m["a_row"])


# --------------------------------------------------------------------- (e) reset


def test_e_reset_restores_the_slot_and_leaves_others_alone():
    h = Harness()
    fresh = {g: bits(t) for g, t in h.state.get(1).flats.items()}
    for key in h.state._shapes:
        rs = h.state.get(1)
        a, b = h.state._leaf_views(rs.flats, key)
        assert same(a, h.state._a_init[key].reshape(a.shape).to(a.device).to(BF16)), key
        assert bool((b == 0).all()), key
    for seed in range(3):
        h.fire({1: h.random_grads(seed), 2: h.random_grads(seed + 10)})
    other = _snapshot(h, 2)
    assert {g: bits(t) for g, t in h.state.get(1).flats.items()} != fresh

    h.state.reset(torch.tensor([1]))
    assert _snapshot(h, 2) == other, "the other slot changed"
    rs = h.state.get(1)
    assert rs.step_ct == 0 and rs.round_ct == 0 and rs.moments_valid == set()
    assert {g: bits(t) for g, t in rs.flats.items()} == fresh
    check_pool_row_matches_state(h.state, rs)

    # A recycled block must behave exactly like a fresh one: its old moment
    # bytes are not read.
    ref = AdamReference(rs.flats)
    grads = h.random_grads(9)
    h.fire({1: grads})
    ref.update(grads)
    ref.check(rs, "after reset")


# ---------------------------------------------------------- non-finite losses


def test_non_finite_loss_leaves_the_state_untouched():
    h = Harness()
    for r in (1, 2):
        h.state.get(r)
    h.fire({1: h.random_grads(0), 2: h.random_grads(1)})
    before = _snapshot(h, 1)
    h.nan_loss = {1}
    h.fire({1: h.random_grads(2)})                     # the single-request path
    assert _snapshot(h, 1) == before
    other_before = _snapshot(h, 2)[3]
    h.fire({1: h.random_grads(3), 2: h.random_grads(4)})  # the batched path
    assert _snapshot(h, 1) == before
    assert _snapshot(h, 2)[3] == other_before + 1, "the finite request did not step"


# --------------------------------------------------------------------- (f) TP


class InProcessTP:
    """All-reduce between one thread per rank."""

    def __init__(self, world_size):
        self.slots = [None] * world_size
        self.barrier = threading.Barrier(world_size, timeout=60)
        self.local = threading.local()

    def all_reduce(self, x):
        self.slots[self.local.rank] = x.detach().clone()
        self.barrier.wait()
        total = sum(self.slots)
        self.barrier.wait()
        return total


def _run_ranks(tp, fn):
    errors = []

    def body(rank):
        tp.local.rank = rank
        try:
            fn(rank)
        except Exception as e:  # noqa: BLE001
            errors.append(e)
            tp.barrier.abort()

    original = dflash_lora_tts.tensor_model_parallel_all_reduce
    dflash_lora_tts.tensor_model_parallel_all_reduce = tp.all_reduce
    try:
        threads = [threading.Thread(target=body, args=(r,)) for r in range(len(tp.slots))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        dflash_lora_tts.tensor_model_parallel_all_reduce = original
    if errors:
        raise errors[0]


def _tp_grads(h, rank, seed, scale=0.2):
    """b_repl's gradient is the same on all ranks. a_repl's is a per-rank partial
    sum, and the sharded leaves have their own."""
    gen_shared = torch.Generator().manual_seed(seed)
    gen_rank = torch.Generator().manual_seed(seed + 1000 * (rank + 1))
    out = {}
    for g, n in h.state._flat_sizes.items():
        gen = gen_shared if g == "b_repl" else gen_rank
        out[g] = (torch.randn(n, generator=gen) * scale).to(DEVICE)
    return out


def _complete_and_clip(per_rank, grad_clip):
    """Per-rank gradients after completion and the global clip, computed by hand."""
    a_repl = per_rank[0]["a_repl"] + per_rank[1]["a_repl"]
    sq = a_repl.pow(2).sum() + per_rank[0]["b_repl"].pow(2).sum()
    for grads in per_rank:
        sq = sq + grads["b_col"].pow(2).sum() + grads["a_row"].pow(2).sum()
    norm = sq.sqrt()
    scale = (grad_clip / (norm + 1e-6)).clamp(max=1.0)
    assert scale < 1.0, "the clip must engage for this test to mean anything"
    out = []
    for grads in per_rank:
        out.append(
            {
                "a_repl": a_repl * scale,
                "b_repl": grads["b_repl"] * scale,
                "b_col": grads["b_col"] * scale,
                "a_row": grads["a_row"] * scale,
            }
        )
    return out


def _tp_run(dtype, reqs, updates, grad_clip=0.05, nan_rank=None):
    harnesses = [Harness(dtype=dtype, grad_clip=grad_clip) for _ in range(2)]
    tp = InProcessTP(2)
    for rank, h in enumerate(harnesses):
        h.state.tp_size = 2
        h.state.tp_rank = rank
        if nan_rank == rank:
            h.nan_loss = set(reqs)

    def body(rank):
        h = harnesses[rank]
        for update in range(updates):
            h.fire({r: _tp_grads(h, rank, update * 10 + r) for r in reqs})

    _run_ranks(tp, body)
    return harnesses


def test_f_replicated_leaves_stay_identical_across_ranks():
    for reqs in ((0,), (0, 1, 2)):                     # single and batched paths
        harnesses = _tp_run("bfloat16", reqs, updates=3)
        for r in reqs:
            rs0, rs1 = harnesses[0].state.get(r), harnesses[1].state.get(r)
            for g in ("a_repl", "b_repl"):
                assert same(rs0.flats[g], rs1.flats[g]), f"req {r}: {g} diverged"
                assert same(rs0.adam_v[g], rs1.adam_v[g]), f"req {r}: v {g} diverged"
            for g in ("b_col", "a_row"):
                assert not same(rs0.flats[g], rs1.flats[g]), f"req {r}: {g} did not differ"


def test_f_update_uses_completed_clipped_gradients():
    for reqs in ((0,), (0, 1)):
        harnesses = _tp_run("bfloat16", reqs, updates=1)
        for r in reqs:
            per_rank = [_tp_grads(harnesses[rank], rank, r) for rank in range(2)]
            clipped = _complete_and_clip(per_rank, 0.05)
            for rank in range(2):
                rs = harnesses[rank].state.get(r)
                fresh = Harness(dtype="bfloat16").state.get(r)
                ref = AdamReference(fresh.flats)
                ref.update(clipped[rank])
                for g in GROUPS:
                    assert same(rs.adam_m[g], ref.m[g]), f"req {r} rank {rank}: m {g}"
                    assert same(rs.adam_v[g], ref.v[g]), f"req {r} rank {rank}: v {g}"
                    assert same(rs.flats[g], ref.w[g]), f"req {r} rank {rank}: w {g}"


def test_f_first_update_equals_float32_rounded():
    for reqs in ((0,), (0, 1)):
        fp, bf = _tp_run("float32", reqs, 1), _tp_run("bfloat16", reqs, 1)
        for rank in range(2):
            for r in reqs:
                _check_first_update_is_float32_rounded(
                    bf[rank].state.get(r), fp[rank].state.get(r)
                )


def test_f_non_finite_on_one_rank_skips_on_all_ranks():
    for reqs in ((0,), (0, 1)):
        harnesses = _tp_run("bfloat16", reqs, updates=2, nan_rank=1)
        for h in harnesses:
            for r in reqs:
                rs = h.state.get(r)
                assert rs.step_ct == 0, f"req {r} stepped"
                assert rs.moments_valid == set(), f"req {r} wrote moments"


# ------------------------------------------------------------- CUDA path matrix


def _sequence(h):
    reqs = [1, 2, 3]
    for r in reqs:
        h.state.get(r)
    for seed in range(4):
        h.fire({r: h.random_grads(seed * 10 + r) for r in reqs})
    # A single-request round, for the unbatched path.
    h.fire({2: h.random_grads(99)})
    return {
        r: {
            "w": {g: bits(h.state.get(r).flats[g]) for g in GROUPS},
            "m": {g: bits(h.state.get(r).adam_m[g]) for g in GROUPS},
            "v": {g: bits(h.state.get(r).adam_v[g]) for g in GROUPS},
        }
        for r in reqs
    }


def test_cuda_offload_async_and_delayed_publish_give_identical_bits():
    if DEV != "cuda":
        print("  skipped: the offloaded and asynchronous paths need CUDA")
        return
    baseline = None
    for offload in (True, False):
        for async_step, delayed in ((False, False), (True, False), (True, True)):
            h = Harness(
                offload_state=offload,
                async_step=async_step,
                delayed_publish=delayed,
                grad_clip=0.05,
            )
            assert h.state._offload is offload and h.state._use_async is async_step
            assert h.state._delayed is delayed
            result = _sequence(h)
            torch.cuda.synchronize()
            if baseline is None:
                baseline = result
            assert result == baseline, f"offload={offload} async={async_step} delayed={delayed}"
            for r in (1, 2, 3):
                check_pool_row_matches_state(h.state, h.state.get(r))


def test_cuda_recycled_pinned_block_is_bf16_and_reused():
    if DEV != "cuda":
        print("  skipped: pinned blocks need CUDA")
        return
    h = Harness(async_step=True)
    h.fire({1: h.random_grads(0)})
    block = h.state.get(1).flats["a_repl"]
    h.state.drop(1)
    h.state.wait_for_pending()
    h.state._drain_deferred()
    assert len(h.state._cpu_state_cache) == 1
    rs = h.state.get(4)
    assert rs.flats["a_repl"] is block, "the block was not recycled"
    assert block.dtype == BF16 and block.is_pinned()
    ref = AdamReference(rs.flats)
    grads = h.random_grads(1)
    h.fire({4: grads})
    ref.update(grads)
    ref.check(rs, "recycled block")


# -------------------------------------------------------------------- publication


def test_published_buffers_hold_the_stored_bits():
    h = Harness()
    state = h.state
    buffers = SimpleNamespace(
        a={
            k: torch.zeros(4, RANK, i, dtype=BF16, device=DEVICE)
            for k, (i, _o) in state._shapes.items()
        },
        b={
            k: torch.zeros(4, o, RANK, dtype=BF16, device=DEVICE)
            for k, (_i, o) in state._shapes.items()
        },
        max_bs=4,
        scaling=state.cfg.scaling,
        block_size=BLOCK,
    )
    state.adopt_static_buffers(buffers)
    h.fire({5: h.random_grads(0), 6: h.random_grads(1)})
    state.publish_static([6, 5])
    for key in state._shapes:
        for row, req in enumerate((6, 5)):
            a, b = state._leaf_views(state.get(req).flats, key)
            assert same(buffers.a[key][row], a), f"A {key} row {row}"
            assert same(buffers.b[key][row], b), f"B {key} row {row}"


# ------------------------------------------------------------------ configuration


def test_startup_checks_and_flags():
    for draft_dtype in (torch.float32, torch.float16):
        try:
            Harness(draft_dtype=draft_dtype)
        except ValueError as e:
            assert "bfloat16 draft model" in str(e) and "float32" in str(e)
        else:
            raise AssertionError(f"a {draft_dtype} draft was accepted")
    Harness(dtype="float32", draft_dtype=torch.float32)
    try:
        Harness(dtype="float16")
    except ValueError as e:
        assert "float16" in str(e)
    else:
        raise AssertionError("an unknown dtype was accepted")

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    base = ["--model-path", "x"]
    args = parser.parse_args(base)
    assert args.speculative_dflash_lora_tts_optimizer_dtype == "float32"
    args = parser.parse_args(
        base
        + [
            "--speculative-dflash-lora-tts-optimizer-dtype",
            "bfloat16",
            "--speculative-dflash-lora-tts-no-offload",
        ]
    )
    assert args.speculative_dflash_lora_tts_optimizer_dtype == "bfloat16"
    assert args.speculative_dflash_lora_tts_offload is False


def _run_all():
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print("ALL PASS" if not failed else f"{failed} FAILED")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
