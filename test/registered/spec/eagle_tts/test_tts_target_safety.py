"""Checks that the target model stays bit-identical under full-mode TTS.

EAGLE-3's `init_lm_head` rebinds the draft's `embed_tokens.weight` (and, for a
draft without a head of its own, its `lm_head.weight`) to the target's tensor
objects (`set_embed` / `set_embed_and_head` in models/llama.py). They share
storage, so an optimizer step on them would modify the target in place.

In speculative decoding the target defines the emitted text, so a modified
target would change what is generated and decoding would stop being lossless.

Runs on CPU with toy modules; needs no checkpoint.

  python test/registered/spec/eagle_tts/test_tts_target_safety.py
"""

import sys

import torch
from torch import nn

from sglang.srt.speculative.eagle_tts import EagleTTSConfig, EagleTTSState

FAILURES = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


class Target(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 8)
        self.lm_head = nn.Linear(8, 16, bias=False)
        self.body = nn.Linear(8, 8, bias=False)


class Draft(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 8)
        self.lm_head = nn.Linear(8, 16, bias=False)
        self.fc = nn.Linear(8, 8, bias=False)


def main():
    torch.manual_seed(0)
    target, draft = Target(), Draft()

    # Reproduce set_embed_and_head: bind both to the target's tensors.
    draft.embed_tokens.weight = target.embed_tokens.weight
    draft.lm_head.weight = target.lm_head.weight

    print("setup")
    check(
        "draft aliases target before TTS (reproduces sglang's set_embed_and_head)",
        draft.embed_tokens.weight.data_ptr() == target.embed_tokens.weight.data_ptr()
        and draft.lm_head.weight.data_ptr() == target.lm_head.weight.data_ptr(),
    )

    before = {n: p.detach().clone() for n, p in target.named_parameters()}

    cfg = EagleTTSConfig(enable=True, lr=1e-2, freq=1, async_step=False)
    state = EagleTTSState(
        draft_model=draft,
        config=cfg,
        device="cpu",
        tp_size=1,
        target_model=target,
    )

    print("privatization")
    check(
        "draft does not alias the target after TTS init",
        draft.embed_tokens.weight.data_ptr() != target.embed_tokens.weight.data_ptr()
        and draft.lm_head.weight.data_ptr() != target.lm_head.weight.data_ptr(),
    )
    check(
        "private copies start numerically identical",
        torch.equal(draft.embed_tokens.weight, target.embed_tokens.weight)
        and torch.equal(draft.lm_head.weight, target.lm_head.weight),
    )
    check(
        "head and embedding are trainable",
        any(p is draft.lm_head.weight for p in state._params)
        and any(p is draft.embed_tokens.weight for p in state._params),
        f"{len(state._params)} trainable tensors",
    )
    check(
        "no trainable tensor shares storage with the target",
        not ({p.data_ptr() for p in state._params}
             & {p.data_ptr() for p in target.parameters()}),
    )

    print("many TTS steps")
    for _ in range(25):
        ids = torch.randint(0, 16, (6,))
        h = draft.fc(draft.embed_tokens(ids))
        draft_logits = draft.lm_head(h)
        target_logits = torch.randn(6, 16)
        state.step(draft_logits=draft_logits, target_logits=target_logits)

    moved = [
        n for n, p in draft.named_parameters()
        if not torch.equal(p, before.get(n.replace("fc", "body"), p))
    ]
    check("draft weights actually moved", len(moved) > 0, f"{len(moved)} changed")

    print("the target after 25 steps")
    unchanged = all(
        torch.equal(p, before[n]) for n, p in target.named_parameters()
    )
    drifted = [n for n, p in target.named_parameters() if not torch.equal(p, before[n])]
    check("EVERY target parameter is bit-identical", unchanged,
          "" if unchanged else f"CORRUPTED: {drifted}")

    print("reset does not write to the target either")
    state.reset()
    still = all(torch.equal(p, before[n]) for n, p in target.named_parameters())
    check("target still bit-identical after reset", still)

    print("\n" + ("FAILED: " + ", ".join(FAILURES) if FAILURES else "all checks passed"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
