"""Tests for the startup validation of full-weight DFlash TTS.

`--speculative-dflash-tts-*` is validated in `ServerArgs`:
`_handle_speculative_decoding` requires `--speculative-algorithm DFLASH` and
calls `_validate_dflash_full_weight_tts`, which rejects every configuration
that full-weight TTS cannot serve. Each rejection is checked for its own
message, and the accepted configurations are checked to pass.

A `ServerArgs` is built on the CPU engine (`device="cpu"`), which needs no
accelerator. `_handle_speculative_decoding` is then called on it with the one
step that reads a checkpoint from the Hugging Face hub
(`_resolve_speculative_algorithm_alias`) replaced by the identity. Limit: the
rest of `ServerArgs.__post_init__` (device, backend and memory defaults) is
exercised with CPU defaults only, so the checks here cover the DFLASH / TTS
validation, not the interplay with accelerator-specific argument handling.

Always runs on CPU; needs no GPU, checkpoint or server.

  python test/registered/spec/dflash_tts/test_dflash_tts_args.py
"""

import argparse
import os
import sys

os.environ.setdefault("SGLANG_USE_CPU_ENGINE", "1")

import sglang.srt.server_args as server_args_module  # noqa: E402
from sglang.srt.server_args import ServerArgs  # noqa: E402

FAILS = []


def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' — ' + extra) if extra else ''}")
    if not cond:
        FAILS.append(name)


# The alias resolution downloads the draft model's config; the algorithm is
# already spelled out here.
server_args_module._resolve_speculative_algorithm_alias = lambda algo, *a, **k: algo


def make_args(**overrides):
    """A CPU ServerArgs for DFLASH with full-weight TTS and valid settings."""
    args = ServerArgs(model_path="dummy", device="cpu")
    args.speculative_algorithm = "DFLASH"
    args.speculative_draft_model_path = "draft"
    args.speculative_dflash_block_size = 8
    args.speculative_dflash_tts_enable = True
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def run(**overrides):
    """(args, message): the arguments after `_handle_speculative_decoding`, and
    the ValueError message it raised (None if it did not raise)."""
    args = make_args(**overrides)
    try:
        args._handle_speculative_decoding()
    except ValueError as e:
        return args, str(e)
    return args, None


def message_of(**overrides):
    return run(**overrides)[1]


def rejects(name, expected, **overrides):
    msg = message_of(**overrides)
    check(
        name,
        msg is not None and expected in msg,
        f"expected {expected!r} in: {(msg or 'no error')[:90]}",
    )


print("\n[1] a valid configuration is accepted")
args, msg = run(max_running_requests=1)
check(
    "default settings with max_running_requests 1 pass",
    msg is None
    and args.max_running_requests == 1
    and args.speculative_algorithm == "DFLASH",
    str(msg),
)
check("default max_running_requests is unset", make_args().max_running_requests is None)
args, msg = run()
check(
    "max_running_requests None becomes 1",
    msg is None and args.max_running_requests == 1,
    f"{args.max_running_requests} {msg}",
)
check(
    "kv_cache_dtype auto is accepted",
    message_of(kv_cache_dtype="auto", max_running_requests=1) is None,
)
for dtype in ("float32", "bfloat16"):
    check(
        f"optimizer dtype {dtype} is accepted",
        message_of(
            speculative_dflash_tts_optimizer_dtype=dtype, max_running_requests=1
        )
        is None,
    )
check(
    "boundary values are accepted (lr 0, lambda 0 and 1, freq 1, kl 0)",
    all(
        message_of(max_running_requests=1, **kw) is None
        for kw in (
            dict(speculative_dflash_tts_lr=0.0),
            dict(speculative_tts_lambda=0.0),
            dict(speculative_tts_lambda=1.0),
            dict(speculative_dflash_tts_freq=1),
            dict(speculative_dflash_tts_kl_threshold=0.0),
        )
    ),
)

print("\n[2] --speculative-dflash-tts-enable requires the DFLASH algorithm")
for algo in (None, "EAGLE3", "NGRAM"):
    rejects(
        f"algorithm {algo}: rejected, naming the DFLASH requirement",
        "--speculative-dflash-tts-enable requires --speculative-algorithm DFLASH",
        speculative_algorithm=algo,
        max_running_requests=1,
    )
args, msg = run(speculative_algorithm="dflash", max_running_requests=1)
check(
    "a lower-case 'dflash' is normalized and accepted",
    msg is None and args.speculative_algorithm == "DFLASH",
    str(msg),
)

print("\n[3] concurrency")
rejects(
    "max_running_requests 2 is rejected as a single-request mode",
    "serves a single in-flight request",
    max_running_requests=2,
)
rejects(
    "max_running_requests 64 is rejected too",
    "single in-flight request",
    max_running_requests=64,
)

print("\n[4] incompatible features")
rejects(
    "combination with LoRA TTS is rejected as mutually exclusive",
    "mutually exclusive",
    speculative_dflash_lora_tts_enable=True,
    max_running_requests=1,
)
rejects(
    "--enable-lora is rejected",
    "incompatible with --enable-lora",
    enable_lora=True,
    max_running_requests=1,
)
rejects(
    "--lora-paths is rejected",
    "incompatible with --enable-lora",
    lora_paths=["adapter=/nonexistent"],
    max_running_requests=1,
)
for dtype in ("fp8_e4m3", "fp8_e5m2"):
    rejects(
        f"kv_cache_dtype {dtype} is rejected",
        "requires --kv-cache-dtype auto",
        kv_cache_dtype=dtype,
        max_running_requests=1,
    )

print("\n[5] numeric settings")
rejects(
    "an unknown optimizer dtype is rejected",
    "optimizer-dtype must be float32 or bfloat16",
    speculative_dflash_tts_optimizer_dtype="float16",
    max_running_requests=1,
)
rejects(
    "a negative learning rate is rejected",
    "tts-lr must be non-negative",
    speculative_dflash_tts_lr=-1e-5,
    max_running_requests=1,
)
rejects(
    "lambda above 1 is rejected",
    "tts-lambda must be in [0, 1]",
    speculative_tts_lambda=1.01,
    max_running_requests=1,
)
rejects(
    "lambda below 0 is rejected",
    "tts-lambda must be in [0, 1]",
    speculative_tts_lambda=-0.01,
    max_running_requests=1,
)
rejects(
    "freq 0 is rejected",
    "tts-freq must be >= 1",
    speculative_dflash_tts_freq=0,
    max_running_requests=1,
)
rejects(
    "gamma 0 is rejected",
    "pos-weight-gamma must be positive",
    speculative_dflash_tts_pos_weight_gamma=0.0,
    max_running_requests=1,
)
rejects(
    "a negative gamma is rejected",
    "pos-weight-gamma must be positive",
    speculative_dflash_tts_pos_weight_gamma=-3.0,
    max_running_requests=1,
)
rejects(
    "a negative KL threshold is rejected",
    "kl-threshold must be non-negative",
    speculative_dflash_tts_kl_threshold=-0.1,
    max_running_requests=1,
)

print("\n[6] with the flag off, none of the TTS validation runs")
off = dict(speculative_dflash_tts_enable=False)
args, msg = run(max_running_requests=8, **off)
check(
    "max_running_requests 8 is left alone",
    msg is None and args.max_running_requests == 8,
    f"{args.max_running_requests} {msg}",
)
args, msg = run(**off)
check(
    "max_running_requests gets the generic DFLASH default, not the cap to 1",
    msg is None
    and args.max_running_requests is not None
    and args.max_running_requests != 1,
    f"{args.max_running_requests} {msg}",
)
check(
    "invalid TTS numbers are ignored",
    message_of(
        speculative_dflash_tts_lr=-1.0,
        speculative_tts_lambda=7.0,
        speculative_dflash_tts_freq=0,
        speculative_dflash_tts_pos_weight_gamma=-1.0,
        speculative_dflash_tts_kl_threshold=-1.0,
        speculative_dflash_tts_optimizer_dtype="float16",
        **off,
    )
    is None,
)
check(
    "kv_cache_dtype fp8 and --lora-paths are not rejected by the TTS validation",
    message_of(
        kv_cache_dtype="fp8_e4m3",
        lora_paths=["adapter=/nonexistent"],
        enable_lora=True,
        **off,
    )
    is None,
)
msg = message_of(speculative_algorithm="NGRAM", **off)
check(
    "another algorithm without the flag is not told to use DFLASH",
    msg is None or "speculative-dflash-tts-enable" not in msg,
    str(msg)[:60],
)

print("\nLoRA TTS replay chunk")
lora = dict(speculative_dflash_tts_enable=False, speculative_dflash_lora_tts_enable=True)
rejects(
    "a replay chunk below 1 is rejected",
    "--speculative-dflash-lora-tts-replay-chunk",
    speculative_dflash_lora_tts_replay_chunk=0,
    **lora,
)
check(
    "the default replay chunk is accepted",
    "replay-chunk" not in str(message_of(**lora)),
)

print("\nthe TTS loss flags are shared by every speculator")
parser = argparse.ArgumentParser()
ServerArgs.add_cli_args(parser)
base = ["--model-path", "dummy"]
parsed = parser.parse_args(base)
check(
    "the default loss is lk",
    parsed.speculative_tts_loss == "lk" and ServerArgs.speculative_tts_loss == "lk",
)
check(
    "--speculative-tts-loss and --speculative-tts-lambda parse",
    (lambda a: (a.speculative_tts_loss, a.speculative_tts_lambda))(
        parser.parse_args(
            base + ["--speculative-tts-loss", "kl", "--speculative-tts-lambda", "0.5"]
        )
    )
    == ("kl", 0.5),
)
check(
    "the per-speculator loss flags are gone",
    not any(
        opt in parser._option_string_actions
        for opt in (
            "--speculative-eagle-tts-loss",
            "--speculative-eagle-tts-lambda",
            "--speculative-dflash-tts-lambda",
            "--speculative-dflash-lora-tts-lambda",
        )
    ),
)
rejects(
    "an unknown loss is rejected",
    "--speculative-tts-loss must be",
    speculative_tts_loss="l2",
    max_running_requests=1,
)
rejects(
    "the shared lambda is validated for per-request DFlash TTS",
    "tts-lambda must be in [0, 1]",
    speculative_tts_lambda=1.5,
    **lora,
)

print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
