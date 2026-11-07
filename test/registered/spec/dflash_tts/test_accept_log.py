"""--speculative-accept-log-dir and the script that tabulates its files.

Needs no GPU, checkpoint or server:

  python test/registered/spec/dflash_tts/test_accept_log.py
"""

import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("SGLANG_USE_CPU_ENGINE", "1")

from sglang.srt.speculative.accept_log import AcceptLog, make_accept_log

FAILS = []


def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' — ' + extra) if extra else ''}")
    if not cond:
        FAILS.append(name)


script = Path(__file__).parents[4] / "examples/tts/accept_length_curve.py"
spec = importlib.util.spec_from_file_location("accept_length_curve", script)
curve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(curve)

with tempfile.TemporaryDirectory() as tmp:
    args = SimpleNamespace(speculative_accept_log_dir=os.path.join(tmp, "logs"))
    check("off without a directory", make_accept_log(SimpleNamespace(), 0) is None)
    check("only rank 0 writes", make_accept_log(args, 1) is None)

    log = make_accept_log(args, 0)
    def req(rid):
        return SimpleNamespace(rid=rid)

    reqs = [req("a"), req("../../evil/b")]
    log.write(reqs, [3, 0])
    log.write(reqs[:1], [5])
    log.write([req("HEALTH_CHECK_1")], [1])
    a = os.path.join(args.speculative_accept_log_dir, "a.txt")
    check("one line per round", curve.read_rounds(a) == [3, 5])
    names = sorted(os.listdir(args.speculative_accept_log_dir))
    check(
        "a client's request id cannot leave the directory",
        # and health checks get no file
        names == [".._.._evil_b.txt", "a.txt"]
        and not os.path.exists(os.path.join(tmp, "evil")),
        str(names),
    )

    not_a_dir = os.path.join(tmp, "file")
    Path(not_a_dir).touch()
    broken = AcceptLog(tmp)
    broken.log_dir = not_a_dir
    broken.write(reqs, [1, 1])
    check("a failed write disables the log and does not raise", not broken._enabled)

print("\n the curve")
rows = curve.binned_accept_length([[1, 3, 7], [0]], bin_size=4)
# Request 0: rounds of 2, 4 and 8 tokens, starting at output lengths 1, 3 and 7.
# Request 1: a round of 1 token at output length 1.
check(
    "rounds are binned by the output length at which they start",
    rows == [(0, 2, 2 + 4 + 1, 3), (4, 1, 8, 1)],
    str(rows),
)
rows = curve.binned_accept_length([[9, 1, 1]], bin_size=10)
check("later rounds land in later bins", rows == [(0, 1, 10, 1), (10, 1, 4, 2)], str(rows))

print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
sys.exit(1 if FAILS else 0)
