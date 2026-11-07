#!/usr/bin/env python3
"""Tabulate acceptance length against output length.

Reads the directory written by a server started with
--speculative-accept-log-dir and pools all its requests:

  python examples/tts/accept_length_curve.py al_logs --bin-size 2000

The server's warm-up request has a file too. It is a few rounds long; to leave it
out, give your requests a `rid` with a common prefix and pass --prefix.
"""

import argparse
import os
from collections import defaultdict


def read_rounds(path):
    """The num_correct_drafts of each verify round of one request."""
    with open(path) as f:
        return [int(line) for line in f if line.strip()]


def binned_accept_length(requests, bin_size):
    """Rows (bin start, requests, tokens, rounds), binned by output length."""
    tokens, rounds, seen = defaultdict(int), defaultdict(int), defaultdict(set)
    for r, per_round in enumerate(requests):
        output_len = 1  # prefill emits the first token
        for num_correct in per_round:
            b = output_len // bin_size
            tokens[b] += num_correct + 1  # the bonus token
            rounds[b] += 1
            seen[b].add(r)
            output_len += num_correct + 1
    return [(b * bin_size, len(seen[b]), tokens[b], rounds[b]) for b in sorted(tokens)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("log_dir")
    ap.add_argument("--bin-size", type=int, default=1000, help="In output tokens.")
    ap.add_argument("--prefix", default="", help="Only request ids with this prefix.")
    args = ap.parse_args()

    paths = [
        os.path.join(args.log_dir, name)
        for name in sorted(os.listdir(args.log_dir))
        if name.startswith(args.prefix) and name.endswith(".txt")
    ]
    if not paths:
        ap.error(f"no request logs in {args.log_dir}")

    print(f"{'output tokens':>15} {'requests':>9} {'tokens':>9} {'rounds':>8} {'AL':>6}")
    for start, num_requests, tokens, rounds in binned_accept_length(
        [read_rounds(p) for p in paths], args.bin_size
    ):
        span = f"{start}-{start + args.bin_size}"
        print(f"{span:>15} {num_requests:>9} {tokens:>9} {rounds:>8} {tokens / rounds:>6.2f}")


if __name__ == "__main__":
    main()
