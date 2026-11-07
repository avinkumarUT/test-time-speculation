"""Rebuild the paper's batched-serving throughput tables from the raw results.

Each file under results/<suite>/<benchmark>.jsonl holds one JSON record per
(batch size, arm) cell, written by the cell runners in this directory. This
script prints, per suite, aggregate throughput (tokens/sec) of the frozen
speculator baseline and of TTS, and the TTS speedup, at each batch size.
Point --results at a directory of your own runs to tabulate those instead.

Usage:
    python experiments/tts/make_tables.py                # all suites, Markdown
    python experiments/tts/make_tables.py --format latex
    python experiments/tts/make_tables.py --suite dflash_qwen3.6-27b
    python experiments/tts/make_tables.py --metric al    # acceptance length
"""

import argparse
import json
import os
from decimal import ROUND_HALF_UP, Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS = os.path.join(HERE, "results")

# Suite directory -> (table title, baseline row label).
SUITES = {
    "dflash_qwen3.6-27b": ("Qwen3.6-27B + DFlash", "DFlash"),
    "dflash_qwen3.5-122b": ("Qwen3.5-122B-A10B + DFlash", "DFlash"),
}

# Paper row order and display names.
BENCHMARKS = [
    ("aime2024", "AIME 2024"),
    ("aime2025", "AIME 2025"),
    ("gpqa_diamond", "GPQA Diamond"),
    ("lcb_hard", "LiveCodeBench"),
    ("math500_hard", "MATH-500"),
    ("theoremqa", "TheoremQA"),
    ("olympiadbench_math", "OlyBench-Math"),
    ("olympiadbench_physics", "OlyBench-Phy"),
]

# metric -> (record field, decimals shown, unit)
#   agg_tok_s  completed tokens / load wall time, as recorded by the runner
#   mean_al    mean over requests of (completion tokens / verify rounds)
METRICS = {
    "throughput": ("agg_tok_s", 1, "tokens/sec"),
    "al": ("mean_al", 2, "mean acceptance length"),
}


def fmt_value(value, decimals):
    """Format a recorded value, rounding half up."""
    quantum = Decimal(1).scaleb(-decimals)
    return str(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def load_suite(suite, results_dir):
    """Return {benchmark: {(bs, 'baseline'|'tts'): record}} and the TTS strides.

    The runners append one record per attempt. Failed attempts (e.g. an
    out-of-memory start that was retried) are ignored, and if a cell was run
    more than once the last ok/partial record wins.
    """
    cells, strides = {}, set()
    suite_dir = os.path.join(results_dir, suite)
    for bench, _ in BENCHMARKS:
        path = os.path.join(suite_dir, f"{bench}.jsonl")
        if not os.path.exists(path):
            continue
        per_bench = {}
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("status") not in ("ok", "partial"):
                    continue
                kind = "baseline" if rec["arm"] == "baseline" else "tts"
                per_bench[(rec["bs"], kind)] = rec
        for (_, kind), rec in per_bench.items():
            if kind == "tts":
                strides.add(rec["tts_freq"])
        cells[bench] = per_bench
    return cells, sorted(strides)


def build_rows(suite, metric, results_dir=DEFAULT_RESULTS):
    field, decimals, _ = METRICS[metric]
    _, base_label = SUITES[suite]
    cells, strides = load_suite(suite, results_dir)
    batch_sizes = sorted({bs for per in cells.values() for bs, _ in per})
    rows, ratios, partial = [], [], False
    for bench, name in BENCHMARKS:
        per = cells.get(bench)
        if per is None:
            continue
        base_cols, tts_cols = [], []
        for bs in batch_sizes:
            base, tts = per.get((bs, "baseline")), per.get((bs, "tts"))
            if base is None or tts is None:
                base_cols.append("-")
                tts_cols.append("-")
                continue
            ratio = tts[field] / base[field]
            ratios.append(ratio)
            star = ""
            if base["status"] != "ok":
                star, partial = "*", True
            base_cols.append(fmt_value(base[field], decimals) + star)
            star = ""
            if tts["status"] != "ok":
                star, partial = "*", True
            tts_cols.append(f"{fmt_value(tts[field], decimals)}{star} ({ratio:.2f}x)")
        rows.append((name, base_label, base_cols))
        rows.append(("", "TTS", tts_cols))
    return batch_sizes, rows, ratios, strides, partial


def print_markdown(suite, metric, results_dir):
    title, _ = SUITES[suite]
    batch_sizes, rows, ratios, strides, partial = build_rows(suite, metric, results_dir)
    if not rows:
        return
    unit = METRICS[metric][2]
    stride = ", ".join(f"S={s}" for s in strides)
    print(f"### {title} (TTS {stride}), {unit}\n")
    print("| Benchmark | Approach | " + " | ".join(f"BS={b}" for b in batch_sizes) + " |")
    print("|---|---|" + "---:|" * len(batch_sizes))
    for name, label, cols in rows:
        print(f"| {name} | {label} | " + " | ".join(cols) + " |")
    print(f"\nTTS / baseline: min {min(ratios):.2f}x, max {max(ratios):.2f}x, "
          f"mean {sum(ratios) / len(ratios):.2f}x over {len(ratios)} cells.")
    if partial:
        print("\\* cell stopped at the per-cell wall-clock cap; the value is "
              "computed over the requests that completed.")
    print()


def print_latex(suite, metric, results_dir):
    title, _ = SUITES[suite]
    batch_sizes, rows, _, strides, _ = build_rows(suite, metric, results_dir)
    if not rows:
        return
    stride = ", ".join(f"S={s}" for s in strides)
    print(f"% {title} (TTS {stride}), {METRICS[metric][2]}")
    print("\\begin{tabular}{ll" + "r" * len(batch_sizes) + "}")
    print("\\toprule")
    print("Benchmark & Approach & " + " & ".join(f"BS={b}" for b in batch_sizes) + " \\\\")
    for name, label, cols in rows:
        if name:
            print("\\midrule")
            name = f"\\multirow{{2}}{{*}}{{{name}}}"
        cols = [c.replace("x)", "$\\times$)").replace("*", "$^{*}$") for c in cols]
        print(f"{name} & {label} & " + " & ".join(cols) + " \\\\")
    print("\\bottomrule")
    print("\\end{tabular}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--suite", choices=sorted(SUITES), action="append",
                    help="Suite(s) to print; default: every suite with results.")
    ap.add_argument("--format", choices=["markdown", "latex"], default="markdown")
    ap.add_argument("--metric", choices=sorted(METRICS), default="throughput")
    ap.add_argument("--results", default=DEFAULT_RESULTS,
                    help="Directory holding <suite>/<benchmark>.jsonl files.")
    args = ap.parse_args()
    printer = print_markdown if args.format == "markdown" else print_latex
    for suite in args.suite or list(SUITES):
        printer(suite, args.metric, args.results)


if __name__ == "__main__":
    main()
