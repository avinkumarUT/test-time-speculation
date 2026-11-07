"""Batched-serving benchmark for DFlash, with and without TTS.

Runs a grid of cells, one server lifecycle per cell:

    launch server -> wait until healthy -> load -> record -> kill -> drain GPUs

A cell is (batch size, arm). Arms:
  baseline   the frozen DFlash speculator.
  tts_s<K>   TTS with update stride K: per-request LoRA adapters with
             delayed adapter publication.

Load is a sliding window: exactly `bs` requests are in flight until N =
max(2*bs, 32) requests have completed (prompts are cycled if the benchmark
has fewer than N). Decoding is greedy with a 32768-token cap and natural
stopping. One JSON record per cell is appended to the output file; cells that
already have an ok/partial record are skipped, so an interrupted grid can
simply be restarted.

Example (the paper's Qwen3.6-27B grid for one benchmark, 4 GPUs):

    python experiments/tts/run_dflash_cells.py \
        --target Qwen/Qwen3.6-27B --draft z-lab/Qwen3.6-27B-DFlash \
        --prompts experiments/tts/prompts/aime2024__qwen3.6-27b.json \
        --benchmark aime2024 --gpus 0,1,2,3 --port 31201 \
        --batch-sizes 8,16,32,64 --arms baseline,tts_s24 \
        --extra-server-args="--enable-deterministic-inference" \
        --out experiments/tts/runs/dflash_qwen3.6-27b/aime2024.jsonl

See experiments/tts/README.md for the full recipes.
"""
import argparse
import json
import os
import random
import re
import signal
import statistics
import string
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
MAX_NEW = 32768


def arm_stride(arm):
    """TTS update stride for an arm name: 'tts_s<K>' -> K, 'baseline' -> None."""
    if arm.startswith("tts_s"):
        return int(arm[len("tts_s"):])
    if arm == "baseline":
        return None
    raise ValueError(f"unknown arm {arm!r}; use 'baseline' or 'tts_s<K>'")


STARTUP_TIMEOUT = 30 * 60          # poll /health_generate this long
CELL_WALL_CAP = 2.5 * 3600         # per-cell load-window cap
DRAIN_MB = 2000                    # GPUs must drop below this before next cell

HEALTH_PATTERNS = [
    "step failed", "disabled", "desync", "Scheduler hit an exception",
    "illegal memory", "CUDA error", "out of memory",
]
# Benign startup/config lines that would otherwise trip the generic patterns.
BENIGN_RE = re.compile(
    r"radix cache|cuda graph|chunked prefill|overlap sched|torch\.compile|"
    r"custom all-?reduce|hierarchical|jit|flashinfer|dp attention|"
    r"disable_radix_cache|disable_cuda_graph|is disabled because|"
    r"deterministic|penalizer|profil",
    re.IGNORECASE,
)
RUNNING_REQ_RE = re.compile(r"#running-req[:=]\s*(\d+)")


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def http_json(url, payload=None, timeout=30):
    if payload is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    try:
        return json.loads(body)
    except Exception:
        return {"raw": body.decode(errors="replace")}


def gpu_used_mb(gpu_ids):
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=60,
    ).stdout
    used = {}
    for line in out.strip().splitlines():
        idx, mem = line.split(",")
        used[int(idx)] = int(mem.strip())
    return [used.get(g, 0) for g in gpu_ids]


class Cell:
    def __init__(self, args, bs, arm, mem_fraction=0.86, attempt=0):
        self.args = args
        self.bs = bs
        self.arm = arm
        self.mf = mem_fraction
        self.attempt = attempt
        self.name = f"{args.benchmark}_bs{bs}_{arm}" + (
            f"_retry{attempt}" if attempt else "")
        self.log_path = os.path.join(args.log_dir, f"server_{self.name}.log")
        self.proc = None

    def is_tts(self):
        return self.arm != "baseline"

    def replay_chunk(self):
        """--speculative-dflash-lora-tts-replay-chunk; None = server default.

        Bounds how many requests' updates are replayed together. At bs=64 a
        small chunk keeps the update's transient memory inside the budget.
        """
        if self.is_tts() and self.bs == 64:
            return self.args.bs64_replay_chunk
        return None

    def server_cmd(self):
        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", self.args.target,
            "--speculative-algorithm", "DFLASH",
            "--speculative-draft-model-path", self.args.draft,
            "--tp", "4",
            "--port", str(self.args.port),
            "--mem-fraction-static", f"{self.mf:g}",
            "--max-running-requests", str(self.bs),
            "--disable-radix-cache",
            "--trust-remote-code",
        ]
        if self.args.context_length:
            # With deterministic inference the attention backend sizes some
            # CUDA-graph index buffers by the context length. The models'
            # default (262144) makes those buffers ~2 GB each, which does not
            # fit next to a bs=64 KV pool. The longest request here is ~3.4K
            # prompt + 32768 generated tokens, so 40960 is never reached.
            cmd += ["--context-length", str(self.args.context_length)]
        if self.args.extra_server_args:
            cmd += self.args.extra_server_args.split()
        if self.is_tts():
            cmd += [
                "--speculative-dflash-lora-tts-enable",
                "--speculative-tts-loss", self.args.tts_loss,
                "--speculative-dflash-lora-tts-freq", str(arm_stride(self.arm)),
                "--speculative-dflash-lora-tts-delayed-publish",
            ]
            ck = self.replay_chunk()
            if ck is not None:
                cmd += ["--speculative-dflash-lora-tts-replay-chunk", str(ck)]
        return cmd

    def launch(self):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in self.args.gpu_ids)
        # FlashInfer workspace size: 4 GB by default; set
        # SGLANG_FLASHINFER_WORKSPACE_SIZE to override (the 122B bs=64 recipe
        # uses 2 GB together with --chunked-prefill-size 4096).
        env.setdefault("SGLANG_FLASHINFER_WORKSPACE_SIZE", str(4096 * 1024 * 1024))
        self.log_f = open(self.log_path, "w")
        self.proc = subprocess.Popen(
            self.server_cmd(), stdout=self.log_f, stderr=subprocess.STDOUT,
            env=env, cwd=REPO, start_new_session=True,
        )
        log(f"{self.name}: launched server pid={self.proc.pid} mf={self.mf:g}")

    def wait_ready(self):
        t0 = time.time()
        url = f"http://127.0.0.1:{self.args.port}/health_generate"
        while time.time() - t0 < self.args.startup_timeout:
            if self.proc.poll() is not None:
                return None  # died
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=20) as r:
                    if r.status == 200:
                        return time.time() - t0
            except Exception:
                pass
            time.sleep(5)
        return None

    def kill(self):
        """SIGTERM then SIGKILL this cell's own process group only."""
        if self.proc is None:
            return
        for sig, wait_s in ((signal.SIGTERM, 30), (signal.SIGKILL, 30)):
            try:
                os.killpg(self.proc.pid, sig)
            except ProcessLookupError:
                break
            except Exception as e:
                log(f"{self.name}: killpg {sig} error: {e}")
            t0 = time.time()
            while time.time() - t0 < wait_s:
                if self.proc.poll() is not None:
                    break
                time.sleep(1)
            if self.proc.poll() is not None:
                break
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except Exception:
            pass
        try:
            self.log_f.close()
        except Exception:
            pass

    def wait_drain(self):
        t0 = time.time()
        used = []
        while time.time() - t0 < 300:
            try:
                used = gpu_used_mb(self.args.gpu_ids)
            except Exception:
                time.sleep(5)
                continue
            if all(u < DRAIN_MB for u in used):
                log(f"{self.name}: GPUs drained {used} MiB")
                return True
            time.sleep(5)
        log(f"{self.name}: WARNING GPUs not drained after 300s: {used}")
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except Exception:
            pass
        time.sleep(15)
        return all(u < DRAIN_MB for u in gpu_used_mb(self.args.gpu_ids))

    def tail_log(self, n=50):
        try:
            with open(self.log_path, errors="replace") as f:
                return f.readlines()[-n:]
        except Exception:
            return []

    def log_has_oom(self):
        try:
            with open(self.log_path, errors="replace") as f:
                txt = f.read()
            return ("out of memory" in txt.lower()
                    or "OutOfMemoryError" in txt
                    or "CUDA error: out of memory" in txt)
        except Exception:
            return False


def grep_health(log_path):
    health, examples = {}, []
    try:
        with open(log_path, errors="replace") as f:
            lines = f.readlines()
        for pat in HEALTH_PATTERNS:
            cnt = 0
            for ln in lines:
                if pat.lower() in ln.lower():
                    if pat == "disabled" and BENIGN_RE.search(ln):
                        continue
                    cnt += 1
                    if len(examples) < 8:
                        examples.append(ln.strip()[:300])
            if cnt:
                health[pat] = cnt
    except Exception:
        pass
    return health, examples


def run_load(cell, prompts):
    """Sliding-window load: exactly bs requests in flight until N complete."""
    args = cell.args
    bs = cell.bs
    n_total = max(2 * bs, 32)
    base = f"http://127.0.0.1:{args.port}"
    deadline = time.time() + args.cell_wall_cap
    results = []
    res_lock = threading.Lock()
    aborted = threading.Event()

    # /abort_request matches request ids by prefix, so ids must be unique and
    # not prefixes of each other: zero-padded index plus a random suffix.
    def make_rid(i):
        suf = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
        return f"c{cell.name}-{i:04d}-{suf}"

    tasks = []
    for i in range(n_total):
        p = prompts[i % len(prompts)]
        tasks.append((i, i % len(prompts), p["formatted"], make_rid(i)))

    task_iter = iter(tasks)
    iter_lock = threading.Lock()

    def worker():
        while True:
            with iter_lock:
                try:
                    idx, p_idx, text, rid = next(task_iter)
                except StopIteration:
                    return
            if aborted.is_set() or time.time() > deadline:
                with res_lock:
                    results.append({"i": idx, "prompt_idx": p_idx,
                                    "status": "skipped_deadline"})
                continue
            t0 = time.time()
            try:
                remaining = max(60, deadline - time.time() + 300)
                out = http_json(
                    f"{base}/generate",
                    {"text": text, "rid": rid,
                     "sampling_params": {"temperature": 0.0,
                                         "max_new_tokens": MAX_NEW}},
                    timeout=remaining,
                )
                mi = out.get("meta_info", {})
                comp = mi.get("completion_tokens")
                vct = mi.get("spec_verify_ct")
                rec = {
                    "i": idx, "prompt_idx": p_idx, "status": "ok",
                    "completion_tokens": comp, "spec_verify_ct": vct,
                    "latency_s": round(time.time() - t0, 1),
                    "finish_reason": (mi.get("finish_reason") or {}).get("type"),
                }
                if comp and vct:
                    rec["al"] = round(comp / vct, 4)
                with res_lock:
                    results.append(rec)
            except Exception as e:
                with res_lock:
                    results.append({
                        "i": idx, "prompt_idx": p_idx, "status": "error",
                        "error": f"{type(e).__name__}: {e}",
                        "latency_s": round(time.time() - t0, 1),
                    })

    # log offset for #running-req parsing over the load window only
    log_offset = os.path.getsize(cell.log_path) if os.path.exists(cell.log_path) else 0

    t_start = time.time()
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(bs)]
    for t in threads:
        t.start()

    partial = False
    while any(t.is_alive() for t in threads):
        if time.time() > deadline and not aborted.is_set():
            log(f"{cell.name}: WALL CAP hit, aborting remaining requests")
            aborted.set()
            partial = True
            try:
                http_json(f"{base}/abort_request", {"abort_all": True},
                          timeout=30)
            except Exception as e:
                log(f"{cell.name}: abort_all failed: {e}")
        time.sleep(5)
    for t in threads:
        t.join(timeout=60)
    wall_s = time.time() - t_start

    # parse #running-req over the load window
    running = []
    try:
        with open(cell.log_path, errors="replace") as f:
            f.seek(log_offset)
            for line in f:
                m = RUNNING_REQ_RE.search(line)
                if m:
                    running.append(int(m.group(1)))
    except Exception:
        pass

    health, examples = grep_health(cell.log_path)

    ok = [r for r in results if r.get("status") == "ok" and r.get("al")]
    comp_sum = sum(r["completion_tokens"] for r in ok)
    als = [r["al"] for r in ok]
    record = {
        "benchmark": args.benchmark,
        "bs": bs,
        "arm": cell.arm,
        "tts_loss": cell.args.tts_loss if cell.is_tts() else None,
        "N": n_total,
        "status": "partial" if partial else "ok",
        "wall_s": round(wall_s, 1),
        "n_complete": len(ok),
        "n_error": sum(1 for r in results if r.get("status") == "error"),
        "n_skipped": sum(1 for r in results if r.get("status") == "skipped_deadline"),
        "sum_completion_tokens": comp_sum,
        "agg_tok_s": round(comp_sum / wall_s, 2) if wall_s > 0 else 0,
        "mean_al": round(statistics.mean(als), 4) if als else None,
        "median_al": round(statistics.median(als), 4) if als else None,
        "mean_completion_tokens": round(comp_sum / len(ok), 1) if ok else None,
        "running_req_max": max(running) if running else None,
        "running_req_p50": int(statistics.median(running)) if running else None,
        "health": health,
        "health_examples": examples,
        "per_request": results,
    }
    return record


def run_cell(args, prompts, bs, arm, mf, attempt):
    """Run one server lifecycle; returns the cell record (never None)."""
    cell = Cell(args, bs, arm, mem_fraction=mf, attempt=attempt)
    log(f"=== CELL {cell.name} start (mf={mf:g}) ===")
    record = None
    try:
        cell.launch()
        startup_s = cell.wait_ready()
        if startup_s is None:
            record = {
                "benchmark": args.benchmark,
                "bs": bs, "arm": arm, "status": "failed",
                "reason": "server did not become healthy before the startup "
                          "timeout (or died)",
                "oom_in_log": cell.log_has_oom(),
                "server_log_tail": cell.tail_log(50),
            }
        else:
            log(f"{cell.name}: server ready in {startup_s:.0f}s")
            record = run_load(cell, prompts)
            record["server_startup_s"] = round(startup_s, 1)
            # A server that crashed mid-load can still have a few early
            # completions. Such a cell is not recorded as ok, so a restart
            # re-runs it.
            crash_sigs = [k for k in record["health"]
                          if k in ("Scheduler hit an exception",
                                   "illegal memory", "CUDA error",
                                   "out of memory")]
            if crash_sigs or record["n_error"] >= max(1, record["N"] // 4):
                record["status"] = "failed"
                record["reason"] = (
                    f"server crashed mid-load: health={crash_sigs}, "
                    f"n_error={record['n_error']}/{record['N']}")
                record["oom_in_log"] = cell.log_has_oom()
            # no completions and an OOM in the server log
            if record["n_complete"] == 0 and cell.log_has_oom():
                record["status"] = "failed"
                record["reason"] = "no completions; OOM in server log"
                record["oom_in_log"] = True
    except Exception as e:
        log(f"{cell.name}: EXCEPTION {type(e).__name__}: {e}")
        traceback.print_exc()
        record = {
            "benchmark": args.benchmark,
            "bs": bs, "arm": arm, "status": "failed",
            "reason": f"driver exception: {type(e).__name__}: {e}",
            "oom_in_log": cell.log_has_oom(),
            "server_log_tail": cell.tail_log(50),
        }
    finally:
        try:
            cell.kill()
        except Exception as e:
            log(f"{cell.name}: kill error {e}")
        try:
            cell.wait_drain()
        except Exception as e:
            log(f"{cell.name}: drain error {e}")
    record["mem_fraction"] = mf
    record["target"] = args.target
    record["draft"] = args.draft
    if cell.is_tts():
        record["tts_freq"] = arm_stride(arm)
        # The speculator was served with its sliding window.
        record["swa"] = True
        record["delayed_publish"] = True
        ck = cell.replay_chunk()
        if ck is not None:
            record["tts_replay_chunk"] = ck
    if attempt:
        record["retry"] = {"attempt": attempt, "mem_fraction": mf,
                           "reason": "OOM at the base memory fraction, "
                                     "retried at a lower one"}
    return record


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--target", required=True, help="Target model (HF id or path).")
    ap.add_argument("--draft", required=True, help="DFlash draft model.")
    ap.add_argument("--prompts", required=True,
                    help="Prompt JSON written by render_prompts.py.")
    ap.add_argument("--benchmark", required=True,
                    help="Benchmark key, recorded with every cell.")
    ap.add_argument("--out", required=True,
                    help="JSONL file that cell records are appended to.")
    ap.add_argument("--gpus", required=True, help="Comma list of 4 GPU ids (TP=4).")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--batch-sizes", default="8,16,32,64",
                    help="Comma list of batch sizes to run, in order.")
    ap.add_argument("--arms", default="baseline,tts_s24",
                    help="Comma list of arms: 'baseline' or 'tts_s<K>' (TTS "
                         "with update stride K).")
    ap.add_argument("--arm-major", action="store_true",
                    help="Order cells arm-major (all batch sizes for arm 1, "
                         "then arm 2) instead of batch-size-major.")
    ap.add_argument("--only-cells", default=None,
                    help="Comma list of bs:arm to run, e.g. '64:tts_s10,32:baseline'.")
    ap.add_argument("--force", action="store_true",
                    help="Re-run cells that already have an ok/partial record "
                         "(the new record is appended; the last one wins).")
    ap.add_argument("--extra-server-args", default="",
                    help="Space-separated extra args appended to launch_server. "
                         "Use the --extra-server-args=\"...\" form.")
    ap.add_argument("--context-length", type=int, default=40960,
                    help="--context-length for the server; 0 keeps the model's "
                         "default.")
    ap.add_argument("--mf", type=float, default=0.86,
                    help="--mem-fraction-static for all cells.")
    ap.add_argument("--mf-bs64", type=float, default=None,
                    help="--mem-fraction-static override for bs=64 cells.")
    ap.add_argument("--retry-mf", type=float, default=0.84,
                    help="First memory fraction to retry at after an OOM; "
                         "further retries step down from it.")
    ap.add_argument("--tts-loss", choices=["kl", "lk"], default="kl",
                    help="TTS loss. The shipped results were produced with kl.")
    ap.add_argument("--bs64-replay-chunk", type=int, default=2,
                    help="--speculative-dflash-lora-tts-replay-chunk for bs=64 TTS cells.")
    ap.add_argument("--startup-timeout", type=float, default=STARTUP_TIMEOUT,
                    help="Seconds to wait for the server to become healthy.")
    ap.add_argument("--cell-wall-cap", type=float, default=CELL_WALL_CAP,
                    help="Per-cell load-window cap in seconds; a cell that hits "
                         "it is recorded as 'partial'.")
    ap.add_argument("--log-dir", default=None,
                    help="Directory for per-cell server logs (default: next to --out).")
    args = ap.parse_args()
    args.gpu_ids = [int(g) for g in args.gpus.split(",")]
    results_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    if args.log_dir is None:
        args.log_dir = os.path.join(os.path.dirname(results_path), "logs")
    os.makedirs(args.log_dir, exist_ok=True)

    with open(args.prompts) as f:
        prompts = json.load(f)
    log(f"benchmark={args.benchmark} gpus={args.gpu_ids} port={args.port} "
        f"prompts={len(prompts)} out={results_path}")

    done = set()
    if os.path.exists(results_path):
        with open(results_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("status") in ("ok", "partial"):
                        done.add((r["bs"], r["arm"]))
                except Exception:
                    pass
    if done:
        log(f"resuming; already done: {sorted(done)}")

    only = None
    if args.only_cells:
        only = set()
        for tok in args.only_cells.split(","):
            b, a = tok.strip().split(":")
            only.add((int(b), a))
        log(f"restricted to cells: {sorted(only)}")

    arms = args.arms.split(",")
    for arm in arms:
        arm_stride(arm)  # validate arm names before launching anything
    bss = [int(x) for x in args.batch_sizes.split(",")]
    if args.arm_major:
        cell_list = [(bs, arm) for arm in arms for bs in bss]
    else:
        cell_list = [(bs, arm) for bs in bss for arm in arms]

    for bs, arm in cell_list:
        if only is not None and (bs, arm) not in only:
            continue
        if (bs, arm) in done and not args.force:
            log(f"skip already-done cell bs={bs} arm={arm}")
            continue
        mf0 = args.mf_bs64 if (bs == 64 and args.mf_bs64 is not None) else args.mf
        record = run_cell(args, prompts, bs, arm, mf0, attempt=0)
        with open(results_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        log(f"=== CELL {args.benchmark}_bs{bs}_{arm} done: status={record['status']} "
            f"al={record.get('mean_al')} tok_s={record.get('agg_tok_s')} ===")
        # OOM retry ladder. Deterministic inference adds CUDA-graph capture
        # and workspace memory on top of the KV pool, and the per-request
        # adapter pool adds more, so a TTS cell at bs>=32 (or a baseline cell
        # at bs=64) can run out of memory at the base fraction. Lowering
        # --mem-fraction-static shrinks the KV pool and leaves more headroom.
        # Three rungs: retry_mf, retry_mf - 0.02, retry_mf - 0.06.
        r = args.retry_mf
        ladder = [r, round(r - 0.02, 4), round(r - 0.06, 4)]
        for attempt, retry_mf in enumerate(ladder, 1):
            if not (record["status"] == "failed" and record.get("oom_in_log")
                    and (bs >= 32 if arm != "baseline" else bs == 64)):
                break
            log(f"{args.benchmark}_bs{bs}_{arm}: OOM detected, retrying at "
                f"MF {retry_mf:g}")
            record = run_cell(args, prompts, bs, arm, retry_mf,
                              attempt=attempt)
            with open(results_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            log(f"=== CELL {args.benchmark}_bs{bs}_{arm} retry{attempt} done: "
                f"status={record['status']} al={record.get('mean_al')} "
                f"tok_s={record.get('agg_tok_s')} ===")

    log(f"DONE {args.benchmark}")


if __name__ == "__main__":
    main()
