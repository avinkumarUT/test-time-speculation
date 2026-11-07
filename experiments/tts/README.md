# Reproducing the batched-serving results

This directory holds everything behind the paper's batched-serving tables:
the benchmark harness, the raw per-cell results, and the script that turns
those results into the tables.

| Suite | Target model | Speculator | TTS stride |
|---|---|---|---|
| `dflash_qwen3.6-27b` | `Qwen/Qwen3.6-27B` | `z-lab/Qwen3.6-27B-DFlash` | 24 |
| `dflash_qwen3.5-122b` | `Qwen/Qwen3.5-122B-A10B` | `z-lab/Qwen3.5-122B-A10B-DFlash` | 10 |

Each suite is 8 benchmarks × 4 batch sizes (8, 16, 32, 64) × 2 arms (the
frozen speculator, and the speculator with per-request LoRA TTS): 64 cells
per suite, 128 in total.

## Rebuild the tables from the shipped results

No GPU needed:

```bash
python experiments/tts/make_tables.py                    # throughput, Markdown
python experiments/tts/make_tables.py --metric al        # acceptance length
python experiments/tts/make_tables.py --format latex --suite dflash_qwen3.6-27b
```

`results/<suite>/<benchmark>.jsonl` has one JSON record per cell with the
aggregate numbers (`agg_tok_s`, `mean_al`, wall time, token counts, memory
fraction) and a `per_request` list (completion tokens, verify rounds,
acceptance length and finish reason for every request). No prompts or model
outputs are stored.

## Protocol

- **Serving:** tensor parallelism over 4 GPUs, deterministic inference
  (`--enable-deterministic-inference`) for both arms.
- **Load:** a cell runs N = max(2 × batch size, 32) requests (32, 32, 64, 128)
  with `batch size` requests in flight. Prompts are cycled when a benchmark
  has fewer than N problems.
- **Decoding:** temperature 0, up to 32,768 new tokens, natural stopping.
  Thinking / high reasoning effort is enabled in the chat template.
- **Throughput** is completed tokens divided by the wall time of the load.
  **Acceptance length** is completion tokens per verify round, averaged over
  requests.
- **Loss:** `kl` with lambda 0.9. The server's default is `lk`, so the harness
  passes `--speculative-tts-loss kl` (`run_dflash_cells.py --tts-loss`).
- **TTS:** rank-16 LoRA adapters (alpha 32) on the DFlash speculator's
  attention query/output and MLP projections, learning rate 1e-4, one adapter
  slot per in-flight request, 16-token block. Updates run asynchronously on a
  side CUDA stream every `S` rounds with delayed adapter publication; `S = 24`
  for the 27B target and `S = 10` for the 122B target.

## Run it yourself

The paper runs used a node with 8× NVIDIA RTX PRO 6000 Blackwell (96 GB), so
two 4-GPU lanes can run side by side. On that hardware one benchmark (all
eight cells) takes 3 to 10 hours on a 4-GPU lane, and a full suite takes
about 40 hours (27B) or 52 hours (122B), not counting model download and the
first, cold model load.

```bash
# 1. Install this repository (from the repo root).
pip install -e "python[all]"

# 2. One-time: enlarge FlashInfer's plan workspace (needed for bs=64 DFlash).
python experiments/tts/scripts/patch_flashinfer.py

# 3. Render and freeze the prompts (GPQA is gated: request access on the Hub
#    and `huggingface-cli login` first). Writes experiments/tts/prompts/.
python experiments/tts/render_prompts.py

# 4. Run a suite. Each script takes an optional list of benchmarks and can be
#    restarted; finished cells are skipped. Output goes to experiments/tts/runs/.
GPUS=0,1,2,3 PORT=31201 experiments/tts/scripts/run_qwen27b_dflash.sh aime2024
GPUS=4,5,6,7 PORT=31202 experiments/tts/scripts/run_qwen122b_dflash.sh aime2024

# 5. Tabulate your runs.
python experiments/tts/make_tables.py --results experiments/tts/runs
```

Benchmark keys: `aime2024`, `aime2025`, `gpqa_diamond`, `lcb_hard`,
`math500_hard`, `theoremqa`, `olympiadbench_math`, `olympiadbench_physics`.
The problem-set filters (for example MATH-500 level 5 only, LiveCodeBench hard
only, text-only OlympiadBench) are in `benchmarks.py`.

### Files

| File | Purpose |
|---|---|
| `benchmarks.py` | Dataset loaders and prompt wording for the eight benchmarks. |
| `render_prompts.py` | Renders prompts once per target model and freezes them to JSON. |
| `run_dflash_cells.py` | A grid of DFlash cells (one server per cell, HTTP load). |
| `scripts/run_*.sh` | The exact grids and settings used for each suite. |
| `scripts/patch_flashinfer.py` | The FlashInfer workspace fix described above. |
| `make_tables.py` | Builds the paper tables from result records. |
| `results/` | The raw results behind the paper tables. |

## Notes on the shipped results

- **Wall-capped cells.** Two baseline cells (Qwen3.5-122B at batch size 64 on
  LiveCodeBench and OlympiadBench-Math) hit the 2.5-hour per-cell cap before
  all 128 requests finished. They are recorded with `"status": "partial"`;
  throughput and acceptance length are computed over the requests that
  completed. `make_tables.py` marks them with `*`.
- **Memory fraction.** `--mem-fraction-static` is 0.86 by default. Cells that
  ran out of memory at startup were retried at a lower fraction, which only
  shrinks the KV-cache pool; the value each cell ran with is in its record
  (`mem_fraction`).
- **Batch-size-64 settings on the 122B target.** The TTS cells at batch size
  64 additionally use `--chunked-prefill-size 4096`, a 2 GB FlashInfer
  workspace and one request per update replay chunk
  (`scripts/run_qwen122b_dflash.sh`). The baseline cells at that batch size
  ran with the default settings. Prompts here are shorter than 4,096 tokens,
  so the chunk size does not split any single prompt.
- **Outputs are not bit-reproducible.** Because of floating-point
  non-determinism the same prompt can produce different text across runs and
  across the two arms (see the note in the top-level README), so the tables
  compare aggregates over the same prompt set.
- **Rounding.** Tables are built from the recorded two-decimal throughput,
  rounded half up to one decimal.
