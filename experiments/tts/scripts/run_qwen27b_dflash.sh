#!/bin/bash
# Qwen3.6-27B + DFlash: baseline vs TTS (stride 24) at batch sizes 8-64.
#
#   GPUS=0,1,2,3 PORT=31201 experiments/tts/scripts/run_qwen27b_dflash.sh [benchmark ...]
#
# Cells already recorded are skipped, so the script can be restarted. Do not
# edit this file while it is running.
set -u
TTS_DIR=$(dirname "$(dirname "$(readlink -f "$0")")")
: "${GPUS:=0,1,2,3}"
: "${PORT:=31201}"
: "${PROMPTS:=$TTS_DIR/prompts}"
: "${RUNS:=$TTS_DIR/runs/dflash_qwen3.6-27b}"
BENCHES=${*:-aime2024 aime2025 gpqa_diamond lcb_hard math500_hard theoremqa olympiadbench_math olympiadbench_physics}
# The batch-invariant matmul can use DeepGEMM, which has no kernels for some
# recent GPU architectures; the Triton fallback is batch-invariant as well.
export SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0
mkdir -p "$RUNS"

for BENCH in $BENCHES; do
  python "$TTS_DIR/run_dflash_cells.py" \
    --target Qwen/Qwen3.6-27B --draft z-lab/Qwen3.6-27B-DFlash \
    --prompts "$PROMPTS/${BENCH}__qwen3.6-27b.json" --benchmark "$BENCH" \
    --gpus "$GPUS" --port "$PORT" \
    --batch-sizes 8,16,32,64 --arms baseline,tts_s24 \
    --extra-server-args="--enable-deterministic-inference" \
    --out "$RUNS/${BENCH}.jsonl" \
    >> "$RUNS/${BENCH}.driver.log" 2>&1
done
echo "done: $(date)"
