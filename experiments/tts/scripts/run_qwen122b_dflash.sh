#!/bin/bash
# Qwen3.5-122B-A10B + DFlash: baseline vs TTS (stride 10) at batch sizes 8-64.
#
#   GPUS=0,1,2,3 PORT=31201 experiments/tts/scripts/run_qwen122b_dflash.sh [benchmark ...]
#
# This target is a hybrid (attention + state-space) MoE model. Two settings
# are needed to serve it with a 16-token DFlash block at high concurrency:
#   --mamba-ssm-dtype bfloat16   the per-request recurrent state is reserved
#                                for every draft position; in fp32 it does not
#                                fit at bs=64. Used for all cells.
#   bs=64 TTS cells only         --chunked-prefill-size 4096, a 2 GB FlashInfer
#                                workspace, memory fraction 0.85, and one
#                                request per update replay chunk.
# Cells already recorded are skipped, so the script can be restarted. Do not
# edit this file while it is running.
set -u
TTS_DIR=$(dirname "$(dirname "$(readlink -f "$0")")")
: "${GPUS:=0,1,2,3}"
: "${PORT:=31201}"
: "${PROMPTS:=$TTS_DIR/prompts}"
: "${RUNS:=$TTS_DIR/runs/dflash_qwen3.5-122b}"
BENCHES=${*:-aime2024 aime2025 gpqa_diamond lcb_hard math500_hard theoremqa olympiadbench_math olympiadbench_physics}
# The batch-invariant matmul can use DeepGEMM, which has no kernels for some
# recent GPU architectures; the Triton fallback is batch-invariant as well.
export SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0
mkdir -p "$RUNS"

run() {  # run <benchmark> <extra server args> [run_dflash_cells.py args ...]
  local BENCH=$1 EXTRA=$2; shift 2
  python "$TTS_DIR/run_dflash_cells.py" \
    --target Qwen/Qwen3.5-122B-A10B --draft z-lab/Qwen3.5-122B-A10B-DFlash \
    --prompts "$PROMPTS/${BENCH}__qwen3.5-122b.json" --benchmark "$BENCH" \
    --gpus "$GPUS" --port "$PORT" \
    --extra-server-args="--mamba-ssm-dtype bfloat16 --enable-deterministic-inference $EXTRA" \
    --out "$RUNS/${BENCH}.jsonl" "$@" \
    >> "$RUNS/${BENCH}.driver.log" 2>&1
}

for BENCH in $BENCHES; do
  # Everything except the bs=64 TTS cell.
  run "$BENCH" "" --arms baseline,tts_s10 --mf 0.86 --retry-mf 0.85 \
      --only-cells "8:baseline,8:tts_s10,16:baseline,16:tts_s10,32:baseline,32:tts_s10,64:baseline"
  # The bs=64 TTS cell.
  SGLANG_FLASHINFER_WORKSPACE_SIZE=$((2048 * 1024 * 1024)) \
  run "$BENCH" "--chunked-prefill-size 4096" --arms tts_s10 --batch-sizes 64 \
      --mf 0.85 --retry-mf 0.83 --bs64-replay-chunk 1
done
echo "done: $(date)"
