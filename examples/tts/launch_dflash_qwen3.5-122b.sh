#!/bin/bash
# Qwen3.5-122B-A10B with its DFlash speculator and TTS, on 8 GPUs.
#
#   bash examples/tts/launch_dflash_qwen3.5-122b.sh            # TTS, per-request LoRA (batched)
#   TTS=full bash examples/tts/launch_dflash_qwen3.5-122b.sh   # TTS, full-weight (one request)
#   TTS=0 bash examples/tts/launch_dflash_qwen3.5-122b.sh      # frozen speculator
#
# This target is a hybrid (attention + state-space) MoE model whose per-request
# recurrent state is reserved for every drafted position, so it needs much more
# memory per concurrent request than a pure-attention model. We recommend 8
# GPUs for it. BS is the number of concurrent requests; with per-request LoRA
# it is also the number of adapter slots. Full-weight TTS serves one request at
# a time and ignores BS. PORT is the HTTP port.
set -eu
# Deterministic inference uses a batch-invariant matmul, which can call DeepGEMM.
# DeepGEMM has no kernels for some recent GPU architectures (for example SM120);
# the Triton fallback is batch-invariant as well.
export SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0
: "${BS:=16}" "${PORT:=30000}" "${TTS:=1}"
MEM=0.86

TTS_ARGS=()
if [ "$TTS" = "1" ]; then
  TTS_ARGS=(
    --speculative-dflash-lora-tts-enable
    --speculative-dflash-lora-tts-rank 16
    --speculative-dflash-lora-tts-lr 1e-4
    --speculative-dflash-lora-tts-freq 10          # update stride S
    --speculative-dflash-lora-tts-delayed-publish  # adapters trained in round N are
                                                   # used from round N+2, so the next
                                                   # round does not wait for the update
  )
elif [ "$TTS" = "full" ]; then
  BS=1
  MEM=0.75   # the optimizer state of the speculator lives outside the KV cache pool
  TTS_ARGS=(
    --speculative-dflash-tts-enable
    --speculative-dflash-tts-lr 1e-5
    --speculative-dflash-tts-freq 10               # update stride S
  )
fi

exec python -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-122B-A10B --tp 8 --port "$PORT" \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/Qwen3.5-122B-A10B-DFlash \
  --max-running-requests "$BS" --mem-fraction-static "$MEM" \
  --mamba-ssm-dtype bfloat16 \
  --context-length 40960 --disable-radix-cache --trust-remote-code \
  --enable-deterministic-inference \
  "${TTS_ARGS[@]}" "$@"
