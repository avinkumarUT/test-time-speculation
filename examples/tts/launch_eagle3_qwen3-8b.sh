#!/bin/bash
# Qwen3-8B with its EAGLE-3 speculator and full-weight TTS, on 1 GPU.
#
#   bash examples/tts/launch_eagle3_qwen3-8b.sh          # TTS, full-weight (one request)
#   TTS=0 bash examples/tts/launch_eagle3_qwen3-8b.sh    # frozen speculator
#
# Full-weight TTS updates the speculator's own weights for the request being
# generated and restores them when the next request starts, so the server
# serves one request at a time. PORT is the HTTP port.
set -eu
: "${PORT:=30000}" "${TTS:=1}"

TTS_ARGS=()
if [ "$TTS" = "1" ]; then
  TTS_ARGS=(
    --speculative-eagle-tts-enable
    --speculative-eagle-tts-mode full
    --speculative-eagle-tts-lr 1e-5
    --speculative-eagle-tts-freq 10            # update stride S
  )
fi

exec python -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B --port "$PORT" \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path AngelSlim/Qwen3-8B_eagle3 \
  --speculative-num-steps 9 --speculative-eagle-topk 1 --speculative-num-draft-tokens 10 \
  --max-running-requests 1 --mem-fraction-static 0.6 \
  --disable-radix-cache --trust-remote-code \
  "${TTS_ARGS[@]}" "$@"
