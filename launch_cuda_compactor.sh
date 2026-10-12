#!/bin/bash
# Dedicated NVIDIA RTX 4060 Context Synthesis & Compaction Server (Port 9001)
# Override with MYAGY_MODEL_DIR and LLAMA_SERVER_BIN.
MODEL_DIR="${MYAGY_MODEL_DIR:-/home/safiyu/models}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-/home/safiyu/llama.cpp-cuda/build/bin/llama-server}"
export CUDA_VISIBLE_DEVICES=0
exec "$LLAMA_SERVER_BIN" \
  --model "$MODEL_DIR/granite-4.2-8b-Q4_K_M.gguf" \
  --port 9001 \
  --host 127.0.0.1 \
  --ctx-size 16384 \
  -ngl 99 \
  --flash-attn on \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --reasoning off \
  --no-warmup \
  --threads 8 \
  --parallel 1 "$@"
