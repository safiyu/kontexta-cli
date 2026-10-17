#!/usr/bin/env bash
# Starts the CUDA compactor llama-server. Override any value via the environment (KXTC_* names match config.py).
HOST="${KXTC_HOST:-127.0.0.1}"
PORT="${KXTC_CUDA_PORT:-9002}"
MODEL_DIR="${KXTC_MODEL_DIR:-${HOME}/models}"
MODEL_FILE="${KXTC_COMPACTOR_FILE:-granite-4.2-8b-instruct-Q4_K_M.gguf}"
CTX_SIZE="${KXTC_COMPACTOR_CTX:-16384}"
GPU_LAYERS="${KXTC_COMPACTOR_GPU_LAYERS:-99}"
KV_DIR="${KXTC_KV_CACHE_DIR:-/dev/shm/kontexta_cli_kv_cache}"

mkdir -p "$KV_DIR"

exec "${LLAMA_SERVER_BIN:-${HOME}/llama.cpp-cuda/build/bin/llama-server}" \
  --model "${MODEL_DIR}/${MODEL_FILE}" \
  --port "$PORT" \
  --host "$HOST" \
  --n-gpu-layers "$GPU_LAYERS" \
  --parallel 1 \
  --ctx-size "$CTX_SIZE" \
  --cache-reuse 256 \
  --slot-save-path "$KV_DIR" \
  --metrics "$@"
