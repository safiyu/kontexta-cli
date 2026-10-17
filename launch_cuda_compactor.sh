#!/usr/bin/env bash
# Projects/myagy/launch_cuda_compactor.sh
mkdir -p /dev/shm/myagy_kv_cache

exec "${LLAMA_SERVER_BIN:-${HOME}/llama.cpp-cuda/build/bin/llama-server}" \
  --model "${MYAGY_MODEL_DIR:-${HOME}/models}/granite-4.2-8b-instruct-Q4_K_M.gguf" \
  --port 9002 \
  --host 127.0.0.1 \
  --n-gpu-layers 99 \
  --parallel 1 \
  --ctx-size 16384 \
  --cache-reuse 256 \
  --slot-save-path /dev/shm/myagy_kv_cache \
  --metrics "$@"
