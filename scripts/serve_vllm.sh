#!/usr/bin/env bash
# Usage: scripts/serve_vllm.sh <hf-model-id> <port> <gpu-ids> [extra vllm args]
#   scripts/serve_vllm.sh Qwen/Qwen3-8B  8001 0
#   scripts/serve_vllm.sh Qwen/Qwen3-32B 8002 1,2
# Requires: pip install vllm  (in a separate env is fine). Tensor parallel = number of GPUs given.
set -euo pipefail
export PATH="$(cd "$(dirname "$0")/.." && pwd)/.conda-vllm/bin:$PATH"
MODEL=$1; PORT=$2; GPUS=$3; shift 3
TP=$(awk -F, '{print NF}' <<<"$GPUS")
CUDA_VISIBLE_DEVICES=$GPUS exec "$(dirname "$0")/../.conda-vllm/bin/vllm" serve "$MODEL" \
  --port "$PORT" --tensor-parallel-size "$TP" \
  --max-model-len 16384 --gpu-memory-utilization 0.90 "$@"
