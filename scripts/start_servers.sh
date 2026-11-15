#!/usr/bin/env bash
# Start the vLLM servers for configs/opensource.yaml. Logs go to logs/vllm-<port>.log
#   GPU0 :8001  companion under test  (default Qwen3-8B; override: COMPANION=unsloth/Llama-3.1-8B-Instruct)
#   GPU1 :8002  child simulator + memory writer  (Gemma-4-31B-it, as in KORA)
#   GPU2 :8003  judge  (Qwen3.6-27B-FP8)
set -euo pipefail
cd "$(dirname "$0")/.."
COMPANION=${COMPANION:-Qwen/Qwen3-8B}
export PATH="$PWD/.conda-vllm/bin:$PATH"
VLLM=.conda-vllm/bin/vllm
mkdir -p logs
start() {  # model port gpu
  CUDA_VISIBLE_DEVICES=$3 nohup $VLLM serve "$1" --port "$2" --max-model-len 16384 \
    --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{"image":0,"video":0,"audio":0}' \
    > "logs/vllm-$2.log" 2>&1 &
  echo "$1 -> :$2 (GPU $3) pid $!"
}
start "$COMPANION" 8001 0
start google/gemma-4-31B-it 8002 1
start Qwen/Qwen3.6-27B-FP8 8003 2
echo "wait until: curl -s localhost:800{1,2,3}/v1/models"
