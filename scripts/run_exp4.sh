#!/usr/bin/env bash
# After the 6-sample sweeps: retrain the controller on all branches (v2 features), then run
# experiment 4 (online head-to-head on 54 new children, engagement prompt, seed 21).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CUDA_VISIBLE_DEVICES=""   # embeddings/controller on CPU; GPUs belong to the vLLM servers
while pgrep -f run_more_samples.sh >/dev/null; do sleep 30; done
echo "sweeps finished $(date +%T)"
$PY -m longkid rq3 --sweeps runs/sweep-persist runs/sweep-scale --runs runs/full-* runs/scale-* \
  --out results/rq3_v2.json > logs/rq3_v2.log 2>&1
echo "controller trained $(date +%T)"
$PY -m longkid online -c configs/scale_engagement.yaml --seed 21 --out runs/online-exp4 \
  --controller results/rq3_v2.pkl --concurrency 108 > logs/online-exp4.log 2>&1
echo "experiment 4 done $(date +%T)"
