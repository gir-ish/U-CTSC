#!/usr/bin/env bash
# Scale-up experiment: 54 personas x {neutral, engagement} x 14 days, then a persistent-strategy sweep.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
for p in 8001 8002 8003; do until curl -sf localhost:$p/v1/models >/dev/null; do sleep 10; done; done
echo "servers ready $(date +%T)"
$PY -m longkid simulate -c configs/scale_neutral.yaml > logs/scale-neutral.log 2>&1 &
$PY -m longkid simulate -c configs/scale_engagement.yaml > logs/scale-engagement.log 2>&1 &
wait
echo "simulations done $(date +%T)"
$PY -m longkid sweep --runs runs/scale-neutral-* runs/scale-engagement-* --out runs/sweep-scale \
  --samples 3 --horizon 6 --concurrency 96 --persist-all > logs/sweep-scale.log 2>&1
echo "sweep done $(date +%T)"
