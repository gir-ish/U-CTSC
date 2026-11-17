#!/usr/bin/env bash
# Experiment 1b: double the samples per decision point (3 -> 6) in both persistent sweeps.
# Finished branches are reused (resume), so only samples 3..5 are simulated.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
for p in 8001 8002 8003; do until curl -sf localhost:$p/v1/models >/dev/null; do sleep 10; done; done
echo "start $(date +%T)"
$PY -m longkid sweep --runs runs/full-* --out runs/sweep-persist \
  --samples 6 --horizon 6 --concurrency 32 --persist-all > logs/sweep-persist-s6.log 2>&1 &
$PY -m longkid sweep --runs runs/scale-neutral-* runs/scale-engagement-* --out runs/sweep-scale \
  --samples 6 --horizon 6 --concurrency 96 --persist-all > logs/sweep-scale-s6.log 2>&1 &
wait
echo "sweeps done $(date +%T)"
