#!/usr/bin/env bash
# Live demo: Pip vs Pip + U-CTSC side by side (needs the vLLM servers). Open http://localhost:7860
cd "$(dirname "$0")/.."
CUDA_VISIBLE_DEVICES="" exec .venv/bin/python -m demo.app "$@"
