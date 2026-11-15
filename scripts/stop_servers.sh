#!/usr/bin/env bash
pkill -f "vllm serve" && echo stopped || echo "no vllm servers running"
