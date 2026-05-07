#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-3B}"
export RUN_NAME="${RUN_NAME:-zero2-4gpu-full-epoch-a6000-perf-3b-$(date +%Y%m%d_%H%M%S)}"
export MAX_TRAIN_SAMPLES=
export MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-1000}"
export MAX_STEPS="${MAX_STEPS:--1}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export LOGGING_STEPS="${LOGGING_STEPS:-50}"
export SAVE_STEPS="${SAVE_STEPS:-500}"
export EVAL_STEPS="${EVAL_STEPS:-500}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"

"$ROOT_DIR/submit_train_full_epoch_4gpu_a6000_perf_fresh.sh" "${1:-edu}"
