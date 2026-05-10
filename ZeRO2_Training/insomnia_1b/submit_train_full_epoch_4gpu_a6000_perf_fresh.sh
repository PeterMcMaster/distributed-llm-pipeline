#!/bin/bash
set -euo pipefail

ACCOUNT="${ACCOUNT:-${1:-edu}}"
PARTITION="${PARTITION:-short}"
QOS="${QOS:-hpc_test}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
MEMORY="${MEMORY:-128G}"
CPUS="${CPUS:-16}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-zero2-4gpu-full-epoch-a6000-perf-$RUN_STAMP}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/$RUN_NAME}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:--1}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-1000}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
LOGGING_STEPS="${LOGGING_STEPS:-50}"
SAVE_STEPS="${SAVE_STEPS:-500}"
EVAL_STEPS="${EVAL_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
REPORT_TO="${REPORT_TO:-none}"
TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-$RUN_NAME}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_NAME="${WANDB_NAME:-$TRAIN_RUN_NAME}"
WANDB_DIR="${WANDB_DIR:-$OUTPUT_DIR/wandb}"
WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$ROOT_DIR/.cache/wandb}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"
NODELIST="${NODELIST:-}"

export WANDB_PROJECT WANDB_ENTITY WANDB_NAME WANDB_DIR WANDB_CACHE_DIR

if [[ -d "$OUTPUT_DIR" ]] && find "$OUTPUT_DIR" -mindepth 1 -print -quit | grep -q .; then
  echo "[submit] ERROR: output_dir already contains files: $OUTPUT_DIR"
  echo "[submit] Refusing to run because this wrapper is for fresh starts only."
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

#NCCL settings for a single-node 4x A6000 run

NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

SBATCH_ARGS=(
  -A "$ACCOUNT"
  --partition="$PARTITION"
  --constraint=A6000
  --gres=gpu:A6000:4
  --time="$TIME_LIMIT"
  --mem="$MEMORY"
  --cpus-per-task="$CPUS"
  --export=ALL,NPROC_PER_NODE=4,RUN_NAME="$RUN_NAME",TRAIN_RUN_NAME="$TRAIN_RUN_NAME",REPORT_TO="$REPORT_TO",OUTPUT_DIR="$OUTPUT_DIR",NUM_TRAIN_EPOCHS="$NUM_TRAIN_EPOCHS",MAX_STEPS="$MAX_STEPS",MAX_TRAIN_SAMPLES="$MAX_TRAIN_SAMPLES",MAX_EVAL_SAMPLES="$MAX_EVAL_SAMPLES",GRADIENT_ACCUMULATION_STEPS="$GRADIENT_ACCUMULATION_STEPS",LOGGING_STEPS="$LOGGING_STEPS",SAVE_STEPS="$SAVE_STEPS",EVAL_STEPS="$EVAL_STEPS",SAVE_TOTAL_LIMIT="$SAVE_TOTAL_LIMIT",RESUME_FROM_CHECKPOINT=,NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE}",NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL}",NCCL_IB_DISABLE="${NCCL_IB_DISABLE}",TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT}",NCCL_DEBUG="${NCCL_DEBUG}"
)

if [[ -n "$QOS" ]]; then
  SBATCH_ARGS+=(--qos="$QOS")
fi

if [[ -n "$EXCLUDE_NODES" ]]; then
  SBATCH_ARGS+=(--exclude="$EXCLUDE_NODES")
fi

if [[ -n "$NODELIST" ]]; then
  SBATCH_ARGS+=(--nodelist="$NODELIST")
fi

echo "[submit] account=$ACCOUNT"
echo "[submit] partition=$PARTITION"
echo "[submit] qos=${QOS:-<none>}"
echo "[submit] constraint=A6000"
echo "[submit] gres=gpu:A6000:4"
echo "[submit] run_name=$RUN_NAME"
echo "[submit] train_run_name=$TRAIN_RUN_NAME"
echo "[submit] output_dir=$OUTPUT_DIR"
echo "[submit] resume_from_checkpoint=<disabled>"
echo "[submit] report_to=$REPORT_TO"
echo "[submit] nccl_p2p_disable=$NCCL_P2P_DISABLE"
echo "[submit] nccl_p2p_level=$NCCL_P2P_LEVEL"
echo "[submit] nccl_ib_disable=$NCCL_IB_DISABLE"
echo "[submit] torch_nccl_blocking_wait=$TORCH_NCCL_BLOCKING_WAIT"
echo "[submit] exclude_nodes=${EXCLUDE_NODES:-<none>}"
echo "[submit] nodelist=${NODELIST:-<none>}"

if [[ -n "$WANDB_PROJECT" ]]; then
  echo "[submit] wandb_project=${WANDB_PROJECT:-<unset>}"
  echo "[submit] wandb_entity=${WANDB_ENTITY:-<unset>}"
  echo "[submit] wandb_dir=$WANDB_DIR"
fi

sbatch "${SBATCH_ARGS[@]}" run_train_zero2.sbatch
