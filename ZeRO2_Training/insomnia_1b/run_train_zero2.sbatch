#!/bin/bash
#SBATCH --job-name=zero2-train
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-%j.out

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-insomnia}"
PYTHON_BIN="$VENV_DIR/bin/python"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-1B}"
DATASET_NAME="${DATASET_NAME:-HuggingFaceH4/ultrachat_200k}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-10}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-64}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
SAVE_STEPS="${SAVE_STEPS:-50}"
EVAL_STEPS="${EVAL_STEPS:-10}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
RUN_NAME="${RUN_NAME:-zero2-test}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/${RUN_NAME}-${SLURM_JOB_ID}}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
REPORT_TO="${REPORT_TO:-none}"
TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-$RUN_NAME}"

export OMP_NUM_THREADS="$(( ${SLURM_CPUS_PER_TASK:-8} / NPROC_PER_NODE ))"
if [[ "$OMP_NUM_THREADS" -lt 1 ]]; then
  export OMP_NUM_THREADS=1
fi

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN before submitting the job.}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT_DIR/.cache}"
export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$ROOT_DIR/.cache/huggingface/datasets}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$ROOT_DIR/.cache/triton}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$ROOT_DIR/.cache/torch_extensions}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$ROOT_DIR/.cache/wandb}"
export WANDB_DIR="${WANDB_DIR:-$OUTPUT_DIR/wandb}"
export TMPDIR="${TMPDIR:-$ROOT_DIR/.tmp}"
export TRANSFORMERS_NO_TF=1
export USE_TF=0
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
unset TRANSFORMERS_CACHE || true

mkdir -p "$OUTPUT_DIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$WANDB_CACHE_DIR" "$WANDB_DIR" "$TMPDIR"

echo "[job] host=$(hostname)"
echo "[job] root_dir=$ROOT_DIR"
echo "[job] venv_dir=$VENV_DIR"
echo "[job] nproc_per_node=$NPROC_PER_NODE"
echo "[job] output_dir=$OUTPUT_DIR"
echo "[job] report_to=$REPORT_TO"
echo "[job] train_run_name=$TRAIN_RUN_NAME"
echo "[job] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[job] nccl_debug=$NCCL_DEBUG"
echo "[job] nccl_p2p_disable=$NCCL_P2P_DISABLE"
echo "[job] nccl_ib_disable=$NCCL_IB_DISABLE"
echo "[job] starting at $(date)"

if [[ -n "${WANDB_PROJECT:-}" ]]; then
  export WANDB_NAME="${WANDB_NAME:-$TRAIN_RUN_NAME}"
  echo "[job] wandb_project=${WANDB_PROJECT:-<unset>}"
  echo "[job] wandb_entity=${WANDB_ENTITY:-<unset>}"
  echo "[job] wandb_dir=$WANDB_DIR"

  if [[ -z "${WANDB_API_KEY:-}" && ! -f "$HOME/.netrc" ]]; then
    echo "[job] WARNING: WANDB_API_KEY is not set and ~/.netrc was not found."
    echo "[job] WARNING: If you have not already run 'wandb login' on Insomnia, W&B logging may fail."
  fi

  if [[ -z "${WANDB_PROJECT:-}" ]]; then
    echo "[job] WARNING: WANDB_PROJECT is not set. W&B will fall back to its default project name."
  fi
fi

cd "$ROOT_DIR"



nvidia-smi

VISIBLE_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | grep -c . || true)"
echo "[job] visible_gpus=${VISIBLE_GPUS:-0}"
if [[ -z "${VISIBLE_GPUS}" || "${VISIBLE_GPUS}" -lt "${NPROC_PER_NODE}" ]]; then
  echo "[job] ERROR: Requested ${NPROC_PER_NODE} processes but only ${VISIBLE_GPUS:-0} GPUs are visible on this node."
  exit 1
fi

./setup_env.sh "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "[job] which python=$(which python)"
echo "[job] which torchrun=$(which torchrun || true)"

PYTHON_CUDA_INFO="$("$PYTHON_BIN" -c "import sys, torch; avail = torch.cuda.is_available(); count = torch.cuda.device_count(); print('[job] python_exe', sys.executable); print('[job] cuda_available', avail); print('[job] gpu_count', count); print(int(avail)); print(count)")"
echo "$PYTHON_CUDA_INFO" | sed '/^[01]$/d'
PYTHON_CUDA_AVAILABLE="$(echo "$PYTHON_CUDA_INFO" | tail -n 2 | head -n 1)"
PYTHON_GPU_COUNT="$(echo "$PYTHON_CUDA_INFO" | tail -n 1)"
if [[ "${PYTHON_CUDA_AVAILABLE}" -ne 1 ]]; then
  echo "[job] ERROR: torch could not initialize CUDA on this node even though GPUs were allocated."
  exit 1
fi
if [[ "${PYTHON_GPU_COUNT}" -lt "${NPROC_PER_NODE}" ]]; then
  echo "[job] ERROR: torch sees only ${PYTHON_GPU_COUNT} GPUs but ${NPROC_PER_NODE} processes were requested."
  exit 1
fi

if [[ "$RESUME_FROM_CHECKPOINT" == "latest" ]]; then
  LATEST_CHECKPOINT="$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1 || true)"
  if [[ -n "$LATEST_CHECKPOINT" ]]; then
    RESUME_FROM_CHECKPOINT="$LATEST_CHECKPOINT"
  else
    RESUME_FROM_CHECKPOINT=""
  fi
fi

TRAIN_ARGS=(
  train_zero2_llama.py
  --model_name "$MODEL_NAME"
  --dataset_name "$DATASET_NAME"
  --output_dir "$OUTPUT_DIR"
  --deepspeed_config "$ROOT_DIR/ds_zero2.json"
  --max_length "$MAX_LENGTH"
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
  --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --learning_rate "$LEARNING_RATE"
  --num_train_epochs "$NUM_TRAIN_EPOCHS"
  --max_steps "$MAX_STEPS"
  --logging_steps "$LOGGING_STEPS"
  --save_steps "$SAVE_STEPS"
  --eval_steps "$EVAL_STEPS"
  --save_total_limit "$SAVE_TOTAL_LIMIT"
  --report_to "$REPORT_TO"
  --run_name "$TRAIN_RUN_NAME"
)

if [[ -n "${MAX_TRAIN_SAMPLES:-}" ]]; then
  TRAIN_ARGS+=(--max_train_samples "$MAX_TRAIN_SAMPLES")
fi

if [[ -n "${MAX_EVAL_SAMPLES:-}" ]]; then
  TRAIN_ARGS+=(--max_eval_samples "$MAX_EVAL_SAMPLES")
fi

if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  echo "[job] resume_from_checkpoint=$RESUME_FROM_CHECKPOINT"
  TRAIN_ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" "${TRAIN_ARGS[@]}"

echo "[job] finished at $(date)"
