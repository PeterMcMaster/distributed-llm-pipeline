#!/bin/bash
#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --job-name=ddp-3b-2gpu
#SBATCH --nodes=1
#SBATCH --nodelist=ins084
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:A6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ddp_3b_2gpu_a6000_%j.out
#SBATCH --error=logs/ddp_3b_2gpu_a6000_%j.err

set -euxo pipefail

mkdir -p logs
source llm-env/bin/activate

export HF_HOME=$PWD/.hf_cache
export HF_DATASETS_CACHE=$PWD/.hf_cache/datasets
export TRANSFORMERS_CACHE=$PWD/.hf_cache
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

export WANDB_PROJECT=distributed-llm-pipeline
export WANDB_RUN_NAME=ddp-llama32-3b-ultrachat-2gpu-a6000-short

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=7200
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

echo "Job started on $(hostname) at $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
which python
which torchrun
nvidia-smi

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=2 \
  --rdzv-conf timeout=7200 \
  train_baseline_3b.py
