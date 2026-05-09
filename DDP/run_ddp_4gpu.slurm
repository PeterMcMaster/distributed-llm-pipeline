#!/bin/bash
#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --job-name=ddp-1b-4gpu
#SBATCH --nodes=1
#SBATCH --nodelist=ins084
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A6000:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ddp_1b_4gpu_a6000_%j.out
#SBATCH --error=logs/ddp_1b_4gpu_a6000_%j.err

set -e
mkdir -p logs

source llm-env/bin/activate

export HF_HOME=$PWD/.hf_cache
export HF_DATASETS_CACHE=$PWD/.hf_cache/datasets
export TRANSFORMERS_CACHE=$PWD/.hf_cache

export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

export WANDB_PROJECT=distributed-llm-pipeline
export WANDB_RUN_NAME=ddp-llama32-1b-ultrachat-4gpu-a6000-short

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=4 \
  train_baseline.py
