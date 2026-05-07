#!/bin/bash
#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --job-name=ddp-1b-2gpu-full
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:A6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --nodelist=ins081
#SBATCH --output=logs/ddp_1b_2gpu_full_%j.out
#SBATCH --error=logs/ddp_1b_2gpu_full_%j.err

set -e
mkdir -p logs

source llm-env/bin/activate

export HF_HOME=$PWD/.hf_cache
export HF_DATASETS_CACHE=$PWD/.hf_cache/datasets
export TRANSFORMERS_CACHE=$PWD/.hf_cache

export HF_TOKEN=$(cat ~/.cache/huggingface/token)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

export WANDB_PROJECT=distributed-llm-pipeline
export WANDB_RUN_NAME=ddp-llama32-1b-ultrachat-2gpu-full-epoch-a6000-ins081

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

torchrun --standalone --nnodes=1 --nproc_per_node=2 train_baseline.py
