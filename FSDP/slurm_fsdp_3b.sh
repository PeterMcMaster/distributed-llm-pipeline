#!/bin/bash
#SBATCH --job-name=fsdp_llama32_3b
#SBATCH --account=edu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=36:00:00
#SBATCH --output=logs/fsdp_3b_%j.log
#SBATCH --error=logs/fsdp_3b_%j.err

# Load required modules
module purge
module load cuda/12.3

# Create logs directory if it doesn't exist
mkdir -p logs

# Set up Python virtual environment
cd /scratch/$USER/distributed-llm-pipeline/FSDP

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate

# Install dependencies (if needed)
pip install -q -r ../requirements.txt

# Set HuggingFace token (you must set this before running)
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN not set. Using huggingface-cli login instead."
fi

# Run FSDP training with 4 GPUs
echo "Starting FSDP Llama-3.1-3B training..."
torchrun --nproc_per_node=4 fsdp_train-2.py --config config_llama32_3b.yaml

echo "Training complete!"
