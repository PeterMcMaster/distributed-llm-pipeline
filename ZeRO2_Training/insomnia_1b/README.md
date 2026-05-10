# HPML Final Project

This folder contains the minimal DeepSpeed ZeRO-2 training scripts used for the successful 1B and 3B fine-tuning runs.

## Successful Training Commands

### 1B model

Successful full-epoch 1B run on `ins093`:

```bash
NODELIST=ins093 ./submit_train_full_epoch_4gpu_a6000_perf_fresh.sh edu
```

### 3B model

Successful half-epoch 3B run on `ins081` with W&B logging:

```bash
export HF_TOKEN="your_hf_token"
export WANDB_API_KEY="your_wandb_api_key"
export WANDB_PROJECT="hpml-final-project"
export WANDB_ENTITY="robsol-columbia-university"

RUN_STAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="zero2-4gpu-half-epoch-a6000-perf-3b-node81-fresh-$RUN_STAMP"

NODELIST=ins081 \
NUM_TRAIN_EPOCHS=0.5 \
REPORT_TO=wandb \
NCCL_P2P_DISABLE=0 \
SAVE_STEPS=3000 \
SAVE_TOTAL_LIMIT=2 \
EVAL_STEPS=500 \
RUN_NAME="$RUN_NAME" \
TRAIN_RUN_NAME="$RUN_NAME" \
./submit_train_full_epoch_4gpu_a6000_perf_3b_fresh.sh edu
```

## Hugging Face Models

- 1B: <https://huggingface.co/robsol/hpml-final-1b-zero2-vllm-export>
- 3B: <https://huggingface.co/robsol/hpml-final-3b-zero2-vllm-export>

## W&B Report

- <https://api.wandb.ai/links/robsol-columbia-university/h9wwgktg>
