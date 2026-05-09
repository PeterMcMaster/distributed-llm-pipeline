# DDP Baseline Training

This directory contains our PyTorch Distributed Data Parallel (DDP) implementation for multi-GPU fine-tuning of Llama 3.2 models on the UltraChat instruction-tuning dataset.

The implementation was developed and benchmarked on the Insomnia GPU cluster using NVIDIA A6000 GPUs.

---

# Models

## Base Models

### Llama 3.2 1B
https://huggingface.co/meta-llama/Llama-3.2-1B

### Llama 3.2 3B
https://huggingface.co/meta-llama/Llama-3.2-3B

---

# Fine-Tuned Models

### 1B Fine-Tuned Checkpoint
https://huggingface.co/Sudarshan1607/ddp-llama32-1b-ultrachat

### 3B Fine-Tuned Checkpoint
https://huggingface.co/Sudarshan1607/ddp-llama32-3b-ultrachat

---

# WandB Tracking

All runs were tracked using Weights & Biases:

https://wandb.ai/

Example run names:
- `ddp-llama32-1b-ultrachat-2gpu-full-epoch`
- `ddp-llama32-3b-ultrachat-4gpu-a6000-short`

Tracked metrics:
- training loss
- wall-clock runtime
- optimizer step timing
- distributed scaling behavior
- GPU utilization
- throughput

---

# Repository Structure

```text
DDP/
├── README.md
├── train_baseline.py
├── train_baseline_3b.py
├── run_ddp_2gpu_1b_full.slurm
├── run_ddp_2gpu_3b_a6000.slurm
├── run_ddp_4gpu.slurm
└── run_ddp_4gpu_3b.slurm
```

---

# Overview

This implementation uses:

- PyTorch DDP (`torch.nn.parallel.DistributedDataParallel`)
- Hugging Face Transformers
- TRL `SFTTrainer`
- Slurm scheduling
- NCCL backend for GPU communication
- BF16 mixed precision training

Each GPU:
- launches its own process,
- holds a replica of the model,
- processes a different mini-batch,
- synchronizes gradients after backpropagation.

Distributed training is launched using:

```bash
torchrun --nproc_per_node=<NUM_GPUS>
```

---

# Training Configuration

## Common Settings

| Parameter | Value |
|---|---|
| Precision | BF16 |
| Optimizer | AdamW |
| LR Scheduler | Cosine |
| Gradient Accumulation | 8 |
| Per-GPU Batch Size | 1 |
| Backend | NCCL |

---

# Effective Global Batch Size

## 2 GPUs
```text
2 × 1 × 8 = 16
```

## 4 GPUs
```text
4 × 1 × 8 = 32
```

---

# Running Jobs

## Activate Environment

```bash
source llm-env/bin/activate
```

---

# 1B Training

## 2 GPU Full-Epoch Run

```bash
sbatch run_ddp_2gpu_1b_full.slurm
```

## 4 GPU Run

```bash
sbatch run_ddp_4gpu.slurm
```

---

# 3B Training

## 2 GPU Run

```bash
sbatch run_ddp_2gpu_3b_a6000.slurm
```

## 4 GPU Run

```bash
sbatch run_ddp_4gpu_3b.slurm
```

---

# Monitoring Jobs

## View Queue

```bash
squeue -u <USERNAME>
```

## Inspect Job

```bash
scontrol show job <JOB_ID>
```

## Monitor Logs

```bash
tail -f logs/<LOG_FILE>.out
```

---

# Systems Challenges Encountered

## Slurm Scheduling

Large multi-GPU jobs frequently remained:

```text
PENDING (Resources)
```

Some nodes repeatedly failed with:

```text
ReqNodeNotAvail
```

Stable nodes identified during experiments:
- `ins081`
- `ins084`
- `ins089`

---

# Distributed Checkpoint Storage Failures

A major issue occurred during checkpoint export:

```text
Disk quota exceeded
```

This was caused by:
- large Hugging Face checkpoints,
- optimizer state serialization,
- repeated intermediate checkpoint saving,
- distributed safetensor exports.

This became especially problematic for:
- full-epoch runs,
- 3B models,
- 4-GPU DDP training.

---

# Solution

To stabilize long distributed runs:

- intermediate checkpointing was disabled,
- periodic evaluation was disabled,
- only final model weights were saved,
- storage overhead was reduced significantly.

Training configuration:

```python
save_strategy="no"
eval_strategy="no"
```

Final model export is performed once after training completion.

---

# Iterative Training Strategy

Before launching full-epoch distributed runs, smaller experiments were performed to:
- validate DDP synchronization,
- confirm checkpoint correctness,
- benchmark throughput,
- reduce debugging time,
- avoid large failed exports.

Once stable, the step limits were removed and full-epoch runs were launched.

---

# Final Outputs

Training exports Hugging Face-compatible checkpoints:

```text
config.json
generation_config.json
model.safetensors
tokenizer.json
tokenizer_config.json
special_tokens_map.json
```

These can be loaded directly with:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("<MODEL_PATH>")
tokenizer = AutoTokenizer.from_pretrained("<MODEL_PATH>")
```
