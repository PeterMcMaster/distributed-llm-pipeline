# DDP Baseline

This folder contains the baseline Distributed Data Parallel (DDP) training pipeline.

## Model
- Qwen2.5-0.5B-Instruct

## Dataset
- Databricks Dolly-15k (subset)

## Environment
- GCP G2 instance (NVIDIA L4 GPU)
- PyTorch 2.6 + CUDA

## Results (Initial Run)
- Train time: ~15.7 min
- Peak GPU memory: ~4.7 GB
- Export time: ~1.5 sec

## Notes
- Output model is exported in Hugging Face format for vLLM compatibility
- Multi-GPU DDP and full-scale experiments are in progress
