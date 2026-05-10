# HPML Final Project: Distributed LLM Pipeline

> **Course:** High Performance Machine Learning
> **Semester:** Spring 2026
> **Instructor:** Dr. Kaoutar El Maghraoui

---

## Team Information

- **Team Name:** Group 25
- **Members:**
  - Peter McMaster (pjm2188) - vLLM deployment and serving pipeline
  - Sudarshan Kadalazhi (sk5573) - DDP fine-tuning 
  - Robert Solomon (rs4759) - DeepSpeed ZeRO-2 fine-tuning
  - Damien Kim (ddk2145) - FSDP fine-tuning

## Submission

- **GitHub repository:** [https://github.com/PeterMcMaster/distributed-llm-pipeline](https://github.com/PeterMcMaster/distributed-llm-pipeline)
- **Final report:**
    [`deliverables/HPML_Final_Report.pdf`](deliverables/HPML_Final_Report.pdf)
- **Final presentation:** [`deliverables/HPML Final Project Presentation.pdf`](deliverables/HPML%20Final%20Project%20Presentation.pdf)
- **Experiment-tracking dashboards:**
  - [DDP WandB](https://wandb.ai/sk5573-columbia-university/distributed-llm-pipeline?nw=nwusersk5573)
  - [ZeRO-2 WandB](https://wandb.ai/robsol-columbia-university/hpml-final-project/reports/ZeRO2-finetuning--VmlldzoxNjgwNjg5Ng?accessToken=0ij9e84cz4jae78scbpqs0e3e4ej5l7qe9gbsizi2ay6t97tvb777ymvnwobm8a8)
  - [FSDP WandB](https://wandb.ai/ddk2145-columbia-university/fsdp-llama/workspace?nw=nwuserddk2145)

---

## 1. Problem Statement

This project explores the end-to-end systems path for fine-tuning and serving LLMs, comparing PyTorch DDP, DeepSpeed ZeRO-2, and PyTorch FSDP. The experiment targets both training and inference: distributed fine-tuning must fit model state, optimizer state, gradients, and activations on limited GPU memory, and the exported checkpoint must still be easy to serve with vLLM. So the question is how training-time choices affect not only throughput and memory, but also checkpoint conversion cost, ease of deployment, server load time, and request latency.

## 2. Model/Application Description

- **Model architecture:** Llama 3.2 1B and 3B language models. The early DDP smoke-test pipeline also includes Qwen2.5-0.5B-Instruct.
- **Training strategies:** PyTorch Distributed Data Parallel (DDP), DeepSpeed ZeRO-2, and PyTorch Fully Sharded Data Parallel (FSDP).
- **Serving stack:** vLLM OpenAI-compatible server with Hugging Face-format checkpoints, Llama 3 chat template, FlashAttention, CUDA graphs, and fixed prompt workloads.
- **Frameworks:** PyTorch, Hugging Face Transformers, Hugging Face Datasets, TRL SFTTrainer, DeepSpeed, FSDP, vLLM, safetensors.
- **Datasets:** HuggingFaceH4/ultrachat_200k for the Llama fine-tuning pipelines; Databricks Dolly-15k subset for the initial Qwen DDP baseline.
- **Custom modifications:** training wrappers record throughput and memory; FSDP and ZeRO-2 include export utilities to convert training checkpoints into vLLM-compatible Hugging Face artifacts; `vllm_deployment_eval/` standardizes serving measurements across strategies.
- **Hardware target:** NVIDIA L4 GPU for the recorded vLLM serving runs.

## 3. Final Results Summary

The table below summarizes the measured serving results checked into `vllm_deployment_eval/results/summary_metrics.csv`. Each serving run used vLLM 0.20.1 on a single NVIDIA L4 with CUDA 13.0, `max_model_len=4096`, and `gpu_memory_utilization=0.85`.

| Metric | Baseline | Optimized / Best Compared Run | Improvement |
| --- | --- | --- | --- |
| 1B average request latency | DDP: 1.6272 s | FSDP: 1.2349 s | 24.1% lower latency |
| 3B average request latency | DDP: 3.9594 s | ZeRO-2: 3.6883 s | 6.8% lower latency |
| 3B completion throughput | DDP: 32.5599 tok/s | ZeRO-2: 34.4365 tok/s | 5.8% higher throughput |
| 3B server load time | DDP: 146.0886 s | FSDP: 124.0405 s | 15.1% faster load |
| 1B first-token latency | DDP: 0.0209 s | ZeRO-2: 0.0205 s | 1.9% lower latency |
| 1B peak serving GPU memory | DDP: 19,482 MB | FSDP / ZeRO-2: 19,482 MB | No measured change |
| DDP smoke-test training time | Qwen2.5-0.5B on Dolly-15k: 939.97 s | N/A | Reference run only |
| DDP smoke-test peak training memory | Qwen2.5-0.5B on Dolly-15k: 4.67 GB | N/A | Reference run only |

**Hardware:** NVIDIA L4 24 GB, driver 580.126.20, CUDA 13.0, Linux 6.1, Python 3.11.2, vLLM 0.20.1 for the committed serving results.

**Headline result:** FSDP produced the best 1B serving latency, reducing average request latency by 24.1% versus the DDP 1B checkpoint, while ZeRO-2 produced the best 3B latency and throughput among the committed 3B runs. While the training strategy seemed to have a slight impact on inference, it was found that DDP was the easiest to checkpoint and export, while FSDP proved to be more complex and difficult.

## 4. Repository Structure

```text
.
|-- README.md
|-- LICENSE
|-- HPML_README_Template-1.md
|-- requirements.txt
|-- deliverables/
|   `-- HPML Final Project Presentation.pdf
|-- DDP/
|   |-- README.md
|   |-- train_baseline.py
|   `-- metrics.json
|-- FSDP/
|   |-- fsdp_train.py
|   |-- fsdp_train-2.py
|   |-- fsdp_export.py
|   |-- sample_config.yaml
|   `-- sample_commands.txt
|-- ZeRO2_Training/
|   |-- train_zero2_llama.py
|   |-- export_for_vllm.py
|   |-- ds_zero2.json
|   `-- notebook/deepspeed_zero2.ipynb
`-- vllm_deployment_eval/
    |-- run_experiment.py
    |-- format_results.py
    |-- configs/
    |   |-- experiments.json
    |   `-- prompts.json
    |-- templates/
    |   `-- llama3-chat-template.jinja
    `-- results/
        |-- summary_metrics.csv
        |-- experiments.jsonl
        |-- DDP/
        |-- FSDP/
        `-- ZeRO2/
```

## 5. Reproducibility Instructions

### A. Environment Setup

```bash
git clone https://github.com/PeterMcMaster/distributed-llm-pipeline.git
cd distributed-llm-pipeline

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the vLLM serving benchmark, install a CUDA-compatible vLLM build. The original setup used a vLLM wheel and Hugging Face CLI authentication:

```bash
pip install openai "huggingface_hub[cli]"
hf auth login
```

Set `HF_TOKEN` before running Llama training or exporting gated Hugging Face models:

```bash
export HF_TOKEN=<your_huggingface_token>
```

**System requirements:** Python 3.10+, CUDA-capable NVIDIA GPU, and enough GPU memory for the selected model. The committed serving runs used an NVIDIA L4 with roughly 23 GB visible memory.

### B. Experiment Tracking Dashboard

Training dashboards are split by strategy:

- DDP: [WandB dashboard](https://wandb.ai/sk5573-columbia-university/distributed-llm-pipeline?nw=nwusersk5573)
- ZeRO-2: [WandB report](https://wandb.ai/robsol-columbia-university/hpml-final-project/reports/ZeRO2-finetuning--VmlldzoxNjgwNjg5Ng?accessToken=0ij9e84cz4jae78scbpqs0e3e4ej5l7qe9gbsizi2ay6t97tvb777ymvnwobm8a8)
- FSDP: [WandB workspace](https://wandb.ai/ddk2145-columbia-university/fsdp-llama/workspace?nw=nwuserddk2145)

The serving-side artifacts are committed under `vllm_deployment_eval/results/`.

### C. Dataset

The datasets are pulled through Hugging Face Datasets at runtime and are not committed to the repository.

- Main Llama fine-tuning: `HuggingFaceH4/ultrachat_200k`
- DDP smoke-test baseline: `databricks/databricks-dolly-15k`

### D. Training

Run the DDP smoke-test baseline:

```bash
cd DDP
python3 train_baseline.py
```

Run ZeRO-2 fine-tuning:

```bash
cd ZeRO2_Training
deepspeed train_zero2_llama.py \
  --model_name meta-llama/Llama-3.2-1B \
  --dataset_name HuggingFaceH4/ultrachat_200k \
  --output_dir ./outputs/llama32_1b_zero2
```

Run FSDP fine-tuning:

```bash
cd FSDP
torchrun --nproc_per_node=4 fsdp_train-2.py --config sample_config.yaml
```

Adjust model size, sample limits, and checkpoint cadence in the relevant script/config before launching full runs.

### E. Export

DDP checkpoints are saved directly in Hugging Face format. ZeRO-2 and FSDP require an explicit export step before vLLM serving.

Export a ZeRO-2 checkpoint:

```bash
cd ZeRO2_Training
python3 export_for_vllm.py \
  --checkpoint_dir ./outputs/llama32_1b_zero2 \
  --export_dir ./outputs/llama32_1b_zero2_vllm_export
```

Export an FSDP checkpoint:

```bash
cd FSDP
torchrun --nproc_per_node=4 fsdp_export.py \
  --config sample_config.yaml \
  --checkpoint_dir ./runs/llama32_1b_fsdp/checkpoints/step_00000010 \
  --export_dir ./runs/llama32_1b_export
```

### F. Evaluation

Run one configured vLLM experiment:

```bash
cd vllm_deployment_eval
./run_experiment.py --config configs/experiments.json --name ddp-llama32-1b-ultrachat
```

Run all committed experiment names from `configs/experiments.json` to regenerate the comparison table:

```bash
./run_experiment.py --config configs/experiments.json --name ddp-llama32-1b-ultrachat
./run_experiment.py --config configs/experiments.json --name ddp-llama32-3b-ultrachat
./run_experiment.py --config configs/experiments.json --name fsdp-llama32-1b-finetuned-step6000
./run_experiment.py --config configs/experiments.json --name fsdp-llama32-3b-finetuned-final
./run_experiment.py --config configs/experiments.json --name zero2-llama32-1b-vllm-export
./run_experiment.py --config configs/experiments.json --name zero2-llama32-3b-vllm-export
```

Rebuild derived summaries after copying or editing result artifacts:

```bash
./format_results.py
```

### G. Profiling

The vLLM runner records:

- server startup/load time
- average request latency
- first-token latency
- completion tokens per second
- GPU memory, power, and utilization samples from `nvidia-smi`
- server logs and vLLM metrics snapshots before and after requests

Artifacts are written under `vllm_deployment_eval/results/<strategy>/<run_id>/`.

### H. Quickstart: Reproduce the Headline Result

The fastest way to reproduce the headline serving comparison is to run the vLLM harness against the published Hugging Face checkpoints:

```bash
cd vllm_deployment_eval
./run_experiment.py --config configs/experiments.json --name ddp-llama32-1b-ultrachat
./run_experiment.py --config configs/experiments.json --name fsdp-llama32-1b-finetuned-step6000
./format_results.py
sed -n '1,20p' results/summary_metrics.csv
```

## 6. Results and Observations

- Checkpoint format matters operationally. DDP produced directly serveable Hugging Face checkpoints, while FSDP and ZeRO-2 needed export/consolidation scripts before vLLM could serve the model.
- FSDP gave the best 1B serving latency in the committed runs: 1.2349 s average request latency versus 1.6272 s for DDP and 1.6276 s for ZeRO-2.
- ZeRO-2 gave the best 3B serving latency and completion throughput: 3.6883 s average latency and 34.4365 completion tokens/s, narrowly ahead of FSDP 3B.
- Peak serving memory was dominated by the vLLM runtime configuration rather than training strategy. The 1B runs all reached 19,482 MB, while the 3B runs all reached 19,114 MB.

Representative committed comparison:

| Strategy | Model | Avg latency (s) | First token (s) | Completion tok/s | Load time (s) | Peak GPU memory (MB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| DDP | 1B | 1.6272 | 0.0209 | 80.6349 | 62.0226 | 19482 |
| FSDP | 1B | 1.2349 | 0.0213 | 78.1572 | 82.0271 | 19482 |
| ZeRO-2 | 1B | 1.6276 | 0.0205 | 80.5498 | 82.0287 | 19482 |
| DDP | 3B | 3.9594 | 0.0487 | 32.5599 | 146.0886 | 19114 |
| FSDP | 3B | 3.7019 | 0.0362 | 34.3225 | 124.0405 | 19114 |
| ZeRO-2 | 3B | 3.6883 | 0.0367 | 34.4365 | 132.0552 | 19114 |

## 7. Notes

- Published checkpoints referenced by the committed configs:
  - [Sudarshan1607/ddp-llama32-1b-ultrachat](https://huggingface.co/Sudarshan1607/ddp-llama32-1b-ultrachat)
  - [Sudarshan1607/ddp-llama32-3b-ultrachat](https://huggingface.co/Sudarshan1607/ddp-llama32-3b-ultrachat)
  - [robsol/hpml-final-1b-zero2-vllm-export](https://huggingface.co/robsol/hpml-final-1b-zero2-vllm-export)
  - [robsol/hpml-final-3b-zero2-vllm-export](https://huggingface.co/robsol/hpml-final-3b-zero2-vllm-export)
  - [ehdus9063/llama32-1b-finetuned-step6000](https://huggingface.co/ehdus9063/llama32-1b-finetuned-step6000)
  - [ehdus9063/llama32-3b-finetuned-final](https://huggingface.co/ehdus9063/llama32-3b-finetuned-final)
- Secrets such as `HF_TOKEN` and any WandB credentials should be supplied through environment variables.

### License

Released under the MIT License. See [`LICENSE`](LICENSE).

### AI Tool Use

**Did your team use any AI tool in completing this project?**

- [x] Yes, we used AI assistance as described below.

**Tool(s) used:** Codex, ChatGPT, Claude

**Specific purpose:** Explained concepts, differences between training strategies, provided guidance on model 
checkpointing and vLLM deployment.

**Sections affected:** `DDP/`, `FSDP/`, `ZeRO2_Training/`, `vllm_deployment_eval/`.

**How we verified correctness:** All implementations were our own. We verified code by running the reported experiments, checking committed metrics and logs, and validating exported checkpoints with vLLM/Hugging Face loading paths.

By submitting this project, the team confirms that the analysis, interpretations, and conclusions are our own, and that any AI assistance is fully disclosed above.

### Citation

If you build on this work, please cite:

```bibtex
@misc{distributedllmpipeline2026hpml,
  title  = {Distributed LLM Pipeline},
  author = {McMaster, Peter and Distributed LLM Pipeline Team},
  year   = {2026},
  note   = {HPML Spring 2026 Final Project, Columbia University},
  url    = {https://github.com/PeterMcMaster/distributed-llm-pipeline}
}
```

### Contact

Open a GitHub Issue at [PeterMcMaster/distributed-llm-pipeline](https://github.com/PeterMcMaster/distributed-llm-pipeline).

---

*HPML Spring 2026 - Dr. Kaoutar El Maghraoui - Columbia University*
