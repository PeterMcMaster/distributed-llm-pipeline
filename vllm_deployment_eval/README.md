# vLLM Deployment Evaluation

Centralized runner for comparing inference-serving behavior across fine-tuning
strategies and model sizes.

Run one configured experiment:

```bash
cd /home/petermcmaster/vllm-deployment-eval
./run_experiment.py --config configs/experiments.json --name ddp-llama32-1b-ultrachat
```

Run an ad hoc model:

```bash
./run_experiment.py \
  --experiment-name ddp-llama32-1b-ultrachat \
  --strategy DDP \
  --model-size 1B \
  --model-id Sudarshan1607/ddp-llama32-1b-ultrachat \
  --chat-template templates/llama3-chat-template.jinja
```

Central outputs:

```text
results/experiments.jsonl
results/summary_metrics.csv
results/<strategy>/<run_id>/
```

Each run directory contains server logs, request/response JSONL, GPU samples,
environment metadata, vLLM metrics, and a Markdown summary.
