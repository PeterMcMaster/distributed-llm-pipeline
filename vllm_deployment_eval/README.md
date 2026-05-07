# vLLM Deployment Evaluation

Centralized runner for comparing vLLM inference behavior across fine-tuning
strategies and model sizes. Runs are grouped by training strategy and every run
gets the same artifact layout.

## Run an experiment

```bash
cd vllm_deployment_eval
./run_experiment.py --config configs/experiments.json --name ddp-llama32-1b-ultrachat
```

Run an ad hoc model with explicit settings:

```bash
./run_experiment.py \
  --experiment-name ddp-llama32-1b-ultrachat \
  --strategy DDP \
  --model-size 1B \
  --model-id Sudarshan1607/ddp-llama32-1b-ultrachat \
  --chat-template templates/llama3-chat-template.jinja
```

## Result layout

```text
results/experiments.jsonl
results/summary_metrics.csv
results/<strategy>/
  <run_id>/
    gpu_samples.csv
    metadata.json
    metrics_before.txt
    metrics_after.txt
    requests.jsonl
    run_record.json
    server.log
    summary.md
```

`summary.md` is the human-readable report for a run. `run_record.json`,
`requests.jsonl`, and `gpu_samples.csv` are the structured sources used to
rebuild reports.

## Reformat existing results

After copying in artifacts or changing report formatting, rebuild the derived
summary files:

```bash
./format_results.py
```

The formatter rewrites `summary.md`, `experiments.jsonl`, and
`summary_metrics.csv`. It does not edit raw logs, request JSONL, GPU samples, or
metrics dumps.
