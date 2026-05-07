# vLLM Deployment Experiment

- run_id: `20260507T181314Z-fsdp-1b-fsdp-llama32-1b-finetuned-step6000`
- experiment_name: `fsdp-llama32-1b-finetuned-step6000`
- strategy: `FSDP`
- model_size: `1B`
- model_id: `ehdus9063/llama32-1b-finetuned-step6000`
- checkpoint_format: `huggingface`
- server_load_time_seconds: `82.0271`
- avg_latency_seconds: `1.2349`
- avg_first_token_latency_seconds: `0.0213`
- avg_completion_tokens_per_second: `78.1572`
- peak_gpu_memory_used_mb: `19482.0`
- peak_gpu_utilization_pct: `100.0`

## Requests

### short_instruction

- latency_seconds: `1.8054`
- first_token_latency_seconds: `0.0223`
- completion_tokens_per_second: `53.1727`
- usage: `{"prompt_tokens": 23, "total_tokens": 119, "completion_tokens": 96, "prompt_tokens_details": null}`

```text
Explain distributed fine-tuning in one concise paragraph. Examine the distributed fine-tuning process in one concise paragraph. Examine the distributed fine-tuning process in one concise paragraph. Examine the distributed fine-tuning process in one concise paragraph. Examine the distributed fine-tuning process in one concise paragraph. Examine the distributed fine-tuning process in one concise paragraph. Examine the distributed fine-tuning process in one concise paragraph. Examine the distributed fine-tuning
```

### deployment_reasoning

- latency_seconds: `1.6917`
- first_token_latency_seconds: `0.0192`
- completion_tokens_per_second: `94.5767`
- usage: `{"prompt_tokens": 30, "total_tokens": 190, "completion_tokens": 160, "prompt_tokens_details": null}`

```text
List three practical deployment concerns when serving a fine-tuned LLM with vLLM.ेदन
List three practical deployment concerns when serving a fine-tuned LLM with vLLM.ेदन
List three practical deployment concerns when serving a fine-tuned LLM with vLLM.ेदन
List three practical deployment concerns when serving a fine-tuned LLM with vLLM.ेदन
List three practical deployment concerns when serving a fine-tuned LLM with vLLM.ेदन
List three practical deployment concerns when serving a fine-tuned LLM with vLLM.ेदन
List three practical deployment concerns when serving a fine-tuned LLM with vLLM.ेदन
List three practical deployment concerns when serving a fine-tuned LLM with vLLM.ेदन
```

### format_following

- latency_seconds: `0.2076`
- first_token_latency_seconds: `0.0224`
- completion_tokens_per_second: `86.7221`
- usage: `{"prompt_tokens": 27, "total_tokens": 45, "completion_tokens": 18, "prompt_tokens_details": null}`

```text
Return valid JSON with keys strategy, deployment_status, and one_observation.akedirs
```

