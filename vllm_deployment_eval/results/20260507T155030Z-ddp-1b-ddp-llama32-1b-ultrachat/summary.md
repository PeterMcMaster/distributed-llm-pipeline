# vLLM Deployment Experiment

- run_id: `20260507T155030Z-ddp-1b-ddp-llama32-1b-ultrachat`
- experiment_name: `ddp-llama32-1b-ultrachat`
- strategy: `DDP`
- model_size: `1B`
- model_id: `Sudarshan1607/ddp-llama32-1b-ultrachat`
- checkpoint_format: `huggingface`
- server_load_time_seconds: `62.0226`
- avg_latency_seconds: `1.6272`
- avg_first_token_latency_seconds: `0.0209`
- avg_completion_tokens_per_second: `80.6349`
- peak_gpu_memory_used_mb: `19482.0`
- peak_gpu_utilization_pct: `100.0`

## Requests

### short_instruction

- latency_seconds: `1.8482`
- first_token_latency_seconds: `0.0222`
- completion_tokens_per_second: `51.9425`
- usage: `{"prompt_tokens": 23, "total_tokens": 119, "completion_tokens": 96, "prompt_tokens_details": null}`

```text
Explain distributed fine-tuning in one concise paragraph. ReferentialAction
Explain distributed fine-tuning in one concise paragraph. ReferentialAction
Explain distributed fine-tuning in one concise paragraph. ReferentialAction
Explain distributed fine-tuning in one concise paragraph. ReferentialAction
Explain distributed fine-tuning in one concise paragraph. ReferentialAction
Explain distributed fine-tuning in one concise paragraph. ReferentialAction
Explain distributed fine-tuning in one concise paragraph. ReferentialAction
Explain distributed fine-t
```

### deployment_reasoning

- latency_seconds: `1.6899`
- first_token_latency_seconds: `0.02`
- completion_tokens_per_second: `94.6798`
- usage: `{"prompt_tokens": 30, "total_tokens": 190, "completion_tokens": 160, "prompt_tokens_details": null}`

```text
List three practical deployment concerns when serving a fine-tuned LLM with vLLM.ेदनinerary
List three practical deployment concerns when serving a fine-tuned LLM with vLLM.ेदनinerary
List three practical deployment concerns when serving a fine-tuned LLM with vLLM.ेदनinerary
List three practical deployment concerns when serving a fine-tuned LLM with vLLM.ेदनinerary
List three practical deployment concerns when serving a fine-tuned LLM with vLLM.ेदनinerary
List three practical deployment concerns when serving a fine-tuned LLM with vLLM.ेदनinerary
List three practical deployment concerns when serving a fine-tuned LLM with vLLM.ेदनinerary
List three practical deployment concerns when serving a fine-tuned LLM
```

### format_following

- latency_seconds: `1.3434`
- first_token_latency_seconds: `0.0205`
- completion_tokens_per_second: `95.2825`
- usage: `{"prompt_tokens": 27, "total_tokens": 155, "completion_tokens": 128, "prompt_tokens_details": null}`

```text
Return valid JSON with keys strategy, deployment_status, and one_observation.### Return valid JSON with keys strategy, deployment_status, and one_observation.### Return valid JSON with keys strategy, deployment_status, and one_observation.### Return valid JSON with keys strategy, deployment_status, and one_observation.### Return valid JSON with keys strategy, deployment_status, and one_observation.### Return valid JSON with keys strategy, deployment_status, and one_observation.### Return valid JSON with keys strategy, deployment_status, and one_observation.### Return valid JSON with keys strategy, deployment_status, and one_observation.
```

