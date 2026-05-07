# ddp-llama32-1b-ultrachat

## Run

| Field | Value |
| --- | --- |
| Run ID | 20260507T155030Z-ddp-1b-ddp-llama32-1b-ultrachat |
| Strategy | DDP |
| Model size | 1B |
| Model ID | Sudarshan1607/ddp-llama32-1b-ultrachat |
| Checkpoint format | huggingface |
| Run directory | results/DDP/20260507T155030Z-ddp-1b-ddp-llama32-1b-ultrachat |

## Serving Metrics

| Metric | Value |
| --- | --- |
| Server load time (s) | 62.0226 |
| Average request latency (s) | 1.6272 |
| Average first token latency (s) | 0.0209 |
| Average completion tokens/s | 80.6349 |
| Peak GPU memory used (MB) | 19482 |
| Peak GPU power (W) | 72.59 |
| Peak GPU utilization (%) | 100 |

## Training And Export Context

| Metric | Value |
| --- | --- |
| Training tokens/s | - |
| Training wall clock to target validation loss (s) | - |
| Training peak GPU memory (MB) | - |
| Export conversion time (s) | - |
| Export peak CPU RAM (MB) | - |
| Export notes | none; checkpoint is already in Hugging Face format |

## Request Results

| Prompt | Latency (s) | First token (s) | Completion tokens/s | Total tokens | Finish reason |
| --- | --- | --- | --- | --- | --- |
| short_instruction | 1.8482 | 0.0222 | 51.9425 | 119 | length |
| deployment_reasoning | 1.6899 | 0.02 | 94.6798 | 190 | length |
| format_following | 1.3434 | 0.0205 | 95.2825 | 155 | length |

## Response Samples

### short_instruction

| Metric | Value |
| --- | --- |
| Latency (s) | 1.8482 |
| First token latency (s) | 0.0222 |
| Completion tokens/s | 51.9425 |
| Usage | {"completion_tokens": 96, "prompt_tokens": 23, "prompt_tokens_details": null, "total_tokens": 119} |

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

| Metric | Value |
| --- | --- |
| Latency (s) | 1.6899 |
| First token latency (s) | 0.02 |
| Completion tokens/s | 94.6798 |
| Usage | {"completion_tokens": 160, "prompt_tokens": 30, "prompt_tokens_details": null, "total_tokens": 190} |

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

| Metric | Value |
| --- | --- |
| Latency (s) | 1.3434 |
| First token latency (s) | 0.0205 |
| Completion tokens/s | 95.2825 |
| Usage | {"completion_tokens": 128, "prompt_tokens": 27, "prompt_tokens_details": null, "total_tokens": 155} |

```text
Return valid JSON with keys strategy, deployment_status, and one_observation.### Return valid JSON with keys strategy, deployment_status, and one_observation.### Return valid JSON with keys strategy, deployment_status, and one_observation.### Return valid JSON with keys strategy, deployment_status, and one_observation.### Return valid JSON with keys strategy, deployment_status, and one_observation.### Return valid JSON with keys strategy, deployment_status, and one_observation.### Return valid JSON with keys strategy, deployment_status, and one_observation.### Return valid JSON with keys strategy, deployment_status, and one_observation.
```

