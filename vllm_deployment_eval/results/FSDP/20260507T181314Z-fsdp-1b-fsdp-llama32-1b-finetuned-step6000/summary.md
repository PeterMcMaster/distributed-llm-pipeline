# fsdp-llama32-1b-finetuned-step6000

## Run

| Field | Value |
| --- | --- |
| Run ID | 20260507T181314Z-fsdp-1b-fsdp-llama32-1b-finetuned-step6000 |
| Strategy | FSDP |
| Model size | 1B |
| Model ID | ehdus9063/llama32-1b-finetuned-step6000 |
| Checkpoint format | huggingface |
| Run directory | results/FSDP/20260507T181314Z-fsdp-1b-fsdp-llama32-1b-finetuned-step6000 |

## Serving Metrics

| Metric | Value |
| --- | --- |
| Server load time (s) | 82.0271 |
| Average request latency (s) | 1.2349 |
| Average first token latency (s) | 0.0213 |
| Average completion tokens/s | 78.1572 |
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
| Export notes | FSDP checkpoint exported to Hugging Face format before vLLM serving |

## Request Results

| Prompt | Latency (s) | First token (s) | Completion tokens/s | Total tokens | Finish reason |
| --- | --- | --- | --- | --- | --- |
| short_instruction | 1.8054 | 0.0223 | 53.1727 | 119 | length |
| deployment_reasoning | 1.6917 | 0.0192 | 94.5767 | 190 | length |
| format_following | 0.2076 | 0.0224 | 86.7221 | 45 | stop |

## Response Samples

### short_instruction

| Metric | Value |
| --- | --- |
| Latency (s) | 1.8054 |
| First token latency (s) | 0.0223 |
| Completion tokens/s | 53.1727 |
| Usage | {"completion_tokens": 96, "prompt_tokens": 23, "prompt_tokens_details": null, "total_tokens": 119} |

```text
Explain distributed fine-tuning in one concise paragraph. Examine the distributed fine-tuning process in one concise paragraph. Examine the distributed fine-tuning process in one concise paragraph. Examine the distributed fine-tuning process in one concise paragraph. Examine the distributed fine-tuning process in one concise paragraph. Examine the distributed fine-tuning process in one concise paragraph. Examine the distributed fine-tuning process in one concise paragraph. Examine the distributed fine-tuning
```

### deployment_reasoning

| Metric | Value |
| --- | --- |
| Latency (s) | 1.6917 |
| First token latency (s) | 0.0192 |
| Completion tokens/s | 94.5767 |
| Usage | {"completion_tokens": 160, "prompt_tokens": 30, "prompt_tokens_details": null, "total_tokens": 190} |

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

| Metric | Value |
| --- | --- |
| Latency (s) | 0.2076 |
| First token latency (s) | 0.0224 |
| Completion tokens/s | 86.7221 |
| Usage | {"completion_tokens": 18, "prompt_tokens": 27, "prompt_tokens_details": null, "total_tokens": 45} |

```text
Return valid JSON with keys strategy, deployment_status, and one_observation.akedirs
```

