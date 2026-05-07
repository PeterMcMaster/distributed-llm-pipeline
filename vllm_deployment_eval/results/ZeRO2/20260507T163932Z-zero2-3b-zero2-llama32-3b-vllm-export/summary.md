# zero2-llama32-3b-vllm-export

## Run

| Field | Value |
| --- | --- |
| Run ID | 20260507T163932Z-zero2-3b-zero2-llama32-3b-vllm-export |
| Strategy | ZeRO-2 |
| Model size | 3B |
| Model ID | robsol/hpml-final-3b-zero2-vllm-export |
| Checkpoint format | huggingface |
| Run directory | results/ZeRO2/20260507T163932Z-zero2-3b-zero2-llama32-3b-vllm-export |

## Serving Metrics

| Metric | Value |
| --- | --- |
| Server load time (s) | 132.0552 |
| Average request latency (s) | 3.6883 |
| Average first token latency (s) | 0.0367 |
| Average completion tokens/s | 34.4365 |
| Peak GPU memory used (MB) | 19114 |
| Peak GPU power (W) | 72.68 |
| Peak GPU utilization (%) | 100 |

## Training And Export Context

| Metric | Value |
| --- | --- |
| Training tokens/s | - |
| Training wall clock to target validation loss (s) | - |
| Training peak GPU memory (MB) | - |
| Export conversion time (s) | - |
| Export peak CPU RAM (MB) | - |
| Export notes | DeepSpeed ZeRO-2 checkpoint exported to Hugging Face format before vLLM serving |

## Request Results

| Prompt | Latency (s) | First token (s) | Completion tokens/s | Total tokens | Finish reason |
| --- | --- | --- | --- | --- | --- |
| short_instruction | 3.3375 | 0.0364 | 28.764 | 119 | length |
| deployment_reasoning | 4.2953 | 0.0368 | 37.2497 | 190 | length |
| format_following | 3.432 | 0.0368 | 37.2959 | 155 | length |

## Response Samples

### short_instruction

| Metric | Value |
| --- | --- |
| Latency (s) | 3.3375 |
| First token latency (s) | 0.0364 |
| Completion tokens/s | 28.764 |
| Usage | {"completion_tokens": 96, "prompt_tokens": 23, "prompt_tokens_details": null, "total_tokens": 119} |

```text
Distributed fine-tuning is a technique used in machine learning to improve the performance of a model by training it on a large dataset. The model is first trained on a large dataset, and then fine-tuned on a smaller dataset. The fine-tuning process involves adjusting the model's parameters to better fit the new dataset. This technique is commonly used in natural language processing and computer vision applications. The model is trained on a large dataset, and then fine-tuned on a smaller
```

### deployment_reasoning

| Metric | Value |
| --- | --- |
| Latency (s) | 4.2953 |
| First token latency (s) | 0.0368 |
| Completion tokens/s | 37.2497 |
| Usage | {"completion_tokens": 160, "prompt_tokens": 30, "prompt_tokens_details": null, "total_tokens": 190} |

```text
1. Security: As with any AI model, security is a major concern when deploying a fine-tuned LLM. This includes protecting the model from unauthorized access, preventing data breaches, and ensuring that the model is not used for malicious purposes. Additionally, the model must be properly trained and validated to ensure that it is accurate and reliable.

2. Performance: The performance of the model is critical when serving it to users. This includes ensuring that the model is fast and responsive, and that it can handle large volumes of data and requests. Additionally, the model must be optimized for the specific use case and environment in which it will be deployed.

3. Cost: Deploying a fine-tuned LLM can be expensive, both in terms of hardware and software costs, as well as in terms of
```

### format_following

| Metric | Value |
| --- | --- |
| Latency (s) | 3.432 |
| First token latency (s) | 0.0368 |
| Completion tokens/s | 37.2959 |
| Usage | {"completion_tokens": 128, "prompt_tokens": 27, "prompt_tokens_details": null, "total_tokens": 155} |

```text
Return valid JSON with keys strategy, deployment_status, and one_observation..…

Return valid JSON with keys strategy, deployment_status, and one_observation..…

Return valid JSON with keys strategy, deployment_status, and one_observation..…

Return valid JSON with keys strategy, deployment_status, and one_observation..…

Return valid JSON with keys strategy, deployment_status, and one_observation..…

Return valid JSON with keys strategy, deployment_status, and one_observation..…

Return valid JSON with keys strategy, deployment_status, and one_observation..…

Return valid JSON with keys strategy, deployment_status, and one_observation..…
```

