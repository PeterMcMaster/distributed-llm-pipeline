# vLLM Deployment Experiment

- run_id: `20260507T163932Z-zero2-3b-zero2-llama32-3b-vllm-export`
- experiment_name: `zero2-llama32-3b-vllm-export`
- strategy: `ZeRO-2`
- model_size: `3B`
- model_id: `robsol/hpml-final-3b-zero2-vllm-export`
- checkpoint_format: `huggingface`
- server_load_time_seconds: `132.0552`
- avg_latency_seconds: `3.6883`
- avg_first_token_latency_seconds: `0.0367`
- avg_completion_tokens_per_second: `34.4365`
- peak_gpu_memory_used_mb: `19114.0`
- peak_gpu_utilization_pct: `100.0`

## Requests

### short_instruction

- latency_seconds: `3.3375`
- first_token_latency_seconds: `0.0364`
- completion_tokens_per_second: `28.764`
- usage: `{"prompt_tokens": 23, "total_tokens": 119, "completion_tokens": 96, "prompt_tokens_details": null}`

```text
Distributed fine-tuning is a technique used in machine learning to improve the performance of a model by training it on a large dataset. The model is first trained on a large dataset, and then fine-tuned on a smaller dataset. The fine-tuning process involves adjusting the model's parameters to better fit the new dataset. This technique is commonly used in natural language processing and computer vision applications. The model is trained on a large dataset, and then fine-tuned on a smaller
```

### deployment_reasoning

- latency_seconds: `4.2953`
- first_token_latency_seconds: `0.0368`
- completion_tokens_per_second: `37.2497`
- usage: `{"prompt_tokens": 30, "total_tokens": 190, "completion_tokens": 160, "prompt_tokens_details": null}`

```text
1. Security: As with any AI model, security is a major concern when deploying a fine-tuned LLM. This includes protecting the model from unauthorized access, preventing data breaches, and ensuring that the model is not used for malicious purposes. Additionally, the model must be properly trained and validated to ensure that it is accurate and reliable.

2. Performance: The performance of the model is critical when serving it to users. This includes ensuring that the model is fast and responsive, and that it can handle large volumes of data and requests. Additionally, the model must be optimized for the specific use case and environment in which it will be deployed.

3. Cost: Deploying a fine-tuned LLM can be expensive, both in terms of hardware and software costs, as well as in terms of
```

### format_following

- latency_seconds: `3.432`
- first_token_latency_seconds: `0.0368`
- completion_tokens_per_second: `37.2959`
- usage: `{"prompt_tokens": 27, "total_tokens": 155, "completion_tokens": 128, "prompt_tokens_details": null}`

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

