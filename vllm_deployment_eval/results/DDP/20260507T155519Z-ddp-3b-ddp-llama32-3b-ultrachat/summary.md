# ddp-llama32-3b-ultrachat

## Run

| Field | Value |
| --- | --- |
| Run ID | 20260507T155519Z-ddp-3b-ddp-llama32-3b-ultrachat |
| Strategy | DDP |
| Model size | 3B |
| Model ID | Sudarshan1607/ddp-llama32-3b-ultrachat |
| Checkpoint format | huggingface |
| Run directory | results/DDP/20260507T155519Z-ddp-3b-ddp-llama32-3b-ultrachat |

## Serving Metrics

| Metric | Value |
| --- | --- |
| Server load time (s) | 146.0886 |
| Average request latency (s) | 3.9594 |
| Average first token latency (s) | 0.0487 |
| Average completion tokens/s | 32.5599 |
| Peak GPU memory used (MB) | 19114 |
| Peak GPU power (W) | 72.66 |
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
| short_instruction | 4.1511 | 0.045 | 23.1267 | 119 | length |
| deployment_reasoning | 4.2973 | 0.0512 | 37.2325 | 190 | length |
| format_following | 3.4297 | 0.0499 | 37.3206 | 155 | length |

## Response Samples

### short_instruction

| Metric | Value |
| --- | --- |
| Latency (s) | 4.1511 |
| First token latency (s) | 0.045 |
| Completion tokens/s | 23.1267 |
| Usage | {"completion_tokens": 96, "prompt_tokens": 23, "prompt_tokens_details": null, "total_tokens": 119} |

```text
Distributed fine-tuning is a technique used in machine learning to improve the performance of a model by training it on multiple datasets simultaneously. It involves splitting the training data into smaller chunks and training the model on each chunk separately, while keeping the model parameters consistent across all chunks. This technique is particularly useful when the datasets are large and heterogeneous, as it allows the model to learn from a variety of data sources and improve its generalization capabilities. Distributed fine-tuning is often used in
```

### deployment_reasoning

| Metric | Value |
| --- | --- |
| Latency (s) | 4.2973 |
| First token latency (s) | 0.0512 |
| Completion tokens/s | 37.2325 |
| Usage | {"completion_tokens": 160, "prompt_tokens": 30, "prompt_tokens_details": null, "total_tokens": 190} |

```text
List three practical deployment concerns when serving a fine-tuned LLM with vLLM. :＼
List three practical deployment concerns when serving a fine-tuned LLM with vLLM. :＼
List three practical deployment concerns when serving a fine-tuned LLM with vLLM. :＼
List three practical deployment concerns when serving a fine-tuned LLM with vLLM. :＼
List three practical deployment concerns when serving a fine-tuned LLM with vLLM. :＼
List three practical deployment concerns when serving a fine-tuned LLM with vLLM. :＼
List three practical deployment concerns when serving a fine-tuned LLM with vLLM. :＼
List three practical deployment concerns when serving a fine-tuned LLM with vLLM. :＼
```

### format_following

| Metric | Value |
| --- | --- |
| Latency (s) | 3.4297 |
| First token latency (s) | 0.0499 |
| Completion tokens/s | 37.3206 |
| Usage | {"completion_tokens": 128, "prompt_tokens": 27, "prompt_tokens_details": null, "total_tokens": 155} |

```text
Return valid JSON with keys strategy, deployment_status, and one_observation.]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
]‏
```

