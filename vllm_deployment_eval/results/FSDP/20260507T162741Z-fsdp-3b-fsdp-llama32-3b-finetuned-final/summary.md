# fsdp-llama32-3b-finetuned-final

## Run

| Field | Value |
| --- | --- |
| Run ID | 20260507T162741Z-fsdp-3b-fsdp-llama32-3b-finetuned-final |
| Strategy | FSDP |
| Model size | 3B |
| Model ID | ehdus9063/llama32-3b-finetuned-final |
| Checkpoint format | huggingface |
| Run directory | results/FSDP/20260507T162741Z-fsdp-3b-fsdp-llama32-3b-finetuned-final |

## Serving Metrics

| Metric | Value |
| --- | --- |
| Server load time (s) | 124.0405 |
| Average request latency (s) | 3.7019 |
| Average first token latency (s) | 0.0362 |
| Average completion tokens/s | 34.3225 |
| Peak GPU memory used (MB) | 19114 |
| Peak GPU power (W) | 72.32 |
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
| short_instruction | 3.3805 | 0.0369 | 28.3983 | 119 | length |
| deployment_reasoning | 4.2949 | 0.036 | 37.2532 | 190 | length |
| format_following | 3.4302 | 0.0358 | 37.3159 | 155 | length |

## Response Samples

### short_instruction

| Metric | Value |
| --- | --- |
| Latency (s) | 3.3805 |
| First token latency (s) | 0.0369 |
| Completion tokens/s | 28.3983 |
| Usage | {"completion_tokens": 96, "prompt_tokens": 23, "prompt_tokens_details": null, "total_tokens": 119} |

```text
# 1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.
```

### deployment_reasoning

| Metric | Value |
| --- | --- |
| Latency (s) | 4.2949 |
| First token latency (s) | 0.036 |
| Completion tokens/s | 37.2532 |
| Usage | {"completion_tokens": 160, "prompt_tokens": 30, "prompt_tokens_details": null, "total_tokens": 190} |

```text
## Introduction

The LLMs (Large Language Models) are the most popular models in the NLP (Natural Language Processing) domain. They are used in various applications such as chatbots, translation, and question answering. The LLMs are trained on large datasets and are capable of generating human-like responses to any input. However, the LLMs are not perfect and have some limitations. One of the limitations is that they are not trained on specific domains or tasks. This means that they may not be able to generate accurate responses for specific tasks. To overcome this limitation, we can fine-tune the LLMs on specific datasets and tasks. This process is called vLLM (Virtual Large Language Model). In this article, we will discuss the practical deployment concerns when serving a fine
```

### format_following

| Metric | Value |
| --- | --- |
| Latency (s) | 3.4302 |
| First token latency (s) | 0.0358 |
| Completion tokens/s | 37.3159 |
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

