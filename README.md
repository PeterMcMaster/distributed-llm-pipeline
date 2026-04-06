# Distributed LLM Pipeline

An end-to-end LLM systems project exploring how distributed fine-tuning strategies affect downstream serving with vLLM. This project compares PyTorch DDP, DeepSpeed ZeRO-2, and PyTorch FSDP on Llama 3.2 1B and 3B models, with a focus not only on training speed and memory efficiency, but also on checkpoint export complexity, deployment readiness, and inference performance. The goal is to understand how training-time systems decisions impact the full path from fine-tuning to production-style model serving.

## vLLM Setup

These steps assume a Linux VM with an NVIDIA GPU and working NVIDIA drivers.

### 1. Create and activate a Python environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

uv venv --python 3.12 --seed
source .venv/bin/activate
```

### 2. Install vLLM

```bash
uv pip install "https://github.com/vllm-project/vllm/releases/download/v0.19.0/vllm-0.19.0+cu130-cp38-abi3-manylinux_2_35_x86_64.whl" --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
uv pip install openai "huggingface_hub[cli]"
```

### 3. Configure the CUDA library path

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
```

To make this persistent across sessions:

```bash
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}' >> ~/.bashrc
source ~/.bashrc
```

### 4. Authenticate with Hugging Face

Create a Hugging Face access token with read access, then log in:

```bash
hf auth login
hf auth whoami
```

### 5. Launch a test model with vLLM

```bash
vllm serve meta-llama/Llama-3.2-1B-Instruct \
  --dtype auto \
  --api-key devkey \
  --generation-config vllm
```

### 6. Test the server from a second terminal

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer devkey" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [
      {"role": "user", "content": "Say hello in one sentence."}
    ],
    "temperature": 0.0,
    "max_tokens": 32
  }'
```

## Planned Evaluation

The serving side of the project measures more than just basic latency. We evaluate model load time, request latency, throughput under load, and qualitative response behavior on a fixed prompt set. These results are later paired with training-side metrics such as throughput, memory usage, and checkpoint export overhead to compare end-to-end tradeoffs across DDP, ZeRO-2, and FSDP.

## Repository Goals

- Serve baseline and fine-tuned Llama 3.2 checkpoints with vLLM
- Benchmark inference latency and throughput
- Compare how different distributed training strategies affect deployment readiness
- Provide an end-to-end view from fine-tuning to production-style serving
