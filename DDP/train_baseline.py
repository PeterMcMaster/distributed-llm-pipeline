import os
import time
import json
import random
import numpy as np
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_NAME = "databricks/databricks-dolly-15k"
OUTPUT_DIR = "./ddp_baseline_output"

MAX_LENGTH = 512
TRAIN_SAMPLES = 4000
VAL_SAMPLES = 500
SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_dolly(example):
    instruction = example["instruction"].strip()
    context = (example.get("context") or "").strip()
    response = example["response"].strip()

    if context:
        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Context:\n{context}\n\n"
            f"### Response:\n{response}"
        )
    else:
        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{response}"
        )

    return {"text": text}


def get_dir_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for fname in files:
            fp = os.path.join(root, fname)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total


def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME)
    formatted = dataset["train"].map(format_dolly)
    formatted = formatted.shuffle(seed=SEED)

    train_end = min(TRAIN_SAMPLES, len(formatted))
    val_end = min(TRAIN_SAMPLES + VAL_SAMPLES, len(formatted))

    train_dataset = formatted.select(range(0, train_end))
    val_dataset = formatted.select(range(train_end, val_end))

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_steps=20,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=20,
        lr_scheduler_type="cosine",
        bf16=True,
        fp16=False,
        report_to="none",
        seed=SEED,
        save_total_limit=2,
        dataloader_num_workers=2,
        max_length=MAX_LENGTH,
        dataset_text_field="text",
        max_grad_norm=1.0,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print("Starting training...")
    start_time = time.time()
    train_result = trainer.train()
    end_time = time.time()

    export_dir = os.path.join(OUTPUT_DIR, "final_hf_export")
    export_start = time.time()
    trainer.model.save_pretrained(export_dir)
    tokenizer.save_pretrained(export_dir)
    export_end = time.time()

    peak_gpu_mem_gb = 0.0
    if torch.cuda.is_available():
        peak_gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    export_size_gb = get_dir_size_bytes(export_dir) / (1024 ** 3)

    report = {
        "model_name": MODEL_NAME,
        "dataset_name": DATASET_NAME,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "max_length": MAX_LENGTH,
        "wall_clock_sec": end_time - start_time,
        "peak_gpu_mem_gb": peak_gpu_mem_gb,
        "export_time_sec": export_end - export_start,
        "export_size_gb": export_size_gb,
        "train_metrics": train_result.metrics,
        "export_dir": export_dir,
    }

    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\nTraining complete.")
    print(json.dumps(report, indent=2))
    print(f"\nSaved model to: {export_dir}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
