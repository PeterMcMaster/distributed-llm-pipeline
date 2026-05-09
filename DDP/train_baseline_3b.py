import os
import time
import json
import torch
import wandb

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

MODEL_NAME = "meta-llama/Llama-3.2-3B"
DATASET_NAME = "HuggingFaceH4/ultrachat_200k"
OUTPUT_DIR = "./output_llama32_3b_4gpu"

MAX_LENGTH = 512
TRAIN_SAMPLES = None
VAL_SAMPLES = 1000
SEED = 42


def get_rank():
    return int(os.environ.get("RANK", "0"))


def get_world_size():
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process():
    return get_rank() == 0


def format_ultrachat(example):
    parts = []

    for msg in example["messages"]:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user":
            parts.append(f"### User:\n{content}")
        elif role == "assistant":
            parts.append(f"### Assistant:\n{content}")

    return {"text": "\n\n".join(parts).strip()}


def get_dir_size_gb(path):
    total = 0
    if not os.path.exists(path):
        return 0.0

    for root, _, files in os.walk(path):
        for filename in files:
            fp = os.path.join(root, filename)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)

    return total / (1024 ** 3)


def main():
    rank = get_rank()
    world_size = get_world_size()
    main_process = is_main_process()

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if main_process:
        print(f"Starting DDP run with world_size={world_size}")
        print(f"Model: {MODEL_NAME}")
        print(f"Dataset: {DATASET_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=hf_token,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if main_process:
        print("Loading UltraChat200K...")

    dataset = load_dataset(DATASET_NAME)

    train_raw = dataset["train_sft"].shuffle(seed=SEED)
    val_raw = dataset["test_sft"].shuffle(seed=SEED)

    if TRAIN_SAMPLES is None:
        train_dataset = train_raw.map(format_ultrachat)
    else:
        train_dataset = train_raw.select(range(TRAIN_SAMPLES)).map(format_ultrachat)
    
    val_dataset = val_raw.select(range(VAL_SAMPLES)).map(format_ultrachat)

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

    train_dataset = train_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    val_dataset = val_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=val_dataset.column_names,
    )

    if main_process:
        print(f"Tokenized train samples: {len(train_dataset)}")
        print(f"Tokenized val samples: {len(val_dataset)}")
        print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=hf_token,
        dtype=torch.bfloat16,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=20,
        lr_scheduler_type="cosine",
        logging_steps=20,
        eval_strategy="no",
        eval_steps=100,
        save_strategy="no",
        save_total_limit=None,
        bf16=True,
        fp16=False,
        report_to="wandb",
        run_name=os.environ.get(
            "WANDB_RUN_NAME",
            f"ddp-llama32-3b-ultrachat-{world_size}gpu",
        ),
        dataloader_num_workers=2,
        seed=SEED,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    if main_process:
        print("Starting training...")

    start_time = time.time()
    train_result = trainer.train()
    end_time = time.time()

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    if main_process:
        export_start = time.time()
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        export_end = time.time()

        peak_gpu_mem_gb = 0.0
        if torch.cuda.is_available():
            peak_gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

        report = {
            "model_name": MODEL_NAME,
            "dataset_name": DATASET_NAME,
            "method": "DDP",
            "world_size": world_size,
            "train_samples": TRAIN_SAMPLES,
            "val_samples": VAL_SAMPLES,
            "max_length": MAX_LENGTH,
            "wall_clock_sec": end_time - start_time,
            "peak_gpu_mem_gb_rank0": peak_gpu_mem_gb,
            "export_time_sec": export_end - export_start,
            "export_size_gb": get_dir_size_gb(OUTPUT_DIR),
            "train_metrics": train_result.metrics,
            "output_dir": OUTPUT_DIR,
        }

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")

        with open(metrics_path, "w") as f:
            json.dump(report, f, indent=2)

        if wandb.run is not None:
            wandb.log(report)
            wandb.finish()

        print("Training complete.")
        print(json.dumps(report, indent=2))
        print(f"Saved model to: {OUTPUT_DIR}")
        print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
