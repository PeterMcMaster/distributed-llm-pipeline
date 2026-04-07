import os
import time
import json
import argparse
from typing import Dict, List, Any

import torch
import torch.distributed as dist

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def reduce_sum(value: int, device: torch.device) -> int:
    if dist.is_available() and dist.is_initialized():
        t = torch.tensor([value], device=device, dtype=torch.long)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return int(t.item())
    return int(value)


def format_messages_as_text(messages: List[Dict[str, str]], eos_token: str) -> str:
    parts = []
    for msg in messages:
        role = msg["role"].strip().lower()
        content = msg["content"].strip()

        if role == "user":
            prefix = "User:"
        elif role == "assistant":
            prefix = "Assistant:"
        else:
            prefix = f"{role.capitalize()}:"

        parts.append(f"{prefix}\n{content}")

    return "\n\n".join(parts) + eos_token


class TokenCountingTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_tokens = 0
        self.window_start = time.time()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if model.training:
            if "attention_mask" in inputs:
                local_tokens = int(inputs["attention_mask"].sum().item())
            else:
                local_tokens = int(inputs["input_ids"].numel())

            device = inputs["input_ids"].device
            global_tokens = reduce_sum(local_tokens, device)
            self.window_tokens += global_tokens

        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    def log(self, logs: Dict[str, float], start_time: float = None) -> None:
        now = time.time()
        elapsed = max(now - self.window_start, 1e-8)

        if self.window_tokens > 0:
            logs["tokens_per_sec"] = self.window_tokens / elapsed

        if torch.cuda.is_available():
            logs["gpu_peak_mem_gb"] = torch.cuda.max_memory_allocated() / (1024 ** 3)

        super().log(logs, start_time=start_time)

        self.window_tokens = 0
        self.window_start = now


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--output_dir", type=str, default="./outputs/llama32_1b_zero2")
    parser.add_argument("--max_length", type=int, default=1024)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)

    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)

    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=1000)

    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    hf_token = os.environ.get("HF_TOKEN", None)
    if hf_token is None:
        raise ValueError("HF_TOKEN is not set. You need access to the gated Llama repo.")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    os.makedirs(args.output_dir, exist_ok=True)

    if is_main_process():
        print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        token=hf_token,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_main_process():
        print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        token=hf_token,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    if is_main_process():
        print("Loading dataset...")
    raw_train = load_dataset(args.dataset_name, split="train_sft")
    raw_eval = load_dataset(args.dataset_name, split="test_sft")

    if args.max_train_samples is not None:
        raw_train = raw_train.select(range(min(args.max_train_samples, len(raw_train))))
    if args.max_eval_samples is not None:
        raw_eval = raw_eval.select(range(min(args.max_eval_samples, len(raw_eval))))

    def preprocess(example: Dict[str, Any]) -> Dict[str, Any]:
        text = format_messages_as_text(example["messages"], tokenizer.eos_token)
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )
        return tokenized

    remove_cols = raw_train.column_names

    train_dataset = raw_train.map(
        preprocess,
        remove_columns=remove_cols,
        desc="Tokenizing train",
    )

    eval_dataset = raw_eval.map(
        preprocess,
        remove_columns=remove_cols,
        desc="Tokenizing eval",
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        deepspeed="ds_zero2.json",

        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,

        lr_scheduler_type="cosine",

        logging_strategy="steps",
        logging_steps=args.logging_steps,

        eval_strategy="steps",
        eval_steps=args.eval_steps,

        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,

        bf16=bf16_ok,
        fp16=not bf16_ok,

        dataloader_num_workers=2,
        dataloader_pin_memory=True,

        gradient_checkpointing=True,
        report_to="none",

        remove_unused_columns=False,
        prediction_loss_only=True,
    )

    trainer = TokenCountingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    if is_main_process():
        print("Starting training...")
    start_time = time.time()

    train_result = trainer.train()

    total_train_time = time.time() - start_time

    if is_main_process():
        metrics = train_result.metrics
        metrics["total_train_time_sec"] = total_train_time
        metrics["peak_gpu_mem_gb"] = (
            torch.cuda.max_memory_allocated() / (1024 ** 3)
            if torch.cuda.is_available() else 0.0
        )

        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)
        trainer.save_state()

        print("\n===== FINAL TRAIN METRICS =====")
        print(json.dumps(metrics, indent=2))

        eval_metrics = trainer.evaluate()
        print("\n===== FINAL EVAL METRICS =====")
        print(json.dumps(eval_metrics, indent=2))

        with open(os.path.join(args.output_dir, "final_metrics.json"), "w") as f:
            json.dump(
                {
                    "train_metrics": metrics,
                    "eval_metrics": eval_metrics,
                },
                f,
                indent=2,
            )

        print(f"\nDone. Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
