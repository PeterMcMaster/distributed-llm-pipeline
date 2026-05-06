import argparse
import inspect
import json
import os
import time
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def distributed_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def reduce_sum(value: int, device: torch.device) -> int:
    if dist.is_available() and dist.is_initialized():
        tensor = torch.tensor([value], device=device, dtype=torch.long)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return int(tensor.item())
    return int(value)


def sanitize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = {}
    for key, value in metrics.items():
        safe_key = key.replace("_mem_", "_memory_")
        sanitized[safe_key] = value
    return sanitized


def parse_report_to(value: str) -> Any:
    cleaned = value.strip()
    if not cleaned or cleaned.lower() == "none":
        return "none"

    targets = [item.strip() for item in cleaned.split(",") if item.strip()]
    return targets or "none"


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


class ExplicitWandbLoggerCallback(TrainerCallback):
    def __init__(self, init_kwargs: Dict[str, Any], config: Dict[str, Any]):
        self.enabled = is_main_process() and bool(init_kwargs.get("project"))
        self.init_kwargs = {key: value for key, value in init_kwargs.items() if value}
        self.config = config
        self.run = None
        self.wandb = None

    def on_train_begin(self, args, state, control, **kwargs):
        if not self.enabled or self.run is not None:
            return control

        import wandb

        self.wandb = wandb
        self.run = self.wandb.init(**self.init_kwargs, config=self.config)
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.run or not logs:
            return control

        payload = sanitize_metrics(dict(logs))
        self.run.log(payload, step=state.global_step)
        return control

    def update_summary(self, metrics: Dict[str, Any], prefix: Optional[str] = None) -> None:
        if not self.run:
            return

        if prefix:
            self.run.summary.update({f"{prefix}_{key}": value for key, value in metrics.items()})
        else:
            self.run.summary.update(metrics)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
            self.run = None


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

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        now = time.time()
        elapsed = max(now - self.window_start, 1e-8)

        if self.window_tokens > 0:
            logs["tokens_per_sec"] = self.window_tokens / elapsed

        if torch.cuda.is_available():
            logs["gpu_peak_memory_gb"] = torch.cuda.max_memory_allocated() / (1024 ** 3)

        super().log(logs, start_time=start_time)

        self.window_tokens = 0
        self.window_start = now


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--output_dir", type=str, default="./outputs/llama32_1b_zero2")
    parser.add_argument("--deepspeed_config", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=1024)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)

    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_steps", type=int, default=-1)

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--run_name", type=str, default=None)

    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=1000)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
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
        return tokenizer(
            text,
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )

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
    ds_config_path = args.deepspeed_config or os.path.join(os.path.dirname(__file__), "ds_zero2.json")
    report_to = parse_report_to(args.report_to)
    wandb_project = os.environ.get("WANDB_PROJECT", "").strip()
    wandb_entity = os.environ.get("WANDB_ENTITY", "").strip()
    wandb_name = os.environ.get("WANDB_NAME", "").strip() or args.run_name or os.path.basename(args.output_dir)
    wandb_tags = [item.strip() for item in os.environ.get("WANDB_TAGS", "").split(",") if item.strip()]
    use_explicit_wandb = bool(wandb_project)

    training_args_kwargs = {
        "output_dir": args.output_dir,
        "overwrite_output_dir": True,
        "deepspeed": ds_config_path,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "lr_scheduler_type": "cosine",
        "logging_strategy": "steps",
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "bf16": bf16_ok,
        "fp16": not bf16_ok,
        "dataloader_num_workers": 2,
        "dataloader_pin_memory": True,
        "gradient_checkpointing": True,
        "report_to": "none" if use_explicit_wandb else report_to,
        "remove_unused_columns": False,
        "prediction_loss_only": True,
    }

    if args.run_name:
        training_args_kwargs["run_name"] = args.run_name

    training_args_signature = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in training_args_signature:
        training_args_kwargs["eval_strategy"] = "steps"
    else:
        training_args_kwargs["evaluation_strategy"] = "steps"

    training_args = TrainingArguments(**training_args_kwargs)

    wandb_callback = None
    if use_explicit_wandb:
        wandb_init_kwargs = {
            "project": wandb_project,
            "entity": wandb_entity or None,
            "name": wandb_name,
            "job_type": "training",
            "dir": os.environ.get("WANDB_DIR"),
            "tags": wandb_tags or None,
        }
        wandb_config = {
            "model_name": args.model_name,
            "dataset_name": args.dataset_name,
            "output_dir": args.output_dir,
            "deepspeed_config": ds_config_path,
            "max_length": args.max_length,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "num_train_epochs": args.num_train_epochs,
            "warmup_ratio": args.warmup_ratio,
            "max_steps": args.max_steps,
            "max_train_samples": args.max_train_samples,
            "max_eval_samples": args.max_eval_samples,
            "seed": args.seed,
            "launcher_report_to": args.report_to,
            "logging_backend": "explicit_wandb_callback",
        }
        wandb_callback = ExplicitWandbLoggerCallback(wandb_init_kwargs, wandb_config)

    trainer = TokenCountingTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    if wandb_callback is not None:
        trainer.add_callback(wandb_callback)

    if is_main_process():
        print("Starting training...")
    start_time = time.time()

    if args.resume_from_checkpoint:
        train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        train_result = trainer.train()
    total_train_time = time.time() - start_time
    distributed_barrier()

    eval_metrics = sanitize_metrics(trainer.evaluate())
    distributed_barrier()

    if is_main_process():
        metrics = sanitize_metrics(dict(train_result.metrics))
        metrics["total_train_time_sec"] = total_train_time
        metrics["peak_gpu_memory_gb"] = (
            torch.cuda.max_memory_allocated() / (1024 ** 3)
            if torch.cuda.is_available() else 0.0
        )

        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        print("\n===== FINAL TRAIN METRICS =====")
        print(json.dumps(metrics, indent=2))

        print("\n===== FINAL EVAL METRICS =====")
        print(json.dumps(eval_metrics, indent=2))

        with open(os.path.join(args.output_dir, "final_metrics.json"), "w") as file_handle:
            json.dump(
                {
                    "train_metrics": metrics,
                    "eval_metrics": eval_metrics,
                },
                file_handle,
                indent=2,
            )

        if wandb_callback is not None:
            wandb_callback.update_summary(metrics)
            wandb_callback.update_summary(eval_metrics, prefix="final_eval")
            wandb_callback.finish()

        print(f"\nDone. Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
