from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from datasets import load_dataset
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy, size_based_auto_wrap_policy
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
)

try:
    import yaml
except Exception:
    yaml = None

try:
    import wandb  # type: ignore
except Exception:
    wandb = None


@dataclass
class Config:
    model_name_or_path: str = "meta-llama/Llama-3.2-1B"
    trust_remote_code: bool = False
    dataset_name: str = "HuggingFaceH4/ultrachat_200k"
    train_split: str = "train_sft"
    val_split: str = "test_sft"
    output_dir: str = "./fsdp_outputs"
    max_seq_length: int = 2048

    epochs: int = 1
    max_steps: int = -1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0

    num_preprocessing_workers: int = 4
    dataloader_num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True

    log_every_n_steps: int = 10
    eval_every_n_steps: int = 200
    save_every_n_steps: int = 500
    seed: int = 42

    sharding_strategy: str = "FULL_SHARD"
    use_orig_params: bool = True
    limit_all_gathers: bool = True
    forward_prefetch: bool = False
    sync_module_states: bool = True
    cpu_offload: bool = False
    auto_wrap_policy: str = "transformer_based"
    min_num_params: int = 1_000_000
    activation_checkpointing: bool = True

    prefer_bf16: bool = True
    param_dtype: str = "bf16"
    reduce_dtype: str = "bf16"
    buffer_dtype: str = "bf16"
    allow_tf32: bool = True

    resume_from_checkpoint: Optional[str] = None
    save_optimizer: bool = True
    save_rng_state: bool = True

    validation_text: str = (
        "def fibonacci(n):\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    return fibonacci(n - 1) + fibonacci(n - 2)\n"
    )

    use_wandb: bool = False
    wandb_project: str = "fsdp-llama"
    wandb_run_name: Optional[str] = None

    use_fast_tokenizer: bool = True
    attn_implementation: str = "sdpa"
    use_cache: bool = False
    # Smoke-test controls.
    build_val_dataset: bool = False
    max_train_samples: Optional[int] = None


@dataclass
class DistEnv:
    rank: int
    local_rank: int
    world_size: int
    backend: str
    device: torch.device


class AppState(Stateful):
    """Stateful wrapper for distributed checkpointing."""

    def __init__(self, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer]) -> None:
        self.model = model
        self.optimizer = optimizer

    def state_dict(self) -> Dict[str, Any]:
        model_state, optim_state = get_state_dict(self.model, self.optimizer)
        return {"model": model_state, "optim": optim_state}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
        )


class WandbLogger:
    """Optional wandb wrapper that becomes a no-op if unavailable."""

    def __init__(self, enabled: bool, cfg: Config, rank: int) -> None:
        self.enabled = bool(enabled and wandb is not None and rank == 0)
        self._run = None
        if enabled and wandb is None and rank == 0:
            logging.getLogger("fsdp_train").warning(
                "wandb requested but not installed; using standard logging."
            )
        if self.enabled:
            self._run = wandb.init(
                project=cfg.wandb_project,
                name=cfg.wandb_run_name,
                config=asdict(cfg),
            )

    def log(self, payload: Dict[str, Any], step: Optional[int] = None) -> None:
        if self.enabled and self._run is not None:
            wandb.log(payload, step=step)

    def finish(self) -> None:
        if self.enabled and self._run is not None:
            wandb.finish()


def load_config(config_path: Optional[str]) -> Config:
    """Load optional YAML/JSON config on top of inline defaults."""
    cfg = Config()
    if config_path is None:
        return cfg

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    if path.suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML not installed but YAML config was provided.")
        with path.open("r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f)
    elif path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            user_cfg = json.load(f)
    else:
        raise ValueError("Config must be .json, .yaml, or .yml")

    if not isinstance(user_cfg, dict):
        raise ValueError("Config must be a top-level mapping.")

    for key, value in user_cfg.items():
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown config key: {key}")
        setattr(cfg, key, value)

    return cfg


def setup_logging(rank: int, output_dir: str) -> logging.Logger:
    """Configure rank-aware console/file logging."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("fsdp_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    class RankFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            record.rank = rank
            return True

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | rank=%(rank)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(RankFilter())
    logger.addHandler(stream_handler)

    if rank == 0:
        file_handler = logging.FileHandler(Path(output_dir) / "train.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.addFilter(RankFilter())
        logger.addHandler(file_handler)

    return logger


def detect_backend() -> str:
    """Select NCCL on CUDA, otherwise gloo."""
    if torch.cuda.is_available():
        return "nccl"
    return "gloo"


def init_distributed() -> DistEnv:
    """Initialize distributed runtime from torchrun environment variables."""
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    backend = detect_backend()

    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if torch.cuda.is_available():
        cuda_count = torch.cuda.device_count()
        if cuda_count > local_rank:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            logging.getLogger("fsdp_train").warning(
                "LOCAL_RANK=%d is out of range for available CUDA devices (%d); falling back to default CUDA device.",
                local_rank,
                cuda_count,
            )
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return DistEnv(rank=rank, local_rank=local_rank, world_size=world_size, backend=backend, device=device)


def cleanup_distributed() -> None:
    """Destroy process group."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    """Synchronize ranks."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def is_main_process() -> bool:
    """Return True on rank 0 or non-distributed runs.

    Avoid calling `dist.get_rank()` unless the process group is initialized.
    """
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dtype_from_name(name: str) -> torch.dtype:
    """Map config dtype names to torch dtypes."""
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def resolve_load_dtype(cfg: Config) -> torch.dtype:
    """Prefer bf16 when supported, otherwise fall back to fp32."""
    if cfg.prefer_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def build_mixed_precision(cfg: Config) -> MixedPrecision:
    """Build an FSDP mixed-precision policy from config."""
    param_dtype = dtype_from_name(cfg.param_dtype)
    reduce_dtype = dtype_from_name(cfg.reduce_dtype)
    buffer_dtype = dtype_from_name(cfg.buffer_dtype)

    bf16_requested = any(d == torch.bfloat16 for d in (param_dtype, reduce_dtype, buffer_dtype))
    if bf16_requested and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        param_dtype = torch.float32
        reduce_dtype = torch.float32
        buffer_dtype = torch.float32

    return MixedPrecision(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        buffer_dtype=buffer_dtype,
    )


def load_model_and_tokenizer(cfg: Config) -> Tuple[torch.nn.Module, Any]:
    """Load the configured Hugging Face CausalLM model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path,
        use_fast=cfg.use_fast_tokenizer,
        trust_remote_code=cfg.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": cfg.trust_remote_code,
        "torch_dtype": resolve_load_dtype(cfg),
    }
    if cfg.attn_implementation:
        model_kwargs["attn_implementation"] = cfg.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, **model_kwargs)
    model.config.use_cache = cfg.use_cache
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def get_transformer_block_classes(model: torch.nn.Module) -> Tuple[type, ...]:
    """Detect common decoder block classes for transformer-based auto-wrap."""
    class_names = {
        "LlamaDecoderLayer",
        "MistralDecoderLayer",
        "Qwen2DecoderLayer",
        "GemmaDecoderLayer",
        "Phi3DecoderLayer",
    }
    classes: List[type] = []
    for module in model.modules():
        cls = module.__class__
        if cls.__name__ in class_names and cls not in classes:
            classes.append(cls)
    return tuple(classes)


def build_auto_wrap_policy(cfg: Config, model: torch.nn.Module) -> Callable[..., bool]:
    """Return the configured FSDP auto-wrap policy."""
    if cfg.auto_wrap_policy == "transformer_based":
        block_classes = get_transformer_block_classes(model)
        if not block_classes:
            raise RuntimeError(
                "Could not detect transformer block classes. "
                "Switch to auto_wrap_policy='size_based' if needed."
            )
        from functools import partial

        return partial(lambda_auto_wrap_policy, lambda_fn=lambda m: isinstance(m, block_classes))

    if cfg.auto_wrap_policy == "size_based":
        from functools import partial
        return partial(size_based_auto_wrap_policy, min_num_params=cfg.min_num_params)

    raise ValueError(f"Unsupported auto_wrap_policy: {cfg.auto_wrap_policy}")


def maybe_apply_activation_checkpointing(model: torch.nn.Module, cfg: Config) -> None:
    """Apply activation checkpointing to detected transformer blocks."""
    if not cfg.activation_checkpointing:
        return

    block_classes = get_transformer_block_classes(model)
    if not block_classes:
        return

    wrapper = lambda m: checkpoint_wrapper(m, checkpoint_impl=CheckpointImpl.NO_REENTRANT)  # noqa: E731

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=wrapper,
        check_fn=lambda m: isinstance(m, block_classes),
    )


def wrap_fsdp(model: torch.nn.Module, cfg: Config, device: torch.device) -> FSDP:
    """Wrap the model with Fully Sharded Data Parallel."""
    sharding = getattr(ShardingStrategy, cfg.sharding_strategy)
    device_id = device.index if device.type == "cuda" else None

    return FSDP(
        model,
        auto_wrap_policy=build_auto_wrap_policy(cfg, model),
        mixed_precision=build_mixed_precision(cfg),
        sharding_strategy=sharding,
        device_id=device_id,
        cpu_offload=CPUOffload(offload_params=cfg.cpu_offload),
        limit_all_gathers=cfg.limit_all_gathers,
        forward_prefetch=cfg.forward_prefetch,
        sync_module_states=cfg.sync_module_states,
        use_orig_params=cfg.use_orig_params,
    )


def format_ultrachat_example(example: Dict[str, Any]) -> str:
    """Convert an UltraChat sample into a simple plain-text conversation."""
    messages = example.get("messages", [])
    if not isinstance(messages, list):
        raise ValueError("Expected UltraChat example to contain 'messages'.")
    lines: List[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        lines.append(f"{role.capitalize()}: {content}")
    return "\n".join(lines)


def build_datasets(cfg: Config, tokenizer: Any) -> Tuple[Any, Optional[Any]]:
    """Load and tokenize the configured dataset splits."""
    dataset = load_dataset(cfg.dataset_name)

    def preprocess(example: Dict[str, Any]) -> Dict[str, Any]:
        text = format_ultrachat_example(example)
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=cfg.max_seq_length,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    train_ds = dataset[cfg.train_split].map(
        preprocess,
        remove_columns=dataset[cfg.train_split].column_names,
        num_proc=cfg.num_preprocessing_workers,
        desc="Tokenizing train split",
    )

    # Skip validation tokenization by default to keep smoke tests fast.
    val_ds = None
    if cfg.build_val_dataset:
        val_ds = dataset[cfg.val_split].map(
            preprocess,
            remove_columns=dataset[cfg.val_split].column_names,
            num_proc=cfg.num_preprocessing_workers,
            desc="Tokenizing val split",
        )

    if cfg.max_train_samples is not None:
        try:
            max_samples = int(cfg.max_train_samples)
        except Exception:
            raise ValueError("max_train_samples must be an integer")
        if max_samples <= 0:
            raise ValueError("max_train_samples must be a positive integer")
        n = min(max_samples, len(train_ds))
        train_ds = train_ds.select(range(n))

    return train_ds, val_ds


def build_dataloader(
    dataset: Any,
    batch_size: int,
    cfg: Config,
    rank: int,
    world_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create a distributed DataLoader for map-style datasets."""
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=False,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=default_data_collator,
        num_workers=cfg.dataloader_num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers if cfg.dataloader_num_workers > 0 else False,
    )


def get_rng_state() -> Dict[str, Any]:
    """Capture RNG state for resume."""
    state: Dict[str, Any] = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_random": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_random"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    """Restore RNG state."""
    random.setstate(state["python_random"])
    np.random.set_state(state["numpy_random"])
    torch.set_rng_state(state["torch_random"])
    if torch.cuda.is_available() and "torch_cuda_random" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda_random"])


def save_checkpoint(
    checkpoint_root: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    step: int,
    epoch: int,
    save_rng_state_flag: bool,
) -> str:
    """Save sharded FSDP checkpoint for resume."""
    ckpt_dir = Path(checkpoint_root) / f"step_{step:08d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    state = {"app": AppState(model, optimizer)}
    dcp.save(state_dict=state, checkpoint_id=str(ckpt_dir))

    if is_main_process():
        with (ckpt_dir / "trainer_state.json").open("w", encoding="utf-8") as f:
            json.dump({"step": step, "epoch": epoch}, f, indent=2)
        if save_rng_state_flag:
            torch.save(get_rng_state(), ckpt_dir / "rng_state.pt")

    barrier()
    return str(ckpt_dir)


def load_checkpoint(
    checkpoint_dir: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    load_rng_state_flag: bool,
) -> Dict[str, int]:
    """Load sharded FSDP checkpoint for training resume."""
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

    state = {"app": AppState(model, optimizer)}
    dcp.load(state_dict=state, checkpoint_id=str(ckpt_dir))

    trainer_state_path = ckpt_dir / "trainer_state.json"
    if not trainer_state_path.exists():
        raise FileNotFoundError(f"Missing trainer_state.json in {checkpoint_dir}")

    with trainer_state_path.open("r", encoding="utf-8") as f:
        trainer_state = json.load(f)

    rng_path = ckpt_dir / "rng_state.pt"
    if load_rng_state_flag and rng_path.exists():
        rng_state = torch.load(rng_path, map_location="cpu")
        set_rng_state(rng_state)

    barrier()
    return {"step": int(trainer_state["step"]), "epoch": int(trainer_state["epoch"])}


@torch.no_grad()
def evaluate_fixed_snippet(
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    text: str,
    max_length: int,
) -> Dict[str, float]:
    """Evaluate perplexity on a fixed text snippet."""
    model.eval()
    batch = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
    )
    loss = float(outputs.loss.detach().float().item())
    perplexity = float(math.exp(min(loss, 20.0)))
    model.train()
    return {"loss": loss, "perplexity": perplexity}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Self-contained FSDP trainer.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML/JSON config file.")
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    """Run distributed FSDP training."""
    args = parse_args()
    cfg = load_config(args.config)

    if args.model_name_or_path is not None:
        cfg.model_name_or_path = args.model_name_or_path
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.resume_from_checkpoint is not None:
        cfg.resume_from_checkpoint = args.resume_from_checkpoint

    dist_env = init_distributed()
    logger = setup_logging(dist_env.rank, cfg.output_dir)
    wandb_logger = WandbLogger(cfg.use_wandb, cfg, dist_env.rank)

    seed_everything(cfg.seed + dist_env.rank)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = cfg.allow_tf32
        torch.backends.cudnn.allow_tf32 = cfg.allow_tf32

    # Keep the unwrapped model on CPU; FSDP handles placement during wrapping.
    model, tokenizer = load_model_and_tokenizer(cfg)
    train_ds, _ = build_datasets(cfg, tokenizer)

    train_loader = build_dataloader(
        train_ds,
        cfg.per_device_train_batch_size,
        cfg,
        dist_env.rank,
        dist_env.world_size,
        shuffle=True,
    )

    maybe_apply_activation_checkpointing(model, cfg)
    fsdp_model = wrap_fsdp(model, cfg, dist_env.device)

    optimizer = torch.optim.AdamW(
        fsdp_model.parameters(),
        lr=cfg.learning_rate,
        betas=cfg.betas,
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )

    updates_per_epoch = max(1, math.ceil(len(train_loader) / cfg.gradient_accumulation_steps))
    total_steps = cfg.max_steps if cfg.max_steps > 0 else updates_per_epoch * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    start_step = 0
    start_epoch = 0
    if cfg.resume_from_checkpoint:
        resume_state = load_checkpoint(
            cfg.resume_from_checkpoint,
            fsdp_model,
            optimizer if cfg.save_optimizer else None,
            cfg.save_rng_state,
        )
        start_step = resume_state["step"]
        start_epoch = resume_state["epoch"]
        logger.info("Resumed from checkpoint: step=%d epoch=%d", start_step, start_epoch)

    logger.info(
        "Starting training | model=%s | backend=%s | world_size=%d | device=%s",
        cfg.model_name_or_path,
        dist_env.backend,
        dist_env.world_size,
        str(dist_env.device),
    )

    effective_global_batch = (
        cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps * dist_env.world_size
    )
    logger.info(
        "Batch sizes | per_device=%d | grad_accum=%d | world_size=%d | effective_global_batch=%d",
        cfg.per_device_train_batch_size,
        cfg.gradient_accumulation_steps,
        dist_env.world_size,
        effective_global_batch,
    )

    global_step = start_step
    optimizer.zero_grad(set_to_none=True)
    fsdp_model.train()

    # Start peak-memory tracking from a clean baseline for this run.
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats(dist_env.device)
        except Exception:
            torch.cuda.reset_peak_memory_stats()

    for epoch in range(start_epoch, cfg.epochs):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(train_loader):
            iter_start = time.perf_counter()

            input_ids = batch["input_ids"].to(dist_env.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(dist_env.device, non_blocking=True)
            labels = batch["labels"].to(dist_env.device, non_blocking=True)

            outputs = fsdp_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / cfg.gradient_accumulation_steps
            loss.backward()

            local_tokens = int(attention_mask.sum().item())
            is_last_batch = (batch_idx + 1) == len(train_loader)
            should_step = ((batch_idx + 1) % cfg.gradient_accumulation_steps) == 0 or is_last_batch

            if should_step:
                # Prefer FSDP grad clipping for sharded/flattened parameters.
                try:
                    clip_fn = getattr(fsdp_model, "clip_grad_norm_", None)
                    if callable(clip_fn):
                        clip_fn(cfg.max_grad_norm)
                    else:
                        clip_grad_norm_(fsdp_model.parameters(), cfg.max_grad_norm)
                except Exception:
                    clip_grad_norm_(fsdp_model.parameters(), cfg.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                elapsed = max(time.perf_counter() - iter_start, 1e-12)
                steps_per_sec = 1.0 / elapsed
                tokens_per_sec = (local_tokens * dist_env.world_size) / elapsed
                full_loss = float(loss.item() * cfg.gradient_accumulation_steps)

                peak_mem_mb = None
                if torch.cuda.is_available():
                    try:
                        peak_mem_mb = torch.cuda.max_memory_allocated(dist_env.device) / (1024 ** 2)
                    except Exception:
                        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

                if global_step % cfg.log_every_n_steps == 0 and is_main_process():
                    logger.info(
                        "step=%d loss=%.6f lr=%.8f steps_per_sec=%.3f tokens_per_sec=%.2f peak_mem_mb=%s",
                        global_step,
                        full_loss,
                        scheduler.get_last_lr()[0],
                        steps_per_sec,
                        tokens_per_sec,
                        f"{peak_mem_mb:.2f}" if peak_mem_mb is not None else "n/a",
                    )
                    wandb_logger.log(
                        {
                            "train/loss": full_loss,
                            "train/steps_per_sec": steps_per_sec,
                            "train/tokens_per_sec": tokens_per_sec,
                            "train/lr": scheduler.get_last_lr()[0],
                            **({"system/peak_gpu_mem_mb": peak_mem_mb} if peak_mem_mb is not None else {}),
                        },
                        step=global_step,
                    )

                    # Reset after each log window for interval-level peak stats.
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.reset_peak_memory_stats(dist_env.device)
                        except Exception:
                            torch.cuda.reset_peak_memory_stats()

                if global_step % cfg.eval_every_n_steps == 0:
                    metrics = evaluate_fixed_snippet(
                        fsdp_model,
                        tokenizer,
                        dist_env.device,
                        cfg.validation_text,
                        cfg.max_seq_length,
                    )
                    if is_main_process():
                        logger.info(
                            "validation step=%d val_loss=%.6f perplexity=%.4f",
                            global_step,
                            metrics["loss"],
                            metrics["perplexity"],
                        )
                        wandb_logger.log(
                            {
                                "val/loss": metrics["loss"],
                                "val/perplexity": metrics["perplexity"],
                            },
                            step=global_step,
                        )

                if global_step % cfg.save_every_n_steps == 0:
                    ckpt = save_checkpoint(
                        checkpoint_root=str(Path(cfg.output_dir) / "checkpoints"),
                        model=fsdp_model,
                        optimizer=optimizer if cfg.save_optimizer else None,
                        step=global_step,
                        epoch=epoch,
                        save_rng_state_flag=cfg.save_rng_state,
                    )
                    if is_main_process():
                        logger.info("Saved checkpoint: %s", ckpt)

                if cfg.max_steps > 0 and global_step >= cfg.max_steps:
                    break

        if cfg.max_steps > 0 and global_step >= cfg.max_steps:
            break

    final_ckpt = save_checkpoint(
        checkpoint_root=str(Path(cfg.output_dir) / "checkpoints"),
        model=fsdp_model,
        optimizer=optimizer if cfg.save_optimizer else None,
        step=global_step,
        epoch=epoch,
        save_rng_state_flag=cfg.save_rng_state,
    )

    if is_main_process():
        logger.info("Training complete. Final checkpoint: %s", final_ckpt)

    wandb_logger.finish()
    cleanup_distributed()


if __name__ == "__main__":
    main()