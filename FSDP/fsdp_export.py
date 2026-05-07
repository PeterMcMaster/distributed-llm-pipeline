from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.fsdp import (
    CPUOffload,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy, size_based_auto_wrap_policy
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import yaml
except Exception:
    yaml = None


@dataclass
class Config:
    model_name_or_path: str = "meta-llama/Llama-3.2-1B"
    trust_remote_code: bool = False
    output_dir: str = "./fsdp_outputs"

    sharding_strategy: str = "FULL_SHARD"
    use_orig_params: bool = True
    limit_all_gathers: bool = True
    forward_prefetch: bool = False
    sync_module_states: bool = True
    cpu_offload: bool = False
    auto_wrap_policy: str = "transformer_based"
    min_num_params: int = 1_000_000

    prefer_bf16: bool = True
    param_dtype: str = "bf16"
    reduce_dtype: str = "bf16"
    buffer_dtype: str = "bf16"

    use_fast_tokenizer: bool = True
    attn_implementation: str = "sdpa"
    use_cache: bool = False

    export_dtype: str = "bf16"
    safe_serialization: bool = True


@dataclass
class DistEnv:
    rank: int
    local_rank: int
    world_size: int
    backend: str
    device: torch.device


class AppState(Stateful):
    """Stateful wrapper for distributed checkpoint loading."""

    def __init__(self, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer]) -> None:
        self.model = model
        self.optimizer = optimizer

    def state_dict(self) -> Dict[str, Any]:
        optimizers = [] if self.optimizer is None else self.optimizer
        model_state, optim_state = get_state_dict(self.model, optimizers)
        return {"model": model_state, "optim": optim_state}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
        )


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
    logger = logging.getLogger("fsdp_export")
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
        file_handler = logging.FileHandler(Path(output_dir) / "export.log", encoding="utf-8")
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
            logging.getLogger("fsdp_export").warning(
                "LOCAL_RANK=%d is out of range for %d CUDA devices; falling back to default CUDA device.",
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
    """Return True on rank 0 or non-distributed execution.

    Avoid calling `dist.get_rank()` unless the process group is initialized.
    """
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


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

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": cfg.trust_remote_code,
        "torch_dtype": resolve_load_dtype(cfg),
    }
    if cfg.attn_implementation:
        model_kwargs["attn_implementation"] = cfg.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, **model_kwargs)
    model.config.use_cache = cfg.use_cache
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


def wrap_fsdp(model: torch.nn.Module, cfg: Config, device: torch.device) -> FSDP:
    """Wrap model in FSDP to reconstruct checkpoint structure."""
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


def load_checkpoint(checkpoint_dir: str, model: torch.nn.Module) -> None:
    """Load sharded FSDP checkpoint."""
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

    from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict
    
    model_state = get_model_state_dict(model)
    state = {"model": model_state}
    
    dcp.load(state_dict=state, checkpoint_id=str(ckpt_dir))
    
    set_model_state_dict(
        model,
        model_state_dict=state["model"],
    )


def export_dtype_to_torch(dtype_name: str) -> torch.dtype:
    """Map export dtype string to torch dtype."""
    mapping = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported export dtype: {dtype_name}")
    return mapping[dtype_name]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Standalone FSDP export utility.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML/JSON config file.")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--export_dir", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--export_dtype", type=str, default=None, choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--safe_serialization", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    """Export a sharded FSDP checkpoint to Hugging Face format."""
    args = parse_args()
    cfg = load_config(args.config)

    if args.model_name_or_path is not None:
        cfg.model_name_or_path = args.model_name_or_path
    if args.export_dtype is not None:
        cfg.export_dtype = args.export_dtype
    if args.safe_serialization is not None:
        cfg.safe_serialization = str(args.safe_serialization).lower() in {"1", "true", "yes", "y"}

    dist_env = init_distributed()
    logger = setup_logging(dist_env.rank, args.export_dir)

    try:
        # Keep the unwrapped model on CPU; FSDP handles device placement.
        model, tokenizer = load_model_and_tokenizer(cfg)
        fsdp_model = wrap_fsdp(model, cfg, dist_env.device)

        load_checkpoint(args.checkpoint_dir, fsdp_model)

        full_state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        # Materialize a rank-0 full state dict on CPU for HF export.
        with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, full_state_cfg):
            full_state_dict = fsdp_model.state_dict()

        # Synchronize before entering rank-0-only export work.
        barrier()

        if is_main_process():
            export_dir = Path(args.export_dir)
            export_dir.mkdir(parents=True, exist_ok=True)

            export_dtype = export_dtype_to_torch(cfg.export_dtype)

            # Convert floating tensors to export dtype and enforce contiguous CPU tensors.
            processed_state: Dict[str, torch.Tensor] = {}
            for k, v in full_state_dict.items():
                t = v.cpu()
                if torch.is_floating_point(t):
                    t = t.to(dtype=export_dtype)
                processed_state[k] = t.contiguous()

            base_model = AutoModelForCausalLM.from_config(model.config)

            def strict_load_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[List[str], List[str]]:
                return base_model.load_state_dict(state_dict, strict=True)

            missing, unexpected = [], []
            try:
                missing, unexpected = strict_load_state_dict(processed_state)
            except RuntimeError as err:
                logger.warning("Strict load_state_dict failed: %s", str(err))

                # Retry with prefix stripping only when every key shares the same prefix.
                proc_keys = list(processed_state.keys())
                if proc_keys:
                    first = proc_keys[0]
                    if "." in first:
                        candidate = first.split(".", 1)[0] + "."
                        if all(k.startswith(candidate) for k in proc_keys):
                            stripped = {k[len(candidate) :]: v for k, v in processed_state.items()}
                            logger.info(
                                "Attempting to strip common prefix '%s' from state-dict keys and retry strict load.",
                                candidate,
                            )
                            try:
                                missing, unexpected = strict_load_state_dict(stripped)
                                processed_state = stripped
                            except RuntimeError as err2:
                                raise RuntimeError(
                                    "Pre-save validation failed after prefix stripping. "
                                    f"Original error: {err}; Retry error: {err2}"
                                )

            if missing or unexpected:
                raise RuntimeError(
                    f"Pre-save validation failed. missing={missing}, unexpected={unexpected}"
                )

            base_model.to("cpu")
            base_model.save_pretrained(
                export_dir,
                state_dict=processed_state,
                safe_serialization=cfg.safe_serialization,
            )
            tokenizer.save_pretrained(export_dir)

            if getattr(model, "generation_config", None) is not None:
                model.generation_config.save_pretrained(export_dir)

            reload_ok = False
            try:
                _ = AutoModelForCausalLM.from_pretrained(export_dir, torch_dtype=export_dtype)
                _ = AutoTokenizer.from_pretrained(export_dir)
                reload_ok = True
            except Exception as err:
                logger.exception("Transformers reload validation failed: %s", str(err))
                raise

            weights_format = "safetensors" if cfg.safe_serialization else "pytorch_bin"

            logger.info(
                "Export complete | dir=%s | dtype=%s | safe_serialization=%s | weights_format=%s | reload_ok=%s",
                str(export_dir),
                cfg.export_dtype,
                cfg.safe_serialization,
                weights_format,
                reload_ok,
            )

        # Keep ranks aligned before distributed cleanup.
        barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()