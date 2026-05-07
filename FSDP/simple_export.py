from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a distributed checkpoint to Hugging Face format.")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--export_dir", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return mapping[name]


def main() -> None:
    args = parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    export_dir = Path(args.export_dir)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

    torch_dtype = dtype_from_name(args.dtype)

    print(f"Loading tokenizer from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model from {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="cpu",
    )
    model.config.use_cache = False

    print(f"Loading checkpoint from {checkpoint_dir}")
    state = {"model": model.state_dict()}
    dcp.load(state_dict=state, checkpoint_id=str(checkpoint_dir))

    print("Applying weights")
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"State dict mismatch. missing={missing}, unexpected={unexpected}")

    print(f"Saving export to {export_dir}")
    export_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(export_dir, safe_serialization=True)
    tokenizer.save_pretrained(export_dir)

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(export_dir)

    print("Verifying export")
    _ = AutoModelForCausalLM.from_pretrained(export_dir, torch_dtype=torch_dtype)
    _ = AutoTokenizer.from_pretrained(export_dir)

    print("Export complete")


if __name__ == "__main__":
    main()