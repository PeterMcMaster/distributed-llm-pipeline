import os
import argparse

import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--export_dir", type=str, required=True)
    parser.add_argument("--repo_id", type=str, default=None)
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    hf_token = os.environ.get("HF_TOKEN", None)
    if hf_token is None:
        raise ValueError("HF_TOKEN is not set.")

    os.makedirs(args.export_dir, exist_ok=True)

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    print(f"Loading checkpoint from: {args.checkpoint_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_dir,
        token=hf_token,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_dir,
        token=hf_token,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Saving clean export to: {args.export_dir}")
    model.save_pretrained(
        args.export_dir,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    tokenizer.save_pretrained(args.export_dir)

    print("Local export complete.")

    if args.repo_id:
        print(f"Pushing to Hugging Face Hub: {args.repo_id}")
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True)
        api.upload_folder(
            repo_id=args.repo_id,
            folder_path=args.export_dir,
            commit_message="Upload ZeRO-2 fine-tuned Llama 3.2 1B export for vLLM",
        )
        print("Push complete.")


if __name__ == "__main__":
    main()
