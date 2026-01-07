#!/usr/bin/env python3
import argparse
import glob
import os
import sys
from typing import Dict, Iterable, Optional, Set, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _normalize_checkpoint_dir(checkpoint_dir: str, tag: Optional[str]) -> Tuple[str, Optional[str]]:
    if tag:
        return checkpoint_dir, tag
    latest_path = os.path.join(checkpoint_dir, "latest")
    if os.path.isfile(latest_path):
        return checkpoint_dir, None
    model_states = glob.glob(os.path.join(checkpoint_dir, "*_model_states.pt"))
    if model_states:
        return os.path.dirname(checkpoint_dir), os.path.basename(checkpoint_dir)
    return checkpoint_dir, None


def _resolve_dtype(name: str, config_dtype: Optional[torch.dtype]) -> Optional[torch.dtype]:
    if name == "auto":
        if isinstance(config_dtype, str):
            config_dtype = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }.get(config_dtype, None)
        return config_dtype or torch.float32
    if name == "fp32":
        return torch.float32
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def _apply_prefix_strip(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    stripped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            stripped[key[len(prefix) :]] = value
        else:
            stripped[key] = value
    return stripped


def _count_matches(keys: Iterable[str], model_keys: Set[str]) -> int:
    return sum(1 for key in keys if key in model_keys)


def _remap_state_dict(state_dict: Dict[str, torch.Tensor], model_keys: Set[str]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    direct_matches = _count_matches(state_dict.keys(), model_keys)
    best_state = state_dict
    best_matches = direct_matches
    for prefix in ("module.", "model."):
        candidate = _apply_prefix_strip(state_dict, prefix)
        matches = _count_matches(candidate.keys(), model_keys)
        if matches > best_matches:
            best_matches = matches
            best_state = candidate
    return best_state


def _load_zero_checkpoint(checkpoint_dir: str, tag: Optional[str]) -> Dict[str, torch.Tensor]:
    try:
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    except ImportError as exc:
        raise RuntimeError("deepspeed is required for checkpoint conversion") from exc
    return get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir, tag)


def _print_key_summary(label: str, keys: Iterable[str], limit: int = 20) -> None:
    keys = list(keys)
    if not keys:
        return
    preview = ", ".join(keys[:limit])
    suffix = "..." if len(keys) > limit else ""
    print(f"[warn] {label} ({len(keys)}): {preview}{suffix}")


def _merge_lora_state_dict(
    state_dict: Dict[str, torch.Tensor],
    base_model: str,
    args: argparse.Namespace,
    dtype: Optional[torch.dtype],
) -> AutoModelForCausalLM:
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError("peft is required for --merge-lora") from exc

    targets = [name.strip() for name in args.lora_targets.split(",") if name.strip()]
    lora_cfg = LoraConfig(
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        target_modules=targets,
        task_type="CAUSAL_LM",
    )
    load_kwargs = {"trust_remote_code": args.trust_remote_code}
    if dtype is not None:
        load_kwargs["torch_dtype"] = dtype
    base = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
    peft_model = get_peft_model(base, lora_cfg)
    remapped = _remap_state_dict(state_dict, set(peft_model.state_dict().keys()))
    missing, unexpected = peft_model.load_state_dict(remapped, strict=False)
    _print_key_summary("missing keys", missing)
    _print_key_summary("unexpected keys", unexpected)
    try:
        merged = peft_model.merge_and_unload()
    except Exception as exc:
        raise RuntimeError(f"failed to merge LoRA weights: {exc}") from exc
    return merged.cpu()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a DeepSpeed ZeRO checkpoint to a Hugging Face checkpoint."
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="DeepSpeed checkpoint root (contains tag folders and latest).",
    )
    parser.add_argument("--tag", default=None, help="Checkpoint tag (e.g., final, global_step1).")
    parser.add_argument(
        "--base-model",
        required=True,
        help="Base model path or name for config/tokenizer (e.g., model/Qwen3-4B-Base).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for Hugging Face checkpoint.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "fp32", "bf16", "fp16"),
        default="auto",
        help="Target dtype for the saved weights.",
    )
    parser.add_argument(
        "--max-shard-size",
        default="5GB",
        help="Max shard size passed to save_pretrained (e.g., 5GB).",
    )
    parser.add_argument(
        "--safe-serialization",
        action="store_true",
        help="Save weights as safetensors if available.",
    )
    parser.add_argument(
        "--merge-lora",
        action="store_true",
        help="Merge LoRA weights into the base model before saving.",
    )
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-targets",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated list of module names for LoRA injection.",
    )
    parser.add_argument(
        "--skip-tokenizer",
        action="store_true",
        help="Skip saving tokenizer files.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model code when loading config/tokenizer.",
    )
    args = parser.parse_args()

    checkpoint_dir, tag = _normalize_checkpoint_dir(args.checkpoint_dir, args.tag)

    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    dtype = _resolve_dtype(args.dtype, getattr(config, "torch_dtype", None))

    print(f"[info] loading DeepSpeed checkpoint from {checkpoint_dir} tag={tag or 'latest'}")
    state_dict = _load_zero_checkpoint(checkpoint_dir, tag)
    if args.merge_lora:
        model = _merge_lora_state_dict(state_dict, args.base_model, args, dtype)
    else:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=args.trust_remote_code)
        if dtype is not None:
            model = model.to(dtype=dtype)
        model = model.cpu()
        state_dict = _remap_state_dict(state_dict, set(model.state_dict().keys()))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        _print_key_summary("missing keys", missing)
        _print_key_summary("unexpected keys", unexpected)

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(
        args.output_dir,
        safe_serialization=args.safe_serialization,
        max_shard_size=args.max_shard_size,
    )
    if not args.skip_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model, trust_remote_code=args.trust_remote_code
        )
        tokenizer.save_pretrained(args.output_dir)

    print(f"[done] saved Hugging Face checkpoint to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
