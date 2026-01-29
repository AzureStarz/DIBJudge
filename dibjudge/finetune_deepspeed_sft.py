from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import time
import warnings
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from .data import DIBJudgeDataset, DIBJudgeExample
from .swanlab_utils import finish_swanlab, init_swanlab, log_swanlab


def _get_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def init_distributed() -> Tuple[int, int, int]:
    rank = _get_env_int("RANK", _get_env_int("SLURM_PROCID", 0))
    world_size = _get_env_int("WORLD_SIZE", _get_env_int("SLURM_NTASKS", 1))
    local_rank = _get_env_int("LOCAL_RANK", _get_env_int("SLURM_LOCALID", 0))
    return rank, world_size, local_rank


def _rank0_print(rank: int, *args: object, **kwargs: object) -> None:
    if rank == 0:
        print(*args, **kwargs)


def _maybe_resize_embeddings(
    model: torch.nn.Module, tokenizer, name: str, rank: int
) -> None:
    if model is None or tokenizer is None:
        return
    embed = model.get_input_embeddings()
    if embed is None:
        return
    vocab_size = int(embed.weight.size(0))
    tok_size = len(tokenizer)
    if tok_size > vocab_size:
        model.resize_token_embeddings(tok_size)
        _rank0_print(rank, f"[stage done] resized {name} embeddings {vocab_size} -> {tok_size}")


def _load_causal_lm(
    name: str,
    attn_implementation: Optional[str],
    local_files_only: bool = False,
    torch_dtype: Optional[torch.dtype] = None,
) -> torch.nn.Module:
    if not attn_implementation:
        return AutoModelForCausalLM.from_pretrained(
            name, local_files_only=local_files_only, torch_dtype=torch_dtype
        )
    try:
        return AutoModelForCausalLM.from_pretrained(
            name,
            attn_implementation=attn_implementation,
            local_files_only=local_files_only,
            torch_dtype=torch_dtype,
        )
    except TypeError:
        warnings.warn(
            f"attn_implementation={attn_implementation} unsupported for {name}; "
            "falling back to default attention.",
            RuntimeWarning,
        )
    except ImportError as exc:
        warnings.warn(
            f"attn_implementation={attn_implementation} unavailable for {name}: {exc}. "
            "Falling back to default attention.",
            RuntimeWarning,
        )
    return AutoModelForCausalLM.from_pretrained(
        name, local_files_only=local_files_only
    )


def _maybe_tqdm(iterable, rank: int, desc: str):
    if rank != 0:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    total = len(iterable) if hasattr(iterable, "__len__") else None
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)


def _load_ds_config(config: object) -> object:
    if isinstance(config, str) and os.path.isfile(config):
        with open(config, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return config


def _configure_deepspeed_activation_checkpointing(
    ds_config: object, rank: int
) -> Tuple[Optional[object], bool]:
    if not isinstance(ds_config, dict):
        return None, True
    cfg = ds_config.get("activation_checkpointing")
    if not cfg:
        return None, True
    try:
        import deepspeed.checkpointing as ds_checkpoint
    except Exception as exc:
        _rank0_print(rank, f"[warn] deepspeed.checkpointing unavailable: {exc}")
        return None, True
    if hasattr(ds_checkpoint, "is_configured"):
        try:
            if ds_checkpoint.is_configured():
                return ds_checkpoint.checkpoint, False
        except Exception:
            pass
    cfg = cfg if isinstance(cfg, dict) else {}
    kwargs = {
        "partition_activations": bool(cfg.get("partition_activations", False)),
        "cpu_checkpointing": bool(cfg.get("cpu_checkpointing", False)),
        "contiguous_memory_optimization": bool(
            cfg.get("contiguous_memory_optimization", False)
        ),
        "synchronize_checkpoint_boundary": bool(
            cfg.get("synchronize_checkpoint_boundary", False)
        ),
    }
    if "profile" in cfg:
        kwargs["profile"] = bool(cfg["profile"])
    try:
        sig = inspect.signature(ds_checkpoint.configure)
        allowed = set(sig.parameters.keys())
        kwargs = {key: value for key, value in kwargs.items() if key in allowed}
        ds_checkpoint.configure(**kwargs)
        _rank0_print(rank, "[stage done] deepspeed activation checkpointing configured")
    except Exception as exc:
        _rank0_print(rank, f"[warn] failed to configure deepspeed checkpointing: {exc}")
        return None, True
    return ds_checkpoint.checkpoint, False




def _move_batch_to_device(
    batch: Dict[str, object],
    device: torch.device,
    keys: Optional[Iterable[str]] = None,
) -> Dict[str, object]:
    """Move only required tensors to GPU to reduce peak memory."""
    key_set = set(keys) if keys is not None else None
    moved: Dict[str, object] = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and (key_set is None or key in key_set):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def _warn_if_zero3_offload_inactive(
    engine, ds_config: object, rank: int
) -> None:
    zero_stage_cfg = _resolve_zero_stage(ds_config)
    if zero_stage_cfg != 3:
        return
    expected_offload = _resolve_zero_offload(ds_config)
    stage_runtime = getattr(engine, "zero_optimization_stage", None)
    if callable(stage_runtime):
        try:
            stage_runtime = stage_runtime()
        except Exception:
            stage_runtime = None
    if stage_runtime is None:
        stage_runtime = getattr(engine, "zero_optimization_stage", None)
    if stage_runtime is not None:
        try:
            if int(stage_runtime) != 3:
                _rank0_print(
                    rank,
                    f"[warn] DeepSpeed reports zero_stage={stage_runtime}; expected 3",
                )
        except (TypeError, ValueError):
            pass
    total_elems = 0
    cuda_elems = 0
    total_bytes = 0
    cuda_bytes = 0
    for param in engine.module.parameters():
        numel = param.numel()
        total_elems += numel
        total_bytes += numel * param.element_size()
        if param.is_cuda:
            cuda_elems += numel
            cuda_bytes += numel * param.element_size()
    if total_elems == 0:
        return
    frac = cuda_elems / max(1, total_elems)
    if expected_offload and cuda_bytes > 0:
        if frac >= 0.05 or cuda_bytes >= 200 * 1024 * 1024:
            _rank0_print(
                rank,
                "[warn] ZeRO-3 offload expected but a large share of parameters remain on GPU "
                f"({cuda_bytes / (1024**2):.1f} MB, {frac:.1%}).",
            )
        else:
            _rank0_print(
                rank,
                "[warn] ZeRO-3 offload expected but some parameters remain on GPU "
                f"({cuda_bytes / (1024**2):.1f} MB, {frac:.1%}).",
            )


def _resolve_torch_dtype(value: Optional[object]) -> Optional[torch.dtype]:
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        name = value.strip().lower()
        if not name:
            return None
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        return mapping.get(name)
    return None


def _preview_text(text: str, limit: int = 120) -> str:
    if not text:
        return ""
    text = text.replace("\n", "\\n")
    return text[:limit]


def _prefilter_long_prompts(
    dataset: DIBJudgeDataset,
    tokenizer,
    max_lm_len: int,
    rank: int,
    log_samples: int = 3,
    preview_len: int = 120,
) -> DIBJudgeDataset:
    if max_lm_len <= 0 or len(dataset) == 0:
        return dataset
    max_check_len = max_lm_len + 1
    kept: List[DIBJudgeExample] = []
    dropped: List[Tuple[int, int, str]] = []
    for idx, ex in enumerate(dataset):
        prompt = ex.judge_prompt or ""
        enc = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_check_len,
        )
        prompt_len = len(enc.get("input_ids", []))
        if prompt_len >= max_lm_len:
            dropped.append((idx, prompt_len, prompt))
        else:
            kept.append(ex)
    if rank == 0:
        total = len(kept) + len(dropped)
        ratio = (len(dropped) / total) if total else 0.0
        _rank0_print(
            rank,
            "[data] prefilter_long_prompts dropped "
            f"{len(dropped)}/{total} ({ratio:.2%}) with max_lm_len={max_lm_len}",
        )
        log_samples = max(0, int(log_samples))
        for idx, (orig_idx, prompt_len, prompt) in enumerate(dropped[:log_samples]):
            preview = _preview_text(prompt, limit=preview_len)
            _rank0_print(
                rank,
                "[data] prefilter_long_prompts sample "
                f"{idx + 1}: idx={orig_idx} prompt_tokens={prompt_len} "
                f"prompt='{preview}'",
            )
    return DIBJudgeDataset(kept)


class CheckpointedCausalLM(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        use_reentrant: bool = True,
        checkpoint_fn=torch_checkpoint,
        *,
        supports_reentrant: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.use_reentrant = bool(use_reentrant)
        self.checkpoint_fn = checkpoint_fn
        self.checkpoint_supports_reentrant = bool(supports_reentrant)

    def set_activation_checkpointing(
        self, checkpoint_fn, *, supports_reentrant: bool = True
    ) -> None:
        self.checkpoint_fn = checkpoint_fn
        self.checkpoint_supports_reentrant = bool(supports_reentrant)

    def _checkpoint_call(self, fn, *args):
        if not self.checkpoint_supports_reentrant:
            return self.checkpoint_fn(fn, *args)
        return self.checkpoint_fn(fn, *args, use_reentrant=self.use_reentrant)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if not self.training or not torch.is_grad_enabled():
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )
        if input_ids is None:
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )
        embed_layer = self.model.get_input_embeddings()
        if embed_layer is None:
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )
        inputs_embeds = embed_layer(input_ids)
        if not inputs_embeds.requires_grad:
            inputs_embeds = inputs_embeds.detach().requires_grad_(True)
        if labels is None:
            def _run_logits(lm_embeds, lm_attention):
                out = self.model(
                    inputs_embeds=lm_embeds,
                    attention_mask=lm_attention,
                    labels=None,
                    use_cache=False,
                    **kwargs,
                )
                return out.logits

            logits = self._checkpoint_call(
                _run_logits,
                inputs_embeds,
                attention_mask,
            )
            return {"loss": None, "logits": logits}

        def _run(lm_embeds, lm_attention, lm_labels):
            out = self.model(
                inputs_embeds=lm_embeds,
                attention_mask=lm_attention,
                labels=lm_labels,
                use_cache=False,
                **kwargs,
            )
            return out.loss, out.logits

        loss, logits = self._checkpoint_call(
            _run,
            inputs_embeds,
            attention_mask,
            labels,
        )
        return {"loss": loss, "logits": logits}

    def __getattr__(self, name: str):
        if name in {"model", "use_reentrant"}:
            return super().__getattr__(name)
        return getattr(self.model, name)


def _load_yaml_config(path: str, parser: argparse.ArgumentParser) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("pyyaml is required for --config/--save-config") from exc
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a mapping of arg names to values.")
    valid = {action.dest for action in parser._actions}
    filtered = {key: value for key, value in data.items() if key in valid}
    unknown = sorted(set(data) - valid)
    if unknown:
        warnings.warn(f"Ignoring unknown config keys: {', '.join(unknown)}")
    return filtered


def _save_yaml_config(args: argparse.Namespace, path: str) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("pyyaml is required for --config/--save-config") from exc
    if not path:
        return
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(vars(args), handle, sort_keys=True)


def _save_deepspeed_checkpoint(
    engine: "deepspeed.DeepSpeedEngine",
    output_dir: str,
    tag: str,
    rank: int,
) -> None:
    if not output_dir:
        return
    os.makedirs(output_dir, exist_ok=True)
    if rank == 0:
        print(f"[checkpoint] saving to {output_dir} tag={tag}", flush=True)
    engine.save_checkpoint(output_dir, tag=tag)


def _normalize_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        normalized[key] = value.cpu()
    return normalized


def _strip_prefix_if_present(
    state_dict: Dict[str, torch.Tensor], prefix: str
) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix) :]: value for key, value in state_dict.items()}
    return state_dict


def _merge_lora_into_lm_state(
    lm_state: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    rank: int,
) -> Optional[Dict[str, torch.Tensor]]:
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        _rank0_print(rank, "[warn] peft is required to merge LoRA weights.")
        return None
    try:
        base = _load_causal_lm(
            args.lm,
            args.attn_implementation,
            local_files_only=True,
            torch_dtype=_resolve_torch_dtype(args.torch_dtype),
        )
    except Exception as exc:
        _rank0_print(rank, f"[warn] failed to load base model for LoRA merge: {exc}")
        return None

    targets = [name.strip() for name in args.lora_targets.split(",") if name.strip()]
    lora_cfg = LoraConfig(
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        target_modules=targets,
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base, lora_cfg)
    missing, unexpected = peft_model.load_state_dict(lm_state, strict=False)
    if missing:
        _rank0_print(rank, f"[warn] missing keys while loading LoRA LM: {len(missing)}")
    if unexpected:
        _rank0_print(rank, f"[warn] unexpected keys while loading LoRA LM: {len(unexpected)}")
    try:
        merged = peft_model.merge_and_unload()
    except Exception as exc:
        _rank0_print(rank, f"[warn] failed to merge LoRA weights: {exc}")
        return None
    return {key: value.cpu() for key, value in merged.state_dict().items()}


def _save_lm_assets(output_dir: str, args: argparse.Namespace, rank: int) -> None:
    try:
        lm_tok = AutoTokenizer.from_pretrained(
            args.lm, local_files_only=True, use_fast=True
        )
        lm_tok.save_pretrained(output_dir)
    except Exception as exc:
        _rank0_print(rank, f"[warn] failed to save LM tokenizer: {exc}")
    try:
        from transformers import GenerationConfig
    except ImportError:
        return
    try:
        gen_cfg = GenerationConfig.from_pretrained(args.lm, local_files_only=True)
        gen_cfg.save_pretrained(output_dir)
    except Exception as exc:
        _rank0_print(rank, f"[warn] failed to save generation config: {exc}")


def _save_hf_checkpoint(
    checkpoint_dir: str,
    tag: str,
    args: argparse.Namespace,
    rank: int,
) -> None:
    output_dir = os.path.join(checkpoint_dir, f"hf-{tag}")
    if rank != 0:
        return
    if not output_dir:
        return
    try:
        from deepspeed.utils.zero_to_fp32 import (
            get_fp32_state_dict_from_zero_checkpoint,
        )
    except ImportError:
        _rank0_print(rank, "[warn] deepspeed is required to export HF checkpoints.")
        return

    os.makedirs(output_dir, exist_ok=True)
    _rank0_print(rank, f"[hf] extracting LM from {checkpoint_dir} tag={tag}")
    raw_state = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir, tag)
    state_dict = _normalize_state_dict(raw_state)
    state_dict = _strip_prefix_if_present(state_dict, "model.")

    if args.use_lora:
        merged = _merge_lora_into_lm_state(state_dict, args, rank)
        if merged is not None:
            state_dict = merged

    state_dict = _strip_prefix_if_present(state_dict, "base_model.model.")
    state_dict = _strip_prefix_if_present(state_dict, "base_model.")
    try:
        model = _load_causal_lm(
            args.lm,
            args.attn_implementation,
            local_files_only=True,
            torch_dtype=_resolve_torch_dtype(args.torch_dtype),
        )
    except Exception as exc:
        _rank0_print(rank, f"[warn] failed to load base model for HF export: {exc}")
        return
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        _rank0_print(rank, f"[warn] missing keys while loading LM: {len(missing)}")
    if unexpected:
        _rank0_print(rank, f"[warn] unexpected keys while loading LM: {len(unexpected)}")
    model.save_pretrained(output_dir)
    _save_lm_assets(output_dir, args, rank)


def _maybe_apply_lora(model: torch.nn.Module, args: argparse.Namespace, rank: int) -> torch.nn.Module:
    if not args.use_lora:
        return model
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError("peft is required for --use-lora") from exc

    target = [name.strip() for name in args.lora_targets.split(",") if name.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target,
    )
    model = get_peft_model(model, lora_cfg)
    _rank0_print(rank, "[stage done] lora applied")
    return model


def _group_weight_decay(params: Dict[str, torch.nn.Parameter], weight_decay: float, lr: float):
    decay: List[torch.nn.Parameter] = []
    no_decay: List[torch.nn.Parameter] = []
    for name, param in params.items():
        if param.ndim == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay, "lr": lr},
        {"params": no_decay, "weight_decay": 0.0, "lr": lr},
    ]


def _build_optimizer_params(
    model: torch.nn.Module,
    lm_lr: float,
    lora_lr: float,
    weight_decay: float,
    lora_weight_decay: float,
):
    groups: Dict[str, Dict[str, torch.nn.Parameter]] = {"lm": {}, "lora": {}}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name:
            groups["lora"][name] = param
        else:
            groups["lm"][name] = param
    param_groups: List[Dict[str, object]] = []
    if groups["lm"]:
        param_groups.extend(_group_weight_decay(groups["lm"], weight_decay, lm_lr))
    if groups["lora"]:
        param_groups.extend(_group_weight_decay(groups["lora"], lora_weight_decay, lora_lr))
    if not param_groups:
        raise ValueError("No trainable parameters found for optimizer.")
    return param_groups


def _zero_offload_enabled(ds_config: object) -> bool:
    if not isinstance(ds_config, dict):
        return False
    zero_cfg = ds_config.get("zero_optimization")
    if not isinstance(zero_cfg, dict):
        return False
    offload = zero_cfg.get("offload_optimizer")
    if not offload:
        return False
    if isinstance(offload, dict):
        device = str(offload.get("device", "")).lower()
        return bool(device) and device not in {"none", "false", "off"}
    if isinstance(offload, str):
        return offload.lower() not in {"none", "false", "off"}
    return True


def _resolve_zero_stage(config: object) -> Optional[int]:
    if not isinstance(config, dict):
        return None
    zero_opt = config.get("zero_optimization")
    if isinstance(zero_opt, dict):
        stage = zero_opt.get("stage")
    elif isinstance(zero_opt, int):
        stage = zero_opt
    else:
        return None
    try:
        return int(stage)
    except (TypeError, ValueError):
        return None


def _resolve_zero_offload(config: object) -> bool:
    if not isinstance(config, dict):
        return False
    zero_opt = config.get("zero_optimization")
    if not isinstance(zero_opt, dict):
        return False
    for key in ("offload_optimizer", "offload_param"):
        offload = zero_opt.get(key)
        if not offload:
            continue
        if isinstance(offload, dict):
            device = str(offload.get("device", "")).lower()
            if device and device not in {"none", "false", "off"}:
                return True
        elif isinstance(offload, str):
            if offload.lower() not in {"none", "false", "off"}:
                return True
        else:
            return True
    return False


def _warmup_steps_from_ratio(total_steps: int, ratio: Optional[float], warmup_steps: int) -> int:
    if ratio is None:
        return warmup_steps
    if ratio < 0:
        raise ValueError("warmup_ratio must be >= 0.")
    return int(total_steps * ratio)


class SFTCollator:
    def __init__(
        self,
        tokenizer,
        max_length: Optional[int],
        append_eos: bool,
        filter_truncated: bool = False,
        min_target_tokens: int = 0,
        drop_truncated: bool = False,
        drop_min_target_tokens: int = 0,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length if max_length and max_length > 0 else None
        self.append_eos = append_eos
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id or 0
        self.padding_side = getattr(self.tokenizer, "padding_side", "right")
        if self.padding_side not in {"left", "right"}:
            self.padding_side = "right"
        self.eos_id = tokenizer.eos_token_id or self.pad_id
        self.filter_truncated = bool(filter_truncated)
        self.min_target_tokens = max(0, int(min_target_tokens))
        self.drop_truncated = bool(drop_truncated)
        self.drop_min_target_tokens = max(0, int(drop_min_target_tokens))

    def _encode_pair(
        self, ex: DIBJudgeExample
    ) -> Tuple[List[int], List[int], bool, int]:
        prompt = ex.judge_prompt or ""
        output = ex.output or ""
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        output_ids = self.tokenizer.encode(output, add_special_tokens=False)
        prompt_len = len(prompt_ids)
        if self.append_eos and (not output_ids or output_ids[-1] != self.eos_id):
            output_ids.append(self.eos_id)
        prompt_overflow = (
            self.max_length is not None and prompt_len >= self.max_length
        )
        if self.max_length is not None:
            total_len = len(prompt_ids) + len(output_ids)
            if total_len > self.max_length:
                overflow = total_len - self.max_length
                if overflow >= len(prompt_ids):
                    drop = overflow - len(prompt_ids)
                    prompt_ids = []
                    if drop > 0:
                        output_ids = output_ids[drop:]
                else:
                    prompt_ids = prompt_ids[overflow:]
        input_ids = prompt_ids + output_ids
        labels = [-100] * len(prompt_ids) + output_ids
        if not input_ids:
            input_ids = [self.pad_id]
            labels = [-100]
        return input_ids, labels, prompt_overflow, len(output_ids)

    def __call__(self, batch: List[DIBJudgeExample]) -> Dict[str, torch.Tensor]:
        sequences: List[List[int]] = []
        labels: List[List[int]] = []
        prompt_overflow: List[bool] = []
        target_lengths: List[int] = []
        for ex in batch:
            seq, lab, overflow, target_len = self._encode_pair(ex)
            sequences.append(seq)
            labels.append(lab)
            prompt_overflow.append(overflow)
            target_lengths.append(target_len)
        drop_count = 0
        drop_seen = len(sequences)
        drop_maxlen_count = 0
        drop_min_target_count = 0
        drop_all_fallback = False
        if self.drop_truncated or self.drop_min_target_tokens > 0:
            drop = []
            for idx in range(drop_seen):
                drop_max = self.drop_truncated and prompt_overflow[idx]
                drop_min = (
                    self.drop_min_target_tokens > 0
                    and target_lengths[idx] < self.drop_min_target_tokens
                )
                drop_maxlen_count += int(drop_max)
                drop_min_target_count += int(drop_min)
                drop.append(drop_max or drop_min)
            drop_count = sum(1 for flag in drop if flag)
            if drop_count:
                keep = [not flag for flag in drop]
                sequences = [seq for seq, keep_item in zip(sequences, keep) if keep_item]
                labels = [lab for lab, keep_item in zip(labels, keep) if keep_item]
                prompt_overflow = [
                    ov for ov, keep_item in zip(prompt_overflow, keep) if keep_item
                ]
                target_lengths = [
                    tl for tl, keep_item in zip(target_lengths, keep) if keep_item
                ]
                if not sequences:
                    drop_all_fallback = True
                    sequences = [[self.pad_id]]
                    labels = [[-100]]
                    prompt_overflow = [False]
                    target_lengths = [0]
        if self.filter_truncated or self.min_target_tokens > 0:
            for idx in range(len(labels)):
                filter_flag = False
                if self.filter_truncated and prompt_overflow[idx]:
                    filter_flag = True
                if self.min_target_tokens > 0 and target_lengths[idx] < self.min_target_tokens:
                    filter_flag = True
                if filter_flag:
                    labels[idx] = [-100] * len(labels[idx])
        max_len = max((len(seq) for seq in sequences), default=1)
        batch_size = len(sequences)
        input_ids = torch.full((batch_size, max_len), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        label_ids = torch.full((batch_size, max_len), -100, dtype=torch.long)
        for idx, seq in enumerate(sequences):
            length = len(seq)
            if self.padding_side == "left":
                start = max_len - length
                input_ids[idx, start:] = torch.tensor(seq, dtype=torch.long)
                attention_mask[idx, start:] = 1
                label_ids[idx, start:] = torch.tensor(labels[idx], dtype=torch.long)
            else:
                input_ids[idx, :length] = torch.tensor(seq, dtype=torch.long)
                attention_mask[idx, :length] = 1
                label_ids[idx, :length] = torch.tensor(labels[idx], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_ids,
            "debug_lm_drop_count": torch.tensor(drop_count, dtype=torch.long),
            "debug_lm_drop_seen": torch.tensor(drop_seen, dtype=torch.long),
            "debug_lm_drop_maxlen_count": torch.tensor(drop_maxlen_count, dtype=torch.long),
            "debug_lm_drop_min_target_count": torch.tensor(drop_min_target_count, dtype=torch.long),
            "debug_lm_drop_all_fallback": torch.tensor(
                1 if drop_all_fallback else 0, dtype=torch.long
            ),
        }


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    pre_args, _ = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(description="DeepSpeed SFT baseline with AutoModelForCausalLM.")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--lm", default=None)
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        help="Attention implementation for the LM (e.g., flash_attention_2, sdpa, eager).",
    )
    parser.add_argument(
        "--padding-side",
        default=None,
        choices=["left", "right"],
        help="Tokenizer padding side; defaults to left when using flash_attention_2.",
    )
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable TF32 matmul kernels for speed (Ampere+).",
    )
    parser.add_argument("--deepspeed-config", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument(
        "--save-at-end",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save a final checkpoint after training.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-lm-len", type=int, default=4096)
    parser.add_argument(
        "--append-eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append EOS token to the SFT target.",
    )
    parser.add_argument(
        "--prefilter-long-prompts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pre-tokenize prompts and drop samples with prompt_len >= max_lm_len.",
    )
    parser.add_argument(
        "--prefilter-long-prompts-log-samples",
        type=int,
        default=3,
        help="Number of dropped prompt samples to log when prefiltering.",
    )
    parser.add_argument(
        "--prefilter-long-prompts-preview",
        type=int,
        default=120,
        help="Max characters to show for dropped prompt previews.",
    )
    parser.add_argument(
        "--filter-lm-truncated",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Mask labels for samples where prompt hits max length.",
    )
    parser.add_argument(
        "--min-target-tokens",
        type=int,
        default=0,
        help="Mask labels for samples with fewer than this many target tokens.",
    )
    parser.add_argument(
        "--drop-lm-truncated",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Hard-drop samples where prompt hits max length.",
    )
    parser.add_argument(
        "--drop-min-target-tokens",
        type=int,
        default=0,
        help="Hard-drop samples with fewer than this many target tokens.",
    )
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--checkpoint-reentrant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use reentrant activation checkpointing in the custom wrapper.",
    )
    parser.add_argument(
        "--torch-autocast",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use torch.autocast outside DeepSpeed (off by default).",
    )
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-targets",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated module names for LoRA injection.",
    )
    parser.add_argument("--lm-lr", type=float, default=2e-5)
    parser.add_argument("--lora-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lora-weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--scheduler-type", default="cosine")
    parser.add_argument("--use-swanlab", action="store_true")
    parser.add_argument("--swanlab-project", default="dibjudge")
    parser.add_argument("--swanlab-run-name", default=None)
    parser.add_argument("--swanlab-tags", default=None)
    parser.add_argument("--swanlab-log-steps", type=int, default=10)
    parser.add_argument("--config", default=None, help="YAML config file to load defaults from.")
    parser.add_argument(
        "--save-config",
        default=None,
        help="Write resolved args to this YAML path (rank0 only).",
    )
    parser.add_argument(
        "--no-save-config",
        action="store_true",
        help="Disable saving the resolved config YAML.",
    )
    parser.add_argument("--local_rank", type=int, default=0)

    if pre_args.config:
        parser.set_defaults(**_load_yaml_config(pre_args.config, parser))
    args = parser.parse_args()
    required = ["data_path", "lm", "deepspeed_config"]
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        parser.error(f"Missing required arguments (or YAML keys): {', '.join(missing)}")
    attn_impl = str(args.attn_implementation or "").strip().lower()
    if attn_impl in {"", "none", "disable", "disabled"}:
        attn_impl = ""
    args.attn_implementation = attn_impl or None
    if args.attn_implementation == "flash_attention_2":
        args.torch_dtype = "bfloat16" if args.bf16 else "float16"
    else:
        args.torch_dtype = None

    import deepspeed

    rank, world_size, local_rank = init_distributed()
    deepspeed.init_distributed()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if rank == 0:
        warnings.filterwarnings("default")
    else:
        warnings.filterwarnings("ignore")
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    if not args.no_save_config and args.save_config is None and args.config is None:
        args.save_config = os.path.join("configs", "finetune_deepspeed_sft_baseline.yaml")
    if args.save_config and rank == 0:
        _save_yaml_config(args, args.save_config)
    if args.log_dir and rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    tok_kwargs = {"use_fast": True}
    tokenizer = AutoTokenizer.from_pretrained(args.lm, **tok_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.padding_side:
        padding_side = args.padding_side
    else:
        padding_side = "left" if args.attn_implementation == "flash_attention_2" else None
    if padding_side:
        tokenizer.padding_side = padding_side
    else:
        padding_side = tokenizer.padding_side
    args.padding_side = padding_side

    dataset = DIBJudgeDataset.from_jsonl(args.data_path)
    _rank0_print(rank, "[stage done] dataset loaded")
    if args.prefilter_long_prompts:
        dataset = _prefilter_long_prompts(
            dataset=dataset,
            tokenizer=tokenizer,
            max_lm_len=int(args.max_lm_len),
            rank=rank,
            log_samples=int(args.prefilter_long_prompts_log_samples),
            preview_len=int(args.prefilter_long_prompts_preview),
        )
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if world_size > 1
        else None
    )
    collator = SFTCollator(
        tokenizer,
        max_length=args.max_lm_len,
        append_eos=args.append_eos,
        filter_truncated=bool(args.filter_lm_truncated),
        min_target_tokens=int(args.min_target_tokens),
        drop_truncated=bool(args.drop_lm_truncated),
        drop_min_target_tokens=int(args.drop_min_target_tokens),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    ds_config = _load_ds_config(args.deepspeed_config)
    zero_stage = _resolve_zero_stage(ds_config)
    use_zero3 = zero_stage == 3
    if not args.gradient_checkpointing and isinstance(ds_config, dict):
        if ds_config.get("activation_checkpointing"):
            args.gradient_checkpointing = True
            _rank0_print(
                rank,
                "[stage done] enabling gradient checkpointing from deepspeed activation_checkpointing",
            )
    if use_zero3:
        _rank0_print(
            rank,
            "[stage done] ZeRO-3 detected; keeping model on CPU before DeepSpeed init",
        )

    load_dtype = _resolve_torch_dtype(args.torch_dtype)
    model = _load_causal_lm(
        args.lm, args.attn_implementation, torch_dtype=load_dtype
    )
    if (
        args.attn_implementation == "flash_attention_2"
        and device.type == "cuda"
        and not use_zero3
    ):
        model = model.to(device)
    _maybe_resize_embeddings(model, tokenizer, "lm", rank)
    model = _maybe_apply_lora(model, args, rank)
    if args.gradient_checkpointing:
        model = CheckpointedCausalLM(model, use_reentrant=bool(args.checkpoint_reentrant))
        _rank0_print(
            rank,
            "[stage done] custom checkpointing enabled "
            f"(reentrant={bool(args.checkpoint_reentrant)})",
        )

    optimizer_params = _build_optimizer_params(
        model,
        lm_lr=args.lm_lr,
        lora_lr=args.lora_lr,
        weight_decay=args.weight_decay,
        lora_weight_decay=args.lora_weight_decay,
    )

    steps_per_epoch = math.ceil(len(dataset) / max(1, args.per_device_train_batch_size))
    if world_size > 1:
        samples_per_rank = math.ceil(len(dataset) / max(1, world_size))
        steps_per_epoch = math.ceil(
            samples_per_rank / max(1, args.per_device_train_batch_size)
        )
    update_steps_per_epoch = math.ceil(steps_per_epoch / max(1, args.grad_accum_steps))
    total_update_steps = max(1, update_steps_per_epoch * max(1, args.epochs))
    warmup_steps = _warmup_steps_from_ratio(
        total_update_steps, args.warmup_ratio, args.warmup_steps
    )

    ds_config = _load_ds_config(args.deepspeed_config)
    if isinstance(ds_config, dict):
        ds_config["gradient_accumulation_steps"] = args.grad_accum_steps
        ds_config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
        ds_config["train_batch_size"] = (
            args.per_device_train_batch_size * args.grad_accum_steps * world_size
        )
        ds_config["gradient_clipping"] = args.max_grad_norm
        if "bf16" in ds_config and isinstance(ds_config["bf16"], dict):
            ds_config["bf16"]["enabled"] = bool(args.bf16)
        if "fp16" in ds_config and isinstance(ds_config["fp16"], dict):
            ds_config["fp16"]["enabled"] = not bool(args.bf16)

    force_ds_cpu = True
    if isinstance(ds_config, dict) and "zero_force_ds_cpu_optimizer" in ds_config:
        force_ds_cpu = bool(ds_config["zero_force_ds_cpu_optimizer"])
    if _zero_offload_enabled(ds_config) and force_ds_cpu:
        try:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
        except ImportError as exc:
            _rank0_print(rank, f"[warn] failed to import DeepSpeedCPUAdam: {exc}")
            optimizer = torch.optim.AdamW(optimizer_params)
            if isinstance(ds_config, dict):
                ds_config["zero_force_ds_cpu_optimizer"] = False
                _rank0_print(
                    rank,
                    "[warn] falling back to AdamW; set zero_force_ds_cpu_optimizer=false",
                )
        else:
            optimizer = DeepSpeedCPUAdam(optimizer_params)
            _rank0_print(rank, "[stage done] using DeepSpeedCPUAdam for ZeRO-Offload")
    else:
        optimizer = torch.optim.AdamW(optimizer_params)

    scheduler = None
    sched_name = str(args.scheduler_type or "").lower()
    if sched_name and sched_name not in {"none", "disable", "disabled"}:
        scheduler = get_scheduler(
            sched_name,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_update_steps,
        )

    ds_kwargs = {
        "model": model,
        "optimizer": optimizer,
        "model_parameters": [p for p in model.parameters() if p.requires_grad],
        "config": ds_config,
    }
    if scheduler is not None:
        ds_kwargs["lr_scheduler"] = scheduler
    engine, optimizer, _, _ = deepspeed.initialize(**ds_kwargs)
    _rank0_print(rank, "[stage done] deepspeed initialized")
    _warn_if_zero3_offload_inactive(engine, ds_config, rank)
    ckpt_fn, ckpt_reentrant = _configure_deepspeed_activation_checkpointing(
        ds_config, rank
    )
    if ckpt_fn is not None and hasattr(engine.module, "set_activation_checkpointing"):
        engine.module.set_activation_checkpointing(
            ckpt_fn, supports_reentrant=ckpt_reentrant
        )
    device = next(engine.module.parameters()).device

    use_amp = args.torch_autocast
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

    _rank0_print(rank, "DeepSpeed config:", ds_config)
    _rank0_print(
        rank,
        "SFT config:",
        {
            "lm_lr": args.lm_lr,
            "lora_lr": args.lora_lr,
            "use_lora": bool(args.use_lora),
            "weight_decay": args.weight_decay,
            "lora_weight_decay": args.lora_weight_decay,
            "warmup_steps": warmup_steps,
            "scheduler_type": args.scheduler_type,
            "grad_accum_steps": args.grad_accum_steps,
        },
    )

    swanlab_client = None
    if rank == 0 and args.use_swanlab:
        tags = None
        if args.swanlab_tags:
            tags = [tag.strip() for tag in args.swanlab_tags.split(",") if tag.strip()]
        swanlab_client = init_swanlab(
            True,
            project=args.swanlab_project,
            run_name=args.swanlab_run_name,
            config={"args": vars(args)},
            log_dir=args.log_dir,
            tags=tags,
        )
        _rank0_print(rank, "[stage done] swanlab initialized")

    step = 0
    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        engine.train()
        epoch_loader = _maybe_tqdm(loader, rank, f"epoch {epoch}")
        tqdm_enabled = hasattr(epoch_loader, "set_postfix")
        tqdm_every = args.swanlab_log_steps if args.swanlab_log_steps > 0 else 50
        totals = {
            "loss": 0.0,
            "steps": 0,
            "lm_drop_count": 0.0,
            "lm_drop_seen": 0.0,
            "lm_drop_maxlen": 0.0,
            "lm_drop_min_target": 0.0,
        }
        gpu_batch_keys = {"input_ids", "attention_mask", "labels"}
        for batch in epoch_loader:
            step_start = time.perf_counter()
            step += 1
            batch = _move_batch_to_device(batch, device, keys=gpu_batch_keys)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = engine(**batch)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
            engine.backward(loss)
            engine.step()

            totals["loss"] += float(loss.item())
            totals["steps"] += 1
            drop_count = batch.get("debug_lm_drop_count")
            drop_seen = batch.get("debug_lm_drop_seen")
            drop_maxlen = batch.get("debug_lm_drop_maxlen_count")
            drop_min_target = batch.get("debug_lm_drop_min_target_count")
            if torch.is_tensor(drop_count):
                totals["lm_drop_count"] += float(drop_count.item())
            if torch.is_tensor(drop_seen):
                totals["lm_drop_seen"] += float(drop_seen.item())
            if torch.is_tensor(drop_maxlen):
                totals["lm_drop_maxlen"] += float(drop_maxlen.item())
            if torch.is_tensor(drop_min_target):
                totals["lm_drop_min_target"] += float(drop_min_target.item())
            if tqdm_enabled and step % tqdm_every == 0:
                epoch_loader.set_postfix(loss=f"{loss.item():.4f}")
            if (
                swanlab_client is not None
                and args.swanlab_log_steps > 0
                and step % args.swanlab_log_steps == 0
            ):
                step_time = max(1e-8, time.perf_counter() - step_start)
                batch_size = int(batch["input_ids"].size(0))
                token_count = int(batch["attention_mask"].sum().item())
                lr_values = [group.get("lr", 0.0) for group in optimizer.param_groups]
                drop_ratio = 0.0
                if torch.is_tensor(drop_count) and torch.is_tensor(drop_seen):
                    seen = float(drop_seen.item())
                    if seen > 0:
                        drop_ratio = float(drop_count.item()) / seen
                log_swanlab(
                    swanlab_client,
                    {
                        "loss": float(loss.item()),
                        "lr": float(max(lr_values) if lr_values else 0.0),
                        "tokens": token_count,
                        "tokens_per_sec": float(token_count / step_time),
                        "samples_per_sec": float(batch_size / step_time),
                        "data/lm_drop_count": float(drop_count.item())
                        if torch.is_tensor(drop_count)
                        else 0.0,
                        "data/lm_drop_seen": float(drop_seen.item())
                        if torch.is_tensor(drop_seen)
                        else 0.0,
                        "data/lm_drop_ratio": drop_ratio,
                        "data/lm_drop_maxlen_count": float(drop_maxlen.item())
                        if torch.is_tensor(drop_maxlen)
                        else 0.0,
                        "data/lm_drop_min_target_count": float(drop_min_target.item())
                        if torch.is_tensor(drop_min_target)
                        else 0.0,
                    },
                    step=step,
                )

            if (
                args.save_every_steps > 0
                and step % args.save_every_steps == 0
                and engine.is_gradient_accumulation_boundary()
            ):
                _save_deepspeed_checkpoint(
                    engine, args.output_dir or "outputs/deepspeed", f"step{step}", rank
                )

        avg_loss = totals["loss"] / max(1, totals["steps"])
        seen = totals.get("lm_drop_seen", 0.0)
        drop_ratio = (totals.get("lm_drop_count", 0.0) / seen) if seen else 0.0
        _rank0_print(
            rank,
            f"epoch={epoch} avg_loss={avg_loss:.4f} "
            f"lm_drop_ratio={drop_ratio:.4f} "
            f"lm_drop_maxlen_ratio={(totals.get('lm_drop_maxlen', 0.0) / seen) if seen else 0.0:.4f} "
            f"lm_drop_min_target_ratio={(totals.get('lm_drop_min_target', 0.0) / seen) if seen else 0.0:.4f}",
        )
        if swanlab_client is not None:
            log_swanlab(
                swanlab_client,
                {"epoch_loss": float(avg_loss), "epoch": epoch},
                step=step,
            )

    if args.save_at_end:
        checkpoint_dir = args.output_dir or "outputs/deepspeed"
        _save_deepspeed_checkpoint(engine, checkpoint_dir, "final", rank)
        _save_hf_checkpoint(checkpoint_dir, "final", args, rank)

    if swanlab_client is not None and rank == 0:
        finish_swanlab(swanlab_client)


if __name__ == "__main__":
    main()
