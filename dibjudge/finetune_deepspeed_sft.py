from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
from typing import Dict, List, Optional, Tuple

import torch
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


def _maybe_tqdm(iterable, rank: int, desc: str):
    if rank != 0:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    total = len(iterable) if hasattr(iterable, "__len__") else None
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)


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
        base = AutoModelForCausalLM.from_pretrained(args.lm, local_files_only=True)
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

    if args.use_lora:
        merged = _merge_lora_into_lm_state(state_dict, args, rank)
        if merged is not None:
            state_dict = merged

    state_dict = _strip_prefix_if_present(state_dict, "base_model.model.")
    state_dict = _strip_prefix_if_present(state_dict, "base_model.")
    try:
        model = AutoModelForCausalLM.from_pretrained(args.lm, local_files_only=True)
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


def _warmup_steps_from_ratio(total_steps: int, ratio: Optional[float], warmup_steps: int) -> int:
    if ratio is None:
        return warmup_steps
    if ratio < 0:
        raise ValueError("warmup_ratio must be >= 0.")
    return int(total_steps * ratio)


class SFTCollator:
    def __init__(self, tokenizer, max_length: Optional[int], append_eos: bool) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length if max_length and max_length > 0 else None
        self.append_eos = append_eos
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id or 0
        self.eos_id = tokenizer.eos_token_id or self.pad_id

    def _encode_pair(self, ex: DIBJudgeExample) -> Tuple[List[int], List[int]]:
        prompt = ex.judge_prompt or ""
        output = ex.output or ""
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        output_ids = self.tokenizer.encode(output, add_special_tokens=False)
        if self.append_eos and (not output_ids or output_ids[-1] != self.eos_id):
            output_ids.append(self.eos_id)
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
        return input_ids, labels

    def __call__(self, batch: List[DIBJudgeExample]) -> Dict[str, torch.Tensor]:
        sequences: List[List[int]] = []
        labels: List[List[int]] = []
        for ex in batch:
            seq, lab = self._encode_pair(ex)
            sequences.append(seq)
            labels.append(lab)
        max_len = max((len(seq) for seq in sequences), default=1)
        batch_size = len(sequences)
        input_ids = torch.full((batch_size, max_len), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        label_ids = torch.full((batch_size, max_len), -100, dtype=torch.long)
        for idx, seq in enumerate(sequences):
            length = len(seq)
            input_ids[idx, :length] = torch.tensor(seq, dtype=torch.long)
            attention_mask[idx, :length] = 1
            label_ids[idx, :length] = torch.tensor(labels[idx], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_ids,
        }


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    pre_args, _ = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(description="DeepSpeed SFT baseline with AutoModelForCausalLM.")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--lm", default=None)
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
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
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

    import deepspeed

    rank, world_size, local_rank = init_distributed()
    deepspeed.init_distributed()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if rank == 0:
        warnings.filterwarnings("default")
    else:
        warnings.filterwarnings("ignore")

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

    dataset = DIBJudgeDataset.from_jsonl(args.data_path)
    _rank0_print(rank, "[stage done] dataset loaded")
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if world_size > 1
        else None
    )
    collator = SFTCollator(tokenizer, max_length=args.max_lm_len, append_eos=args.append_eos)
    loader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    model = AutoModelForCausalLM.from_pretrained(args.lm)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    _maybe_resize_embeddings(model, tokenizer, "lm", rank)
    model = _maybe_apply_lora(model, args, rank)

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
        steps_per_epoch = math.ceil(samples_per_rank / max(1, args.per_device_train_batch_size))
    update_steps_per_epoch = math.ceil(steps_per_epoch / max(1, args.grad_accum_steps))
    total_update_steps = max(1, update_steps_per_epoch * max(1, args.epochs))
    warmup_steps = _warmup_steps_from_ratio(
        total_update_steps, args.warmup_ratio, args.warmup_steps
    )

    ds_config = args.deepspeed_config
    if isinstance(ds_config, str) and os.path.isfile(ds_config):
        with open(ds_config, "r", encoding="utf-8") as handle:
            ds_config = json.load(handle)
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
        totals = {"loss": 0.0, "steps": 0}
        for batch in epoch_loader:
            step_start = time.perf_counter()
            step += 1
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = engine(**batch)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
            engine.backward(loss)
            engine.step()

            totals["loss"] += float(loss.item())
            totals["steps"] += 1
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
                log_swanlab(
                    swanlab_client,
                    {
                        "loss": float(loss.item()),
                        "lr": float(max(lr_values) if lr_values else 0.0),
                        "tokens": token_count,
                        "tokens_per_sec": float(token_count / step_time),
                        "samples_per_sec": float(batch_size / step_time),
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
        _rank0_print(rank, f"[epoch {epoch}] loss={avg_loss:.4f}")
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
