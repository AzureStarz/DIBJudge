from __future__ import annotations

import argparse
import math
import warnings
import os
import json
import time
from dataclasses import asdict
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from .data import DIBJudgeCollator, DIBJudgeDataset
from .modeling import DIBJudgeConfig, DIBJudgeModel
from .proxy_tasks import ProxyTaskConfig, compute_proxy_losses
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


def _grl_schedule(
    max_lambda: float, step: int, total_steps: int, warmup_steps: int, gamma: float
) -> float:
    if max_lambda <= 0:
        return 0.0
    if total_steps <= 0:
        return max_lambda
    if step < warmup_steps:
        return 0.0
    denom = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, float(step - warmup_steps + 1) / float(denom)))
    return max_lambda * (2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0)


def _phase_boundary(total_steps: int, grl_start_ratio: float) -> int:
    if total_steps <= 0:
        return 0
    ratio = min(1.0, max(0.0, float(grl_start_ratio)))
    core_steps = int(total_steps * ratio)
    return min(total_steps, max(0, core_steps))


def _warmup_steps_from_ratio(
    total_steps: int, ratio: Optional[float], warmup_steps: int
) -> int:
    if ratio is None:
        return warmup_steps
    if ratio < 0:
        raise ValueError("warmup_ratio must be >= 0.")
    return int(total_steps * ratio)


def _compute_bias_terms(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return compute_proxy_losses(outputs, batch)


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


def _param_grad_norm(params: List[torch.nn.Parameter], device: torch.device) -> float:
    total = torch.zeros((), device=device)
    for param in params:
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        total += grad.pow(2).sum()
    return float(total.sqrt().item())


def _grad_norm_from_loss(
    loss: torch.Tensor, params: List[torch.nn.Parameter]
) -> torch.Tensor:
    if not params:
        return loss.new_tensor(0.0)
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    total = loss.new_tensor(0.0)
    for grad in grads:
        if grad is None:
            continue
        total = total + grad.detach().float().pow(2).sum()
    return total.sqrt()


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


def _parse_bins(text: str, cast=float) -> List:
    if text is None:
        return []
    if isinstance(text, (list, tuple)):
        return [cast(val) for val in text]
    parts = []
    for raw in str(text).split(","):
        raw = raw.strip().strip("[]")
        if not raw:
            continue
        parts.append(cast(raw))
    return parts


def _encode_ids(tokenizer, text: str, max_length: Optional[int]) -> List[int]:
    kwargs = {"add_special_tokens": False, "truncation": True}
    if max_length is not None:
        kwargs["max_length"] = max_length
    return tokenizer(text, **kwargs)["input_ids"]


def _has_response(text: Optional[str]) -> bool:
    return bool(text and text.strip())


def _compute_length_quantile_bins(
    dataset: DIBJudgeDataset,
    tokenizer,
    max_length: Optional[int],
    bins: int = 10,
) -> List[int]:
    lengths: List[int] = []
    for ex in dataset:
        lengths.append(len(_encode_ids(tokenizer, ex.response_a or "", max_length)))
        if _has_response(ex.response_b):
            lengths.append(len(_encode_ids(tokenizer, ex.response_b or "", max_length)))
    if not lengths or bins < 2:
        return list(ProxyTaskConfig().length_bins)
    lengths.sort()
    n = len(lengths)
    cuts: List[int] = []
    for idx in range(1, bins):
        q = idx / float(bins)
        pos = int(math.ceil(q * n)) - 1
        pos = min(max(pos, 0), n - 1)
        cuts.append(lengths[pos])
    max_len = lengths[-1]
    return [0] + cuts + [max_len]


def _collect_proxy_values(dataset: DIBJudgeDataset, attrs: Tuple[str, ...]) -> List[float]:
    values: List[float] = []
    for ex in dataset:
        for attr in attrs:
            val = getattr(ex, attr, None)
            if val is None:
                continue
            if isinstance(val, float) and math.isnan(val):
                continue
            values.append(float(val))
    return values


def _compute_value_quantile_bins(
    values: List[float],
    bins: int,
    fallback: Tuple[float, ...],
) -> List[float]:
    if not values or bins < 2:
        return list(fallback)
    values.sort()
    n = len(values)
    cuts: List[float] = []
    for idx in range(1, bins):
        q = idx / float(bins)
        pos = int(math.ceil(q * n)) - 1
        pos = min(max(pos, 0), n - 1)
        cuts.append(values[pos])
    return [values[0]] + cuts + [values[-1]]


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
    # DeepSpeed requires all ranks to participate in checkpointing.
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


def _save_state_dict(
    state_dict: Dict[str, torch.Tensor],
    output_dir: str,
    rank: int,
    filename: str = "model.safetensors",
) -> str:
    try:
        from safetensors.torch import save_file
    except ImportError:
        path = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(state_dict, path)
        _rank0_print(rank, f"[hf] saved pytorch checkpoint: {path}")
        return path
    path = os.path.join(output_dir, filename)
    save_file(state_dict, path)
    _rank0_print(rank, f"[hf] saved safetensors checkpoint: {path}")
    return path


def _save_lm_assets(output_dir: str, args: argparse.Namespace, rank: int) -> None:
    try:
        lm_tok = AutoTokenizer.from_pretrained(
            args.lm, local_files_only=True, use_fast=False
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


def _save_encoder_tokenizer(output_dir: str, args: argparse.Namespace, rank: int) -> None:
    try:
        enc_tok = AutoTokenizer.from_pretrained(
            args.judge_encoder, local_files_only=True, use_fast=False
        )
        enc_tok.save_pretrained(output_dir)
    except Exception as exc:
        _rank0_print(rank, f"[warn] failed to save judge encoder tokenizer: {exc}")


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


def _strip_prefix_if_present(
    state_dict: Dict[str, torch.Tensor], prefix: str
) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix) :]: value for key, value in state_dict.items()}
    return state_dict


def _save_hf_checkpoint(
    checkpoint_dir: str,
    tag: str,
    args: argparse.Namespace,
    rank: int,
) -> None:
    output_dir = os.path.join(checkpoint_dir, f"hf-{tag}")
    _save_hf_from_zero(
        checkpoint_dir=checkpoint_dir,
        tag=tag,
        output_dir=output_dir,
        args=args,
        rank=rank,
    )


def _save_hf_from_zero(
    checkpoint_dir: str,
    tag: str,
    output_dir: str,
    args: argparse.Namespace,
    rank: int,
) -> None:
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
    _rank0_print(rank, f"[hf] extracting DIBJudge from {checkpoint_dir} tag={tag}")
    raw_state = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir, tag)
    state_dict = _normalize_state_dict(raw_state)

    dibjudge_state = {
        key: value for key, value in state_dict.items() if not key.startswith("judge_lm.")
    }
    if args.use_lora:
        lm_state = {
            key[len("judge_lm.") :]: value
            for key, value in state_dict.items()
            if key.startswith("judge_lm.")
        }
        if lm_state:
            merged = _merge_lora_into_lm_state(lm_state, args, rank)
            if merged is not None:
                state_dict = {
                    key: value for key, value in state_dict.items() if not key.startswith("judge_lm.")
                }
                for key, value in merged.items():
                    state_dict[f"judge_lm.{key}"] = value
                lm_state = merged
        else:
            _rank0_print(rank, "[warn] no judge_lm weights found for LoRA merge.")
        dibjudge_state = {
            key: value for key, value in state_dict.items() if not key.startswith("judge_lm.")
        }
    else:
        lm_state = {
            key[len("judge_lm.") :]: value
            for key, value in state_dict.items()
            if key.startswith("judge_lm.")
        }

    lm_dir = os.path.join(output_dir, "lm")
    dibjudge_dir = os.path.join(output_dir, "dibjudge")
    os.makedirs(lm_dir, exist_ok=True)
    os.makedirs(dibjudge_dir, exist_ok=True)

    if lm_state:
        lm_state = _strip_prefix_if_present(lm_state, "base_model.model.")
        lm_state = _strip_prefix_if_present(lm_state, "base_model.")
        try:
            lm_model = AutoModelForCausalLM.from_pretrained(
                args.lm, local_files_only=True
            )
        except Exception as exc:
            _rank0_print(rank, f"[warn] failed to load base model for LM export: {exc}")
        else:
            missing, unexpected = lm_model.load_state_dict(lm_state, strict=False)
            if missing:
                _rank0_print(rank, f"[warn] missing keys while loading LM: {len(missing)}")
            if unexpected:
                _rank0_print(rank, f"[warn] unexpected keys while loading LM: {len(unexpected)}")
            lm_model.save_pretrained(lm_dir)
            _save_lm_assets(lm_dir, args, rank)
    else:
        _rank0_print(rank, "[warn] no judge_lm weights found for LM export.")

    config = DIBJudgeConfig(
        judge_encoder_name=args.judge_encoder,
        judge_lm_name=args.lm,
        z_latent_dim=args.z_latent_dim,
        z_prompt_len=args.z_prompt_len,
        z_prompt_prefix_len=args.z_prompt_prefix_len,
        z_prompt_postfix_len=args.z_prompt_postfix_len,
        prompt_mlp_hidden=args.prompt_mlp_hidden,
        prompt_mlp_layers=args.prompt_mlp_layers,
        prompt_mlp_dropout=args.prompt_mlp_dropout,
        grl_lambda=args.grl_lambda,
        bottleneck_noise_alpha=args.bottleneck_noise_alpha,
        bias_proxy_hidden=args.bias_proxy_hidden,
        bias_proxy_layers=args.bias_proxy_layers,
        bias_proxy_dropout=args.bias_proxy_dropout,
        low_recon_layer=args.low_recon_layer,
        compact_prior=args.compact_prior,
        compact_mu_token_id=args.compact_mu_token_id,
        compact_head_hidden=args.compact_head_hidden,
        compact_head_layers=args.compact_head_layers,
        compact_head_dropout=args.compact_head_dropout,
        proxy_length_classes=getattr(args, "proxy_length_classes", DIBJudgeConfig.proxy_length_classes),
    )
    config_path = os.path.join(dibjudge_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as handle:
        payload = asdict(config)
        payload["model_type"] = "dibjudge"
        json.dump(payload, handle, indent=2)
    _save_state_dict(dibjudge_state, dibjudge_dir, rank)
    _save_encoder_tokenizer(dibjudge_dir, args, rank)


def _print_trainable_params(model: torch.nn.Module, rank: int) -> None:
    total = 0
    trainable = 0
    for param in model.parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
    pct = (100.0 * trainable / total) if total else 0.0
    _rank0_print(rank, f"[params] trainable={trainable:,} total={total:,} ({pct:.2f}%)")


def _set_lm_trainable(model: DIBJudgeModel, trainable: bool, rank: int) -> None:
    for param in model.judge_lm.parameters():
        param.requires_grad = trainable
    state = "trainable" if trainable else "frozen"
    _rank0_print(rank, f"[stage done] judge_lm {state}")


def _set_shared_encoder_trainable(model: DIBJudgeModel, mode: str, rank: int) -> bool:
    mode = (mode or "all").lower()
    if mode == "all":
        for param in model.shared_encoder.parameters():
            param.requires_grad = True
        _rank0_print(rank, "[stage done] shared encoder fully trainable")
        return False
    if mode == "none":
        for param in model.shared_encoder.parameters():
            param.requires_grad = False
        _rank0_print(rank, "[stage done] shared encoder frozen")
        return True
    if mode != "last":
        _rank0_print(rank, f"[warn] unknown encoder_trainable={mode}, using all")
        for param in model.shared_encoder.parameters():
            param.requires_grad = True
        return False
    for param in model.shared_encoder.parameters():
        param.requires_grad = False
    last_layer = None
    enc = model.shared_encoder
    layer_paths = (
        ("encoder", "block"),
        ("encoder", "layers"),
        ("encoder", "layer"),
        ("encoder", "h"),
        ("layers",),
        ("layer",),
        ("block",),
        ("blocks",),
        ("h",),
    )
    for path in layer_paths:
        mod = enc
        ok = True
        for attr in path:
            if not hasattr(mod, attr):
                ok = False
                break
            mod = getattr(mod, attr)
        if not ok:
            continue
        if isinstance(mod, (torch.nn.ModuleList, list)) and len(mod) > 0:
            last_layer = mod[-1]
            break
    if last_layer is None:
        _rank0_print(rank, "[warn] unable to locate encoder layers; shared encoder fully trainable")
        for param in model.shared_encoder.parameters():
            param.requires_grad = True
        return False
    for param in last_layer.parameters():
        param.requires_grad = True
    last_ids = {id(param) for param in last_layer.parameters()}
    trainable = [param for param in model.shared_encoder.parameters() if param.requires_grad]
    only_last = bool(trainable) and all(id(param) in last_ids for param in trainable)
    _rank0_print(rank, "[stage done] shared encoder trainable=last")
    return only_last


def _maybe_apply_lora(model: DIBJudgeModel, args: argparse.Namespace) -> None:
    if not args.use_lora:
        return
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
    model.judge_lm = get_peft_model(model.judge_lm, lora_cfg)


def _build_optimizer_params(
    model: torch.nn.Module,
    encoder_lr: float,
    lm_lr: float,
    lora_lr: float,
    head_lr: float,
    weight_decay: float,
    head_weight_decay: float,
) -> List[Dict[str, object]]:
    groups: Dict[str, List[torch.nn.Parameter]] = {
        "encoder": [],
        "lm": [],
        "lora": [],
        "head": [],
    }
    head_prefixes = (
        "eng_domain_head.",
        "position_head.",
        "low_recon_head.",
        "length_bin_head.",
        "nll_bin_head.",
        "ttr_bin_head.",
        "compact_head.",
        "prompt_mlp.",
        "z_to_lm.",
        "z_prompt_prefix",
        "z_prompt_postfix",
    )
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name:
            groups["lora"].append(param)
        elif name.startswith("shared_encoder."):
            groups["encoder"].append(param)
        elif name.startswith("judge_lm."):
            groups["lm"].append(param)
        elif name.startswith(head_prefixes):
            groups["head"].append(param)
        else:
            groups["head"].append(param)

    def _split(
        params: List[torch.nn.Parameter], lr: float, decay_value: float
    ) -> List[Dict[str, object]]:
        decay, no_decay = [], []
        for param in params:
            if param.ndim == 1:
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {"params": decay, "weight_decay": decay_value, "lr": lr},
            {"params": no_decay, "weight_decay": 0.0, "lr": lr},
        ]

    param_groups: List[Dict[str, object]] = []
    if groups["encoder"]:
        param_groups.extend(_split(groups["encoder"], encoder_lr, weight_decay))
    if groups["lm"]:
        param_groups.extend(_split(groups["lm"], lm_lr, weight_decay))
    if groups["lora"]:
        param_groups.extend(_split(groups["lora"], lora_lr, head_weight_decay))
    if groups["head"]:
        param_groups.extend(_split(groups["head"], head_lr, head_weight_decay))
    return param_groups


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    pre_args, _ = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(description="DeepSpeed LoRA finetuning for DIBJudge.")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--judge-encoder", default=None)
    parser.add_argument("--lm", default=None)
    parser.add_argument("--z-latent-dim", type=int, default=256)
    parser.add_argument("--z-prompt-len", type=int, default=16)
    parser.add_argument("--prompt-mlp-hidden", type=int, default=0)
    parser.add_argument("--prompt-mlp-layers", type=int, default=1)
    parser.add_argument("--prompt-mlp-dropout", type=float, default=0.1)
    parser.add_argument("--bottleneck-noise-alpha", type=float, default=8.0)
    parser.add_argument("--bottleneck-noise-warmup-ratio", type=float, default=0.2)
    parser.add_argument("--z-prompt-prefix-len", type=int, default=1)
    parser.add_argument("--z-prompt-postfix-len", type=int, default=1)
    parser.add_argument(
        "--encoder-trainable",
        default="all",
        choices=["all", "last", "none"],
        help="Shared encoder trainable params: all, last, or none.",
    )
    parser.add_argument(
        "--freeze-lm-when-no-lora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze LM parameters when LoRA is disabled.",
    )
    parser.add_argument("--bias-proxy-hidden", type=int, default=0)
    parser.add_argument("--bias-proxy-layers", type=int, default=-1)
    parser.add_argument("--bias-proxy-dropout", type=float, default=-1.0)
    parser.add_argument("--low-recon-layer", type=int, default=2)
    parser.add_argument("--compact-prior", type=float, default=0.3)
    parser.add_argument("--compact-mu-token-id", type=int, default=-1)
    parser.add_argument("--compact-head-hidden", type=int, default=0)
    parser.add_argument("--compact-head-layers", type=int, default=1)
    parser.add_argument("--compact-head-dropout", type=float, default=0.1)
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
    parser.add_argument("--max-bias-len", type=int, default=1024)
    parser.add_argument("--max-ref-len", type=int, default=1024)
    parser.add_argument("--max-lm-len", type=int, default=4096)
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
    parser.add_argument("--lambda-compression", type=float, default=1.0)
    parser.add_argument("--lambda-compression-warmup-ratio", type=float, default=0.05)
    parser.add_argument("--mask-loss-weight", type=float, default=1.0)
    parser.add_argument("--consistency-loss-weight", type=float, default=1.0)
    parser.add_argument("--lambda-bias", type=float, default=1.0)
    parser.add_argument("--grl-lambda", type=float, default=1.0)
    parser.add_argument("--grl-start-ratio", type=float, default=0.3)
    parser.add_argument("--grl-gamma", type=float, default=10.0)
    parser.add_argument("--bias-decoder-steps", type=int, default=1)
    parser.add_argument("--eng-domain-weight", type=float, default=1.0)
    parser.add_argument("--low-recon-weight", type=float, default=0.5)
    parser.add_argument("--z-l2-weight", type=float, default=0.1)
    parser.add_argument("--nll-bin-weight", type=float, default=0.5)
    parser.add_argument(
        "--ppl-bin-weight",
        dest="nll_bin_weight",
        type=float,
        default=0.5,
        help="Deprecated alias for --nll-bin-weight.",
    )
    parser.add_argument("--ttr-bin-weight", type=float, default=0.5)
    parser.add_argument("--length-bin-weight", type=float, default=0.5)
    parser.add_argument(
        "--position-weight",
        type=float,
        default=0.0,
        help="Weight for adversarial position-discriminator loss.",
    )
    parser.add_argument(
        "--proxy-nll-bins",
        dest="proxy_nll_bins",
        default="0,2.3026,2.9957,3.6889,4.3820,5.0752,13.8155",
        help="Comma-separated bins for NLL (log PPL) ordinal regression.",
    )
    parser.add_argument(
        "--proxy-ppl-bins",
        dest="proxy_nll_bins",
        default="0,2.3026,2.9957,3.6889,4.3820,5.0752,13.8155",
        help="Deprecated alias for --proxy-nll-bins.",
    )
    parser.add_argument(
        "--proxy-nll-quantiles",
        dest="proxy_nll_quantiles",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use training-set NLL quantiles instead of fixed bins.",
    )
    parser.add_argument(
        "--proxy-ppl-quantiles",
        dest="proxy_nll_quantiles",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deprecated alias for --proxy-nll-quantiles.",
    )
    parser.add_argument(
        "--proxy-ttr-bins",
        default="0,0.2,0.4,0.6,0.8,1.0",
        help="Comma-separated bins for TTR ordinal regression.",
    )
    parser.add_argument(
        "--proxy-ttr-quantiles",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use training-set TTR quantiles instead of fixed bins.",
    )
    parser.add_argument(
        "--proxy-length-bins",
        default="0,50,100,200,400,1000000",
        help="Comma-separated bins for length ordinal regression.",
    )
    parser.add_argument(
        "--proxy-length-quantiles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use decile length quantiles from the training set instead of fixed bins.",
    )
    parser.add_argument(
        "--proxy-soft-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use soft interpolation between ordinal bins.",
    )
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--lm-lr", type=float, default=2e-5)
    parser.add_argument("--lora-lr", type=float, default=2e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--head-weight-decay", type=float, default=0.001)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--scheduler-type", default="cosine")
    parser.add_argument("--debug-data", action="store_true")
    parser.add_argument(
        "--debug-aux-checks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable auxiliary-loss diagnostics (grad norms, coverage).",
    )
    parser.add_argument("--debug-aux-checks-interval", type=int, default=200)
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
    required = ["data_path", "judge_encoder", "lm", "deepspeed_config"]
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
        args.save_config = os.path.join("configs", "finetune_deepspeed.yaml")
    if args.save_config and rank == 0:
        _save_yaml_config(args, args.save_config)
    if args.log_dir and rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    enc_tok_kwargs = {"use_fast": False, "legacy": True}
    lm_tok_kwargs = {"use_fast": True}
    judge_tok = AutoTokenizer.from_pretrained(args.judge_encoder, **enc_tok_kwargs)
    lm_tok = AutoTokenizer.from_pretrained(args.lm, **lm_tok_kwargs)
    if judge_tok.pad_token_id is None:
        judge_tok.pad_token = judge_tok.eos_token
    if lm_tok.pad_token_id is None:
        lm_tok.pad_token = lm_tok.eos_token

    mu_id = lm_tok.convert_tokens_to_ids(".")
    if mu_id is None or mu_id == lm_tok.unk_token_id:
        mu_id = lm_tok.eos_token_id
    if mu_id is None:
        mu_id = lm_tok.pad_token_id or 0
    args.compact_mu_token_id = int(mu_id)

    length_bins = _parse_bins(args.proxy_length_bins, float)
    if len(length_bins) < 2:
        length_bins = list(ProxyTaskConfig().length_bins)
    nll_bins = _parse_bins(args.proxy_nll_bins, float)
    if len(nll_bins) < 2:
        nll_bins = list(ProxyTaskConfig().nll_bins)
    ttr_bins = _parse_bins(args.proxy_ttr_bins, float)
    if len(ttr_bins) < 2:
        ttr_bins = list(ProxyTaskConfig().ttr_bins)

    dataset = DIBJudgeDataset.from_jsonl(args.data_path)
    _rank0_print(rank, "[stage done] dataset loaded")
    if args.proxy_length_quantiles:
        length_bins = _compute_length_quantile_bins(
            dataset, lm_tok, args.max_bias_len, bins=10
        )
    if args.proxy_nll_quantiles:
        nll_bins = _compute_value_quantile_bins(
            _collect_proxy_values(dataset, ("proxy_nll_a", "proxy_nll_b")),
            bins=10,
            fallback=tuple(nll_bins),
        )
    if args.proxy_ttr_quantiles:
        ttr_bins = _compute_value_quantile_bins(
            _collect_proxy_values(dataset, ("proxy_ttr_a", "proxy_ttr_b")),
            bins=10,
            fallback=tuple(ttr_bins),
        )
    proxy_config = ProxyTaskConfig(
        nll_bins=tuple(nll_bins),
        ttr_bins=tuple(ttr_bins),
        length_bins=tuple(length_bins),
        use_soft_labels=bool(args.proxy_soft_labels),
    )
    args.proxy_nll_classes = max(2, len(nll_bins) - 1)
    args.proxy_ttr_classes = max(2, len(ttr_bins) - 1)
    args.proxy_length_classes = max(2, len(length_bins) - 1)
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if world_size > 1
        else None
    )
    samples_per_rank = math.ceil(len(dataset) / max(1, world_size))
    steps_per_epoch = math.ceil(samples_per_rank / max(1, args.per_device_train_batch_size))
    total_steps = max(1, steps_per_epoch * max(1, args.epochs))
    grl_start = _phase_boundary(total_steps, args.grl_start_ratio)
    collator = DIBJudgeCollator(
        lm_tok,
        max_bias_len=args.max_bias_len,
        max_ref_len=args.max_ref_len,
        max_lm_len=args.max_lm_len,
        proxy_config=proxy_config,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )
    _rank0_print(rank, "[stage done] dataloader ready")
    if args.debug_data and rank == 0:
        batch = next(iter(loader))
        shapes = {k: tuple(v.shape) for k, v in batch.items() if torch.is_tensor(v)}
        _rank0_print(rank, "Debug batch shapes:", shapes)
        for key in ("original_attention_mask", "lm_attention_mask"):
            if key in batch:
                mask = batch[key]
                lengths = mask.sum(dim=-1).view(-1).tolist()
                _rank0_print(rank, f"{key} lengths (first 8):", lengths[:8])

    model = DIBJudgeModel.init_from_backbones(
        judge_encoder_name=args.judge_encoder,
        judge_lm_name=args.lm,
        z_latent_dim=args.z_latent_dim,
        z_prompt_len=args.z_prompt_len,
        prompt_mlp_hidden=args.prompt_mlp_hidden,
        prompt_mlp_layers=args.prompt_mlp_layers,
        prompt_mlp_dropout=args.prompt_mlp_dropout,
        bottleneck_noise_alpha=args.bottleneck_noise_alpha,
        grl_lambda=args.grl_lambda,
        z_prompt_prefix_len=args.z_prompt_prefix_len,
        z_prompt_postfix_len=args.z_prompt_postfix_len,
        low_recon_layer=args.low_recon_layer,
        bias_proxy_hidden=args.bias_proxy_hidden,
        bias_proxy_layers=args.bias_proxy_layers,
        bias_proxy_dropout=args.bias_proxy_dropout,
        compact_prior=args.compact_prior,
        compact_mu_token_id=args.compact_mu_token_id,
        compact_head_hidden=args.compact_head_hidden,
        compact_head_layers=args.compact_head_layers,
        compact_head_dropout=args.compact_head_dropout,
        proxy_nll_classes=max(2, len(nll_bins) - 1),
        proxy_ttr_classes=max(2, len(ttr_bins) - 1),
        proxy_length_classes=max(2, len(length_bins) - 1),
    ).to(device)
    _maybe_resize_embeddings(model.shared_encoder, judge_tok, "shared_encoder", rank)
    _maybe_resize_embeddings(model.judge_lm, lm_tok, "judge_lm", rank)
    skip_encoder_checkpointing = _set_shared_encoder_trainable(
        model, args.encoder_trainable, rank
    )
    _rank0_print(rank, "[stage done] model initialized")
    if args.gradient_checkpointing:
        # DeepSpeed ZeRO-3 can trigger metadata mismatch in non-reentrant checkpointing.
        gc_kwargs = {"use_reentrant": True}
        if not skip_encoder_checkpointing:
            model.shared_encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gc_kwargs
            )
        model.judge_lm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gc_kwargs
        )
    _maybe_apply_lora(model, args)
    if args.freeze_lm_when_no_lora and not args.use_lora:
        _set_lm_trainable(model, False, rank)
    _rank0_print(rank, "[stage done] lora applied" if args.use_lora else "[stage done] lora skipped")

    optimizer_params = _build_optimizer_params(
        model,
        encoder_lr=args.encoder_lr,
        lm_lr=args.lm_lr,
        lora_lr=args.lora_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        head_weight_decay=args.head_weight_decay,
    )
    optimizer = torch.optim.AdamW(optimizer_params)

    steps_per_epoch = math.ceil(len(dataset) / max(1, args.per_device_train_batch_size))
    if world_size > 1:
        steps_per_epoch = math.ceil(samples_per_rank / max(1, args.per_device_train_batch_size))
    update_steps_per_epoch = math.ceil(steps_per_epoch / max(1, args.grad_accum_steps))
    total_update_steps = max(1, update_steps_per_epoch * max(1, args.epochs))
    warmup_steps = _warmup_steps_from_ratio(
        total_update_steps, args.warmup_ratio, args.warmup_steps
    )
    scheduler = None
    sched_name = str(args.scheduler_type or "").lower()
    if sched_name and sched_name not in {"none", "disable", "disabled"}:
        scheduler = get_scheduler(
            sched_name,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_update_steps,
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

    use_external_scheduler = scheduler is not None and isinstance(ds_config, dict) and "scheduler" not in ds_config
    ds_kwargs = {
        "model": model,
        "optimizer": optimizer,
        "model_parameters": [p for p in model.parameters() if p.requires_grad],
        "config": ds_config,
    }
    if use_external_scheduler:
        ds_kwargs["lr_scheduler"] = scheduler
    engine, optimizer, _, _ = deepspeed.initialize(**ds_kwargs)
    _rank0_print(rank, "[stage done] deepspeed initialized")
    _print_trainable_params(engine.module, rank)
    model_dtype = None
    for param in engine.module.parameters():
        if param.requires_grad:
            model_dtype = param.dtype
            break
    if model_dtype is None:
        for param in engine.module.parameters():
            model_dtype = param.dtype
            break
    if model_dtype is None:
        model_dtype = torch.bfloat16 if args.bf16 else torch.float32
    encoder_params: List[torch.nn.Parameter] = []
    lm_params: List[torch.nn.Parameter] = []
    if args.debug_aux_checks:
        for name, param in engine.module.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("shared_encoder."):
                encoder_params.append(param)
            elif name.startswith("judge_lm."):
                lm_params.append(param)

    use_amp = args.torch_autocast
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

    _rank0_print(rank, "DeepSpeed config:", ds_config)
    _rank0_print(
        rank,
        "Loss config:",
        {
            "lambda_bias": args.lambda_bias,
            "lambda_compression": args.lambda_compression,
            "lambda_compression_warmup_ratio": args.lambda_compression_warmup_ratio,
            "grl_lambda": args.grl_lambda,
            "grl_start_ratio": args.grl_start_ratio,
            "grl_gamma": args.grl_gamma,
            "bottleneck_noise_alpha": args.bottleneck_noise_alpha,
            "bottleneck_noise_warmup_ratio": args.bottleneck_noise_warmup_ratio,
            "eng_domain_weight": args.eng_domain_weight,
            "low_recon_weight": args.low_recon_weight,
            "z_l2_weight": args.z_l2_weight,
            "length_bin_weight": args.length_bin_weight,
            "position_weight": args.position_weight,
            "mask_loss_weight": args.mask_loss_weight,
            "consistency_loss_weight": args.consistency_loss_weight,
        },
    )
    _rank0_print(
        rank,
        "Scheduler config:",
        {
            "scheduler_type": args.scheduler_type if use_external_scheduler else "deepspeed",
            "warmup_steps": warmup_steps if use_external_scheduler else None,
            "total_update_steps": total_update_steps if use_external_scheduler else None,
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
    last_update_diag: Dict[str, float] = {}
    total_steps_est = max(1, args.epochs * len(loader))
    grl_start = _phase_boundary(total_steps_est, args.grl_start_ratio)
    warmup_phase_steps = grl_start if grl_start > 0 else total_steps_est
    compression_warmup_steps = (
        int(warmup_phase_steps * args.lambda_compression_warmup_ratio)
        if args.lambda_compression_warmup_ratio is not None
        else 0
    )
    noise_warmup_steps = (
        int(warmup_phase_steps * args.bottleneck_noise_warmup_ratio)
        if args.bottleneck_noise_warmup_ratio is not None
        else 0
    )
    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        engine.train()
        epoch_loader = _maybe_tqdm(loader, rank, f"epoch {epoch}")
        tqdm_enabled = hasattr(epoch_loader, "set_postfix")
        tqdm_every = args.swanlab_log_steps if args.swanlab_log_steps > 0 else 50
        totals = {
            "loss": 0.0,
            "lm_loss": 0.0,
            "compact_kl_loss": 0.0,
            "domain_loss": 0.0,
            "position_loss": 0.0,
            "low_recon_loss": 0.0,
            "z_l2_loss": 0.0,
            "nll_bin_loss": 0.0,
            "ttr_bin_loss": 0.0,
            "length_bin_loss": 0.0,
            "nll_bin_mae": 0.0,
            "ttr_bin_mae": 0.0,
            "length_bin_mae": 0.0,
            "compression_loss": 0.0,
            "mask_loss": 0.0,
            "consistency_loss": 0.0,
            "steps": 0,
        }

        for batch in epoch_loader:
            step_start = time.perf_counter()
            step_index = step + 1
            accum_boundary = step_index % args.grad_accum_steps == 0
            optim_step = step_index // args.grad_accum_steps
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            if step < grl_start:
                progress = float(step + 1) / float(max(1, grl_start))
                lambda_bias = args.lambda_bias * min(1.0, progress)
                grl_lambda = 0.0
                bias_detach = True
            else:
                lambda_bias = args.lambda_bias
                grl_lambda = _grl_schedule(
                    args.grl_lambda,
                    step,
                    total_steps_est,
                    grl_start,
                    args.grl_gamma,
                )
                bias_detach = False
            if compression_warmup_steps > 0:
                lambda_compression = min(
                    args.lambda_compression,
                    args.lambda_compression * float(step + 1) / float(compression_warmup_steps),
                )
            else:
                lambda_compression = args.lambda_compression
            if noise_warmup_steps > 0:
                noise_alpha = min(
                    args.bottleneck_noise_alpha,
                    args.bottleneck_noise_alpha * float(step + 1) / float(noise_warmup_steps),
                )
            else:
                noise_alpha = args.bottleneck_noise_alpha
            batch["grl_lambda"] = batch["lm_input_ids"].new_tensor(grl_lambda)
            batch["bottleneck_noise_alpha"] = batch["lm_input_ids"].new_tensor(noise_alpha)
            batch["bias_detach"] = bias_detach

            diag_metrics: Dict[str, float] = {}
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = engine(batch)
                proxy_losses = _compute_bias_terms(outputs, batch)
                domain_loss = proxy_losses["domain_loss"]
                position_loss = proxy_losses["position_loss"]
                low_recon_loss = proxy_losses["low_recon_loss"]
                z_l2_loss = proxy_losses["z_l2_loss"]
                nll_bin_loss = proxy_losses["nll_bin_loss"]
                ttr_bin_loss = proxy_losses["ttr_bin_loss"]
                length_bin_loss = proxy_losses["length_bin_loss"]
                nll_bin_mae = proxy_losses.get("nll_bin_mae", outputs["lm_loss"].new_tensor(0.0))
                ttr_bin_mae = proxy_losses.get("ttr_bin_mae", outputs["lm_loss"].new_tensor(0.0))
                length_bin_mae = proxy_losses.get("length_bin_mae", outputs["lm_loss"].new_tensor(0.0))

                mask_loss = outputs.get(
                    "compact_mask_loss", outputs["lm_loss"].new_tensor(0.0)
                )
                consistency_loss = outputs.get(
                    "compact_con_loss", outputs["lm_loss"].new_tensor(0.0)
                )
                compression_loss = (
                    args.mask_loss_weight * mask_loss
                    + args.consistency_loss_weight * consistency_loss
                )
                bias_loss = (
                    args.eng_domain_weight * domain_loss
                    + args.position_weight * position_loss
                    + args.low_recon_weight * low_recon_loss
                    + args.z_l2_weight * z_l2_loss
                    + args.nll_bin_weight * nll_bin_loss
                    + args.ttr_bin_weight * ttr_bin_loss
                    + args.length_bin_weight * length_bin_loss
                )
                core_lm_loss = outputs["lm_loss"] + outputs["compact_kl_loss"]
            if (
                args.debug_aux_checks
                and step % args.debug_aux_checks_interval == 0
                and not args.gradient_checkpointing
            ):
                compression_term = lambda_compression * compression_loss
                bias_term = lambda_bias * bias_loss
                lm_term = core_lm_loss
                enc_lm_grad = _grad_norm_from_loss(lm_term, encoder_params)
                enc_bias_grad = _grad_norm_from_loss(bias_term, encoder_params)
                enc_comp_grad = _grad_norm_from_loss(compression_term, encoder_params)
                lm_lm_grad = _grad_norm_from_loss(lm_term, lm_params)
                lm_bias_grad = _grad_norm_from_loss(bias_term, lm_params)
                lm_comp_grad = _grad_norm_from_loss(compression_term, lm_params)
                aux_enc = enc_bias_grad + enc_comp_grad
                aux_lm = lm_bias_grad + lm_comp_grad
                diag_metrics.update(
                    {
                        "diag/grad_norm/lm/encoder": float(enc_lm_grad.item()),
                        "diag/grad_norm/bias/encoder": float(enc_bias_grad.item()),
                        "diag/grad_norm/compression/encoder": float(enc_comp_grad.item()),
                        "diag/grad_norm/lm/lm": float(lm_lm_grad.item()),
                        "diag/grad_norm/bias/lm": float(lm_bias_grad.item()),
                        "diag/grad_norm/compression/lm": float(lm_comp_grad.item()),
                        "diag/grad_ratio/aux_vs_lm/encoder": float(
                            aux_enc.item() / (enc_lm_grad.item() + 1e-8)
                        ),
                        "diag/grad_ratio/aux_vs_lm/lm": float(
                            aux_lm.item() / (lm_lm_grad.item() + 1e-8)
                        ),
                    }
                )
            elif args.debug_aux_checks and step % args.debug_aux_checks_interval == 0:
                diag_metrics["diag/grad_norm/skipped_checkpointing"] = 1.0

            total_loss = (
                core_lm_loss
                + lambda_compression * compression_loss
                + lambda_bias * bias_loss
            )

            if args.bf16:
                if total_loss.dtype != torch.float32:
                    total_loss = total_loss.float()
            elif model_dtype is not None and total_loss.dtype != model_dtype:
                total_loss = total_loss.to(model_dtype)

            engine.backward(total_loss)
            if args.debug_aux_checks and accum_boundary:
                total_params = [p for p in engine.module.parameters() if p.requires_grad]
                total_norm = _param_grad_norm(total_params, total_loss.device)
                last_update_diag = {
                    "diag/grad_norm_total/encoder": _param_grad_norm(
                        encoder_params, total_loss.device
                    ),
                    "diag/grad_norm_total/lm": _param_grad_norm(
                        lm_params, total_loss.device
                    ),
                    "grad_norm/ratio": float(
                        total_norm / args.max_grad_norm if args.max_grad_norm > 0 else 0.0
                    ),
                    "grad_norm/clipped": float(
                        total_norm > args.max_grad_norm + 1e-6 if args.max_grad_norm > 0 else 0.0
                    ),
                }
            engine.step()

            if (
                args.bias_decoder_steps > 1
                and step >= grl_start
                and accum_boundary
            ):
                for _ in range(args.bias_decoder_steps - 1):
                    engine.zero_grad()
                    bias_batch = dict(batch)
                    bias_batch["grl_lambda"] = batch["lm_input_ids"].new_tensor(0.0)
                    bias_batch["bias_detach"] = True
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                        bias_outputs = engine(bias_batch)
                        proxy_losses_b = _compute_bias_terms(bias_outputs, bias_batch)
                        bias_loss_b = (
                            args.eng_domain_weight * proxy_losses_b["domain_loss"]
                            + args.position_weight * proxy_losses_b["position_loss"]
                            + args.low_recon_weight * proxy_losses_b["low_recon_loss"]
                            + args.z_l2_weight * proxy_losses_b["z_l2_loss"]
                            + args.nll_bin_weight * proxy_losses_b["nll_bin_loss"]
                            + args.ttr_bin_weight * proxy_losses_b["ttr_bin_loss"]
                            + args.length_bin_weight * proxy_losses_b["length_bin_loss"]
                        )
                        bias_loss_b = bias_loss_b * lambda_bias
                    if args.bf16:
                        if bias_loss_b.dtype != torch.float32:
                            bias_loss_b = bias_loss_b.float()
                    elif model_dtype is not None and bias_loss_b.dtype != model_dtype:
                        bias_loss_b = bias_loss_b.to(model_dtype)
                    engine.backward(bias_loss_b)
                    saved_sched = getattr(engine, "lr_scheduler", None)
                    if saved_sched is not None:
                        engine.lr_scheduler = None
                    engine.step()
                    if saved_sched is not None:
                        engine.lr_scheduler = saved_sched

            totals["loss"] += total_loss.item()
            totals["lm_loss"] += outputs["lm_loss"].item()
            totals["compact_kl_loss"] += outputs["compact_kl_loss"].item()
            totals["domain_loss"] += domain_loss.item()
            totals["position_loss"] += position_loss.item()
            totals["low_recon_loss"] += low_recon_loss.item()
            totals["z_l2_loss"] += z_l2_loss.item()
            totals["nll_bin_loss"] += nll_bin_loss.item()
            totals["ttr_bin_loss"] += ttr_bin_loss.item()
            totals["length_bin_loss"] += length_bin_loss.item()
            totals["nll_bin_mae"] += nll_bin_mae.item()
            totals["ttr_bin_mae"] += ttr_bin_mae.item()
            totals["length_bin_mae"] += length_bin_mae.item()
            totals.setdefault("compression_loss", 0.0)
            totals["compression_loss"] += compression_loss.item()
            totals["mask_loss"] += mask_loss.item()
            totals["consistency_loss"] += consistency_loss.item()
            totals["steps"] += 1
            if tqdm_enabled and step % tqdm_every == 0:
                epoch_loader.set_postfix(
                    loss=f"{total_loss.item():.4f}",
                    lm=f"{outputs['lm_loss'].item():.4f}",
                )
            if (
                swanlab_client is not None
                and args.swanlab_log_steps > 0
                and step % args.swanlab_log_steps == 0
            ):
                step_time = max(1e-8, time.perf_counter() - step_start)
                batch_size = None
                if "lm_input_ids" in batch:
                    batch_size = int(batch["lm_input_ids"].size(0))
                elif "original_input_ids" in batch:
                    batch_size = int(batch["original_input_ids"].size(0))
                lm_tokens = (
                    int(batch["lm_attention_mask"].sum().item())
                    if "lm_attention_mask" in batch
                    else 0
                )
                original_tokens = (
                    int(batch["original_attention_mask"].sum().item())
                    if "original_attention_mask" in batch
                    else 0
                )
                perf = {"perf/step_time": step_time}
                if batch_size is not None:
                    perf["perf/samples_per_sec"] = batch_size / step_time
                if lm_tokens:
                    perf["perf/lm_tokens_per_sec"] = lm_tokens / step_time
                if original_tokens:
                    perf["perf/original_tokens_per_sec"] = original_tokens / step_time
                mem = {}
                if torch.cuda.is_available():
                    mem = {
                        "mem/allocated_mb": torch.cuda.memory_allocated() / (1024**2),
                        "mem/reserved_mb": torch.cuda.memory_reserved() / (1024**2),
                    }
                lrs = [group.get("lr", 0.0) for group in optimizer.param_groups]
                lr_stats = {}
                if lrs:
                    lr_stats = {
                        "lr/main_min": float(min(lrs)),
                        "lr/main_max": float(max(lrs)),
                        "lr/main_mean": float(sum(lrs) / len(lrs)),
                    }
                log_swanlab(
                    swanlab_client,
                    {
                            "loss": total_loss.item(),
                            "lm_loss": outputs["lm_loss"].item(),
                            "compact_kl_loss": outputs["compact_kl_loss"].item(),
                            "domain_loss": domain_loss.item(),
                            "position_loss": position_loss.item(),
                            "low_recon_loss": low_recon_loss.item(),
                            "z_l2_loss": z_l2_loss.item(),
                            "nll_bin_loss": nll_bin_loss.item(),
                            "ttr_bin_loss": ttr_bin_loss.item(),
                            "length_bin_loss": length_bin_loss.item(),
                            "nll_bin_mae": nll_bin_mae.item(),
                            "ttr_bin_mae": ttr_bin_mae.item(),
                            "length_bin_mae": length_bin_mae.item(),
                            "compression/mask_loss": mask_loss.item(),
                            "compression/consistency_loss": consistency_loss.item(),
                            "compression/loss": compression_loss.item(),
                            "compact/pi_mean": float(outputs.get("compact_pi_mean", 0.0)),
                            "compact/mask_mean": float(outputs.get("compact_mask_mean", 0.0)),
                            "compact/pi_saturation": float(outputs.get("compact_pi_saturation", 0.0)),
                            "compact/kl_loss": outputs["compact_kl_loss"].item(),
                            "weights/lambda_bias": lambda_bias,
                            "weights/lambda_compression": lambda_compression,
                            "weights/grl_lambda": grl_lambda,
                            "weights/bottleneck_noise_alpha": noise_alpha,
                            "weights/eng_domain": args.eng_domain_weight,
                            "weights/position": args.position_weight,
                            "weights/low_recon": args.low_recon_weight,
                            "weights/z_l2": args.z_l2_weight,
                            "weights/nll_bin": args.nll_bin_weight,
                            "weights/ttr_bin": args.ttr_bin_weight,
                            "weights/length_bin": args.length_bin_weight,
                            "weights/mask_loss": args.mask_loss_weight,
                            "weights/consistency_loss": args.consistency_loss_weight,
                            **last_update_diag,
                            "batch/lm_tokens": lm_tokens,
                            "batch/original_tokens": original_tokens,
                            **lr_stats,
                        **perf,
                        **mem,
                        **diag_metrics,
                    },
                    step=step,
                )
            if (
                args.save_every_steps is not None
                and args.save_every_steps > 0
                and accum_boundary
                and optim_step > 0
                and optim_step % args.save_every_steps == 0
            ):
                _save_deepspeed_checkpoint(
                    engine,
                    args.output_dir or "outputs/deepspeed",
                    tag=f"step-{optim_step}",
                    rank=rank,
                )
                _save_hf_checkpoint(
                    args.output_dir or "outputs/deepspeed",
                    tag=f"step-{optim_step}",
                    args=args,
                    rank=rank,
                )
            step += 1

        if rank == 0:
            denom = max(1, totals["steps"])
            metrics = {k: (v / denom) if k != "steps" else v for k, v in totals.items()}
            _rank0_print(rank, f"epoch={epoch} metrics={metrics}")
            if swanlab_client is not None:
                epoch_metrics = {f"epoch/{k}": v for k, v in metrics.items() if k != "steps"}
                epoch_metrics["epoch"] = epoch
                log_swanlab(swanlab_client, epoch_metrics, step=step)
        if args.save_at_end:
            _save_deepspeed_checkpoint(
                engine,
                args.output_dir or "outputs/deepspeed",
                tag=f"epoch-{epoch}",
                rank=rank,
            )
            _save_hf_checkpoint(
                args.output_dir or "outputs/deepspeed",
                tag=f"epoch-{epoch}",
                args=args,
                rank=rank,
            )
    if swanlab_client is not None and rank == 0:
        finish_swanlab(swanlab_client)


if __name__ == "__main__":
    main()
