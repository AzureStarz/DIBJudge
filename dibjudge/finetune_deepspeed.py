from __future__ import annotations

import argparse
import contextlib
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


def _warmup_steps_from_ratio(
    total_steps: int, ratio: Optional[float], warmup_steps: int
) -> int:
    if ratio is None:
        return warmup_steps
    if ratio < 0:
        raise ValueError("warmup_ratio must be >= 0.")
    return int(total_steps * ratio)


def _piecewise_linear(progress: float, points: List[Tuple[float, float]]) -> float:
    if not points:
        return 0.0
    progress = max(0.0, min(1.0, float(progress)))
    points = sorted(points, key=lambda x: x[0])
    if progress <= points[0][0]:
        return float(points[0][1])
    for (p0, v0), (p1, v1) in zip(points, points[1:]):
        if progress <= p1:
            if p1 <= p0 + 1e-8:
                return float(v1)
            t = (progress - p0) / (p1 - p0)
            return float(v0 + t * (v1 - v0))
    return float(points[-1][1])


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


def _install_checkpoint_warning_hook(rank: int, log_path: Optional[str] = None) -> None:
    if rank != 0:
        return
    if getattr(_install_checkpoint_warning_hook, "_installed", False):
        return
    import torch
    import torch.utils.checkpoint as _checkpoint
    import inspect
    import traceback
    import warnings as _warnings

    orig_showwarning = _warnings.showwarning
    orig_checkpoint = _checkpoint.checkpoint
    orig_forward = _checkpoint.CheckpointFunction.forward
    _install_checkpoint_warning_hook._count = 0
    max_logs = 5

    def _log_line(text: str) -> None:
        _rank0_print(rank, text)
        if not log_path:
            return
        try:
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(text + "\n")
        except OSError:
            pass

    def _showwarning(message, category, filename, lineno, file=None, line=None):
        text = str(message)
        if "None of the inputs have requires_grad=True" in text:
            stack = traceback.extract_stack()
            idx = None
            for i in range(len(stack) - 1, -1, -1):
                if stack[i].filename.endswith("torch/utils/checkpoint.py"):
                    idx = i
                    break
            caller = stack[idx - 1] if idx is not None and idx > 0 else None
            _log_line(f"[checkpoint warn] {text}")
            if caller is not None:
                _log_line(
                    f"[checkpoint warn] caller={caller.filename}:{caller.lineno} in {caller.name}"
                )
            if idx is None:
                idx = len(stack) - 1
            start = max(0, idx - 4)
            end = min(len(stack), idx + 2)
            snippet = "".join(traceback.format_list(stack[start:end])).rstrip()
            if snippet:
                _log_line("[checkpoint warn] stack:\n" + snippet)
        return orig_showwarning(message, category, filename, lineno, file=file, line=line)

    def _describe_run_function(run_function) -> None:
        if _install_checkpoint_warning_hook._count >= max_logs:
            return
        _install_checkpoint_warning_hook._count += 1
        name = getattr(run_function, "__qualname__", None) or getattr(run_function, "__name__", None)
        if not name:
            name = repr(run_function)
        src_file = None
        try:
            src_file = inspect.getsourcefile(run_function)
        except TypeError:
            src_file = None
        _log_line(f"[checkpoint warn] run_function={name}")
        if src_file:
            _log_line(f"[checkpoint warn] run_function_file={src_file}")
        owner = getattr(run_function, "__self__", None)
        if isinstance(owner, torch.nn.Module):
            _log_line(f"[checkpoint warn] bound_module={owner.__class__.__name__}")
        elif owner is not None:
            _log_line(f"[checkpoint warn] bound_self={type(owner).__name__}")
        inner = getattr(run_function, "func", None)
        if inner is not None:
            _log_line(f"[checkpoint warn] partial_func={repr(inner)}")
            inner_self = getattr(inner, "__self__", None)
            if isinstance(inner_self, torch.nn.Module):
                _log_line(f"[checkpoint warn] partial_module={inner_self.__class__.__name__}")
        closure = getattr(run_function, "__closure__", None)
        if closure:
            for cell in closure:
                try:
                    value = cell.cell_contents
                except ValueError:
                    continue
                if isinstance(value, torch.nn.Module):
                    _log_line(f"[checkpoint warn] closure_module={value.__class__.__name__}")
                    break

    def _checkpoint_wrapper(function, *args, **kwargs):
        has_grad_input = any(
            torch.is_tensor(arg) and arg.requires_grad for arg in args
        )
        if not has_grad_input:
            stack = traceback.extract_stack()
            caller = None
            for i in range(len(stack) - 1, -1, -1):
                if stack[i].filename.endswith("torch/utils/checkpoint.py"):
                    if i > 0:
                        caller = stack[i - 1]
                    break
            _log_line("[checkpoint warn] checkpoint() called with no grad inputs")
            if caller is not None:
                _log_line(
                    f"[checkpoint warn] caller={caller.filename}:{caller.lineno} in {caller.name}"
                )
            _describe_run_function(function)
        return orig_checkpoint(function, *args, **kwargs)

    def _forward_wrapper(ctx, run_function, preserve_rng_state, *args):
        has_grad_input = any(
            torch.is_tensor(arg) and arg.requires_grad for arg in args
        )
        if not has_grad_input:
            _log_line("[checkpoint warn] CheckpointFunction.forward has no grad inputs")
            _describe_run_function(run_function)
        return orig_forward(ctx, run_function, preserve_rng_state, *args)

    _warnings.showwarning = _showwarning
    _checkpoint.checkpoint = _checkpoint_wrapper
    _checkpoint.CheckpointFunction.forward = staticmethod(_forward_wrapper)
    _install_checkpoint_warning_hook._installed = True


def _param_grad_norm(params: List[torch.nn.Parameter], device: torch.device) -> float:
    total = torch.zeros((), device=device)
    for param in params:
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        total += grad.pow(2).sum()
    return float(total.sqrt().item())


def _collect_vq_task_samples(
    model: DIBJudgeModel,
    dataset: DIBJudgeDataset,
    collator: DIBJudgeCollator,
    max_samples: int,
    batch_size: int,
    device: torch.device,
    seed: int,
    rank: int,
    torch_dtype: Optional[torch.dtype] = None,
    enable_autocast: bool = True,
    world_size: int = 1,
) -> Optional[torch.Tensor]:
    if max_samples <= 0 or len(dataset) == 0:
        return None
    batch_size = max(1, int(batch_size))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    if world_size > 1:
        indices = indices[int(rank) :: int(world_size)]
    samples: List[torch.Tensor] = []
    total = 0
    batch_examples = []
    model_was_training = model.training
    model.eval()
    amp_dtype = torch_dtype
    if amp_dtype is None:
        for param in model.shared_encoder.parameters():
            amp_dtype = param.dtype
            break
    if enable_autocast and device.type == "cuda":
        if amp_dtype not in (torch.float16, torch.bfloat16):
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    attn_impl = getattr(model.config, "attn_implementation", None)
    if attn_impl == "flash_attention_2" and device.type == "cuda":
        if amp_dtype not in (torch.float16, torch.bfloat16):
            amp_dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
        model.shared_encoder = model.shared_encoder.to(device=device, dtype=amp_dtype)
    use_autocast = bool(
        enable_autocast
        and device.type == "cuda"
        and amp_dtype in (torch.float16, torch.bfloat16)
    )

    def _random_subset(vectors: torch.Tensor, count: int) -> torch.Tensor:
        if count <= 0 or vectors.numel() == 0:
            return vectors.new_zeros((0, vectors.size(-1)))
        if vectors.size(0) <= count:
            return vectors
        idx = torch.randperm(vectors.size(0), device=vectors.device)[:count]
        return vectors[idx]

    def _sample_task_tokens(task_tokens: torch.Tensor, count: int) -> torch.Tensor:
        if count <= 0 or task_tokens.numel() == 0:
            return task_tokens.new_zeros((0, task_tokens.size(-1)))
        if task_tokens.dim() != 3:
            flat = task_tokens.reshape(-1, task_tokens.size(-1))
            return _random_subset(flat, count)
        prompt_len = task_tokens.size(1)
        if prompt_len <= 1:
            flat = task_tokens.reshape(-1, task_tokens.size(-1))
            return _random_subset(flat, count)
        per_pos = int(math.ceil(float(count) / float(prompt_len)))
        per_pos = max(1, per_pos)
        buckets = []
        for idx in range(prompt_len):
            buckets.append(_random_subset(task_tokens[:, idx, :], per_pos))
        flat = torch.cat(buckets, dim=0)
        return _random_subset(flat, count)

    try:
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
            if use_autocast
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), amp_ctx:
            for idx in indices:
                batch_examples.append(dataset[idx])
                if len(batch_examples) < batch_size:
                    continue
                batch = collator(batch_examples)
                batch_examples = []
                orig_ids = batch["original_input_ids"].to(device)
                orig_mask = batch["original_attention_mask"].to(device)
                orig_ids, orig_mask, _ = model._flatten_pair_inputs(orig_ids, orig_mask)
                outputs = model.shared_encoder(
                    input_ids=orig_ids,
                    attention_mask=orig_mask,
                    output_hidden_states=False,
                    use_cache=False,
                )
                hidden = model._get_hidden(outputs)
                pooled = model._pool_hidden(hidden, orig_mask)
                task_tokens = model.task_mlp(pooled)
                needed = max_samples - total
                if needed <= 0:
                    break
                flat = _sample_task_tokens(task_tokens, needed)
                samples.append(flat.detach().cpu())
                total += flat.size(0)
                if total >= max_samples:
                    break
            if batch_examples and total < max_samples:
                batch = collator(batch_examples)
                orig_ids = batch["original_input_ids"].to(device)
                orig_mask = batch["original_attention_mask"].to(device)
                orig_ids, orig_mask, _ = model._flatten_pair_inputs(orig_ids, orig_mask)
                outputs = model.shared_encoder(
                    input_ids=orig_ids,
                    attention_mask=orig_mask,
                    output_hidden_states=False,
                    use_cache=False,
                )
                hidden = model._get_hidden(outputs)
                pooled = model._pool_hidden(hidden, orig_mask)
                task_tokens = model.task_mlp(pooled)
                needed = max_samples - total
                if needed > 0:
                    flat = _sample_task_tokens(task_tokens, needed)
                    samples.append(flat.detach().cpu())
                    total += flat.size(0)
    finally:
        if model_was_training:
            model.train()
    if not samples:
        _rank0_print(rank, "[warn] VQ init requested but no samples were collected.")
        return None
    collected = torch.cat(samples, dim=0)
    _rank0_print(rank, f"[stage done] collected {collected.size(0)} VQ init samples")
    return collected


def _resolve_vq_init_samples(args: argparse.Namespace, dataset_len: int, rank: int) -> int:
    requested = int(getattr(args, "vq_init_samples", 0))
    if requested > 0:
        return requested
    prompt_len = max(1, int(getattr(args, "z_prompt_len", 1)))
    num_codes = max(1, int(getattr(args, "task_codebook_size", 1)))
    num_codebooks = max(1, int(getattr(args, "vq_num_codebooks", 1)))
    target = int(num_codes * num_codebooks * prompt_len * 8)
    target = max(10000, target)
    if dataset_len > 0:
        target = min(target, int(dataset_len) * prompt_len)
    if rank == 0:
        _rank0_print(rank, f"[stage done] auto vq_init_samples={target}")
    return int(target)


def _grad_norm_from_loss(
    loss: torch.Tensor, params: List[torch.nn.Parameter]
) -> torch.Tensor:
    if not params:
        return loss.new_tensor(0.0)
    if not torch.is_tensor(loss) or not loss.requires_grad:
        return loss.new_tensor(0.0)
    try:
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    except RuntimeError as exc:
        message = str(exc)
        if "Direct calls to tensor.backward()" in message:
            return loss.new_tensor(0.0)
        raise
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


def _resolve_zero_stage(config: object) -> Optional[int]:
    cfg = None
    if isinstance(config, str) and os.path.isfile(config):
        try:
            with open(config, "r", encoding="utf-8") as handle:
                cfg = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None
    elif isinstance(config, dict):
        cfg = config
    if not isinstance(cfg, dict):
        return None
    zero_opt = cfg.get("zero_optimization")
    if isinstance(zero_opt, dict):
        stage = zero_opt.get("stage")
    elif isinstance(zero_opt, int):
        stage = zero_opt
    else:
        stage = None
    if stage is None:
        return None
    try:
        return int(stage)
    except (TypeError, ValueError):
        return None


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
        prompt = ex.judge_prompt
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
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        engine.save_checkpoint(output_dir, tag=tag)
    except Exception as exc:
        _rank0_print(rank, f"[warn] deepspeed save_checkpoint failed: {exc}")
        if rank != 0:
            return
        ckpt_dir = os.path.join(output_dir, tag) if tag else output_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        state_path = os.path.join(ckpt_dir, "mp_rank_00_model_states.pt")
        try:
            torch.save({"module": engine.module.state_dict()}, state_path)
            _rank0_print(
                rank,
                f"[warn] saved minimal model state (no optimizer) to {state_path}",
            )
        except Exception as save_exc:
            _rank0_print(rank, f"[warn] fallback save failed: {save_exc}")


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


def _resolve_hf_save_dtype(args: argparse.Namespace) -> Optional[torch.dtype]:
    raw = getattr(args, "hf_save_dtype", None)
    if raw is None:
        raw = "auto"
    raw = str(raw).strip().lower()
    if raw in {"", "none", "disable", "disabled"}:
        return None
    if raw == "auto":
        return torch.bfloat16 if bool(getattr(args, "bf16", False)) else torch.float32
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return mapping.get(raw)


def _cast_state_dict(
    state_dict: Dict[str, torch.Tensor],
    dtype: Optional[torch.dtype],
) -> Dict[str, torch.Tensor]:
    if dtype is None:
        return state_dict
    casted: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            casted[key] = value
            continue
        if value.is_floating_point():
            casted[key] = value.detach().to(dtype=dtype).cpu()
        else:
            casted[key] = value.detach().cpu()
    return casted


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


def _has_lora_keys(state_dict: Dict[str, torch.Tensor]) -> bool:
    return any("lora_" in key for key in state_dict)


def _extract_lora_state_from_model(
    model: torch.nn.Module, rank: int
) -> Optional[Dict[str, torch.Tensor]]:
    try:
        from peft import get_peft_model_state_dict
    except ImportError:
        _rank0_print(rank, "[warn] peft is required to extract LoRA state.")
        return None
    if not hasattr(model, "peft_config"):
        return None
    try:
        state = get_peft_model_state_dict(model)
    except Exception as exc:
        _rank0_print(rank, f"[warn] failed to extract LoRA adapter state: {exc}")
        return None
    return {key: value.detach().cpu() for key, value in state.items()}


def _save_lora_adapter(model: torch.nn.Module, output_dir: str, rank: int) -> None:
    if not hasattr(model, "peft_config"):
        return
    try:
        model.save_pretrained(output_dir)
        _rank0_print(rank, f"[hf] saved LoRA adapter: {output_dir}")
    except Exception as exc:
        _rank0_print(rank, f"[warn] failed to save LoRA adapter: {exc}")


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
    remapped = _remap_state_dict(lm_state, set(peft_model.state_dict().keys()))
    missing, unexpected = peft_model.load_state_dict(remapped, strict=False)
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


def _apply_prefix_strip(
    state_dict: Dict[str, torch.Tensor], prefix: str
) -> Dict[str, torch.Tensor]:
    stripped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            stripped[key[len(prefix) :]] = value
        else:
            stripped[key] = value
    return stripped


def _count_matches(keys: Iterable[str], model_keys: set[str]) -> int:
    return sum(1 for key in keys if key in model_keys)


def _remap_state_dict(
    state_dict: Dict[str, torch.Tensor], model_keys: set[str]
) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    best = state_dict
    best_matches = _count_matches(best.keys(), model_keys)
    for prefix in ("module.", "model.", "base_model.", "base_model.model."):
        candidate = _apply_prefix_strip(state_dict, prefix)
        matches = _count_matches(candidate.keys(), model_keys)
        if matches > best_matches:
            best = candidate
            best_matches = matches
    return best


def _load_state_dict_from_ds_checkpoint(
    checkpoint_dir: str, tag: str, rank: int
) -> Optional[Dict[str, torch.Tensor]]:
    ckpt_dir = os.path.join(checkpoint_dir, tag) if tag else checkpoint_dir
    if not os.path.isdir(ckpt_dir):
        return None
    preferred = ("mp_rank_00_model_states.pt", "mp_rank_0_model_states.pt")
    candidates = []
    for name in os.listdir(ckpt_dir):
        if name.endswith("_model_states.pt"):
            candidates.append(os.path.join(ckpt_dir, name))
    for name in preferred:
        path = os.path.join(ckpt_dir, name)
        if path in candidates:
            candidates.remove(path)
            candidates.insert(0, path)
            break
    if not candidates:
        return None
    path = candidates[0]
    try:
        state = torch.load(path, map_location="cpu")
    except (OSError, RuntimeError) as exc:
        _rank0_print(rank, f"[warn] failed to load DS model state {path}: {exc}")
        return None
    if isinstance(state, dict):
        for key in ("module", "model", "state_dict", "module_state_dict"):
            value = state.get(key)
            if isinstance(value, dict):
                return value
    if isinstance(state, dict):
        return state
    return None


def _save_hf_checkpoint(
    checkpoint_dir: str,
    tag: str,
    args: argparse.Namespace,
    rank: int,
    engine: Optional["deepspeed.DeepSpeedEngine"] = None,
) -> None:
    output_dir = os.path.join(checkpoint_dir, f"hf-{tag}")
    _save_hf_from_zero(
        checkpoint_dir=checkpoint_dir,
        tag=tag,
        output_dir=output_dir,
        args=args,
        rank=rank,
        engine=engine,
    )


def _save_hf_from_zero(
    checkpoint_dir: str,
    tag: str,
    output_dir: str,
    args: argparse.Namespace,
    rank: int,
    engine: Optional["deepspeed.DeepSpeedEngine"] = None,
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
    raw_state = None
    zero_stage = _resolve_zero_stage(args.deepspeed_config)
    if zero_stage in (0, 1, 2) and engine is not None:
        raw_state = engine.module.state_dict()
    if raw_state is None:
        try:
            raw_state = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir, tag)
        except FileNotFoundError as exc:
            _rank0_print(rank, f"[warn] HF export fallback (no ZeRO states): {exc}")
            raw_state = _load_state_dict_from_ds_checkpoint(checkpoint_dir, tag, rank)
    if raw_state is None:
        _rank0_print(rank, "[warn] HF export skipped: no model states found.")
        return
    state_dict = _normalize_state_dict(raw_state)
    save_dtype = _resolve_hf_save_dtype(args)
    if save_dtype is not None:
        _rank0_print(rank, f"[hf] saving checkpoint dtype={save_dtype}")

    dibjudge_state = {
        key: value for key, value in state_dict.items() if not key.startswith("judge_lm.")
    }
    if args.use_lora:
        lm_state = {
            key[len("judge_lm.") :]: value
            for key, value in state_dict.items()
            if key.startswith("judge_lm.")
        }
        if not lm_state:
            _rank0_print(rank, "[warn] no judge_lm weights found for LoRA merge.")
        if lm_state and not _has_lora_keys(lm_state):
            _rank0_print(
                rank,
                "[warn] judge_lm state has no lora_ keys; trying live adapter weights.",
            )
        if (not lm_state or not _has_lora_keys(lm_state)) and engine is not None:
            live_lora = _extract_lora_state_from_model(engine.module.judge_lm, rank)
            if live_lora and _has_lora_keys(live_lora):
                lm_state = live_lora
                _rank0_print(rank, "[hf] using live LoRA adapter weights for merge.")
        if lm_state and _has_lora_keys(lm_state):
            merged = _merge_lora_into_lm_state(lm_state, args, rank)
            if merged is not None:
                state_dict = {
                    key: value for key, value in state_dict.items() if not key.startswith("judge_lm.")
                }
                for key, value in merged.items():
                    state_dict[f"judge_lm.{key}"] = value
                lm_state = merged
        elif lm_state:
            _rank0_print(rank, "[warn] skipping LoRA merge: no adapter weights found.")
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
    if args.use_lora and engine is not None:
        _save_lora_adapter(engine.module.judge_lm, os.path.join(lm_dir, "lora"), rank)

    if lm_state:
        try:
            lm_model = AutoModelForCausalLM.from_pretrained(
                args.lm, local_files_only=True
            )
        except Exception as exc:
            _rank0_print(rank, f"[warn] failed to load base model for LM export: {exc}")
        else:
            lm_state = _strip_prefix_if_present(lm_state, "base_model.model.")
            lm_state = _strip_prefix_if_present(lm_state, "base_model.")
            lm_state = _remap_state_dict(lm_state, set(lm_model.state_dict().keys()))
            missing, unexpected = lm_model.load_state_dict(lm_state, strict=False)
            if missing:
                _rank0_print(rank, f"[warn] missing keys while loading LM: {len(missing)}")
            if unexpected:
                _rank0_print(rank, f"[warn] unexpected keys while loading LM: {len(unexpected)}")
            lm_state_to_save = _cast_state_dict(lm_state, save_dtype)
            lm_model.save_pretrained(lm_dir, state_dict=lm_state_to_save)
            _save_lm_assets(lm_dir, args, rank)
    else:
        _rank0_print(rank, "[warn] no judge_lm weights found for LM export.")

    config = DIBJudgeConfig(
        judge_encoder_name=args.judge_encoder,
        judge_lm_name=args.lm,
        attn_implementation=args.attn_implementation,
        padding_side=args.padding_side,
        torch_dtype=args.torch_dtype,
        use_rms_norm=args.use_rms_norm,
        rms_norm_eps=args.rms_norm_eps,
        use_swiglu=args.use_swiglu,
        z_latent_dim=args.z_latent_dim,
        z_prompt_len=args.z_prompt_len,
        bias_prompt_len=args.bias_prompt_len,
        task_codebook_size=args.task_codebook_size,
        vq_num_codebooks=args.vq_num_codebooks,
        vq_commitment_gamma=args.vq_commitment_gamma,
        vq_ema_decay=args.vq_ema_decay,
        vq_use_ema=args.vq_use_ema,
        vq_normalize_inputs=args.vq_normalize_inputs,
        vq_codebook_trainable=args.vq_codebook_trainable,
        vq_dead_code_threshold=args.vq_dead_code_threshold,
        vq_reset_dead_codes=args.vq_reset_dead_codes,
        vq_align_samples=args.vq_align_samples,
        z_prompt_prefix_len=args.z_prompt_prefix_len,
        z_prompt_postfix_len=args.z_prompt_postfix_len,
        prompt_mlp_hidden=args.prompt_mlp_hidden,
        prompt_mlp_layers=args.prompt_mlp_layers,
        prompt_mlp_dropout=args.prompt_mlp_dropout,
        bottleneck_noise_alpha=args.bottleneck_noise_alpha,
        bias_proxy_hidden=args.bias_proxy_hidden,
        bias_proxy_layers=args.bias_proxy_layers,
        bias_proxy_dropout=args.bias_proxy_dropout,
        low_recon_layer=args.low_recon_layer,
        compact_prior=args.compact_keep_init,
        compact_mu_token_id=args.compact_mu_token_id,
        compact_head_hidden=args.compact_head_hidden,
        compact_head_layers=args.compact_head_layers,
        compact_head_dropout=args.compact_head_dropout,
        compact_pi_init=args.compact_pi_init,
        lm_loss_chunk_size=args.lm_loss_chunk_size,
        compact_kl_chunk_size=args.compact_kl_chunk_size,
        proxy_length_classes=getattr(args, "proxy_length_classes", DIBJudgeConfig.proxy_length_classes),
    )
    config_path = os.path.join(dibjudge_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as handle:
        payload = asdict(config)
        payload["model_type"] = "dibjudge"
        payload["z_soft_prompt"] = not bool(args.disable_z_prompt_insertion)
        payload["use_compactor"] = not bool(args.disable_compactor)
        json.dump(payload, handle, indent=2)
    dibjudge_state = _cast_state_dict(dibjudge_state, save_dtype)
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


def _freeze_non_lora_lm_params(model: DIBJudgeModel, rank: int) -> None:
    frozen = 0
    for name, param in model.judge_lm.named_parameters():
        if "lora_" in name:
            continue
        if param.requires_grad:
            param.requires_grad = False
            frozen += param.numel()
    if frozen > 0:
        _rank0_print(rank, f"[stage done] froze {frozen:,} non-LoRA LM params")


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
        task_type="CAUSAL_LM",
    )
    model.judge_lm = get_peft_model(model.judge_lm, lora_cfg)


def _validate_lora(model: DIBJudgeModel, args: argparse.Namespace, rank: int) -> None:
    if not args.use_lora:
        return
    targets = [name.strip() for name in args.lora_targets.split(",") if name.strip()]
    matched = {name: 0 for name in targets}
    for name, _ in model.judge_lm.named_modules():
        for target in targets:
            if name.endswith(target):
                matched[target] += 1
    missing = [name for name, count in matched.items() if count == 0]
    if missing:
        _rank0_print(
            rank,
            "[warn] LoRA targets not found in LM modules:",
            ", ".join(missing),
        )
    lora_params = [
        (name, param)
        for name, param in model.named_parameters()
        if "lora_" in name
    ]
    lora_count = sum(param.numel() for _, param in lora_params)
    if lora_count == 0:
        raise RuntimeError(
            "LoRA enabled but no lora_ parameters were created. "
            "Check --lora-targets against the LM module names."
        )
    frozen = [name for name, param in lora_params if not param.requires_grad]
    if frozen:
        _rank0_print(
            rank,
            f"[warn] {len(frozen)} LoRA params are frozen; check optimizer setup.",
        )
    trainable = sum(param.numel() for _, param in lora_params if param.requires_grad)
    _rank0_print(
        rank,
        f"[lora] params={lora_count:,} trainable={trainable:,} targets={matched}",
    )


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
        "task_mlp.",
        "bias_mlp.",
        "vq_task.",
        "task_vq_to_lm.",
        "low_recon_head.",
        "length_bin_head.",
        "nll_bin_head.",
        "ttr_bin_head.",
        "compact_head.",
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


def _build_optimizer(
    optimizer_params: List[Dict[str, object]],
    args: argparse.Namespace,
    rank: int,
    zero_stage: Optional[int],
) -> torch.optim.Optimizer:
    use_low_precision = bool(args.bf16) or str(getattr(args, "torch_dtype", "")).lower() in {
        "float16",
        "bfloat16",
    }
    if zero_stage == 0 and args.use_lora and use_low_precision:
        try:
            from deepspeed.ops.adam import FusedAdam
        except ImportError as exc:
            _rank0_print(
                rank,
                "[warn] DeepSpeed FusedAdam unavailable; "
                f"falling back to AdamW ({exc}).",
            )
        else:
            _rank0_print(
                rank,
                "[stage done] using DeepSpeed FusedAdam (FP32 optimizer states) "
                "for ZeRO-0 LoRA stability",
            )
            return FusedAdam(optimizer_params, adam_w_mode=True)
    if zero_stage == 0 and args.use_lora and use_low_precision:
        _rank0_print(
            rank,
            "[warn] ZeRO-0 + low-precision params + AdamW can underflow LoRA updates; "
            "consider enabling FusedAdam.",
        )
    return torch.optim.AdamW(optimizer_params)


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    pre_args, _ = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(description="DeepSpeed LoRA finetuning for DIBJudge.")
    parser.add_argument("--data-path", default=None)
    parser.add_argument(
        "--proxy-cache-path",
        default=None,
        help="Optional JSONL file with proxy labels aligned to --data-path.",
    )
    parser.add_argument(
        "--require-both-responses",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop samples with empty response_B when the response_B field is present.",
    )
    parser.add_argument(
        "--max-training-samples",
        type=int,
        default=0,
        help="Cap the number of training samples (0 = no cap).",
    )
    parser.add_argument("--judge-encoder", default=None)
    parser.add_argument("--lm", default=None)
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        help="Attention implementation for both encoder and LM (e.g., flash_attention_2, sdpa, eager).",
    )
    parser.add_argument(
        "--padding-side",
        default=None,
        choices=["left", "right"],
        help="Tokenizer padding side; defaults to left when using flash_attention_2.",
    )
    parser.add_argument(
        "--use-rms-norm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use RMSNorm in custom heads instead of LayerNorm.",
    )
    parser.add_argument(
        "--rms-norm-eps",
        type=float,
        default=1e-6,
        help="RMSNorm epsilon for custom heads.",
    )
    parser.add_argument(
        "--use-swiglu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use SwiGLU activations in custom MLP heads.",
    )
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable TF32 matmul kernels for speed (Ampere+).",
    )
    parser.add_argument("--z-latent-dim", type=int, default=256)
    parser.add_argument("--z-prompt-len", type=int, default=16)
    parser.add_argument("--bias-prompt-len", type=int, default=8)
    parser.add_argument("--prompt-mlp-hidden", type=int, default=0)
    parser.add_argument("--prompt-mlp-layers", type=int, default=1)
    parser.add_argument("--prompt-mlp-dropout", type=float, default=0.1)
    parser.add_argument("--task-codebook-size", type=int, default=1024)
    parser.add_argument("--vq-num-codebooks", type=int, default=4)
    parser.add_argument("--vq-commitment-gamma", type=float, default=0.05)
    parser.add_argument("--vq-ema-decay", type=float, default=0.99)
    parser.add_argument(
        "--vq-use-ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use EMA updates for VQ codebooks.",
    )
    parser.add_argument(
        "--vq-normalize-inputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize encoder outputs before VQ.",
    )
    parser.add_argument(
        "--vq-codebook-trainable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow gradient updates to VQ codebooks in addition to EMA.",
    )
    parser.add_argument("--vq-dead-code-threshold", type=float, default=0.1)
    parser.add_argument(
        "--vq-reset-dead-codes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reset dead codes to random batch vectors.",
    )
    parser.add_argument("--vq-align-samples", type=int, default=512)
    parser.add_argument(
        "--vq-usage-weight",
        type=float,
        default=0.1,
        help="Weight for code usage regularization.",
    )
    parser.add_argument(
        "--vq-init-samples",
        type=int,
        default=0,
        help="Number of task tokens to sample for VQ codebook kmeans++ init (0 = auto).",
    )
    parser.add_argument(
        "--vq-init-batch-size",
        type=int,
        default=0,
        help="Batch size for VQ init forward passes (0 = per_device_train_batch_size).",
    )
    parser.add_argument(
        "--vq-init-autocast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable autocast during VQ codebook initialization.",
    )
    parser.add_argument(
        "--vq-init-spherical",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize samples before VQ codebook initialization.",
    )
    parser.add_argument(
        "--vq-init-seed",
        type=int,
        default=0,
        help="Random seed for VQ codebook init sampling.",
    )
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
    parser.add_argument(
        "--freeze-lm-when-lora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze non-LoRA LM parameters when LoRA is enabled.",
    )
    parser.add_argument("--bias-proxy-hidden", type=int, default=0)
    parser.add_argument("--bias-proxy-layers", type=int, default=-1)
    parser.add_argument("--bias-proxy-dropout", type=float, default=-1.0)
    parser.add_argument("--low-recon-layer", type=int, default=2)
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
    parser.add_argument("--max-response-len", type=int, default=1024)
    parser.add_argument("--max-lm-len", type=int, default=4096)
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
    parser.add_argument("--lambda-compression", type=float, default=1.0)
    parser.add_argument("--mask-loss-weight", type=float, default=1.0)
    parser.add_argument("--consistency-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--mask-group-weight",
        type=float,
        default=0.5,
        help="Weight for (mask_loss + consistency_loss) group loss.",
    )
    parser.add_argument(
        "--loss-schedule",
        choices=["static", "dynamic"],
        default="dynamic",
        help="Use dynamic scheduling for auxiliary losses (dynamic) or keep static weights.",
    )
    parser.add_argument(
        "--disable-compactor",
        action="store_true",
        help="Disable compact masking and compact losses (KL/mask/consistency).",
    )
    parser.add_argument(
        "--disable-z-prompt-insertion",
        action="store_true",
        help="Disable insertion of task prompt tokens into the LM input.",
    )
    parser.add_argument("--compact-keep-init", type=float, default=0.9)
    parser.add_argument(
        "--compact-keep-final",
        type=float,
        default=0.7,
        help="Final keep ratio target for dynamic scheduling.",
    )
    parser.add_argument(
        "--compact-pi-init",
        type=float,
        default=0.95,
        help="Initialize compact head bias so sigmoid(pi) ~= this value.",
    )
    parser.add_argument("--kl-weight-max", type=float, default=1.0)
    parser.add_argument(
        "--compact-kl-chunk-size",
        type=int,
        default=0,
        help="Optional chunk size for compact KL computation (0 = disabled).",
    )
    parser.add_argument("--z-dropout", type=float, default=0.0)
    parser.add_argument(
        "--lm-loss-chunk-size",
        type=int,
        default=0,
        help="Optional chunk size for causal LM loss to reduce peak memory (0 = disabled).",
    )
    parser.add_argument("--lambda-bias", type=float, default=1.0)
    parser.add_argument("--low-recon-weight", type=float, default=0.5)
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
    parser.add_argument("--vq-task-weight", type=float, default=1.0)
    parser.add_argument("--vq-align-weight", type=float, default=0.0)
    parser.add_argument("--disentangle-weight", type=float, default=1.0)
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
    parser.add_argument(
        "--debug-loss-spike",
        action="store_true",
        help="Log batch-level diagnostics when lm_loss spikes.",
    )
    parser.add_argument(
        "--debug-loss-spike-threshold",
        type=float,
        default=0.0,
        help="Trigger spike logging when lm_loss exceeds this value.",
    )
    parser.add_argument(
        "--debug-loss-spike-multiplier",
        type=float,
        default=0.0,
        help="Trigger spike logging when lm_loss exceeds EMA * multiplier.",
    )
    parser.add_argument(
        "--debug-loss-spike-max-samples",
        type=int,
        default=3,
        help="Max samples to print when a loss spike is detected.",
    )
    parser.add_argument(
        "--debug-loss-spike-preview",
        type=int,
        default=120,
        help="Max characters to show in debug prompt/output previews.",
    )
    parser.add_argument(
        "--filter-lm-truncated",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Mask LM labels for samples where prompt hits max length or tokenizer reports truncation.",
    )
    parser.add_argument(
        "--min-target-tokens",
        type=int,
        default=0,
        help="Mask LM labels for samples with fewer than this many target tokens.",
    )
    parser.add_argument(
        "--drop-lm-truncated",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Hard-drop samples where prompt hits max length or tokenizer reports truncation.",
    )
    parser.add_argument(
        "--drop-min-target-tokens",
        type=int,
        default=0,
        help="Hard-drop samples with fewer than this many target tokens.",
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
        "--debug-lm-sample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log one detokenized LM input sample before training.",
    )
    parser.add_argument(
        "--debug-lm-sample-max-tokens",
        type=int,
        default=None,
        help="Max LM tokens to detokenize for the debug LM sample.",
    )
    parser.add_argument(
        "--debug-checkpoint-warning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log stack/caller info when checkpoint warns about missing grads.",
    )
    parser.add_argument("--debug-aux-checks-interval", type=int, default=200)
    parser.add_argument("--use-swanlab", action="store_true")
    parser.add_argument("--swanlab-project", default="dibjudge")
    parser.add_argument("--swanlab-run-name", default=None)
    parser.add_argument("--swanlab-tags", default=None)
    parser.add_argument("--swanlab-log-steps", type=int, default=10)
    parser.add_argument(
        "--hf-save-dtype",
        default="auto",
        help="HF export dtype (auto|float32|bfloat16|float16).",
    )
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
    attn_impl = str(args.attn_implementation or "").strip().lower()
    if attn_impl in {"", "none", "disable", "disabled"}:
        attn_impl = ""
    args.attn_implementation = attn_impl or None
    if args.attn_implementation == "flash_attention_2":
        args.torch_dtype = "bfloat16" if args.bf16 else "float16"
    else:
        args.torch_dtype = None
    if args.debug_lm_sample_max_tokens is None or args.debug_lm_sample_max_tokens <= 0:
        args.debug_lm_sample_max_tokens = int(args.max_lm_len)

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
    if args.debug_checkpoint_warning:
        log_base = args.log_dir or args.output_dir or "."
        log_path = os.path.join(log_base, "checkpoint_warn_rank0.log")
        _install_checkpoint_warning_hook(rank, log_path=log_path)

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
    if args.padding_side:
        padding_side = args.padding_side
    else:
        padding_side = "left" if args.attn_implementation == "flash_attention_2" else None
    if padding_side:
        judge_tok.padding_side = padding_side
        lm_tok.padding_side = padding_side
    else:
        padding_side = lm_tok.padding_side
    args.padding_side = padding_side

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

    dataset = DIBJudgeDataset.from_jsonl(
        args.data_path,
        proxy_cache_path=args.proxy_cache_path,
        require_both_responses=bool(args.require_both_responses),
    )
    max_samples = int(args.max_training_samples)
    if max_samples > 0 and len(dataset) > max_samples:
        dataset = DIBJudgeDataset(dataset.samples[:max_samples])
        _rank0_print(rank, f"[stage done] dataset capped to {max_samples} samples")
    _rank0_print(rank, "[stage done] dataset loaded")
    if args.prefilter_long_prompts:
        dataset = _prefilter_long_prompts(
            dataset=dataset,
            tokenizer=lm_tok,
            max_lm_len=int(args.max_lm_len),
            rank=rank,
            log_samples=int(args.prefilter_long_prompts_log_samples),
            preview_len=int(args.prefilter_long_prompts_preview),
        )
    if args.proxy_length_quantiles:
        length_bins = _compute_length_quantile_bins(
            dataset, lm_tok, args.max_response_len, bins=10
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
    proxy_labels_enabled = bool(args.proxy_cache_path)
    collator = DIBJudgeCollator(
        lm_tok,
        max_response_len=args.max_response_len,
        max_lm_len=args.max_lm_len,
        proxy_config=proxy_config,
        enable_proxy_labels=proxy_labels_enabled,
        debug_spike=bool(args.debug_loss_spike),
        debug_preview_len=int(args.debug_loss_spike_preview),
        filter_truncated=bool(args.filter_lm_truncated),
        min_target_tokens=int(args.min_target_tokens),
        drop_truncated=bool(args.drop_lm_truncated),
        drop_min_target_tokens=int(args.drop_min_target_tokens),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collator,
    )
    _rank0_print(rank, "[stage done] dataloader ready")
    if rank == 0 and (args.debug_data or args.debug_lm_sample):
        batch = next(iter(loader))
        if args.debug_data:
            shapes = {k: tuple(v.shape) for k, v in batch.items() if torch.is_tensor(v)}
            _rank0_print(rank, "Debug batch shapes:", shapes)
            for key in ("original_attention_mask", "lm_attention_mask"):
                if key in batch:
                    mask = batch[key]
                    lengths = mask.sum(dim=-1).view(-1).tolist()
                    _rank0_print(rank, f"{key} lengths (first 8):", lengths[:8])
        if args.debug_lm_sample:
            lm_ids = batch.get("lm_input_ids")
            lm_mask = batch.get("lm_attention_mask")
            lm_labels = batch.get("lm_labels")
            if torch.is_tensor(lm_ids):
                max_tokens = max(1, int(args.debug_lm_sample_max_tokens))
                take = int(lm_ids.size(1))
                if torch.is_tensor(lm_mask):
                    take = int(lm_mask[0].sum().item())
                take = min(take, max_tokens)
                if lm_tok.padding_side == "left":
                    ids = lm_ids[0, -take:].tolist()
                else:
                    ids = lm_ids[0, :take].tolist()
                decoded = lm_tok.decode(ids, skip_special_tokens=False)
                _rank0_print(
                    rank,
                    f"[lm sample] tokens={take} text='{decoded}'",
                )
                if torch.is_tensor(lm_labels):
                    label_ids = lm_labels[0].tolist()
                    label_ids = [tok for tok in label_ids if tok != -100][:take]
                    label_decoded = lm_tok.decode(
                        label_ids, skip_special_tokens=False
                    ) if label_ids else ""
                    _rank0_print(
                        rank,
                        f"[lm sample] labels='{label_decoded}'",
                    )

    model = DIBJudgeModel.init_from_backbones(
        judge_encoder_name=args.judge_encoder,
        judge_lm_name=args.lm,
        attn_implementation=args.attn_implementation,
        padding_side=args.padding_side,
        use_rms_norm=args.use_rms_norm,
        rms_norm_eps=args.rms_norm_eps,
        use_swiglu=args.use_swiglu,
        z_latent_dim=args.z_latent_dim,
        z_prompt_len=args.z_prompt_len,
        bias_prompt_len=args.bias_prompt_len,
        task_codebook_size=args.task_codebook_size,
        vq_num_codebooks=args.vq_num_codebooks,
        vq_commitment_gamma=args.vq_commitment_gamma,
        vq_ema_decay=args.vq_ema_decay,
        vq_use_ema=args.vq_use_ema,
        vq_normalize_inputs=args.vq_normalize_inputs,
        vq_codebook_trainable=args.vq_codebook_trainable,
        vq_dead_code_threshold=args.vq_dead_code_threshold,
        vq_reset_dead_codes=args.vq_reset_dead_codes,
        vq_align_samples=args.vq_align_samples,
        prompt_mlp_hidden=args.prompt_mlp_hidden,
        prompt_mlp_layers=args.prompt_mlp_layers,
        prompt_mlp_dropout=args.prompt_mlp_dropout,
        z_prompt_prefix_len=args.z_prompt_prefix_len,
        z_prompt_postfix_len=args.z_prompt_postfix_len,
        low_recon_layer=args.low_recon_layer,
        bias_proxy_hidden=args.bias_proxy_hidden,
        bias_proxy_layers=args.bias_proxy_layers,
        bias_proxy_dropout=args.bias_proxy_dropout,
        compact_prior=args.compact_keep_init,
        compact_mu_token_id=args.compact_mu_token_id,
        compact_head_hidden=args.compact_head_hidden,
        compact_head_layers=args.compact_head_layers,
        compact_head_dropout=args.compact_head_dropout,
        compact_pi_init=args.compact_pi_init,
        lm_loss_chunk_size=args.lm_loss_chunk_size,
        compact_kl_chunk_size=args.compact_kl_chunk_size,
        proxy_nll_classes=max(2, len(nll_bins) - 1),
        proxy_ttr_classes=max(2, len(ttr_bins) - 1),
        proxy_length_classes=max(2, len(length_bins) - 1),
    ).to(device)
    _maybe_resize_embeddings(model.shared_encoder, judge_tok, "shared_encoder", rank)
    _maybe_resize_embeddings(model.judge_lm, lm_tok, "judge_lm", rank)
    skip_encoder_checkpointing = _set_shared_encoder_trainable(
        model, args.encoder_trainable, rank
    )
    if args.disable_z_prompt_insertion:
        vq_init_samples = 0
        args.vq_init_samples = 0
        args.vq_task_weight = 0.0
        args.vq_align_weight = 0.0
        args.vq_usage_weight = 0.0
        args.disentangle_weight = 0.0
        args.vq_align_samples = 0
        _rank0_print(
            rank,
            "[stage done] disable_z_prompt_insertion=true; skipping VQ init/losses/align",
        )
    else:
        vq_init_samples = _resolve_vq_init_samples(args, len(dataset), rank)
    args.vq_init_samples = vq_init_samples
    if vq_init_samples > 0:
        vq_init_dtype = None
        if args.vq_init_autocast:
            vq_init_dtype = torch.bfloat16 if args.bf16 else torch.float16
        init_batch_size = (
            int(args.vq_init_batch_size)
            if int(args.vq_init_batch_size) > 0
            else int(args.per_device_train_batch_size)
        )
        per_rank_target = int(math.ceil(float(vq_init_samples) / float(max(1, world_size))))
        samples = _collect_vq_task_samples(
            model=model,
            dataset=dataset,
            collator=collator,
            max_samples=per_rank_target,
            batch_size=init_batch_size,
            device=device,
            seed=int(args.vq_init_seed),
            rank=rank,
            torch_dtype=vq_init_dtype,
            enable_autocast=bool(args.vq_init_autocast),
            world_size=world_size,
        )
        merged_samples = samples
        if dist.is_initialized() and world_size > 1:
            local = samples
            if local is None:
                local = torch.zeros(
                    (0, model.vq_task.dim),
                    device=device,
                    dtype=model.vq_task.codebook.dtype,
                )
            else:
                local = local.to(device=device, dtype=model.vq_task.codebook.dtype)
            local_size = torch.tensor([local.size(0)], device=device, dtype=torch.long)
            size_list = [torch.zeros_like(local_size) for _ in range(world_size)]
            dist.all_gather(size_list, local_size)
            max_size = max(int(size.item()) for size in size_list)
            if local.size(0) < max_size:
                pad = torch.zeros(
                    (max_size - local.size(0), local.size(1)),
                    device=device,
                    dtype=local.dtype,
                )
                local = torch.cat([local, pad], dim=0)
            gather_list = [torch.zeros_like(local) for _ in range(world_size)]
            dist.all_gather(gather_list, local)
            merged_samples = None
            if rank == 0:
                chunks = []
                for tensor, size in zip(gather_list, size_list):
                    size_int = int(size.item())
                    if size_int > 0:
                        chunks.append(tensor[:size_int].cpu())
                if chunks:
                    merged = torch.cat(chunks, dim=0)
                    if merged.size(0) > vq_init_samples:
                        merged = merged[:vq_init_samples]
                    merged_samples = merged
        if rank == 0 and merged_samples is not None:
            model.vq_task.initialize_codebook(
                merged_samples.to(device),
                max_samples=int(vq_init_samples),
                seed=int(args.vq_init_seed),
                spherical=bool(args.vq_init_spherical),
            )
            _rank0_print(rank, "[stage done] VQ codebook initialized with kmeans++")
        if dist.is_initialized() and world_size > 1:
            dist.barrier(device_ids=[device.index] if device.type == "cuda" else None)
            with torch.no_grad():
                dist.broadcast(model.vq_task.codebook, src=0)
                dist.broadcast(model.vq_task.ema_w, src=0)
                dist.broadcast(model.vq_task.ema_cluster_size, src=0)
            dist.barrier(device_ids=[device.index] if device.type == "cuda" else None)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    _rank0_print(rank, "[stage done] model initialized")
    _maybe_apply_lora(model, args)
    if args.use_lora and args.freeze_lm_when_lora:
        _freeze_non_lora_lm_params(model, rank)
    _validate_lora(model, args, rank)
    if args.freeze_lm_when_no_lora and not args.use_lora:
        _set_lm_trainable(model, False, rank)
    _rank0_print(rank, "[stage done] lora applied" if args.use_lora else "[stage done] lora skipped")
    if args.gradient_checkpointing:
        lm_trainable = any(param.requires_grad for param in model.judge_lm.parameters())
        encoder_ckpt = any(param.requires_grad for param in model.shared_encoder.parameters())
        model.set_gradient_checkpointing(
            encoder=encoder_ckpt,
            lm=lm_trainable,
            use_reentrant=bool(args.checkpoint_reentrant),
        )
        zero_stage = _resolve_zero_stage(args.deepspeed_config)
        zero_stage_str = str(zero_stage) if zero_stage is not None else "unknown"
        _rank0_print(
            rank,
            "[stage done] gradient checkpointing enabled "
            f"(encoder={encoder_ckpt} lm={lm_trainable} "
            f"reentrant={bool(args.checkpoint_reentrant)} "
            f"zero_stage={zero_stage_str})",
        )
    else:
        model.set_gradient_checkpointing(encoder=False, lm=False, use_reentrant=False)

    optimizer_params = _build_optimizer_params(
        model,
        encoder_lr=args.encoder_lr,
        lm_lr=args.lm_lr,
        lora_lr=args.lora_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        head_weight_decay=args.head_weight_decay,
    )
    zero_stage = _resolve_zero_stage(args.deepspeed_config)
    optimizer = _build_optimizer(optimizer_params, args, rank, zero_stage)

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
        if sched_name in {"none", "disable", "disabled"} and "scheduler" in ds_config:
            _rank0_print(
                rank,
                "[warn] scheduler disabled via args; removing scheduler from deepspeed config",
            )
            ds_config.pop("scheduler", None)
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
    lora_params: List[torch.nn.Parameter] = []
    if args.debug_aux_checks:
        for name, param in engine.module.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("shared_encoder."):
                encoder_params.append(param)
            elif name.startswith("judge_lm."):
                lm_params.append(param)
                if args.use_lora and "lora_" in name:
                    lora_params.append(param)

    use_amp = bool(args.torch_autocast)
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

    _rank0_print(rank, "DeepSpeed config:", ds_config)
    _rank0_print(
        rank,
        "Loss config:",
        {
            "loss_schedule": args.loss_schedule,
            "disable_compactor": args.disable_compactor,
            "disable_z_prompt_insertion": args.disable_z_prompt_insertion,
            "lambda_bias": args.lambda_bias,
            "lambda_compression": args.lambda_compression,
            "compact_keep_init": args.compact_keep_init,
            "compact_keep_final": args.compact_keep_final,
            "kl_weight_max": args.kl_weight_max,
            "z_dropout": args.z_dropout,
            "low_recon_weight": args.low_recon_weight,
            "vq_task_weight": args.vq_task_weight,
            "vq_usage_weight": args.vq_usage_weight,
            "vq_align_weight": args.vq_align_weight,
            "disentangle_weight": args.disentangle_weight,
            "length_bin_weight": args.length_bin_weight,
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
    ema_lm_loss: Optional[float] = None
    last_update_diag: Dict[str, float] = {}
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
            "low_recon_loss": 0.0,
            "vq_task_loss": 0.0,
            "vq_align_loss": 0.0,
            "vq_usage_loss": 0.0,
            "disentangle_loss": 0.0,
            "vq_task_perplexity": 0.0,
            "vq_task_perplexity_ema": 0.0,
            "vq_task_active_codes": 0.0,
            "vq_task_active_fraction": 0.0,
            "vq_task_usage_entropy": 0.0,
            "vq_task_unique_codes": 0.0,
            "vq_task_dead_fraction": 0.0,
            "vq_task_avg_distance": 0.0,
            "nll_bin_loss": 0.0,
            "ttr_bin_loss": 0.0,
            "length_bin_loss": 0.0,
            "nll_bin_mae": 0.0,
            "ttr_bin_mae": 0.0,
            "length_bin_mae": 0.0,
            "compression_loss": 0.0,
            "mask_group_loss": 0.0,
            "mask_loss": 0.0,
            "consistency_loss": 0.0,
            "lm_drop_count": 0.0,
            "lm_drop_seen": 0.0,
            "lm_drop_maxlen": 0.0,
            "lm_drop_min_target": 0.0,
            "steps": 0,
        }

        for batch in epoch_loader:
            step_start = time.perf_counter()
            step_index = step + 1
            accum_boundary = step_index % args.grad_accum_steps == 0
            optim_step = step_index // args.grad_accum_steps
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            if total_update_steps > 0:
                progress = min(1.0, float(optim_step) / float(total_update_steps))
            else:
                progress = 0.0
            device = batch["lm_input_ids"].device
            disable_compactor = bool(args.disable_compactor)
            disable_z_prompt_insertion = bool(args.disable_z_prompt_insertion)
            if args.loss_schedule == "dynamic":
                keep_init = float(args.compact_keep_init)
                keep_final = float(args.compact_keep_final)
                keep_ratio = _piecewise_linear(
                    progress,
                    [
                        (0.0, keep_init),
                        (0.2, keep_init),
                        (0.6, keep_final),
                        (1.0, keep_final),
                    ],
                )
                keep_ratio = min(max(keep_ratio, 0.0), 1.0)
                kl_weight = _piecewise_linear(
                    progress,
                    [
                        (0.0, 0.1 * float(args.kl_weight_max)),
                        (0.2, 0.1 * float(args.kl_weight_max)),
                        (0.5, float(args.kl_weight_max)),
                        (1.0, float(args.kl_weight_max)),
                    ],
                )
                lambda_compression = float(args.lambda_compression) * _piecewise_linear(
                    progress,
                    [
                        (0.0, 0.1),
                        (0.1, 0.1),
                        (0.4, 1.0),
                        (1.0, 0.3),
                    ],
                )
                lambda_bias = float(args.lambda_bias) * _piecewise_linear(
                    progress,
                    [
                        (0.0, 0.2),
                        (0.05, 0.2),
                        (0.3, 1.0),
                        (0.7, 1.0),
                        (1.0, 0.2),
                    ],
                )
                disentangle_weight = float(args.disentangle_weight) * _piecewise_linear(
                    progress,
                    [
                        (0.0, 0.0),
                        (0.2, 0.0),
                        (0.6, 1.0),
                        (1.0, 0.5),
                    ],
                )
                vq_commit_ramp = _piecewise_linear(
                    progress,
                    [
                        (0.0, 0.0),
                        (0.1, 1.0),
                        (1.0, 1.0),
                    ],
                )
                mask_group_scale = 1.0
            else:
                keep_ratio = min(max(float(args.compact_keep_init), 0.0), 1.0)
                kl_weight = float(args.kl_weight_max)
                lambda_compression = float(args.lambda_compression)
                lambda_bias = float(args.lambda_bias)
                disentangle_weight = float(args.disentangle_weight)
                vq_commit_ramp = 1.0
                mask_group_scale = 1.0
            if disable_compactor:
                keep_ratio = 1.0
                kl_weight = 0.0
                mask_group_scale = 0.0
            z_dropout = float(args.z_dropout)
            batch["compact_prior"] = keep_ratio
            batch["z_task_dropout"] = z_dropout
            batch["compact_mask_scale"] = mask_group_scale
            batch["disable_compactor"] = disable_compactor
            batch["disable_z_prompt_insertion"] = disable_z_prompt_insertion
            batch["compute_compact_kl"] = kl_weight > 0.0
            batch["enable_low_recon"] = float(args.low_recon_weight) > 0.0

            diag_metrics: Dict[str, float] = {}
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = engine(batch)
                proxy_losses = _compute_bias_terms(outputs, batch)
                low_recon_loss = proxy_losses["low_recon_loss"]
                nll_bin_loss = proxy_losses["nll_bin_loss"]
                ttr_bin_loss = proxy_losses["ttr_bin_loss"]
                length_bin_loss = proxy_losses["length_bin_loss"]
                nll_bin_mae = proxy_losses.get("nll_bin_mae", outputs["lm_loss"].new_tensor(0.0))
                ttr_bin_mae = proxy_losses.get("ttr_bin_mae", outputs["lm_loss"].new_tensor(0.0))
                length_bin_mae = proxy_losses.get("length_bin_mae", outputs["lm_loss"].new_tensor(0.0))
                vq_task_loss = outputs.get("vq_task_loss", outputs["lm_loss"].new_tensor(0.0))
                vq_align_loss = outputs.get("vq_align_loss", outputs["lm_loss"].new_tensor(0.0))
                vq_usage_loss = outputs.get(
                    "vq_task_usage_loss", outputs["lm_loss"].new_tensor(0.0)
                )
                disentangle_loss = outputs.get("disentangle_loss", outputs["lm_loss"].new_tensor(0.0))

                mask_loss = outputs.get(
                    "compact_mask_loss", outputs["lm_loss"].new_tensor(0.0)
                )
                consistency_loss = outputs.get(
                    "compact_con_loss", outputs["lm_loss"].new_tensor(0.0)
                )
                mask_group_loss = (
                    args.mask_loss_weight * mask_loss
                    + args.consistency_loss_weight * consistency_loss
                )
                compression_loss = (
                    args.vq_task_weight * vq_task_loss * vq_commit_ramp
                    + args.vq_align_weight * vq_align_loss
                    + args.vq_usage_weight * vq_usage_loss * vq_commit_ramp
                )
                bias_loss = (
                    + args.low_recon_weight * low_recon_loss
                    + args.nll_bin_weight * nll_bin_loss
                    + args.ttr_bin_weight * ttr_bin_loss
                    + args.length_bin_weight * length_bin_loss
                )
                core_lm_loss = outputs["lm_loss"] + kl_weight * outputs["compact_kl_loss"]
            total_loss = (
                core_lm_loss
                + (args.mask_group_weight * mask_group_scale) * mask_group_loss
                + lambda_compression * compression_loss
                + lambda_bias * bias_loss
                + disentangle_weight * disentangle_loss
            )
            if args.debug_loss_spike and rank == 0:
                lm_loss_val = float(outputs["lm_loss"].detach().item())
                ema_ref = ema_lm_loss if ema_lm_loss is not None else lm_loss_val
                spike = False
                if args.debug_loss_spike_threshold > 0:
                    spike = spike or lm_loss_val >= float(args.debug_loss_spike_threshold)
                if args.debug_loss_spike_multiplier > 0 and ema_ref > 0:
                    spike = spike or lm_loss_val >= float(args.debug_loss_spike_multiplier) * ema_ref
                if spike:
                    labels = batch.get("lm_labels")
                    attention = batch.get("lm_attention_mask")
                    label_counts = None
                    pad_ratio = None
                    filtered = batch.get("debug_lm_filtered")
                    filtered_count = 0
                    if torch.is_tensor(filtered):
                        filtered_count = int(filtered.sum().item())
                    if torch.is_tensor(labels) and torch.is_tensor(attention):
                        label_counts = labels.ne(-100).sum(dim=1).detach().cpu()
                        pad_counts = attention.eq(0).sum(dim=1).detach().cpu()
                        pad_ratio = pad_counts.float() / max(1, attention.size(1))
                    truncated = batch.get("debug_lm_truncated")
                    prompt_tokens = batch.get("debug_prompt_tokens")
                    prompt_preview = batch.get("debug_prompt_preview")
                    output_preview = batch.get("debug_output_preview")
                    trunc_list = (
                        truncated.detach().cpu().tolist()
                        if torch.is_tensor(truncated)
                        else None
                    )
                    prompt_tok_list = (
                        prompt_tokens.detach().cpu().tolist()
                        if torch.is_tensor(prompt_tokens)
                        else None
                    )
                    _rank0_print(
                        rank,
                        "[loss spike] "
                        f"step={step} lm_loss={lm_loss_val:.4f} total_loss={total_loss.item():.4f} "
                        f"ema={ema_ref:.4f} "
                        f"labels_mean={(label_counts.float().mean().item() if label_counts is not None else 0.0):.2f} "
                        f"labels_min={(int(label_counts.min().item()) if label_counts is not None else 0)} "
                        f"pad_ratio_mean={(pad_ratio.mean().item() if pad_ratio is not None else 0.0):.3f} "
                        f"truncated={(sum(trunc_list) if trunc_list is not None else 0)} "
                        f"filtered={filtered_count}",
                    )
                    if label_counts is not None:
                        max_samples = max(1, int(args.debug_loss_spike_max_samples))
                        idx = torch.argsort(label_counts, dim=0)
                        for rank_idx in range(min(max_samples, idx.numel())):
                            i = int(idx[rank_idx].item())
                            label_count = int(label_counts[i].item())
                            pad_r = float(pad_ratio[i].item()) if pad_ratio is not None else 0.0
                            trunc_flag = trunc_list[i] if trunc_list is not None else 0
                            prompt_tok = prompt_tok_list[i] if prompt_tok_list is not None else 0
                            _rank0_print(
                                rank,
                                "[loss spike sample] "
                                f"idx={i} label_tokens={label_count} pad_ratio={pad_r:.3f} "
                                f"truncated={trunc_flag} prompt_tokens={prompt_tok}",
                            )
                            if prompt_preview is not None:
                                _rank0_print(
                                    rank,
                                    f"[loss spike sample] prompt='{prompt_preview[i]}'",
                                )
                            if output_preview is not None:
                                _rank0_print(
                                    rank,
                                    f"[loss spike sample] output='{output_preview[i]}'",
                                )
                ema_lm_loss = 0.95 * ema_ref + 0.05 * lm_loss_val
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
                if args.use_lora:
                    lora_lm_grad = _grad_norm_from_loss(lm_term, lora_params)
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
                if args.use_lora:
                    diag_metrics["diag/grad_norm/lm/lora"] = float(
                        lora_lm_grad.item()
                    )
            elif args.debug_aux_checks and step % args.debug_aux_checks_interval == 0:
                diag_metrics["diag/grad_norm/skipped_checkpointing"] = 1.0

            if args.bf16:
                if total_loss.dtype != torch.float32:
                    total_loss = total_loss.float()
            elif model_dtype is not None and total_loss.dtype != model_dtype:
                total_loss = total_loss.to(model_dtype)

            sequence_len = 0
            lm_attention = batch.get("lm_attention_mask")
            if torch.is_tensor(lm_attention):
                sequence_len = int(lm_attention.size(-1))
            else:
                lm_ids = batch.get("lm_input_ids")
                if torch.is_tensor(lm_ids):
                    sequence_len = int(lm_ids.size(-1))

            try:
                engine.backward(total_loss)
            except RuntimeError as exc:
                msg = str(exc).lower()
                if "out of memory" not in msg and "cublas_status_alloc_failed" not in msg:
                    raise
                exc_msg = str(exc).splitlines()[0] if exc else "CUDA out of memory"
                print(
                    f"[warn] OOM error occurred | sequence_len={sequence_len} | details={exc_msg}"
                )
                engine.zero_grad()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
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
                if args.use_lora:
                    last_update_diag["diag/grad_norm_total/lora"] = _param_grad_norm(
                        lora_params, total_loss.device
                    )
            engine.step()


            totals["loss"] += total_loss.item()
            totals["lm_loss"] += outputs["lm_loss"].item()
            totals["compact_kl_loss"] += outputs["compact_kl_loss"].item()
            totals["low_recon_loss"] += low_recon_loss.item()
            totals["vq_task_loss"] += vq_task_loss.item()
            totals["vq_align_loss"] += vq_align_loss.item()
            totals["vq_usage_loss"] += vq_usage_loss.item()
            totals["disentangle_loss"] += disentangle_loss.item()
            totals["vq_task_perplexity"] += float(outputs.get("vq_task_perplexity", 0.0))
            totals["vq_task_perplexity_ema"] += float(outputs.get("vq_task_perplexity_ema", 0.0))
            totals["vq_task_active_codes"] += float(outputs.get("vq_task_active_codes", 0.0))
            totals["vq_task_active_fraction"] += float(outputs.get("vq_task_active_fraction", 0.0))
            totals["vq_task_usage_entropy"] += float(outputs.get("vq_task_usage_entropy", 0.0))
            totals["vq_task_unique_codes"] += float(outputs.get("vq_task_unique_codes", 0.0))
            totals["vq_task_dead_fraction"] += float(outputs.get("vq_task_dead_fraction", 0.0))
            totals["vq_task_avg_distance"] += float(outputs.get("vq_task_avg_distance", 0.0))
            totals["nll_bin_loss"] += nll_bin_loss.item()
            totals["ttr_bin_loss"] += ttr_bin_loss.item()
            totals["length_bin_loss"] += length_bin_loss.item()
            totals["nll_bin_mae"] += nll_bin_mae.item()
            totals["ttr_bin_mae"] += ttr_bin_mae.item()
            totals["length_bin_mae"] += length_bin_mae.item()
            totals.setdefault("compression_loss", 0.0)
            totals["compression_loss"] += compression_loss.item()
            totals["mask_group_loss"] += mask_group_loss.item()
            totals["mask_loss"] += mask_loss.item()
            totals["consistency_loss"] += consistency_loss.item()
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
                            "low_recon_loss": low_recon_loss.item(),
                            "vq_task_loss": vq_task_loss.item(),
                            "vq_align_loss": vq_align_loss.item(),
                            "vq_usage_loss": vq_usage_loss.item(),
                            "disentangle_loss": disentangle_loss.item(),
                            "vq/task_commitment_loss": float(outputs.get("vq_task_commitment_loss", 0.0)),
                            "vq/task_codebook_loss": float(outputs.get("vq_task_codebook_loss", 0.0)),
                            "vq/task_perplexity": float(outputs.get("vq_task_perplexity", 0.0)),
                            "vq/task_perplexity_ema": float(outputs.get("vq_task_perplexity_ema", 0.0)),
                            "vq/task_usage_loss": float(outputs.get("vq_task_usage_loss", 0.0)),
                            "vq/task_active_codes": float(outputs.get("vq_task_active_codes", 0.0)),
                            "vq/task_active_fraction": float(outputs.get("vq_task_active_fraction", 0.0)),
                            "vq/task_usage_entropy": float(outputs.get("vq_task_usage_entropy", 0.0)),
                            "vq/task_unique_codes": float(outputs.get("vq_task_unique_codes", 0.0)),
                            "vq/task_dead_fraction": float(outputs.get("vq_task_dead_fraction", 0.0)),
                            "vq/task_avg_distance": float(outputs.get("vq_task_avg_distance", 0.0)),
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
                            "weights/kl_weight": kl_weight,
                            "weights/keep_ratio": keep_ratio,
                            "weights/z_task_dropout": z_dropout,
                            "weights/vq_commit_scale": vq_commit_ramp,
                            "weights/low_recon": args.low_recon_weight,
                            "weights/vq_task": args.vq_task_weight,
                            "weights/vq_usage": args.vq_usage_weight,
                            "weights/vq_align": args.vq_align_weight,
                            "weights/disentangle": disentangle_weight,
                            "weights/nll_bin": args.nll_bin_weight,
                            "weights/ttr_bin": args.ttr_bin_weight,
                            "weights/length_bin": args.length_bin_weight,
                            "weights/mask_group": args.mask_group_weight * mask_group_scale,
                            "weights/mask_group_scale": mask_group_scale,
                            "weights/mask_loss": args.mask_loss_weight,
                            "weights/consistency_loss": args.consistency_loss_weight,
                            "data/lm_drop_count": float(drop_count.item()) if torch.is_tensor(drop_count) else 0.0,
                            "data/lm_drop_seen": float(drop_seen.item()) if torch.is_tensor(drop_seen) else 0.0,
                            "data/lm_drop_ratio": float(
                                (drop_count.item() / drop_seen.item())
                                if torch.is_tensor(drop_count) and torch.is_tensor(drop_seen) and drop_seen.item() > 0
                                else 0.0
                            ),
                            "data/lm_drop_maxlen_count": float(drop_maxlen.item())
                            if torch.is_tensor(drop_maxlen)
                            else 0.0,
                            "data/lm_drop_min_target_count": float(drop_min_target.item())
                            if torch.is_tensor(drop_min_target)
                            else 0.0,
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
                    engine=engine,
                )
            step += 1

        if rank == 0:
            denom = max(1, totals["steps"])
            metrics = {k: (v / denom) if k != "steps" else v for k, v in totals.items()}
            seen = totals.get("lm_drop_seen", 0.0)
            if seen > 0:
                metrics["lm_drop_ratio"] = totals.get("lm_drop_count", 0.0) / seen
                metrics["lm_drop_maxlen_ratio"] = totals.get("lm_drop_maxlen", 0.0) / seen
                metrics["lm_drop_min_target_ratio"] = totals.get("lm_drop_min_target", 0.0) / seen
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
                engine=engine,
            )
    if swanlab_client is not None and rank == 0:
        finish_swanlab(swanlab_client)


if __name__ == "__main__":
    main()
