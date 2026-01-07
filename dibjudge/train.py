from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from transformers import get_scheduler

from .modeling import DIBJudgeModel
from .proxy_tasks import compute_proxy_losses


@dataclass
class TrainConfig:
    lr: float = 2e-5
    encoder_lr: float = 2e-5
    lm_lr: float = 2e-5
    lora_lr: float = 2e-4
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    head_weight_decay: float = 0.001
    warmup_steps: int = 0
    warmup_ratio: Optional[float] = 0.03
    total_steps: int = 10000
    scheduler_type: str = "cosine"
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    use_amp: bool = True
    amp_dtype: torch.dtype = torch.float16
    lambda_bias: float = 1.0
    grl_lambda: float = 1.0
    grl_start_ratio: float = 0.3
    grl_gamma: float = 10.0
    bias_decoder_steps: int = 1
    bottleneck_noise_alpha: float = 8.0
    bottleneck_noise_warmup_ratio: Optional[float] = 0.2
    eng_domain_weight: float = 1.0
    low_recon_weight: float = 0.5
    z_l2_weight: float = 0.1
    nll_bin_weight: float = 0.5
    ttr_bin_weight: float = 0.5
    length_bin_weight: float = 0.5
    lambda_compression: float = 1.0
    lambda_compression_warmup_ratio: Optional[float] = 0.05
    mask_loss_weight: float = 1.0
    consistency_loss_weight: float = 1.0
    debug_aux_checks: bool = False
    debug_aux_checks_interval: int = 200


def move_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _param_grad_norm(params: List[nn.Parameter], device: torch.device) -> float:
    total = torch.zeros((), device=device)
    for param in params:
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        total += grad.pow(2).sum()
    return float(total.sqrt().item())


def _grad_norm_from_loss(
    loss: torch.Tensor, params: List[nn.Parameter]
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


def _compression_coverage(batch: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    if "lm_attention_mask" not in batch:
        return None
    prompt_mask = batch["lm_attention_mask"].bool()
    if "lm_labels" in batch:
        prompt_mask = prompt_mask & batch["lm_labels"].eq(-100)
    response_types = batch.get("lm_response_types")
    if response_types is None:
        return prompt_mask.float().mean()
    response_mask = response_types > 0
    covered = prompt_mask & response_mask
    denom = prompt_mask.sum().clamp_min(1)
    return covered.sum().float() / denom


def _group_weight_decay(
    params: Dict[str, nn.Parameter], weight_decay: float, lr: float
) -> List[Dict[str, object]]:
    decay, no_decay = [], []
    for name, param in params.items():
        if param.ndim == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay, "lr": lr},
        {"params": no_decay, "weight_decay": 0.0, "lr": lr},
    ]


def _collect_param_groups(model: nn.Module) -> Dict[str, Dict[str, nn.Parameter]]:
    groups = {
        "encoder": {},
        "lm": {},
        "lora": {},
        "head": {},
    }
    head_prefixes = (
        "eng_domain_head.",
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
            groups["lora"][name] = param
            continue
        if name.startswith("shared_encoder."):
            groups["encoder"][name] = param
            continue
        if name.startswith("judge_lm."):
            groups["lm"][name] = param
            continue
        if name.startswith(head_prefixes):
            groups["head"][name] = param
            continue
        groups["head"][name] = param
    return groups


def create_optimizers(model: DIBJudgeModel, config: TrainConfig) -> AdamW:
    groups = _collect_param_groups(model)
    main_groups: List[Dict[str, object]] = []
    if groups["encoder"]:
        main_groups.extend(_group_weight_decay(groups["encoder"], config.weight_decay, config.encoder_lr))
    if groups["lm"]:
        main_groups.extend(_group_weight_decay(groups["lm"], config.weight_decay, config.lm_lr))
    if groups["lora"]:
        main_groups.extend(_group_weight_decay(groups["lora"], config.head_weight_decay, config.lora_lr))
    if groups["head"]:
        main_groups.extend(_group_weight_decay(groups["head"], config.head_weight_decay, config.head_lr))
    if not main_groups:
        raise ValueError("No trainable parameters found for main optimizer.")
    main_optimizer = AdamW(main_groups)
    return main_optimizer


def create_scheduler(optimizer: AdamW, config: TrainConfig):
    sched_name = config.scheduler_type
    if sched_name is None:
        return None
    sched_name = str(sched_name).lower()
    if sched_name in {"none", "disable", "disabled"}:
        return None
    warmup_steps = config.warmup_steps
    if config.warmup_ratio is not None:
        if config.warmup_ratio < 0:
            raise ValueError("warmup_ratio must be >= 0.")
        warmup_steps = int(config.total_steps * config.warmup_ratio)
    return get_scheduler(
        sched_name,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=config.total_steps,
    )


def _lr_stats(optimizer: AdamW, prefix: str) -> Dict[str, float]:
    lrs = [group.get("lr", 0.0) for group in optimizer.param_groups]
    if not lrs:
        return {}
    return {
        f"lr/{prefix}_min": float(min(lrs)),
        f"lr/{prefix}_max": float(max(lrs)),
        f"lr/{prefix}_mean": float(sum(lrs) / len(lrs)),
    }


def _ramp_weight(weight: float, step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return weight
    return weight * min(1.0, float(step + 1) / float(warmup_steps))


def _warmup_steps_from_ratio(total_steps: int, ratio: Optional[float]) -> int:
    if ratio is None:
        return 0
    if ratio < 0:
        raise ValueError("warmup_ratio must be >= 0.")
    return int(total_steps * ratio)


def _phase_boundary(total_steps: int, grl_start_ratio: float) -> int:
    if total_steps <= 0:
        return 0
    ratio = min(1.0, max(0.0, float(grl_start_ratio)))
    core_steps = int(total_steps * ratio)
    return min(total_steps, max(0, core_steps))


def _compute_bias_terms(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return compute_proxy_losses(outputs, batch)


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


def _ramp_linear(start: float, end: float, step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return end
    progress = min(1.0, float(step + 1) / float(warmup_steps))
    return start + (end - start) * progress


def train_one_epoch(
    model: DIBJudgeModel,
    dataloader: Iterable[Dict[str, torch.Tensor]],
    optimizer: AdamW,
    scheduler: Optional[object],
    config: TrainConfig,
    device: torch.device,
    start_step: int = 0,
    log_fn: Optional[Callable[[Dict[str, float], int], None]] = None,
    log_interval: int = 0,
) -> Dict[str, float]:
    model.train()
    scaler = GradScaler(
        "cuda",
        enabled=config.use_amp and config.amp_dtype == torch.float16,
    )
    encoder_params: List[nn.Parameter] = []
    lm_params: List[nn.Parameter] = []
    if config.debug_aux_checks:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("shared_encoder."):
                encoder_params.append(param)
            elif name.startswith("judge_lm."):
                lm_params.append(param)
    totals = {
        "loss": 0.0,
        "lm_loss": 0.0,
        "compact_kl_loss": 0.0,
        "domain_loss": 0.0,
        "low_recon_loss": 0.0,
        "z_l2_loss": 0.0,
        "nll_bin_loss": 0.0,
        "ttr_bin_loss": 0.0,
        "length_bin_loss": 0.0,
        "nll_bin_mae": 0.0,
        "ttr_bin_mae": 0.0,
        "length_bin_mae": 0.0,
        "mask_loss": 0.0,
        "consistency_loss": 0.0,
        "compression_loss": 0.0,
        "steps": 0,
    }

    optimizer.zero_grad(set_to_none=True)
    last_grad_norm = 0.0
    last_update_diag: Dict[str, float] = {}
    last_clip_diag: Dict[str, float] = {}
    grl_start = _phase_boundary(config.total_steps, config.grl_start_ratio)
    for step, batch in enumerate(dataloader, start=1):
        step_start = time.perf_counter()
        batch = move_to_device(batch, device)
        # Marginals are built online from the current batch inside the model.
        global_step = start_step + step - 1
        accum_boundary = step % config.grad_accum_steps == 0
        optim_step = step // config.grad_accum_steps
        warmup_phase_steps = grl_start if grl_start > 0 else config.total_steps
        compression_warmup_steps = _warmup_steps_from_ratio(
            warmup_phase_steps, config.lambda_compression_warmup_ratio
        )
        noise_warmup_steps = _warmup_steps_from_ratio(
            warmup_phase_steps, config.bottleneck_noise_warmup_ratio
        )
        lambda_compression = _ramp_linear(
            0.0, config.lambda_compression, global_step, compression_warmup_steps
        )
        if global_step < grl_start:
            progress = (global_step + 1) / float(max(1, grl_start))
            lambda_bias = config.lambda_bias * min(1.0, progress)
            grl_lambda = 0.0
            bias_detach = True
        else:
            lambda_bias = config.lambda_bias
            grl_lambda = _grl_schedule(
                config.grl_lambda,
                global_step,
                config.total_steps,
                grl_start,
                config.grl_gamma,
            )
            bias_detach = False
        noise_alpha = _ramp_weight(
            config.bottleneck_noise_alpha, global_step, noise_warmup_steps
        )
        batch["grl_lambda"] = batch["lm_input_ids"].new_tensor(grl_lambda)
        batch["bottleneck_noise_alpha"] = batch["lm_input_ids"].new_tensor(noise_alpha)
        batch["bias_detach"] = bias_detach
        diag_metrics: Dict[str, float] = {}
        with autocast("cuda", enabled=config.use_amp, dtype=config.amp_dtype):
            outputs = model(batch)
            if outputs["lm_loss"] is None:
                raise ValueError("lm_labels must be provided for training.")
            proxy_losses = _compute_bias_terms(outputs, batch)
            domain_loss = proxy_losses["domain_loss"]
            low_recon_loss = proxy_losses["low_recon_loss"]
            z_l2_loss = proxy_losses["z_l2_loss"]
            nll_bin_loss = proxy_losses["nll_bin_loss"]
            ttr_bin_loss = proxy_losses["ttr_bin_loss"]
            length_bin_loss = proxy_losses["length_bin_loss"]
            nll_bin_mae = proxy_losses.get("nll_bin_mae", outputs["lm_loss"].new_tensor(0.0))
            ttr_bin_mae = proxy_losses.get("ttr_bin_mae", outputs["lm_loss"].new_tensor(0.0))
            length_bin_mae = proxy_losses.get("length_bin_mae", outputs["lm_loss"].new_tensor(0.0))

            mask_loss = outputs.get("compact_mask_loss", torch.zeros((), device=outputs["lm_loss"].device))
            consistency_loss = outputs.get(
                "compact_con_loss", torch.zeros((), device=outputs["lm_loss"].device)
            )
            compression_loss = (
                config.mask_loss_weight * mask_loss
                + config.consistency_loss_weight * consistency_loss
            )
            bias_loss = (
                config.eng_domain_weight * domain_loss
                + config.low_recon_weight * low_recon_loss
                + config.z_l2_weight * z_l2_loss
                + config.nll_bin_weight * nll_bin_loss
                + config.ttr_bin_weight * ttr_bin_loss
                + config.length_bin_weight * length_bin_loss
            )
            core_lm_loss = outputs["lm_loss"] + outputs["compact_kl_loss"]
            if config.debug_aux_checks and step % config.debug_aux_checks_interval == 0:
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
                coverage = _compression_coverage(batch)
                if coverage is not None:
                    diag_metrics["diag/compression_coverage"] = float(coverage.item())
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
            total_loss = (
                core_lm_loss
                + lambda_compression * compression_loss
                + lambda_bias * bias_loss
            )
            total_loss = total_loss / config.grad_accum_steps

        scaler.scale(total_loss).backward()

        if accum_boundary:
            scaler.unscale_(optimizer)
            if config.debug_aux_checks and (encoder_params or lm_params):
                last_update_diag = {
                    "diag/grad_norm_total/encoder": _param_grad_norm(
                        encoder_params, total_loss.device
                    ),
                    "diag/grad_norm_total/lm": _param_grad_norm(
                        lm_params, total_loss.device
                    ),
                }
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            last_grad_norm = float(grad_norm)
            clip_ratio = (
                last_grad_norm / config.max_grad_norm if config.max_grad_norm > 0 else 0.0
            )
            last_clip_diag = {
                "grad_norm/ratio": float(clip_ratio),
                "grad_norm/clipped": float(last_grad_norm > config.max_grad_norm + 1e-6),
            }
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

            if config.bias_decoder_steps > 1 and global_step >= grl_start:
                for _ in range(config.bias_decoder_steps - 1):
                    optimizer.zero_grad(set_to_none=True)
                    bias_batch = dict(batch)
                    bias_batch["grl_lambda"] = batch["lm_input_ids"].new_tensor(0.0)
                    bias_batch["bias_detach"] = True
                    with autocast("cuda", enabled=config.use_amp, dtype=config.amp_dtype):
                        bias_outputs = model(bias_batch)
                        proxy_losses_b = _compute_bias_terms(bias_outputs, bias_batch)
                        bias_loss_b = (
                            config.eng_domain_weight * proxy_losses_b["domain_loss"]
                            + config.low_recon_weight * proxy_losses_b["low_recon_loss"]
                            + config.z_l2_weight * proxy_losses_b["z_l2_loss"]
                            + config.nll_bin_weight * proxy_losses_b["nll_bin_loss"]
                            + config.ttr_bin_weight * proxy_losses_b["ttr_bin_loss"]
                            + config.length_bin_weight * proxy_losses_b["length_bin_loss"]
                        )
                        bias_loss_b = bias_loss_b * lambda_bias
                    scaler.scale(bias_loss_b).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

        total_loss_value = total_loss.item() * config.grad_accum_steps
        totals["loss"] += total_loss_value
        totals["lm_loss"] += outputs["lm_loss"].item()
        totals["compact_kl_loss"] += outputs["compact_kl_loss"].item()
        totals["domain_loss"] += domain_loss.item()
        totals["low_recon_loss"] += low_recon_loss.item()
        totals["z_l2_loss"] += z_l2_loss.item()
        totals["nll_bin_loss"] += nll_bin_loss.item()
        totals["ttr_bin_loss"] += ttr_bin_loss.item()
        totals["length_bin_loss"] += length_bin_loss.item()
        totals["nll_bin_mae"] += nll_bin_mae.item()
        totals["ttr_bin_mae"] += ttr_bin_mae.item()
        totals["length_bin_mae"] += length_bin_mae.item()
        totals["mask_loss"] += mask_loss.item()
        totals["consistency_loss"] += consistency_loss.item()
        totals.setdefault("compression_loss", 0.0)
        totals["compression_loss"] += compression_loss.item()
        totals["steps"] += 1
        if log_fn and log_interval > 0 and step % log_interval == 0:
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
            perf = {
                "perf/step_time": step_time,
            }
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
            log_fn(
                {
                    "loss": total_loss_value,
                    "lm_loss": outputs["lm_loss"].item(),
                    "domain_loss": domain_loss.item(),
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
                    "mask_loss": mask_loss.item(),
                    "consistency_loss": consistency_loss.item(),
                    "compact_kl_loss": outputs["compact_kl_loss"].item(),
                    "compact/pi_mean": float(outputs.get("compact_pi_mean", 0.0)),
                    "compact/mask_mean": float(outputs.get("compact_mask_mean", 0.0)),
                    "compact/pi_saturation": float(outputs.get("compact_pi_saturation", 0.0)),
                    "compact/kl_loss": outputs["compact_kl_loss"].item(),
                    "weights/lambda_bias": lambda_bias,
                    "weights/lambda_compression": lambda_compression,
                    "weights/grl_lambda": grl_lambda,
                    "weights/bottleneck_noise_alpha": noise_alpha,
                    "weights/eng_domain": config.eng_domain_weight,
                    "weights/low_recon": config.low_recon_weight,
                    "weights/z_l2": config.z_l2_weight,
                    "weights/nll_bin": config.nll_bin_weight,
                    "weights/ttr_bin": config.ttr_bin_weight,
                    "weights/length_bin": config.length_bin_weight,
                    "weights/mask_loss": config.mask_loss_weight,
                    "weights/consistency_loss": config.consistency_loss_weight,
                    "grad_norm/main": last_grad_norm,
                    **last_update_diag,
                    **last_clip_diag,
                    "batch/lm_tokens": lm_tokens,
                    "batch/original_tokens": original_tokens,
                    **_lr_stats(optimizer, "main"),
                    **perf,
                    **mem,
                    **diag_metrics,
                },
                global_step,
            )

    denom = max(1, totals["steps"])
    return {k: (v / denom) if k != "steps" else v for k, v in totals.items()}
