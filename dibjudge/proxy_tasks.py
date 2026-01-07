from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class ProxyTaskConfig:
    nll_bins: Tuple[float, ...] = (0.0, 2.3026, 2.9957, 3.6889, 4.3820, 5.0752, 13.8155)
    ttr_bins: Tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    length_bins: Tuple[int, ...] = (0, 50, 100, 200, 400, 1000000)
    use_soft_labels: bool = True


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    if mask.any():
        return values[mask].mean()
    return fallback


def _wasserstein_1d(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    fallback: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = F.softmax(logits.float(), dim=-1)
    targets = targets.to(probs.device).float()
    cdf_pred = probs.cumsum(dim=-1)
    cdf_true = targets.cumsum(dim=-1)
    emd = (cdf_pred - cdf_true).abs().sum(dim=-1)
    idx = torch.arange(probs.size(-1), device=probs.device, dtype=probs.dtype)
    pred_mean = (probs * idx).sum(dim=-1)
    target_mean = (targets * idx).sum(dim=-1)
    mae = (pred_mean - target_mean).abs()
    return _masked_mean(emd, mask, fallback), _masked_mean(mae, mask, fallback)


def compute_proxy_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    losses: Dict[str, torch.Tensor] = {}
    # Optional per-response mask to skip single-response entries.
    response_mask = batch.get("response_mask")
    if torch.is_tensor(response_mask):
        response_mask = response_mask.view(-1).bool()
    else:
        response_mask = None

    domain_logits = outputs.get("domain_logits")
    if domain_logits is None:
        losses["domain_loss"] = outputs["lm_loss"].new_tensor(0.0)
    else:
        half = domain_logits.size(0) // 2
        domain_labels = torch.cat(
            [
                torch.zeros(half, device=domain_logits.device, dtype=torch.long),
                torch.ones(domain_logits.size(0) - half, device=domain_logits.device, dtype=torch.long),
            ],
            dim=0,
        )
        weight = None
        counts = torch.bincount(domain_labels, minlength=2).float()
        if torch.all(counts > 0):
            weight = (counts.sum() / (2.0 * counts)).clamp(max=5.0)
            weight = weight.to(domain_logits.device)
        if response_mask is not None and domain_logits.size(0) == response_mask.numel() * 2:
            domain_mask = response_mask.to(domain_logits.device).repeat(2)
            if domain_mask.any():
                masked_logits = domain_logits[domain_mask]
                masked_labels = domain_labels[domain_mask]
                counts = torch.bincount(masked_labels, minlength=2).float()
                weight = None
                if torch.all(counts > 0):
                    weight = (counts.sum() / (2.0 * counts)).clamp(max=5.0)
                    weight = weight.to(domain_logits.device)
                losses["domain_loss"] = F.cross_entropy(
                    masked_logits.float(), masked_labels, weight=weight
                )
            else:
                losses["domain_loss"] = outputs["lm_loss"].new_tensor(0.0)
        else:
            losses["domain_loss"] = F.cross_entropy(domain_logits.float(), domain_labels, weight=weight)

    position_logits = outputs.get("position_logits")
    position_labels = batch.get("proxy_position_label")
    if position_logits is None or position_labels is None:
        losses["position_loss"] = outputs["lm_loss"].new_tensor(0.0)
    else:
        labels = position_labels.view(-1).to(position_logits.device).long()
        mask = labels.ne(-100)
        if response_mask is not None:
            mask = mask & response_mask.to(position_logits.device)
        if mask.any():
            losses["position_loss"] = F.cross_entropy(position_logits[mask].float(), labels[mask])
        else:
            losses["position_loss"] = outputs["lm_loss"].new_tensor(0.0)

    low_recon_pred = outputs.get("low_recon_pred")
    low_recon_target = outputs.get("low_recon_target")
    if low_recon_pred is None or low_recon_target is None:
        losses["low_recon_loss"] = outputs["lm_loss"].new_tensor(0.0)
    else:
        diff = (low_recon_pred.float() - low_recon_target.float()).pow(2).mean(dim=-1)
        if response_mask is not None and diff.size(0) == response_mask.numel():
            mask = response_mask.to(diff.device)
            losses["low_recon_loss"] = diff[mask].mean() if mask.any() else outputs["lm_loss"].new_tensor(0.0)
        else:
            losses["low_recon_loss"] = diff.mean()

    losses["z_l2_loss"] = outputs.get("z_l2_loss", outputs["lm_loss"].new_tensor(0.0))

    length_logits = outputs.get("length_bin_logits")
    length_labels = batch.get("proxy_length_label")
    length_targets = batch.get("proxy_length_target")
    if length_logits is None or length_labels is None or length_targets is None:
        losses["length_bin_loss"] = outputs["lm_loss"].new_tensor(0.0)
        losses["length_bin_mae"] = outputs["lm_loss"].new_tensor(0.0)
    else:
        labels = length_labels.view(-1).to(length_logits.device).long()
        targets = length_targets.view(-1, length_targets.size(-1)).to(length_logits.device)
        mask = labels.ne(-100)
        loss, mae = _wasserstein_1d(length_logits, targets, mask, outputs["lm_loss"].new_tensor(0.0))
        losses["length_bin_loss"] = loss
        losses["length_bin_mae"] = mae

    nll_logits = outputs.get("nll_bin_logits")
    nll_labels = batch.get("proxy_nll_label")
    nll_targets = batch.get("proxy_nll_target")
    if nll_logits is None or nll_labels is None or nll_targets is None:
        losses["nll_bin_loss"] = outputs["lm_loss"].new_tensor(0.0)
        losses["nll_bin_mae"] = outputs["lm_loss"].new_tensor(0.0)
    else:
        labels = nll_labels.view(-1).to(nll_logits.device).long()
        targets = nll_targets.view(-1, nll_targets.size(-1)).to(nll_logits.device)
        mask = labels.ne(-100)
        loss, mae = _wasserstein_1d(nll_logits, targets, mask, outputs["lm_loss"].new_tensor(0.0))
        losses["nll_bin_loss"] = loss
        losses["nll_bin_mae"] = mae

    ttr_logits = outputs.get("ttr_bin_logits")
    ttr_labels = batch.get("proxy_ttr_label")
    ttr_targets = batch.get("proxy_ttr_target")
    if ttr_logits is None or ttr_labels is None or ttr_targets is None:
        losses["ttr_bin_loss"] = outputs["lm_loss"].new_tensor(0.0)
        losses["ttr_bin_mae"] = outputs["lm_loss"].new_tensor(0.0)
    else:
        labels = ttr_labels.view(-1).to(ttr_logits.device).long()
        targets = ttr_targets.view(-1, ttr_targets.size(-1)).to(ttr_logits.device)
        mask = labels.ne(-100)
        loss, mae = _wasserstein_1d(ttr_logits, targets, mask, outputs["lm_loss"].new_tensor(0.0))
        losses["ttr_bin_loss"] = loss
        losses["ttr_bin_mae"] = mae

    return losses
