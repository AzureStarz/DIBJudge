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
    pred_mean = (probs.detach() * idx).sum(dim=-1)
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

    low_recon_mean_pred = outputs.get("low_recon_mean_pred")
    low_recon_logvar_pred = outputs.get("low_recon_logvar_pred")
    low_recon_mean_target = outputs.get("low_recon_mean_target")
    low_recon_logvar_target = outputs.get("low_recon_logvar_target")
    low_recon_mag_logits = outputs.get("low_recon_mag_logits")
    low_recon_mag_target = outputs.get("low_recon_mag_target")
    if (
        low_recon_mean_pred is None
        or low_recon_logvar_pred is None
        or low_recon_mean_target is None
        or low_recon_logvar_target is None
    ):
        losses["low_recon_mean_loss"] = outputs["lm_loss"].new_tensor(0.0)
        losses["low_recon_logvar_loss"] = outputs["lm_loss"].new_tensor(0.0)
        losses["low_recon_loss"] = outputs["lm_loss"].new_tensor(0.0)
    else:
        mean_diff = (
            (low_recon_mean_pred.float() - low_recon_mean_target.float())
            .pow(2)
            .mean(dim=-1)
        )
        logvar_diff = F.smooth_l1_loss(
            low_recon_logvar_pred.float(),
            low_recon_logvar_target.float(),
            reduction="none",
        ).mean(dim=-1)
        if response_mask is not None and mean_diff.size(0) == response_mask.numel():
            mask = response_mask.to(mean_diff.device)
            if mask.any():
                losses["low_recon_mean_loss"] = mean_diff[mask].mean()
                losses["low_recon_logvar_loss"] = logvar_diff[mask].mean()
            else:
                losses["low_recon_mean_loss"] = outputs["lm_loss"].new_tensor(0.0)
                losses["low_recon_logvar_loss"] = outputs["lm_loss"].new_tensor(0.0)
        else:
            losses["low_recon_mean_loss"] = mean_diff.mean()
            losses["low_recon_logvar_loss"] = logvar_diff.mean()
        losses["low_recon_loss"] = (
            losses["low_recon_mean_loss"] + losses["low_recon_logvar_loss"]
        )
    if low_recon_mag_logits is None or low_recon_mag_target is None:
        losses["low_recon_mag_loss"] = outputs["lm_loss"].new_tensor(0.0)
    else:
        targets = low_recon_mag_target.to(low_recon_mag_logits.device)
        if response_mask is not None and targets.size(0) == response_mask.numel():
            mask = response_mask.to(targets.device).bool()
        else:
            mask = targets.sum(dim=-1).gt(0)
        loss, _ = _wasserstein_1d(
            low_recon_mag_logits, targets, mask, outputs["lm_loss"].new_tensor(0.0)
        )
        losses["low_recon_mag_loss"] = loss

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
