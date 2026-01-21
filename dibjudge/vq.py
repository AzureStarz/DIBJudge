from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class VQOutput:
    quantized: torch.Tensor
    indices: torch.Tensor
    loss: torch.Tensor
    commitment_loss: torch.Tensor
    codebook_loss: torch.Tensor
    perplexity: torch.Tensor
    usage_loss: torch.Tensor
    dead_fraction: torch.Tensor
    avg_distance: torch.Tensor


class VectorQuantizerEMA(nn.Module):
    def __init__(
        self,
        num_codes: int,
        dim: int,
        num_codebooks: int = 1,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
        use_ema: bool = True,
        codebook_trainable: bool = False,
        dead_code_threshold: float = 0.1,
        reset_dead_codes: bool = True,
        normalize_inputs: bool = True,
    ) -> None:
        super().__init__()
        self.num_codes = int(num_codes)
        self.dim = int(dim)
        self.num_codebooks = max(1, int(num_codebooks))
        self.commitment_cost = float(commitment_cost)
        self.decay = float(decay)
        self.eps = float(eps)
        self.use_ema = bool(use_ema)
        self.codebook_trainable = bool(codebook_trainable)
        self.dead_code_threshold = float(dead_code_threshold)
        self.reset_dead_codes = bool(reset_dead_codes)
        self.normalize_inputs = bool(normalize_inputs)

        if self.num_codebooks > 1 and self.dim % self.num_codebooks != 0:
            raise ValueError(
                "dim must be divisible by num_codebooks for product quantization."
            )
        self.sub_dim = self.dim // self.num_codebooks
        self.is_pq = self.num_codebooks > 1

        if self.is_pq:
            init = torch.randn(self.num_codebooks, self.num_codes, self.sub_dim) * 0.02
            if self.codebook_trainable:
                self.codebook = nn.Parameter(init)
            else:
                self.register_buffer("codebook", init)
            self.register_buffer(
                "ema_cluster_size", torch.zeros(self.num_codebooks, self.num_codes)
            )
            self.register_buffer("ema_w", init.clone())
        else:
            init = torch.randn(self.num_codes, self.dim) * 0.02
            if self.codebook_trainable:
                self.codebook = nn.Parameter(init)
            else:
                self.register_buffer("codebook", init)
            self.register_buffer("ema_cluster_size", torch.zeros(self.num_codes))
            self.register_buffer("ema_w", init.clone())

    def _distance(self, flat_inputs: torch.Tensor) -> torch.Tensor:
        if self.is_pq:
            raise ValueError("Use _distance_pq for product quantization.")
        codebook = self.codebook
        inputs_norm = flat_inputs.pow(2).sum(dim=1, keepdim=True)
        codebook_norm = codebook.pow(2).sum(dim=1)
        return inputs_norm + codebook_norm - 2.0 * flat_inputs @ codebook.t()

    def _distance_pq(self, flat_inputs: torch.Tensor) -> torch.Tensor:
        codebook = self.codebook
        inputs_norm = flat_inputs.pow(2).sum(dim=2, keepdim=True)
        codebook_norm = codebook.pow(2).sum(dim=2)
        dot = torch.einsum("nmd,mkd->nmk", flat_inputs, codebook)
        return inputs_norm + codebook_norm.unsqueeze(0) - 2.0 * dot

    def _embed_pq(self, indices: torch.Tensor) -> torch.Tensor:
        codebook = self.codebook
        m_ids = torch.arange(self.num_codebooks, device=indices.device)[:, None]
        quantized = codebook[m_ids, indices.transpose(0, 1)]
        return quantized.transpose(0, 1)

    def _usage_loss(self, avg_probs: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        if avg_probs.numel() == 0:
            return avg_probs.new_tensor(0.0)
        if avg_probs.dim() == 1:
            return (avg_probs * (avg_probs + eps).log()).sum() + math.log(self.num_codes)
        per_codebook = (avg_probs * (avg_probs + eps).log()).sum(dim=-1) + math.log(
            self.num_codes
        )
        return per_codebook.mean()

    def _counts_from_indices(
        self, indices: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        if indices.numel() == 0:
            shape = (
                (self.num_codebooks, self.num_codes) if self.is_pq else (self.num_codes,)
            )
            return indices.new_zeros(shape, dtype=dtype)
        if self.is_pq:
            counts = indices.new_zeros(
                (self.num_codebooks, self.num_codes), dtype=dtype
            )
            idx = indices.transpose(0, 1).contiguous()
            ones = torch.ones_like(idx, dtype=dtype)
            counts.scatter_add_(1, idx, ones)
            return counts
        counts = torch.bincount(indices, minlength=self.num_codes)
        return counts.to(dtype)

    def _ema_update(
        self,
        flat_inputs: torch.Tensor,
        indices: torch.Tensor,
        counts: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if not self.use_ema:
            return None
        if indices.numel() == 0:
            return flat_inputs.new_tensor(0.0)
        with torch.no_grad():
            if counts is None:
                counts = self._counts_from_indices(indices, self.ema_cluster_size.dtype)
            else:
                counts = counts.to(dtype=self.ema_cluster_size.dtype)
            if self.is_pq:
                self.ema_cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
                dw = torch.zeros_like(self.ema_w)
                for idx in range(self.num_codebooks):
                    dw[idx].index_add_(0, indices[:, idx], flat_inputs[:, idx, :])
                self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)

                n = self.ema_cluster_size.sum(dim=1, keepdim=True)
                cluster_size = (
                    (self.ema_cluster_size + self.eps)
                    / (n + self.num_codes * self.eps)
                    * n
                )
                codebook = self.ema_w / cluster_size.unsqueeze(-1)

                dead = cluster_size <= float(self.dead_code_threshold)
                dead_fraction = dead.float().mean()
                if self.reset_dead_codes and dead.any():
                    codebook = codebook.clone()
                    for idx in range(self.num_codebooks):
                        dead_idx = dead[idx]
                        if not dead_idx.any():
                            continue
                        replace_pool = flat_inputs[:, idx, :]
                        replace_ids = torch.randint(
                            0,
                            replace_pool.size(0),
                            (int(dead_idx.sum().item()),),
                            device=replace_pool.device,
                        )
                        replace = replace_pool[replace_ids]
                        codebook[idx, dead_idx] = replace
                        self.ema_cluster_size[idx, dead_idx] = 1.0
                        self.ema_w[idx, dead_idx] = replace
                if self.codebook_trainable:
                    self.codebook.data.copy_(codebook)
                else:
                    self.codebook.copy_(codebook)
            else:
                self.ema_cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
                dw = torch.zeros_like(self.ema_w)
                dw.index_add_(0, indices, flat_inputs)
                self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)

                n = self.ema_cluster_size.sum()
                cluster_size = (
                    (self.ema_cluster_size + self.eps)
                    / (n + self.num_codes * self.eps)
                    * n
                )
                codebook = self.ema_w / cluster_size.unsqueeze(1)

                dead = cluster_size <= float(self.dead_code_threshold)
                dead_fraction = dead.float().mean()
                if self.reset_dead_codes and dead.any():
                    replace = flat_inputs[
                        torch.randint(
                            0,
                            flat_inputs.size(0),
                            (int(dead.sum().item()),),
                            device=flat_inputs.device,
                        )
                    ]
                    codebook = codebook.clone()
                    codebook[dead] = replace
                    self.ema_cluster_size[dead] = 1.0
                    self.ema_w[dead] = replace
                if self.codebook_trainable:
                    self.codebook.data.copy_(codebook)
                else:
                    self.codebook.copy_(codebook)
        return dead_fraction

    def alignment_loss(self, token_embeddings: torch.Tensor, sample_size: int) -> torch.Tensor:
        if sample_size <= 0 or token_embeddings.numel() == 0:
            return token_embeddings.new_tensor(0.0)
        flat = token_embeddings.reshape(-1, token_embeddings.size(-1))
        if flat.size(0) == 0:
            return token_embeddings.new_tensor(0.0)
        if flat.size(0) > sample_size:
            idx = torch.randint(0, flat.size(0), (sample_size,), device=flat.device)
            flat = flat[idx]
        if self.is_pq:
            flat = flat.view(-1, self.num_codebooks, self.sub_dim)
            flat = F.normalize(flat, dim=-1, eps=1e-6)
            codebook = F.normalize(self.codebook, dim=-1, eps=1e-6)
            sim = torch.einsum("nmd,mkd->nmk", flat, codebook)
            min_loss = 1.0 - sim.max(dim=2).values
            return min_loss.mean()
        flat = F.normalize(flat, dim=1, eps=1e-6)
        codebook = F.normalize(self.codebook, dim=1, eps=1e-6)
        sim = flat @ codebook.t()
        min_loss = 1.0 - sim.max(dim=1).values
        return min_loss.mean()

    def initialize_codebook(
        self,
        samples: torch.Tensor,
        max_samples: Optional[int] = None,
        seed: int = 0,
        spherical: bool = True,
    ) -> None:
        if samples is None or samples.numel() == 0:
            return
        with torch.no_grad():
            flat = samples.reshape(-1, self.dim)
            if flat.numel() == 0:
                return
            device = self.codebook.device
            dtype = self.codebook.dtype
            flat = flat.to(device=device, dtype=dtype)
            if max_samples is not None and max_samples > 0 and flat.size(0) > max_samples:
                perm = torch.randperm(flat.size(0), device=flat.device)
                flat = flat[perm[: int(max_samples)]]
            if flat.size(0) == 0:
                return
            use_spherical = bool(spherical) or self.normalize_inputs
            generator = torch.Generator(device=flat.device)
            generator.manual_seed(int(seed))

            def _kmeanspp_init(vectors: torch.Tensor) -> torch.Tensor:
                n_samples = vectors.size(0)
                n_codes = int(self.num_codes)
                if n_samples == 1:
                    return vectors.repeat(n_codes, 1)
                first = torch.randint(0, n_samples, (1,), generator=generator, device=vectors.device)
                selected = [int(first.item())]
                center = vectors[first]
                if use_spherical:
                    min_dist = 1.0 - (vectors @ center.t()).squeeze(-1).clamp(-1.0, 1.0)
                    min_dist = min_dist.clamp_min(0.0).pow(2)
                else:
                    min_dist = (vectors - center).pow(2).sum(dim=1)
                while len(selected) < n_codes:
                    total = float(min_dist.sum().item())
                    if total <= 0:
                        break
                    probs = min_dist / min_dist.sum()
                    idx = torch.multinomial(probs, 1, generator=generator)
                    selected.append(int(idx.item()))
                    center = vectors[idx]
                    if use_spherical:
                        dist = 1.0 - (vectors @ center.t()).squeeze(-1).clamp(-1.0, 1.0)
                        dist = dist.clamp_min(0.0).pow(2)
                    else:
                        dist = (vectors - center).pow(2).sum(dim=1)
                    min_dist = torch.minimum(min_dist, dist)
                if len(selected) < n_codes:
                    extra = torch.randint(
                        0, n_samples, (n_codes - len(selected),), generator=generator, device=vectors.device
                    ).tolist()
                    selected.extend(extra)
                chosen = torch.tensor(selected, device=vectors.device)
                return vectors[chosen]

            if self.is_pq:
                flat = flat.view(-1, self.num_codebooks, self.sub_dim)
                if use_spherical:
                    flat = F.normalize(flat, dim=-1, eps=1e-6)
                codebooks: List[torch.Tensor] = []
                for idx in range(self.num_codebooks):
                    codebooks.append(_kmeanspp_init(flat[:, idx, :]))
                codebook = torch.stack(codebooks, dim=0)
            else:
                if use_spherical:
                    flat = F.normalize(flat, dim=1, eps=1e-6)
                codebook = _kmeanspp_init(flat)
            if self.codebook_trainable:
                self.codebook.data.copy_(codebook)
            else:
                self.codebook.copy_(codebook)
            self.ema_w.copy_(codebook)
            self.ema_cluster_size.fill_(1.0)

    def forward(self, inputs: torch.Tensor) -> VQOutput:
        orig_dtype = inputs.dtype
        if inputs.dtype != self.codebook.dtype:
            inputs = inputs.to(dtype=self.codebook.dtype)
        flat_inputs = inputs.reshape(-1, self.dim)
        if self.is_pq:
            pq_inputs = flat_inputs.view(-1, self.num_codebooks, self.sub_dim)
            if self.normalize_inputs:
                pq_inputs = F.normalize(pq_inputs, dim=-1, eps=1e-6)
            flat_inputs = pq_inputs.reshape(-1, self.dim)
            distances = self._distance_pq(pq_inputs)
            indices = distances.argmin(dim=2)
            quantized_flat = self._embed_pq(indices).reshape(-1, self.dim)
        else:
            if self.normalize_inputs:
                flat_inputs = F.normalize(flat_inputs, dim=1, eps=1e-6)
            distances = self._distance(flat_inputs)
            indices = distances.argmin(dim=1)
            quantized_flat = F.embedding(indices, self.codebook)
        counts = self._counts_from_indices(indices, flat_inputs.dtype)

        dead_fraction = None
        if self.training:
            dead_fraction = self._ema_update(
                pq_inputs if self.is_pq else flat_inputs, indices, counts
            )

        codebook_loss = (flat_inputs.detach() - quantized_flat).pow(2).mean()
        commitment_loss = (flat_inputs - quantized_flat.detach()).pow(2).mean()
        loss = codebook_loss + self.commitment_cost * commitment_loss

        inputs_for_st = flat_inputs.view_as(inputs)
        quantized = inputs_for_st + (quantized_flat.view_as(inputs) - inputs_for_st).detach()
        if quantized.dtype != orig_dtype:
            quantized = quantized.to(dtype=orig_dtype)

        if self.is_pq:
            denom = counts.sum(dim=-1, keepdim=True).clamp_min(1.0)
            avg_probs = counts / denom
        else:
            denom = counts.sum().clamp_min(1.0)
            avg_probs = counts / denom
        if avg_probs.dim() == 1:
            perplexity = torch.exp(-torch.sum(avg_probs * (avg_probs + 1e-8).log()))
        else:
            per_codebook = torch.exp(
                -torch.sum(avg_probs * (avg_probs + 1e-8).log(), dim=-1)
            )
            perplexity = per_codebook.mean()
        usage_loss = self._usage_loss(avg_probs)
        usage = self.ema_cluster_size if self.use_ema else counts
        if dead_fraction is None:
            dead_fraction = (usage <= float(self.dead_code_threshold)).float().mean()
        if self.is_pq:
            avg_distance = distances.min(dim=2).values.sum(dim=1).mean().sqrt()
        else:
            avg_distance = distances.min(dim=1).values.mean().sqrt()

        return VQOutput(
            quantized=quantized,
            indices=indices,
            loss=loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            perplexity=perplexity,
            usage_loss=usage_loss,
            dead_fraction=dead_fraction,
            avg_distance=avg_distance,
        )
