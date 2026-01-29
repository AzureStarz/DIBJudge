from __future__ import annotations

from typing import NamedTuple, Optional

import torch
from torch import nn
from torch.nn import functional as F


class _SwiGLULinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim * 2)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.proj(x).chunk(2, dim=-1)
        return self.act(value) * gate


def _build_mlp(
    in_dim: int,
    hidden_dim: int,
    layers: int,
    dropout: float,
    use_swiglu: bool,
) -> Optional[nn.Module]:
    if layers <= 0 or hidden_dim <= 0:
        return None
    blocks = []
    for idx in range(layers):
        in_features = in_dim if idx == 0 else hidden_dim
        if use_swiglu:
            blocks.append(_SwiGLULinear(in_features, hidden_dim))
        else:
            blocks.append(nn.Linear(in_features, hidden_dim))
            blocks.append(nn.GELU())
        if dropout > 0:
            blocks.append(nn.Dropout(dropout))
    return nn.Sequential(*blocks)


class VIBOutput(NamedTuple):
    embeds: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    kl: torch.Tensor


class GaussianVIB(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int = 0,
        layers: int = 2,
        dropout: float = 0.0,
        norm_eps: float = 1e-6,
        use_swiglu: bool = False,
        logvar_min: float = -10.0,
        logvar_max: float = 10.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim) if hidden_dim and hidden_dim > 0 else dim
        layers = max(0, int(layers))
        self.ln = nn.LayerNorm(dim, eps=norm_eps)
        self.mlp = _build_mlp(dim, hidden_dim, layers, dropout, use_swiglu)
        proj_in = hidden_dim if self.mlp is not None else dim
        self.mu = nn.Linear(proj_in, dim)
        self.logvar = nn.Linear(proj_in, dim)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        sample: Optional[bool] = None,
    ) -> VIBOutput:
        x = self.ln(tokens)
        if self.mlp is not None:
            x = self.mlp(x)
        else:
            x = F.gelu(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        if self.logvar_min < self.logvar_max:
            logvar = logvar.clamp(self.logvar_min, self.logvar_max)
        if sample is None:
            sample = self.training
        # Reuse exp(logvar) for both std and KL to avoid a second exp allocation.
        var = logvar.exp()
        if sample:
            std = var.sqrt()
            eps = torch.randn_like(std)
            embeds = mu + eps * std
        else:
            embeds = mu
        kl = 0.5 * (mu.pow(2) + var - 1.0 - logvar)
        kl = kl.sum(dim=-1)
        return VIBOutput(
            embeds=embeds,
            mu=mu,
            logvar=logvar,
            kl=kl,
        )
