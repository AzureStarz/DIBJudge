from __future__ import annotations

from dataclasses import dataclass
import math
import time
import warnings
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel, AutoModelForCausalLM

from .vq import VectorQuantizerEMA

try:
    import torch.distributed as dist
except ImportError:  # pragma: no cover - torch distributed not available
    dist = None
try:
    import torch.distributed.nn.functional as dist_nn
except ImportError:  # pragma: no cover - torch distributed not available
    dist_nn = None
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class SwiGLULinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim * 2)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.proj(x).chunk(2, dim=-1)
        return self.act(value) * gate


def _build_norm(norm_type: str, dim: int, eps: float) -> nn.Module:
    if norm_type == "rms":
        return RMSNorm(dim, eps=eps)
    return nn.LayerNorm(dim)


def _build_mlp(
    in_dim: int,
    hidden_dim: int,
    layers: int,
    dropout: float,
    use_swiglu: bool,
) -> Optional[nn.Module]:
    if layers <= 0 or hidden_dim <= 0:
        return None
    blocks: List[nn.Module] = []
    for idx in range(layers):
        in_features = in_dim if idx == 0 else hidden_dim
        if use_swiglu:
            blocks.append(SwiGLULinear(in_features, hidden_dim))
        else:
            blocks.append(nn.Linear(in_features, hidden_dim))
            blocks.append(nn.GELU())
        if dropout > 0:
            blocks.append(nn.Dropout(dropout))
    return nn.Sequential(*blocks)


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


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (hidden * mask).sum(dim=1) / denom


def masked_var(hidden: torch.Tensor, mask: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    denom = mask.sum(dim=1).clamp_min(1.0)
    diff = (hidden - mean.unsqueeze(1)) ** 2
    return (diff * mask).sum(dim=1) / denom


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class _GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        ctx.input_dtype = inputs.dtype
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        grad = grad_output.neg() * ctx.scale
        if grad.dtype != ctx.input_dtype:
            grad = grad.to(ctx.input_dtype)
        return grad, None


def gradient_reversal(inputs: torch.Tensor, scale: float) -> torch.Tensor:
    if scale <= 0:
        return inputs
    return _GradientReversalFn.apply(inputs, scale)


class DeterministicProjection(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, clip: float) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, latent_dim)
        self.clip = float(clip)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        if self.clip > 0:
            z = self.clip * torch.tanh(z / self.clip)
        return z


class LatentHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        latent_clip: float,
        hidden_dim: int = 0,
        layers: int = 0,
        dropout: float = 0.0,
        norm_type: str = "layernorm",
        norm_eps: float = 1e-6,
        use_swiglu: bool = False,
    ) -> None:
        super().__init__()
        self.ln = _build_norm(norm_type, in_dim, eps=norm_eps)
        self.act = nn.SiLU() if use_swiglu else nn.GELU()
        proj_in = in_dim
        if layers > 0 and hidden_dim > 0:
            self.mlp = _build_mlp(
                in_dim,
                hidden_dim,
                layers=layers,
                dropout=dropout,
                use_swiglu=use_swiglu,
            )
            proj_in = hidden_dim
        else:
            self.mlp = None
        self.proj = DeterministicProjection(proj_in, latent_dim, latent_clip)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        pooled = self.ln(pooled)
        if self.mlp is not None:
            pooled = self.mlp(pooled)
        else:
            pooled = self.act(pooled)
        return self.proj(pooled)


class _TokenClassifierHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 0,
        layers: int = 0,
        dropout: float = 0.0,
        norm_type: str = "layernorm",
        norm_eps: float = 1e-6,
        use_swiglu: bool = False,
    ) -> None:
        super().__init__()
        self.ln = _build_norm(norm_type, in_dim, eps=norm_eps)
        self.act = nn.SiLU() if use_swiglu else nn.GELU()
        proj_in = in_dim
        if layers > 0 and hidden_dim > 0:
            self.mlp = _build_mlp(
                in_dim,
                hidden_dim,
                layers=layers,
                dropout=dropout,
                use_swiglu=use_swiglu,
            )
            proj_in = hidden_dim
        else:
            self.mlp = None
        self.classifier = nn.Linear(proj_in, 2)

    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        tokens = self.ln(tokens)
        if self.mlp is not None:
            tokens = self.mlp(tokens)
        else:
            tokens = self.act(tokens)
        if tokens.dim() == 2:
            pooled = tokens
        elif mask is not None:
            if tokens.dim() == 4:
                bsz, pairs, seq_len, dim = tokens.shape
                tokens = tokens.view(bsz * pairs, seq_len, dim)
                mask = mask.view(bsz * pairs, seq_len)
            pooled = masked_mean(tokens, mask)
        else:
            pooled = tokens.mean(dim=1)
        return self.classifier(pooled)


class DomainDiscriminatorHead(_TokenClassifierHead):
    pass


class HetsDiscriminatorHead(_TokenClassifierHead):
    pass


class PositionDiscriminatorHead(_TokenClassifierHead):
    pass


class ProxyClassifierHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dim: int = 0,
        layers: int = 0,
        dropout: float = 0.0,
        norm_type: str = "layernorm",
        norm_eps: float = 1e-6,
        use_swiglu: bool = False,
    ) -> None:
        super().__init__()
        self.ln = _build_norm(norm_type, in_dim, eps=norm_eps)
        self.act = nn.SiLU() if use_swiglu else nn.GELU()
        proj_in = in_dim
        if layers > 0 and hidden_dim > 0:
            self.mlp = _build_mlp(
                in_dim,
                hidden_dim,
                layers=layers,
                dropout=dropout,
                use_swiglu=use_swiglu,
            )
            proj_in = hidden_dim
        else:
            self.mlp = None
        self.classifier = nn.Linear(proj_in, num_classes)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        pooled = self.ln(pooled)
        if self.mlp is not None:
            pooled = self.mlp(pooled)
        else:
            pooled = self.act(pooled)
        return self.classifier(pooled)


class TokenReconstructionHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 0,
        layers: int = 0,
        dropout: float = 0.0,
        norm_type: str = "layernorm",
        norm_eps: float = 1e-6,
        use_swiglu: bool = False,
    ) -> None:
        super().__init__()
        self.ln = _build_norm(norm_type, in_dim, eps=norm_eps)
        self.act = nn.SiLU() if use_swiglu else nn.GELU()
        proj_in = in_dim
        if layers > 0 and hidden_dim > 0:
            self.mlp = _build_mlp(
                in_dim,
                hidden_dim,
                layers=layers,
                dropout=dropout,
                use_swiglu=use_swiglu,
            )
            proj_in = hidden_dim
        else:
            self.mlp = None
        self.proj = nn.Linear(proj_in, out_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.ln(tokens)
        if self.mlp is not None:
            tokens = self.mlp(tokens)
        else:
            tokens = self.act(tokens)
        return self.proj(tokens)


class TokenMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 0,
        layers: int = 1,
        dropout: float = 0.1,
        norm_type: str = "layernorm",
        norm_eps: float = 1e-6,
        use_swiglu: bool = False,
    ) -> None:
        super().__init__()
        self.use_identity = (
            int(layers) <= 0 and int(hidden_dim) <= 0 and int(out_dim) == int(in_dim)
        )
        if self.use_identity:
            self.ln = None
            self.act = None
            self.mlp = None
            self.proj = None
            return
        self.ln = _build_norm(norm_type, in_dim, eps=norm_eps)
        self.act = nn.SiLU() if use_swiglu else nn.GELU()
        proj_in = in_dim
        hidden_dim = int(hidden_dim) if hidden_dim and hidden_dim > 0 else in_dim
        layers = max(0, int(layers))
        self.mlp = _build_mlp(
            proj_in,
            hidden_dim,
            layers=layers,
            dropout=dropout,
            use_swiglu=use_swiglu,
        )
        if self.mlp is not None:
            proj_in = hidden_dim
        self.proj = nn.Linear(proj_in, out_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.use_identity:
            return tokens
        tokens = self.ln(tokens)
        if self.mlp is not None:
            tokens = self.mlp(tokens)
        else:
            tokens = self.act(tokens)
        return self.proj(tokens)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float,
        use_swiglu: bool,
    ) -> None:
        super().__init__()
        self.use_swiglu = bool(use_swiglu)
        if self.use_swiglu:
            self.proj = SwiGLULinear(dim, hidden_dim)
            self.out = nn.Linear(hidden_dim, dim)
        else:
            self.proj = nn.Linear(dim, hidden_dim)
            self.act = nn.GELU()
            self.out = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_swiglu:
            x = self.proj(x)
        else:
            x = self.act(self.proj(x))
        x = self.dropout(x)
        return self.out(x)


@dataclass
class DIBJudgeConfig:
    judge_encoder_name: str = "google/mt5-base"
    judge_lm_name: str = "gpt2"
    attn_implementation: Optional[str] = None
    padding_side: str = "right"
    torch_dtype: Optional[str] = None
    use_rms_norm: bool = False
    rms_norm_eps: float = 1e-6
    use_swiglu: bool = False
    z_latent_dim: int = 256
    task_codebook_size: int = 1024
    vq_num_codebooks: int = 4
    vq_commitment_gamma: float = 0.05
    vq_ema_decay: float = 0.99
    vq_use_ema: bool = True
    vq_codebook_trainable: bool = False
    vq_dead_code_threshold: float = 0.1
    vq_reset_dead_codes: bool = True
    vq_align_samples: int = 512
    vq_normalize_inputs: bool = True
    task_mlp_hidden: int = 0
    task_mlp_layers: int = 0
    task_mlp_dropout: float = 0.0
    task_lm_gate_hidden: int = 0
    task_lm_gate_layers: int = 1
    task_lm_gate_dropout: float = 0.0
    bias_mlp_hidden: int = 0
    bias_mlp_layers: int = 1
    bias_mlp_dropout: float = 0.1
    bottleneck_noise_alpha: float = 0.05
    bias_proxy_hidden: int = 0
    bias_proxy_layers: int = 1
    bias_proxy_dropout: float = 0.0
    proxy_nll_classes: int = 6
    proxy_ttr_classes: int = 5
    proxy_length_classes: int = 5
    low_recon_layer: int = 2
    compact_prior: float = 0.3
    compact_mu_token_id: int = 0
    compact_head_hidden: int = 0
    compact_head_layers: int = 1
    compact_head_dropout: float = 0.1
    compact_pi_init: float = 0.95
    lm_loss_chunk_size: int = 0
    compact_kl_chunk_size: int = 0
    disentangle_cos_weight: float = 1.0
    disentangle_cov_weight: float = 0.0
    disentangle_cov_min_batch: int = 4
    disentangle_cov_eps: float = 1e-6
    disentangle_cov_gather: bool = True
    disentangle_cov_full_grad: bool = False
    disentangle_cov_queue_size: int = 0
    low_recon_mag_bins: Tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)


class DIBJudgeModel(nn.Module):
    @staticmethod
    def _load_backbone(
        model_cls,
        name: str,
        attn_implementation: Optional[str],
        torch_dtype: Optional[torch.dtype],
    ) -> torch.nn.Module:
        use_flash = attn_implementation == "flash_attention_2"
        use_cuda = use_flash and torch.cuda.is_available()
        if use_flash and torch_dtype is None and torch.cuda.is_available():
            torch_dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
        kwargs: Dict[str, object] = {}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        if use_flash:
            kwargs["attn_implementation"] = attn_implementation
        if use_cuda:
            kwargs["device_map"] = "cuda"
            kwargs["low_cpu_mem_usage"] = True
        try:
            model = model_cls.from_pretrained(name, **kwargs)
        except (TypeError, ValueError, ImportError) as exc:
            if use_flash:
                warnings.warn(
                    f"attn_implementation={attn_implementation} unavailable for {name}: {exc}. "
                    "Falling back to default attention.",
                    RuntimeWarning,
                )
                kwargs.pop("attn_implementation", None)
            if "device_map" in kwargs:
                kwargs.pop("device_map", None)
                kwargs.pop("low_cpu_mem_usage", None)
            model = model_cls.from_pretrained(name, **kwargs)
        if use_cuda:
            model = model.to("cuda")
        return model

    def __init__(self, config: DIBJudgeConfig) -> None:
        super().__init__()
        self.config = config
        attn_impl = getattr(config, "attn_implementation", None)
        torch_dtype = _resolve_torch_dtype(getattr(config, "torch_dtype", None))
        self.shared_encoder = self._load_backbone(
            AutoModel, config.judge_encoder_name, attn_impl, torch_dtype
        )
        self.judge_lm = self._load_backbone(
            AutoModelForCausalLM, config.judge_lm_name, attn_impl, torch_dtype
        )
        if (
            attn_impl == "flash_attention_2"
            and torch_dtype in (torch.float16, torch.bfloat16)
        ):
            self.shared_encoder = self.shared_encoder.to(dtype=torch_dtype)
            self.judge_lm = self.judge_lm.to(dtype=torch_dtype)
        self._checkpoint_encoder = False
        self._checkpoint_lm = False
        self._checkpoint_use_reentrant = True

        encoder_hidden = getattr(self.shared_encoder.config, "hidden_size", None)
        if encoder_hidden is None:
            encoder_hidden = getattr(self.shared_encoder.config, "d_model", None)
        if encoder_hidden is None:
            raise ValueError("Unable to resolve shared encoder hidden size.")

        bias_mlp_hidden = int(getattr(config, "bias_mlp_hidden", 0))
        if bias_mlp_hidden <= 0:
            bias_mlp_hidden = 2 * encoder_hidden
        lm_hidden = getattr(self.judge_lm.config, "hidden_size", None)
        if lm_hidden is None:
            lm_hidden = getattr(self.judge_lm.config, "n_embd", None)
        if lm_hidden is None:
            raise ValueError("Unable to resolve judge LM hidden size.")

        norm_type = "rms" if bool(config.use_rms_norm) else "layernorm"
        norm_eps = float(config.rms_norm_eps)
        use_swiglu = bool(config.use_swiglu)
        # configured_latent_dim = int(getattr(config, "z_latent_dim", 0))
        task_latent_dim = encoder_hidden
        # if configured_latent_dim > 0 and configured_latent_dim != encoder_hidden:
        #     warnings.warn(
        #         "z_latent_dim ignored; using shared encoder hidden size for Z_task.",
        #         RuntimeWarning,
        #     )
        self.vq_task = VectorQuantizerEMA(
            num_codes=int(getattr(config, "task_codebook_size", 1024)),
            dim=task_latent_dim,
            num_codebooks=int(getattr(config, "vq_num_codebooks", 4)),
            commitment_cost=float(getattr(config, "vq_commitment_gamma", 0.05)),
            decay=float(getattr(config, "vq_ema_decay", 0.99)),
            use_ema=bool(getattr(config, "vq_use_ema", True)),
            codebook_trainable=bool(getattr(config, "vq_codebook_trainable", False)),
            dead_code_threshold=float(getattr(config, "vq_dead_code_threshold", 0.1)),
            reset_dead_codes=bool(getattr(config, "vq_reset_dead_codes", True)),
            normalize_inputs=bool(getattr(config, "vq_normalize_inputs", True)),
        )
        if task_latent_dim == lm_hidden:
            self.task_latent_to_lm = nn.Identity()
        else:
            self.task_latent_to_lm = nn.Linear(task_latent_dim, lm_hidden)
        if task_latent_dim == encoder_hidden:
            self.task_latent_to_encoder = nn.Identity()
        else:
            self.task_latent_to_encoder = nn.Linear(task_latent_dim, encoder_hidden)
        gate_hidden = int(getattr(config, "task_lm_gate_hidden", 0))
        gate_layers = max(0, int(getattr(config, "task_lm_gate_layers", 1)))
        gate_dropout = float(getattr(config, "task_lm_gate_dropout", 0.0))
        self.task_lm_gate = TokenMLP(
            2 * lm_hidden,
            lm_hidden,
            hidden_dim=gate_hidden,
            layers=gate_layers,
            dropout=gate_dropout,
            norm_type=norm_type,
            norm_eps=norm_eps,
            use_swiglu=use_swiglu,
        )
        task_mlp_hidden = int(getattr(config, "task_mlp_hidden", 0))
        task_mlp_layers = max(0, int(getattr(config, "task_mlp_layers", 0)))
        task_mlp_dropout = float(getattr(config, "task_mlp_dropout", 0.0))
        self.task_mlp = TokenMLP(
            encoder_hidden,
            encoder_hidden,
            hidden_dim=task_mlp_hidden,
            layers=task_mlp_layers,
            dropout=task_mlp_dropout,
            norm_type=norm_type,
            norm_eps=norm_eps,
            use_swiglu=use_swiglu,
        )
        self.bias_mlp = TokenMLP(
            encoder_hidden,
            encoder_hidden,
            hidden_dim=bias_mlp_hidden,
            layers=max(0, int(getattr(config, "bias_mlp_layers", 0))),
            dropout=float(getattr(config, "bias_mlp_dropout", 0.0)),
            norm_type=norm_type,
            norm_eps=norm_eps,
            use_swiglu=use_swiglu,
        )
        self.disentangle_queue_size = max(
            0, int(getattr(config, "disentangle_cov_queue_size", 0))
        )
        if self.disentangle_queue_size > 0:
            self.register_buffer(
                "disentangle_task_queue",
                torch.zeros(self.disentangle_queue_size, encoder_hidden),
            )
            self.register_buffer(
                "disentangle_bias_queue",
                torch.zeros(self.disentangle_queue_size, encoder_hidden),
            )
            self.register_buffer(
                "disentangle_queue_ptr", torch.zeros((), dtype=torch.long)
            )
            self.register_buffer(
                "disentangle_queue_filled", torch.zeros((), dtype=torch.long)
            )
        else:
            self.disentangle_task_queue = None
            self.disentangle_bias_queue = None
            self.disentangle_queue_ptr = None
            self.disentangle_queue_filled = None
        bias_hidden = self._resolve_bias_proxy_hidden(encoder_hidden, 2)
        bias_layers = self._resolve_bias_proxy_layers()
        bias_dropout = self._resolve_bias_proxy_dropout()
        recon_hidden = self._resolve_bias_proxy_hidden(
            encoder_hidden, 2 * encoder_hidden
        )
        self.low_recon_mean_head = TokenReconstructionHead(
            encoder_hidden,
            encoder_hidden,
            hidden_dim=recon_hidden,
            layers=bias_layers,
            dropout=bias_dropout,
            norm_type=norm_type,
            norm_eps=norm_eps,
            use_swiglu=use_swiglu,
        )
        mag_bins = getattr(config, "low_recon_mag_bins", None)
        if not mag_bins:
            mag_bins = DIBJudgeConfig.low_recon_mag_bins
        self.low_recon_mag_bins = tuple(float(val) for val in mag_bins)
        mag_classes = max(2, len(self.low_recon_mag_bins) - 1)
        self.low_recon_mag_head = ProxyClassifierHead(
            encoder_hidden,
            num_classes=mag_classes,
            hidden_dim=bias_hidden,
            layers=bias_layers,
            dropout=bias_dropout,
            norm_type=norm_type,
            norm_eps=norm_eps,
            use_swiglu=use_swiglu,
        )
        self.low_recon_logvar_head = TokenReconstructionHead(
            encoder_hidden,
            encoder_hidden,
            hidden_dim=recon_hidden,
            layers=bias_layers,
            dropout=bias_dropout,
            norm_type=norm_type,
            norm_eps=norm_eps,
            use_swiglu=use_swiglu,
        )
        self.length_bin_head = ProxyClassifierHead(
            encoder_hidden,
            num_classes=max(2, int(config.proxy_length_classes)),
            hidden_dim=bias_hidden,
            layers=bias_layers,
            dropout=bias_dropout,
            norm_type=norm_type,
            norm_eps=norm_eps,
            use_swiglu=use_swiglu,
        )
        self.nll_bin_head = ProxyClassifierHead(
            encoder_hidden,
            num_classes=max(2, int(config.proxy_nll_classes)),
            hidden_dim=bias_hidden,
            layers=bias_layers,
            dropout=bias_dropout,
            norm_type=norm_type,
            norm_eps=norm_eps,
            use_swiglu=use_swiglu,
        )
        self.ttr_bin_head = ProxyClassifierHead(
            encoder_hidden,
            num_classes=max(2, int(config.proxy_ttr_classes)),
            hidden_dim=bias_hidden,
            layers=bias_layers,
            dropout=bias_dropout,
            norm_type=norm_type,
            norm_eps=norm_eps,
            use_swiglu=use_swiglu,
        )
        compact_hidden = int(config.compact_head_hidden)
        compact_layers = max(0, int(config.compact_head_layers))
        compact_dropout = float(config.compact_head_dropout)
        self.compact_head = LatentHead(
            encoder_hidden,
            1,
            latent_clip=0.0,
            hidden_dim=compact_hidden,
            layers=compact_layers,
            dropout=compact_dropout,
            norm_type=norm_type,
            norm_eps=norm_eps,
            use_swiglu=use_swiglu,
        )
        self._init_compact_head_bias(float(config.compact_pi_init))
        self.register_buffer(
            "compact_mu_id",
            torch.tensor(int(config.compact_mu_token_id), dtype=torch.long),
            persistent=False,
        )

    def _init_compact_head_bias(self, target: float = 0.95) -> None:
        if target <= 0.0 or target >= 1.0:
            return
        bias = math.log(target / (1.0 - target))
        proj = getattr(self.compact_head, "proj", None)
        if proj is None:
            return
        if hasattr(proj, "proj"):
            proj = proj.proj
        if not isinstance(proj, nn.Linear) or proj.bias is None:
            return
        with torch.no_grad():
            proj.bias.fill_(bias)

    def set_gradient_checkpointing(
        self, encoder: bool, lm: bool, use_reentrant: bool = True
    ) -> None:
        self._checkpoint_encoder = bool(encoder)
        self._checkpoint_lm = bool(lm)
        self._checkpoint_use_reentrant = bool(use_reentrant)

    @staticmethod
    def _get_hidden(outputs) -> torch.Tensor:
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states:
            return hidden_states[-1]
        raise ValueError("Shared encoder output missing hidden states.")

    @classmethod
    def init_from_backbones(
        cls,
        judge_encoder_name: str,
        judge_lm_name: str,
        **config_overrides: object,
    ) -> "DIBJudgeModel":
        config = DIBJudgeConfig(
            judge_encoder_name=judge_encoder_name,
            judge_lm_name=judge_lm_name,
            **config_overrides,
        )
        return cls(config)

    def load_from_pretrained(self, checkpoint_path: str) -> Tuple[List[str], List[str]]:
        state = torch.load(checkpoint_path, map_location="cpu")
        state_dict = state.get("model", state)
        keep_prefixes = (
            "shared_encoder.",
            "judge_lm.",
            "vq_task.",
            "task_latent_to_lm.",
            "task_latent_to_encoder.",
            "task_lm_gate",
            "task_mlp.",
            "bias_mlp.",
            "eng_domain_head.",
            "position_head.",
            "low_recon_head.",
            "low_recon_mean_head.",
            "low_recon_logvar_head.",
            "low_recon_mag_head.",
            "length_bin_head.",
            "nll_bin_head.",
            "ttr_bin_head.",
            "compact_head.",
        )
        filtered = {
            key: value for key, value in state_dict.items() if key.startswith(keep_prefixes)
        }
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        return missing, unexpected

    def _resolve_bias_proxy_hidden(self, in_dim: int, out_dim: int) -> int:
        hidden = int(self.config.bias_proxy_hidden)
        if hidden > 0:
            return hidden
        if out_dim > 2:
            return max(1, int(out_dim // 2))
        return int(in_dim)

    def _resolve_bias_proxy_layers(self) -> int:
        return max(0, int(self.config.bias_proxy_layers))

    def _resolve_bias_proxy_dropout(self) -> float:
        return max(0.0, float(self.config.bias_proxy_dropout))

    def _resolve_encoder_layers(self) -> Optional[torch.nn.ModuleList]:
        cached = getattr(self, "_encoder_layers_cache", None)
        if cached is not None:
            return cached
        enc = self.shared_encoder
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
                self._encoder_layers_cache = mod
                return mod
        self._encoder_layers_cache = None
        return None

    def _get_encoder_layer(self, idx: int) -> Optional[torch.nn.Module]:
        layers = self._resolve_encoder_layers()
        if not layers:
            return None
        idx = max(0, min(int(idx), len(layers) - 1))
        return layers[idx]

    @staticmethod
    def _flatten_pair_inputs(
        inputs: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[int, int]]]:
        if inputs.dim() == 2:
            return inputs, mask, None
        if inputs.dim() != 3:
            raise ValueError("Expected inputs of shape (batch, seq) or (batch, pairs, seq).")
        bsz, pairs, seq_len = inputs.shape
        flat = inputs.view(bsz * pairs, seq_len)
        flat_mask = mask.view(bsz * pairs, seq_len) if mask is not None else None
        return flat, flat_mask, (bsz, pairs)

    def _encode_bias_bundle(
        self,
        orig_ids: torch.Tensor,
        orig_mask: torch.Tensor,
        need_low_hidden: bool = True,
        response_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[Tuple[int, int]],
        Optional[torch.Tensor],
    ]:
        orig_ids, orig_mask, orig_shape = self._flatten_pair_inputs(orig_ids, orig_mask)
        if orig_mask is None:
            orig_mask = torch.ones_like(orig_ids, dtype=torch.long)
        flat_response_mask: Optional[torch.Tensor] = None
        if torch.is_tensor(response_mask):
            if response_mask.dim() == 2 and orig_shape is not None:
                flat_response_mask = response_mask.view(-1)
            elif response_mask.dim() == 1 and response_mask.numel() == orig_ids.size(0):
                flat_response_mask = response_mask
        keep = None
        if flat_response_mask is not None:
            keep = flat_response_mask.to(orig_mask.device).bool()

        empty = orig_mask.sum(dim=1).eq(0)
        empty_kept = empty if keep is None else (empty & keep)
        if empty_kept.any():
            if not getattr(self, "_warned_empty_response", False):
                count = int(empty_kept.sum().item())
                detail = ""
                if orig_shape is not None:
                    bsz, pairs = orig_shape
                    if pairs == 2:
                        empty_pairs = empty_kept.view(bsz, pairs)
                        count_a = int(empty_pairs[:, 0].sum().item())
                        count_b = int(empty_pairs[:, 1].sum().item())
                        detail = f" (A: {count_a}, B: {count_b})"
                elif (
                    torch.is_tensor(response_mask)
                    and response_mask.dim() == 2
                    and response_mask.size(1) == 2
                    and empty_kept.numel() == response_mask.numel()
                ):
                    empty_pairs = empty_kept.view(response_mask.size(0), response_mask.size(1))
                    count_a = int(empty_pairs[:, 0].sum().item())
                    count_b = int(empty_pairs[:, 1].sum().item())
                    detail = f" (A: {count_a}, B: {count_b})"
                warnings.warn(
                    f"Detected {count} empty response sequences{detail}; "
                    "injecting a pad token to keep FlashAttention stable.",
                    RuntimeWarning,
                )
                self._warned_empty_response = True
            pad_side = getattr(self.config, "padding_side", "right")
            pad_side = pad_side if pad_side in {"left", "right"} else "right"
            pad_idx = -1 if pad_side == "left" else 0
            orig_mask[empty_kept, pad_idx] = 1
            pad_id = getattr(self.shared_encoder.config, "pad_token_id", None)
            if pad_id is None:
                pad_id = getattr(self.judge_lm.config, "pad_token_id", None)
            if pad_id is None:
                pad_id = 0
            orig_ids[empty_kept, pad_idx] = int(pad_id)
        orig_len = orig_ids.size(1)
        use_ckpt = (
            self.training and self._checkpoint_encoder and torch.is_grad_enabled()
        )
        enc_embed_layer = self.shared_encoder.get_input_embeddings() if use_ckpt else None
        idx = int(self.config.low_recon_layer)
        hook_layer = self._get_encoder_layer(idx) if need_low_hidden else None

        def _maybe_capture(output: object) -> Optional[torch.Tensor]:
            if torch.is_tensor(output):
                return output
            if isinstance(output, (tuple, list)) and output:
                first = output[0]
                if torch.is_tensor(first):
                    return first
            return None

        def _forward_encoder(
            *,
            input_ids: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            handle = None
            captured: Dict[str, torch.Tensor] = {}
            if hook_layer is not None:
                def _hook(_module, _inputs, output):
                    tensor = _maybe_capture(output)
                    if tensor is not None:
                        captured["tensor"] = tensor
                handle = hook_layer.register_forward_hook(_hook)
            try:
                outputs = self.shared_encoder(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    output_hidden_states=need_low_hidden and hook_layer is None,
                    use_cache=False,
                    return_dict=True,
                )
            finally:
                if handle is not None:
                    handle.remove()
            hidden = self._get_hidden(outputs)
            low_hidden_local = None
            if need_low_hidden:
                if hook_layer is not None:
                    low_hidden_local = captured.get("tensor")
                    if low_hidden_local is None:
                        raise ValueError("Shared encoder hook did not capture hidden states.")
                else:
                    hidden_states = getattr(outputs, "hidden_states", None)
                    if not hidden_states:
                        raise ValueError(
                            "Shared encoder output missing hidden states for reconstruction."
                        )
                    idx_local = max(0, min(idx, len(hidden_states) - 1))
                    low_hidden_local = hidden_states[idx_local]
                if low_hidden_local is not None:
                    low_hidden_local = low_hidden_local[:, :orig_len]
            return hidden, low_hidden_local

        def _encode(
            inputs_ids: torch.Tensor, inputs_mask: torch.Tensor
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            if use_ckpt and enc_embed_layer is not None:
                encoder_embeds = enc_embed_layer(inputs_ids)
                if not encoder_embeds.requires_grad:
                    encoder_embeds = encoder_embeds.detach().requires_grad_(True)

                def _run_encoder(inputs_embeds: torch.Tensor, attention_mask: torch.Tensor):
                    return _forward_encoder(
                        inputs_embeds=inputs_embeds, attention_mask=attention_mask
                    )

                return checkpoint(
                    _run_encoder,
                    encoder_embeds,
                    inputs_mask,
                    use_reentrant=self._checkpoint_use_reentrant,
                )
            return _forward_encoder(input_ids=inputs_ids, attention_mask=inputs_mask)

        if keep is not None and keep.numel() == orig_ids.size(0) and not keep.all():
            if keep.any():
                kept_hidden, kept_low = _encode(orig_ids[keep], orig_mask[keep])
                full_hidden = kept_hidden.new_zeros(
                    (orig_ids.size(0), orig_ids.size(1), kept_hidden.size(-1))
                )
                full_hidden[keep] = kept_hidden
                full_low = None
                if need_low_hidden:
                    full_low = kept_hidden.new_zeros(
                        (orig_ids.size(0), orig_ids.size(1), kept_hidden.size(-1))
                    )
                    if kept_low is not None:
                        full_low[keep] = kept_low
                orig_hidden = full_hidden
                low_hidden = full_low
            else:
                hidden_size = getattr(self.shared_encoder.config, "hidden_size", None)
                if hidden_size is None:
                    hidden_size = getattr(self.shared_encoder.config, "d_model", None)
                if hidden_size is None:
                    raise ValueError("Unable to resolve shared encoder hidden size.")
                embed_layer = self.shared_encoder.get_input_embeddings()
                dtype = embed_layer.weight.dtype if embed_layer is not None else torch.float32
                orig_hidden = torch.zeros(
                    (orig_ids.size(0), orig_ids.size(1), hidden_size),
                    device=orig_ids.device,
                    dtype=dtype,
                )
                low_hidden = (
                    orig_hidden.clone() if need_low_hidden else None
                )
        else:
            orig_hidden, low_hidden = _encode(orig_ids, orig_mask)

        return (
            orig_hidden,
            orig_mask,
            orig_shape,
            low_hidden,
        )

    def _pool_hidden(self, hidden: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return hidden[:, -1]
        return last_token_pool(hidden, mask)

    def _pool_hidden_mean(
        self, hidden: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if mask is None:
            return hidden.mean(dim=1)
        return masked_mean(hidden, mask)

    @staticmethod
    def _scatter_compact_logits(
        response_types: torch.Tensor,
        pi_a: torch.Tensor,
        pi_b: Optional[torch.Tensor],
    ) -> torch.Tensor:
        pi_logits = pi_a.new_zeros(response_types.size())

        def _fill(label: int, pi_tokens: Optional[torch.Tensor]) -> None:
            if pi_tokens is None or pi_tokens.size(1) == 0:
                return
            mask = response_types.eq(label)
            if not mask.any():
                return
            idx = mask.long().cumsum(dim=1) - 1
            valid = mask & (idx < pi_tokens.size(1))
            if not valid.any():
                return
            idx = idx.clamp(min=0, max=pi_tokens.size(1) - 1)
            gathered = torch.gather(pi_tokens, 1, idx)
            pi_logits[valid] = gathered[valid]

        _fill(1, pi_a)
        _fill(2, pi_b)
        return pi_logits

    @staticmethod
    def _scatter_response_latents(
        response_types: torch.Tensor,
        latents: torch.Tensor,
        mask: Optional[torch.Tensor],
        pairs: int,
    ) -> torch.Tensor:
        bsz = response_types.size(0)
        if latents.size(0) != bsz * max(1, int(pairs)):
            raise ValueError("Latents batch size does not match response types.")
        seq_len = latents.size(1)
        dim = latents.size(2)
        pair_count = max(1, int(pairs))
        if seq_len == 0 or dim == 0:
            return latents.new_zeros(bsz, response_types.size(1), dim)
        latents_pair = latents.view(bsz, pair_count, seq_len, dim)
        mask_pair = (
            mask.view(bsz, pair_count, seq_len) if torch.is_tensor(mask) else None
        )
        if mask_pair is None:
            packed = latents_pair
            valid_len = latents.new_full((bsz, pair_count), seq_len, dtype=torch.long)
        else:
            mask_bool = mask_pair.bool()
            valid_len = mask_bool.sum(dim=-1)
            if mask_bool.all():
                packed = latents_pair
            else:
                positions = torch.arange(seq_len, device=latents.device).view(1, 1, seq_len)
                key = positions + (~mask_bool) * seq_len
                gather_idx = torch.argsort(key, dim=-1)
                packed = latents_pair.gather(
                    2, gather_idx.unsqueeze(-1).expand(-1, -1, -1, dim)
                )
        out = latents.new_zeros(bsz, response_types.size(1), dim)
        for pair_idx, label in enumerate((1, 2)):
            if pair_idx >= pair_count:
                break
            resp_mask = response_types.eq(label)
            if not resp_mask.any():
                continue
            resp_rank = resp_mask.long().cumsum(dim=1) - 1
            valid = resp_mask & (resp_rank < valid_len[:, pair_idx].unsqueeze(1))
            if not valid.any():
                continue
            max_idx = packed.size(2) - 1
            gather_idx = resp_rank.clamp(min=0, max=max_idx)
            selected = packed[:, pair_idx].gather(
                1, gather_idx.unsqueeze(-1).expand(-1, -1, dim)
            )
            out[valid] = selected[valid]
        return out

    @staticmethod
    def _normalize_token_features(hidden: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        hidden_f = hidden.float()
        normed = F.layer_norm(hidden_f, (hidden_f.size(-1),), eps=eps)
        return normed.to(hidden.dtype)

    def _compute_low_stats(
        self, hidden: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        normed = self._normalize_token_features(hidden)
        if mask is None:
            mean = normed.mean(dim=1)
            var = normed.var(dim=1, unbiased=False)
        else:
            mean = masked_mean(normed, mask)
            var = masked_var(normed, mask, mean)
        logvar = (var.clamp_min(1e-6)).log()
        return mean, logvar

    def _soft_histogram(
        self,
        values: torch.Tensor,
        bins: Tuple[float, ...],
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if values.numel() == 0:
            return values.new_zeros((values.size(0), max(1, len(bins) - 1)))
        values_f = values.float()
        edges = values_f.new_tensor(bins)
        num_bins = max(1, edges.numel() - 1)
        values_f = values_f.clamp(min=float(edges[0].item()), max=float(edges[-1].item()))
        idx = torch.bucketize(values_f, edges) - 1
        idx = idx.clamp(min=0, max=num_bins - 1)
        left = edges[idx]
        right = edges[(idx + 1).clamp(max=edges.numel() - 1)]
        denom = (right - left).clamp_min(1e-6)
        w_right = ((values_f - left) / denom).clamp(0.0, 1.0)
        is_last = idx.eq(num_bins - 1)
        w_right = w_right.masked_fill(is_last, 0.0)
        w_left = 1.0 - w_right
        if mask is None:
            mask = torch.ones_like(values_f, dtype=torch.bool)
        mask_f = mask.to(values_f.dtype)
        w_left = w_left * mask_f
        w_right = w_right * mask_f
        bsz, seq_len = values_f.shape
        batch_ids = torch.arange(bsz, device=values_f.device).unsqueeze(1).expand(bsz, seq_len)
        flat_batch = batch_ids.reshape(-1)
        flat_left = idx.reshape(-1)
        flat_right = (idx + 1).clamp(max=num_bins - 1).reshape(-1)
        flat_w_left = w_left.reshape(-1)
        flat_w_right = w_right.reshape(-1)
        hist = values_f.new_zeros((bsz * num_bins,))
        hist.scatter_add_(0, flat_batch * num_bins + flat_left, flat_w_left)
        hist.scatter_add_(0, flat_batch * num_bins + flat_right, flat_w_right)
        hist = hist.view(bsz, num_bins)
        denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        return hist / denom

    def _compute_low_mag_hist(
        self, hidden: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        normed = self._normalize_token_features(hidden)
        token_mag = normed.abs().mean(dim=-1)
        return self._soft_histogram(token_mag, self.low_recon_mag_bins, mask)

    @staticmethod
    def _compute_causal_lm_loss(
        logits: torch.Tensor,
        labels: Optional[torch.Tensor],
        chunk_size: int = 0,
    ) -> Optional[torch.Tensor]:
        if labels is None:
            return None
        if logits.numel() == 0:
            return logits.new_zeros(())
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        if shift_labels.numel() == 0:
            return logits.new_zeros(())
        chunk_size = int(chunk_size) if chunk_size is not None else 0
        if chunk_size <= 0:
            return F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
                reduction="mean",
            )
        total_loss = shift_logits.new_zeros(())
        total_count = shift_labels.new_zeros((), dtype=torch.long)
        for start in range(0, shift_logits.size(1), chunk_size):
            end = min(start + chunk_size, shift_logits.size(1))
            chunk_logits = shift_logits[:, start:end, :]
            chunk_labels = shift_labels[:, start:end]
            if chunk_labels.numel() == 0:
                continue
            chunk_loss = F.cross_entropy(
                chunk_logits.reshape(-1, chunk_logits.size(-1)),
                chunk_labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            total_loss = total_loss + chunk_loss
            total_count = total_count + chunk_labels.ne(-100).sum()
        return total_loss / total_count.clamp_min(1).to(total_loss.dtype)

    @staticmethod
    def _compute_compact_kl_loss(
        full_logits: torch.Tensor,
        masked_logits: torch.Tensor,
        label_mask: torch.Tensor,
        chunk_size: int = 0,
    ) -> torch.Tensor:
        if label_mask is None or not label_mask.any():
            return masked_logits.new_zeros(())
        chunk_size = int(chunk_size) if chunk_size is not None else 0
        if chunk_size > 0:
            total_kl = full_logits.new_zeros(())
            total_count = label_mask.new_zeros((), dtype=torch.long)
            seq_len = full_logits.size(1)
            for start in range(0, seq_len, chunk_size):
                end = min(start + chunk_size, seq_len)
                chunk_mask = label_mask[:, start:end]
                if not chunk_mask.any():
                    continue
                chunk_full = full_logits[:, start:end, :][chunk_mask]
                chunk_masked = masked_logits[:, start:end, :][chunk_mask]
                full_logp = F.log_softmax(chunk_full.float(), dim=-1)
                masked_logp = F.log_softmax(chunk_masked.float(), dim=-1)
                kl = (full_logp.exp() * (full_logp - masked_logp)).sum(dim=-1)
                total_kl = total_kl + kl.sum()
                total_count = total_count + chunk_mask.sum()
            return total_kl / total_count.clamp_min(1).to(total_kl.dtype)
        full_logits = full_logits[label_mask]
        masked_logits = masked_logits[label_mask]
        full_logp = F.log_softmax(full_logits.float(), dim=-1)
        masked_logp = F.log_softmax(masked_logits.float(), dim=-1)
        kl = (full_logp.exp() * (full_logp - masked_logp)).sum(dim=-1)
        return kl.mean()

    def _apply_z_dropout(self, z: torch.Tensor, dropout: float) -> torch.Tensor:
        if not self.training or dropout <= 0:
            return z
        keep = torch.rand(z.size(0), z.size(1), device=z.device) >= float(dropout)
        keep = keep.to(z.dtype).unsqueeze(-1)
        return z * keep

    def _codebook_alignment_loss(self, quantizer: VectorQuantizerEMA) -> torch.Tensor:
        sample_size = int(self.config.vq_align_samples)
        if sample_size <= 0:
            return torch.zeros((), device=self.judge_lm.device)
        token_embeddings = self.shared_encoder.get_input_embeddings().weight
        return quantizer.alignment_loss(token_embeddings, sample_size)

    def _gather_disentangle_batch(
        self,
        task: torch.Tensor,
        bias: torch.Tensor,
        full_grad: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if dist is None or not dist.is_available() or not dist.is_initialized():
            return task, bias
        world_size = dist.get_world_size()
        if world_size <= 1:
            return task, bias
        device = task.device
        local_size = torch.tensor([task.size(0)], device=device, dtype=torch.long)
        size_list = [local_size.clone() for _ in range(world_size)]
        dist.all_gather(size_list, local_size)
        max_size = max(int(size.item()) for size in size_list)
        if max_size <= 0:
            return task, bias
        task_pad = task
        bias_pad = bias
        if task.size(0) < max_size:
            pad = torch.zeros(
                (max_size - task.size(0), task.size(1)),
                device=device,
                dtype=task.dtype,
            )
            task_pad = torch.cat([task, pad], dim=0)
        if bias.size(0) < max_size:
            pad = torch.zeros(
                (max_size - bias.size(0), bias.size(1)),
                device=device,
                dtype=bias.dtype,
            )
            bias_pad = torch.cat([bias, pad], dim=0)
        use_full_grad = bool(full_grad) and dist_nn is not None
        if use_full_grad:
            gather_task = list(dist_nn.all_gather(task_pad))
            gather_bias = list(dist_nn.all_gather(bias_pad))
        else:
            gather_task = [torch.zeros_like(task_pad) for _ in range(world_size)]
            gather_bias = [torch.zeros_like(bias_pad) for _ in range(world_size)]
            dist.all_gather(gather_task, task_pad.detach())
            dist.all_gather(gather_bias, bias_pad.detach())
            rank = dist.get_rank()
            gather_task[rank] = task_pad
            gather_bias[rank] = bias_pad
        task_chunks = []
        bias_chunks = []
        for task_chunk, bias_chunk, size in zip(gather_task, gather_bias, size_list):
            size_int = int(size.item())
            if size_int <= 0:
                continue
            task_chunks.append(task_chunk[:size_int])
            bias_chunks.append(bias_chunk[:size_int])
        if not task_chunks:
            return task, bias
        return torch.cat(task_chunks, dim=0), torch.cat(bias_chunks, dim=0)

    def _get_disentangle_queue(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        size = int(getattr(self, "disentangle_queue_size", 0))
        if size <= 0:
            return None
        if (
            self.disentangle_task_queue is None
            or self.disentangle_bias_queue is None
            or self.disentangle_queue_filled is None
        ):
            return None
        filled = int(self.disentangle_queue_filled.item())
        if filled <= 0:
            return None
        if filled < size:
            return (
                self.disentangle_task_queue[:filled],
                self.disentangle_bias_queue[:filled],
            )
        return self.disentangle_task_queue, self.disentangle_bias_queue

    def _update_disentangle_queue(
        self, task: torch.Tensor, bias: torch.Tensor
    ) -> None:
        if not self.training:
            return
        size = int(getattr(self, "disentangle_queue_size", 0))
        if size <= 0 or task.numel() == 0 or bias.numel() == 0:
            return
        if (
            self.disentangle_task_queue is None
            or self.disentangle_bias_queue is None
            or self.disentangle_queue_ptr is None
            or self.disentangle_queue_filled is None
        ):
            return
        with torch.no_grad():
            queue_task = self.disentangle_task_queue
            queue_bias = self.disentangle_bias_queue
            task_det = task.detach().to(device=queue_task.device, dtype=queue_task.dtype)
            bias_det = bias.detach().to(device=queue_bias.device, dtype=queue_bias.dtype)
            bsz = int(task_det.size(0))
            if bsz >= size:
                queue_task.copy_(task_det[-size:])
                queue_bias.copy_(bias_det[-size:])
                self.disentangle_queue_ptr.zero_()
                self.disentangle_queue_filled.fill_(size)
                return
            ptr = int(self.disentangle_queue_ptr.item())
            end = ptr + bsz
            if end <= size:
                queue_task[ptr:end].copy_(task_det)
                queue_bias[ptr:end].copy_(bias_det)
            else:
                first = size - ptr
                queue_task[ptr:].copy_(task_det[:first])
                queue_bias[ptr:].copy_(bias_det[:first])
                remaining = bsz - first
                queue_task[:remaining].copy_(task_det[first:])
                queue_bias[:remaining].copy_(bias_det[first:])
            self.disentangle_queue_ptr.fill_(end % size)
            filled = int(self.disentangle_queue_filled.item())
            self.disentangle_queue_filled.fill_(min(size, filled + bsz))

    def _disentangle_loss(
        self,
        task: torch.Tensor,
        bias: torch.Tensor,
        cos_weight: Optional[float] = None,
        cov_weight: Optional[float] = None,
        cov_gather: Optional[bool] = None,
        cov_full_grad: Optional[bool] = None,
    ) -> torch.Tensor:
        if task.numel() == 0 or bias.numel() == 0:
            return task.new_zeros(())
        if cos_weight is None:
            cos_weight = float(getattr(self.config, "disentangle_cos_weight", 1.0))
        if cov_weight is None:
            cov_weight = float(getattr(self.config, "disentangle_cov_weight", 0.0))
        cov_min_batch = int(getattr(self.config, "disentangle_cov_min_batch", 4))
        cov_eps = float(getattr(self.config, "disentangle_cov_eps", 1e-6))
        if cov_gather is None:
            cov_gather = bool(getattr(self.config, "disentangle_cov_gather", True))
        if cov_full_grad is None:
            cov_full_grad = bool(
                getattr(self.config, "disentangle_cov_full_grad", False)
            )

        task_norm = F.normalize(task.float(), dim=-1)
        bias_norm = F.normalize(bias.float(), dim=-1)
        corr = (task_norm * bias_norm).sum(dim=-1).abs()
        loss = corr.mean() * cos_weight

        if cov_weight <= 0:
            self._update_disentangle_queue(task, bias)
            return loss.to(task.dtype)

        cov_task = task
        cov_bias = bias
        if cov_gather:
            cov_task, cov_bias = self._gather_disentangle_batch(
                task, bias, full_grad=bool(cov_full_grad)
            )
        if self.training:
            queue = self._get_disentangle_queue()
            if queue is not None:
                queue_task, queue_bias = queue
                cov_task = torch.cat([cov_task, queue_task], dim=0)
                cov_bias = torch.cat([cov_bias, queue_bias], dim=0)
        if cov_task.size(0) < cov_min_batch:
            self._update_disentangle_queue(task, bias)
            return loss.to(task.dtype)

        z_task = cov_task.float()
        z_bias = cov_bias.float()
        z_task = z_task - z_task.mean(dim=0, keepdim=True)
        z_bias = z_bias - z_bias.mean(dim=0, keepdim=True)
        z_task = z_task / z_task.std(dim=0, unbiased=False).clamp_min(cov_eps)
        z_bias = z_bias / z_bias.std(dim=0, unbiased=False).clamp_min(cov_eps)
        cov = (z_task.t() @ z_bias) / z_task.size(0)
        cov_loss = cov.pow(2).mean()
        self._update_disentangle_queue(task, bias)
        return (loss + cov_weight * cov_loss).to(task.dtype)

    def benchmark_disentangle_overhead(
        self,
        task: torch.Tensor,
        bias: torch.Tensor,
        *,
        iters: int = 50,
        warmup: int = 10,
        full_grad: bool = False,
    ) -> Dict[str, float]:
        if task.numel() == 0 or bias.numel() == 0:
            return {"mean_ms": 0.0, "iters": 0, "full_grad": float(full_grad)}
        iters = max(1, int(iters))
        warmup = max(0, int(warmup))
        task_bench = task.detach().requires_grad_(True)
        bias_bench = bias.detach().requires_grad_(True)
        device = task_bench.device
        use_cuda = device.type == "cuda"
        if dist is not None and dist.is_available() and dist.is_initialized():
            dist.barrier()
        if use_cuda:
            torch.cuda.synchronize(device)

        for _ in range(warmup):
            loss = self._disentangle_loss(
                task_bench,
                bias_bench,
                cos_weight=0.0,
                cov_weight=1.0,
                cov_gather=True,
                cov_full_grad=full_grad,
            )
            loss.backward()
            task_bench.grad = None
            bias_bench.grad = None
        if dist is not None and dist.is_available() and dist.is_initialized():
            dist.barrier()
        if use_cuda:
            torch.cuda.synchronize(device)

        start = time.perf_counter()
        for _ in range(iters):
            loss = self._disentangle_loss(
                task_bench,
                bias_bench,
                cos_weight=0.0,
                cov_weight=1.0,
                cov_gather=True,
                cov_full_grad=full_grad,
            )
            loss.backward()
            task_bench.grad = None
            bias_bench.grad = None
        if use_cuda:
            torch.cuda.synchronize(device)
        if dist is not None and dist.is_available() and dist.is_initialized():
            dist.barrier()
        elapsed = time.perf_counter() - start
        return {
            "mean_ms": elapsed * 1000.0 / float(iters),
            "iters": float(iters),
            "full_grad": float(full_grad),
        }

    def lm_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        pi_logits: Optional[torch.Tensor] = None,
        response_types: Optional[torch.Tensor] = None,
        z_task_addition: Optional[torch.Tensor] = None,
        compact_prior: Optional[torch.Tensor] = None,
        disable_compactor: bool = False,
        compute_compact_kl: bool = True,
        lm_loss_chunk_size: Optional[int] = None,
        compact_kl_chunk_size: Optional[int] = None,
        return_logits: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if lm_loss_chunk_size is None:
            lm_loss_chunk_size = int(getattr(self.config, "lm_loss_chunk_size", 0) or 0)
        elif torch.is_tensor(lm_loss_chunk_size):
            lm_loss_chunk_size = int(lm_loss_chunk_size.detach().item())
        else:
            lm_loss_chunk_size = int(lm_loss_chunk_size)
        if compact_kl_chunk_size is None:
            compact_kl_chunk_size = int(
                getattr(self.config, "compact_kl_chunk_size", 0) or 0
            )
        elif torch.is_tensor(compact_kl_chunk_size):
            compact_kl_chunk_size = int(compact_kl_chunk_size.detach().item())
        else:
            compact_kl_chunk_size = int(compact_kl_chunk_size)
        embed_layer = self.judge_lm.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids)
        use_ckpt = self.training and self._checkpoint_lm and torch.is_grad_enabled()
        need_logits = bool(return_logits) or labels is None
        if use_ckpt and not inputs_embeds.requires_grad:
            inputs_embeds = inputs_embeds.detach().requires_grad_(True)
        compact_mask_loss = inputs_embeds.new_zeros(())
        compact_con_loss = inputs_embeds.new_zeros(())
        compact_kl_loss = inputs_embeds.new_zeros(())
        compact_pi_mean = inputs_embeds.new_zeros(())
        compact_mask_mean = inputs_embeds.new_zeros(())
        compact_pi_saturation = inputs_embeds.new_zeros(())
        gated_mean = inputs_embeds.new_zeros(())
        gated_add_norm = inputs_embeds.new_zeros(())
        if pi_logits is None:
            pi_logits = inputs_embeds.new_zeros(inputs_embeds.size(0), inputs_embeds.size(1))
        if z_task_addition is not None:
            if response_types is None:
                warnings.warn(
                    "z_task_addition provided without response types; skipping.",
                    RuntimeWarning,
                )
            else:
                if z_task_addition.shape != inputs_embeds.shape:
                    raise ValueError("z_task_addition shape must match LM inputs.")
                response_mask = response_types > 0
                response_mask = response_mask & attention_mask.bool()
                if response_mask.any():
                    gate_inputs = torch.cat([inputs_embeds, z_task_addition], dim=-1)
                    gate = torch.sigmoid(self.task_lm_gate(gate_inputs)).to(
                        dtype=inputs_embeds.dtype, device=inputs_embeds.device
                    )
                    gated = z_task_addition.to(
                        dtype=inputs_embeds.dtype, device=inputs_embeds.device
                    )
                    gated = gated * gate
                    masked_gated = gated * response_mask.unsqueeze(-1).to(gated.dtype)
                    inputs_embeds = inputs_embeds + masked_gated
                    gated_mean = gate[response_mask].mean()
                    gated_add_norm = masked_gated[response_mask].norm(dim=-1).mean()
        if disable_compactor:
            if use_ckpt and labels is not None and not need_logits:
                def _run_lm_loss(lm_embeds: torch.Tensor, lm_attention: torch.Tensor) -> torch.Tensor:
                    out = self.judge_lm(
                        inputs_embeds=lm_embeds,
                        attention_mask=lm_attention,
                        labels=None,
                        use_cache=False,
                    )
                    return self._compute_causal_lm_loss(
                        out.logits, labels, chunk_size=lm_loss_chunk_size
                    )

                lm_loss = checkpoint(
                    _run_lm_loss,
                    inputs_embeds,
                    attention_mask,
                    use_reentrant=self._checkpoint_use_reentrant,
                )
                lm_logits = None
            else:
                outputs = self.judge_lm(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=None,
                    use_cache=False,
                )
                lm_logits = outputs.logits
                lm_loss = self._compute_causal_lm_loss(
                    lm_logits, labels, chunk_size=lm_loss_chunk_size
                )
            result = {
                "loss": lm_loss,
                "compact_mask_loss": inputs_embeds.new_zeros(()),
                "compact_con_loss": inputs_embeds.new_zeros(()),
                "compact_kl_loss": inputs_embeds.new_zeros(()),
                "compact_pi_mean": inputs_embeds.new_zeros(()),
                "compact_mask_mean": inputs_embeds.new_zeros(()),
                "compact_pi_saturation": inputs_embeds.new_zeros(()),
                "gated_mean": gated_mean,
                "gated_add_norm": gated_add_norm,
            }
            if need_logits and lm_logits is not None:
                result["logits"] = lm_logits
            return result
        if labels is not None:
            labels = labels.masked_fill(attention_mask.eq(0), -100)
        pi = torch.sigmoid(pi_logits.float())
        pi = torch.nan_to_num(pi, nan=0.5, posinf=1.0, neginf=0.0)
        pi = pi.clamp(min=0.0, max=1.0).to(pi_logits.dtype)
        prompt_mask = attention_mask.bool()
        if labels is not None:
            prompt_mask = prompt_mask & labels.eq(-100)
        response_mask = None
        if response_types is not None:
            response_mask = response_types > 0
        if response_mask is not None:
            response_mask = response_mask & prompt_mask
            if (~response_mask.any(dim=1)).any():
                warnings.warn(
                    "lm_response_types has no response spans for at least one sample; "
                    "compact masking is skipped for those samples.",
                    RuntimeWarning,
                )
            response_mask_f = response_mask.to(pi.dtype)
            pi = pi * response_mask_f + (1.0 - response_mask_f)
        prompt_mask_f = prompt_mask.to(pi.dtype)
        pi = pi * prompt_mask_f + (1.0 - prompt_mask_f)
        m = torch.bernoulli(pi)
        m = (m - pi).detach() + pi
        prompt_mask_m = prompt_mask.to(m.dtype)
        m = m * prompt_mask_m + (1.0 - prompt_mask_m)
        mask_for_compact = prompt_mask
        if response_mask is not None:
            mask_for_compact = mask_for_compact & response_mask
        mask_for_compact_f = mask_for_compact.to(pi.dtype)
        mask_count = mask_for_compact.sum().clamp_min(1)
        compact_pi_mean = (pi * mask_for_compact_f).sum() / mask_count
        compact_mask_mean = (m * mask_for_compact_f.to(m.dtype)).sum() / mask_count
        r = float(self.config.compact_prior)
        if compact_prior is not None:
            if torch.is_tensor(compact_prior):
                r = float(compact_prior.detach().item())
            else:
                r = float(compact_prior)
        eps = 1e-6
        if mask_for_compact.any():
            pi_prompt = pi[mask_for_compact].float().clamp(min=eps, max=1.0 - eps)
            r = min(max(r, eps), 1.0 - eps)
            compact_mask_loss = (
                pi_prompt * (torch.log(pi_prompt) - math.log(r))
                + (1.0 - pi_prompt)
                * (torch.log(1.0 - pi_prompt) - math.log(1.0 - r))
            ).mean()
            compact_mask_loss = compact_mask_loss.to(pi.dtype)
            compact_pi_saturation = (
                (pi_prompt < 0.05) | (pi_prompt > 0.95)
            ).float().mean()
            pair_mask = mask_for_compact[:, 1:] & mask_for_compact[:, :-1]
            if pair_mask.any():
                diffs = (pi[:, 1:] - pi[:, :-1]) ** 2
                compact_con_loss = diffs[pair_mask].mean()
        mu_embed = embed_layer(self.compact_mu_id.to(inputs_embeds.device))
        mu_embed = mu_embed.view(1, 1, -1)
        masked_embeds = m.unsqueeze(-1) * inputs_embeds + (1.0 - m).unsqueeze(-1) * mu_embed
        label_mask = labels.ne(-100) if labels is not None else None
        compute_kl = bool(compute_compact_kl and label_mask is not None and label_mask.any())
        if use_ckpt and labels is not None and not need_logits:
            def _run_lm_loss(
                lm_embeds: torch.Tensor, lm_attention: torch.Tensor
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                out = self.judge_lm(
                    inputs_embeds=lm_embeds,
                    attention_mask=lm_attention,
                    labels=None,
                    use_cache=False,
                )
                lm_logits_local = out.logits
                lm_loss_local = self._compute_causal_lm_loss(
                    lm_logits_local, labels, chunk_size=lm_loss_chunk_size
                )
                if compute_kl:
                    with torch.no_grad():
                        full_out = self.judge_lm(
                            inputs_embeds=inputs_embeds,
                            attention_mask=lm_attention,
                            labels=None,
                            use_cache=False,
                        )
                    compact_kl_local = self._compute_compact_kl_loss(
                        full_out.logits,
                        lm_logits_local,
                        label_mask,
                        chunk_size=compact_kl_chunk_size,
                    )
                else:
                    compact_kl_local = lm_logits_local.new_zeros(())
                return lm_loss_local, compact_kl_local

            lm_loss, compact_kl_loss = checkpoint(
                _run_lm_loss,
                masked_embeds,
                attention_mask,
                use_reentrant=self._checkpoint_use_reentrant,
            )
            lm_logits = None
        elif use_ckpt and labels is None:
            def _run_lm_logits(lm_embeds: torch.Tensor, lm_attention: torch.Tensor):
                out = self.judge_lm(
                    inputs_embeds=lm_embeds,
                    attention_mask=lm_attention,
                    labels=None,
                    use_cache=False,
                )
                return out.logits

            lm_logits = checkpoint(
                _run_lm_logits,
                masked_embeds,
                attention_mask,
                use_reentrant=self._checkpoint_use_reentrant,
            )
            lm_loss = None
        else:
            outputs = self.judge_lm(
                inputs_embeds=masked_embeds,
                attention_mask=attention_mask,
                labels=None,
                use_cache=False,
            )
            lm_logits = outputs.logits
            lm_loss = self._compute_causal_lm_loss(
                lm_logits, labels, chunk_size=lm_loss_chunk_size
            )
            if compute_kl and lm_loss is not None:
                with torch.no_grad():
                    full_out = self.judge_lm(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        labels=None,
                        use_cache=False,
                    )
                compact_kl_loss = self._compute_compact_kl_loss(
                    full_out.logits,
                    lm_logits,
                    label_mask,
                    chunk_size=compact_kl_chunk_size,
                )
        return {
            "loss": lm_loss,
            "logits": lm_logits if need_logits else None,
            "compact_mask_loss": compact_mask_loss,
            "compact_con_loss": compact_con_loss,
            "compact_kl_loss": compact_kl_loss,
            "compact_pi_mean": compact_pi_mean,
            "compact_mask_mean": compact_mask_mean,
            "compact_pi_saturation": compact_pi_saturation,
            "gated_mean": gated_mean,
            "gated_add_norm": gated_add_norm,
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        disable_compactor = bool(batch.get("disable_compactor", False))
        disable_z_prompt = bool(batch.get("disable_z_prompt_insertion", False))
        lm_loss_chunk_size = batch.get("lm_loss_chunk_size", None)
        compact_kl_chunk_size = batch.get("compact_kl_chunk_size", None)
        if disable_compactor and disable_z_prompt:
            compute_compact_kl = bool(batch.get("compute_compact_kl", True))
            lm_out = self.lm_forward(
                batch["lm_input_ids"],
                batch["lm_attention_mask"],
                labels=batch.get("lm_labels"),
                pi_logits=None,
                response_types=None,
                z_task_addition=None,
                compact_prior=batch.get("compact_prior"),
                disable_compactor=True,
                compute_compact_kl=compute_compact_kl,
                lm_loss_chunk_size=lm_loss_chunk_size,
                compact_kl_chunk_size=compact_kl_chunk_size,
                return_logits=bool(batch.get("return_lm_logits", False)),
            )
            gated_mean = lm_out.get("gated_mean", lm_out["loss"].new_zeros(()))
            gated_add_norm = lm_out.get("gated_add_norm", lm_out["loss"].new_zeros(()))
            base = lm_out.get("logits")
            if base is None:
                base = lm_out.get("loss")
            if base is None:
                base = lm_out.get("compact_mask_loss")
            if base is None:
                base = lm_out.get("compact_kl_loss")
            if base is None:
                base = batch["lm_input_ids"]
            zero = base.new_zeros(())
            outputs = {
                "lm_loss": lm_out["loss"],
                "compact_mask_loss": lm_out["compact_mask_loss"],
                "compact_con_loss": lm_out["compact_con_loss"],
                "compact_kl_loss": lm_out["compact_kl_loss"],
                "compact_pi_mean": lm_out["compact_pi_mean"],
                "compact_mask_mean": lm_out["compact_mask_mean"],
                "compact_pi_saturation": lm_out["compact_pi_saturation"],
                "low_recon_mean_pred": None,
                "low_recon_logvar_pred": None,
                "low_recon_mean_target": None,
                "low_recon_logvar_target": None,
                "low_recon_mag_logits": None,
                "low_recon_mag_target": None,
                "low_recon_pred": None,
                "low_recon_target": None,
                "vq_task_loss": zero,
                "vq_task_commitment_loss": zero,
                "vq_task_codebook_loss": zero,
                "vq_task_perplexity": zero,
                "vq_task_usage_loss": zero,
                "vq_task_dead_fraction": zero,
                "vq_task_avg_distance": zero,
                "vq_align_loss": zero,
                "disentangle_loss": zero,
                "gated_mean": gated_mean,
                "gated_add_norm": gated_add_norm,
            }
            if bool(batch.get("return_lm_logits", False)):
                outputs["lm_logits"] = lm_out["logits"]
            return outputs
        enable_low_recon = bool(batch.get("enable_low_recon", True))
        (
            orig_hidden,
            orig_mask,
            orig_shape,
            low_hidden,
        ) = self._encode_bias_bundle(
            batch.get("original_prompt_input_ids", batch["original_input_ids"]),
            batch.get("original_prompt_attention_mask", batch["original_attention_mask"]),
            need_low_hidden=enable_low_recon,
            response_mask=batch.get("response_mask"),
        )
        if orig_shape is None:
            bsz = orig_mask.size(0)
            pairs = 1
        else:
            bsz, pairs = orig_shape

        disable_z_prompt = bool(batch.get("disable_z_prompt_insertion", False))
        if disable_z_prompt:
            z_task_pool = None
            z_bias_orig_pool = None
            z_task_addition = None
            low_recon_mean_pred = None
            low_recon_logvar_pred = None
            low_recon_mean_target = None
            low_recon_logvar_target = None
            low_recon_mag_logits = None
            low_recon_mag_target = None
            low_recon_pred = None
            low_recon_target = None
            zero = orig_hidden.new_zeros(())
            vq_task_loss = zero
            vq_task_commitment_loss = zero
            vq_task_codebook_loss = zero
            vq_task_perplexity = zero
            vq_task_usage_loss = zero
            vq_task_dead_fraction = zero
            vq_task_avg_distance = zero
        else:
            task_latents = self.task_mlp(orig_hidden)
            vq_task = self.vq_task(task_latents)
            z_task = vq_task.quantized
            z_task_dropout = float(batch.get("z_task_dropout", 0.0))
            z_task_lm = self._apply_z_dropout(z_task, z_task_dropout)
            lm_in_dtype = z_task_lm.dtype
            proj_weight = getattr(self.task_latent_to_lm, "weight", None)
            if proj_weight is not None and z_task_lm.dtype != proj_weight.dtype:
                z_task_lm = z_task_lm.to(dtype=proj_weight.dtype)
            z_task_lm = self.task_latent_to_lm(z_task_lm)
            if z_task_lm.dtype != lm_in_dtype:
                z_task_lm = z_task_lm.to(dtype=lm_in_dtype)

            z_task_pool = self._pool_hidden_mean(task_latents, orig_mask)
            z_task_pool = self.task_latent_to_encoder(z_task_pool)
            bias_tokens_orig = self.bias_mlp(orig_hidden)
            z_bias_orig_pool = self._pool_hidden_mean(bias_tokens_orig, orig_mask)

            if enable_low_recon and low_hidden is not None:
                low_recon_mean_pred = self.low_recon_mean_head(z_bias_orig_pool)
                low_recon_logvar_pred = self.low_recon_logvar_head(z_bias_orig_pool)
                low_mean, low_logvar = self._compute_low_stats(low_hidden, orig_mask)
                low_recon_mean_target = low_mean.detach()
                low_recon_logvar_target = low_logvar.detach()
                low_recon_mag_logits = self.low_recon_mag_head(z_bias_orig_pool)
                low_recon_mag_target = self._compute_low_mag_hist(
                    low_hidden, orig_mask
                ).detach()
                low_recon_pred = torch.cat(
                    [low_recon_mean_pred, low_recon_logvar_pred], dim=-1
                )
                low_recon_target = torch.cat(
                    [low_recon_mean_target, low_recon_logvar_target], dim=-1
                )
            else:
                low_recon_mean_pred = None
                low_recon_logvar_pred = None
                low_recon_mean_target = None
                low_recon_logvar_target = None
                low_recon_mag_logits = None
                low_recon_mag_target = None
                low_recon_pred = None
                low_recon_target = None
            vq_task_loss = vq_task.loss
            vq_task_commitment_loss = vq_task.commitment_loss
            vq_task_codebook_loss = vq_task.codebook_loss
            vq_task_perplexity = vq_task.perplexity
            vq_task_usage_loss = vq_task.usage_loss
            vq_task_dead_fraction = vq_task.dead_fraction
            vq_task_avg_distance = vq_task.avg_distance
            z_task_addition = None
        pi_logits = None
        lm_response_types = batch.get("lm_response_types")
        disable_compactor = bool(batch.get("disable_compactor", False))
        if lm_response_types is not None and not disable_compactor:
            orig_len = orig_hidden.size(1)
            orig_pair = orig_hidden.view(bsz, pairs, orig_len, orig_hidden.size(-1))
            pi_a = self.compact_head(orig_pair[:, 0]).squeeze(-1)
            pi_b = None
            if pairs > 1:
                pi_b = self.compact_head(orig_pair[:, 1]).squeeze(-1)
            pi_logits = self._scatter_compact_logits(lm_response_types, pi_a, pi_b)

        if lm_response_types is not None and not disable_z_prompt:
            z_task_addition = self._scatter_response_latents(
                lm_response_types, z_task_lm, orig_mask, pairs
            )

        compute_compact_kl = bool(batch.get("compute_compact_kl", True))
        lm_out = self.lm_forward(
            batch["lm_input_ids"],
            batch["lm_attention_mask"],
            labels=batch.get("lm_labels"),
            pi_logits=pi_logits,
            response_types=lm_response_types,
            z_task_addition=z_task_addition,
            compact_prior=batch.get("compact_prior"),
            disable_compactor=disable_compactor,
            compute_compact_kl=compute_compact_kl,
            lm_loss_chunk_size=lm_loss_chunk_size,
            compact_kl_chunk_size=compact_kl_chunk_size,
            return_logits=bool(batch.get("return_lm_logits", False)),
        )
        gated_mean = lm_out.get("gated_mean", lm_out["loss"].new_zeros(()))
        gated_add_norm = lm_out.get("gated_add_norm", lm_out["loss"].new_zeros(()))
        return_lm_logits = bool(batch.get("return_lm_logits", False))

        disentangle_loss = orig_hidden.new_zeros(())
        vq_align_loss = orig_hidden.new_zeros(())
        if not disable_z_prompt:
            response_mask = batch.get("response_mask")
            if torch.is_tensor(response_mask):
                mask = response_mask.view(-1).to(z_task_pool.device).bool()
                if mask.any():
                    task_dis = z_task_pool[mask]
                    bias_dis = z_bias_orig_pool[mask]
                else:
                    task_dis = z_task_pool
                    bias_dis = z_bias_orig_pool
            else:
                task_dis = z_task_pool
                bias_dis = z_bias_orig_pool
            disentangle_loss = self._disentangle_loss(task_dis, bias_dis)
            vq_align_loss = self._codebook_alignment_loss(self.vq_task)
        outputs = {
            "lm_loss": lm_out["loss"],
            "compact_mask_loss": lm_out["compact_mask_loss"],
            "compact_con_loss": lm_out["compact_con_loss"],
            "compact_kl_loss": lm_out["compact_kl_loss"],
            "compact_pi_mean": lm_out["compact_pi_mean"],
            "compact_mask_mean": lm_out["compact_mask_mean"],
            "compact_pi_saturation": lm_out["compact_pi_saturation"],
            "low_recon_mean_pred": low_recon_mean_pred,
            "low_recon_logvar_pred": low_recon_logvar_pred,
            "low_recon_mean_target": low_recon_mean_target,
            "low_recon_logvar_target": low_recon_logvar_target,
            "low_recon_mag_logits": low_recon_mag_logits,
            "low_recon_mag_target": low_recon_mag_target,
            "low_recon_pred": low_recon_pred,
            "low_recon_target": low_recon_target,
            "vq_task_loss": vq_task_loss,
            "vq_task_commitment_loss": vq_task_commitment_loss,
            "vq_task_codebook_loss": vq_task_codebook_loss,
            "vq_task_perplexity": vq_task_perplexity,
            "vq_task_usage_loss": vq_task_usage_loss,
            "vq_task_dead_fraction": vq_task_dead_fraction,
            "vq_task_avg_distance": vq_task_avg_distance,
            "vq_align_loss": vq_align_loss,
            "disentangle_loss": disentangle_loss,
            "gated_mean": gated_mean,
            "gated_add_norm": gated_add_norm,
        }
        proxy_enabled = bool(batch.get("proxy_labels_enabled", True))
        if proxy_enabled and not disable_z_prompt:
            outputs["length_bin_logits"] = self.length_bin_head(z_bias_orig_pool)
            outputs["nll_bin_logits"] = self.nll_bin_head(z_bias_orig_pool)
            outputs["ttr_bin_logits"] = self.ttr_bin_head(z_bias_orig_pool)
        if return_lm_logits:
            outputs["lm_logits"] = lm_out["logits"]
        return outputs
