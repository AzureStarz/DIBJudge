from __future__ import annotations

from dataclasses import dataclass
import math
import warnings
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from transformers import AutoModel, AutoModelForCausalLM


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
    ) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.mlp = None
        self.act = nn.GELU()
        proj_in = in_dim
        if layers > 0 and hidden_dim > 0:
            blocks = []
            for idx in range(layers):
                in_features = in_dim if idx == 0 else hidden_dim
                blocks.append(nn.Linear(in_features, hidden_dim))
                blocks.append(nn.GELU())
                if dropout > 0:
                    blocks.append(nn.Dropout(dropout))
            self.mlp = nn.Sequential(*blocks)
            proj_in = hidden_dim
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
    ) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.mlp = None
        self.act = nn.GELU()
        proj_in = in_dim
        if layers > 0 and hidden_dim > 0:
            blocks = []
            for idx in range(layers):
                in_features = in_dim if idx == 0 else hidden_dim
                blocks.append(nn.Linear(in_features, hidden_dim))
                blocks.append(nn.GELU())
                if dropout > 0:
                    blocks.append(nn.Dropout(dropout))
            self.mlp = nn.Sequential(*blocks)
            proj_in = hidden_dim
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
            pooled = last_token_pool(tokens, mask)
        else:
            pooled = tokens[:, -1]
        return self.classifier(pooled)


class DomainDiscriminatorHead(_TokenClassifierHead):
    pass


class HetsDiscriminatorHead(_TokenClassifierHead):
    pass


class PositionDiscriminatorHead(_TokenClassifierHead):
    pass


class ProxyRegressionHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 0,
        layers: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.mlp = None
        self.act = nn.GELU()
        proj_in = in_dim
        if layers > 0 and hidden_dim > 0:
            blocks = []
            for idx in range(layers):
                in_features = in_dim if idx == 0 else hidden_dim
                blocks.append(nn.Linear(in_features, hidden_dim))
                blocks.append(nn.GELU())
                if dropout > 0:
                    blocks.append(nn.Dropout(dropout))
            self.mlp = nn.Sequential(*blocks)
            proj_in = hidden_dim
        self.proj = nn.Linear(proj_in, 1)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        pooled = self.ln(pooled)
        if self.mlp is not None:
            pooled = self.mlp(pooled)
        else:
            pooled = self.act(pooled)
        return self.proj(pooled).squeeze(-1)


class ProxyClassifierHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dim: int = 0,
        layers: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.mlp = None
        self.act = nn.GELU()
        proj_in = in_dim
        if layers > 0 and hidden_dim > 0:
            blocks = []
            for idx in range(layers):
                in_features = in_dim if idx == 0 else hidden_dim
                blocks.append(nn.Linear(in_features, hidden_dim))
                blocks.append(nn.GELU())
                if dropout > 0:
                    blocks.append(nn.Dropout(dropout))
            self.mlp = nn.Sequential(*blocks)
            proj_in = hidden_dim
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
    ) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.mlp = None
        self.act = nn.GELU()
        proj_in = in_dim
        if layers > 0 and hidden_dim > 0:
            blocks = []
            for idx in range(layers):
                in_features = in_dim if idx == 0 else hidden_dim
                blocks.append(nn.Linear(in_features, hidden_dim))
                blocks.append(nn.GELU())
                if dropout > 0:
                    blocks.append(nn.Dropout(dropout))
            self.mlp = nn.Sequential(*blocks)
            proj_in = hidden_dim
        self.proj = nn.Linear(proj_in, out_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.ln(tokens)
        if self.mlp is not None:
            tokens = self.mlp(tokens)
        else:
            tokens = self.act(tokens)
        return self.proj(tokens)


class PromptProjector(nn.Module):
    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        prompt_len: int,
        hidden_dim: int = 0,
        layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.prompt_len = max(1, int(prompt_len))
        self.ln = nn.LayerNorm(in_dim)
        self.act = nn.GELU()
        proj_in = in_dim
        blocks = []
        hidden_dim = int(hidden_dim) if hidden_dim and hidden_dim > 0 else in_dim
        layers = max(0, int(layers))
        if layers > 0:
            for idx in range(layers):
                in_features = proj_in if idx == 0 else hidden_dim
                blocks.append(nn.Linear(in_features, hidden_dim))
                blocks.append(nn.GELU())
                if dropout > 0:
                    blocks.append(nn.Dropout(dropout))
            proj_in = hidden_dim
            self.mlp = nn.Sequential(*blocks)
        else:
            self.mlp = None
        self.proj = nn.Linear(proj_in, self.prompt_len * latent_dim)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        pooled = self.ln(pooled)
        if self.mlp is not None:
            pooled = self.mlp(pooled)
        else:
            pooled = self.act(pooled)
        out = self.proj(pooled)
        return out.view(pooled.size(0), self.prompt_len, -1)


@dataclass
class DIBJudgeConfig:
    judge_encoder_name: str = "google/mt5-base"
    judge_lm_name: str = "gpt2"
    z_latent_dim: int = 256
    z_prompt_prefix_len: int = 1
    z_prompt_postfix_len: int = 1
    z_prompt_len: int = 16
    prompt_mlp_hidden: int = 0
    prompt_mlp_layers: int = 1
    prompt_mlp_dropout: float = 0.1
    grl_lambda: float = 1.0
    bottleneck_noise_alpha: float = 8.0
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


class DIBJudgeModel(nn.Module):
    def __init__(self, config: DIBJudgeConfig) -> None:
        super().__init__()
        self.config = config
        self.shared_encoder = AutoModel.from_pretrained(config.judge_encoder_name)
        self.judge_lm = AutoModelForCausalLM.from_pretrained(config.judge_lm_name)

        encoder_hidden = getattr(self.shared_encoder.config, "hidden_size", None)
        if encoder_hidden is None:
            encoder_hidden = getattr(self.shared_encoder.config, "d_model", None)
        if encoder_hidden is None:
            raise ValueError("Unable to resolve shared encoder hidden size.")

        prompt_hidden = int(config.prompt_mlp_hidden)
        if prompt_hidden <= 0:
            prompt_hidden = 2 * encoder_hidden
        self.prompt_mlp = PromptProjector(
            encoder_hidden,
            config.z_latent_dim,
            prompt_len=max(1, int(config.z_prompt_len)),
            hidden_dim=prompt_hidden,
            layers=max(0, int(config.prompt_mlp_layers)),
            dropout=float(config.prompt_mlp_dropout),
        )
        self.prompt_noise_ln = nn.LayerNorm(config.z_latent_dim)
        self.z_to_lm = nn.Linear(config.z_latent_dim, self.judge_lm.config.hidden_size)
        self.z_prompt_prefix = nn.Parameter(
            torch.randn(1, max(0, int(config.z_prompt_prefix_len)), self.judge_lm.config.hidden_size)
            * 0.02
        )
        self.z_prompt_postfix = nn.Parameter(
            torch.randn(1, max(0, int(config.z_prompt_postfix_len)), self.judge_lm.config.hidden_size)
            * 0.02
        )
        bias_hidden = self._resolve_bias_proxy_hidden(config.z_latent_dim, 2)
        bias_layers = self._resolve_bias_proxy_layers()
        bias_dropout = self._resolve_bias_proxy_dropout()
        self.eng_domain_head = DomainDiscriminatorHead(
            config.z_latent_dim,
            hidden_dim=bias_hidden,
            layers=bias_layers,
            dropout=bias_dropout,
        )
        self.position_head = PositionDiscriminatorHead(
            config.z_latent_dim,
            hidden_dim=bias_hidden,
            layers=bias_layers,
            dropout=bias_dropout,
        )
        self.surface_stat_dim = 2 * encoder_hidden
        recon_hidden = self._resolve_bias_proxy_hidden(
            config.z_latent_dim, self.surface_stat_dim
        )
        self.low_recon_head = TokenReconstructionHead(
            config.z_latent_dim,
            self.surface_stat_dim,
            hidden_dim=recon_hidden,
            layers=bias_layers,
            dropout=bias_dropout,
        )
        self.length_bin_head = ProxyClassifierHead(
            config.z_latent_dim,
            num_classes=max(2, int(config.proxy_length_classes)),
            hidden_dim=bias_hidden,
            layers=bias_layers,
            dropout=bias_dropout,
        )
        self.nll_bin_head = ProxyClassifierHead(
            config.z_latent_dim,
            num_classes=max(2, int(config.proxy_nll_classes)),
            hidden_dim=bias_hidden,
            layers=bias_layers,
            dropout=bias_dropout,
        )
        self.ttr_bin_head = ProxyClassifierHead(
            config.z_latent_dim,
            num_classes=max(2, int(config.proxy_ttr_classes)),
            hidden_dim=bias_hidden,
            layers=bias_layers,
            dropout=bias_dropout,
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
        )
        self.register_buffer(
            "compact_mu_id",
            torch.tensor(int(config.compact_mu_token_id), dtype=torch.long),
            persistent=False,
        )

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
        current = self.state_dict()
        filtered = {}
        for key, value in state_dict.items():
            if not key.startswith(keep_prefixes):
                continue
            if key not in current:
                continue
            if current[key].shape != value.shape:
                continue
            filtered[key] = value
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        return list(missing), list(unexpected)

    @staticmethod
    def _flatten_pair_inputs(
        ids: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[int, int]]]:
        if ids.ndim == 3:
            bsz, pairs, seq_len = ids.shape
            return ids.reshape(bsz * pairs, seq_len), mask.reshape(bsz * pairs, seq_len), (bsz, pairs)
        return ids, mask, None

    @staticmethod
    def _pad_to_len(
        ids: torch.Tensor, mask: torch.Tensor, target_len: int, pad_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if ids.size(1) == target_len:
            return ids, mask
        padded_ids = ids.new_full((ids.size(0), target_len), pad_id)
        padded_mask = mask.new_zeros((mask.size(0), target_len))
        padded_ids[:, : ids.size(1)] = ids
        padded_mask[:, : mask.size(1)] = mask
        return padded_ids, padded_mask

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

    def _encode_bias_bundle(
        self,
        orig_ids: torch.Tensor,
        orig_mask: torch.Tensor,
        eng_ids: torch.Tensor,
        eng_mask: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[Tuple[int, int]],
        torch.Tensor,
    ]:
        orig_ids, orig_mask, orig_shape = self._flatten_pair_inputs(orig_ids, orig_mask)
        eng_ids, eng_mask, _ = self._flatten_pair_inputs(eng_ids, eng_mask)
        orig_len = orig_ids.size(1)

        orig_outputs = self.shared_encoder(
            input_ids=orig_ids,
            attention_mask=orig_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        orig_hidden = self._get_hidden(orig_outputs)
        hidden_states = getattr(orig_outputs, "hidden_states", None)
        if not hidden_states:
            raise ValueError("Shared encoder output missing hidden states for reconstruction.")
        idx = int(self.config.low_recon_layer)
        idx = max(0, min(idx, len(hidden_states) - 1))
        low_hidden = hidden_states[idx][:, :orig_len]

        eng_outputs = self.shared_encoder(
            input_ids=eng_ids,
            attention_mask=eng_mask,
            output_hidden_states=False,
            use_cache=False,
        )
        eng_hidden = self._get_hidden(eng_outputs)
        return (
            orig_hidden,
            orig_mask,
            eng_hidden,
            eng_mask,
            orig_shape,
            low_hidden,
        )

    def _build_z_prompts(
        self, hidden: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if mask is None:
            pooled = hidden[:, -1]
        else:
            pooled = last_token_pool(hidden, mask)
        return self.prompt_mlp(pooled)

    def _build_prompt_tokens(self, z_tokens: torch.Tensor) -> torch.Tensor:
        z_mapped = self.z_to_lm(z_tokens)
        prefix = self.z_prompt_prefix
        postfix = self.z_prompt_postfix
        if prefix.numel() == 0:
            prefix = z_mapped.new_zeros((1, 0, z_mapped.size(-1)))
        if postfix.numel() == 0:
            postfix = z_mapped.new_zeros((1, 0, z_mapped.size(-1)))
        return torch.cat([prefix.expand(z_mapped.size(0), -1, -1), z_mapped, postfix.expand(z_mapped.size(0), -1, -1)], dim=1)

    @staticmethod
    def _insert_prompt_tokens(
        embeds: torch.Tensor,
        attn: torch.Tensor,
        labels: Optional[torch.Tensor],
        response_types: torch.Tensor,
        pi_logits: Optional[torch.Tensor],
        inserts: List[Tuple[int, torch.Tensor]],
        pi_fill: float = 10.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        seq_len = embeds.size(0)
        new_embeds = embeds
        new_attn = attn
        new_labels = labels
        new_resp = response_types
        new_pi = pi_logits
        offset = 0
        for idx, prompt in inserts:
            insert_at = max(0, min(int(idx) + offset, new_embeds.size(0)))
            if prompt.numel() == 0:
                continue
            prompt_len = prompt.size(0)
            new_embeds = torch.cat([new_embeds[:insert_at], prompt, new_embeds[insert_at:]], dim=0)
            new_attn = torch.cat(
                [new_attn[:insert_at], new_attn.new_ones(prompt_len), new_attn[insert_at:]], dim=0
            )
            if new_labels is not None:
                new_labels = torch.cat(
                    [
                        new_labels[:insert_at],
                        new_labels.new_full((prompt_len,), -100),
                        new_labels[insert_at:],
                    ],
                    dim=0,
                )
            new_resp = torch.cat(
                [new_resp[:insert_at], new_resp.new_zeros(prompt_len), new_resp[insert_at:]], dim=0
            )
            if new_pi is not None:
                new_pi = torch.cat(
                    [
                        new_pi[:insert_at],
                        new_pi.new_full((prompt_len,), float(pi_fill)),
                        new_pi[insert_at:],
                    ],
                    dim=0,
                )
            offset += prompt_len
        return new_embeds, new_attn, new_labels, new_resp, new_pi

    def _compute_low_stats(
        self, hidden: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if mask is None:
            mean = hidden.mean(dim=1)
            var = hidden.var(dim=1, unbiased=False)
        else:
            mean = masked_mean(hidden, mask)
            var = masked_var(hidden, mask, mean)
        logvar = (var.clamp_min(1e-6)).log()
        return torch.cat([mean, logvar], dim=-1)

    def _resolve_grl_scale(self, batch: Dict[str, torch.Tensor]) -> float:
        grl_scale = batch.get("grl_lambda", self.config.grl_lambda)
        if torch.is_tensor(grl_scale):
            grl_scale = float(grl_scale.detach().item())
        return float(grl_scale)

    def _apply_bottleneck_noise(self, z: torch.Tensor, noise_alpha: float) -> Tuple[torch.Tensor, torch.Tensor]:
        l2_loss = z.pow(2).mean()
        if self.training and noise_alpha > 0:
            scale = noise_alpha / math.sqrt(z.size(1) * z.size(2))
            z = z + torch.randn_like(z) * scale
        z = self.prompt_noise_ln(z)
        return z, l2_loss

    def lm_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        pi_logits: Optional[torch.Tensor] = None,
        response_types: Optional[torch.Tensor] = None,
        z_prompts: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        embed_layer = self.judge_lm.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids)
        compact_mask_loss = inputs_embeds.new_zeros(())
        compact_con_loss = inputs_embeds.new_zeros(())
        compact_kl_loss = inputs_embeds.new_zeros(())
        compact_pi_mean = inputs_embeds.new_zeros(())
        compact_mask_mean = inputs_embeds.new_zeros(())
        compact_pi_saturation = inputs_embeds.new_zeros(())
        if pi_logits is None:
            pi_logits = inputs_embeds.new_zeros(inputs_embeds.size(0), inputs_embeds.size(1))
        if response_types is not None and z_prompts is not None:
            embeds_list = []
            attn_list = []
            labels_list = []
            resp_list = []
            pi_list = []
            max_len = 0
            for idx in range(inputs_embeds.size(0)):
                embeds = inputs_embeds[idx]
                attn = attention_mask[idx]
                lbl = labels[idx] if labels is not None else None
                resp = response_types[idx]
                pi = pi_logits[idx] if pi_logits is not None else None
                inserts = []
                a_idx = (resp == 1).nonzero(as_tuple=False).view(-1)
                if a_idx.numel() > 0 and "a" in z_prompts:
                    inserts.append((int(a_idx[-1].item()) + 1, z_prompts["a"][idx]))
                b_idx = (resp == 2).nonzero(as_tuple=False).view(-1)
                if b_idx.numel() > 0 and "b" in z_prompts:
                    inserts.append((int(b_idx[-1].item()) + 1, z_prompts["b"][idx]))
                inserts.sort(key=lambda x: x[0])
                embeds, attn, lbl, resp, pi = self._insert_prompt_tokens(
                    embeds, attn, lbl, resp, pi, inserts
                )
                embeds_list.append(embeds)
                attn_list.append(attn)
                resp_list.append(resp)
                pi_list.append(pi)
                if lbl is not None:
                    labels_list.append(lbl)
                max_len = max(max_len, embeds.size(0))

            hidden_dim = inputs_embeds.size(-1)
            padded_embeds = inputs_embeds.new_zeros((len(embeds_list), max_len, hidden_dim))
            padded_attn = attention_mask.new_zeros((len(attn_list), max_len))
            padded_resp = response_types.new_zeros((len(resp_list), max_len))
            padded_pi = pi_logits.new_zeros((len(pi_list), max_len)) if pi_logits is not None else None
            padded_labels = labels.new_full((len(labels_list), max_len), -100) if labels is not None else None
            for i, emb in enumerate(embeds_list):
                length = emb.size(0)
                padded_embeds[i, :length] = emb
                padded_attn[i, :length] = attn_list[i]
                padded_resp[i, :length] = resp_list[i]
                if padded_pi is not None and pi_list[i] is not None:
                    padded_pi[i, :length] = pi_list[i]
                if padded_labels is not None:
                    padded_labels[i, :length] = labels_list[i]
            inputs_embeds = padded_embeds
            attention_mask = padded_attn
            response_types = padded_resp
            pi_logits = padded_pi
            labels = padded_labels
        pi = torch.sigmoid(pi_logits)
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
        eps = 1e-6
        if mask_for_compact.any():
            pi_prompt = pi[mask_for_compact].clamp(min=eps, max=1.0 - eps)
            r = min(max(r, eps), 1.0 - eps)
            compact_mask_loss = (
                pi_prompt * (torch.log(pi_prompt) - math.log(r))
                + (1.0 - pi_prompt)
                * (torch.log(1.0 - pi_prompt) - math.log(1.0 - r))
            ).mean()
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
        outputs = self.judge_lm(
            inputs_embeds=masked_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
        if labels is not None and outputs.loss is not None:
            label_mask = labels.ne(-100)
            if label_mask.any():
                with torch.no_grad():
                    full_out = self.judge_lm(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        labels=None,
                        use_cache=False,
                    )
                full_logp = F.log_softmax(full_out.logits.float(), dim=-1)
                masked_logp = F.log_softmax(outputs.logits.float(), dim=-1)
                full_p = full_logp.exp()
                kl = (full_p * (full_logp - masked_logp)).sum(dim=-1)
                compact_kl_loss = kl[label_mask].mean()
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "compact_mask_loss": compact_mask_loss,
            "compact_con_loss": compact_con_loss,
            "compact_kl_loss": compact_kl_loss,
            "compact_pi_mean": compact_pi_mean,
            "compact_mask_mean": compact_mask_mean,
            "compact_pi_saturation": compact_pi_saturation,
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Expected batch keys: original_input_ids, original_attention_mask,
        english_input_ids, english_attention_mask, lm_input_ids,
        lm_attention_mask, lm_labels (optional). Optional proxy keys:
        proxy_* labels.
        Judge/english tensors can be shaped (batch, seq) or (batch, 2, seq)."""
        (
            orig_hidden,
            orig_mask,
            eng_ref_hidden,
            eng_ref_mask,
            orig_shape,
            low_hidden,
        ) = self._encode_bias_bundle(
            batch["original_input_ids"],
            batch["original_attention_mask"],
            batch["english_input_ids"],
            batch["english_attention_mask"],
        )
        if orig_shape is None:
            bsz = orig_mask.size(0)
            pairs = 1
        else:
            bsz, pairs = orig_shape

        z_orig = self._build_z_prompts(orig_hidden, orig_mask)
        z_ref = self._build_z_prompts(eng_ref_hidden, eng_ref_mask)
        noise_alpha = float(batch.get("bottleneck_noise_alpha", self.config.bottleneck_noise_alpha))
        z_orig, z_orig_l2 = self._apply_bottleneck_noise(z_orig, noise_alpha)
        z_ref, z_ref_l2 = self._apply_bottleneck_noise(z_ref, noise_alpha)
        grl_scale = self._resolve_grl_scale(batch)
        z_l2_loss = (z_orig_l2 + z_ref_l2) / 2.0
        z_orig_adv = gradient_reversal(z_orig, grl_scale)
        z_ref_adv = gradient_reversal(z_ref, grl_scale)
        if batch.get("bias_detach", False):
            z_orig_adv = z_orig_adv.detach()
            z_ref_adv = z_ref_adv.detach()
            z_l2_loss = z_l2_loss.detach()
        z_orig_pool = z_orig_adv.mean(dim=1)
        z_ref_pool = z_ref_adv.mean(dim=1)
        low_recon_pred = self.low_recon_head(z_orig_pool)
        low_recon_target = self._compute_low_stats(low_hidden, orig_mask).detach()
        pi_logits = None
        lm_response_types = batch.get("lm_response_types")
        if lm_response_types is not None:
            orig_len = orig_hidden.size(1)
            orig_pair = orig_hidden.view(bsz, pairs, orig_len, orig_hidden.size(-1))
            pi_a = self.compact_head(orig_pair[:, 0]).squeeze(-1)
            pi_b = None
            if pairs > 1:
                pi_b = self.compact_head(orig_pair[:, 1]).squeeze(-1)
            pi_logits = self._scatter_compact_logits(lm_response_types, pi_a, pi_b)

        z_prompts = None
        if lm_response_types is not None:
            z_pair = z_orig.view(bsz, pairs, z_orig.size(1), z_orig.size(2))
            z_prompts = {"a": self._build_prompt_tokens(z_pair[:, 0])}
            if pairs > 1:
                z_prompts["b"] = self._build_prompt_tokens(z_pair[:, 1])

        lm_out = self.lm_forward(
            batch["lm_input_ids"],
            batch["lm_attention_mask"],
            labels=batch.get("lm_labels"),
            pi_logits=pi_logits,
            response_types=lm_response_types,
            z_prompts=z_prompts,
        )
        return_lm_logits = bool(batch.get("return_lm_logits", False))

        domain_logits = self.eng_domain_head(torch.cat([z_orig_pool, z_ref_pool], dim=0))
        position_logits = self.position_head(z_orig_pool)
        outputs = {
            "lm_loss": lm_out["loss"],
            "compact_mask_loss": lm_out["compact_mask_loss"],
            "compact_con_loss": lm_out["compact_con_loss"],
            "compact_kl_loss": lm_out["compact_kl_loss"],
            "compact_pi_mean": lm_out["compact_pi_mean"],
            "compact_mask_mean": lm_out["compact_mask_mean"],
            "compact_pi_saturation": lm_out["compact_pi_saturation"],
            "domain_logits": domain_logits,
            "position_logits": position_logits,
            "low_recon_pred": low_recon_pred,
            "low_recon_target": low_recon_target,
            "z_l2_loss": z_l2_loss,
        }
        outputs["length_bin_logits"] = self.length_bin_head(z_orig_pool)
        outputs["nll_bin_logits"] = self.nll_bin_head(z_orig_pool)
        outputs["ttr_bin_logits"] = self.ttr_bin_head(z_orig_pool)
        if return_lm_logits:
            outputs["lm_logits"] = lm_out["logits"]
        return outputs
