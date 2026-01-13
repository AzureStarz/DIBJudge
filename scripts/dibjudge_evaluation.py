#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import EmbedsPrompt

if __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dibjudge.data import _find_response_span
from dibjudge.modeling import BranchMLP, DIBJudgeModel, LatentHead, last_token_pool
from dibjudge.vq import VectorQuantizerEMA
import vanilla_evaluation as ve


def _read_checkpoint(path: str) -> Dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu")
    return state.get("model", state)


def _load_checkpoint_state(path: str) -> Dict[str, torch.Tensor]:
    if os.path.isdir(path):
        index_path = os.path.join(path, "pytorch_model.bin.index.json")
        if os.path.isfile(index_path):
            with open(index_path, "r", encoding="utf-8") as handle:
                index = json.load(handle)
            state: Dict[str, torch.Tensor] = {}
            for shard in sorted(set(index.get("weight_map", {}).values())):
                shard_path = os.path.join(path, shard)
                state.update(torch.load(shard_path, map_location="cpu"))
            return state
        bin_path = os.path.join(path, "pytorch_model.bin")
        if os.path.isfile(bin_path):
            return _read_checkpoint(bin_path)
        safetensors_path = os.path.join(path, "model.safetensors")
        if os.path.isfile(safetensors_path):
            try:
                from safetensors.torch import load_file
            except ImportError as exc:
                raise RuntimeError("safetensors is required to load model.safetensors") from exc
            return load_file(safetensors_path)
        safetensors_index = os.path.join(path, "model.safetensors.index.json")
        if os.path.isfile(safetensors_index):
            try:
                from safetensors.torch import load_file
            except ImportError as exc:
                raise RuntimeError("safetensors is required to load sharded safetensors") from exc
            with open(safetensors_index, "r", encoding="utf-8") as handle:
                index = json.load(handle)
            state: Dict[str, torch.Tensor] = {}
            for shard in sorted(set(index.get("weight_map", {}).values())):
                shard_path = os.path.join(path, shard)
                state.update(load_file(shard_path))
            return state
        raise FileNotFoundError(f"No checkpoint weights found under: {path}")
    return _read_checkpoint(path)


def _load_checkpoint_config(path: str) -> Optional[dict]:
    if not os.path.isdir(path):
        return None
    for name in ("dibjudge_config.json", "config.json"):
        cfg_path = os.path.join(path, name)
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
    return None


def _looks_like_tokenizer(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    candidates = [
        "tokenizer.json",
        "tokenizer_config.json",
        "spiece.model",
        "vocab.json",
        "merges.txt",
    ]
    return any(os.path.isfile(os.path.join(path, name)) for name in candidates)


def _resolve_encoder_name(
    config: Optional[dict], fallback: Optional[str], checkpoint_path: str
) -> str:
    if config:
        for key in ("judge_encoder_name", "judge_encoder", "encoder_name", "encoder"):
            value = config.get(key)
            if value:
                return str(value)
    if fallback:
        return str(fallback)
    if _looks_like_tokenizer(checkpoint_path):
        return checkpoint_path
    raise ValueError(
        "Missing judge encoder name. Provide --judge-encoder or ensure checkpoint has a config/tokenizer."
    )


def _load_dibjudge_bundle(
    checkpoint: str,
    judge_encoder: Optional[str],
    device: torch.device,
    dtype: torch.dtype,
    trust_remote_code: bool,
) -> DIBJudgePromptBundle:
    config = _load_checkpoint_config(checkpoint)
    encoder_name = _resolve_encoder_name(config, judge_encoder, checkpoint)
    state = _load_checkpoint_state(checkpoint)
    return DIBJudgePromptBundle(
        encoder_name, state, device, dtype, trust_remote_code, config=config
    )


def _infer_branch_mlp_config(
    state: Dict[str, torch.Tensor], prefix: str
) -> Tuple[int, int, int, float]:
    linear_indices = []
    for key in state:
        if key.startswith(f"{prefix}mlp.") and key.endswith(".weight"):
            parts = key.split(".")
            if len(parts) > 2:
                linear_indices.append(int(parts[2]))
    linear_indices = sorted(set(linear_indices))
    if not linear_indices:
        return 0, 0, 0, 0.0
    hidden_dim = state[f"{prefix}mlp.{linear_indices[0]}.weight"].shape[0]
    layers = len(linear_indices)
    dropout = 0.0
    if len(linear_indices) > 1 and (linear_indices[1] - linear_indices[0]) == 3:
        dropout = 0.1
    return layers, hidden_dim, linear_indices[0], dropout


class DIBJudgePromptBundle(torch.nn.Module):
    def __init__(
        self,
        encoder_name: str,
        state: Dict[str, torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
        trust_remote_code: bool,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.config = config or {}
        self.shared_encoder = AutoModel.from_pretrained(
            encoder_name, trust_remote_code=trust_remote_code
        )
        self.shared_encoder.to(device)
        self.shared_encoder.eval()
        enc_hidden = getattr(self.shared_encoder.config, "hidden_size", None)
        if enc_hidden is None:
            enc_hidden = getattr(self.shared_encoder.config, "d_model", None)
        if enc_hidden is None:
            raise ValueError("Unable to resolve encoder hidden size.")

        task_vq_to_lm_weight = state.get("task_vq_to_lm.weight")
        if task_vq_to_lm_weight is None:
            raise ValueError("Checkpoint missing task_vq_to_lm weights.")
        z_dim = task_vq_to_lm_weight.shape[1]
        lm_hidden = task_vq_to_lm_weight.shape[0]
        prompt_out = state["task_mlp.proj.weight"].shape[0]
        if prompt_out % z_dim != 0:
            raise ValueError("Task prompt projection shape is incompatible with latent dim.")
        prompt_len = prompt_out // z_dim
        layers, hidden_dim, _first_idx, dropout = _infer_branch_mlp_config(state, "task_mlp.")
        if hidden_dim <= 0:
            hidden_dim = 2 * enc_hidden
        self.task_mlp = BranchMLP(
            enc_hidden,
            z_dim,
            prompt_len=prompt_len,
            hidden_dim=hidden_dim,
            layers=layers,
            dropout=dropout,
        )
        codebook = state.get("vq_task.codebook")
        if codebook is None:
            raise ValueError("Checkpoint missing vq_task.codebook.")
        if codebook.dim() == 3:
            num_codebooks, num_codes, sub_dim = codebook.shape
            vq_dim = int(num_codebooks * sub_dim)
            inferred_num_codebooks = int(num_codebooks)
        else:
            num_codes, vq_dim = codebook.shape
            inferred_num_codebooks = 1
        if vq_dim != z_dim:
            raise ValueError("VQ codebook dim does not match task_vq_to_lm input.")
        vq_num_codebooks = int(self.config.get("vq_num_codebooks", inferred_num_codebooks))
        if vq_num_codebooks != inferred_num_codebooks:
            vq_num_codebooks = inferred_num_codebooks
        vq_commitment = float(self.config.get("vq_commitment_gamma", 0.05))
        vq_decay = float(self.config.get("vq_ema_decay", 0.99))
        vq_use_ema = bool(self.config.get("vq_use_ema", True))
        vq_codebook_trainable = bool(self.config.get("vq_codebook_trainable", False))
        vq_dead_code = float(self.config.get("vq_dead_code_threshold", 0.1))
        vq_reset_dead = bool(self.config.get("vq_reset_dead_codes", True))
        vq_normalize = bool(self.config.get("vq_normalize_inputs", True))
        self.vq_task = VectorQuantizerEMA(
            num_codes=int(num_codes),
            dim=int(vq_dim),
            num_codebooks=vq_num_codebooks,
            commitment_cost=vq_commitment,
            decay=vq_decay,
            use_ema=vq_use_ema,
            codebook_trainable=vq_codebook_trainable,
            dead_code_threshold=vq_dead_code,
            reset_dead_codes=vq_reset_dead,
            normalize_inputs=vq_normalize,
        )
        self.task_vq_to_lm = torch.nn.Linear(z_dim, lm_hidden)
        compact_state = {
            k.replace("compact_head.", ""): v
            for k, v in state.items()
            if k.startswith("compact_head.")
        }
        self.compact_head = None
        if compact_state:
            compact_layers, compact_hidden, _first_idx, compact_dropout = (
                _infer_branch_mlp_config(state, "compact_head.")
            )
            compact_hidden = int(
                self.config.get("compact_head_hidden", compact_hidden or 0)
            )
            compact_layers = int(
                self.config.get("compact_head_layers", compact_layers or 0)
            )
            compact_dropout = float(
                self.config.get("compact_head_dropout", compact_dropout)
            )
            self.compact_head = LatentHead(
                enc_hidden,
                1,
                latent_clip=0.0,
                hidden_dim=compact_hidden,
                layers=compact_layers,
                dropout=compact_dropout,
            )

        shared_state = {
            k.replace("shared_encoder.", ""): v
            for k, v in state.items()
            if k.startswith("shared_encoder.")
        }
        task_state = {
            k.replace("task_mlp.", ""): v
            for k, v in state.items()
            if k.startswith("task_mlp.")
        }
        vq_state = {
            k.replace("vq_task.", ""): v
            for k, v in state.items()
            if k.startswith("vq_task.")
        }
        proj_state = {
            k.replace("task_vq_to_lm.", ""): v
            for k, v in state.items()
            if k.startswith("task_vq_to_lm.")
        }
        self.shared_encoder.load_state_dict(shared_state, strict=False)
        self.task_mlp.load_state_dict(task_state, strict=False)
        self.vq_task.load_state_dict(vq_state, strict=False)
        self.task_vq_to_lm.load_state_dict(proj_state, strict=False)
        if self.compact_head is not None:
            self.compact_head.load_state_dict(compact_state, strict=False)
        self.compact_mu_id = None
        if "compact_mu_token_id" in self.config:
            self.compact_mu_id = int(self.config["compact_mu_token_id"])

        self.to(device=device, dtype=dtype)
        self.eval()

    @torch.no_grad()
    def build_prompt_features(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        outputs = self.shared_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        pooled = last_token_pool(hidden, attention_mask)
        task_tokens = self.task_mlp(pooled)
        z_quant = self.vq_task(task_tokens).quantized
        z_prompt = self.task_vq_to_lm(z_quant)
        compact_logits = None
        if self.compact_head is not None:
            compact_logits = self.compact_head(hidden).squeeze(-1)
        return z_prompt, compact_logits


def _tokenize_response(
    tokenizer, text: str, max_length: Optional[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    kwargs = {
        "add_special_tokens": False,
        "truncation": True,
        "return_tensors": "pt",
    }
    if max_length is not None:
        kwargs["max_length"] = max_length
    enc = tokenizer(text, **kwargs)
    return enc["input_ids"], enc["attention_mask"]


def _pad_pair(
    a_ids: torch.Tensor,
    a_mask: torch.Tensor,
    b_ids: torch.Tensor,
    b_mask: torch.Tensor,
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(a_ids.size(1), b_ids.size(1))
    if a_ids.size(1) != max_len:
        padded = a_ids.new_full((a_ids.size(0), max_len), pad_id)
        padded_mask = a_mask.new_zeros((a_mask.size(0), max_len))
        padded[:, : a_ids.size(1)] = a_ids
        padded_mask[:, : a_mask.size(1)] = a_mask
        a_ids, a_mask = padded, padded_mask
    if b_ids.size(1) != max_len:
        padded = b_ids.new_full((b_ids.size(0), max_len), pad_id)
        padded_mask = b_mask.new_zeros((b_mask.size(0), max_len))
        padded[:, : b_ids.size(1)] = b_ids
        padded_mask[:, : b_mask.size(1)] = b_mask
        b_ids, b_mask = padded, padded_mask
    return a_ids, a_mask, b_ids, b_mask


def _build_response_token_types(
    prompt: str,
    response_a: str,
    response_b: str,
    offset_mapping: Optional[torch.Tensor],
    seq_len: int,
) -> torch.Tensor:
    token_types = torch.zeros((seq_len,), dtype=torch.long)
    if offset_mapping is None:
        return token_types
    span_a = _find_response_span(prompt, response_a)
    span_b = None
    if response_b:
        span_b = _find_response_span(prompt, response_b, start_at=span_a[1] if span_a else None)
    offsets = offset_mapping.tolist()
    for tok_idx, (start, end) in enumerate(offsets):
        if start == end == 0:
            continue
        if span_a and start < span_a[1] and end > span_a[0]:
            token_types[tok_idx] = 1
        elif span_b and start < span_b[1] and end > span_b[0]:
            token_types[tok_idx] = 2
    return token_types


def _build_prompt_embeds(
    bundle: Optional[DIBJudgePromptBundle],
    lm_embed,
    lm_tokenizer,
    prompt: str,
    user_prompt: str,
    response_a: str,
    response_b: str,
    max_response_len: Optional[int],
    device: torch.device,
    dtype: torch.dtype,
    use_z_soft_prompt: bool,
    use_compactor: bool,
) -> torch.Tensor:
    def _combine(_prompt: str, resp: str) -> str:
        if not resp.strip():
            return ""
        return resp

    enc = lm_tokenizer(
        prompt,
        add_special_tokens=True,
        return_tensors="pt",
        return_offsets_mapping=getattr(lm_tokenizer, "is_fast", False),
    )
    input_ids = enc["input_ids"]
    offsets = enc.get("offset_mapping")
    if offsets is not None:
        offsets = offsets[0]
    inputs_embeds = lm_embed(input_ids).to(device=device, dtype=dtype)
    embeds = inputs_embeds.squeeze(0)
    attn = enc["attention_mask"].to(device=device).squeeze(0)

    response_types = _build_response_token_types(
        prompt, response_a, response_b, offsets, embeds.size(0)
    ).to(device=device)
    pi_logits = embeds.new_zeros(embeds.size(0))
    z_a = None
    z_b = None
    if bundle is not None:
        a_ids, a_mask = _tokenize_response(
            lm_tokenizer, _combine(user_prompt, response_a), max_response_len
        )
        b_ids, b_mask = _tokenize_response(
            lm_tokenizer, _combine(user_prompt, response_b), max_response_len
        )
        pad_id = lm_tokenizer.pad_token_id
        if pad_id is None:
            pad_id = lm_tokenizer.eos_token_id or 0
        a_ids, a_mask, b_ids, b_mask = _pad_pair(a_ids, a_mask, b_ids, b_mask, pad_id)
        a_ids = a_ids.to(device)
        a_mask = a_mask.to(device)
        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)

        z_tokens, compact_logits = bundle.build_prompt_features(
            torch.cat([a_ids, b_ids], dim=0),
            torch.cat([a_mask, b_mask], dim=0),
        )
        z_a = z_tokens[0]
        z_b = z_tokens[1] if z_tokens.size(0) > 1 else None
        if use_compactor and compact_logits is not None:
            pi_a = compact_logits[0]
            pi_b = compact_logits[1] if compact_logits.size(0) > 1 else None
            pi_logits = DIBJudgeModel._scatter_compact_logits(
                response_types.unsqueeze(0),
                pi_a.unsqueeze(0),
                pi_b.unsqueeze(0) if pi_b is not None else None,
            ).squeeze(0)
    elif use_z_soft_prompt or use_compactor:
        raise ValueError("z soft prompt is enabled but no prompt bundle was provided.")

    if use_z_soft_prompt:
        a_idx = (response_types == 1).nonzero(as_tuple=False).view(-1)
        b_idx = (response_types == 2).nonzero(as_tuple=False).view(-1)
        inserts: List[Tuple[int, torch.Tensor]] = []
        if a_idx.numel() > 0 and z_a is not None:
            inserts.append((int(a_idx.min().item()), z_a))
        if b_idx.numel() > 0 and z_b is not None:
            inserts.append((int(b_idx.min().item()), z_b))
        inserts.sort(key=lambda x: x[0])
        embeds, attn, _labels, response_types, pi_logits = DIBJudgeModel._insert_prompt_tokens(
            embeds, attn, None, response_types, pi_logits, inserts
        )

    if bundle is None or not use_compactor:
        return embeds

    prompt_mask = attn.bool()
    response_mask = response_types > 0
    pi = torch.sigmoid(pi_logits)
    response_mask = response_mask & prompt_mask
    if not response_mask.any():
        warnings.warn(
            "lm_response_types has no response spans for this prompt; "
            "compact masking is skipped.",
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
    mu_id = bundle.compact_mu_id
    if mu_id is None:
        mu_id = lm_tokenizer.pad_token_id
        if mu_id is None:
            mu_id = lm_tokenizer.eos_token_id or 0
    mu_ids = torch.tensor([mu_id], device=lm_embed.weight.device)
    mu_embed = lm_embed(mu_ids).to(device=embeds.device, dtype=embeds.dtype).view(1, -1)
    masked_embeds = m.unsqueeze(-1) * embeds + (1.0 - m).unsqueeze(-1) * mu_embed
    return masked_embeds


def _build_prompt_embeds_batch(
    bundle: Optional[DIBJudgePromptBundle],
    lm_embed,
    lm_tokenizer,
    prompts: List[str],
    user_prompts: List[str],
    responses_a: List[str],
    responses_b: List[str],
    max_response_len: Optional[int],
    device: torch.device,
    dtype: torch.dtype,
    use_z_soft_prompt: bool,
    use_compactor: bool,
) -> List[torch.Tensor]:
    if not prompts:
        return []
    if not (
        len(prompts)
        == len(user_prompts)
        == len(responses_a)
        == len(responses_b)
    ):
        raise ValueError("Prompt/embed inputs must have matching lengths.")

    def _combine(_prompt: str, resp: str) -> str:
        if not resp.strip():
            return ""
        return resp

    enc = lm_tokenizer(
        prompts,
        add_special_tokens=True,
        return_tensors="pt",
        padding=True,
        return_offsets_mapping=getattr(lm_tokenizer, "is_fast", False),
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    offsets = enc.get("offset_mapping")

    inputs_embeds = lm_embed(input_ids).to(device=device, dtype=dtype)

    z_tokens = None
    compact_logits = None
    if bundle is not None:
        response_texts: List[str] = []
        for user_prompt, resp_a, resp_b in zip(user_prompts, responses_a, responses_b):
            response_texts.append(_combine(user_prompt, resp_a))
            response_texts.append(_combine(user_prompt, resp_b))
        resp_kwargs = {
            "add_special_tokens": False,
            "truncation": True,
            "return_tensors": "pt",
            "padding": True,
        }
        if max_response_len is not None:
            resp_kwargs["max_length"] = max_response_len
        resp_enc = lm_tokenizer(response_texts, **resp_kwargs)
        resp_ids = resp_enc["input_ids"].to(device)
        resp_mask = resp_enc["attention_mask"].to(device)
        z_tokens, compact_logits = bundle.build_prompt_features(resp_ids, resp_mask)
    elif use_z_soft_prompt or use_compactor:
        raise ValueError("z soft prompt is enabled but no prompt bundle was provided.")

    embeds_list: List[torch.Tensor] = []
    for i, prompt in enumerate(prompts):
        length = int(attention_mask[i].sum().item())
        if length <= 0:
            length = inputs_embeds.size(1)
        embeds = inputs_embeds[i, :length]
        attn = attention_mask[i, :length].to(device=device)
        offsets_i = None
        if offsets is not None:
            if torch.is_tensor(offsets):
                offsets_i = offsets[i, :length]
            else:
                offsets_i = offsets[i][:length]

        response_types = _build_response_token_types(
            prompt, responses_a[i], responses_b[i], offsets_i, embeds.size(0)
        ).to(device=device)
        pi_logits = embeds.new_zeros(embeds.size(0))
        if bundle is not None and use_compactor and compact_logits is not None:
            pi_a = compact_logits[2 * i]
            pi_b = None
            if compact_logits.size(0) > (2 * i + 1):
                pi_b = compact_logits[2 * i + 1]
            pi_logits = DIBJudgeModel._scatter_compact_logits(
                response_types.unsqueeze(0),
                pi_a.unsqueeze(0),
                pi_b.unsqueeze(0) if pi_b is not None else None,
            ).squeeze(0)

        if use_z_soft_prompt:
            a_idx = (response_types == 1).nonzero(as_tuple=False).view(-1)
            b_idx = (response_types == 2).nonzero(as_tuple=False).view(-1)
            inserts: List[Tuple[int, torch.Tensor]] = []
            if a_idx.numel() > 0 and z_tokens is not None:
                inserts.append((int(a_idx.min().item()), z_tokens[2 * i]))
            if b_idx.numel() > 0 and z_tokens is not None:
                if z_tokens.size(0) > (2 * i + 1):
                    inserts.append((int(b_idx.min().item()), z_tokens[2 * i + 1]))
            inserts.sort(key=lambda x: x[0])
            embeds, attn, _labels, response_types, pi_logits = DIBJudgeModel._insert_prompt_tokens(
                embeds, attn, None, response_types, pi_logits, inserts
            )

        if bundle is None or not use_compactor:
            embeds_list.append(embeds)
            continue

        prompt_mask = attn.bool()
        response_mask = response_types > 0
        pi = torch.sigmoid(pi_logits)
        response_mask = response_mask & prompt_mask
        if not response_mask.any():
            warnings.warn(
                "lm_response_types has no response spans for this prompt; "
                "compact masking is skipped.",
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
        mu_id = bundle.compact_mu_id
        if mu_id is None:
            mu_id = lm_tokenizer.pad_token_id
            if mu_id is None:
                mu_id = lm_tokenizer.eos_token_id or 0
        mu_ids = torch.tensor([mu_id], device=lm_embed.weight.device)
        mu_embed = lm_embed(mu_ids).to(device=embeds.device, dtype=embeds.dtype).view(1, -1)
        masked_embeds = m.unsqueeze(-1) * embeds + (1.0 - m).unsqueeze(-1) * mu_embed
        embeds_list.append(masked_embeds)
    return embeds_list


def main() -> None:
    parser = argparse.ArgumentParser(description="DIBJudge evaluation with vLLM embeds.")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to DIBJudge HF checkpoint directory or a state dict file.",
    )
    parser.add_argument(
        "--judge-encoder",
        default=None,
        help="Fallback encoder name when checkpoint config is missing.",
    )
    parser.add_argument(
        "--benchmark",
        default="both",
        choices=["MM-Eval", "multilingual-reward-bench", "both"],
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Languages to evaluate (default: all available languages per benchmark).",
    )
    parser.add_argument(
        "--template_path",
        default="configs/eval_config",
        help="Directory containing template json files.",
    )
    parser.add_argument("--template", default=None, help="Template name or json path.")
    parser.add_argument("--verdict-pattern-a", default=None)
    parser.add_argument("--verdict-pattern-b", default=None)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument(
        "--use_vllm",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=256,
        help="Batch size for building prompt embeddings (smaller uses less memory).",
    )
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--enable-prompt-embeds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable prompt embeddings in vLLM.",
    )
    parser.add_argument(
        "--z-soft-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable z soft prompts (default: follow checkpoint config).",
    )
    parser.add_argument(
        "--use-compactor",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable compact masking using predicted pi logits.",
    )
    parser.add_argument(
        "--debug-prompt-only",
        action="store_true",
        help="Print one constructed prompt and exit without running evaluation.",
    )
    parser.add_argument(
        "--reuse_results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing raw/parsed results in output_dir when available.",
    )
    args = parser.parse_args()

    ckpt_config = _load_checkpoint_config(args.checkpoint)
    use_compactor = bool(args.use_compactor)
    if args.z_soft_prompt is None:
        if ckpt_config is not None:
            use_z_soft_prompt = bool(ckpt_config.get("z_soft_prompt", True))
        else:
            use_z_soft_prompt = True
    else:
        use_z_soft_prompt = bool(args.z_soft_prompt)
    if use_z_soft_prompt and not args.enable_prompt_embeds and not args.debug_prompt_only:
        raise ValueError("z soft prompt requires --enable-prompt-embeds.")
    if use_compactor and not args.enable_prompt_embeds and not args.debug_prompt_only:
        raise ValueError("compactor requires --enable-prompt-embeds.")

    template, verdict_a, verdict_b = ve._load_template(
        args.template_path,
        args.template,
        args.verdict_pattern_a,
        args.verdict_pattern_b,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    summary = {"benchmarks": {}}
    table_rows: List[List[str]] = []
    rng = random.Random(args.seed)

    mm_grouped = None
    mm_selected: List[str] = []
    mreward_pairs: List[Tuple[str, str]] = []
    expected_groups: List[Tuple[str, str]] = []
    if args.benchmark in {"MM-Eval", "both"}:
        mm_dir = os.path.join("data", "eval_data", "MM-Eval")
        dataset = ve._load_mm_eval(mm_dir)
        grouped = ve._group_by_language(dataset)
        grouped = ve._filter_mm_eval_core_languages(grouped)
        available = sorted(grouped.keys())
        mm_selected = args.languages or available
        missing = [lang for lang in mm_selected if lang not in grouped]
        if missing:
            raise ValueError(f"MM-Eval missing languages: {', '.join(missing)}")
        mm_grouped = grouped
        expected_groups.extend([("MM-Eval", lang) for lang in mm_selected])

    if args.benchmark in {"multilingual-reward-bench", "both"}:
        mreward_dir = os.path.join("data", "eval_data", "multilingual-reward-bench")
        available = ve._available_mreward_languages(mreward_dir)
        if not available:
            available = sorted(ve.SHORT_TO_CONFIG.values())
        selected, missing = ve._resolve_mreward_languages(args.languages, available)
        if missing:
            raise ValueError(
                "multilingual-reward-bench missing languages: " + ", ".join(missing)
            )
        mreward_pairs = [
            (lang, ve.CONFIG_TO_SHORT.get(lang, lang)) for lang in selected
        ]
        expected_groups.extend(
            [("multilingual-reward-bench", display) for _, display in mreward_pairs]
        )

    if not expected_groups:
        raise ValueError("No evaluation prompts were built for the selected benchmarks.")

    if args.debug_prompt_only:
        lm_tokenizer = AutoTokenizer.from_pretrained(
            args.model, use_fast=True, trust_remote_code=args.trust_remote_code
        )
        if lm_tokenizer.pad_token_id is None:
            lm_tokenizer.pad_token = lm_tokenizer.eos_token
        ve._PRINTED_TEST_PROMPT = True
        prompt = None
        if mm_grouped is not None:
            lang = mm_selected[0] if mm_selected else next(iter(mm_grouped.keys()))
            rows = mm_grouped[lang]
            if not rows:
                raise ValueError("MM-Eval has no rows to build a debug prompt.")
            prompt, _ = ve._build_task(
                rows[0], template, lm_tokenizer, rng, "MM-Eval", language_override=lang
            )
        elif mreward_pairs:
            mreward_dir = os.path.join("data", "eval_data", "multilingual-reward-bench")
            lang, display_lang = mreward_pairs[0]
            dataset = ve._load_mreward_language(mreward_dir, lang)
            rows = list(dataset)
            if not rows:
                raise ValueError("multilingual-reward-bench has no rows to build a debug prompt.")
            prompt, _ = ve._build_task(
                rows[0],
                template,
                lm_tokenizer,
                rng,
                "multilingual-reward-bench",
                language_override=display_lang,
            )
        else:
            raise ValueError("No datasets loaded to build a debug prompt.")
        print(prompt)
        return

    raw_by_group: Dict[Tuple[str, str], List[dict]] = {}
    parsed_by_group: Dict[Tuple[str, str], List[dict]] = {}
    missing_groups = expected_groups
    if args.reuse_results:
        raw_by_group, parsed_by_group, missing_groups = ve._load_partial_results(
            args.output_dir, expected_groups
        )
    missing_set = set(missing_groups)
    generated_groups: set[Tuple[str, str]] = set()

    if missing_groups:
        if not args.use_vllm:
            raise ValueError("DIBJudge evaluation requires --use_vllm for inference.")

        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        device = torch.device(args.device)

        lm_tokenizer = AutoTokenizer.from_pretrained(
            args.model, use_fast=True, trust_remote_code=args.trust_remote_code
        )
        if lm_tokenizer.pad_token_id is None:
            lm_tokenizer.pad_token = lm_tokenizer.eos_token

        bundle: Optional[DIBJudgePromptBundle] = None
        if use_z_soft_prompt or use_compactor:
            bundle = _load_dibjudge_bundle(
                args.checkpoint, args.judge_encoder, device, dtype, args.trust_remote_code
            )

        lm_for_embed = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=dtype,
            device_map="cpu",
            trust_remote_code=args.trust_remote_code,
        )
        lm_embed = lm_for_embed.get_input_embeddings()
        lm_embed.eval()

        tasks: List[dict] = []
        prompts: List[str] = []
        if mm_grouped is not None:
            for lang in mm_selected:
                if ("MM-Eval", lang) not in missing_set:
                    continue
                rows = mm_grouped[lang]
                if args.limit:
                    rows = rows[: args.limit]
                for row in rows:
                    prompt, meta = ve._build_task(
                        row,
                        template,
                        lm_tokenizer,
                        rng,
                        "MM-Eval",
                        language_override=lang,
                    )
                    prompts.append(prompt)
                    tasks.append(meta)

        if mreward_pairs:
            mreward_dir = os.path.join(
                "data", "eval_data", "multilingual-reward-bench"
            )
            for lang, display_lang in mreward_pairs:
                if ("multilingual-reward-bench", display_lang) not in missing_set:
                    continue
                dataset = ve._load_mreward_language(mreward_dir, lang)
                rows = list(dataset)
                if args.limit:
                    rows = rows[: args.limit]
                for row in rows:
                    prompt, meta = ve._build_task(
                        row,
                        template,
                        lm_tokenizer,
                        rng,
                        "multilingual-reward-bench",
                        language_override=display_lang,
                    )
                    prompts.append(prompt)
                    tasks.append(meta)

        embed_prompts: List[EmbedsPrompt] = []
        embed_batch_size = max(1, int(args.embed_batch_size))
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(
                range(0, len(prompts), embed_batch_size),
                desc="build_prompt_embeds",
                dynamic_ncols=True,
            )
        except ImportError:
            iterator = range(0, len(prompts), embed_batch_size)
        with torch.inference_mode():
            for start in iterator:
                chunk = list(range(start, min(start + embed_batch_size, len(prompts))))
                chunk_prompts = [prompts[idx] for idx in chunk]
                chunk_user_prompts = [str(tasks[idx].get("prompt", "")) for idx in chunk]
                chunk_a = [str(tasks[idx]["answer_a"]) for idx in chunk]
                chunk_b = [str(tasks[idx]["answer_b"]) for idx in chunk]
                embeds_list = _build_prompt_embeds_batch(
                    bundle,
                    lm_embed,
                    lm_tokenizer,
                    chunk_prompts,
                    chunk_user_prompts,
                    chunk_a,
                    chunk_b,
                    args.max_model_len,
                    device,
                    dtype,
                    use_z_soft_prompt,
                    use_compactor,
                )
                embed_prompts.extend(
                    EmbedsPrompt(prompt_embeds=embeds.cpu()) for embeds in embeds_list
                )

        if device.type == "cuda":
            bundle.to("cpu")
            del lm_embed
            del lm_for_embed
            torch.cuda.empty_cache()

        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            seed=args.seed,
            enable_prompt_embeds=args.enable_prompt_embeds,
        )
        sampling = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k if args.top_k is not None else -1,
        )

        completions: List[str] = []
        for chunk in ve._chunked(list(range(len(embed_prompts))), args.batch_size):
            batch_prompts = [embed_prompts[idx] for idx in chunk]
            outputs = llm.generate(batch_prompts, sampling)
            for out in outputs:
                text = ""
                if out.outputs:
                    text = out.outputs[0].text
                completions.append(text)

        if len(completions) != len(tasks):
            raise RuntimeError(
                f"Expected {len(tasks)} completions, received {len(completions)}."
            )

        new_raw = defaultdict(list)
        new_parsed = defaultdict(list)
        for idx, completion in enumerate(completions):
            payload = tasks[idx]
            verdict = ve._parse_verdict(completion, verdict_a, verdict_b)
            expected = payload.get("expected_verdict")
            correct = verdict == expected
            group_key = (payload.get("benchmark", ""), payload.get("language", ""))
            new_raw[group_key].append(
                {
                    "id": payload.get("id"),
                    "language": payload.get("language"),
                    "prompt": payload.get("prompt"),
                    "answer_a": payload.get("answer_a"),
                    "answer_b": payload.get("answer_b"),
                    "completion": completion,
                    "verdict": verdict,
                    "expected_verdict": expected,
                    "swapped": payload.get("swapped"),
                }
            )
            new_parsed[group_key].append(
                {
                    "id": payload.get("id"),
                    "language": payload.get("language"),
                    "verdict": verdict,
                    "expected_verdict": expected,
                    "correct": bool(correct),
                    "swapped": payload.get("swapped"),
                }
            )

        raw_by_group.update(new_raw)
        parsed_by_group.update(new_parsed)
        generated_groups = set(missing_groups)

        ve._shutdown_vllm(llm)

    write_outputs = bool(generated_groups)
    if args.benchmark in {"MM-Eval", "both"}:
        benchmark_rows = {}
        mm_total = 0
        mm_correct = 0
        for lang in mm_selected:
            parsed = parsed_by_group.get(("MM-Eval", lang))
            if not parsed:
                continue
            raw = raw_by_group.get(("MM-Eval", lang), [])
            if write_outputs and ("MM-Eval", lang) in generated_groups:
                raw_path = os.path.join(args.output_dir, f"mm_eval_{lang}_raw.jsonl")
                parsed_path = os.path.join(args.output_dir, f"mm_eval_{lang}_parsed.jsonl")
                ve._save_jsonl(raw_path, raw)
                ve._save_jsonl(parsed_path, parsed)
            total, correct, acc = ve._summarize(parsed)
            mm_total += total
            mm_correct += correct
            benchmark_rows[lang] = {"total": total, "correct": correct, "accuracy": acc}
            table_rows.append(
                ["MM-Eval", str(lang), str(total), str(correct), f"{acc:.4f}"]
            )
        if mm_total:
            benchmark_rows["_overall"] = {
                "total": mm_total,
                "correct": mm_correct,
                "accuracy": float(mm_correct) / mm_total,
            }
        summary["benchmarks"]["MM-Eval"] = benchmark_rows

    if args.benchmark in {"multilingual-reward-bench", "both"}:
        benchmark_rows = {}
        mr_total = 0
        mr_correct = 0
        for _lang, display_lang in mreward_pairs:
            parsed = parsed_by_group.get(("multilingual-reward-bench", display_lang))
            if not parsed:
                continue
            raw = raw_by_group.get(("multilingual-reward-bench", display_lang), [])
            if write_outputs and ("multilingual-reward-bench", display_lang) in generated_groups:
                raw_path = os.path.join(args.output_dir, f"mreward_{display_lang}_raw.jsonl")
                parsed_path = os.path.join(
                    args.output_dir, f"mreward_{display_lang}_parsed.jsonl"
                )
                ve._save_jsonl(raw_path, raw)
                ve._save_jsonl(parsed_path, parsed)
            total, correct, acc = ve._summarize(parsed)
            mr_total += total
            mr_correct += correct
            benchmark_rows[display_lang] = {
                "total": total,
                "correct": correct,
                "accuracy": acc,
            }
            table_rows.append(
                ["mreward", str(display_lang), str(total), str(correct), f"{acc:.4f}"]
            )
        if mr_total:
            benchmark_rows["_overall"] = {
                "total": mr_total,
                "correct": mr_correct,
                "accuracy": float(mr_correct) / mr_total,
            }
        summary["benchmarks"]["multilingual-reward-bench"] = benchmark_rows

    headers = ["Benchmark", "Language", "Total", "Correct", "Accuracy"]
    print(ve._format_table(table_rows, headers))

    overall_total = 0
    overall_correct = 0
    for benchmark in summary["benchmarks"].values():
        for lang, stats in benchmark.items():
            if lang == "_overall":
                continue
            overall_total += stats["total"]
            overall_correct += stats["correct"]
    overall_acc = float(overall_correct) / overall_total if overall_total else 0.0
    print("")
    print(
        f"Summary: ✅ {overall_correct} pass / ❌ {overall_total - overall_correct} fail"
    )
    print(f"Overall accuracy: {overall_acc:.4f}")

    weighted_total = 0
    weighted_correct = 0
    for bench_name, bench in summary["benchmarks"].items():
        stats = bench.get("_overall")
        if not stats:
            continue
        weighted_total += stats["total"]
        weighted_correct += stats["correct"]
        bench_acc = float(stats["accuracy"])
        print(f"{bench_name} weighted accuracy: {bench_acc:.4f}")
    if weighted_total:
        weighted_acc = float(weighted_correct) / weighted_total
        print(f"Weighted average accuracy (benchmarks): {weighted_acc:.4f}")

    summary["overall"] = {
        "total": overall_total,
        "correct": overall_correct,
        "accuracy": overall_acc,
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
