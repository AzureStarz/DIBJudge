#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.multiprocessing as mp
try:
    import torch.distributed as dist
except ImportError:  # pragma: no cover - torch distributed not available
    dist = None
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import EmbedsPrompt

import eval_stats
if __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dibjudge.data import _find_response_span
from dibjudge.modeling import (
    DIBJudgeModel,
    RMSNorm,
    TokenMLP,
)
from dibjudge.bottlenecks import GaussianVIB
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


def _resolve_attn_impl(config: Optional[dict], override: Optional[str]) -> Optional[str]:
    if override:
        value = str(override).strip()
        return value.lower() if value else None
    if config:
        value = config.get("attn_implementation")
        if value:
            return str(value).strip().lower()
    return None


def _resolve_padding_side(
    config: Optional[dict], override: Optional[str], attn_impl: Optional[str]
) -> str:
    if override:
        return override
    if config:
        value = config.get("padding_side")
        if value:
            return str(value).strip().lower()
    if attn_impl == "flash_attention_2":
        return "left"
    return "right"


def _load_dibjudge_bundle(
    checkpoint: str,
    judge_encoder: Optional[str],
    device: torch.device,
    dtype: torch.dtype,
    trust_remote_code: bool,
    attn_implementation: Optional[str],
) -> DIBJudgePromptBundle:
    config = _load_checkpoint_config(checkpoint)
    encoder_name = _resolve_encoder_name(config, judge_encoder, checkpoint)
    state = _load_checkpoint_state(checkpoint)
    return DIBJudgePromptBundle(
        encoder_name,
        state,
        device,
        dtype,
        trust_remote_code,
        config=config,
        attn_implementation=attn_implementation,
    )


def _prompt_has_embeds(prompt: object) -> bool:
    if isinstance(prompt, dict):
        return "prompt_embeds" in prompt
    return hasattr(prompt, "prompt_embeds")


def _extract_prompt_embeds(prompt: object) -> Optional[torch.Tensor]:
    if isinstance(prompt, dict):
        embeds = prompt.get("prompt_embeds")
    else:
        embeds = getattr(prompt, "prompt_embeds", None)
    return embeds if torch.is_tensor(embeds) else None


def _append_prompt_log(path: str, prompts: List[object], start_idx: int) -> int:
    if not path:
        return start_idx
    os.makedirs(os.path.dirname(path), exist_ok=True)
    next_idx = start_idx
    with open(path, "a", encoding="utf-8") as handle:
        for prompt in prompts:
            if _prompt_has_embeds(prompt):
                embeds = _extract_prompt_embeds(prompt)
                shape = tuple(embeds.shape) if embeds is not None else None
                payload = {"idx": next_idx, "type": "embeds", "shape": shape}
            else:
                payload = {"idx": next_idx, "type": "text", "prompt": str(prompt)}
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            next_idx += 1
    return next_idx


def _normalize_embed_distributed(value: object) -> str:
    if value is None:
        return "false"
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return "true"
    if text in {"0", "false", "no", "n", "off"}:
        return "false"
    if text == "auto":
        return "auto"
    raise ValueError("embed-distributed must be one of: true, false, auto")


def _init_embed_distributed(enabled: bool) -> Optional[Dict[str, int]]:
    if not enabled or dist is None or not dist.is_available():
        return None
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return None
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return {
        "rank": rank,
        "world_size": dist.get_world_size(),
        "local_rank": local_rank,
    }


def _embed_spawn_worker(local_rank: int, world_size: int) -> None:
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    main()


def _maybe_spawn_embed_distributed(args: argparse.Namespace) -> bool:
    mode = _normalize_embed_distributed(args.embed_distributed)
    args.embed_distributed = mode
    if mode != "auto":
        return False
    if os.environ.get("DIBJUDGE_EMBED_SPAWNED") == "1":
        return False
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size_env > 1:
        args.embed_distributed = "true"
        return False
    world_size = int(getattr(args, "embed_world_size", 0) or 0)
    if world_size <= 0:
        world_size = torch.cuda.device_count()
    if world_size <= 1:
        print(
            "[warn] embed-distributed=auto requested but only one GPU detected; "
            "falling back to single-process prompt embedding.",
        )
        args.embed_distributed = "false"
        return False
    os.environ["DIBJUDGE_EMBED_SPAWNED"] = "1"
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    mp.spawn(_embed_spawn_worker, nprocs=world_size, args=(world_size,), join=True)
    return True


def _infer_branch_mlp_config(
    state: Dict[str, torch.Tensor], prefix: str
) -> Tuple[int, int, int, float]:
    linear_indices: List[int] = []
    weight_keys: Dict[int, str] = {}
    for key in state:
        if not (key.startswith(f"{prefix}mlp.") and key.endswith(".weight")):
            continue
        parts = key.split(".")
        if len(parts) < 3:
            continue
        idx_str = parts[2]
        if not idx_str.isdigit():
            continue
        idx = int(idx_str)
        # Accept direct Linear weights or SwiGLU projection weights.
        if len(parts) == 4 or (len(parts) == 5 and parts[3] == "proj"):
            linear_indices.append(idx)
            weight_keys.setdefault(idx, key)
    linear_indices = sorted(set(linear_indices))
    if not linear_indices:
        return 0, 0, 0, 0.0
    first_key = weight_keys[linear_indices[0]]
    hidden_weight = state[first_key]
    # SwiGLU projection doubles the output dimension.
    hidden_dim = (
        hidden_weight.shape[0] // 2
        if first_key.endswith(".proj.weight")
        else hidden_weight.shape[0]
    )
    layers = len(linear_indices)
    dropout = 0.0
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
        attn_implementation: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.config = config or {}
        if attn_implementation:
            try:
                self.shared_encoder = AutoModel.from_pretrained(
                    encoder_name,
                    trust_remote_code=trust_remote_code,
                    attn_implementation=attn_implementation,
                    torch_dtype=dtype,
                )
            except TypeError:
                warnings.warn(
                    f"attn_implementation={attn_implementation} unsupported for {encoder_name}; "
                    "falling back to default attention.",
                    RuntimeWarning,
                )
                self.shared_encoder = AutoModel.from_pretrained(
                    encoder_name,
                    trust_remote_code=trust_remote_code,
                    torch_dtype=dtype,
                )
            except ImportError as exc:
                warnings.warn(
                    f"attn_implementation={attn_implementation} unavailable for {encoder_name}: {exc}. "
                    "Falling back to default attention.",
                    RuntimeWarning,
                )
                self.shared_encoder = AutoModel.from_pretrained(
                    encoder_name,
                    trust_remote_code=trust_remote_code,
                    torch_dtype=dtype,
                )
        else:
            self.shared_encoder = AutoModel.from_pretrained(
                encoder_name, trust_remote_code=trust_remote_code, torch_dtype=dtype
            )
        self.shared_encoder.to(device)
        self.shared_encoder.eval()
        enc_hidden = getattr(self.shared_encoder.config, "hidden_size", None)
        if enc_hidden is None:
            enc_hidden = getattr(self.shared_encoder.config, "d_model", None)
        if enc_hidden is None:
            raise ValueError("Unable to resolve encoder hidden size.")

        latent_dim = enc_hidden
        vib_proj_weight = state.get("vib_to_lm.weight")
        if vib_proj_weight is not None:
            lm_hidden = vib_proj_weight.shape[0]
            if vib_proj_weight.shape[1] != latent_dim:
                raise ValueError("vib_to_lm input dim does not match encoder latents.")
        else:
            lm_hidden = latent_dim
        norm_type = "rms" if bool(self.config.get("use_rms_norm", False)) else "layernorm"
        norm_eps = float(self.config.get("rms_norm_eps", 1e-6))
        use_swiglu = bool(self.config.get("use_swiglu", False))
        task_layers, task_hidden, _first_idx, task_dropout = _infer_branch_mlp_config(
            state, "task_mlp."
        )
        task_hidden = int(self.config.get("task_mlp_hidden", task_hidden or 0))
        task_layers = int(self.config.get("task_mlp_layers", task_layers or 0))
        task_dropout = float(self.config.get("task_mlp_dropout", task_dropout))
        use_vib_cfg = self.config.get("use_vib")
        if use_vib_cfg is None:
            use_vib_cfg = any(k.startswith("vib_task.") for k in state) or (
                "vib_to_lm.weight" in state
            )
        self.use_vib = bool(use_vib_cfg)
        self.task_mlp = TokenMLP(
            enc_hidden,
            enc_hidden,
            hidden_dim=task_hidden,
            layers=task_layers,
            dropout=task_dropout,
            norm_type=norm_type,
            norm_eps=norm_eps,
            use_swiglu=use_swiglu,
        )
        self.task_post_norm = RMSNorm(enc_hidden, eps=norm_eps)
        if self.use_vib:
            vib_hidden = int(self.config.get("vib_hidden", 0))
            vib_layers = int(self.config.get("vib_layers", 2))
            vib_dropout = float(self.config.get("vib_dropout", 0.0))
            self.vib_task = GaussianVIB(
                enc_hidden,
                hidden_dim=vib_hidden,
                layers=vib_layers,
                dropout=vib_dropout,
                norm_eps=norm_eps,
                use_swiglu=use_swiglu,
                logvar_min=float(self.config.get("vib_logvar_min", -10.0)),
                logvar_max=float(self.config.get("vib_logvar_max", 10.0)),
            )
            self.vib_to_lm = torch.nn.Linear(enc_hidden, lm_hidden)
        else:
            self.vib_task = None
            self.vib_to_lm = None
        shared_state = {
            k.replace("shared_encoder.", ""): v
            for k, v in state.items()
            if k.startswith("shared_encoder.")
        }
        task_mlp_state = {
            k.replace("task_mlp.", ""): v
            for k, v in state.items()
            if k.startswith("task_mlp.")
        }
        vib_state = {k.replace("vib_task.", ""): v for k, v in state.items() if k.startswith("vib_task.")}
        vib_proj_state = {
            k.replace("vib_to_lm.", ""): v for k, v in state.items() if k.startswith("vib_to_lm.")
        }
        self.shared_encoder.load_state_dict(shared_state, strict=False)
        if task_mlp_state:
            self.task_mlp.load_state_dict(task_mlp_state, strict=False)
        task_norm_state = {
            k.replace("task_post_norm.", ""): v
            for k, v in state.items()
            if k.startswith("task_post_norm.")
        }
        if task_norm_state:
            self.task_post_norm.load_state_dict(task_norm_state, strict=False)
        if self.vib_task is not None and vib_state:
            self.vib_task.load_state_dict(vib_state, strict=False)
        if self.vib_to_lm is not None and vib_proj_state:
            self.vib_to_lm.load_state_dict(vib_proj_state, strict=False)
        self.compact_head = None
        self.compact_mu_id = None

        self.to(device=device, dtype=dtype)
        self.eval()

    @torch.no_grad()
    def build_prompt_features(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Optional[torch.Tensor]:
        outputs = self.shared_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        if self.vib_task is None or self.vib_to_lm is None:
            return None
        task_tokens = self.task_post_norm(self.task_mlp(hidden))
        vib_out = self.vib_task(task_tokens, sample=False)
        robust = self.vib_to_lm(vib_out.embeds)
        return robust


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
    pad_side: str = "right",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if pad_side not in {"left", "right"}:
        pad_side = "right"
    max_len = max(a_ids.size(1), b_ids.size(1))
    if a_ids.size(1) != max_len:
        padded = a_ids.new_full((a_ids.size(0), max_len), pad_id)
        padded_mask = a_mask.new_zeros((a_mask.size(0), max_len))
        if pad_side == "left":
            padded[:, -a_ids.size(1) :] = a_ids
            padded_mask[:, -a_mask.size(1) :] = a_mask
        else:
            padded[:, : a_ids.size(1)] = a_ids
            padded_mask[:, : a_mask.size(1)] = a_mask
        a_ids, a_mask = padded, padded_mask
    if b_ids.size(1) != max_len:
        padded = b_ids.new_full((b_ids.size(0), max_len), pad_id)
        padded_mask = b_mask.new_zeros((b_mask.size(0), max_len))
        if pad_side == "left":
            padded[:, -b_ids.size(1) :] = b_ids
            padded_mask[:, -b_mask.size(1) :] = b_mask
        else:
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


def _apply_z_task_addition(
    embeds: torch.Tensor,
    response_types: torch.Tensor,
    z_a: Optional[torch.Tensor],
    z_b: Optional[torch.Tensor],
    a_mask: Optional[torch.Tensor],
    b_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if z_a is None and z_b is None:
        return embeds
    if response_types.dim() == 1:
        response_types = response_types.unsqueeze(0)
    latents: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    if z_a is not None:
        latents.append(z_a)
        if a_mask is not None:
            masks.append(a_mask)
    if z_b is not None:
        latents.append(z_b)
        if b_mask is not None:
            masks.append(b_mask)
    pairs = max(1, len(latents))
    latents_tensor = torch.stack(latents, dim=0)
    mask_tensor = None
    if masks and len(masks) == len(latents):
        mask_tensor = torch.stack(masks, dim=0)
    addition = DIBJudgeModel._scatter_response_latents(
        response_types,
        latents_tensor,
        mask_tensor,
        pairs,
    )
    addition = addition.to(dtype=embeds.dtype, device=embeds.device)
    if addition.dim() == 3:
        addition = addition.squeeze(0)
    return embeds + addition


def _build_prompt_embeds(
    bundle: Optional[DIBJudgePromptBundle],
    lm_embed,
    lm_tokenizer,
    encoder_tokenizer,
    prompt: str,
    user_prompt: str,
    response_a: str,
    response_b: str,
    max_response_len: Optional[int],
    device: torch.device,
    dtype: torch.dtype,
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
    embed_device = lm_embed.weight.device
    input_ids = input_ids.to(embed_device)
    inputs_embeds = lm_embed(input_ids)
    if inputs_embeds.device != device or inputs_embeds.dtype != dtype:
        inputs_embeds = inputs_embeds.to(device=device, dtype=dtype)
    embeds = inputs_embeds.squeeze(0)
    attn = enc["attention_mask"].to(device=device).squeeze(0)

    response_types = _build_response_token_types(
        prompt, response_a, response_b, offsets, embeds.size(0)
    ).to(device=device)
    z_a = None
    z_b = None
    z_vib = None
    a_mask = None
    b_mask = None
    if bundle is not None:
        tok = encoder_tokenizer or lm_tokenizer
        if encoder_tokenizer is None:
            warnings.warn(
                "Encoder tokenizer unavailable; falling back to LM tokenizer for VIB features.",
                RuntimeWarning,
            )
        a_ids, a_mask = _tokenize_response(
            tok, _combine(user_prompt, response_a), max_response_len
        )
        b_ids, b_mask = _tokenize_response(
            tok, _combine(user_prompt, response_b), max_response_len
        )
        pad_id = tok.pad_token_id
        if pad_id is None:
            pad_id = tok.eos_token_id or 0
        pad_side = getattr(tok, "padding_side", "right")
        a_ids, a_mask, b_ids, b_mask = _pad_pair(
            a_ids, a_mask, b_ids, b_mask, pad_id, pad_side=pad_side
        )
        a_ids = a_ids.to(device)
        a_mask = a_mask.to(device)
        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)

        z_vib = bundle.build_prompt_features(
            torch.cat([a_ids, b_ids], dim=0),
            torch.cat([a_mask, b_mask], dim=0),
        )

    if z_vib is not None:
        z_a_vib = z_vib[0]
        z_b_vib = z_vib[1] if z_vib.size(0) > 1 else None
        embeds = _apply_z_task_addition(
            embeds,
            response_types,
            z_a_vib,
            z_b_vib,
            a_mask.squeeze(0) if torch.is_tensor(a_mask) else None,
            b_mask.squeeze(0) if torch.is_tensor(b_mask) else None,
        )
    return embeds


def _build_prompt_embeds_batch(
    bundle: Optional[DIBJudgePromptBundle],
    lm_embed,
    lm_tokenizer,
    encoder_tokenizer,
    prompts: List[str],
    user_prompts: List[str],
    responses_a: List[str],
    responses_b: List[str],
    max_response_len: Optional[int],
    device: torch.device,
    dtype: torch.dtype,
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
    pad_side = getattr(lm_tokenizer, "padding_side", "right")
    left_padding = pad_side == "left"

    embed_device = lm_embed.weight.device
    input_ids = input_ids.to(embed_device)
    inputs_embeds = lm_embed(input_ids)
    if inputs_embeds.device != device or inputs_embeds.dtype != dtype:
        inputs_embeds = inputs_embeds.to(device=device, dtype=dtype)

    z_vib = None
    resp_mask = None
    if bundle is not None:
        response_texts: List[str] = []
        for user_prompt, resp_a, resp_b in zip(user_prompts, responses_a, responses_b):
            response_texts.append(_combine(user_prompt, resp_a))
            response_texts.append(_combine(user_prompt, resp_b))
        tok = encoder_tokenizer or lm_tokenizer
        if encoder_tokenizer is None:
            warnings.warn(
                "Encoder tokenizer unavailable; falling back to LM tokenizer for VIB features.",
                RuntimeWarning,
            )
        resp_kwargs = {
            "add_special_tokens": False,
            "truncation": True,
            "return_tensors": "pt",
            "padding": True,
        }
        if max_response_len is not None:
            resp_kwargs["max_length"] = max_response_len
        resp_enc = tok(response_texts, **resp_kwargs)
        resp_ids = resp_enc["input_ids"].to(device)
        resp_mask = resp_enc["attention_mask"].to(device)
        z_vib = bundle.build_prompt_features(resp_ids, resp_mask)

    embeds_list: List[torch.Tensor] = []
    for i, prompt in enumerate(prompts):
        length = int(attention_mask[i].sum().item())
        if length <= 0:
            length = inputs_embeds.size(1)
        start = inputs_embeds.size(1) - length if left_padding else 0
        embeds = inputs_embeds[i, start : start + length]
        attn = attention_mask[i, start : start + length].to(device=device)
        offsets_i = None
        if offsets is not None:
            if torch.is_tensor(offsets):
                offsets_i = offsets[i, start : start + length]
            else:
                offsets_i = offsets[i][start : start + length]

        response_types = _build_response_token_types(
            prompt, responses_a[i], responses_b[i], offsets_i, embeds.size(0)
        ).to(device=device)

        z_a = None
        z_b = None
        a_mask = None
        b_mask = None
        if resp_mask is not None:
            a_mask = resp_mask[2 * i]
            if resp_mask.size(0) > (2 * i + 1):
                b_mask = resp_mask[2 * i + 1]
        if z_vib is not None:
            z_a = z_vib[2 * i]
            z_b = z_vib[2 * i + 1] if z_vib.size(0) > (2 * i + 1) else None
        embeds = _apply_z_task_addition(
            embeds,
            response_types,
            z_a,
            z_b,
            a_mask,
            b_mask,
        )
        embeds_list.append(embeds.detach().cpu())
    return embeds_list


def _prepare_seed_inputs(
    args: argparse.Namespace,
    seed: int,
    template: Optional[Union[str, Dict[str, object]]],
    mm_grouped: Optional[Dict[str, List[dict]]],
    mm_selected: List[str],
    mreward_pairs: List[Tuple[str, str]],
    bias_grouped: Optional[Dict[str, List[dict]]],
    bias_selected: List[str],
    expected_groups: List[Tuple[str, str]],
    use_compactor: bool,
    embed_resources: Optional[Dict[str, object]],
    dist_ctx: Optional[Dict[str, int]],
) -> Dict[str, object]:
    eval_stats.seed_everything(seed)
    seed_output_dir = os.path.join(args.output_dir, f"seed_{seed}")
    os.makedirs(seed_output_dir, exist_ok=True)

    raw_by_group: Dict[Tuple[str, str], List[dict]] = {}
    parsed_by_group: Dict[Tuple[str, str], List[dict]] = {}
    missing_groups = expected_groups
    if args.reuse_results:
        raw_by_group, parsed_by_group, missing_groups = ve._load_partial_results(
            seed_output_dir, expected_groups
        )

    tasks: List[dict] = []
    embed_prompts: List[Union[EmbedsPrompt, str]] = []

    if missing_groups:
        if embed_resources is None:
            raise ValueError("Embed resources are required to build prompts.")
        lm_tokenizer = embed_resources["tokenizer"]
        rng = random.Random(seed)

        prompts: List[str] = []
        if mm_grouped is not None:
            for lang in mm_selected:
                if ("MM-Eval", lang) not in missing_groups:
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
                        seed_value=seed,
                        max_model_len=args.max_model_len,
                    )
                    prompts.append(prompt)
                    tasks.append(meta)

        if mreward_pairs:
            mreward_dir = os.path.join(
                "data", "eval_data", "multilingual-reward-bench"
            )
            for lang, display_lang in mreward_pairs:
                if ("multilingual-reward-bench", display_lang) not in missing_groups:
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
                        seed_value=seed,
                        max_model_len=args.max_model_len,
                    )
                    prompts.append(prompt)
                    tasks.append(meta)

        if bias_grouped is not None:
            for dataset in bias_selected:
                if (ve.JUDGMENT_BENCHMARK, dataset) not in missing_groups:
                    continue
                rows = bias_grouped[dataset]
                if args.limit:
                    rows = ve._limit_judgment_pairs(rows, args.limit)
                for entry in rows:
                    for prompt, meta in ve._build_judgment_tasks(
                        entry,
                        dataset,
                        template,
                        lm_tokenizer,
                        seed_value=seed,
                        max_model_len=args.max_model_len,
                    ):
                        prompts.append(prompt)
                        tasks.append(meta)

        if args.enable_prompt_embeds:
            dtype = embed_resources["dtype"]
            device = embed_resources["device"]
            bundle = embed_resources.get("bundle")
            lm_embed = embed_resources.get("lm_embed")
            encoder_tokenizer = embed_resources.get("encoder_tokenizer")
            if lm_embed is None:
                raise ValueError("Prompt embeds enabled but lm_embed is missing.")

            embed_batch_size = max(1, int(args.embed_batch_size))
            rank = dist_ctx["rank"] if dist_ctx is not None else 0
            world_size = dist_ctx["world_size"] if dist_ctx is not None else 1
            indices = list(range(len(prompts)))
            if world_size > 1:
                indices = [idx for idx in indices if idx % world_size == rank]
            try:
                from tqdm.auto import tqdm

                iterator = tqdm(
                    range(0, len(indices), embed_batch_size),
                    desc="build_prompt_embeds",
                    dynamic_ncols=True,
                    disable=world_size > 1 and rank != 0,
                )
            except ImportError:
                iterator = range(0, len(indices), embed_batch_size)
            local_pairs: List[Tuple[int, EmbedsPrompt]] = []
            with torch.inference_mode():
                for start in iterator:
                    chunk_idx = indices[start : start + embed_batch_size]
                    if not chunk_idx:
                        continue
                    chunk_prompts = [prompts[idx] for idx in chunk_idx]
                    chunk_user_prompts = [
                        str(tasks[idx].get("prompt", "")) for idx in chunk_idx
                    ]
                    chunk_a = [str(tasks[idx]["answer_a"]) for idx in chunk_idx]
                    chunk_b = [str(tasks[idx]["answer_b"]) for idx in chunk_idx]
                    embeds_list = _build_prompt_embeds_batch(
                        bundle,
                        lm_embed,
                        lm_tokenizer,
                        encoder_tokenizer,
                        chunk_prompts,
                        chunk_user_prompts,
                        chunk_a,
                        chunk_b,
                        args.max_model_len,
                        device,
                        dtype,
                        use_compactor,
                    )
                    for idx, embeds in zip(chunk_idx, embeds_list):
                        local_pairs.append(
                            (idx, EmbedsPrompt(prompt_embeds=embeds.detach().cpu()))
                        )
            if world_size > 1 and dist is not None and dist.is_initialized():
                gathered: Optional[List[List[Tuple[int, EmbedsPrompt]]]] = None
                if rank == 0:
                    gathered = [None for _ in range(world_size)]
                dist.gather_object(local_pairs, gathered, dst=0)
                if rank == 0 and gathered is not None:
                    flat: List[Tuple[int, EmbedsPrompt]] = []
                    for shard in gathered:
                        if shard:
                            flat.extend(shard)
                    flat.sort(key=lambda item: item[0])
                    embed_prompts = [prompt for _, prompt in flat]
                else:
                    embed_prompts = []
            else:
                embed_prompts = [prompt for _, prompt in local_pairs]
        else:
            embed_prompts = prompts

    return {
        "seed_output_dir": seed_output_dir,
        "raw_by_group": raw_by_group,
        "parsed_by_group": parsed_by_group,
        "missing_groups": missing_groups,
        "tasks": tasks,
        "embed_prompts": embed_prompts,
    }


def _evaluate_seed(
    args: argparse.Namespace,
    seed: int,
    template: Optional[Union[str, Dict[str, object]]],
    verdict_a: str,
    verdict_b: str,
    mm_grouped: Optional[Dict[str, List[dict]]],
    mm_selected: List[str],
    mreward_pairs: List[Tuple[str, str]],
    bias_selected: List[str],
    expected_groups: List[Tuple[str, str]],
    use_compactor: bool,
    llm_handle: Optional[LLM],
    prepared: Dict[str, object],
) -> dict:
    summary = {"seed": seed, "benchmarks": {}}

    seed_output_dir = prepared["seed_output_dir"]
    raw_by_group = prepared["raw_by_group"]
    parsed_by_group = prepared["parsed_by_group"]
    missing_groups = prepared["missing_groups"]
    generated_groups: set[Tuple[str, str]] = set()

    if missing_groups:
        if not args.use_vllm:
            raise ValueError("DIBJudge evaluation requires --use_vllm for inference.")

        tasks = prepared["tasks"]
        embed_prompts = prepared["embed_prompts"]

        llm = llm_handle
        created_llm = False
        if llm is None:
            llm = LLM(
                model=args.model,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                dtype=args.dtype,
                trust_remote_code=args.trust_remote_code,
                seed=seed,
                enable_prompt_embeds=args.enable_prompt_embeds,
            )
            created_llm = True
        sampling_kwargs = {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k if args.top_k is not None else -1,
            "seed": seed,
        }
        try:
            sampling = SamplingParams(**sampling_kwargs)
        except TypeError:
            sampling_kwargs.pop("seed", None)
            sampling = SamplingParams(**sampling_kwargs)

        prompt_log = None
        prompt_log_idx = 0
        if args.debug_vllm_prompts:
            prompt_log = os.path.join(seed_output_dir, "vllm_prompts.jsonl")
            with open(prompt_log, "w", encoding="utf-8") as handle:
                handle.write("")

        completions: List[str] = []
        for chunk in ve._chunked(list(range(len(embed_prompts))), args.batch_size):
            batch_prompts = [embed_prompts[idx] for idx in chunk]
            if prompt_log is not None:
                prompt_log_idx = _append_prompt_log(
                    prompt_log, batch_prompts, prompt_log_idx
                )
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
            raw_entry = {
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
            parsed_entry = {
                "id": payload.get("id"),
                "language": payload.get("language"),
                "verdict": verdict,
                "expected_verdict": expected,
                "correct": bool(correct),
                "swapped": payload.get("swapped"),
            }
            if payload.get("pair_id") is not None:
                raw_entry["pair_id"] = payload.get("pair_id")
                parsed_entry["pair_id"] = payload.get("pair_id")
            if payload.get("sample_language") is not None:
                raw_entry["sample_language"] = payload.get("sample_language")
            if payload.get("group") is not None:
                raw_entry["group"] = payload.get("group")
                parsed_entry["group"] = payload.get("group")
            if payload.get("gold") is not None:
                raw_entry["gold"] = payload.get("gold")
            new_raw[group_key].append(raw_entry)
            new_parsed[group_key].append(parsed_entry)

        raw_by_group.update(new_raw)
        parsed_by_group.update(new_parsed)
        generated_groups = set(missing_groups)

        if created_llm:
            ve._shutdown_vllm(llm)

    write_outputs = bool(generated_groups)
    overall_flags: List[float] = []
    benchmark_flags: Dict[str, List[float]] = {"MM-Eval": [], "multilingual-reward-bench": []}

    if args.benchmark in {"MM-Eval", "both", "all"}:
        benchmark_rows = {}
        mm_total = 0
        mm_correct = 0
        for idx, lang in enumerate(mm_selected):
            parsed = parsed_by_group.get(("MM-Eval", lang))
            if not parsed:
                continue
            raw = raw_by_group.get(("MM-Eval", lang), [])
            if write_outputs and ("MM-Eval", lang) in generated_groups:
                raw_path = os.path.join(seed_output_dir, f"mm_eval_{lang}_raw.jsonl")
                parsed_path = os.path.join(
                    seed_output_dir, f"mm_eval_{lang}_parsed.jsonl"
                )
                ve._save_jsonl(raw_path, raw)
                ve._save_jsonl(parsed_path, parsed)
            stats = ve._summarize_with_ci(
                parsed, args.bootstrap_samples, args.bootstrap_confidence, seed + idx
            )
            mm_total += int(stats["total"])
            mm_correct += int(stats["correct"])
            benchmark_rows[lang] = stats
            flags = [1.0 if row.get("correct") else 0.0 for row in parsed]
            benchmark_flags["MM-Eval"].extend(flags)
            overall_flags.extend(flags)
        if mm_total:
            overall = {
                "total": mm_total,
                "correct": mm_correct,
                "accuracy": float(mm_correct) / mm_total,
            }
            if args.bootstrap_samples > 0 and benchmark_flags["MM-Eval"]:
                rng = random.Random(seed + 991)
                ci = eval_stats.bootstrap_mean_ci(
                    benchmark_flags["MM-Eval"],
                    args.bootstrap_samples,
                    args.bootstrap_confidence,
                    rng,
                )
                if ci is not None:
                    overall["accuracy_ci"] = {
                        "low": ci[0],
                        "high": ci[1],
                        "confidence": args.bootstrap_confidence,
                    }
            benchmark_rows["_overall"] = overall
        summary["benchmarks"]["MM-Eval"] = benchmark_rows

    if args.benchmark in {"multilingual-reward-bench", "both", "all"}:
        benchmark_rows = {}
        mr_total = 0
        mr_correct = 0
        for idx, (_lang, display_lang) in enumerate(mreward_pairs):
            parsed = parsed_by_group.get(("multilingual-reward-bench", display_lang))
            if not parsed:
                continue
            raw = raw_by_group.get(("multilingual-reward-bench", display_lang), [])
            if write_outputs and ("multilingual-reward-bench", display_lang) in generated_groups:
                raw_path = os.path.join(
                    seed_output_dir, f"mreward_{display_lang}_raw.jsonl"
                )
                parsed_path = os.path.join(
                    seed_output_dir, f"mreward_{display_lang}_parsed.jsonl"
                )
                ve._save_jsonl(raw_path, raw)
                ve._save_jsonl(parsed_path, parsed)
            stats = ve._summarize_with_ci(
                parsed,
                args.bootstrap_samples,
                args.bootstrap_confidence,
                seed + 1000 + idx,
            )
            mr_total += int(stats["total"])
            mr_correct += int(stats["correct"])
            benchmark_rows[display_lang] = stats
            flags = [1.0 if row.get("correct") else 0.0 for row in parsed]
            benchmark_flags["multilingual-reward-bench"].extend(flags)
            overall_flags.extend(flags)
        if mr_total:
            overall = {
                "total": mr_total,
                "correct": mr_correct,
                "accuracy": float(mr_correct) / mr_total,
            }
            if args.bootstrap_samples > 0 and benchmark_flags["multilingual-reward-bench"]:
                rng = random.Random(seed + 1991)
                ci = eval_stats.bootstrap_mean_ci(
                    benchmark_flags["multilingual-reward-bench"],
                    args.bootstrap_samples,
                    args.bootstrap_confidence,
                    rng,
                )
                if ci is not None:
                    overall["accuracy_ci"] = {
                        "low": ci[0],
                        "high": ci[1],
                        "confidence": args.bootstrap_confidence,
                    }
            benchmark_rows["_overall"] = overall
        summary["benchmarks"]["multilingual-reward-bench"] = benchmark_rows

    if args.benchmark in {ve.JUDGMENT_BENCHMARK, "all"}:
        bias_rows = {}
        for dataset in bias_selected:
            parsed = parsed_by_group.get((ve.JUDGMENT_BENCHMARK, dataset))
            if not parsed:
                continue
            raw = raw_by_group.get((ve.JUDGMENT_BENCHMARK, dataset), [])
            if write_outputs and (ve.JUDGMENT_BENCHMARK, dataset) in generated_groups:
                raw_path = os.path.join(
                    seed_output_dir, f"judgment_requests_{dataset}_raw.jsonl"
                )
                parsed_path = os.path.join(
                    seed_output_dir, f"judgment_requests_{dataset}_parsed.jsonl"
                )
                ve._save_jsonl(raw_path, raw)
                ve._save_jsonl(parsed_path, parsed)
            bias_rows[dataset] = ve._summarize_bias(parsed)
        summary["bias_benchmarks"] = bias_rows

    overall_total = len(overall_flags)
    overall_correct = int(sum(overall_flags))
    overall_acc = float(overall_correct) / overall_total if overall_total else 0.0
    summary["overall"] = {
        "total": overall_total,
        "correct": overall_correct,
        "accuracy": overall_acc,
    }
    if args.bootstrap_samples > 0 and overall_flags:
        rng = random.Random(seed + 7777)
        ci = eval_stats.bootstrap_mean_ci(
            overall_flags, args.bootstrap_samples, args.bootstrap_confidence, rng
        )
        if ci is not None:
            summary["overall"]["accuracy_ci"] = {
                "low": ci[0],
                "high": ci[1],
                "confidence": args.bootstrap_confidence,
            }

    summary_path = os.path.join(seed_output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary



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
        "--attn-implementation",
        default="flash_attention_2",
        help="Attention implementation override for the shared encoder.",
    )
    parser.add_argument(
        "--padding-side",
        choices=["left", "right"],
        default=None,
        help="Tokenizer padding side override (defaults to checkpoint config).",
    )
    parser.add_argument(
        "--benchmark",
        default="both",
        choices=[
            "MM-Eval",
            "multilingual-reward-bench",
            ve.JUDGMENT_BENCHMARK,
            "both",
            "all",
        ],
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
        "--judgment-request-dir",
        default=ve.DEFAULT_JUDGMENT_REQUEST_DIR,
        help="Directory containing judgment request jsonl datasets.",
    )
    parser.add_argument(
        "--judgment-datasets",
        nargs="+",
        default=None,
        help="Optional subset of judgment request datasets to evaluate.",
    )
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
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--limit-per-language",
        type=int,
        default=None,
        help="Limit number of judgment request pairs per language file.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--seed-step", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--compare-dir", default=None)
    parser.add_argument("--compare-label", default="baseline")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--enable-prompt-embeds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable prompt embeddings in vLLM.",
    )
    parser.add_argument(
        "--embed-distributed",
        nargs="?",
        const="true",
        default="false",
        choices=["true", "false", "auto"],
        help="Build prompt embeddings across multiple GPUs (true|false|auto).",
    )
    parser.add_argument(
        "--embed-world-size",
        type=int,
        default=0,
        help="Number of GPU worker processes for embed-distributed=auto (0=all visible).",
    )
    parser.add_argument(
        "--embed-lm-on-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Move the LM input embedding layer to GPU during prompt embed building.",
    )
    parser.add_argument(
        "--debug-prompt-only",
        action="store_true",
        help="Print one constructed prompt and exit without running evaluation.",
    )
    parser.add_argument(
        "--debug-vllm-prompts",
        action="store_true",
        help="Write prompts sent to vLLM.generate into seed output directories.",
    )
    parser.add_argument(
        "--reuse_results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing raw/parsed results in output_dir when available.",
    )
    args = parser.parse_args()
    if _maybe_spawn_embed_distributed(args):
        return

    ckpt_config = _load_checkpoint_config(args.checkpoint)
    attn_impl = _resolve_attn_impl(ckpt_config, args.attn_implementation)
    padding_side = _resolve_padding_side(ckpt_config, args.padding_side, attn_impl)
    use_compactor = False

    template, verdict_a, verdict_b = ve._load_template(
        args.template_path,
        args.template,
        args.verdict_pattern_a,
        args.verdict_pattern_b,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    seeds = eval_stats.resolve_seeds(
        args.seed, args.seeds, args.num_seeds, seed_step=args.seed_step
    )

    mm_grouped = None
    mm_selected: List[str] = []
    mreward_pairs: List[Tuple[str, str]] = []
    expected_groups: List[Tuple[str, str]] = []
    if args.benchmark in {"MM-Eval", "both", "all"}:
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

    if args.benchmark in {"multilingual-reward-bench", "both", "all"}:
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

    bias_grouped = None
    bias_selected: List[str] = []
    if args.benchmark in {ve.JUDGMENT_BENCHMARK, "all"}:
        bias_grouped = ve._load_judgment_requests(
            args.judgment_request_dir,
            args.judgment_datasets,
            limit_per_language=args.limit_per_language,
        )
        bias_selected = sorted(bias_grouped.keys())
        expected_groups.extend(
            [(ve.JUDGMENT_BENCHMARK, dataset) for dataset in bias_selected]
        )

    if not expected_groups:
        raise ValueError("No evaluation prompts were built for the selected benchmarks.")

    if args.debug_prompt_only:
        lm_tokenizer = AutoTokenizer.from_pretrained(
            args.model, use_fast=True, trust_remote_code=args.trust_remote_code
        )
        if lm_tokenizer.pad_token_id is None:
            lm_tokenizer.pad_token = lm_tokenizer.eos_token
        lm_tokenizer.padding_side = padding_side
        ve._PRINTED_TEST_PROMPT = True
        prompt = None
        rng = random.Random(seeds[0])
        if mm_grouped is not None:
            lang = mm_selected[0] if mm_selected else next(iter(mm_grouped.keys()))
            rows = mm_grouped[lang]
            if not rows:
                raise ValueError("MM-Eval has no rows to build a debug prompt.")
            prompt, _ = ve._build_task(
                rows[0],
                template,
                lm_tokenizer,
                rng,
                "MM-Eval",
                language_override=lang,
                seed_value=seeds[0],
                max_model_len=args.max_model_len,
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
                seed_value=seeds[0],
                max_model_len=args.max_model_len,
            )
        elif bias_grouped is not None and bias_selected:
            dataset = bias_selected[0]
            entries = bias_grouped[dataset]
            if not entries:
                raise ValueError("Judgment requests have no rows to build a debug prompt.")
            prompts = ve._build_judgment_tasks(
                entries[0],
                dataset,
                template,
                lm_tokenizer,
                seed_value=seeds[0],
                max_model_len=args.max_model_len,
            )
            print(prompts[0][0])
            return
        else:
            raise ValueError("No datasets loaded to build a debug prompt.")
        print(prompt)
        return

    any_missing = not args.reuse_results
    if args.reuse_results:
        for seed in seeds:
            seed_output_dir = os.path.join(args.output_dir, f"seed_{seed}")
            _, _, missing_groups = ve._load_partial_results(
                seed_output_dir, expected_groups
            )
            if missing_groups:
                any_missing = True
                break

    embed_resources: Optional[Dict[str, object]] = None
    dist_ctx = _init_embed_distributed(
        args.embed_distributed == "true" and bool(args.enable_prompt_embeds)
    )
    if args.embed_distributed == "true" and dist_ctx is None:
        print(
            "[warn] embed-distributed requested but WORLD_SIZE<=1; "
            "launch with torchrun --nproc_per_node=N to use multiple GPUs."
        )
    prepared_by_seed: Dict[int, Dict[str, object]] = {}
    if any_missing:
        if not args.use_vllm:
            raise ValueError("DIBJudge evaluation requires --use_vllm for inference.")
        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        if dist_ctx is not None:
            device = torch.device("cuda", dist_ctx["local_rank"])
        else:
            device = torch.device(args.device)
        lm_tokenizer = AutoTokenizer.from_pretrained(
            args.model, use_fast=True, trust_remote_code=args.trust_remote_code
        )
        if lm_tokenizer.pad_token_id is None:
            lm_tokenizer.pad_token = lm_tokenizer.eos_token
        lm_tokenizer.padding_side = padding_side

        bundle: Optional[DIBJudgePromptBundle] = None
        lm_embed = None
        lm_for_embed = None
        encoder_tokenizer = None
        if args.enable_prompt_embeds:
            bundle = _load_dibjudge_bundle(
                args.checkpoint,
                args.judge_encoder,
                device,
                dtype,
                args.trust_remote_code,
                attn_implementation=attn_impl,
            )
            encoder_name = _resolve_encoder_name(ckpt_config, args.judge_encoder, args.checkpoint)
            try:
                encoder_tokenizer = AutoTokenizer.from_pretrained(
                    encoder_name,
                    use_fast=False,
                    legacy=True,
                    trust_remote_code=args.trust_remote_code,
                )
            except TypeError:
                encoder_tokenizer = AutoTokenizer.from_pretrained(
                    encoder_name,
                    use_fast=False,
                    trust_remote_code=args.trust_remote_code,
                )
            if encoder_tokenizer.pad_token_id is None:
                encoder_tokenizer.pad_token = encoder_tokenizer.eos_token
            encoder_tokenizer.padding_side = padding_side
            lm_for_embed = AutoModelForCausalLM.from_pretrained(
                args.model,
                dtype=dtype,
                device_map="cpu",
                trust_remote_code=args.trust_remote_code,
            )
            lm_embed = lm_for_embed.get_input_embeddings()
            if bool(args.embed_lm_on_gpu):
                lm_embed = lm_embed.to(device=device, dtype=dtype)
            lm_embed.eval()

        embed_resources = {
            "dtype": dtype,
            "device": device,
            "tokenizer": lm_tokenizer,
            "encoder_tokenizer": encoder_tokenizer,
            "bundle": bundle,
            "lm_embed": lm_embed,
            "lm_for_embed": lm_for_embed,
        }

    for seed in seeds:
        prepared_by_seed[seed] = _prepare_seed_inputs(
            args,
            seed,
            template,
            mm_grouped,
            mm_selected,
            mreward_pairs,
            bias_grouped,
            bias_selected,
            expected_groups,
            use_compactor,
            embed_resources,
            dist_ctx,
        )

    if dist_ctx is not None and dist is not None and dist.is_initialized():
        dist.barrier()
        if dist_ctx["rank"] != 0:
            return

    if embed_resources is not None:
        bundle = embed_resources.get("bundle")
        if bundle is not None:
            bundle.to("cpu")
        if embed_resources.get("lm_embed") is not None:
            del embed_resources["lm_embed"]
        if embed_resources.get("lm_for_embed") is not None:
            del embed_resources["lm_for_embed"]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    llm_handle = None
    need_generation = any(
        prepared["missing_groups"] for prepared in prepared_by_seed.values()
    )
    if need_generation:
        llm_handle = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            seed=seeds[0],
            enable_prompt_embeds=args.enable_prompt_embeds,
        )
        effective_len = ve._get_vllm_max_model_len(llm_handle)
        if effective_len is not None and (
            args.max_model_len is None or effective_len < args.max_model_len
        ):
            args.max_model_len = int(effective_len)

    seed_summaries: List[dict] = []
    for seed in seeds:
        print(f"Running evaluation for seed={seed}")
        seed_summary = _evaluate_seed(
            args,
            seed,
            template,
            verdict_a,
            verdict_b,
            mm_grouped,
            mm_selected,
            mreward_pairs,
            bias_selected,
            expected_groups,
            use_compactor,
            llm_handle,
            prepared_by_seed[seed],
        )
        seed_summaries.append(seed_summary)

    if llm_handle is not None:
        ve._shutdown_vllm(llm_handle)

    aggregate = ve._aggregate_seed_summaries(
        seed_summaries,
        args.bootstrap_samples,
        args.bootstrap_confidence,
        bootstrap_seed=seeds[0],
    )

    table_rows: List[List[str]] = []
    for bench_name, bench in aggregate.get("benchmarks", {}).items():
        for lang, stats in bench.items():
            if lang == "_overall":
                continue
            total = int(stats.get("total", 0))
            correct = stats.get("correct", {}).get("formatted", "0.0000 ± 0.0000")
            accuracy = stats.get("accuracy", {}).get("formatted", "0.0000 ± 0.0000")
            table_rows.append(
                [bench_name, str(lang), str(total), str(correct), str(accuracy)]
            )
    headers = [
        "Benchmark",
        "Language",
        "Total",
        "Correct (mean ± std)",
        "Accuracy (mean ± std)",
    ]
    print(ve._format_table(table_rows, headers))

    overall_stats = aggregate.get("overall", {})
    if overall_stats:
        overall_acc = overall_stats.get("accuracy", {}).get("formatted", "0.0000 ± 0.0000")
        print("")
        print(f"Overall accuracy: {overall_acc} (n={len(seeds)})")
        if args.bootstrap_samples > 0 and "accuracy_mean_ci" in overall_stats:
            ci = overall_stats["accuracy_mean_ci"]
            print(
                f"Overall accuracy mean {ci['confidence']:.0%} CI: [{ci['low']:.4f}, {ci['high']:.4f}]"
            )

    for bench_name, bench in aggregate.get("benchmarks", {}).items():
        stats = bench.get("_overall")
        if not stats:
            continue
        bench_acc = stats.get("accuracy", {}).get("formatted", "0.0000 ± 0.0000")
        print(f"{bench_name} weighted accuracy: {bench_acc}")
        if args.bootstrap_samples > 0 and "accuracy_mean_ci" in stats:
            ci = stats["accuracy_mean_ci"]
            print(
                f"{bench_name} accuracy mean {ci['confidence']:.0%} CI: [{ci['low']:.4f}, {ci['high']:.4f}]"
            )

    bias_benchmarks = aggregate.get("bias_benchmarks", {})
    if bias_benchmarks:
        bias_rows: List[List[str]] = []
        bias_language_rows: List[List[str]] = []
        for name, stats in bias_benchmarks.items():
            overall = stats.get("overall", {})
            total_pairs = int(overall.get("total_pairs", 0))
            human = overall.get("consistent_human_win", {}).get(
                "formatted", "0.0000 ± 0.0000"
            )
            machine = overall.get("consistent_machine_win", {}).get(
                "formatted", "0.0000 ± 0.0000"
            )
            severity = overall.get("bias_severity", {}).get(
                "formatted", "0.0000 ± 0.0000"
            )
            bias_rows.append(
                [
                    name,
                    "_overall",
                    str(total_pairs),
                    str(human),
                    str(machine),
                    str(severity),
                ]
            )
            for group_name, group_stats in stats.get("by_group", {}).items():
                total_pairs = int(group_stats.get("total_pairs", 0))
                human = group_stats.get("consistent_human_win", {}).get(
                    "formatted", "0.0000 ± 0.0000"
                )
                machine = group_stats.get("consistent_machine_win", {}).get(
                    "formatted", "0.0000 ± 0.0000"
                )
                severity = group_stats.get("bias_severity", {}).get(
                    "formatted", "0.0000 ± 0.0000"
                )
                bias_rows.append(
                    [
                        name,
                        str(group_name),
                        str(total_pairs),
                        str(human),
                    str(machine),
                    str(severity),
                ]
            )
            for lang_name in sorted(stats.get("by_language", {}).keys()):
                lang_stats = stats.get("by_language", {}).get(lang_name, {})
                total_pairs = int(lang_stats.get("total_pairs", 0))
                human = lang_stats.get("consistent_human_win", {}).get(
                    "formatted", "0.0000 ± 0.0000"
                )
                machine = lang_stats.get("consistent_machine_win", {}).get(
                    "formatted", "0.0000 ± 0.0000"
                )
                severity = lang_stats.get("bias_severity", {}).get(
                    "formatted", "0.0000 ± 0.0000"
                )
                bias_language_rows.append(
                    [
                        name,
                        str(lang_name),
                        str(total_pairs),
                        str(human),
                        str(machine),
                        str(severity),
                    ]
                )
        headers = [
            "Benchmark",
            "Group",
            "Total Pairs",
            "Consistent Human",
            "Consistent Machine",
            "Bias Severity",
        ]
        print("")
        print(ve._format_table(bias_rows, headers))
        if bias_language_rows:
            language_headers = [
                "Benchmark",
                "Language",
                "Total Pairs",
                "Consistent Human",
                "Consistent Machine",
                "Bias Severity",
            ]
            print("")
            print(ve._format_table(bias_language_rows, language_headers))

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, ensure_ascii=False, indent=2)

    if args.compare_dir:
        compare_summaries, missing = eval_stats.load_seed_summaries(
            args.compare_dir, seeds
        )
        if missing:
            print(f"[warn] missing comparison summaries for seeds: {missing}")
        primary_by_seed = {summary["seed"]: summary for summary in seed_summaries}
        compare_by_seed = {summary["seed"]: summary for summary in compare_summaries}
        shared_seeds = sorted(set(primary_by_seed) & set(compare_by_seed))
        tests: Dict[str, Dict[str, float]] = {}
        if shared_seeds:
            primary_overall = [
                primary_by_seed[seed]["overall"]["accuracy"] for seed in shared_seeds
            ]
            compare_overall = [
                compare_by_seed[seed]["overall"]["accuracy"] for seed in shared_seeds
            ]
            tests["overall.accuracy"] = eval_stats.paired_ttest(
                primary_overall, compare_overall
            )
            for bench_name, bench in aggregate.get("benchmarks", {}).items():
                if "_overall" not in bench:
                    continue
                primary_vals = []
                compare_vals = []
                for seed in shared_seeds:
                    primary_bench = (
                        primary_by_seed[seed].get("benchmarks", {}).get(bench_name, {})
                    )
                    compare_bench = (
                        compare_by_seed[seed].get("benchmarks", {}).get(bench_name, {})
                    )
                    if "_overall" not in primary_bench or "_overall" not in compare_bench:
                        continue
                    primary_vals.append(primary_bench["_overall"]["accuracy"])
                    compare_vals.append(compare_bench["_overall"]["accuracy"])
                if primary_vals and compare_vals:
                    tests[f"{bench_name}.accuracy"] = eval_stats.paired_ttest(
                        primary_vals, compare_vals
                    )
        aggregate["paired_t_tests"] = {
            "compare_dir": args.compare_dir,
            "compare_label": args.compare_label,
            "alpha": args.alpha,
            "tests": tests,
        }
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(aggregate, handle, ensure_ascii=False, indent=2)
        ve._print_paired_tests(args.compare_label, tests, args.alpha)


if __name__ == "__main__":
    main()
