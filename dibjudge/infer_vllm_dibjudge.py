from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from transformers import AutoConfig, AutoTokenizer, T5EncoderModel
from vllm import LLM, SamplingParams
from vllm.inputs import EmbedsPrompt

from .modeling import PromptProjector, last_token_pool


def _read_jsonl(path: str) -> Iterable[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _render_prompt(example: Dict[str, object], prompt_field: str) -> str:
    prompt = str(example.get(prompt_field, ""))
    if "{instruction}" in prompt:
        prompt = re.sub(r"\{instruction\}", str(example.get("instruction", "")), prompt)
    if "{response_A}" in prompt:
        prompt = re.sub(r"\{response_A\}", str(example.get("response_A", "")), prompt)
    if "{response_B}" in prompt:
        prompt = re.sub(r"\{response_B\}", str(example.get("response_B", "")), prompt)
    return prompt


def _find_span(text: str, substring: str) -> Optional[Tuple[int, int]]:
    if not substring:
        return None
    start = text.find(substring)
    if start == -1:
        return None
    return start, start + len(substring)


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
    span_a = _find_span(prompt, response_a)
    span_b = _find_span(prompt, response_b)
    offsets = offset_mapping.tolist()
    for tok_idx, (start, end) in enumerate(offsets):
        if start == end == 0:
            continue
        if span_a and start < span_a[1] and end > span_a[0]:
            token_types[tok_idx] = 1
        elif span_b and start < span_b[1] and end > span_b[0]:
            token_types[tok_idx] = 2
    return token_types


@dataclass
class JudgeBottleneckConfig:
    judge_encoder_name: str
    z_latent_dim: int
    lm_hidden_size: int
    z_prompt_len: int = 16
    prompt_mlp_hidden: int = 0
    prompt_mlp_layers: int = 1
    prompt_mlp_dropout: float = 0.1


class JudgeBottleneck(torch.nn.Module):
    def __init__(self, cfg: JudgeBottleneckConfig) -> None:
        super().__init__()
        self.shared_encoder = T5EncoderModel.from_pretrained(cfg.judge_encoder_name)
        hidden_dim = self.shared_encoder.config.d_model
        prompt_hidden = cfg.prompt_mlp_hidden if cfg.prompt_mlp_hidden > 0 else 2 * hidden_dim
        self.prompt_mlp = PromptProjector(
            hidden_dim,
            cfg.z_latent_dim,
            prompt_len=cfg.z_prompt_len,
            hidden_dim=prompt_hidden,
            layers=cfg.prompt_mlp_layers,
            dropout=cfg.prompt_mlp_dropout,
        )

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim == 3:
            batch, pairs, seq = input_ids.shape
            flat_ids = input_ids.reshape(batch * pairs, seq)
            flat_mask = attention_mask.reshape(batch * pairs, seq)
            outputs = self.shared_encoder(input_ids=flat_ids, attention_mask=flat_mask)
            pooled = last_token_pool(outputs.last_hidden_state, flat_mask)
            z_tokens = self.prompt_mlp(pooled)
            z = z_tokens.mean(dim=1)
            return z.view(batch, pairs, -1)
        outputs = self.shared_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = last_token_pool(outputs.last_hidden_state, attention_mask)
        z_tokens = self.prompt_mlp(pooled)
        return z_tokens.mean(dim=1)


def _load_bottleneck(checkpoint_path: str, bundle: JudgeBottleneck) -> None:
    state = torch.load(checkpoint_path, map_location="cpu")
    state_dict = state.get("model", state)
    wanted = {}
    for key, value in state_dict.items():
        if key.startswith("shared_encoder."):
            wanted[key.replace("shared_encoder.", "shared_encoder.")] = value
        elif key.startswith("prompt_mlp."):
            wanted[key.replace("prompt_mlp.", "prompt_mlp.")] = value
    bundle.load_state_dict(wanted, strict=False)


def _build_prompt_embeds(
    bundle: JudgeBottleneck,
    lm_embed: torch.nn.Module,
    lm_tokenizer,
    prompt: str,
    response_a: str,
    response_b: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    enc = lm_tokenizer(
        prompt,
        add_special_tokens=True,
        return_tensors="pt",
        return_offsets_mapping=getattr(lm_tokenizer, "is_fast", False),
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    inputs_embeds = lm_embed(input_ids).to(dtype)
    return inputs_embeds.squeeze(0).to(dtype)


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM inference with DIBJudge bottleneck.")
    parser.add_argument("--model", required=True, help="Path or HF ID of the LM.")
    parser.add_argument("--checkpoint", required=True, help="DIBJudge checkpoint path.")
    parser.add_argument("--judge-encoder", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--prompt-field", default="judge_prompt")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--z-latent-dim", type=int, default=256)
    parser.add_argument("--z-prompt-len", type=int, default=16)
    parser.add_argument("--prompt-mlp-hidden", type=int, default=0)
    parser.add_argument("--prompt-mlp-layers", type=int, default=1)
    parser.add_argument("--prompt-mlp-dropout", type=float, default=0.1)
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    device = torch.device(args.device)

    lm_config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    lm_hidden = getattr(lm_config, "hidden_size", None) or getattr(lm_config, "n_embd")
    if lm_hidden is None:
        raise ValueError("Unable to determine LM hidden size.")

    bundle_cfg = JudgeBottleneckConfig(
        judge_encoder_name=args.judge_encoder,
        z_latent_dim=args.z_latent_dim,
        lm_hidden_size=lm_hidden,
        z_prompt_len=args.z_prompt_len,
        prompt_mlp_hidden=args.prompt_mlp_hidden,
        prompt_mlp_layers=args.prompt_mlp_layers,
        prompt_mlp_dropout=args.prompt_mlp_dropout,
    )
    bundle = JudgeBottleneck(bundle_cfg).to(device)
    bundle.eval()
    _load_bottleneck(args.checkpoint, bundle)

    lm_tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=True, trust_remote_code=args.trust_remote_code
    )
    if lm_tokenizer.pad_token_id is None:
        lm_tokenizer.pad_token = lm_tokenizer.eos_token

    # Load LM embeddings on CPU to avoid duplicating vLLM weights on GPU.
    from transformers import AutoModelForCausalLM

    lm_for_embed = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=args.trust_remote_code,
    )
    lm_embed = lm_for_embed.get_input_embeddings()
    lm_embed.eval()

    raw_examples = list(_read_jsonl(args.data_path))
    if args.limit is not None:
        raw_examples = raw_examples[: args.limit]

    prompts: List[EmbedsPrompt] = []
    for ex in raw_examples:
        prompt = _render_prompt(ex, args.prompt_field)
        response_a = str(ex.get("response_A", ""))
        response_b = str(ex.get("response_B", ""))
        prompt_embeds = _build_prompt_embeds(
            bundle,
            lm_embed,
            lm_tokenizer,
            prompt,
            response_a,
            response_b,
            device,
            dtype,
        )
        prompts.append(EmbedsPrompt(prompt_embeds=prompt_embeds))

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        seed=args.seed,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )

    with open(args.output_path, "w", encoding="utf-8") as handle:
        offset = 0
        for idx in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[idx : idx + args.batch_size]
            outputs = llm.generate(batch_prompts, sampling)
            for out in outputs:
                text = ""
                if out.outputs:
                    text = out.outputs[0].text
                record = {
                    "id": offset,
                    "prediction": text,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                offset += 1


if __name__ == "__main__":
    main()
