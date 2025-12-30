from __future__ import annotations

import argparse
from typing import List

import torch
from transformers import AutoTokenizer

from .data import DIBJudgeCollator, DIBJudgeExample
from .modeling import DIBJudgeConfig, DIBJudgeModel


def _build_examples() -> List[DIBJudgeExample]:
    return [
        DIBJudgeExample(
            instruction="What is 2+2?",
            response_a="The answer is 4.",
            response_b="It is 4.",
            response_a_bt="The answer is 4.",
            response_b_bt="It is 4.",
            judge_prompt="Judge:\n{instruction}\n\nA:\n{response_A}\n\nB:\n{response_B}\n\nFeedback:",
            output="Both are correct. [RESULT] A",
        ),
        DIBJudgeExample(
            instruction="Capital of France?",
            response_a="Paris.",
            response_b="The capital is Paris.",
            response_a_bt="Paris.",
            response_b_bt="The capital is Paris.",
            judge_prompt="Judge:\n{instruction}\n\nA:\n{response_A}\n\nB:\n{response_B}\n\nFeedback:",
            output="Both are correct. [RESULT] B",
        ),
    ]


def run_smoke(args: argparse.Namespace) -> None:
    judge_tokenizer = AutoTokenizer.from_pretrained(args.judge_encoder, use_fast=True)
    lm_tokenizer = AutoTokenizer.from_pretrained(args.lm, use_fast=True)

    collator = DIBJudgeCollator(lm_tokenizer, max_lm_len=64)
    batch = collator(_build_examples())

    expected = [
        "original_input_ids",
        "original_attention_mask",
        "shuffle_input_ids",
        "shuffle_attention_mask",
        "original_shuffle_labels",
        "shuffle_labels",
        "english_input_ids",
        "english_attention_mask",
        "lm_input_ids",
        "lm_attention_mask",
        "lm_labels",
    ]
    for key in expected:
        if key not in batch:
            raise KeyError(f"Missing batch key: {key}")

    if args.with_model:
        device = torch.device(args.device)
        cfg = DIBJudgeConfig(
            judge_encoder_name=args.judge_encoder,
            judge_lm_name=args.lm,
        )
        model = DIBJudgeModel(cfg).to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(batch)
        if outputs["lm_loss"].ndim != 0:
            raise RuntimeError("Expected scalar LM loss.")


def main() -> None:
    parser = argparse.ArgumentParser(description="DIBJudge smoke test.")
    parser.add_argument("--judge-encoder", default="google/mt5-base")
    parser.add_argument("--lm", default="gpt2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--with-model", action="store_true")
    args = parser.parse_args()
    run_smoke(args)


if __name__ == "__main__":
    main()
