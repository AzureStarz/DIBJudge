from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Tuple
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dibjudge.finetune_deepspeed import _save_hf_from_zero
from dibjudge.modeling import DIBJudgeConfig


def _load_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("pyyaml is required to read YAML configs.") from exc
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("Config file must be a mapping of keys to values.")
    return data


def _resolve_checkpoint(checkpoint_dir: str, tag: str | None) -> Tuple[str, str]:
    if tag:
        return checkpoint_dir, tag
    latest_path = os.path.join(checkpoint_dir, "latest")
    if os.path.isfile(latest_path):
        with open(latest_path, "r", encoding="utf-8") as handle:
            latest_tag = handle.read().strip()
        if latest_tag:
            return checkpoint_dir, latest_tag
    model_state = os.path.join(checkpoint_dir, "mp_rank_00_model_states.pt")
    if os.path.isfile(model_state):
        return os.path.dirname(checkpoint_dir), os.path.basename(checkpoint_dir)
    raise ValueError("Unable to resolve checkpoint tag; provide --tag.")


def _build_parser(config: Dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a DeepSpeed ZeRO checkpoint to a unified HF-style DIBJudge checkpoint."
    )
    parser.add_argument("--config", help="Path to finetune YAML/JSON config.")
    parser.add_argument("--checkpoint-dir", required=True, help="DeepSpeed checkpoint root directory.")
    parser.add_argument("--tag", help="Checkpoint tag (defaults to latest).")
    parser.add_argument("--output-dir", help="Output directory for HF checkpoint.")
    parser.add_argument("--judge-encoder", dest="judge_encoder", default=DIBJudgeConfig.judge_encoder_name)
    parser.add_argument("--lm", default=DIBJudgeConfig.judge_lm_name)
    parser.add_argument("--z-latent-dim", type=int, default=DIBJudgeConfig.z_latent_dim)
    parser.add_argument("--z-prompt-len", type=int, default=DIBJudgeConfig.z_prompt_len)
    parser.add_argument("--z-prompt-prefix-len", type=int, default=DIBJudgeConfig.z_prompt_prefix_len)
    parser.add_argument("--z-prompt-postfix-len", type=int, default=DIBJudgeConfig.z_prompt_postfix_len)
    parser.add_argument("--prompt-mlp-hidden", type=int, default=DIBJudgeConfig.prompt_mlp_hidden)
    parser.add_argument("--prompt-mlp-layers", type=int, default=DIBJudgeConfig.prompt_mlp_layers)
    parser.add_argument("--prompt-mlp-dropout", type=float, default=DIBJudgeConfig.prompt_mlp_dropout)
    parser.add_argument("--grl-lambda", type=float, default=DIBJudgeConfig.grl_lambda)
    parser.add_argument("--bottleneck-noise-alpha", type=float, default=DIBJudgeConfig.bottleneck_noise_alpha)
    parser.add_argument("--bias-proxy-hidden", type=int, default=DIBJudgeConfig.bias_proxy_hidden)
    parser.add_argument("--bias-proxy-layers", type=int, default=DIBJudgeConfig.bias_proxy_layers)
    parser.add_argument("--bias-proxy-dropout", type=float, default=DIBJudgeConfig.bias_proxy_dropout)
    parser.add_argument("--low-recon-layer", type=int, default=DIBJudgeConfig.low_recon_layer)
    parser.add_argument("--compact-prior", type=float, default=DIBJudgeConfig.compact_prior)
    parser.add_argument("--compact-mu-token-id", type=int, default=DIBJudgeConfig.compact_mu_token_id)
    parser.add_argument("--compact-head-hidden", type=int, default=DIBJudgeConfig.compact_head_hidden)
    parser.add_argument("--compact-head-layers", type=int, default=DIBJudgeConfig.compact_head_layers)
    parser.add_argument("--compact-head-dropout", type=float, default=DIBJudgeConfig.compact_head_dropout)
    parser.add_argument("--use-lora", dest="use_lora", action="store_true")
    parser.add_argument("--no-use-lora", dest="use_lora", action="store_false")
    parser.set_defaults(use_lora=False)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-targets",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated list of LM module names for LoRA.",
    )
    if config:
        parser.set_defaults(**config)
    return parser


def main() -> None:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config")
    config_args, _ = config_parser.parse_known_args()
    config = _load_config(config_args.config)

    parser = _build_parser(config)
    args = parser.parse_args()

    checkpoint_dir, tag = _resolve_checkpoint(args.checkpoint_dir, args.tag)
    output_dir = args.output_dir or os.path.join(checkpoint_dir, f"hf-{tag}")
    _save_hf_from_zero(
        checkpoint_dir=checkpoint_dir,
        tag=tag,
        output_dir=output_dir,
        args=args,
        rank=0,
    )
    print(f"[done] saved HF checkpoints to {output_dir}/lm and {output_dir}/dibjudge")


if __name__ == "__main__":
    main()
