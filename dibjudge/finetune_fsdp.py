from __future__ import annotations

import argparse
import warnings
import os
import json
import math
from dataclasses import asdict
from typing import List, Optional, Tuple
import functools

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from .data import DIBJudgeCollator, DIBJudgeDataset
from .modeling import DIBJudgeConfig, DIBJudgeModel
from .proxy_tasks import ProxyTaskConfig
from .swanlab_utils import finish_swanlab, init_swanlab, log_swanlab
from .train import TrainConfig, create_optimizers, create_scheduler, train_one_epoch


def _get_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def init_distributed() -> Tuple[int, int, int]:
    if dist.is_initialized():
        local_rank = _get_env_int("LOCAL_RANK", _get_env_int("SLURM_LOCALID", 0))
        return dist.get_rank(), dist.get_world_size(), local_rank
    rank = _get_env_int("RANK", _get_env_int("SLURM_PROCID", 0))
    world_size = _get_env_int("WORLD_SIZE", _get_env_int("SLURM_NTASKS", 1))
    local_rank = _get_env_int("LOCAL_RANK", _get_env_int("SLURM_LOCALID", 0))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank


def _rank0_print(rank: int, *args: object, **kwargs: object) -> None:
    if rank == 0:
        print(*args, **kwargs)


def _maybe_resize_embeddings(
    model: torch.nn.Module, tokenizer, name: str, rank: int
) -> None:
    if model is None or tokenizer is None:
        return
    embed = model.get_input_embeddings()
    if embed is None:
        return
    vocab_size = int(embed.weight.size(0))
    tok_size = len(tokenizer)
    if tok_size > vocab_size:
        model.resize_token_embeddings(tok_size)
        _rank0_print(rank, f"[stage done] resized {name} embeddings {vocab_size} -> {tok_size}")


def _maybe_tqdm(iterable, rank: int, desc: str):
    if rank != 0:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    total = len(iterable) if hasattr(iterable, "__len__") else None
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)


def _load_yaml_config(path: str, parser: argparse.ArgumentParser) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("pyyaml is required for --config/--save-config") from exc
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a mapping of arg names to values.")
    valid = {action.dest for action in parser._actions}
    filtered = {key: value for key, value in data.items() if key in valid}
    unknown = sorted(set(data) - valid)
    if unknown:
        warnings.warn(f"Ignoring unknown config keys: {', '.join(unknown)}")
    return filtered


def _save_yaml_config(args: argparse.Namespace, path: str) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("pyyaml is required for --config/--save-config") from exc
    if not path:
        return
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(vars(args), handle, sort_keys=True)


def _parse_bins(text: str, cast=float) -> List:
    if text is None:
        return []
    if isinstance(text, (list, tuple)):
        return [cast(val) for val in text]
    parts = []
    for raw in str(text).split(","):
        raw = raw.strip().strip("[]")
        if not raw:
            continue
        parts.append(cast(raw))
    return parts


def _encode_ids(tokenizer, text: str, max_length: Optional[int]) -> List[int]:
    kwargs = {"add_special_tokens": False, "truncation": True}
    if max_length is not None:
        kwargs["max_length"] = max_length
    return tokenizer(text, **kwargs)["input_ids"]


def _has_response(text: Optional[str]) -> bool:
    return bool(text and text.strip())


def _compute_length_quantile_bins(
    dataset: DIBJudgeDataset,
    tokenizer,
    max_length: Optional[int],
    bins: int = 10,
) -> List[int]:
    lengths: List[int] = []
    for ex in dataset:
        lengths.append(len(_encode_ids(tokenizer, ex.response_a or "", max_length)))
        if _has_response(ex.response_b):
            lengths.append(len(_encode_ids(tokenizer, ex.response_b or "", max_length)))
    if not lengths or bins < 2:
        return list(ProxyTaskConfig().length_bins)
    lengths.sort()
    n = len(lengths)
    cuts: List[int] = []
    for idx in range(1, bins):
        q = idx / float(bins)
        pos = int(math.ceil(q * n)) - 1
        pos = min(max(pos, 0), n - 1)
        cuts.append(lengths[pos])
    max_len = lengths[-1]
    return [0] + cuts + [max_len]


def _collect_proxy_values(dataset: DIBJudgeDataset, attrs: Tuple[str, ...]) -> List[float]:
    values: List[float] = []
    for ex in dataset:
        for attr in attrs:
            val = getattr(ex, attr, None)
            if val is None:
                continue
            if isinstance(val, float) and math.isnan(val):
                continue
            values.append(float(val))
    return values


def _compute_value_quantile_bins(
    values: List[float],
    bins: int,
    fallback: Tuple[float, ...],
) -> List[float]:
    if not values or bins < 2:
        return list(fallback)
    values.sort()
    n = len(values)
    cuts: List[float] = []
    for idx in range(1, bins):
        q = idx / float(bins)
        pos = int(math.ceil(q * n)) - 1
        pos = min(max(pos, 0), n - 1)
        cuts.append(values[pos])
    return [values[0]] + cuts + [values[-1]]


def _resolve_wrap_classes(model: torch.nn.Module, names: List[str]) -> List[type]:
    if not names:
        return []
    target = set(names)
    classes: List[type] = []
    for module in model.modules():
        cls = module.__class__
        if cls.__name__ in target and cls not in classes:
            classes.append(cls)
    return classes


def _print_trainable_params(model: torch.nn.Module, rank: int) -> None:
    total = 0
    trainable = 0
    for param in model.parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
    pct = (100.0 * trainable / total) if total else 0.0
    _rank0_print(rank, f"[params] trainable={trainable:,} total={total:,} ({pct:.2f}%)")


def _set_lm_trainable(model: DIBJudgeModel, trainable: bool, rank: int) -> None:
    for param in model.judge_lm.parameters():
        param.requires_grad = trainable
    state = "trainable" if trainable else "frozen"
    _rank0_print(rank, f"[stage done] judge_lm {state}")


def _set_shared_encoder_trainable(model: DIBJudgeModel, mode: str, rank: int) -> bool:
    mode = (mode or "all").lower()
    if mode == "all":
        for param in model.shared_encoder.parameters():
            param.requires_grad = True
        _rank0_print(rank, "[stage done] shared encoder fully trainable")
        return False
    if mode == "none":
        for param in model.shared_encoder.parameters():
            param.requires_grad = False
        _rank0_print(rank, "[stage done] shared encoder frozen")
        return True
    if mode != "last":
        _rank0_print(rank, f"[warn] unknown encoder_trainable={mode}, using all")
        for param in model.shared_encoder.parameters():
            param.requires_grad = True
        return False
    for param in model.shared_encoder.parameters():
        param.requires_grad = False
    last_layer = None
    enc = model.shared_encoder
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
            last_layer = mod[-1]
            break
    if last_layer is None:
        _rank0_print(rank, "[warn] unable to locate encoder layers; shared encoder fully trainable")
        for param in model.shared_encoder.parameters():
            param.requires_grad = True
        return False
    for param in last_layer.parameters():
        param.requires_grad = True
    last_ids = {id(param) for param in last_layer.parameters()}
    trainable = [param for param in model.shared_encoder.parameters() if param.requires_grad]
    only_last = bool(trainable) and all(id(param) in last_ids for param in trainable)
    _rank0_print(rank, "[stage done] shared encoder trainable=last")
    return only_last


def _setup_fsdp(
    model: DIBJudgeModel,
    wrap_cls_names: List[str],
    use_bf16: bool,
) -> torch.nn.Module:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    wrap_classes = _resolve_wrap_classes(model, wrap_cls_names)
    auto_wrap = None
    if wrap_classes:
        auto_wrap = functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls=set(wrap_classes)
        )
    mp = None
    if use_bf16:
        mp = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    return FSDP(
        model,
        auto_wrap_policy=auto_wrap,
        mixed_precision=mp,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=True,
        device_id=torch.cuda.current_device(),
    )


def _maybe_apply_lora(model: DIBJudgeModel, args: argparse.Namespace) -> None:
    if not args.use_lora:
        return
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError("peft is required for --use-lora") from exc

    target = [name.strip() for name in args.lora_targets.split(",") if name.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target,
    )
    model.judge_lm = get_peft_model(model.judge_lm, lora_cfg)

def _apply_fsdp_activation_checkpointing(module: torch.nn.Module) -> None:
    try:
        from torch.distributed.fsdp.wrap import (
            apply_activation_checkpointing,
            checkpoint_wrapper,
            CheckpointImpl,
        )
    except ImportError:
        return
    check_fn = lambda sub: sub.__class__.__name__ in {"T5Block", "MT5Block"}
    wrapper = functools.partial(
        checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
    )
    apply_activation_checkpointing(module, checkpoint_wrapper_fn=wrapper, check_fn=check_fn)


def _save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    output_dir: str,
    epoch: int,
    rank: int,
) -> None:
    if rank != 0:
        return
    os.makedirs(output_dir, exist_ok=True)
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_cfg):
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
    path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save(state, path)


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    pre_args, _ = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(description="FSDP LoRA finetuning for DIBJudge.")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--judge-encoder", default=None)
    parser.add_argument("--lm", default=None)
    parser.add_argument("--z-latent-dim", type=int, default=256)
    parser.add_argument("--z-prompt-len", type=int, default=16)
    parser.add_argument("--prompt-mlp-hidden", type=int, default=0)
    parser.add_argument("--prompt-mlp-layers", type=int, default=1)
    parser.add_argument("--prompt-mlp-dropout", type=float, default=0.1)
    parser.add_argument("--bottleneck-noise-alpha", type=float, default=8.0)
    parser.add_argument("--bottleneck-noise-warmup-ratio", type=float, default=0.2)
    parser.add_argument("--z-prompt-prefix-len", type=int, default=1)
    parser.add_argument("--z-prompt-postfix-len", type=int, default=1)
    parser.add_argument(
        "--encoder-trainable",
        default="all",
        choices=["all", "last", "none"],
        help="Shared encoder trainable params: all, last, or none.",
    )
    parser.add_argument(
        "--freeze-lm-when-no-lora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze LM parameters when LoRA is disabled.",
    )
    parser.add_argument("--bias-proxy-hidden", type=int, default=0)
    parser.add_argument("--bias-proxy-layers", type=int, default=-1)
    parser.add_argument("--bias-proxy-dropout", type=float, default=-1.0)
    parser.add_argument("--low-recon-layer", type=int, default=2)
    parser.add_argument("--compact-prior", type=float, default=0.3)
    parser.add_argument("--compact-mu-token-id", type=int, default=-1)
    parser.add_argument("--compact-head-hidden", type=int, default=0)
    parser.add_argument("--compact-head-layers", type=int, default=1)
    parser.add_argument("--compact-head-dropout", type=float, default=0.1)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-bias-len", type=int, default=1024)
    parser.add_argument("--max-ref-len", type=int, default=1024)
    parser.add_argument("--max-lm-len", type=int, default=4096)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-targets",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated module names for LoRA injection.",
    )
    parser.add_argument(
        "--fsdp-wrap-cls",
        default="",
        help="Comma-separated transformer layer class names for auto-wrapping.",
    )
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--lm-lr", type=float, default=2e-5)
    parser.add_argument("--lora-lr", type=float, default=2e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--head-weight-decay", type=float, default=0.001)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--total-steps", type=int, default=10000)
    parser.add_argument("--scheduler-type", default="cosine")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lambda-compression", type=float, default=1.0)
    parser.add_argument("--lambda-compression-warmup-ratio", type=float, default=0.05)
    parser.add_argument("--mask-loss-weight", type=float, default=1.0)
    parser.add_argument("--consistency-loss-weight", type=float, default=1.0)
    parser.add_argument("--lambda-bias", type=float, default=1.0)
    parser.add_argument("--grl-lambda", type=float, default=1.0)
    parser.add_argument("--grl-start-ratio", type=float, default=0.3)
    parser.add_argument("--grl-gamma", type=float, default=10.0)
    parser.add_argument("--bias-decoder-steps", type=int, default=1)
    parser.add_argument("--eng-domain-weight", type=float, default=1.0)
    parser.add_argument("--low-recon-weight", type=float, default=0.5)
    parser.add_argument("--z-l2-weight", type=float, default=0.1)
    parser.add_argument("--nll-bin-weight", type=float, default=0.5)
    parser.add_argument(
        "--ppl-bin-weight",
        dest="nll_bin_weight",
        type=float,
        default=0.5,
        help="Deprecated alias for --nll-bin-weight.",
    )
    parser.add_argument("--ttr-bin-weight", type=float, default=0.5)
    parser.add_argument("--length-bin-weight", type=float, default=0.5)
    parser.add_argument(
        "--proxy-nll-bins",
        dest="proxy_nll_bins",
        default="0,2.3026,2.9957,3.6889,4.3820,5.0752,13.8155",
        help="Comma-separated bins for NLL (log PPL) ordinal regression.",
    )
    parser.add_argument(
        "--proxy-ppl-bins",
        dest="proxy_nll_bins",
        default="0,2.3026,2.9957,3.6889,4.3820,5.0752,13.8155",
        help="Deprecated alias for --proxy-nll-bins.",
    )
    parser.add_argument(
        "--proxy-nll-quantiles",
        dest="proxy_nll_quantiles",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use training-set NLL quantiles instead of fixed bins.",
    )
    parser.add_argument(
        "--proxy-ppl-quantiles",
        dest="proxy_nll_quantiles",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deprecated alias for --proxy-nll-quantiles.",
    )
    parser.add_argument(
        "--proxy-ttr-bins",
        default="0,0.2,0.4,0.6,0.8,1.0",
        help="Comma-separated bins for TTR ordinal regression.",
    )
    parser.add_argument(
        "--proxy-ttr-quantiles",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use training-set TTR quantiles instead of fixed bins.",
    )
    parser.add_argument(
        "--proxy-length-bins",
        default="0,50,100,200,400,1000000",
        help="Comma-separated bins for length ordinal regression.",
    )
    parser.add_argument(
        "--proxy-length-quantiles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use decile length quantiles from the training set instead of fixed bins.",
    )
    parser.add_argument(
        "--proxy-soft-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use soft interpolation between ordinal bins.",
    )
    parser.add_argument("--hard-neg-k", type=int, default=32)
    parser.add_argument("--hard-neg-gamma", type=float, default=1.2)

    parser.add_argument("--debug-data", action="store_true")
    parser.add_argument(
        "--debug-aux-checks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable auxiliary-loss diagnostics (grad norms, coverage).",
    )
    parser.add_argument("--debug-aux-checks-interval", type=int, default=200)
    parser.add_argument("--use-swanlab", action="store_true")
    parser.add_argument("--swanlab-project", default="dibjudge")
    parser.add_argument("--swanlab-run-name", default=None)
    parser.add_argument("--swanlab-tags", default=None)
    parser.add_argument("--swanlab-log-steps", type=int, default=10)
    parser.add_argument("--config", default=None, help="YAML config file to load defaults from.")
    parser.add_argument(
        "--save-config",
        default=None,
        help="Write resolved args to this YAML path (rank0 only).",
    )
    parser.add_argument(
        "--no-save-config",
        action="store_true",
        help="Disable saving the resolved config YAML.",
    )

    if pre_args.config:
        parser.set_defaults(**_load_yaml_config(pre_args.config, parser))
    args = parser.parse_args()
    required = ["data_path", "judge_encoder", "lm"]
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        parser.error(f"Missing required arguments (or YAML keys): {', '.join(missing)}")

    rank, world_size, local_rank = init_distributed()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if rank == 0:
        warnings.filterwarnings("default")
    else:
        warnings.filterwarnings("ignore")

    if not args.no_save_config and args.save_config is None and args.config is None:
        args.save_config = os.path.join("configs", "finetune_fsdp.yaml")
    if args.save_config and rank == 0:
        _save_yaml_config(args, args.save_config)
    if args.log_dir and rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)

    enc_tok_kwargs = {"use_fast": False, "legacy": True}
    lm_tok_kwargs = {"use_fast": True}
    judge_tok = AutoTokenizer.from_pretrained(args.judge_encoder, **enc_tok_kwargs)
    lm_tok = AutoTokenizer.from_pretrained(args.lm, **lm_tok_kwargs)
    if judge_tok.pad_token_id is None:
        judge_tok.pad_token = judge_tok.eos_token
    if lm_tok.pad_token_id is None:
        lm_tok.pad_token = lm_tok.eos_token
    mu_id = lm_tok.convert_tokens_to_ids(".")
    if mu_id is None or mu_id == lm_tok.unk_token_id:
        mu_id = lm_tok.eos_token_id
    if mu_id is None:
        mu_id = lm_tok.pad_token_id or 0
    args.compact_mu_token_id = int(mu_id)

    length_bins = _parse_bins(args.proxy_length_bins, float)
    if len(length_bins) < 2:
        length_bins = list(ProxyTaskConfig().length_bins)
    nll_bins = _parse_bins(args.proxy_nll_bins, float)
    if len(nll_bins) < 2:
        nll_bins = list(ProxyTaskConfig().nll_bins)
    ttr_bins = _parse_bins(args.proxy_ttr_bins, float)
    if len(ttr_bins) < 2:
        ttr_bins = list(ProxyTaskConfig().ttr_bins)

    dataset = DIBJudgeDataset.from_jsonl(args.data_path)
    _rank0_print(rank, "[stage done] dataset loaded")
    if args.proxy_length_quantiles:
        length_bins = _compute_length_quantile_bins(
            dataset, lm_tok, args.max_bias_len, bins=10
        )
    if args.proxy_nll_quantiles:
        nll_bins = _compute_value_quantile_bins(
            _collect_proxy_values(dataset, ("proxy_nll_a", "proxy_nll_b")),
            bins=10,
            fallback=tuple(nll_bins),
        )
    if args.proxy_ttr_quantiles:
        ttr_bins = _compute_value_quantile_bins(
            _collect_proxy_values(dataset, ("proxy_ttr_a", "proxy_ttr_b")),
            bins=10,
            fallback=tuple(ttr_bins),
        )
    proxy_config = ProxyTaskConfig(
        nll_bins=tuple(nll_bins),
        ttr_bins=tuple(ttr_bins),
        length_bins=tuple(length_bins),
        use_soft_labels=bool(args.proxy_soft_labels),
    )
    args.proxy_nll_classes = max(2, len(nll_bins) - 1)
    args.proxy_ttr_classes = max(2, len(ttr_bins) - 1)
    args.proxy_length_classes = max(2, len(length_bins) - 1)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    collator = DIBJudgeCollator(
        lm_tok,
        max_bias_len=args.max_bias_len,
        max_ref_len=args.max_ref_len,
        max_lm_len=args.max_lm_len,
        proxy_config=proxy_config,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )
    _rank0_print(rank, "[stage done] dataloader ready")
    if args.debug_data and rank == 0:
        batch = next(iter(loader))
        shapes = {k: tuple(v.shape) for k, v in batch.items() if torch.is_tensor(v)}
        _rank0_print(rank, "Debug batch shapes:", shapes)
        for key in ("original_attention_mask", "lm_attention_mask"):
            if key in batch:
                mask = batch[key]
                lengths = mask.sum(dim=-1).view(-1).tolist()
                _rank0_print(rank, f"{key} lengths (first 8):", lengths[:8])

    model = DIBJudgeModel.init_from_backbones(
        judge_encoder_name=args.judge_encoder,
        judge_lm_name=args.lm,
        z_latent_dim=args.z_latent_dim,
        z_prompt_len=args.z_prompt_len,
        prompt_mlp_hidden=args.prompt_mlp_hidden,
        prompt_mlp_layers=args.prompt_mlp_layers,
        prompt_mlp_dropout=args.prompt_mlp_dropout,
        bottleneck_noise_alpha=args.bottleneck_noise_alpha,
        grl_lambda=args.grl_lambda,
        z_prompt_prefix_len=args.z_prompt_prefix_len,
        z_prompt_postfix_len=args.z_prompt_postfix_len,
        low_recon_layer=args.low_recon_layer,
        bias_proxy_hidden=args.bias_proxy_hidden,
        bias_proxy_layers=args.bias_proxy_layers,
        bias_proxy_dropout=args.bias_proxy_dropout,
        compact_prior=args.compact_prior,
        compact_mu_token_id=args.compact_mu_token_id,
        compact_head_hidden=args.compact_head_hidden,
        compact_head_layers=args.compact_head_layers,
        compact_head_dropout=args.compact_head_dropout,
        proxy_nll_classes=max(2, len(nll_bins) - 1),
        proxy_ttr_classes=max(2, len(ttr_bins) - 1),
        proxy_length_classes=max(2, len(length_bins) - 1),
    ).to(device)
    _maybe_resize_embeddings(model.shared_encoder, judge_tok, "shared_encoder", rank)
    _maybe_resize_embeddings(model.judge_lm, lm_tok, "judge_lm", rank)
    skip_encoder_checkpointing = _set_shared_encoder_trainable(
        model, args.encoder_trainable, rank
    )
    _rank0_print(rank, "[stage done] model initialized")
    if args.gradient_checkpointing:
        if world_size > 1:
            if not skip_encoder_checkpointing:
                _apply_fsdp_activation_checkpointing(model.shared_encoder)
        else:
            gc_kwargs = {"use_reentrant": False}
            if not skip_encoder_checkpointing:
                model.shared_encoder.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=gc_kwargs
                )
            model.judge_lm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gc_kwargs
            )
    _maybe_apply_lora(model, args)
    if args.freeze_lm_when_no_lora and not args.use_lora:
        _set_lm_trainable(model, False, rank)
    _rank0_print(rank, "[stage done] lora applied" if args.use_lora else "[stage done] lora skipped")
    wrap_names = [name.strip() for name in args.fsdp_wrap_cls.split(",") if name.strip()]
    model = _setup_fsdp(model, wrap_names, use_bf16=args.bf16)
    _rank0_print(rank, "[stage done] fsdp wrapped")
    _print_trainable_params(model, rank)

    train_cfg = TrainConfig(
        lr=args.lr,
        encoder_lr=args.encoder_lr,
        lm_lr=args.lm_lr,
        lora_lr=args.lora_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        head_weight_decay=args.head_weight_decay,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        total_steps=args.total_steps,
        scheduler_type=args.scheduler_type,
        grad_accum_steps=args.grad_accum_steps,
        max_grad_norm=args.max_grad_norm,
        use_amp=True,
        amp_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        lambda_bias=args.lambda_bias,
        grl_lambda=args.grl_lambda,
        grl_start_ratio=args.grl_start_ratio,
        grl_gamma=args.grl_gamma,
        bias_decoder_steps=args.bias_decoder_steps,
        bottleneck_noise_alpha=args.bottleneck_noise_alpha,
        bottleneck_noise_warmup_ratio=args.bottleneck_noise_warmup_ratio,
        lambda_compression=args.lambda_compression,
        lambda_compression_warmup_ratio=args.lambda_compression_warmup_ratio,
        mask_loss_weight=args.mask_loss_weight,
        consistency_loss_weight=args.consistency_loss_weight,
        eng_domain_weight=args.eng_domain_weight,
        low_recon_weight=args.low_recon_weight,
        z_l2_weight=args.z_l2_weight,
        nll_bin_weight=args.nll_bin_weight,
        ttr_bin_weight=args.ttr_bin_weight,
        length_bin_weight=args.length_bin_weight,
        debug_aux_checks=args.debug_aux_checks,
        debug_aux_checks_interval=args.debug_aux_checks_interval,
    )

    optimizer = create_optimizers(model, train_cfg)
    scheduler = create_scheduler(optimizer, train_cfg)
    _rank0_print(rank, "[stage done] optimizers and scheduler ready")

    swanlab_client = None
    if rank == 0 and args.use_swanlab:
        tags = None
        if args.swanlab_tags:
            tags = [tag.strip() for tag in args.swanlab_tags.split(",") if tag.strip()]
        swanlab_client = init_swanlab(
            True,
            project=args.swanlab_project,
            run_name=args.swanlab_run_name,
            config={"args": vars(args), "train": asdict(train_cfg)},
            log_dir=args.log_dir,
            tags=tags,
        )
        _rank0_print(rank, "[stage done] swanlab initialized")

    _rank0_print(rank, "TrainConfig:", asdict(train_cfg))

    step = 0
    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch_loader = _maybe_tqdm(loader, rank, f"epoch {epoch}")
        metrics = train_one_epoch(
            model,
            epoch_loader,
            optimizer,
            scheduler,
            train_cfg,
            device,
            start_step=step,
            log_fn=(
                (lambda m, s: log_swanlab(swanlab_client, m, step=s))
                if swanlab_client is not None
                else None
            ),
            log_interval=args.swanlab_log_steps if swanlab_client is not None else 0,
        )
        if hasattr(epoch_loader, "close"):
            epoch_loader.close()
        step += metrics.get("steps", 0)
        _rank0_print(rank, f"epoch={epoch} metrics={metrics}")
        if swanlab_client is not None and rank == 0:
            epoch_metrics = {f"epoch/{k}": v for k, v in metrics.items() if k != "steps"}
            epoch_metrics["epoch"] = epoch
            log_swanlab(swanlab_client, epoch_metrics, step=step)
        if args.output_dir:
            _save_checkpoint(model, optimizer, args.output_dir, epoch, rank)
    if swanlab_client is not None and rank == 0:
        finish_swanlab(swanlab_client)


if __name__ == "__main__":
    main()
