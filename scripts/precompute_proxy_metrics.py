#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _is_null(value: Optional[object]) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped.lower() in {"none", "null"}:
            return True
    return False


def _resolve_output_path(input_path: str, output_path: Optional[str]) -> str:
    if output_path:
        return output_path
    base = Path(input_path)
    return str(base.with_suffix(".proxy.jsonl"))


def _iter_jsonl(path: str) -> Iterable[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)


def _iter_jsonl_range(path: str, start: int, end: Optional[int]) -> Iterable[Dict[str, object]]:
    idx = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if idx < start:
                idx += 1
                continue
            if end is not None and idx >= end:
                break
            yield json.loads(line)
            idx += 1


def _count_lines(path: str) -> int:
    count = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _maybe_tqdm(iterable, total: Optional[int], desc: str):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit="ex", dynamic_ncols=True)


def _compute_nll(
    model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    if logits.size(1) < 2:
        return logits.new_zeros((logits.size(0),))
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()
    vocab = shift_logits.size(-1)
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, vocab),
        shift_labels.view(-1),
        reduction="none",
    ).view(shift_labels.size())
    loss = loss * shift_mask
    denom = shift_mask.sum(dim=1).clamp_min(1)
    return loss.sum(dim=1) / denom


def _token_metrics(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[Dict[str, float]]:
    metrics: List[Dict[str, float]] = []
    lengths = attention_mask.sum(dim=1).tolist()
    for row, length in enumerate(lengths):
        length = int(length)
        if length <= 0:
            metrics.append({"length": 0.0, "ttr": 0.0})
            continue
        ids = input_ids[row, :length].tolist()
        metrics.append({"length": float(length), "ttr": float(len(set(ids)) / float(length))})
    return metrics


def _prepare_tasks(
    records: List[Dict[str, object]],
    overwrite: bool,
) -> List[Dict[str, object]]:
    tasks: List[Dict[str, object]] = []
    for idx, record in enumerate(records):
        for suffix in ("A", "B"):
            response = record.get(f"response_{suffix}")
            if response is None or not str(response).strip():
                continue
            nll_key = f"proxy_nll_{suffix}"
            ppl_key = f"proxy_ppl_{suffix}"
            ttr_key = f"proxy_ttr_{suffix}"
            len_key = f"proxy_length_{suffix}"
            existing_nll = record.get(nll_key)
            if _is_null(existing_nll):
                ppl_val = record.get(ppl_key)
                if not _is_null(ppl_val):
                    try:
                        ppl_float = float(ppl_val)
                    except (TypeError, ValueError):
                        ppl_float = None
                    if ppl_float is not None and ppl_float > 0:
                        record[nll_key] = math.log(ppl_float)
                        existing_nll = record[nll_key]

            need_nll = overwrite or _is_null(existing_nll)
            need_ttr = overwrite or _is_null(record.get(ttr_key))
            need_len = overwrite or _is_null(record.get(len_key))
            if not (need_nll or need_ttr or need_len):
                continue
            tasks.append(
                {
                    "record_idx": idx,
                    "suffix": suffix,
                    "text": str(response),
                    "need_nll": need_nll,
                    "need_ttr": need_ttr,
                    "need_len": need_len,
                }
            )
    return tasks


def _write_proxy_line(handle, record: Dict[str, object]) -> None:
    payload: Dict[str, object] = {}
    for key in (
        "proxy_nll_A",
        "proxy_nll_B",
        "proxy_ttr_A",
        "proxy_ttr_B",
        "proxy_length_A",
        "proxy_length_B",
    ):
        if key in record and not _is_null(record.get(key)):
            payload[key] = record[key]
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute proxy NLL/TTR/length metrics.")
    parser.add_argument("--input", required=True, help="Input JSONL dataset path.")
    parser.add_argument("--output", default=None, help="Output proxy cache JSONL path.")
    parser.add_argument("--lm", required=True, help="LM name or path for NLL computation.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default=None, help="Device string (e.g., cuda, cuda:0, cpu).")
    parser.add_argument(
        "--dtype",
        default=None,
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype for the LM.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    output_path = _resolve_output_path(args.input, args.output)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1.")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must be within [0, num-shards).")
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = None
    if args.dtype == "float16":
        torch_dtype = torch.float16
    elif args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "float32":
        torch_dtype = torch.float32
    if device.startswith("cpu") and torch_dtype in {torch.float16, torch.bfloat16}:
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.lm, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.lm,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    model.to(device)

    with open(output_path, "w", encoding="utf-8") as out_handle:
        start_idx = 0
        end_idx = None
        total_lines = None
        if args.num_shards > 1:
            total = _count_lines(args.input)
            shard_size = total // args.num_shards
            remainder = total % args.num_shards
            start_idx = args.shard_index * shard_size + min(args.shard_index, remainder)
            length = shard_size + (1 if args.shard_index < remainder else 0)
            end_idx = start_idx + length
            total_lines = length
            print(
                f"[shard] index={args.shard_index} num={args.num_shards} range={start_idx}:{end_idx}",
                flush=True,
            )
        else:
            total_lines = _count_lines(args.input)
        if args.max_samples and total_lines is not None:
            total_lines = min(total_lines, args.max_samples)
        chunk: List[Dict[str, object]] = []
        processed = 0
        iterator = _iter_jsonl_range(args.input, start_idx, end_idx)
        iterator = _maybe_tqdm(iterator, total_lines, "precompute")
        for record in iterator:
            chunk.append(record)
            if args.max_samples and processed + len(chunk) >= args.max_samples:
                chunk = chunk[: max(0, args.max_samples - processed)]
            if len(chunk) >= args.chunk_size or (args.max_samples and processed + len(chunk) >= args.max_samples):
                processed += _process_chunk(
                    chunk,
                    out_handle,
                    tokenizer,
                    model,
                    device,
                    args.max_length if args.max_length > 0 else None,
                    args.batch_size,
                    args.overwrite,
                )
                chunk = []
            if args.max_samples and processed >= args.max_samples:
                break
        if chunk and (not args.max_samples or processed < args.max_samples):
            _process_chunk(
                chunk,
                out_handle,
                tokenizer,
                model,
                device,
                args.max_length if args.max_length > 0 else None,
                args.batch_size,
                args.overwrite,
            )


def _process_chunk(
    records: List[Dict[str, object]],
    out_handle,
    tokenizer,
    model,
    device: str,
    max_length: Optional[int],
    batch_size: int,
    overwrite: bool,
) -> int:
    tasks = _prepare_tasks(records, overwrite)
    for start in range(0, len(tasks), batch_size):
        batch_tasks = tasks[start : start + batch_size]
        texts = [task["text"] for task in batch_tasks]
        enc = tokenizer(
            texts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        metrics = _token_metrics(input_ids, attention_mask)

        need_nll = any(task["need_nll"] for task in batch_tasks)
        nll_values: List[float] = [0.0] * len(batch_tasks)
        if need_nll:
            with torch.no_grad():
                nll = _compute_nll(model, input_ids, attention_mask).detach().float().cpu()
            nll_values = nll.tolist()

        for idx, task in enumerate(batch_tasks):
            record = records[task["record_idx"]]
            suffix = task["suffix"]
            if task["need_len"]:
                record[f"proxy_length_{suffix}"] = metrics[idx]["length"]
            if task["need_ttr"]:
                record[f"proxy_ttr_{suffix}"] = metrics[idx]["ttr"]
            if task["need_nll"]:
                record[f"proxy_nll_{suffix}"] = float(nll_values[idx])

    for record in records:
        _write_proxy_line(out_handle, record)
    return len(records)


if __name__ == "__main__":
    main()
