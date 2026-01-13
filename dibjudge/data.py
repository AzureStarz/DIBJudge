from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from torch.utils.data import Dataset

from .proxy_tasks import ProxyTaskConfig


@dataclass
class DIBJudgeExample:
    instruction: str
    response_a: str
    response_b: Optional[str]
    judge_prompt: str
    output: str
    proxy_length_a: Optional[float] = None
    proxy_length_b: Optional[float] = None
    proxy_nll_a: Optional[float] = None
    proxy_nll_b: Optional[float] = None
    proxy_ttr_a: Optional[float] = None
    proxy_ttr_b: Optional[float] = None

    @classmethod
    def from_dict(cls, raw: Dict[str, object]) -> "DIBJudgeExample":
        response_a = str(raw.get("response_A", ""))
        response_b = _coerce_optional_text(raw.get("response_B"))
        judge_prompt = _coerce_optional_text(
            raw.get("judge_prompt", raw.get("judge_instruction"))
        )
        instruction = _coerce_optional_text(
            raw.get("instruction", raw.get("prompt"))
        )
        return cls(
            instruction=instruction,
            response_a=response_a,
            response_b=response_b,
            judge_prompt=judge_prompt,
            output=str(raw.get("output", "")),
            proxy_length_a=_coerce_optional_float(raw.get("proxy_length_A")),
            proxy_length_b=_coerce_optional_float(raw.get("proxy_length_B")),
            proxy_nll_a=_coerce_optional_nll(raw.get("proxy_nll_A"), raw.get("proxy_ppl_A")),
            proxy_nll_b=_coerce_optional_nll(raw.get("proxy_nll_B"), raw.get("proxy_ppl_B")),
            proxy_ttr_a=_coerce_optional_float(raw.get("proxy_ttr_A")),
            proxy_ttr_b=_coerce_optional_float(raw.get("proxy_ttr_B")),
        )


class DIBJudgeDataset(Dataset):
    def __init__(self, samples: Sequence[DIBJudgeExample]) -> None:
        self.samples = list(samples)

    @classmethod
    def from_jsonl(
        cls, path: str, proxy_cache_path: Optional[str] = None
    ) -> "DIBJudgeDataset":
        samples: List[DIBJudgeExample] = []
        filtered_empty = 0
        cache_path = proxy_cache_path or _infer_proxy_cache_path(path)
        if proxy_cache_path:
            cache_path = proxy_cache_path
            if not Path(cache_path).exists():
                raise FileNotFoundError(f"Proxy cache not found: {cache_path}")
        cache_iter = _iter_jsonl(cache_path) if cache_path else None
        cache_exhausted = False
        missing_proxy = False
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                cache_raw = None
                if cache_iter is not None and not cache_exhausted:
                    try:
                        cache_raw = next(cache_iter)
                    except StopIteration:
                        cache_exhausted = True
                        cache_raw = None
                        warnings.warn(
                            "Proxy cache ended early; remaining samples will use in-file fields.",
                            RuntimeWarning,
                        )
                if cache_raw is not None:
                    _merge_proxy_cache(raw, cache_raw)
                if not _has_response(_coerce_optional_text(raw.get("response_A"))):
                    filtered_empty += 1
                    continue
                if _missing_proxy_fields(raw):
                    missing_proxy = True
                samples.append(DIBJudgeExample.from_dict(raw))
        if cache_iter is not None and not cache_exhausted:
            try:
                next(cache_iter)
            except StopIteration:
                pass
            else:
                warnings.warn(
                    "Proxy cache has extra entries; ensure it matches the dataset order.",
                    RuntimeWarning,
                )
        if filtered_empty:
            warnings.warn(
                f"Filtered {filtered_empty} samples with empty response_A.",
                RuntimeWarning,
            )
        if missing_proxy:
            if cache_path is None:
                warnings.warn(
                    "Proxy fields missing in dataset and no proxy cache found. "
                    "Provide a proxy cache path or include proxy fields in the dataset.",
                    RuntimeWarning,
                )
            else:
                warnings.warn(
                    "Proxy fields missing after applying proxy cache; "
                    "verify the cache file matches the dataset order.",
                    RuntimeWarning,
                )
        return cls(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> DIBJudgeExample:
        return self.samples[idx]


def _encode_texts(tokenizer, texts: List[str], max_length: Optional[int], add_special_tokens: bool = True):
    kwargs = {
        "padding": True,
        "truncation": True,
        "return_tensors": "pt",
        "add_special_tokens": add_special_tokens,
    }
    if max_length is not None:
        kwargs["max_length"] = max_length
    enc = tokenizer(texts, **kwargs)
    input_ids = enc["input_ids"]
    if input_ids.size(1) == 0:
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id or 0
        enc["input_ids"] = torch.full(
            (input_ids.size(0), 1), pad_id, dtype=input_ids.dtype
        )
        enc["attention_mask"] = torch.zeros((input_ids.size(0), 1), dtype=input_ids.dtype)
    return enc

def _encode_ids(tokenizer, text: str, max_length: Optional[int]) -> List[int]:
    kwargs = {"add_special_tokens": False, "truncation": True}
    if max_length is not None:
        kwargs["max_length"] = max_length
    return tokenizer(text, **kwargs)["input_ids"]


def _find_subseq(haystack: Sequence[int], needle: Sequence[int]) -> Optional[Tuple[int, int]]:
    if not needle:
        return None
    max_start = len(haystack) - len(needle)
    if max_start < 0:
        return None
    for idx in range(max_start + 1):
        if haystack[idx : idx + len(needle)] == list(needle):
            return idx, idx + len(needle)
    return None


def _coerce_text(value: Optional[object], fallback: str) -> str:
    if value is None:
        return fallback
    text = str(value)
    return text if text else fallback


def _coerce_optional_text(value: Optional[object]) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _has_response(text: Optional[str]) -> bool:
    return bool(text and text.strip())


def _bucketize_int(value: int, bins: Sequence[int]) -> int:
    if len(bins) < 2:
        return 0
    for idx in range(len(bins) - 1):
        if bins[idx] <= value <= bins[idx + 1]:
            return idx
    if value < bins[0]:
        return 0
    return len(bins) - 2


def _soft_bin_target(
    value: Optional[float],
    bins: Sequence[float],
    use_soft: bool,
) -> Tuple[int, List[float]]:
    num_bins = max(0, len(bins) - 1)
    if num_bins <= 0:
        return -100, []
    if value is None:
        return -100, [0.0] * num_bins
    if isinstance(value, float) and math.isnan(value):
        return -100, [0.0] * num_bins
    idx = _bucketize_int(float(value), bins)
    target = [0.0] * num_bins
    if not use_soft or num_bins == 1:
        target[idx] = 1.0
        return idx, target
    left = float(bins[idx])
    right = float(bins[min(idx + 1, len(bins) - 1)])
    if idx >= num_bins - 1 or right <= left:
        target[idx] = 1.0
        return idx, target
    ratio = (float(value) - left) / (right - left)
    ratio = min(max(ratio, 0.0), 1.0)
    target[idx] = 1.0 - ratio
    target[idx + 1] = ratio
    return idx, target


def _coerce_optional_int(value: Optional[object]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_float(value: Optional[object]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_nll(nll_value: Optional[object], ppl_value: Optional[object]) -> Optional[float]:
    nll = _coerce_optional_float(nll_value)
    if nll is not None:
        return nll
    ppl = _coerce_optional_float(ppl_value)
    if ppl is None:
        return None
    if ppl <= 0:
        return None
    return math.log(ppl)


_PROXY_CACHE_FIELDS = (
    "proxy_length_A",
    "proxy_length_B",
    "proxy_nll_A",
    "proxy_nll_B",
    "proxy_ppl_A",
    "proxy_ppl_B",
    "proxy_ttr_A",
    "proxy_ttr_B",
)


def _iter_jsonl(path: Optional[str]) -> Iterable[Dict[str, object]]:
    if path is None:
        return iter(())
    def _gen() -> Iterable[Dict[str, object]]:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                yield json.loads(line)
    return _gen()


def _infer_proxy_cache_path(path: str) -> Optional[str]:
    base = Path(path)
    candidates = [
        base.with_suffix(".proxy.jsonl"),
        base.with_suffix(".proxy_cache.jsonl"),
        base.with_name(base.name + ".proxy.jsonl"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _is_null_value(value: Optional[object]) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _missing_proxy_fields(raw: Dict[str, object]) -> bool:
    response_b = _coerce_optional_text(raw.get("response_B"))
    has_b = _has_response(response_b)
    def _missing_nll(prefix: str) -> bool:
        nll = raw.get(f"proxy_nll_{prefix}")
        ppl = raw.get(f"proxy_ppl_{prefix}")
        return _is_null_value(nll) and _is_null_value(ppl)

    if _missing_nll("A"):
        return True
    if has_b and _missing_nll("B"):
        return True
    return False


def _merge_proxy_cache(raw: Dict[str, object], cache_raw: Dict[str, object]) -> None:
    for field in _PROXY_CACHE_FIELDS:
        if _is_null_value(raw.get(field)) and not _is_null_value(cache_raw.get(field)):
            raw[field] = cache_raw.get(field)


def _preview_text(text: str, limit: int = 120) -> str:
    if not text:
        return ""
    text = text.replace("\n", "\\n")
    return text[:limit]


def _format_span_debug(text: str, start: int, end: int, window: int = 120) -> str:
    if start < 0 or end < 0:
        return _preview_text(text, limit=window)
    left = max(0, start - window)
    right = min(len(text), end + window)
    return _preview_text(text[left:right], limit=window * 2)


def _compact_text_with_map(text: str) -> Tuple[str, List[int]]:
    compact = []
    mapping: List[int] = []
    for idx, ch in enumerate(text):
        if ch.isspace():
            continue
        compact.append(ch)
        mapping.append(idx)
    return "".join(compact), mapping


def _find_response_span(
    prompt: str, response: str, start_at: Optional[int] = None
) -> Optional[Tuple[int, int]]:
    response = (response or "").strip()
    if not response:
        return None
    if start_at is not None and start_at >= 0:
        start = prompt.find(response, start_at)
        if start >= 0:
            return start, start + len(response)
    start = prompt.find(response)
    if start >= 0:
        return start, start + len(response)

    compact_prompt, mapping = _compact_text_with_map(prompt)
    compact_resp, _ = _compact_text_with_map(response)
    if not compact_resp:
        return None
    compact_start = 0
    if start_at is not None and start_at >= 0:
        for idx, orig_idx in enumerate(mapping):
            if orig_idx >= start_at:
                compact_start = idx
                break
    match = compact_prompt.find(compact_resp, compact_start)
    if match < 0:
        return None
    start_idx = mapping[match]
    end_idx = mapping[match + len(compact_resp) - 1] + 1
    return start_idx, end_idx


def _find_section_span(
    text: str, start_marker: str, end_marker: Optional[str]
) -> Optional[Tuple[int, int]]:
    start = text.find(start_marker)
    if start < 0:
        return None
    start += len(start_marker)
    while start < len(text) and text[start] in " \n\r\t":
        start += 1
    end = len(text)
    if end_marker:
        idx = text.find(end_marker, start)
        if idx >= 0:
            end = idx
    while end > start and text[end - 1] in " \n\r\t":
        end -= 1
    if end <= start:
        return None
    return start, end


def _pad_sequences(
    sequences: List[List[int]],
    pad_id: int,
    max_length: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max((len(seq) for seq in sequences), default=0)
    if max_length is not None:
        max_len = min(max_len, max_length)
    batch = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for idx, seq in enumerate(sequences):
        seq = seq[:max_len]
        if not seq:
            continue
        length = len(seq)
        batch[idx, :length] = torch.tensor(seq, dtype=torch.long)
        mask[idx, :length] = 1
    return batch, mask


class DIBJudgeCollator:
    def __init__(
        self,
        tokenizer,
        max_response_len: Optional[int] = 1024,
        max_lm_len: Optional[int] = 4096,
        proxy_config: Optional[ProxyTaskConfig] = None,
        enable_proxy_labels: bool = True,
        debug_spike: bool = False,
        debug_preview_len: int = 120,
        filter_truncated: bool = False,
        min_target_tokens: int = 0,
        drop_truncated: bool = False,
        drop_min_target_tokens: int = 0,
    ) -> None:
        if max_response_len is None:
            max_response_len = 1024
        if max_lm_len is None:
            max_lm_len = 4096
        self.tokenizer = tokenizer
        self.max_response_len = max_response_len
        self.max_lm_len = max_lm_len
        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.tokenizer.eos_token_id or 0
        self.proxy_config = proxy_config or ProxyTaskConfig()
        self.enable_proxy_labels = bool(enable_proxy_labels)
        self.debug_spike = bool(debug_spike)
        self.debug_preview_len = max(20, int(debug_preview_len))
        self.filter_truncated = bool(filter_truncated)
        self.min_target_tokens = max(0, int(min_target_tokens))
        self.drop_truncated = bool(drop_truncated)
        self.drop_min_target_tokens = max(0, int(drop_min_target_tokens))

    def __call__(self, batch: Sequence[DIBJudgeExample]) -> Dict[str, torch.Tensor]:
        base_sequences: List[List[int]] = []
        response_masks: List[List[int]] = []
        enable_proxy_labels = self.enable_proxy_labels
        length_labels: List[int] = []
        length_targets: List[List[float]] = []
        nll_labels: List[int] = []
        nll_targets: List[List[float]] = []
        ttr_labels: List[int] = []
        ttr_targets: List[List[float]] = []
        response_mask: List[int] = []
        use_soft = bool(getattr(self.proxy_config, "use_soft_labels", True))
        for ex in batch:
            # Encode A/B directly as separate sequences for each line.
            resp_a = ex.response_a
            resp_b = ex.response_b
            has_a = _has_response(resp_a)
            has_b = _has_response(resp_b)
            response_mask.extend([1 if has_a else 0, 1 if has_b else 0])
            responses = (
                (
                    resp_a or "",
                    ex.proxy_length_a,
                    ex.proxy_nll_a,
                    ex.proxy_ttr_a,
                    has_a,
                ),
                (
                    resp_b or "",
                    ex.proxy_length_b,
                    ex.proxy_nll_b,
                    ex.proxy_ttr_b,
                    has_b,
                ),
            )
            for response, length_val, nll_val, ttr_val, present in responses:
                token_ids = _encode_ids(self.tokenizer, response, self.max_response_len)
                base_sequences.append(token_ids)
                resp_mask = [0] * len(token_ids)
                if present:
                    resp_mask = [1] * len(token_ids)
                response_masks.append(resp_mask)
                if not enable_proxy_labels:
                    continue
                if not present:
                    # Missing responses (e.g., single-response samples) should not emit proxy targets.
                    length_labels.append(-100)
                    length_targets.append([0.0] * max(0, len(self.proxy_config.length_bins) - 1))
                    nll_labels.append(-100)
                    nll_targets.append([0.0] * max(0, len(self.proxy_config.nll_bins) - 1))
                    ttr_labels.append(-100)
                    ttr_targets.append([0.0] * max(0, len(self.proxy_config.ttr_bins) - 1))
                    continue
                if length_val is None:
                    length_val = float(len(token_ids))
                length_label, length_target = _soft_bin_target(
                    length_val, self.proxy_config.length_bins, use_soft
                )
                length_labels.append(int(length_label))
                length_targets.append(length_target)

                nll_label, nll_target = _soft_bin_target(
                    nll_val, self.proxy_config.nll_bins, use_soft
                )
                nll_labels.append(int(nll_label))
                nll_targets.append(nll_target)

                if ttr_val is None:
                    token_count = len(token_ids)
                    if token_count:
                        ttr_val = float(len(set(token_ids))) / float(token_count)
                    else:
                        ttr_val = 0.0
                ttr_label, ttr_target = _soft_bin_target(
                    ttr_val, self.proxy_config.ttr_bins, use_soft
                )
                ttr_labels.append(int(ttr_label))
                ttr_targets.append(ttr_target)
        original_input_ids, original_attention_mask = _pad_sequences(
            base_sequences,
            self.pad_id,
            self.max_response_len,
        )
        original_response_mask, _ = _pad_sequences(
            response_masks,
            0,
            self.max_response_len,
        )
        need_meta = (
            self.debug_spike
            or self.filter_truncated
            or self.min_target_tokens > 0
            or self.drop_truncated
            or self.drop_min_target_tokens > 0
        )
        lm_inputs, lm_labels, lm_response_types, debug_info = self._build_lm_inputs(
            batch, return_debug=self.debug_spike, return_meta=need_meta
        )

        batch_size = len(batch)
        original_input_ids = original_input_ids.view(batch_size, 2, -1)
        original_attention_mask = original_attention_mask.view(batch_size, 2, -1)
        original_response_mask = original_response_mask.view(batch_size, 2, -1)
        response_mask_tensor = torch.tensor(response_mask, dtype=torch.long).view(batch_size, 2)

        if debug_info is not None and (self.filter_truncated or self.min_target_tokens > 0):
            prompt_tokens = debug_info.get("prompt_tokens", [])
            truncated = debug_info.get("truncated", [])
            label_tokens = debug_info.get("label_tokens", [])
            drop = []
            for idx in range(batch_size):
                drop_flag = False
                if self.filter_truncated:
                    prompt_overflow = False
                    if idx < len(prompt_tokens):
                        prompt_overflow = int(prompt_tokens[idx]) >= int(self.max_lm_len)
                    trunc_flag = False
                    if idx < len(truncated):
                        trunc_flag = bool(truncated[idx])
                    drop_flag = drop_flag or trunc_flag or prompt_overflow
                if self.min_target_tokens > 0 and idx < len(label_tokens):
                    drop_flag = drop_flag or int(label_tokens[idx]) < self.min_target_tokens
                drop.append(drop_flag)
            if any(drop):
                drop_mask = torch.tensor(drop, dtype=torch.bool)
                lm_labels = lm_labels.masked_fill(drop_mask.unsqueeze(1), -100)
                if torch.is_tensor(lm_response_types):
                    lm_response_types = lm_response_types.masked_fill(
                        drop_mask.unsqueeze(1), 0
                    )
                debug_info["lm_filtered"] = drop

        batch_dict = {
            "original_input_ids": original_input_ids,
            "original_attention_mask": original_attention_mask,
            "original_response_mask": original_response_mask,
            "lm_input_ids": lm_inputs["input_ids"],
            "lm_attention_mask": lm_inputs["attention_mask"],
            "lm_labels": lm_labels,
            "lm_response_types": lm_response_types,
            "response_mask": response_mask_tensor,
            "proxy_labels_enabled": enable_proxy_labels,
        }
        drop_count = 0
        drop_maxlen_count = 0
        drop_min_target_count = 0
        drop_all_fallback = False
        if debug_info is not None and (self.drop_truncated or self.drop_min_target_tokens > 0):
            prompt_tokens = debug_info.get("prompt_tokens", [])
            truncated = debug_info.get("truncated", [])
            label_tokens = debug_info.get("label_tokens", [])
            drop = []
            drop_maxlen = []
            drop_min_target = []
            for idx in range(batch_size):
                drop_max = False
                if self.drop_truncated:
                    prompt_overflow = False
                    if idx < len(prompt_tokens):
                        prompt_overflow = int(prompt_tokens[idx]) >= int(self.max_lm_len)
                    trunc_flag = False
                    if idx < len(truncated):
                        trunc_flag = bool(truncated[idx])
                    drop_max = prompt_overflow or trunc_flag
                drop_min = False
                if self.drop_min_target_tokens > 0 and idx < len(label_tokens):
                    drop_min = int(label_tokens[idx]) < self.drop_min_target_tokens
                drop_maxlen.append(drop_max)
                drop_min_target.append(drop_min)
                drop.append(drop_max or drop_min)
            drop_count = sum(1 for flag in drop if flag)
            drop_maxlen_count = sum(1 for flag in drop_maxlen if flag)
            drop_min_target_count = sum(1 for flag in drop_min_target if flag)
            if drop_count:
                keep = [not flag for flag in drop]
                keep_mask = torch.tensor(keep, dtype=torch.bool)
                if keep_mask.sum().item() == 0:
                    keep_mask = torch.zeros_like(keep_mask)
                    keep_mask[0] = True
                    drop_all_fallback = True
                for key, value in list(batch_dict.items()):
                    if torch.is_tensor(value) and value.dim() > 0 and value.size(0) == batch_size:
                        batch_dict[key] = value[keep_mask]
                    elif isinstance(value, list) and len(value) == batch_size:
                        batch_dict[key] = [v for v, keep_item in zip(value, keep) if keep_item]
                if debug_info is not None:
                    for key, value in list(debug_info.items()):
                        if isinstance(value, list) and len(value) == batch_size:
                            debug_info[key] = [
                                v for v, keep_item in zip(value, keep) if keep_item
                            ]
                if drop_all_fallback:
                    if torch.is_tensor(batch_dict.get("lm_labels")):
                        batch_dict["lm_labels"] = batch_dict["lm_labels"].masked_fill(
                            torch.ones_like(batch_dict["lm_labels"], dtype=torch.bool), -100
                        )
                    if torch.is_tensor(batch_dict.get("lm_response_types")):
                        batch_dict["lm_response_types"] = batch_dict["lm_response_types"].zero_()
                    if torch.is_tensor(batch_dict.get("response_mask")):
                        batch_dict["response_mask"] = batch_dict["response_mask"].zero_()
                    for key in ("proxy_length_label", "proxy_nll_label", "proxy_ttr_label"):
                        if torch.is_tensor(batch_dict.get(key)):
                            batch_dict[key] = batch_dict[key].masked_fill(
                                torch.ones_like(batch_dict[key], dtype=torch.bool), -100
                            )
                    for key in ("proxy_length_target", "proxy_nll_target", "proxy_ttr_target"):
                        if torch.is_tensor(batch_dict.get(key)):
                            batch_dict[key] = batch_dict[key].zero_()
        if debug_info is not None:
            if "prompt_preview" in debug_info:
                batch_dict["debug_prompt_preview"] = debug_info["prompt_preview"]
            if "output_preview" in debug_info:
                batch_dict["debug_output_preview"] = debug_info["output_preview"]
            if "prompt_tokens" in debug_info:
                batch_dict["debug_prompt_tokens"] = torch.tensor(
                    debug_info["prompt_tokens"], dtype=torch.long
                )
            if "lm_tokens" in debug_info:
                batch_dict["debug_lm_tokens"] = torch.tensor(
                    debug_info["lm_tokens"], dtype=torch.long
                )
            if "truncated" in debug_info:
                batch_dict["debug_lm_truncated"] = torch.tensor(
                    debug_info["truncated"], dtype=torch.long
                )
            if "label_tokens" in debug_info:
                batch_dict["debug_label_tokens"] = torch.tensor(
                    debug_info["label_tokens"], dtype=torch.long
                )
            if "lm_filtered" in debug_info:
                batch_dict["debug_lm_filtered"] = torch.tensor(
                    debug_info["lm_filtered"], dtype=torch.long
                )
        if self.drop_truncated or self.drop_min_target_tokens > 0:
            batch_dict["debug_lm_drop_count"] = torch.tensor(drop_count, dtype=torch.long)
            batch_dict["debug_lm_drop_ratio"] = torch.tensor(
                drop_count / batch_size if batch_size > 0 else 0.0, dtype=torch.float
            )
            batch_dict["debug_lm_drop_seen"] = torch.tensor(batch_size, dtype=torch.long)
            batch_dict["debug_lm_drop_maxlen_count"] = torch.tensor(
                drop_maxlen_count, dtype=torch.long
            )
            batch_dict["debug_lm_drop_min_target_count"] = torch.tensor(
                drop_min_target_count, dtype=torch.long
            )
            batch_dict["debug_lm_drop_all_fallback"] = torch.tensor(
                1 if drop_all_fallback else 0, dtype=torch.long
            )
        if enable_proxy_labels:
            proxy_len = torch.tensor(length_labels, dtype=torch.long).view(batch_size, 2)
            proxy_len_target = torch.tensor(length_targets, dtype=torch.float).view(
                batch_size, 2, -1
            )
            proxy_nll = torch.tensor(nll_labels, dtype=torch.long).view(batch_size, 2)
            proxy_nll_target = torch.tensor(nll_targets, dtype=torch.float).view(
                batch_size, 2, -1
            )
            proxy_ttr = torch.tensor(ttr_labels, dtype=torch.long).view(batch_size, 2)
            proxy_ttr_target = torch.tensor(ttr_targets, dtype=torch.float).view(
                batch_size, 2, -1
            )
            batch_dict.update(
                {
                    "proxy_length_label": proxy_len,
                    "proxy_length_target": proxy_len_target,
                    "proxy_nll_label": proxy_nll,
                    "proxy_nll_target": proxy_nll_target,
                    "proxy_ttr_label": proxy_ttr,
                    "proxy_ttr_target": proxy_ttr_target,
                }
            )
        return batch_dict

    def _build_lm_inputs(
        self,
        batch: Sequence[DIBJudgeExample],
        return_debug: bool = False,
        return_meta: bool = False,
    ) -> Tuple[
        Dict[str, torch.Tensor],
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[Dict[str, object]],
    ]:
        prompts = [self._build_prompt(ex) for ex in batch]
        targets = [ex.output for ex in batch]
        debug_info = None
        if return_debug or return_meta:
            debug_info = {
                "prompt_tokens": [],
                "lm_tokens": [],
                "label_tokens": [],
                "truncated": [],
            }
            if return_debug:
                debug_info["prompt_preview"] = []
                debug_info["output_preview"] = []

        full_texts = [p + t for p, t in zip(prompts, targets)]
        full_enc = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_lm_len,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            add_special_tokens=True,
        )
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            input_ids = full_enc["input_ids"]
            attention_mask = full_enc["attention_mask"]
            max_len = input_ids.size(1)
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = eos_id or 0
            extended = False
            for i in range(input_ids.size(0)):
                length = int(attention_mask[i].sum().item())
                if length <= 0:
                    continue
                last_idx = length - 1
                if input_ids[i, last_idx].item() == eos_id:
                    continue
                if length >= max_len and not extended:
                    # Extend tensors so we can append EOS without overwriting tokens.
                    input_ids = torch.cat(
                        [input_ids, input_ids.new_full((input_ids.size(0), 1), pad_id)],
                        dim=1,
                    )
                    attention_mask = torch.cat(
                        [
                            attention_mask,
                            attention_mask.new_zeros((attention_mask.size(0), 1)),
                        ],
                        dim=1,
                    )
                    max_len = input_ids.size(1)
                    extended = True
                if length < max_len:
                    input_ids[i, length] = eos_id
                    attention_mask[i, length] = 1
            full_enc["input_ids"] = input_ids
            full_enc["attention_mask"] = attention_mask
            if extended:
                special_tokens_mask = full_enc.get("special_tokens_mask")
                if torch.is_tensor(special_tokens_mask):
                    full_enc["special_tokens_mask"] = torch.cat(
                        [
                            special_tokens_mask,
                            special_tokens_mask.new_zeros(
                                (special_tokens_mask.size(0), 1)
                            ),
                        ],
                        dim=1,
                    )
        special_mask = full_enc.get("special_tokens_mask")
        if torch.is_tensor(special_mask):
            special_mask = special_mask.tolist()

        prompt_enc = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_lm_len,
            return_tensors="pt",
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        prompt_offsets = prompt_enc.pop("offset_mapping", None)
        if torch.is_tensor(prompt_offsets):
            prompt_offsets = prompt_offsets.tolist()

        labels = full_enc["input_ids"].clone()
        response_types = torch.zeros_like(labels)
        for i, ex in enumerate(batch):
            has_b = _has_response(ex.response_b)
            prompt_len = int(prompt_enc["attention_mask"][i].sum().item())
            prompt_len = min(prompt_len, labels.size(1))
            lead_special = 0
            if special_mask is not None:
                for flag in special_mask[i]:
                    if flag == 1:
                        lead_special += 1
                    else:
                        break
            prompt_len = min(prompt_len + lead_special, labels.size(1))
            # Mask prompt tokens so loss is only on the target continuation.
            labels[i, :prompt_len] = -100

            prompt = prompts[i]
            span_a = _find_response_span(prompt, ex.response_a)
            span_b = None
            if has_b:
                span_b = _find_response_span(
                    prompt, ex.response_b or "", start_at=span_a[1] if span_a else None
                )
            used_token_fallback = False
            if prompt_offsets is not None and (span_a or span_b):
                for tok_idx, (start, end) in enumerate(prompt_offsets[i][:prompt_len]):
                    if start == end == 0:
                        continue
                    if span_a and start < span_a[1] and end > span_a[0]:
                        response_types[i, lead_special + tok_idx] = 1
                    elif span_b and start < span_b[1] and end > span_b[0]:
                        response_types[i, lead_special + tok_idx] = 2
            else:
                used_token_fallback = True
                prompt_ids = prompt_enc["input_ids"][i][:prompt_len].tolist()
                lead = lead_special
                resp_a_ids = _encode_ids(self.tokenizer, ex.response_a, self.max_lm_len)
                resp_b_ids = (
                    _encode_ids(self.tokenizer, ex.response_b or "", self.max_lm_len)
                    if has_b
                    else []
                )
                span_a_ids = _find_subseq(prompt_ids, resp_a_ids)
                if span_a_ids is None and resp_a_ids:
                    span_a_ids = _find_subseq(prompt_ids, resp_a_ids[1:])
                span_b_ids = _find_subseq(prompt_ids, resp_b_ids)
                if span_b_ids is None and resp_b_ids:
                    span_b_ids = _find_subseq(prompt_ids, resp_b_ids[1:])
                if span_a_ids is not None:
                    start = min(response_types.size(1), lead + span_a_ids[0])
                    end = min(response_types.size(1), lead + span_a_ids[1])
                    response_types[i, start:end] = 1
                if span_b_ids is not None:
                    start = min(response_types.size(1), lead + span_b_ids[0])
                    end = min(response_types.size(1), lead + span_b_ids[1])
                    response_types[i, start:end] = 2

            truncated = False
            if hasattr(full_enc, "encodings") and full_enc.encodings:
                try:
                    truncated = bool(full_enc.encodings[i].truncated)
                except (AttributeError, IndexError):
                    truncated = False
            lm_tokens = int(full_enc["attention_mask"][i].sum().item())
            prompt_tokens = int(prompt_enc["attention_mask"][i].sum().item())
            if response_types[i].eq(1).sum().item() == 0:
                probe = _preview_text(ex.response_a, limit=80)
                probe_idx = prompt.find(probe) if probe else -1
                snippet = _format_span_debug(prompt, probe_idx, probe_idx + len(probe))
                warnings.warn(
                    "lm_response_types span A not found; "
                    f"prompt_tokens={prompt_tokens} lm_tokens={lm_tokens} "
                    f"truncated={'yes' if truncated else 'no'} "
                    f"response_span={'yes' if span_a is not None else 'no'} "
                    f"fallback={'token' if used_token_fallback else 'response'}; "
                    f"resp_preview='{_preview_text(ex.response_a)}'; "
                    f"prompt_snippet='{snippet}'",
                    RuntimeWarning,
                )
            if has_b and response_types[i].eq(2).sum().item() == 0:
                probe = _preview_text(ex.response_b or "", limit=80)
                probe_idx = prompt.find(probe) if probe else -1
                snippet = _format_span_debug(prompt, probe_idx, probe_idx + len(probe))
                warnings.warn(
                    "lm_response_types span B not found; "
                    f"prompt_tokens={prompt_tokens} lm_tokens={lm_tokens} "
                    f"truncated={'yes' if truncated else 'no'} "
                    f"response_span={'yes' if span_b is not None else 'no'} "
                    f"fallback={'token' if used_token_fallback else 'response'}; "
                    f"resp_preview='{_preview_text(ex.response_b or '')}'; "
                    f"prompt_snippet='{snippet}'",
                    RuntimeWarning,
                )

            if debug_info is not None and return_debug:
                debug_info["prompt_preview"].append(
                    _preview_text(prompts[i], limit=self.debug_preview_len)
                )
                debug_info["output_preview"].append(
                    _preview_text(targets[i], limit=self.debug_preview_len)
                )
            if debug_info is not None:
                debug_info["prompt_tokens"].append(int(prompt_len))
                debug_info["truncated"].append(1 if truncated else 0)

        attention_mask = full_enc.get("attention_mask")
        if torch.is_tensor(attention_mask):
            labels = labels.masked_fill(attention_mask.eq(0), -100)

        if debug_info is not None:
            label_counts = labels.ne(-100).sum(dim=1).cpu().tolist()
            debug_info["label_tokens"] = label_counts
            attention = full_enc.get("attention_mask")
            if torch.is_tensor(attention):
                debug_info["lm_tokens"] = attention.sum(dim=1).cpu().tolist()
            else:
                debug_info["lm_tokens"] = [0] * len(batch)

        return full_enc, labels, response_types, debug_info

    def _build_prompt(self, ex: DIBJudgeExample) -> str:
        prompt = ex.judge_prompt
        return prompt
