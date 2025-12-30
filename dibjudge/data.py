from __future__ import annotations

import json
import random
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from torch.utils.data import Dataset


@dataclass
class DIBJudgeExample:
    instruction: str
    response_a: str
    response_b: str
    response_a_bt: str
    response_b_bt: str
    judge_prompt: str
    output: str

    @classmethod
    def from_dict(cls, raw: Dict[str, object]) -> "DIBJudgeExample":
        response_a = str(raw.get("response_A", ""))
        response_b = str(raw.get("response_B", ""))
        response_a_bt = _coerce_text(
            raw.get("response_A_bt", raw.get("response_A_eng")), response_a
        )
        response_b_bt = _coerce_text(
            raw.get("response_B_bt", raw.get("response_B_eng")), response_b
        )
        return cls(
            instruction=str(raw.get("instruction", "")),
            response_a=response_a,
            response_b=response_b,
            response_a_bt=response_a_bt,
            response_b_bt=response_b_bt,
            judge_prompt=str(raw.get("judge_prompt", "")),
            output=str(raw.get("output", "")),
        )


class DIBJudgeDataset(Dataset):
    def __init__(self, samples: Sequence[DIBJudgeExample]) -> None:
        self.samples = list(samples)

    @classmethod
    def from_jsonl(cls, path: str) -> "DIBJudgeDataset":
        samples: List[DIBJudgeExample] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                samples.append(DIBJudgeExample.from_dict(raw))
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


def _pad_sequences_with_labels(
    sequences: List[List[int]],
    labels: List[List[int]],
    pad_id: int,
    max_length: Optional[int],
    label_pad: int = -100,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max((len(seq) for seq in sequences), default=0)
    if max_length is not None:
        max_len = min(max_len, max_length)
    batch = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
    label_batch = torch.full((len(sequences), max_len), label_pad, dtype=torch.long)
    for idx, seq in enumerate(sequences):
        seq = seq[:max_len]
        lab = labels[idx][:max_len]
        if not seq:
            continue
        length = len(seq)
        batch[idx, :length] = torch.tensor(seq, dtype=torch.long)
        mask[idx, :length] = 1
        label_batch[idx, :length] = torch.tensor(lab, dtype=torch.long)
    return batch, mask, label_batch


def _split_token_spans(
    token_ids: List[int],
    min_len: int,
    max_len: int,
    rng: random.Random,
) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    idx = 0
    total = len(token_ids)
    while idx < total:
        span_len = min_len if max_len <= min_len else rng.randint(min_len, max_len)
        end = min(total, idx + span_len)
        spans.append((idx, end))
        idx = end
    return spans


def _select_span_indices(
    spans: List[Tuple[int, int]],
    total_tokens: int,
    target_ratio: float,
    rng: random.Random,
) -> List[int]:
    if not spans or total_tokens <= 0:
        return []
    span_indices = list(range(len(spans)))
    rng.shuffle(span_indices)
    target = max(1, int(total_tokens * target_ratio))
    selected: List[int] = []
    covered = 0
    for idx in span_indices:
        start, end = spans[idx]
        selected.append(idx)
        covered += end - start
        if covered >= target:
            break
    return selected


def _apply_shuffle_or_perturb(
    token_ids: List[int],
    rng: random.Random,
    perturb_sources: List[List[int]],
    shuffle_ratio: float,
    min_len: int = 3,
    max_len: int = 8,
) -> Tuple[List[int], List[int]]:
    if not token_ids:
        return token_ids, []
    spans = _split_token_spans(token_ids, min_len, max_len, rng)
    selected = _select_span_indices(spans, len(token_ids), shuffle_ratio, rng)
    if not selected:
        labels = [1] * len(token_ids)
        return token_ids, labels

    op = "shuffle" if rng.random() < 0.5 else "perturb"
    if op == "shuffle" and len(selected) < 2:
        op = "perturb"

    new_tokens = token_ids[:]
    if op == "shuffle":
        # Align permutation with positional order so spans actually swap locations.
        selected_sorted = sorted(selected)
        selected_spans = [spans[idx] for idx in selected_sorted]
        permuted = selected_spans[:]
        rng.shuffle(permuted)
        if permuted == selected_spans and len(permuted) > 1:
            permuted = permuted[1:] + permuted[:1]
        span_iter = iter(permuted)
        rebuilt: List[int] = []
        selected_set = set(selected_sorted)
        for idx, (start, end) in enumerate(spans):
            if idx in selected_set:
                src_start, src_end = next(span_iter)
                rebuilt.extend(token_ids[src_start:src_end])
            else:
                rebuilt.extend(token_ids[start:end])
        new_tokens = rebuilt
    else:
        changed = False
        for idx in selected:
            start, end = spans[idx]
            span_len = end - start
            if span_len <= 0:
                continue
            candidates = [
                seq for seq in perturb_sources if seq is not token_ids and len(seq) >= span_len
            ]
            if not candidates:
                continue
            source = rng.choice(candidates)
            src_start = rng.randrange(0, len(source) - span_len + 1)
            new_tokens[start:end] = source[src_start : src_start + span_len]
            changed = True
        if not changed and len(new_tokens) > 1:
            i, j = rng.sample(range(len(new_tokens)), 2)
            new_tokens[i], new_tokens[j] = new_tokens[j], new_tokens[i]

    labels = [1] * len(token_ids)
    for idx in selected:
        start, end = spans[idx]
        for pos in range(start, end):
            if 0 <= pos < len(labels):
                labels[pos] = 0
    return new_tokens, labels


class DIBJudgeCollator:
    def __init__(
        self,
        tokenizer,
        max_bias_len: Optional[int] = 1024,
        max_ref_len: Optional[int] = 1024,
        max_lm_len: Optional[int] = 4096,
        shuffle_ratio_min: float = 0.05,
        shuffle_ratio_max: float = 0.2,
        shuffle_ratio_step: float = 0.02,
        shuffle_schedule_start_step: int = 0,
    ) -> None:
        if max_bias_len is None:
            max_bias_len = 1024
        if max_lm_len is None:
            max_lm_len = 4096
        self.tokenizer = tokenizer
        self.max_bias_len = max_bias_len
        self.max_ref_len = max_ref_len
        self.max_lm_len = max_lm_len
        self.shuffle_rng = random.Random(13)
        self.shuffle_pad_id = self.tokenizer.pad_token_id
        if self.shuffle_pad_id is None:
            self.shuffle_pad_id = self.tokenizer.eos_token_id or 0
        self.shuffle_ratio_min = min(shuffle_ratio_min, shuffle_ratio_max)
        self.shuffle_ratio_max = max(shuffle_ratio_min, shuffle_ratio_max)
        self.shuffle_ratio_step = abs(shuffle_ratio_step)
        self.shuffle_schedule_start_step = max(0, int(shuffle_schedule_start_step))
        self.shuffle_step = 0
        # Curriculum: start easy (more shuffled), then anneal to harder (less shuffled).
        self.shuffle_ratio = self.shuffle_ratio_max

    def __call__(self, batch: Sequence[DIBJudgeExample]) -> Dict[str, torch.Tensor]:
        current_ratio = self.shuffle_ratio
        base_sequences: List[List[int]] = []
        original_labels: List[List[int]] = []
        shuffle_sequences: List[List[int]] = []
        shuffle_labels: List[List[int]] = []
        english_texts: List[str] = []
        for ex in batch:
            # Encode A/B directly as separate sequences for each line.
            english_texts.extend([ex.response_a_bt, ex.response_b_bt])
            for response in (ex.response_a, ex.response_b):
                token_ids = _encode_ids(self.tokenizer, response, self.max_bias_len)
                base_sequences.append(token_ids)
                original_labels.append([1] * len(token_ids))
        for token_ids in base_sequences:
            shuffled_ids, token_labels = _apply_shuffle_or_perturb(
                token_ids,
                rng=self.shuffle_rng,
                perturb_sources=base_sequences,
                shuffle_ratio=current_ratio,
            )
            shuffle_sequences.append(shuffled_ids)
            shuffle_labels.append(token_labels)
        original_input_ids, original_attention_mask, original_shuffle_labels = _pad_sequences_with_labels(
            base_sequences,
            original_labels,
            self.shuffle_pad_id,
            self.max_bias_len,
            label_pad=-100,
        )
        shuffle_input_ids, shuffle_attention_mask, shuffle_labels = _pad_sequences_with_labels(
            shuffle_sequences,
            shuffle_labels,
            self.shuffle_pad_id,
            self.max_bias_len,
            label_pad=-100,
        )
        english_enc = _encode_texts(
            self.tokenizer,
            english_texts,
            self.max_ref_len,
            add_special_tokens=False,
        )

        lm_inputs, lm_labels, lm_response_types = self._build_lm_inputs(batch)

        batch_size = len(batch)
        original_input_ids = original_input_ids.view(batch_size, 2, -1)
        original_attention_mask = original_attention_mask.view(batch_size, 2, -1)
        original_shuffle_labels = original_shuffle_labels.view(batch_size, 2, -1)
        shuffle_input_ids = shuffle_input_ids.view(batch_size, 2, -1)
        shuffle_attention_mask = shuffle_attention_mask.view(batch_size, 2, -1)
        shuffle_labels = shuffle_labels.view(batch_size, 2, -1)
        english_input_ids = english_enc["input_ids"].view(batch_size, 2, -1)
        english_attention_mask = english_enc["attention_mask"].view(
            batch_size, 2, -1
        )
        self.shuffle_step += 1
        if self.shuffle_ratio_step > 0 and self.shuffle_step > self.shuffle_schedule_start_step:
            self.shuffle_ratio = max(
                self.shuffle_ratio_min, self.shuffle_ratio - self.shuffle_ratio_step
            )

        return {
            "original_input_ids": original_input_ids,
            "original_attention_mask": original_attention_mask,
            "shuffle_input_ids": shuffle_input_ids,
            "shuffle_attention_mask": shuffle_attention_mask,
            "original_shuffle_labels": original_shuffle_labels,
            "shuffle_labels": shuffle_labels,
            "english_input_ids": english_input_ids,
            "english_attention_mask": english_attention_mask,
            "shuffle_ratio": current_ratio,
            "lm_input_ids": lm_inputs["input_ids"],
            "lm_attention_mask": lm_inputs["attention_mask"],
            "lm_labels": lm_labels,
            "lm_response_types": lm_response_types,
        }

    def _build_lm_inputs(
        self, batch: Sequence[DIBJudgeExample]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        prompts = [self._build_prompt(ex) for ex in batch]
        targets = [ex.output for ex in batch]

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
            span_a = _find_section_span(
                prompt, "###Response A to evaluate:", "###Response B to evaluate:"
            )
            span_b = _find_section_span(
                prompt, "###Response B to evaluate:", "###Evaluation Criteria:"
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
                resp_b_ids = _encode_ids(self.tokenizer, ex.response_b, self.max_lm_len)
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
                    f"marker_span={'yes' if span_a is not None else 'no'} "
                    f"fallback={'token' if used_token_fallback else 'marker'}; "
                    f"resp_preview='{_preview_text(ex.response_a)}'; "
                    f"prompt_snippet='{snippet}'",
                    RuntimeWarning,
                )
            if response_types[i].eq(2).sum().item() == 0:
                probe = _preview_text(ex.response_b, limit=80)
                probe_idx = prompt.find(probe) if probe else -1
                snippet = _format_span_debug(prompt, probe_idx, probe_idx + len(probe))
                warnings.warn(
                    "lm_response_types span B not found; "
                    f"prompt_tokens={prompt_tokens} lm_tokens={lm_tokens} "
                    f"truncated={'yes' if truncated else 'no'} "
                    f"marker_span={'yes' if span_b is not None else 'no'} "
                    f"fallback={'token' if used_token_fallback else 'marker'}; "
                    f"resp_preview='{_preview_text(ex.response_b)}'; "
                    f"prompt_snippet='{snippet}'",
                    RuntimeWarning,
                )

        return full_enc, labels, response_types

    def _build_prompt(self, ex: DIBJudgeExample) -> str:
        prompt = ex.judge_prompt
        return prompt
