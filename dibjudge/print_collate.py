from __future__ import annotations

import argparse
import re
from typing import List, Tuple

from transformers import AutoTokenizer

from .data import DIBJudgeCollator, DIBJudgeDataset


def _ensure_pad(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _collect_spans(labels: List[int], target: int = 0) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start = None
    for idx, label in enumerate(labels):
        if label == target:
            if start is None:
                start = idx
        elif start is not None:
            spans.append((start, idx))
            start = None
    if start is not None:
        spans.append((start, len(labels)))
    return spans


def _decode_marked(tokenizer, ids: List[int], labels: List[int]) -> str:
    if not ids:
        return ""
    spans = _collect_spans(labels, target=0)
    if not spans:
        return _normalize(tokenizer.decode(ids, skip_special_tokens=True))
    parts: List[str] = []
    cursor = 0
    for start, end in spans:
        if start > cursor:
            chunk = _normalize(tokenizer.decode(ids[cursor:start], skip_special_tokens=True))
            if chunk:
                parts.append(chunk)
        marked = _normalize(tokenizer.decode(ids[start:end], skip_special_tokens=True))
        if marked:
            parts.append(f"<<{marked}>>")
        cursor = end
    if cursor < len(ids):
        chunk = _normalize(tokenizer.decode(ids[cursor:], skip_special_tokens=True))
        if chunk:
            parts.append(chunk)
    return _normalize(" ".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description="Print collated batch from data/test.jsonl.")
    parser.add_argument("--data-path", default="data/test.jsonl")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lm", default="Qwen/Qwen3-4B")
    args = parser.parse_args()

    lm_tok = AutoTokenizer.from_pretrained(args.lm, use_fast=True)

    _ensure_pad(lm_tok)
    judge_tok = lm_tok

    dataset = DIBJudgeDataset.from_jsonl(args.data_path)
    if len(dataset) == 0:
        raise SystemExit("No examples found in the dataset.")
    batch_items: List = [dataset[i] for i in range(min(args.batch_size, len(dataset)))]
    collator = DIBJudgeCollator(lm_tok)
    batch = collator(batch_items)

    print("Keys:", sorted(batch.keys()))
    for key, value in batch.items():
        if not hasattr(value, "shape"):
            print(f"{key}: type={type(value).__name__} value={value}")
            continue
        print(f"{key}: shape={tuple(value.shape)} dtype={value.dtype}")

    # Show small slices for sanity checking.
    print("\nSample slices:")
    for key in sorted(batch.keys()):
        tensor = batch[key]
        if not hasattr(tensor, "ndim"):
            print(f"{key}: {tensor}")
            continue
        if tensor.ndim == 1:
            sample = tensor[:40].tolist()
        elif tensor.ndim == 2:
            sample = tensor[0][:40].tolist()
        elif tensor.ndim == 3:
            sample = tensor[0][0][:40].tolist()
        else:
            sample = tensor.flatten()[:40].tolist()
        print(f"{key}[0][:40] {sample}")

    if "lm_response_types" in batch:
        # Decode response A/B spans based on lm_response_types.
        print("\nDecoded response spans:")
        lm_ids = batch["lm_input_ids"]
        resp_types = batch["lm_response_types"]
        for idx, ex in enumerate(batch_items):
            ids = lm_ids[idx].tolist()
            types = resp_types[idx].tolist()
            a_ids = [tid for tid, t in zip(ids, types) if t == 1]
            b_ids = [tid for tid, t in zip(ids, types) if t == 2]
            decoded_a = lm_tok.decode(a_ids, skip_special_tokens=True)
            decoded_b = lm_tok.decode(b_ids, skip_special_tokens=True)
            norm_a = _normalize(decoded_a)
            norm_b = _normalize(decoded_b)
            ref_a = _normalize(ex.response_a)
            ref_b = _normalize(ex.response_b)
            print(f"\nExample {idx}:")
            print("response_a:", ref_a[:200])
            print("decoded_a :", norm_a[:200])
            print("match_a  :", ref_a in norm_a or norm_a in ref_a)
            print("response_b:", ref_b[:200])
            print("decoded_b :", norm_b[:200])
            print("match_b  :", ref_b in norm_b or norm_b in ref_b)
    else:
        print("\nDecoded response spans: (lm_response_types not available)")

    print("\nEOS label check:")
    eos_id = lm_tok.eos_token_id
    if eos_id is None:
        print("Tokenizer has no eos_token_id; cannot check EOS in labels.")
    else:
        lm_labels = batch.get("lm_labels")
        lm_input_ids = batch.get("lm_input_ids")
        lm_attention = batch.get("lm_attention_mask")
        if lm_labels is None or lm_input_ids is None:
            print("lm_labels or lm_input_ids missing; cannot check EOS in labels.")
        else:
            for idx in range(min(len(batch_items), lm_labels.size(0))):
                labels = lm_labels[idx]
                input_ids = lm_input_ids[idx]
                label_positions = (labels != -100).nonzero(as_tuple=False).view(-1).tolist()
                if not label_positions:
                    print(f"Example {idx}: no supervised labels.")
                    continue
                label_ids = labels[label_positions].tolist()
                last_pos = label_positions[-1]
                last_label_id = labels[last_pos].item()
                last_input_id = input_ids[last_pos].item()
                eos_in_labels = eos_id in label_ids
                last_is_eos = last_label_id == eos_id
                if lm_attention is not None:
                    active_len = int(lm_attention[idx].sum().item())
                else:
                    active_len = len(input_ids)
                print(
                    f"Example {idx}: labels={len(label_ids)} active_len={active_len} "
                    f"eos_in_labels={eos_in_labels} last_label_is_eos={last_is_eos} "
                    f"last_label_id={last_label_id} last_input_id={last_input_id}"
                    f"Tok deocde lm_labels: {lm_tok.decode(label_ids)}"
                )

    print("\nOriginal responses (judge tokenizer):")
    original_ids = batch["original_input_ids"]
    original_mask = batch["original_attention_mask"]
    for idx in range(min(len(batch_items), original_ids.size(0))):
        print(f"\nExample {idx}:")
        for resp_idx in range(2):
            ids = original_ids[idx, resp_idx].tolist()
            mask = original_mask[idx, resp_idx].tolist()
            tok_ids = [tid for tid, m in zip(ids, mask) if m == 1]
            decoded = judge_tok.decode(tok_ids, skip_special_tokens=True)
            print(f"original_resp_{resp_idx}:", _normalize(decoded)[:200])

    print("\nShuffle span comparison (marked spans show shuffled/perturbed tokens):")
    shuffle_ids = batch["shuffle_input_ids"]
    shuffle_mask = batch["shuffle_attention_mask"]
    shuffle_labels = batch["shuffle_labels"]
    max_spans = 4
    for idx in range(min(len(batch_items), shuffle_ids.size(0))):
        print(f"\nExample {idx}:")
        for resp_idx in range(2):
            ids_before = original_ids[idx, resp_idx].tolist()
            ids_after = shuffle_ids[idx, resp_idx].tolist()
            mask = shuffle_mask[idx, resp_idx].tolist()
            labels = shuffle_labels[idx, resp_idx].tolist()
            length = sum(mask)
            ids_before = ids_before[:length]
            ids_after = ids_after[:length]
            labels = labels[:length]
            spans = _collect_spans(labels, target=0)
            if not spans:
                print(f"resp_{resp_idx}: <no shuffled spans>")
                continue
            before_marked = _decode_marked(judge_tok, ids_before, labels)
            after_marked = _decode_marked(judge_tok, ids_after, labels)
            print(f"resp_{resp_idx}_before:", before_marked[:400])
            print(f"resp_{resp_idx}_after :", after_marked[:400])
            for span_idx, (start, end) in enumerate(spans[:max_spans]):
                before_span = _normalize(judge_tok.decode(ids_before[start:end], skip_special_tokens=True))
                after_span = _normalize(judge_tok.decode(ids_after[start:end], skip_special_tokens=True))
                print(
                    f"  span_{span_idx}: "
                    f"before='{before_span[:160]}' after='{after_span[:160]}'"
                )
            if len(spans) > max_spans:
                print(f"  ... {len(spans) - max_spans} more spans")

    print("\nShuffle visualization (judge tokenizer tokens):")
    for idx in range(min(len(batch_items), shuffle_ids.size(0))):
        print(f"\nExample {idx}:")
        for resp_idx in range(2):
            ids = shuffle_ids[idx, resp_idx].tolist()
            mask = shuffle_mask[idx, resp_idx].tolist()
            tok_ids = [tid for tid, m in zip(ids, mask) if m == 1]
            if not tok_ids:
                print(f"shuffle_resp_{resp_idx}: <empty>")
                continue
            decoded = judge_tok.decode(tok_ids, skip_special_tokens=True)
            print(f"shuffle_resp_{resp_idx}:", _normalize(decoded)[:200])


if __name__ == "__main__":
    main()
