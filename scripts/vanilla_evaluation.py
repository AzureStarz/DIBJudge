#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import json
import os
import random
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from datasets import load_dataset


DEFAULT_SYSTEM_PROMPT = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. "
    "The question provided is in {src_lang}. "
    "You should choose the assistant that follows the user's instructions and answers the user's question better. "
    "Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. "
    "Also, make sure that the assistant responses are in {tgt_lang}. "
    "Begin your evaluation by comparing the two responses and provide a short explanation. "
    "Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. "
    "Do not allow the length of the responses to influence your evaluation. "
    "Do not favor certain names of the assistants. "
    "Be as objective as possible. "
    "After providing your explanation, output your final verdict by strictly following this format: "
    "\"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better."
)

DEFAULT_USER_PROMPT = (
    "[User Question]\n{question}\n\n"
    "[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n"
    "[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]"
)

DEFAULT_VERDICT_A = "[[A]]"
DEFAULT_VERDICT_B = "[[B]]"


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _chunked(items: Sequence[str], size: int) -> Iterable[List[str]]:
    if size <= 0:
        yield list(items)
        return
    for idx in range(0, len(items), size):
        yield list(items[idx : idx + size])


def _format_table(rows: List[List[str]], headers: List[str]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    line = "+".join("-" * (w + 2) for w in widths)
    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    out = [line, f" {header_line} ", line]
    for row in rows:
        out.append(" " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " ")
    out.append(line)
    return "\n".join(out)


def _parse_verdict(text: str, pattern_a: str, pattern_b: str) -> Optional[str]:
    if text is None:
        return None
    idx_a = text.find(pattern_a) if pattern_a else -1
    idx_b = text.find(pattern_b) if pattern_b else -1
    if idx_a == -1 and idx_b == -1:
        lower = text.lower()
        idx_a = lower.find(pattern_a.lower()) if pattern_a else -1
        idx_b = lower.find(pattern_b.lower()) if pattern_b else -1
    if idx_a == -1 and idx_b == -1:
        return None
    if idx_a != -1 and idx_b != -1:
        return "A" if idx_a < idx_b else "B"
    return "A" if idx_a != -1 else "B"


def _load_template(
    template_path: Optional[str],
    template_name: Optional[str],
    verdict_a: Optional[str],
    verdict_b: Optional[str],
) -> Tuple[Optional[str], str, str]:
    template = None
    config_verdict_a = None
    config_verdict_b = None
    if template_name:
        candidate = template_name
        if not os.path.isfile(candidate):
            base = template_path or ""
            if os.path.isdir(base):
                candidate = os.path.join(base, template_name)
            if not candidate.endswith(".json"):
                candidate = candidate + ".json"
        with open(candidate, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        template = data.get("template")
        config_verdict_a = data.get("verdict_answer_A_pattern")
        config_verdict_b = data.get("verdict_answer_B_pattern")
    verdict_a = verdict_a or config_verdict_a or DEFAULT_VERDICT_A
    verdict_b = verdict_b or config_verdict_b or DEFAULT_VERDICT_B
    return template, verdict_a, verdict_b


def _render_default_prompt(
    tokenizer,
    question: str,
    answer_a: str,
    answer_b: str,
    src_lang: str,
    tgt_lang: str,
) -> str:
    system_prompt = DEFAULT_SYSTEM_PROMPT.format(src_lang=src_lang, tgt_lang=tgt_lang)
    user_prompt = DEFAULT_USER_PROMPT.format(
        question=question, answer_a=answer_a, answer_b=answer_b
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if tokenizer is not None:
        apply = getattr(tokenizer, "apply_chat_template", None)
        if callable(apply):
            try:
                return apply(messages, tokenize=False, add_generation_prompt=True)
            except TypeError:
                return apply(messages, tokenize=False)
    return system_prompt + "\n\n" + user_prompt


def _render_template_prompt(
    template: str,
    question: str,
    answer_a: str,
    answer_b: str,
    src_lang: str,
    tgt_lang: str,
) -> str:
    payload = _SafeDict(
        {
            "prompt": question,
            "question": question,
            "response_a": answer_a,
            "response_b": answer_b,
            "answer_a": answer_a,
            "answer_b": answer_b,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
        }
    )
    return template.format_map(payload)


def _load_mm_eval(local_dir: str):
    # parquet_files = glob.glob(os.path.join(local_dir, "data", "test-*.parquet"))
    # if parquet_files:
    #     return load_dataset(
    #         "parquet", data_files={"test": parquet_files}, split="test"
    #     )
    # return load_dataset("prometheus-eval/MM-Eval", split="test")
    return load_dataset(local_dir, split="test")


def _load_mreward_language(local_dir: str, language: str):
    # lang_dir = os.path.join(local_dir, language)
    # if os.path.isdir(lang_dir):
    #     candidate = os.path.join(lang_dir, "filtered.json")
    #     if not os.path.isfile(candidate):
    #         candidate = os.path.join(lang_dir, "raw.json")
    #     if os.path.isfile(candidate):
    #         return load_dataset("json", data_files={"test": candidate}, split="test")
    # return load_dataset(
    #     "CohereLabsCommunity/multilingual-reward-bench", language, split="test"
    # )
    return load_dataset(local_dir, language, split="test")


def _group_by_language(dataset) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in dataset:
        lang = str(row.get("language", "unknown"))
        grouped[lang].append(row)
    return grouped


def _filter_mm_eval_core_languages(grouped: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    core = {lang: rows for lang, rows in grouped.items() if "_" not in lang}
    return core


def _available_mreward_languages(local_dir: str) -> List[str]:
    if os.path.isdir(local_dir):
        langs: List[str] = []
        for name in os.listdir(local_dir):
            if name.startswith(".") or name in {"translation"}:
                continue
            full = os.path.join(local_dir, name)
            if not os.path.isdir(full):
                continue
            filtered = os.path.join(full, "filtered.json")
            raw = os.path.join(full, "raw.json")
            parquet = glob.glob(os.path.join(full, "test-*.parquet"))
            if os.path.isfile(filtered) or os.path.isfile(raw) or parquet:
                langs.append(name)
        return sorted(langs)
    return []


SHORT_TO_CONFIG = {
    "ar": "arb_Arab",
    "cs": "ces_Latn",
    "de": "deu_Latn",
    "el": "ell_Grek",
    # "en": "eng_Latn",
    "fr": "fra_Latn",
    "he": "heb_Hebr",
    "hi": "hin_Deva",
    "id": "ind_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "nl": "nld_Latn",
    "fa": "pes_Arab",
    "pl": "pol_Latn",
    "pt": "por_Latn",
    "ro": "ron_Latn",
    "ru": "rus_Cyrl",
    "es": "spa_Latn",
    "tr": "tur_Latn",
    "uk": "ukr_Cyrl",
    "vi": "vie_Latn",
    "zh": "zho_Hans",
}

CONFIG_TO_SHORT = {value: key for key, value in SHORT_TO_CONFIG.items()}


def _resolve_mreward_languages(
    requested: Optional[List[str]], available: List[str]
) -> Tuple[List[str], List[str]]:
    if not requested:
        return available, []
    resolved: List[str] = []
    missing: List[str] = []
    available_set = set(available)
    for lang in requested:
        if lang in available_set:
            resolved.append(lang)
            continue
        mapped = SHORT_TO_CONFIG.get(lang)
        if mapped and mapped in available_set:
            resolved.append(mapped)
            continue
        missing.append(lang)
    return resolved, missing


def _build_task(
    row: dict,
    template: Optional[str],
    tokenizer,
    rng: random.Random,
    benchmark: str,
    language_override: Optional[str] = None,
) -> Tuple[str, dict]:
    question = str(row.get("prompt", ""))
    chosen = str(row.get("chosen", ""))
    rejected = str(row.get("rejected", ""))
    swap = rng.random() < 0.5
    if swap:
        answer_a = rejected
        answer_b = chosen
        expected = "B"
    else:
        answer_a = chosen
        answer_b = rejected
        expected = "A"
    src_lang = str(language_override or row.get("language", ""))
    tgt_lang = src_lang
    if template:
        prompt = _render_template_prompt(
            template, question, answer_a, answer_b, src_lang, tgt_lang
        )
    else:
        prompt = _render_default_prompt(
            tokenizer, question, answer_a, answer_b, src_lang, tgt_lang
        )
    meta = {
        "benchmark": benchmark,
        "id": row.get("id"),
        "language": src_lang,
        "prompt": question,
        "answer_a": answer_a,
        "answer_b": answer_b,
        "chosen_model": row.get("chosen_model"),
        "rejected_model": row.get("rejected_model"),
        "expected_verdict": expected,
        "swapped": swap,
    }
    return prompt, meta


def _run_vllm(
    model: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: Optional[int],
    trust_remote_code: bool,
) -> Tuple["LLM", object]:
    from vllm import LLM

    llm = LLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=trust_remote_code,
    )
    return llm, llm.get_tokenizer()


def _shutdown_vllm(llm) -> None:
    if llm is None:
        return
    for attr in ("shutdown", "close"):
        fn = getattr(llm, attr, None)
        if callable(fn):
            try:
                fn()
                return
            except Exception:
                pass
    engine = getattr(llm, "engine", None) or getattr(llm, "_engine", None)
    if engine is None:
        return
    for attr in ("shutdown", "close"):
        fn = getattr(engine, attr, None)
        if callable(fn):
            try:
                fn()
                return
            except Exception:
                pass


def _generate_vllm(
    llm: "LLM",
    prompts: List[str],
    batch_size: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> List[str]:
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k if top_k is not None else -1,
    )
    outputs: List[str] = []
    for chunk in _chunked(prompts, batch_size):
        results = llm.generate(chunk, sampling_params)
        for out in results:
            if not out.outputs:
                outputs.append("")
            else:
                outputs.append(out.outputs[0].text)
    return outputs


def _init_transformers(
    model: str,
    trust_remote_code: bool,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_obj = AutoModelForCausalLM.from_pretrained(
        model, dtype=torch.float16, device_map="auto", trust_remote_code=trust_remote_code
    )
    model_obj.eval()
    return model_obj, tokenizer


def _generate_transformers(
    model_obj,
    tokenizer,
    prompts: List[str],
    batch_size: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
):
    import torch

    outputs: List[str] = []
    with torch.inference_mode():
        for chunk in _chunked(prompts, batch_size):
            enc = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(model_obj.device)
            gen = model_obj.generate(
                **enc,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k if top_k is not None else 0,
            )
            text = tokenizer.batch_decode(
                gen[:, enc["input_ids"].size(1) :], skip_special_tokens=True
            )
            outputs.extend(text)
    return outputs


def _build_records(
    tasks: List[dict],
    completions: List[str],
    verdict_a: str,
    verdict_b: str,
) -> Tuple[Dict[Tuple[str, str], List[dict]], Dict[Tuple[str, str], List[dict]]]:
    raw_by_group: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    parsed_by_group: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for idx, completion in enumerate(completions):
        payload = tasks[idx]
        verdict = _parse_verdict(completion, verdict_a, verdict_b)
        expected = payload.get("expected_verdict")
        correct = verdict == expected
        group_key = (payload.get("benchmark", ""), payload.get("language", ""))
        raw_by_group[group_key].append(
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
        parsed_by_group[group_key].append(
            {
                "id": payload.get("id"),
                "language": payload.get("language"),
                "verdict": verdict,
                "expected_verdict": expected,
                "correct": bool(correct),
                "swapped": payload.get("swapped"),
            }
        )
    return raw_by_group, parsed_by_group


def _save_jsonl(path: str, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _result_paths(output_dir: str, benchmark: str, lang: str) -> Tuple[str, str]:
    if benchmark == "MM-Eval":
        prefix = "mm_eval"
    elif benchmark == "multilingual-reward-bench":
        prefix = "mreward"
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    raw_path = os.path.join(output_dir, f"{prefix}_{lang}_raw.jsonl")
    parsed_path = os.path.join(output_dir, f"{prefix}_{lang}_parsed.jsonl")
    return raw_path, parsed_path


def _parsed_from_raw(raw_rows: List[dict]) -> Optional[List[dict]]:
    parsed: List[dict] = []
    for row in raw_rows:
        verdict = row.get("verdict")
        expected = row.get("expected_verdict")
        correct = row.get("correct")
        if correct is None and verdict is not None and expected is not None:
            correct = verdict == expected
        if correct is None:
            return None
        parsed.append(
            {
                "id": row.get("id"),
                "language": row.get("language"),
                "verdict": verdict,
                "expected_verdict": expected,
                "correct": bool(correct),
                "swapped": row.get("swapped"),
            }
        )
    return parsed


def _load_existing_results(
    output_dir: str,
    expected_groups: List[Tuple[str, str]],
) -> Optional[Tuple[Dict[Tuple[str, str], List[dict]], Dict[Tuple[str, str], List[dict]]]]:
    raw_by_group: Dict[Tuple[str, str], List[dict]] = {}
    parsed_by_group: Dict[Tuple[str, str], List[dict]] = {}
    for benchmark, lang in expected_groups:
        raw_path, parsed_path = _result_paths(output_dir, benchmark, lang)
        raw_rows: Optional[List[dict]] = None
        parsed_rows: Optional[List[dict]] = None
        if os.path.isfile(raw_path):
            raw_rows = _load_jsonl(raw_path)
        if os.path.isfile(parsed_path):
            parsed_rows = _load_jsonl(parsed_path)
        if parsed_rows is None and raw_rows is not None:
            parsed_rows = _parsed_from_raw(raw_rows)
        if parsed_rows is None:
            return None
        parsed_by_group[(benchmark, lang)] = parsed_rows
        if raw_rows is not None:
            raw_by_group[(benchmark, lang)] = raw_rows
    return raw_by_group, parsed_by_group


def _load_partial_results(
    output_dir: str,
    expected_groups: List[Tuple[str, str]],
) -> Tuple[
    Dict[Tuple[str, str], List[dict]],
    Dict[Tuple[str, str], List[dict]],
    List[Tuple[str, str]],
]:
    raw_by_group: Dict[Tuple[str, str], List[dict]] = {}
    parsed_by_group: Dict[Tuple[str, str], List[dict]] = {}
    missing: List[Tuple[str, str]] = []
    for benchmark, lang in expected_groups:
        raw_path, parsed_path = _result_paths(output_dir, benchmark, lang)
        raw_rows: Optional[List[dict]] = None
        parsed_rows: Optional[List[dict]] = None
        if os.path.isfile(raw_path):
            raw_rows = _load_jsonl(raw_path)
        if os.path.isfile(parsed_path):
            parsed_rows = _load_jsonl(parsed_path)
        if parsed_rows is None and raw_rows is not None:
            parsed_rows = _parsed_from_raw(raw_rows)
        if parsed_rows is None:
            missing.append((benchmark, lang))
            continue
        parsed_by_group[(benchmark, lang)] = parsed_rows
        if raw_rows is not None:
            raw_by_group[(benchmark, lang)] = raw_rows
    return raw_by_group, parsed_by_group, missing


def _summarize(parsed: List[dict]) -> Tuple[int, int, float]:
    total = len(parsed)
    correct = sum(1 for row in parsed if row.get("correct"))
    accuracy = float(correct) / total if total else 0.0
    return total, correct, accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Vanilla LLM evaluation with vLLM.")
    parser.add_argument("--model", required=True)
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
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reuse_results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing raw/parsed results in output_dir when available.",
    )
    args = parser.parse_args()

    template, verdict_a, verdict_b = _load_template(
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
        dataset = _load_mm_eval(mm_dir)
        grouped = _group_by_language(dataset)
        grouped = _filter_mm_eval_core_languages(grouped)
        available = sorted(grouped.keys())
        mm_selected = args.languages or available
        missing = [lang for lang in mm_selected if lang not in grouped]
        if missing:
            raise ValueError(f"MM-Eval missing languages: {', '.join(missing)}")
        mm_grouped = grouped
        expected_groups.extend([("MM-Eval", lang) for lang in mm_selected])

    if args.benchmark in {"multilingual-reward-bench", "both"}:
        mreward_dir = os.path.join("data", "eval_data", "multilingual-reward-bench")
        available = _available_mreward_languages(mreward_dir)
        if not available:
            available = sorted(SHORT_TO_CONFIG.values())
        selected, missing = _resolve_mreward_languages(args.languages, available)
        if missing:
            raise ValueError(
                "multilingual-reward-bench missing languages: " + ", ".join(missing)
            )
        mreward_pairs = [(lang, CONFIG_TO_SHORT.get(lang, lang)) for lang in selected]
        expected_groups.extend(
            [("multilingual-reward-bench", display) for _, display in mreward_pairs]
        )

    if not expected_groups:
        raise ValueError("No evaluation prompts were built for the selected benchmarks.")

    raw_by_group: Dict[Tuple[str, str], List[dict]] = {}
    parsed_by_group: Dict[Tuple[str, str], List[dict]] = {}
    missing_groups = expected_groups
    if args.reuse_results:
        raw_by_group, parsed_by_group, missing_groups = _load_partial_results(
            args.output_dir, expected_groups
        )
    missing_set = set(missing_groups)
    generated_groups: set[Tuple[str, str]] = set()

    if missing_groups:
        tokenizer = None
        generate_fn = None
        llm_handle = None
        try:
            if args.use_vllm:
                llm_handle, tokenizer = _run_vllm(
                    args.model,
                    args.tensor_parallel_size,
                    args.gpu_memory_utilization,
                    args.max_model_len,
                    args.trust_remote_code,
                )

                def _generate(
                    prompts: List[str],
                    batch_size: int,
                    max_tokens: int,
                    temperature: float,
                    top_p: float,
                    top_k: int,
                ) -> List[str]:
                    return _generate_vllm(
                        llm_handle,
                        prompts,
                        batch_size,
                        max_tokens,
                        temperature,
                        top_p,
                        top_k,
                    )

                generate_fn = _generate
            else:
                model_obj, tokenizer = _init_transformers(
                    args.model, trust_remote_code=args.trust_remote_code
                )

                def _generate(
                    prompts: List[str],
                    batch_size: int,
                    max_tokens: int,
                    temperature: float,
                    top_p: float,
                    top_k: int,
                ) -> List[str]:
                    return _generate_transformers(
                        model_obj,
                        tokenizer,
                        prompts,
                        batch_size,
                        max_tokens,
                        temperature,
                        top_p,
                        top_k,
                    )

                generate_fn = _generate

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
                        prompt, meta = _build_task(
                            row,
                            template,
                            tokenizer,
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
                    if lang == "eng_Latn":
                        continue
                    dataset = _load_mreward_language(mreward_dir, lang)
                    rows = list(dataset)
                    if args.limit:
                        rows = rows[: args.limit]
                    for row in rows:
                        prompt, meta = _build_task(
                            row,
                            template,
                            tokenizer,
                            rng,
                            "multilingual-reward-bench",
                            language_override=display_lang,
                        )
                        prompts.append(prompt)
                        tasks.append(meta)

            completions = generate_fn(
                prompts,
                args.batch_size,
                args.max_tokens,
                args.temperature,
                args.top_p,
                args.top_k,
            )
            if len(completions) != len(tasks):
                raise RuntimeError(
                    f"Expected {len(tasks)} completions, received {len(completions)}."
                )

            new_raw, new_parsed = _build_records(
                tasks, completions, verdict_a, verdict_b
            )
            raw_by_group.update(new_raw)
            parsed_by_group.update(new_parsed)
            generated_groups = set(missing_groups)
        finally:
            if args.use_vllm:
                _shutdown_vllm(llm_handle)

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
                _save_jsonl(raw_path, raw)
                _save_jsonl(parsed_path, parsed)
            total, correct, acc = _summarize(parsed)
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
                raw_path = os.path.join(
                    args.output_dir, f"mreward_{display_lang}_raw.jsonl"
                )
                parsed_path = os.path.join(
                    args.output_dir, f"mreward_{display_lang}_parsed.jsonl"
                )
                _save_jsonl(raw_path, raw)
                _save_jsonl(parsed_path, parsed)
            total, correct, acc = _summarize(parsed)
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
    print(_format_table(table_rows, headers))

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
    print(f"Summary: ✅ {overall_correct} pass / ❌ {overall_total - overall_correct} fail")
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
