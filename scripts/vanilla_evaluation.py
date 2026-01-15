#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import re
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from datasets import load_dataset
import eval_stats


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
QWEN3_SYSTEM_PREFIX = "You are Qwen3, an AI assistant that reasons, thinks, and answers strictly in English."
_PRINTED_TEST_PROMPT = False


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
) -> Tuple[Optional[Union[str, Dict[str, object]]], str, str]:
    template: Optional[Union[str, Dict[str, object]]] = None
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
        if template is None and isinstance(data, dict) and "general" in data:
            template = data
        if template is not None:
            config_verdict_a = data.get("verdict_answer_A_pattern")
            config_verdict_b = data.get("verdict_answer_B_pattern")
    if template is not None and config_verdict_a is None and config_verdict_b is None:
        if isinstance(template, dict) and "schema" in template:
            config_verdict_a = "\"score\": \"Assistant A\""
            config_verdict_b = "\"score\": \"Assistant B\""
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
                return apply(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
            except TypeError:
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
    try:
        return template.format_map(payload)
    except ValueError:
        return _safe_format_template(template, payload)


def _safe_format_template(template: str, payload: Dict[str, object]) -> str:
    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in payload:
            return str(payload[key])
        return match.group(0)

    return re.sub(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", _replace, template)


def _get_reproducible_rubric_str(
    rubric_list: Sequence[object],
    row_id: object,
    seed_value: Optional[object] = None,
) -> str:
    if not rubric_list:
        return ""
    if seed_value is None:
        seed_text = str(row_id)
    else:
        seed_text = f"{seed_value}:{row_id}"
    digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    idx = int(digest, 16) % len(rubric_list)
    rubric = rubric_list[idx]
    if isinstance(rubric, dict):
        preferred = ["Assistant A", "Assistant B", "Tie", "Both Bad"]
        keys = [k for k in preferred if k in rubric]
        keys.extend([k for k in rubric.keys() if k not in keys])
        return "\n".join(f"{key}: {rubric[key]}" for key in keys)
    if isinstance(rubric, (list, tuple)):
        return "\n".join(str(item) for item in rubric)
    return str(rubric)


def _select_mr3_template_type(
    templated_dict: Dict[str, object],
    benchmark: str,
    row: dict,
) -> str:
    if benchmark == "multilingual-reward-bench":
        data_source = str(row.get("category") or row.get("subset") or "")
    else:
        data_source = str(row.get("subset") or "")
    data_source = data_source.strip().lower()
    if "chat" in data_source:
        return "general"
    if data_source and data_source in templated_dict:
        return data_source
    for key in templated_dict:
        if key in {"schema", "tags", "general"}:
            continue
        if key in data_source:
            return key
    return "general"


def _render_mr3_prompt(
    tokenizer,
    templated_dict: Dict[str, object],
    template_type: str,
    question: str,
    answer_a: str,
    answer_b: str,
    row_id: object,
    seed_value: Optional[object],
) -> str:
    instruction_msg = str(templated_dict.get("instruction_msg") or "Instruction")
    rubric_list = templated_dict.get(template_type, {}).get("rubric_list", [])
    shuffled_rubric = _get_reproducible_rubric_str(
        rubric_list, row_id, seed_value=seed_value
    )
    task_description = templated_dict.get(template_type, {}).get("task_description", "")
    tags = templated_dict.get("tags", {})
    schema = templated_dict.get(template_type, {}).get("schema")
    if not schema:
        schema = templated_dict.get("schema", {})
    developer_text = (
        f"{QWEN3_SYSTEM_PREFIX}\n\n"
        f"# {instruction_msg}\n"
        f"{task_description}\n\n"
        f"# {tags.get('evaluation_rubric_tag', 'Evaluation Rubric')}\n"
        f"{shuffled_rubric}\n\n"
        f"# {tags.get('response_format_tag', 'Response Format')}\n\n"
        f"{schema}"
    )
    user_text = (
        f"# {tags.get('input_tag', 'Input')}\n"
        f"{question}\n\n"
        "# Assistant A\n"
        f"{answer_a}\n\n"
        "# Assistant B\n"
        f"{answer_b}\n\n"
        f"# {tags.get('your_response_tag', 'Your Response')}\n"
    )
    messages = [
        {"role": "system", "content": developer_text},
        {"role": "user", "content": user_text},
    ]
    if tokenizer is not None:
        apply = getattr(tokenizer, "apply_chat_template", None)
        if callable(apply):
            try:
                return apply(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
            except TypeError:
                try:
                    return apply(messages, tokenize=False, add_generation_prompt=True)
                except TypeError:
                    return apply(messages, tokenize=False)
    return developer_text + "\n\n" + user_text


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
    template: Optional[Union[str, Dict[str, object]]],
    tokenizer,
    rng: random.Random,
    benchmark: str,
    language_override: Optional[str] = None,
    seed_value: Optional[int] = None,
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
        if isinstance(template, dict):
            template_type = _select_mr3_template_type(template, benchmark, row)
            prompt = _render_mr3_prompt(
                tokenizer,
                template,
                template_type,
                question,
                answer_a,
                answer_b,
                row.get("id"),
                seed_value,
            )
        else:
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
        "subset": row.get("subset"),
        "category": row.get("category"),
        "chosen_model": row.get("chosen_model"),
        "rejected_model": row.get("rejected_model"),
        "expected_verdict": expected,
        "swapped": swap,
    }
    global _PRINTED_TEST_PROMPT
    if not _PRINTED_TEST_PROMPT:
        print("[test] final_prompt:\n", prompt)
        _PRINTED_TEST_PROMPT = True
    return prompt, meta


def _write_prompt_log(path: str, prompts: List[str]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for idx, prompt in enumerate(prompts):
            payload = {"idx": idx, "prompt": prompt}
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _run_vllm(
    model: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: Optional[int],
    trust_remote_code: bool,
    seed: Optional[int] = None,
) -> Tuple["LLM", object]:
    from vllm import LLM

    kwargs = {
        "model": model,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "trust_remote_code": trust_remote_code,
    }
    if seed is not None:
        kwargs["seed"] = seed
    try:
        llm = LLM(**kwargs)
    except TypeError:
        kwargs.pop("seed", None)
        llm = LLM(**kwargs)
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
    seed: Optional[int] = None,
) -> List[str]:
    from vllm import SamplingParams

    sampling_kwargs = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k if top_k is not None else -1,
    }
    if seed is not None:
        sampling_kwargs["seed"] = seed
    try:
        sampling_params = SamplingParams(**sampling_kwargs)
    except TypeError:
        sampling_kwargs.pop("seed", None)
        sampling_params = SamplingParams(**sampling_kwargs)
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


def _init_tokenizer_only(model: str, trust_remote_code: bool):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _generate_transformers(
    model_obj,
    tokenizer,
    prompts: List[str],
    batch_size: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: Optional[int] = None,
):
    import torch

    outputs: List[str] = []
    with torch.inference_mode():
        if seed is not None:
            torch.manual_seed(seed)
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


def _summarize_with_ci(
    parsed: List[dict],
    bootstrap_samples: int,
    bootstrap_confidence: float,
    seed: int,
) -> Dict[str, object]:
    total, correct, accuracy = _summarize(parsed)
    summary: Dict[str, object] = {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
    }
    if bootstrap_samples > 0 and total > 0:
        rng = random.Random(seed)
        values = [1.0 if row.get("correct") else 0.0 for row in parsed]
        ci = eval_stats.bootstrap_mean_ci(
            values, bootstrap_samples, bootstrap_confidence, rng
        )
        if ci is not None:
            summary["accuracy_ci"] = {
                "low": ci[0],
                "high": ci[1],
                "confidence": bootstrap_confidence,
            }
    return summary


def _aggregate_seed_summaries(
    seed_summaries: List[dict],
    bootstrap_samples: int,
    bootstrap_confidence: float,
    bootstrap_seed: int,
) -> Dict[str, object]:
    aggregate: Dict[str, object] = {"seeds": [], "benchmarks": {}, "overall": {}}
    seeds = [int(summary.get("seed")) for summary in seed_summaries if "seed" in summary]
    aggregate["seeds"] = seeds

    bench_names = sorted(
        {
            bench
            for summary in seed_summaries
            for bench in summary.get("benchmarks", {}).keys()
        }
    )
    seed_counter = 0
    for bench in bench_names:
        lang_names = sorted(
            {
                lang
                for summary in seed_summaries
                for lang in summary.get("benchmarks", {}).get(bench, {}).keys()
            }
        )
        bench_out: Dict[str, object] = {}
        for lang in lang_names:
            accuracy_values: List[float] = []
            correct_values: List[float] = []
            totals: List[int] = []
            for summary in seed_summaries:
                metrics = summary.get("benchmarks", {}).get(bench, {}).get(lang)
                if metrics is None:
                    continue
                accuracy_values.append(float(metrics.get("accuracy", 0.0)))
                correct_values.append(float(metrics.get("correct", 0)))
                totals.append(int(metrics.get("total", 0)))
            if not accuracy_values:
                continue
            acc_mean, acc_std = eval_stats.mean_std(accuracy_values)
            corr_mean, corr_std = eval_stats.mean_std(correct_values)
            total = totals[0] if totals else 0
            entry: Dict[str, object] = {
                "total": total,
                "correct": {
                    "mean": corr_mean,
                    "std": corr_std,
                    "formatted": eval_stats.format_mean_std(corr_mean, corr_std),
                    "values": correct_values,
                },
                "accuracy": {
                    "mean": acc_mean,
                    "std": acc_std,
                    "formatted": eval_stats.format_mean_std(acc_mean, acc_std),
                    "values": accuracy_values,
                },
            }
            if bootstrap_samples > 0:
                rng = random.Random(bootstrap_seed + seed_counter)
                seed_counter += 1
                ci = eval_stats.bootstrap_mean_ci(
                    accuracy_values, bootstrap_samples, bootstrap_confidence, rng
                )
                if ci is not None:
                    entry["accuracy_mean_ci"] = {
                        "low": ci[0],
                        "high": ci[1],
                        "confidence": bootstrap_confidence,
                    }
            bench_out[lang] = entry
        if bench_out:
            aggregate["benchmarks"][bench] = bench_out

    overall_values = [
        float(summary.get("overall", {}).get("accuracy", 0.0))
        for summary in seed_summaries
        if summary.get("overall") is not None
    ]
    overall_totals = [
        int(summary.get("overall", {}).get("total", 0))
        for summary in seed_summaries
        if summary.get("overall") is not None
    ]
    overall_corrects = [
        float(summary.get("overall", {}).get("correct", 0))
        for summary in seed_summaries
        if summary.get("overall") is not None
    ]
    if overall_values:
        acc_mean, acc_std = eval_stats.mean_std(overall_values)
        corr_mean, corr_std = eval_stats.mean_std(overall_corrects)
        total = overall_totals[0] if overall_totals else 0
        aggregate["overall"] = {
            "total": total,
            "correct": {
                "mean": corr_mean,
                "std": corr_std,
                "formatted": eval_stats.format_mean_std(corr_mean, corr_std),
                "values": overall_corrects,
            },
            "accuracy": {
                "mean": acc_mean,
                "std": acc_std,
                "formatted": eval_stats.format_mean_std(acc_mean, acc_std),
                "values": overall_values,
            },
        }
        if bootstrap_samples > 0:
            rng = random.Random(bootstrap_seed + seed_counter)
            seed_counter += 1
            ci = eval_stats.bootstrap_mean_ci(
                overall_values, bootstrap_samples, bootstrap_confidence, rng
            )
            if ci is not None:
                aggregate["overall"]["accuracy_mean_ci"] = {
                    "low": ci[0],
                    "high": ci[1],
                    "confidence": bootstrap_confidence,
                }
    return aggregate


def _evaluate_seed(
    args: argparse.Namespace,
    seed: int,
    template: Optional[Union[str, Dict[str, object]]],
    verdict_a: str,
    verdict_b: str,
    mm_grouped: Optional[Dict[str, List[dict]]],
    mm_selected: List[str],
    mreward_pairs: List[Tuple[str, str]],
    expected_groups: List[Tuple[str, str]],
    tokenizer: Optional[object],
    generate_fn: Optional[
        Callable[
            [List[str], int, int, float, float, int, Optional[int]], List[str]
        ]
    ],
) -> dict:
    eval_stats.seed_everything(seed)
    seed_output_dir = os.path.join(args.output_dir, f"seed_{seed}")
    os.makedirs(seed_output_dir, exist_ok=True)

    summary = {"seed": seed, "benchmarks": {}}
    rng = random.Random(seed)

    raw_by_group: Dict[Tuple[str, str], List[dict]] = {}
    parsed_by_group: Dict[Tuple[str, str], List[dict]] = {}
    missing_groups = expected_groups
    if args.reuse_results:
        raw_by_group, parsed_by_group, missing_groups = _load_partial_results(
            seed_output_dir, expected_groups
        )
    missing_set = set(missing_groups)
    generated_groups: set[Tuple[str, str]] = set()

    if missing_groups:
        if tokenizer is None or generate_fn is None:
            raise ValueError("Generation requested but no model was initialized.")

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
                        seed_value=seed,
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
                        seed_value=seed,
                    )
                    prompts.append(prompt)
                    tasks.append(meta)

        if args.debug_vllm_prompts:
            prompt_log = os.path.join(seed_output_dir, "vllm_prompts.jsonl")
            _write_prompt_log(prompt_log, prompts)

        completions = generate_fn(
            prompts,
            args.batch_size,
            args.max_tokens,
            args.temperature,
            args.top_p,
            args.top_k,
            seed,
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

    write_outputs = bool(generated_groups)
    overall_flags: List[float] = []
    benchmark_flags: Dict[str, List[float]] = {"MM-Eval": [], "multilingual-reward-bench": []}

    if args.benchmark in {"MM-Eval", "both"}:
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
                _save_jsonl(raw_path, raw)
                _save_jsonl(parsed_path, parsed)
            stats = _summarize_with_ci(
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

    if args.benchmark in {"multilingual-reward-bench", "both"}:
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
                _save_jsonl(raw_path, raw)
                _save_jsonl(parsed_path, parsed)
            stats = _summarize_with_ci(
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


def _print_paired_tests(
    label: str,
    tests: Dict[str, Dict[str, float]],
    alpha: float,
) -> None:
    if not tests:
        return
    print("")
    print(f"Paired t-test vs {label} (alpha={alpha:.2f})")
    for metric, stats in tests.items():
        t_stat = stats.get("t_stat", float("nan"))
        p_value = stats.get("p_value", float("nan"))
        mean_diff = stats.get("mean_diff", float("nan"))
        n = int(stats.get("n", 0))
        sig = "significant" if p_value < alpha else "not significant"
        print(
            f"{metric}: t={t_stat:.4f}, p={p_value:.4f}, mean_diff={mean_diff:.4f} (n={n}, {sig})"
        )


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
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--seed-step", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--compare-dir", default=None)
    parser.add_argument("--compare-label", default="baseline")
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

    template, verdict_a, verdict_b = _load_template(
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

    if args.debug_prompt_only:
        tokenizer = None
        if args.model:
            try:
                tokenizer = _init_tokenizer_only(
                    args.model, trust_remote_code=args.trust_remote_code
                )
            except Exception as exc:
                print(f"[warn] failed to load tokenizer: {exc}")
                tokenizer = None
        global _PRINTED_TEST_PROMPT
        _PRINTED_TEST_PROMPT = True
        prompt = None
        rng = random.Random(seeds[0])
        if mm_grouped is not None:
            lang = mm_selected[0] if mm_selected else next(iter(mm_grouped.keys()))
            rows = mm_grouped[lang]
            if not rows:
                raise ValueError("MM-Eval has no rows to build a debug prompt.")
            prompt, _ = _build_task(
                rows[0],
                template,
                tokenizer,
                rng,
                "MM-Eval",
                language_override=lang,
                seed_value=seeds[0],
            )
        elif mreward_pairs:
            mreward_dir = os.path.join("data", "eval_data", "multilingual-reward-bench")
            lang, display_lang = mreward_pairs[0]
            dataset = _load_mreward_language(mreward_dir, lang)
            rows = list(dataset)
            if not rows:
                raise ValueError("multilingual-reward-bench has no rows to build a debug prompt.")
            prompt, _ = _build_task(
                rows[0],
                template,
                tokenizer,
                rng,
                "multilingual-reward-bench",
                language_override=display_lang,
                seed_value=seeds[0],
            )
        else:
            raise ValueError("No datasets loaded to build a debug prompt.")
        print(prompt)
        return

    any_missing = not args.reuse_results
    if args.reuse_results:
        for seed in seeds:
            seed_output_dir = os.path.join(args.output_dir, f"seed_{seed}")
            _, _, missing_groups = _load_partial_results(
                seed_output_dir, expected_groups
            )
            if missing_groups:
                any_missing = True
                break

    llm_handle = None
    model_obj = None
    tokenizer = None
    generate_fn = None
    if any_missing:
        if args.use_vllm:
            llm_handle, tokenizer = _run_vllm(
                args.model,
                args.tensor_parallel_size,
                args.gpu_memory_utilization,
                args.max_model_len,
                args.trust_remote_code,
                seed=seeds[0],
            )

            def _generate(
                prompts: List[str],
                batch_size: int,
                max_tokens: int,
                temperature: float,
                top_p: float,
                top_k: int,
                sample_seed: Optional[int],
            ) -> List[str]:
                return _generate_vllm(
                    llm_handle,
                    prompts,
                    batch_size,
                    max_tokens,
                    temperature,
                    top_p,
                    top_k,
                    seed=sample_seed,
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
                sample_seed: Optional[int],
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
                    seed=sample_seed,
                )

            generate_fn = _generate

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
            expected_groups,
            tokenizer,
            generate_fn,
        )
        seed_summaries.append(seed_summary)

    if args.use_vllm and llm_handle is not None:
        _shutdown_vllm(llm_handle)

    aggregate = _aggregate_seed_summaries(
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
    print(_format_table(table_rows, headers))

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
        _print_paired_tests(args.compare_label, tests, args.alpha)


if __name__ == "__main__":
    main()
