#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import Dataset, load_dataset

M_PREF_ROOT = Path(
    "data/raw/M-Preference-Collection"
)
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "m_preference_collection_50k_qwen3-4b.jsonl"

SAMPLE_SIZE = 50000
SEED = 42

REF_BLOCK_RE = re.compile(r"\n###Reference Answer:\s*.*?(?=\n###|\Z)", re.S)
EXTRACT_RE = re.compile(
    r"###Task Description:\s*(?P<task_description>[\s\S]*?)"
    r"###The instruction to evaluate:\s*(?P<instruction>[\s\S]*?)"
    r"###Response A to evaluate:\s*(?P<response_A>[\s\S]*?)"
    r"###Response B to evaluate:\s*(?P<response_B>[\s\S]*?)"
    # r"###Reference Answer:\s*(?P<reference_answer>[\s\S]*?)"
    r"###Evaluation Criteria:\s*(?P<evaluation_criteria>[\s\S]*?)"
    r"###Feedback:\s*(?P<feedback>[\s\S]*?)$"
)
TRANSLATION_PROMPT = (
    "Translate the following model response into English.\n"
    "Be careful to translate only the relevant content and avoid translating any elements that are not meant to be translated, such as mathematical expressions, formulas, or any input data like tables or code. Ensure that only the necessary parts are translated and that all other elements remain unchanged.\n"
    "Output only the translated text with no comments, explanations, or additional content."
)
BACK_TRANSLATION_PROMPT = (
    "Translate the following model response into the language identified by ISO 639-1 code '{language}'.\n"
    "Be careful to translate only the relevant content and avoid translating any elements that are not meant to be translated, such as mathematical expressions, formulas, or any input data like tables or code. Ensure that only the necessary parts are translated and that all other elements remain unchanged.\n"
    "Output only the translated text with no comments, explanations, or additional content."
)
SGLANG_MODEL = "Qwen/Qwen3-4B"

try:
    import sglang as sgl
except ImportError:
    sgl = None
try:
    import langid
except ImportError:
    langid = None
try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

_CHAT_TOKENIZER = None


def _list_parquet_files(root: Path) -> List[str]:
    files = sorted((root / "data").glob("train-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {root / 'data'}")
    return [str(f) for f in files]


def _is_null(value: Optional[object]) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return True
        if stripped.lower() in {"none", "null"}:
            return True
    return False


def _remove_reference_block(prompt: str) -> str:
    return REF_BLOCK_RE.sub("", prompt)


def _extract_parts(prompt: str) -> Optional[Tuple[str, str, str]]:
    if not prompt:
        return None
    match = EXTRACT_RE.search(prompt)
    if not match:
        return None
    return (
        match.group("instruction").strip(),
        match.group("response_A").strip(),
        match.group("response_B").strip(),
    )

def _reservoir_sample(dataset: Dataset, rng: random.Random) -> List[Dict[str, object]]:
    filtered = dataset.filter(
        lambda ex: _is_null(ex.get("orig_criteria")),
        desc="Filtering orig_criteria",
    )
    if len(filtered) < SAMPLE_SIZE:
        print(f"Warning: only {len(filtered)} examples matched the filter; returning {len(filtered)}.")
    seed = rng.randint(0, 2**32 - 1)
    shuffled = filtered.shuffle(seed=seed)
    take = min(SAMPLE_SIZE, len(shuffled))
    return list(shuffled.select(range(take)))


def _init_sglang_translator():
    if sgl is None:
        raise RuntimeError("sglang is not installed; cannot run offline translation.")
    llm = sgl.Engine(model_path=SGLANG_MODEL, tp_size=4, mem_fraction_static=0.7, cuda_graph_max_bs=32)
    return llm


def _get_chat_tokenizer(llm):
    global _CHAT_TOKENIZER
    if _CHAT_TOKENIZER is not None:
        return _CHAT_TOKENIZER
    tokenizer = getattr(llm, "tokenizer", None)
    if tokenizer is None and hasattr(llm, "get_tokenizer"):
        tokenizer = llm.get_tokenizer()
    if tokenizer is None:
        if AutoTokenizer is None:
            raise RuntimeError("transformers is not installed; cannot load chat tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(SGLANG_MODEL)
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError("Tokenizer with apply_chat_template is required for chat prompts.")
    _CHAT_TOKENIZER = tokenizer
    return tokenizer


def _format_chat_prompt(llm, content: str) -> str:
    tokenizer = _get_chat_tokenizer(llm)
    messages = [{"role": "user", "content": content}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False)


def _translate_texts_batch(llm, texts: List[str]) -> List[str]:
    translations = ["" for _ in texts]
    if not texts:
        return translations

    prompts = [_format_chat_prompt(llm, f"{TRANSLATION_PROMPT}\n\n{text}") for text in texts]
    outputs = llm.generate(prompts, {"temperature": 0.9, "top_p": 0.95, "max_new_tokens": 8192})
    if len(outputs) != len(prompts):
        raise RuntimeError("sglang batch translation count mismatch.")
    for idx, out in enumerate(outputs):
        # print("===============================")
        # print(f"Prompt: {prompts[idx]}\nGenerated text: {out.get('text', '')}")
        translations[idx] = str(out.get("text", "")).strip()
    return translations


def _detect_language(text: str) -> str:
    if not text:
        return ""
    if langid is None:
        raise RuntimeError("langid is not installed; cannot run language detection.")
    lang, _score = langid.classify(text)
    return lang


def _back_translate_texts_batch(llm, texts: List[str], languages: List[str]) -> List[str]:
    if len(texts) != len(languages):
        raise ValueError("texts and languages must be the same length for back translation.")

    translations = ["" for _ in texts]
    if not texts:
        return translations

    for idx, language in enumerate(languages):
        if not language:
            raise ValueError(f"Missing language for back translation at index {idx}.")

    prompts = [
        _format_chat_prompt(llm, f"{BACK_TRANSLATION_PROMPT.format(language=language)}\n\n{text}")
        for text, language in zip(texts, languages)
    ]
    outputs = llm.generate(prompts, {"temperature": 0.9, "top_p": 0.95, "max_new_tokens": 8192})
    if len(outputs) != len(prompts):
        raise RuntimeError("sglang batch translation count mismatch.")
    for idx, out in enumerate(outputs):
        # print("===============================")
        # print(f"Prompt: {prompts[idx]}\nGenerated text: {out.get('text', '')}")
        translations[idx] = str(out.get("text", "")).strip()
    return translations


def main() -> None:
    rng_sample = random.Random(SEED)
    m_files = _list_parquet_files(M_PREF_ROOT)
    m_dataset = load_dataset("parquet", data_files=m_files, split="train")
    samples = _reservoir_sample(m_dataset, rng_sample)

    llm = _init_sglang_translator()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    processed: List[Dict[str, object]] = []
    responses: List[str] = []
    response_languages: List[str] = []
    for ex in samples:
        prompt = str(ex.get("instruction") or "")
        cleaned_prompt = _remove_reference_block(prompt)
        extracted = _extract_parts(cleaned_prompt)
        if extracted is None:
            instruction = str(ex.get("orig_instruction") or "").strip()
            response_a = str(ex.get("orig_response_A") or "").strip()
            response_b = str(ex.get("orig_response_B") or "").strip()
        else:
            instruction, response_a, response_b = extracted
        judge_prompt = cleaned_prompt

        responses.extend([response_a, response_b])
        response_languages.extend([_detect_language(response_a), _detect_language(response_b)])
        processed.append(
            {
                "response_A_eng": "",
                "response_B_eng": "",
                "response_A_bt": "",
                "response_B_bt": "",
                "instruction": instruction,
                "response_A": response_a,
                "response_B": response_b,
                "judge_prompt": judge_prompt,
                "output": str(ex.get("output") or ""),
            }
        )

    translated = _translate_texts_batch(llm, responses)
    back_translated = _back_translate_texts_batch(llm, translated, response_languages)
    for idx, item in enumerate(processed):
        item["response_A_eng"] = translated[2 * idx]
        item["response_B_eng"] = translated[2 * idx + 1]
        item["response_A_bt"] = back_translated[2 * idx]
        item["response_B_bt"] = back_translated[2 * idx + 1]

    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for record in processed:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(samples)} samples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
