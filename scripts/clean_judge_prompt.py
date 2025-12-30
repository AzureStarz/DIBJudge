import argparse
import json
import os
import re
import sys
from pathlib import Path

DEFAULT_INPUT = Path(__file__).with_name("m_preference_collection_50k.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove 'a reference answer,' from judge_prompt fields in JSONL.",
    )
    parser.add_argument(
        "-i",
        "--input",
        default=str(DEFAULT_INPUT),
        help="Input JSONL path.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output JSONL path. Defaults to <input>_no_ref.jsonl.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input file in place.",
    )
    parser.add_argument(
        "--skip-bad-lines",
        action="store_true",
        help="Skip lines that are not valid JSON.",
    )
    return parser.parse_args()


def process_record(data: dict) -> dict:
    new_data = dict(data)
    judge_prompt = new_data.get("judge_prompt")
    if isinstance(judge_prompt, str):
        cleaned = judge_prompt.replace("a reference answer, ", "")
        cleaned = re.sub(r"###Evaluation Criteria:\n.*?\n\n", "", cleaned, flags=re.DOTALL)
        new_data["judge_prompt"] = cleaned
    return new_data


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 2

    if args.in_place and args.output:
        print("Use either --in-place or --output, not both.", file=sys.stderr)
        return 2

    if args.in_place:
        output_path = input_path.with_suffix(input_path.suffix + ".tmp")
    elif args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(f"{input_path.stem}_no_ref.jsonl")

    processed = 0
    skipped = 0

    completed = False
    try:
        with input_path.open("r", encoding="utf-8") as f_in, output_path.open(
            "w", encoding="utf-8"
        ) as f_out:
            for line_no, line in enumerate(f_in, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    data = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    if args.skip_bad_lines:
                        skipped += 1
                        print(
                            f"Skipping line {line_no}: {exc}",
                            file=sys.stderr,
                        )
                        continue
                    raise ValueError(
                        f"{input_path}:{line_no}: invalid JSON: {exc}"
                    ) from exc
                new_data = process_record(data)
                f_out.write(json.dumps(new_data, ensure_ascii=False))
                f_out.write("\n")
                processed += 1
        completed = True
    finally:
        if args.in_place and completed and output_path.exists():
            os.replace(output_path, input_path)

    if skipped:
        print(f"Processed {processed} lines, skipped {skipped} bad lines.", file=sys.stderr)
    else:
        print(f"Processed {processed} lines.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
