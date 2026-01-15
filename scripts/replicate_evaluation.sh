#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/replicate_evaluation.sh --model <replicate/model> [--prompt-key prompt] [--system-message-key system_instruction] [--] <extra args>
#
# Examples:
#   ./scripts/replicate_evaluation.sh --model google/gemini-2.5-flash -- --benchmark MM-Eval --limit 100
#   ./scripts/replicate_evaluation.sh --test

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${REPLICATE_MODEL:-}"
PROMPT_KEY="${PROMPT_KEY:-prompt}"
SYSTEM_MESSAGE_KEY="${SYSTEM_MESSAGE_KEY:-system_instruction}"
RUN_TEST=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
replicate_evaluation.sh

Required:
  --model <replicate/model>    Replicate model name (unless --test is used).

Optional:
  --prompt-key <key>           Input key for user prompt (default: prompt).
  --system-message-key <key>   Input key for system message (default: system_instruction).
  --test                       Run a small local sanity test and exit.
  --                           Pass remaining args directly to replicate_evaluation.py.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --prompt-key)
      PROMPT_KEY="$2"
      shift 2
      ;;
    --system-message-key)
      SYSTEM_MESSAGE_KEY="$2"
      shift 2
      ;;
    --test)
      RUN_TEST=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "$RUN_TEST" -eq 1 ]]; then
  TEST_MODEL="${MODEL:-google/gemini-2.5-flash}"
  echo "[test] Running py_compile..."
  "$PYTHON_BIN" -m py_compile scripts/replicate_evaluation.py
  echo "[test] Running debug prompt generation..."
  output="$("$PYTHON_BIN" scripts/replicate_evaluation.py \
    --model "$TEST_MODEL" \
    --benchmark MM-Eval \
    --limit 1 \
    --debug-prompt-only \
    --prompt-key "$PROMPT_KEY" \
    --system-message-key "$SYSTEM_MESSAGE_KEY")"
  if [[ "$output" != *"[User Question]"* ]]; then
    echo "[test] Missing user prompt marker in debug output." >&2
    exit 1
  fi
  if [[ "$output" != *"Please act as an impartial judge"* ]]; then
    echo "[test] Missing system prompt marker in debug output." >&2
    exit 1
  fi
  echo "[test] OK"
  exit 0
fi

if [[ -z "$MODEL" ]]; then
  echo "Missing --model. Use --help for usage." >&2
  exit 1
fi

if [[ -z "${REPLICATE_API_TOKEN:-}" ]]; then
  echo "REPLICATE_API_TOKEN is required to run evaluation." >&2
  exit 1
fi

"$PYTHON_BIN" scripts/replicate_evaluation.py \
  --model "$MODEL" \
  --prompt-key "$PROMPT_KEY" \
  --system-message-key "$SYSTEM_MESSAGE_KEY" \
  "${EXTRA_ARGS[@]}"
