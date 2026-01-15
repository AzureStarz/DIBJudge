#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/precompute_proxy_metrics.sh INPUT_JSONL MODEL_PATH [OUTPUT_JSONL] [NUM_SHARDS]
#
# Example:
#   ./scripts/precompute_proxy_metrics.sh \
#     data/m_preference_collection_50k_qwen3-4b.jsonl \
#     /path/to/causal-lm \
#     data/m_preference_collection_50k_qwen3-4b.proxy.jsonl

cd DIBJUDGE_ROOT

source ~/online1/miniconda3/bin/activate sglang
source DIBJUDGE_ENV_FILE

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NVLS_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_P2P_LEVEL=LOC
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
PADDING_SIDE="${PADDING_SIDE:-left}"
DTYPE="${DTYPE:-bfloat16}"
ALLOW_TF32="${ALLOW_TF32:-1}"

module load amd/gcc_compiler/11.3.0
num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected ${num_gpus} GPUs"


if [[ $# -lt 2 ]]; then
  echo "Usage: $0 INPUT_JSONL MODEL_PATH [OUTPUT_JSONL] [NUM_SHARDS]" >&2
  exit 1
fi

INPUT_PATH="$1"
MODEL_PATH="$2"
OUTPUT_PATH="${3:-}"

IFS=',' read -r -a VISIBLE_DEVICES <<< "${CUDA_VISIBLE_DEVICES:-}"
if [[ $# -ge 4 ]]; then
  NUM_SHARDS="$4"
elif [[ ${#VISIBLE_DEVICES[@]} -gt 0 ]]; then
  NUM_SHARDS="${#VISIBLE_DEVICES[@]}"
else
  NUM_SHARDS="$num_gpus"
fi

if [[ -z "$OUTPUT_PATH" ]]; then
  OUTPUT_PATH="${INPUT_PATH%.*}.proxy.jsonl"
fi

if [[ "$NUM_SHARDS" -le 1 ]]; then
  python scripts/precompute_proxy_metrics.py \
    --input "$INPUT_PATH" \
    --lm "$MODEL_PATH" \
    --output "$OUTPUT_PATH" \
    --batch-size 4 \
    --max-length 16384 \
    --trust-remote-code \
    --attn-implementation "$ATTN_IMPL" \
    --padding-side "$PADDING_SIDE" \
    --dtype "$DTYPE" \
    $( [[ "$ALLOW_TF32" == "1" ]] && printf '%s' "--allow-tf32" )
  exit 0
fi

BASE_NO_EXT="${OUTPUT_PATH%.jsonl}"
SHARD_OUTPUTS=()

for ((i=0; i<NUM_SHARDS; i++)); do
  SHARD_OUT="${BASE_NO_EXT}.shard${i}.jsonl"
  SHARD_OUTPUTS+=("$SHARD_OUT")
  if [[ ${#VISIBLE_DEVICES[@]} -gt 0 ]]; then
    CUDA_VISIBLE_DEVICES="${VISIBLE_DEVICES[$i]}" \
      python scripts/precompute_proxy_metrics.py \
        --input "$INPUT_PATH" \
        --lm "$MODEL_PATH" \
        --output "$SHARD_OUT" \
        --batch-size 4 \
        --max-length 16384 \
        --trust-remote-code \
        --attn-implementation "$ATTN_IMPL" \
        --padding-side "$PADDING_SIDE" \
        --dtype "$DTYPE" \
        $( [[ "$ALLOW_TF32" == "1" ]] && printf '%s' "--allow-tf32" ) \
        --num-shards "$NUM_SHARDS" \
        --shard-index "$i" &
  else
    CUDA_VISIBLE_DEVICES="$i" \
      python scripts/precompute_proxy_metrics.py \
        --input "$INPUT_PATH" \
        --lm "$MODEL_PATH" \
        --output "$SHARD_OUT" \
        --batch-size 4 \
        --max-length 16384 \
        --trust-remote-code \
        --attn-implementation "$ATTN_IMPL" \
        --padding-side "$PADDING_SIDE" \
        --dtype "$DTYPE" \
        $( [[ "$ALLOW_TF32" == "1" ]] && printf '%s' "--allow-tf32" ) \
        --num-shards "$NUM_SHARDS" \
        --shard-index "$i" &
  fi
done

wait

cat "${SHARD_OUTPUTS[@]}" > "$OUTPUT_PATH"
echo "Wrote merged proxy cache to $OUTPUT_PATH"
