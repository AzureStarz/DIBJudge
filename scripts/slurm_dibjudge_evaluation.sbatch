#!/bin/bash
#SBATCH --job-name=dibjudge_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  sbatch scripts/slurm_dibjudge_evaluation.sbatch -- [options] [-- dibjudge_evaluation.py args...]

Options:
  -m, --model PATH                 Model path or HF id (env: DIBJUDGE_MODEL)
  -c, --checkpoint PATH            DIBJudge checkpoint path (env: DIBJUDGE_CHECKPOINT)
  -o, --output-dir PATH            Output dir (env: DIBJUDGE_OUTPUT_DIR)
  -l, --languages CSV              Comma-separated language list (env: DIBJUDGE_LANGUAGES)
  -t, --template NAME              Prompt template (env: DIBJUDGE_TEMPLATE)
  -p, --tensor-parallel-size NUM   Tensor parallel size (env: DIBJUDGE_TENSOR_PARALLEL_SIZE)
  -h, --help                       Show this help

Environment:
  DIBJUDGE_ROOT=/path/to/DIBJudge
  DIBJUDGE_CONDA_ENV=dibjudge
  DIBJUDGE_ENV_FILE=/path/to/private_env.sh
  DIBJUDGE_MODULES="cuda/12 gcc/11"
USAGE
}

MODEL="${DIBJUDGE_MODEL:-outputs/quickstart/hf-epoch-1/lm}"
CHECKPOINT="${DIBJUDGE_CHECKPOINT:-outputs/quickstart/hf-epoch-1/dibjudge}"
OUTPUT_DIR="${DIBJUDGE_OUTPUT_DIR:-results/quickstart/dibjudge}"
LANGUAGES="${DIBJUDGE_LANGUAGES:-zh}"
TEMPLATE="${DIBJUDGE_TEMPLATE:-mr3_templates}"
TENSOR_PARALLEL_SIZE="${DIBJUDGE_TENSOR_PARALLEL_SIZE:-}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model) MODEL="$2"; shift 2 ;;
    --model=*) MODEL="${1#*=}"; shift ;;
    -c|--checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --checkpoint=*) CHECKPOINT="${1#*=}"; shift ;;
    -o|--output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --output-dir=*) OUTPUT_DIR="${1#*=}"; shift ;;
    -l|--languages) LANGUAGES="$2"; shift 2 ;;
    --languages=*) LANGUAGES="${1#*=}"; shift ;;
    -t|--template) TEMPLATE="$2"; shift 2 ;;
    --template=*) TEMPLATE="${1#*=}"; shift ;;
    -p|--tensor-parallel-size) TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
    --tensor-parallel-size=*) TENSOR_PARALLEL_SIZE="${1#*=}"; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

mkdir -p "${OUTPUT_DIR}" logs
unset CUDA_LAUNCH_BLOCKING NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_NVLS_DISABLE NCCL_IB_DISABLE
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"

if command -v nvidia-smi >/dev/null 2>&1; then
  NUM_GPUS="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
else
  NUM_GPUS="${SLURM_GPUS_ON_NODE:-1}"
fi
if [[ -z "${TENSOR_PARALLEL_SIZE}" ]]; then
  TENSOR_PARALLEL_SIZE="${NUM_GPUS}"
fi
IFS=',' read -r -a LANG_ARRAY <<< "${LANGUAGES}"

BASE_ARGS=(
  --model "${MODEL}"
  --checkpoint "${CHECKPOINT}"
  --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}"
  --output_dir "${OUTPUT_DIR}"
  --languages "${LANG_ARRAY[@]}"
  --template "${TEMPLATE}"
)

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  echo "Phase 1: distributed prompt embedding with ${NUM_GPUS} GPUs"
  torchrun --nproc_per_node="${NUM_GPUS}" scripts/dibjudge_evaluation.py \
    --embed-distributed \
    --embed-on-gpu \
    --embed-only \
    "${BASE_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"

  echo "Phase 2: vLLM generation from cached embeddings"
  python scripts/dibjudge_evaluation.py \
    --load-embed-shards \
    "${BASE_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
else
  python scripts/dibjudge_evaluation.py \
    --embed-on-gpu \
    "${BASE_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
fi
