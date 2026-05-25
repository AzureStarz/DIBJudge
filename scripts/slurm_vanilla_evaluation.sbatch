#!/bin/bash
#SBATCH --job-name=dibjudge_vanilla_eval
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
  sbatch scripts/slurm_vanilla_evaluation.sbatch -- [options] [-- vanilla_evaluation.py args...]

Options:
  -m, --model PATH                 Model path or HF id (env: VANILLA_MODEL)
  -o, --output-dir PATH            Output dir (env: VANILLA_OUTPUT_DIR)
  -l, --languages CSV              Comma-separated language list (env: VANILLA_LANGUAGES)
  -t, --template NAME              Template name/json path (env: VANILLA_TEMPLATE)
  -T, --template-path PATH         Template directory (env: VANILLA_TEMPLATE_PATH)
  -b, --benchmark NAME             MM-Eval|multilingual-reward-bench|judgment-requests|both|all
  -p, --tensor-parallel-size NUM   Tensor parallel size (env: VANILLA_TENSOR_PARALLEL_SIZE)
  -h, --help                       Show this help

Environment:
  DIBJUDGE_ROOT=/path/to/DIBJudge
  DIBJUDGE_CONDA_ENV=dibjudge
  DIBJUDGE_ENV_FILE=/path/to/private_env.sh
  DIBJUDGE_MODULES="cuda/12 gcc/11"
USAGE
}

MODEL="${VANILLA_MODEL:-Qwen/Qwen3-4B}"
OUTPUT_DIR="${VANILLA_OUTPUT_DIR:-results/quickstart/vanilla}"
LANGUAGES="${VANILLA_LANGUAGES:-}"
TEMPLATE="${VANILLA_TEMPLATE:-mr3_templates}"
TEMPLATE_PATH="${VANILLA_TEMPLATE_PATH:-configs/eval_config}"
BENCHMARK="${VANILLA_BENCHMARK:-}"
TENSOR_PARALLEL_SIZE="${VANILLA_TENSOR_PARALLEL_SIZE:-}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model) MODEL="$2"; shift 2 ;;
    --model=*) MODEL="${1#*=}"; shift ;;
    -o|--output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --output-dir=*) OUTPUT_DIR="${1#*=}"; shift ;;
    -l|--languages) LANGUAGES="$2"; shift 2 ;;
    --languages=*) LANGUAGES="${1#*=}"; shift ;;
    -t|--template) TEMPLATE="$2"; shift 2 ;;
    --template=*) TEMPLATE="${1#*=}"; shift ;;
    -T|--template-path) TEMPLATE_PATH="$2"; shift 2 ;;
    --template-path=*) TEMPLATE_PATH="${1#*=}"; shift ;;
    -b|--benchmark) BENCHMARK="$2"; shift 2 ;;
    --benchmark=*) BENCHMARK="${1#*=}"; shift ;;
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

CMD=(
  python scripts/vanilla_evaluation.py
  --model "${MODEL}"
  --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}"
  --output_dir "${OUTPUT_DIR}"
)
[[ -n "${TEMPLATE}" ]] && CMD+=(--template "${TEMPLATE}")
[[ -n "${TEMPLATE_PATH}" ]] && CMD+=(--template_path "${TEMPLATE_PATH}")
[[ -n "${BENCHMARK}" ]] && CMD+=(--benchmark "${BENCHMARK}")
if [[ -n "${LANGUAGES}" ]]; then
  IFS=',' read -r -a LANG_ARRAY <<< "${LANGUAGES}"
  CMD+=(--languages "${LANG_ARRAY[@]}")
fi
CMD+=("${EXTRA_ARGS[@]}")

"${CMD[@]}"
