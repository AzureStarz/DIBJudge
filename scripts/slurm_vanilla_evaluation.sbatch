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
  cat <<'EOF'
Usage:
  sbatch slurm_vanilla_evaluation.sbatch -- [options] [-- vanilla_evaluation.py args...]

Options:
  -m, --model PATH                 Model path (env: VANILLA_MODEL)
  -o, --output-dir PATH            Output dir (env: VANILLA_OUTPUT_DIR)
  -l, --languages CSV              Comma-separated list (env: VANILLA_LANGUAGES)
  -t, --template NAME              Template name or json path (env: VANILLA_TEMPLATE)
  -T, --template-path PATH         Template directory (env: VANILLA_TEMPLATE_PATH)
  -b, --benchmark NAME             Benchmark: MM-Eval|multilingual-reward-bench|both
                                   (env: VANILLA_BENCHMARK)
  -p, --tensor-parallel-size NUM   Tensor parallel size (env: VANILLA_TENSOR_PARALLEL_SIZE)
  -h, --help                       Show this help

Examples:
  sbatch slurm_vanilla_evaluation.sbatch -- -m model/Qwen3-4B-Base -t mr3_templates
  VANILLA_MODEL=model/Qwen3-4B-Base sbatch slurm_vanilla_evaluation.sbatch --
EOF
}

# Defaults (can be overridden by env or flags)
MODEL="${VANILLA_MODEL:-model/mR3-Qwen3-4B-en-prompt-en-thinking}"
OUTPUT_DIR="${VANILLA_OUTPUT_DIR:-results/mR3-Qwen3-4B-en-prompt-en-thinking}"
LANGUAGES="${VANILLA_LANGUAGES:-}"
TEMPLATE="${VANILLA_TEMPLATE:-mr3_templates}"
TEMPLATE_PATH="${VANILLA_TEMPLATE_PATH:-configs/eval_config}"
BENCHMARK="${VANILLA_BENCHMARK:-}"
TENSOR_PARALLEL_SIZE="${VANILLA_TENSOR_PARALLEL_SIZE:-}"

while getopts ":m:o:l:t:T:b:p:h-:" opt; do
  case "${opt}" in
    m) MODEL="${OPTARG}" ;;
    o) OUTPUT_DIR="${OPTARG}" ;;
    l) LANGUAGES="${OPTARG}" ;;
    t) TEMPLATE="${OPTARG}" ;;
    T) TEMPLATE_PATH="${OPTARG}" ;;
    b) BENCHMARK="${OPTARG}" ;;
    p) TENSOR_PARALLEL_SIZE="${OPTARG}" ;;
    h)
      usage
      exit 0
      ;;
    -)
      case "${OPTARG}" in
        model) MODEL="${!OPTIND}"; OPTIND=$((OPTIND + 1)) ;;
        model=*) MODEL="${OPTARG#*=}" ;;
        output-dir) OUTPUT_DIR="${!OPTIND}"; OPTIND=$((OPTIND + 1)) ;;
        output-dir=*) OUTPUT_DIR="${OPTARG#*=}" ;;
        languages) LANGUAGES="${!OPTIND}"; OPTIND=$((OPTIND + 1)) ;;
        languages=*) LANGUAGES="${OPTARG#*=}" ;;
        template) TEMPLATE="${!OPTIND}"; OPTIND=$((OPTIND + 1)) ;;
        template=*) TEMPLATE="${OPTARG#*=}" ;;
        template-path) TEMPLATE_PATH="${!OPTIND}"; OPTIND=$((OPTIND + 1)) ;;
        template-path=*) TEMPLATE_PATH="${OPTARG#*=}" ;;
        benchmark) BENCHMARK="${!OPTIND}"; OPTIND=$((OPTIND + 1)) ;;
        benchmark=*) BENCHMARK="${OPTARG#*=}" ;;
        tensor-parallel-size) TENSOR_PARALLEL_SIZE="${!OPTIND}"; OPTIND=$((OPTIND + 1)) ;;
        tensor-parallel-size=*) TENSOR_PARALLEL_SIZE="${OPTARG#*=}" ;;
        help)
          usage
          exit 0
          ;;
        *)
          echo "Unknown option --${OPTARG}" >&2
          usage >&2
          exit 2
          ;;
      esac
      ;;
    \?)
      echo "Unknown option -${OPTARG}" >&2
      usage >&2
      exit 2
      ;;
    :)
      echo "Option -${OPTARG} requires an argument." >&2
      usage >&2
      exit 2
      ;;
  esac
done
shift $((OPTIND - 1))

# ---------------------------
# Host-side setup (outside container)
# ---------------------------
cd DIBJUDGE_ROOT
source DIBJUDGE_ENV_FILE

module load amd/singularity/3.10.4

# Paths
SIF=/online1/hpc_containers/base_hpc_glibc_235.sif
CONDA_ENV=/path/to/workspace/online1/miniconda3/envs/vllm_0112_new/
BIND1=/online1/hpc_containers/conda/
BIND2=/path/to/bind/

# ---------------------------
# Run everything INSIDE the container
# ---------------------------
VANILLA_MODEL="${MODEL}" \
VANILLA_OUTPUT_DIR="${OUTPUT_DIR}" \
VANILLA_LANGUAGES="${LANGUAGES}" \
VANILLA_TEMPLATE="${TEMPLATE}" \
VANILLA_TEMPLATE_PATH="${TEMPLATE_PATH}" \
VANILLA_BENCHMARK="${BENCHMARK}" \
VANILLA_TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE}" \
singularity exec --nv \
  -B "${BIND1}" \
  -B "${BIND2}" \
  "${SIF}" \
  bash -lc "
    set -euo pipefail

    # Make conda available and activate env
    source ~/.bashrc
    conda activate ${CONDA_ENV}

    unset CUDA_LAUNCH_BLOCKING
    unset NCCL_P2P_DISABLE
    unset NCCL_SHM_DISABLE
    unset NCCL_NVLS_DISABLE
    unset NCCL_IB_DISABLE
    export NCCL_P2P_LEVEL=NVL
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

    num_gpus=\$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo \"Detected \${num_gpus} GPUs\"

    cd DIBJUDGE_ROOT

    tensor_parallel_size=\"\${VANILLA_TENSOR_PARALLEL_SIZE:-}\"
    if [[ -z \"\${tensor_parallel_size}\" ]]; then
      tensor_parallel_size=\"\${num_gpus}\"
    fi

    cmd=(
      python scripts/vanilla_evaluation.py
      --model \"\${VANILLA_MODEL}\"
      --tensor_parallel_size \"\${tensor_parallel_size}\"
      --output_dir \"\${VANILLA_OUTPUT_DIR}\"
    )

    if [[ -n \"\${VANILLA_TEMPLATE}\" ]]; then
      cmd+=(--template \"\${VANILLA_TEMPLATE}\")
    fi
    if [[ -n \"\${VANILLA_TEMPLATE_PATH}\" ]]; then
      cmd+=(--template_path \"\${VANILLA_TEMPLATE_PATH}\")
    fi
    if [[ -n \"\${VANILLA_BENCHMARK}\" ]]; then
      cmd+=(--benchmark \"\${VANILLA_BENCHMARK}\")
    fi
    if [[ -n \"\${VANILLA_LANGUAGES}\" ]]; then
      IFS=',' read -r -a languages_arr <<< \"\${VANILLA_LANGUAGES}\"
      cmd+=(--languages \"\${languages_arr[@]}\")
    fi

    \"\${cmd[@]}\" \"\$@\"
  " -- "$@"
