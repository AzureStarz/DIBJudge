#!/bin/bash
#SBATCH --job-name=dibjudge_deepspeed_lora
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
  sbatch slurm_dibjudge_evaluation.sbatch -- [options] [-- dibjudge_evaluation.py args...]

Options:
  -m, --model PATH                 Model path (env: DIBJUDGE_MODEL)
  -c, --checkpoint PATH            Checkpoint path (env: DIBJUDGE_CHECKPOINT)
  -o, --output-dir PATH            Output dir (env: DIBJUDGE_OUTPUT_DIR)
  -l, --languages CSV              Comma-separated list (env: DIBJUDGE_LANGUAGES)
  -t, --template NAME              Prompt template (env: DIBJUDGE_TEMPLATE)
  -p, --tensor-parallel-size NUM   Tensor parallel size (env: DIBJUDGE_TENSOR_PARALLEL_SIZE)
  -h, --help                       Show this help

Examples:
  sbatch slurm_dibjudge_evaluation.sbatch -- -m /path/to/model -l zh,en
  DIBJUDGE_MODEL=/path/to/model sbatch slurm_dibjudge_evaluation.sbatch --
EOF
}

# Defaults (can be overridden by env or flags)
MODEL="${DIBJUDGE_MODEL:-outputs/deepspeed/debug/hf-epoch-1/lm}"
CHECKPOINT="${DIBJUDGE_CHECKPOINT:-outputs/deepspeed/debug/hf-epoch-1/dibjudge}"
OUTPUT_DIR="${DIBJUDGE_OUTPUT_DIR:-results/Qwen3-4B-Base/DIBJudge}"
LANGUAGES="${DIBJUDGE_LANGUAGES:-zh}"
TEMPLATE="${DIBJUDGE_TEMPLATE:-prometheus}"
TENSOR_PARALLEL_SIZE="${DIBJUDGE_TENSOR_PARALLEL_SIZE:-}"

while getopts ":m:c:o:l:t:p:h-:" opt; do
  case "${opt}" in
    m) MODEL="${OPTARG}" ;;
    c) CHECKPOINT="${OPTARG}" ;;
    o) OUTPUT_DIR="${OPTARG}" ;;
    l) LANGUAGES="${OPTARG}" ;;
    t) TEMPLATE="${OPTARG}" ;;
    p) TENSOR_PARALLEL_SIZE="${OPTARG}" ;;
    h)
      usage
      exit 0
      ;;
    -)
      case "${OPTARG}" in
        model) MODEL="${!OPTIND}"; OPTIND=$((OPTIND + 1)) ;;
        model=*) MODEL="${OPTARG#*=}" ;;
        checkpoint) CHECKPOINT="${!OPTIND}"; OPTIND=$((OPTIND + 1)) ;;
        checkpoint=*) CHECKPOINT="${OPTARG#*=}" ;;
        output-dir) OUTPUT_DIR="${!OPTIND}"; OPTIND=$((OPTIND + 1)) ;;
        output-dir=*) OUTPUT_DIR="${OPTARG#*=}" ;;
        languages) LANGUAGES="${!OPTIND}"; OPTIND=$((OPTIND + 1)) ;;
        languages=*) LANGUAGES="${OPTARG#*=}" ;;
        template) TEMPLATE="${!OPTIND}"; OPTIND=$((OPTIND + 1)) ;;
        template=*) TEMPLATE="${OPTARG#*=}" ;;
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

module load amd/singularity/3.10.4 amd/cuda/11.8.89

# Paths
SIF=/online1/hpc_containers/base_hpc_glibc_235.sif
CONDA_ENV=/path/to/workspace/.conda/envs/vllm_0112_new01/
BIND1=/online1/hpc_containers/conda/
BIND2=/path/to/bind/

# ---------------------------
# Run everything INSIDE the container
# ---------------------------
DIBJUDGE_MODEL="${MODEL}" \
DIBJUDGE_CHECKPOINT="${CHECKPOINT}" \
DIBJUDGE_OUTPUT_DIR="${OUTPUT_DIR}" \
DIBJUDGE_LANGUAGES="${LANGUAGES}" \
DIBJUDGE_TEMPLATE="${TEMPLATE}" \
DIBJUDGE_TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE}" \
singularity exec --nv \
  -B "${BIND1}" \
  -B "${BIND2}" \
  "${SIF}" \
  bash -lc "
    set -euo pipefail

    # Make conda available and activate env
    source ~/.bashrc
    conda activate ${CONDA_ENV}

    # NCCL configs
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
    export OMP_NUM_THREADS=8

    # CUDA configs
    # export PATH=/online1/public/support/amd/cuda/11.8/bin:$PATH
    # export LD_LIBRARY_PATH=/online1/public/support/amd/cuda/11.8/lib64:$LD_LIBRARY_PATH

    num_gpus=\$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo \"Detected \${num_gpus} GPUs\"

    cd DIBJUDGE_ROOT

    tensor_parallel_size=\"\${DIBJUDGE_TENSOR_PARALLEL_SIZE:-}\"
    if [[ -z \"\${tensor_parallel_size}\" ]]; then
      tensor_parallel_size=\"\${num_gpus}\"
    fi

    IFS=',' read -r -a languages_arr <<< \"\${DIBJUDGE_LANGUAGES}\"

    base_args=(
      --model \"\${DIBJUDGE_MODEL}\"
      --checkpoint \"\${DIBJUDGE_CHECKPOINT}\"
      --tensor_parallel_size \"\${tensor_parallel_size}\"
      --output_dir \"\${DIBJUDGE_OUTPUT_DIR}\"
      --languages \"\${languages_arr[@]}\"
      --template \"\${DIBJUDGE_TEMPLATE}\"
    )

    if [[ \"\${num_gpus}\" -gt 1 ]]; then
      echo \"Phase 1: distributed prompt embedding (torchrun)\"
      torchrun --nproc_per_node=\"\${num_gpus}\" scripts/dibjudge_evaluation.py \
        --embed-distributed \
        --embed-on-gpu \
        --embed-only \
        \"\${base_args[@]}\" \
        \"\$@\"

      echo \"Phase 2: vLLM evaluation (single process, reuse embeds)\"
      python scripts/dibjudge_evaluation.py \
        --load-embed-shards \
        \"\${base_args[@]}\" \
        \"\$@\"
    else
      python scripts/dibjudge_evaluation.py \
        --embed-on-gpu \
        \"\${base_args[@]}\" \
        \"\$@\"
    fi
  " -- "$@"
