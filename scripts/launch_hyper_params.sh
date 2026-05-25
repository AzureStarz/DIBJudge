#!/bin/bash
#SBATCH --job-name=dibjudge_hyperparams
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

BASE_DIR="${DIBJUDGE_ROOT}"
CONFIG_ROOT="${BASE_DIR}/configs/hyper_params"

unset CUDA_LAUNCH_BLOCKING
unset NCCL_P2P_DISABLE
unset NCCL_SHM_DISABLE
unset NCCL_NVLS_DISABLE
unset NCCL_IB_DISABLE
export NCCL_P2P_LEVEL=NVL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

if command -v module >/dev/null 2>&1; then
  module load "${DIBJUDGE_GCC_MODULE:-amd/gcc_compiler/11.3.0}"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')
else
  num_gpus="${SLURM_GPUS_ON_NODE:-${NUM_GPUS:-1}}"
fi
echo "Detected ${num_gpus} GPUs"

get_yaml_value() {
  local key="$1"
  local file="$2"
  local value
  value=$(awk -v k="${key}" '
    $0 ~ "^[[:space:]]*" k ":[[:space:]]*" {
      line=$0
      sub("^[[:space:]]*" k ":[[:space:]]*", "", line)
      sub(/[[:space:]]+#.*$/, "", line)
      gsub(/^["'"'"']|["'"'"']$/, "", line)
      print line
      exit
    }
  ' "${file}" || true)
  echo "${value}"
}

shopt -s globstar nullglob

for cfg in "${CONFIG_ROOT}"/**/*.yaml; do
  output_dir="$(get_yaml_value output_dir "${cfg}")"
  log_dir="$(get_yaml_value log_dir "${cfg}")"
  epochs="$(get_yaml_value epochs "${cfg}")"
  max_lm_len="$(get_yaml_value max_lm_len "${cfg}")"

  if [[ -z "${output_dir}" ]]; then
    rel_cfg="${cfg#${CONFIG_ROOT}/}"
    run_name="${rel_cfg%.yaml}"
    run_name="${run_name//\//_}"
    output_dir="outputs/${run_name}"
  fi
  if [[ -z "${log_dir}" ]]; then
    run_name="$(basename "${output_dir}")"
    log_dir="logs/${run_name}"
  fi
  if [[ -z "${epochs}" ]]; then
    epochs=3
  fi
  max_lm_len_override=8192
  if [[ -n "${max_lm_len}" && "${max_lm_len}" =~ ^[0-9]+$ ]]; then
    if (( max_lm_len > 8192 )); then
      max_lm_len_override=8192
    fi
  fi
  if [[ -n "${max_lm_len_override}" ]]; then
    final_max_lm_len="${max_lm_len_override}"
  else
    final_max_lm_len="${max_lm_len:-config/default}"
  fi

  run_name="$(basename "${output_dir}")"
  hf_dir="${output_dir}/hf-epoch-${epochs}"
  results_dir="results/DIBJudge/hyper_params/${run_name}/hf-epoch-${epochs}"
  summary_path="${results_dir}/summary.json"
  hf_dir_abs="${hf_dir}"
  if [[ "${hf_dir_abs}" != /* ]]; then
    hf_dir_abs="${BASE_DIR}/${hf_dir_abs}"
  fi
  checkpoint_dir="${output_dir}"
  if [[ "${checkpoint_dir}" != /* ]]; then
    checkpoint_dir="${BASE_DIR}/${checkpoint_dir}"
  fi

  echo "=== Running ${cfg}"
  echo "output_dir=${output_dir}"
  echo "log_dir=${log_dir}"
  echo "max_lm_len=${final_max_lm_len}"
  echo "results_dir=${results_dir}"
  if [[ -f "${summary_path}" ]]; then
    echo "summary.json exists, skipping: ${summary_path}"
    continue
  fi
  ckpt_exists=false
  if [[ -d "${hf_dir_abs}/lm" && -d "${hf_dir_abs}/dibjudge" ]]; then
    ckpt_exists=true
  fi

  train_args=(
    deepspeed
    --num_gpus="${num_gpus}"
    --module dibjudge.finetune_deepspeed
    --config "${cfg}"
  )
  if [[ -n "${max_lm_len_override}" ]]; then
    train_args+=(--max-lm-len "${max_lm_len_override}")
  fi

  if [[ "${ckpt_exists}" == true ]]; then
    echo "checkpoint exists without summary, running evaluation only"
  else
    echo "run training"
    "${train_args[@]}"
  fi

  bash scripts/slurm_dibjudge_evaluation.sbatch -- \
    --model "${hf_dir}/lm" \
    --checkpoint "${hf_dir}/dibjudge" \
    --output_dir "${results_dir}" \
    --languages zh \
    --template mr3_templates \
    --tensor_parallel_size "${num_gpus}"

  if [[ -f "${summary_path}" ]]; then
    if [[ "${checkpoint_dir}" == "${BASE_DIR}"/outputs/* ]]; then
      echo "summary.json exists, deleting checkpoint dir: ${checkpoint_dir}"
      rm -rf "${checkpoint_dir}"
    else
      echo "summary.json exists, but checkpoint dir is not under outputs/: ${checkpoint_dir}"
    fi
  fi
done
