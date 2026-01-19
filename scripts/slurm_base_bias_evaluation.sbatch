#!/bin/bash
#SBATCH --job-name=dibjudge_sft_baseline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd DIBJUDGE_ROOT

source ~/online1/miniconda3/bin/activate sglang
source DIBJUDGE_ENV_FILE

unset CUDA_LAUNCH_BLOCKING
unset NCCL_P2P_DISABLE
unset NCCL_SHM_DISABLE
unset NCCL_NVLS_DISABLE
unset NCCL_IB_DISABLE
export NCCL_P2P_LEVEL=NVL
export VLLM_WORKER_MULTIPROC_METHOD=spawn

module load amd/gcc_compiler/11.3.0
num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected ${num_gpus} GPUs"

# python scripts/vanilla_evaluation.py \
#     --model model/mR3-Qwen3-4B-en-prompt-en-thinking \
#     --tensor_parallel_size ${num_gpus} \
#     --output_dir results/mR3-Qwen3-4B-en-prompt-en-thinking \
#     --template mr3_templates

# Add model paths here; each entry will get its own results/<model_basename> output dir.
MODEL_PATHS=(
    # model/mR3-Qwen3-4B-en-prompt-en-thinking
    # model/mR3-Qwen3-8B-en-prompt-en-thinking
    # model/Qwen3-4B
    # model/Qwen3-8B
    # model/Qwen2.5-7B-Instruct
    # model/Qwen2.5-3B-Instruct
    outputs/Qwen3-4B_mR3_baseline_vanilla_LoRA/Qwen3-4B_mR3_baseline_vanilla_LoRA
)

for model_path in "${MODEL_PATHS[@]}"; do
    model_name="$(basename "${model_path}")"
    output_dir="bias_results/${model_name}"
    mkdir -p "${output_dir}"
    log_file="${output_dir}/run.log"

    # bash scripts/slurm_vanilla_evaluation.sbatch \
    python scripts/vanilla_evaluation.py \
        --model "${model_path}" \
        --output_dir "${output_dir}" \
        --template mr3_templates \
        --benchmark judgment-requests \
        --tensor_parallel_size "${num_gpus}" \
        >"${log_file}" 2>&1
done
