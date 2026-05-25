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

# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=warn
# export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-3}

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NVLS_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_P2P_LEVEL=LOC

if command -v module >/dev/null 2>&1; then
  module load "${DIBJUDGE_GCC_MODULE:-amd/gcc_compiler/11.3.0}"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')
else
  num_gpus="${SLURM_GPUS_ON_NODE:-${NUM_GPUS:-1}}"
fi
echo "Detected ${num_gpus} GPUs"

deepspeed --num_gpus=${num_gpus} --module dibjudge.finetune_deepspeed_sft \
    --config configs/finetune_deepspeed_sft_baseline.yaml

# python scripts/convert_deepspeed_to_hf.py \
#     --checkpoint-dir outputs/deepspeed/sft_baseline/final \
#     --base-model model/Qwen3-4B-Base \
#     --output-dir outputs/sft_baseline/final/hf

# python scripts/vanilla_evaluation.py \
#     --model outputs/sft_baseline/final/hf \
#     --tensor_parallel_size ${num_gpus} \
#     --template prometheus \
#     --output_dir results/sft_baseline