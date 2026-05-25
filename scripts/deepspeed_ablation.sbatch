#!/bin/bash
#SBATCH --job-name=dibjudge_deepspeed_lora
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

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

deepspeed --num_gpus=${num_gpus} --module dibjudge.finetune_deepspeed \
    --config configs/deepspeed_ablation/zero0_gradient_ckpt.yaml

bash scripts/slurm_vanilla_evaluation.sbatch -- \
    --model outputs/zero0_gradient_ckpt/hf-epoch-1/lm \
    --output_dir results/Qwen3-4B/zero0_gradient_ckpt \
    --languages zh \
    --template mr3_templates \
    --tensor_parallel_size ${num_gpus}

deepspeed --num_gpus=${num_gpus} --module dibjudge.finetune_deepspeed \
    --config configs/deepspeed_ablation/zero0_no_gradient_ckpt.yaml

bash scripts/slurm_vanilla_evaluation.sbatch -- \
    --model outputs/zero0_no_gradient_ckpt/hf-epoch-1/lm \
    --output_dir results/Qwen3-4B/zero0_no_gradient_ckpt \
    --languages zh \
    --template mr3_templates \
    --tensor_parallel_size ${num_gpus}

# deepspeed --num_gpus=${num_gpus} --module dibjudge.finetune_deepspeed \
#     --config configs/deepspeed_ablation/zero2_gradient_ckpt.yaml

# bash scripts/slurm_vanilla_evaluation.sbatch -- \
#     --model outputs/zero2_gradient_ckpt/hf-epoch-1/lm \
#     --output_dir results/Qwen3-4B/zero2_gradient_ckpt \
#     --languages zh \
#     --template mr3_templates \
#     --tensor_parallel_size ${num_gpus}

# deepspeed --num_gpus=${num_gpus} --module dibjudge.finetune_deepspeed \
#     --config configs/deepspeed_ablation/zero2_no_gradient_ckpt.yaml

# bash scripts/slurm_vanilla_evaluation.sbatch -- \
#     --model outputs/zero2_no_gradient_ckpt/hf-epoch-1/lm \
#     --output_dir results/Qwen3-4B/zero2_no_gradient_ckpt \
#     --languages zh \
#     --template mr3_templates \
#     --tensor_parallel_size ${num_gpus}
