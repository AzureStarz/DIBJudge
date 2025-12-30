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

cd DIBJUDGE_ROOT

source ~/online1/miniconda3/bin/activate sglang
source DIBJUDGE_ENV_FILE

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NVLS_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_P2P_LEVEL=LOC

num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected ${num_gpus} GPUs"

# Ablate Token Task
deepspeed --num_gpus=${num_gpus} --module dibjudge.finetune_deepspeed \
    --config configs/finetune_deepspeed_shuffle_ablation.yaml

bash scripts/slurm_dibjudge_evaluation.sbatch -- \
    --model outputs/deepspeed/shuffle_ablation/hf-epoch-1/lm \
    --checkpoint outputs/deepspeed/shuffle_ablation/hf-epoch-1/dibjudge \
    --output-dir results/Qwen3-4B-Base/ablation/shuffle_ablation \
    --languages zh \
    --template prometheus \
    --tensor-parallel-size ${num_gpus}

# Ablate BT Task
deepspeed --num_gpus=${num_gpus} --module dibjudge.finetune_deepspeed \
    --config configs/finetune_deepspeed_bt_ablation.yaml

bash scripts/slurm_dibjudge_evaluation.sbatch -- \
    --model outputs/deepspeed/bt_ablation/hf-epoch-1/lm \
    --checkpoint outputs/deepspeed/bt_ablation/hf-epoch-1/dibjudge \
    --output-dir results/Qwen3-4B-Base/ablation/bt_ablation \
    --languages zh \
    --template prometheus \
    --tensor-parallel-size ${num_gpus}

# Ablate Bias Branch
deepspeed --num_gpus=${num_gpus} --module dibjudge.finetune_deepspeed \
    --config configs/finetune_deepspeed_bias_ablation.yaml

bash scripts/slurm_dibjudge_evaluation.sbatch -- \
    --model outputs/deepspeed/bias_ablation/hf-epoch-1/lm \
    --checkpoint outputs/deepspeed/bias_ablation/hf-epoch-1/dibjudge \
    --output-dir results/Qwen3-4B-Base/ablation/bias_ablation \
    --languages zh \
    --template prometheus \
    --tensor-parallel-size ${num_gpus}