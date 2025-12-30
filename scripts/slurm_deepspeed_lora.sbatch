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

# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=warn
# export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-3}

# deepspeed --num_gpus=3 --module dibjudge.finetune_deepspeed \
#   --config configs/finetune_deepspeed.yaml

cd DIBJUDGE_ROOT

source ~/online1/miniconda3/bin/activate sglang
source DIBJUDGE_ENV_FILE

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NVLS_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_P2P_LEVEL=LOC

num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
# num_gpus=4
echo "Detected ${num_gpus} GPUs"

### Deprecated
# python -m dibjudge.print_collate
# python scripts/convet_to_hf.py
###

deepspeed --num_gpus=${num_gpus} --module dibjudge.finetune_deepspeed \
    --config configs/finetune_deepspeed.yaml

# python scripts/convert_zero_to_hf.py \
#     --checkpoint-dir outputs/deepspeed/debug \
#     --tag epoch-1 \
#     --config configs/finetune_deepspeed.yaml \
#     --use-lora

bash scripts/slurm_dibjudge_evaluation.sbatch