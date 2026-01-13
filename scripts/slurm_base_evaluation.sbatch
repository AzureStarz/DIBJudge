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

cd DIBJUDGE_ROOT

source ~/online1/miniconda3/bin/activate sglang
source DIBJUDGE_ENV_FILE

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NVLS_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_P2P_LEVEL=LOC
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

module load amd/gcc_compiler/11.3.0
num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected ${num_gpus} GPUs"

python scripts/vanilla_evaluation.py \
    --model model/mR3-Qwen3-4B-en-prompt-en-thinking \
    --tensor_parallel_size ${num_gpus} \
    --output_dir results/baselines/mR3-Qwen3-4B-en-prompt-en-thinking \
    --languages zh \
    --template mr3_templates