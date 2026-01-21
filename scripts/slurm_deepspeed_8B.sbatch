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

unset CUDA_LAUNCH_BLOCKING
unset NCCL_P2P_DISABLE
unset NCCL_SHM_DISABLE
unset NCCL_NVLS_DISABLE
unset NCCL_IB_DISABLE
export NCCL_P2P_LEVEL=NVL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

module load amd/gcc_compiler/11.3.0
num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected ${num_gpus} GPUs"

deepspeed --num_gpus=${num_gpus} --module dibjudge.finetune_deepspeed \
    --config configs/finetune_deepspeed_8B.yaml

bash scripts/slurm_dibjudge_evaluation.sbatch -- \
    --model outputs/8B_bernoulli_default/hf-epoch-1/lm \
    --checkpoint outputs/8B_bernoulli_default/hf-epoch-1/dibjudge \
    --output_dir results/DIBJudge/8B_bernoulli_default/hf-epoch-1 \
    --languages zh \
    --template mr3_templates \
    --use-compactor \
    --tensor_parallel_size ${num_gpus}
