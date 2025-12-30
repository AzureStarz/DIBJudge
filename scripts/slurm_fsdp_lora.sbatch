#!/bin/bash
#SBATCH --job-name=dibjudge_fsdp_lora
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=warn
# export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

torchrun --nnodes=1 --nproc_per_node=3 \
  -m dibjudge.finetune_fsdp \
  --data-path data/m_preference_collection_10k.jsonl \
  --judge-encoder /path/to/workspace/online1/data/pretrained-models/mt5-xl \
  --lm Qwen/Qwen3-4B \
  --epochs 1 \
  --batch-size 2 \
  --bf16 \
  --use-lora \
  --lora-targets q_proj,k_proj,v_proj,o_proj \
  --fsdp-wrap-cls LlamaDecoderLayer,T5Block \
  --output-dir outputs/fsdp_lora \
  --debug-data
