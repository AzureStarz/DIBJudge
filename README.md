<div align="center">

# DIBJudge

### Mitigating Translationese Bias in Multilingual LLM-as-a-Judge via Disentangled Information Bottleneck

[![arXiv](https://img.shields.io/badge/arXiv-2603.10351-b31b1b.svg)](https://arxiv.org/abs/2603.10351)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-DeepSpeed-ee4c2c.svg)](https://www.deepspeed.ai/)
[![License](https://img.shields.io/badge/license-TBD-lightgrey.svg)](#license)

**DIBJudge** is a robust fine-tuning framework for multilingual LLM-as-a-Judge. It learns a judgment-critical representation with variational information compression, separates translationese-related spurious factors into a bias branch, and discourages dependence between robust and bias representations.

</div>

---

## News

- **2026-03-11**: Paper available on arXiv: [arXiv:2603.10351](https://arxiv.org/abs/2603.10351).
- Public checkpoints and processed data links will be added when the release artifacts are finalized.

## Contents

- [Installation](#installation)
- [Repository Layout](#repository-layout)
- [Data Format](#data-format)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Useful Scripts](#useful-scripts)
- [Citation](#citation)

## Installation

> The code is designed for GPU machines. Install the PyTorch build that matches your CUDA driver before installing the remaining dependencies if your platform needs a custom wheel.

```bash
git clone https://github.com/AzureStarz/DIBJudge.git
cd DIBJudge

conda create -n dibjudge python=3.10 -y
conda activate dibjudge

pip install -r requirements.txt
pip install -e .
```

Optional speedups:

- `flash-attn` for FlashAttention-compatible training/evaluation.
- `vllm` for fast evaluation generation.
- `swanlab` only if `use_swanlab: true` in a training config.

## Repository Layout

```text
dibjudge/                  # DIBJudge model, bottlenecks, data collator, training loops
configs/                   # DeepSpeed, training, ablation, eval, and quickstart configs
configs/quickstart/        # Minimal public smoke-run config
scripts/                   # Data preprocessing, proxy metrics, conversion, and evaluation
scripts/env.sh             # Public-safe shell/SLURM environment bootstrap
```

Large local artifacts are intentionally ignored by git: `data/`, `model/`, `outputs/`, `logs/`, `results/`, checkpoints, and `.env*` files.

## Data Format

Training uses JSONL. Each line should contain one preference/judgment example:

```json
{
  "instruction": "User task or question",
  "response_A": "First candidate response",
  "response_B": "Second candidate response",
  "judge_prompt": "Full judge prompt containing both responses",
  "output": "Target judge continuation/verdict"
}
```

Optional proxy fields can be stored in the same file or a parallel proxy cache JSONL:

```json
{
  "proxy_length_A": 128,
  "proxy_length_B": 96,
  "proxy_nll_A": 3.2,
  "proxy_nll_B": 3.7,
  "proxy_ttr_A": 0.54,
  "proxy_ttr_B": 0.49
}
```

Expected local layout:

```text
data/
  train_data/
    mr3_preprocessed.jsonl
    mr3_preprocessed.Qwen3-4B.proxy.jsonl
  eval_data/
    MM-Eval/
    multilingual-reward-bench/
    judgment_requests/
```

## Quick Start

### 1) Precompute proxy metrics

```bash
python scripts/precompute_proxy_metrics.py \
  --input data/train_data/mr3_preprocessed.jsonl \
  --lm Qwen/Qwen3-4B \
  --output data/train_data/mr3_preprocessed.Qwen3-4B.proxy.jsonl \
  --batch-size 4 \
  --max-length 4096 \
  --trust-remote-code
```

### 2) Launch a smoke training run

```bash
NUM_GPUS=${NUM_GPUS:-1}

deepspeed --num_gpus ${NUM_GPUS} --module dibjudge.finetune_deepspeed \
  --config configs/quickstart/dibjudge_4b.yaml
```

The quickstart config saves Hugging Face-style artifacts under:

```text
outputs/quickstart/hf-epoch-1/
  dibjudge/    # DIBJudge heads/config/tokenizer assets
  lm/          # Judge LM assets/adapters
```

### 3) Evaluate DIBJudge

```bash
python scripts/dibjudge_evaluation.py \
  --model outputs/quickstart/hf-epoch-1/lm \
  --checkpoint outputs/quickstart/hf-epoch-1/dibjudge \
  --benchmark MM-Eval \
  --languages zh \
  --template mr3_templates \
  --output_dir results/quickstart/dibjudge \
  --tensor_parallel_size ${NUM_GPUS}
```

### 4) Evaluate a vanilla judge baseline

```bash
python scripts/vanilla_evaluation.py \
  --model Qwen/Qwen3-4B \
  --benchmark MM-Eval \
  --languages zh \
  --template mr3_templates \
  --output_dir results/quickstart/vanilla \
  --tensor_parallel_size ${NUM_GPUS}
```

## Training

DIBJudge training entrypoint:

```bash
deepspeed --num_gpus ${NUM_GPUS:-8} --module dibjudge.finetune_deepspeed \
  --config configs/finetune_deepspeed_4B.yaml
```

SFT baseline entrypoint:

```bash
deepspeed --num_gpus ${NUM_GPUS:-8} --module dibjudge.finetune_deepspeed_sft \
  --config configs/finetune_deepspeed_sft_baseline.yaml
```

Important config fields:

| Field | Meaning |
| --- | --- |
| `judge_encoder` | Encoder backbone used for robust/bias representations. |
| `lm` | Causal LM backbone used as the judge. |
| `data_path` | Training JSONL path. |
| `proxy_cache_path` | Optional aligned JSONL with proxy labels. |
| `deepspeed_config` | DeepSpeed JSON config. |
| `use_vib` | Enable variational information bottleneck. |
| `disentangle_cov_weight` | Weight for cross-covariance disentanglement. |
| `use_lora` | Train LM LoRA adapters instead of all LM parameters. |

## Evaluation

DIBJudge evaluation supports prompt-embedding injection and multi-GPU embedding precomputation:

```bash
torchrun --nproc_per_node=${NUM_GPUS:-8} scripts/dibjudge_evaluation.py \
  --embed-distributed \
  --embed-on-gpu \
  --embed-only \
  --model outputs/quickstart/hf-epoch-1/lm \
  --checkpoint outputs/quickstart/hf-epoch-1/dibjudge \
  --output_dir results/quickstart/dibjudge \
  --languages zh \
  --template mr3_templates \
  --tensor_parallel_size ${NUM_GPUS:-8}

python scripts/dibjudge_evaluation.py \
  --load-embed-shards \
  --model outputs/quickstart/hf-epoch-1/lm \
  --checkpoint outputs/quickstart/hf-epoch-1/dibjudge \
  --output_dir results/quickstart/dibjudge \
  --languages zh \
  --template mr3_templates \
  --tensor_parallel_size ${NUM_GPUS:-8}
```

For SLURM clusters, use the public-safe wrappers and provide local paths through environment variables:

```bash
cp .env.example .env
# edit .env locally; do not commit it
source .env

sbatch scripts/slurm_dibjudge_evaluation.sbatch -- \
  --model outputs/quickstart/hf-epoch-1/lm \
  --checkpoint outputs/quickstart/hf-epoch-1/dibjudge \
  --languages zh \
  --template mr3_templates
```

## Useful Scripts

| Script | Purpose |
| --- | --- |
| `scripts/preprocess_data.py` | Build DIBJudge JSONL from M-Preference-Collection-style parquet files. |
| `scripts/precompute_proxy_metrics.py` | Compute length/NLL/TTR proxy labels. |
| `scripts/convert_deepspeed_to_hf.py` | Convert a DeepSpeed checkpoint to a Hugging Face checkpoint. |
| `scripts/dibjudge_evaluation.py` | Evaluate a trained DIBJudge model. |
| `scripts/vanilla_evaluation.py` | Evaluate a vanilla LLM-as-a-Judge baseline. |
| `scripts/replicate_evaluation.py` | Evaluate Replicate-hosted model APIs. |

Example preprocessing command:

```bash
python scripts/preprocess_data.py \
  --m-pref-root /path/to/M-Preference-Collection \
  --output data/train_data/mr3_preprocessed.jsonl \
  --sglang-model Qwen/Qwen3-4B \
  --sample-size 50000
```

## Privacy-Safe Release Notes

This repository avoids committing local user paths, proxy files, credentials, datasets, checkpoints, and experiment outputs. Use environment variables or an untracked `.env` file for private paths:

```bash
export DIBJUDGE_ROOT=/path/to/DIBJudge
export DIBJUDGE_CONDA_ENV=dibjudge
export DIBJUDGE_ENV_FILE=/path/to/private_env.sh
export DIBJUDGE_JUDGMENT_REQUEST_DIR=data/eval_data/judgment_requests
```

Before publishing, run:

```bash
rg -n --hidden \
  -e '/home/' -e '/Users/' -e 'BEGIN .*PRIVATE KEY' -e 'api[_-]?key\\s*[:=]' \
  . --glob '!.git/**' --glob '!.omx/**'
```

If sensitive values were ever committed to history, rewrite history or create a fresh public repository before release.

## Citation

```bibtex
@misc{zhang2026dibjudge,
  title        = {Mitigating Translationese Bias in Multilingual LLM-as-a-Judge via Disentangled Information Bottleneck},
  author       = {Zhang, Hongbin and Chen, Kehai and Bai, Xuefen and Pan, Youcheng and Xiang, Yang and Wang, Jinpeng and Zhang, Min},
  year         = {2026},
  eprint       = {2603.10351},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2603.10351}
}
```

## License

A license file has not been added yet. Add the intended open-source license before a public release.
