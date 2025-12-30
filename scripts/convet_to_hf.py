import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dibjudge import finetune_deepspeed as ft

args = argparse.Namespace(
    use_lora=True,
    lm="model/Qwen3-4B-Base",
    lora_targets="q_proj,k_proj,v_proj,o_proj",
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
)

ft._save_lora_adapter(
    checkpoint_dir="outputs/deepspeed/debug",
    tag="epoch-1",
    args=args,
    rank=0,
)
print("Saved to outputs/deepspeed/debug/lora-epoch-1")
