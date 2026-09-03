import numpy as np
import torch
from safetensors import safe_open

MODEL_DIR = "model/Qwen3.5-2B"
SHARD = f"{MODEL_DIR}/model.safetensors.000001-of-000001.safetensors"

def load_raw(key: str):
    with safe_open(SHARD, framework="numpy") as f:
        t = f.get_tensor(key).to(torch.float32)
    return t

def raw_keys() -> list[str]:
    with safe_open(SHARD, framework="numpy") as f:
        return list(f.keys())

def load_hf():
    from transformers import AutoModelForCausalLM
    
    