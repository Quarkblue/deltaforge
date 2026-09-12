"""NumPy-vs-HuggingFace correctness harness (Session 01).

Proves three ops that bracket the transformer stack — token embedding, final
RMSNorm, and the tied lm_head projection — against HF as the oracle.

Two weight sources:
  - RAW  : safetensors on disk (bf16) -> fp32 numpy. This is "our impl" input.
  - HF   : full model on CPU in fp32. This is ground truth.

Run:  python harness.py
"""
from __future__ import annotations

import functools
import numpy as np
import torch
from safetensors import safe_open


MODEL_DIR = "../model/Qwen3.5-2B"
SHARD = f"{MODEL_DIR}/model.safetensors-00001-of-00001.safetensors"

# Disk keys (verified from the shard). No lm_head on disk — it is TIED to embed.
EMBED_KEY = "model.language_model.embed_tokens.weight"
NORM_KEY = "model.language_model.norm.weight"
RMS_EPS = 1e-06
HIDDEN = 2048
VOCAB = 248320


# ── weight sources ──────────────────────────────────────────────────────────
def load_raw(key: str) -> np.ndarray:
    """One tensor by key from the shard, as fp32 numpy.

    numpy has no bfloat16 dtype, so `safe_open(framework="numpy").get_tensor`
    raises on a bf16 tensor. We open with framework="pt" (torch *does* have
    bf16), widen bf16->fp32 (an exact bit-widening), then hand off to numpy.
    """
    with safe_open(SHARD, framework="pt") as f:
        t = f.get_tensor(key)  # torch.bfloat16 for most tensors
    return t.float().numpy()  # bf16 -> fp32, exact


def raw_keys() -> list[str]:
    with safe_open(SHARD, framework="pt") as f:
        return list(f.keys())


@functools.lru_cache(maxsize=1)
def load_hf():
    """Full model on CPU in fp32 (the oracle). ~8.5 GB RAM, slow — cached.

    AutoModelForImageTextToText resolves Qwen3_5Config to the full
    Qwen3_5ForConditionalGeneration (has .lm_head and .model.language_model),
    matching python/reference.py.
    """
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_DIR, dtype=torch.float32, low_cpu_mem_usage=True
    )
    model.eval()
    return model


# Module accessors — the attribute path differs from the disk key strings:
# the disk key is "model.language_model.embed_tokens.weight", but the loaded
# object is a ForConditionalGeneration whose `.model` is the Qwen3_5Model.
def hf_embed(model):
    return model.model.language_model.embed_tokens


def hf_norm(model):
    return model.model.language_model.norm


def hf_head(model):
    return model.lm_head


# ── the reusable comparator (kept for sessions 02/03) ───────────────────────
def compare(name: str, mine, theirs, atol: float) -> bool:
    mine = np.asarray(mine, np.float32)
    theirs = np.asarray(theirs, np.float32)
    maxerr = float(np.abs(mine - theirs).max())
    ok = maxerr <= atol
    tag = "OK  " if ok else "FAIL"
    print(f"[{tag}] {name:22s} max|d|={maxerr:.2e}  atol={atol:.0e}  shape={tuple(mine.shape)}")
    return ok


# ── the three ops (our NumPy reference impls) ───────────────────────────────
def embed(input_ids: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Token embedding = a pure row gather."""
    return weight[input_ids]


def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = RMS_EPS) -> np.ndarray:
    """RMSNorm in fp32.

        y = x / sqrt(mean(x^2, axis=-1) + eps) * (1.0 + weight)

    No mean-subtraction and no bias (that is LayerNorm). RMSNorm rescales by the
    root-mean-square only — it fixes the vector's *magnitude*, not its mean.

    Two traps, both in Qwen3_5RMSNorm.forward:
      1. UNIT-OFFSET weight. The scale is (1.0 + weight): the parameter is
         initialized to zeros and stores the *delta from 1.0* (Gemma-style).
         Using `weight` directly is wrong by ~the channel scale.
      2. fp32 math. It upcasts x to fp32 before the rsqrt; bf16 misses tolerance.
    """
    x = x.astype(np.float32)
    var = np.mean(x * x, axis=-1, keepdims=True)
    x = x * (1.0 / np.sqrt(var + eps))
    return x * (1.0 + weight.astype(np.float32))


def tied_head(h: np.ndarray, embed_weight: np.ndarray) -> np.ndarray:
    """logits = h @ embed_tokens.weight.T  (no bias). lm_head is tied to embed."""
    return h.astype(np.float32) @ embed_weight.astype(np.float32).T


# ── tasks ───────────────────────────────────────────────────────────────────
def task1_weights_agree():
    print("\n== task 1: raw vs HF embedding weight agree in fp32 ==")
    w_raw = load_raw(EMBED_KEY)
    w_hf = hf_embed(load_hf()).weight.detach().numpy()
    # Both are the same bf16 tensor widened to fp32 -> expect bit-exact.
    compare("embed_tokens.weight", w_raw, w_hf, atol=0.0)


def task2_embedding():
    print("\n== task 2: embedding lookup ==")
    ids = np.array([0, 1, 42, 100, VOCAB - 1], dtype=np.int64)
    w_raw = load_raw(EMBED_KEY)
    mine = embed(ids, w_raw)
    theirs = hf_embed(load_hf())(torch.from_numpy(ids)).detach().numpy()
    compare("embed(ids)", mine, theirs, atol=0.0)


def task3_rmsnorm():
    print("\n== task 3: RMSNorm (fp32) ==")
    rng = np.random.default_rng(0)
    x = rng.standard_normal((7, HIDDEN)).astype(np.float32)
    w = load_raw(NORM_KEY)
    mine = rmsnorm(x, w, RMS_EPS)
    with torch.no_grad():
        theirs = hf_norm(load_hf())(torch.from_numpy(x)).numpy()
    compare("rmsnorm", mine, theirs, atol=1e-5)


def task4_tied_head():
    print("\n== task 4: tied head ==")
    model = load_hf()
    # The tie, structurally: lm_head and embed share one storage.
    same = hf_head(model).weight.data_ptr() == hf_embed(model).weight.data_ptr()
    print(f"       tie: lm_head.weight is embed_tokens.weight -> {same}")
    rng = np.random.default_rng(1)
    h = rng.standard_normal((3, HIDDEN)).astype(np.float32)
    w_raw = load_raw(EMBED_KEY)
    mine = tied_head(h, w_raw)
    with torch.no_grad():
        theirs = hf_head(model)(torch.from_numpy(h)).numpy()
    # bf16-origin weights + fp32 matmul; reduction-order differs from torch.
    compare("logits = h @ W.T", mine, theirs, atol=1e-3)


def task5_stub():
    print("\n== task 5: end-to-end stub (plumbing only — NOT a correctness check) ==")
    ids = np.array([1234, 5678], dtype=np.int64)
    w_embed = load_raw(EMBED_KEY)
    w_norm = load_raw(NORM_KEY)
    h = embed(ids, w_embed)          # [seq, 2048]
    h = rmsnorm(h, w_norm, RMS_EPS)  # no layers in between yet!
    logits = tied_head(h, w_embed)   # [seq, 248320]
    print(f"       ids{tuple(ids.shape)} -> logits{tuple(logits.shape)}  "
          f"(expected ({len(ids)}, {VOCAB}))")
    assert logits.shape == (len(ids), VOCAB)


if __name__ == "__main__":
    # `python harness.py` runs this file as the module named "__main__".
    # harness_attn does `from harness import ...`, which would import this same
    # file a SECOND time under the name "harness" — a distinct module object with
    # its own load_hf lru_cache, so the 8.5 GB model would load twice (~17 GB
    # peak). Aliasing the two names to one module object makes the cache shared.
    import sys

    sys.modules.setdefault("harness", sys.modules["__main__"])

    task1_weights_agree()
    task2_embedding()
    task3_rmsnorm()
    task4_tied_head()
    task5_stub()

    # Session 02 lives in harness_attn.py, which imports this module. Importing
    # it here (inside __main__, not at the top) keeps that one-directional: by
    # the time this line runs, `harness` is fully defined, so there is no
    # circular-import problem.
    import harness_attn

    raise SystemExit(1 if harness_attn.main() else 0)
