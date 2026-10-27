"""Reference tables for the C++ bf16 helpers (brief 04, task 5).

torch is the oracle: the engine must cast exactly as the dumps were produced.
No model load, so this runs in about a second.

Writes dumps/bf16_table.safetensors with
  widen     F32[65536]  bf16 bit pattern i, widened by torch
  cast_in   F32[N]      fp32 inputs for the narrowing direction
  cast_out  F32[N]      torch's bf16 result, widened back to fp32

cast_out is stored *widened* on purpose: bf16 -> fp32 is exact and injective, so
comparing widened fp32 bit patterns is a bf16 bit comparison. That keeps DType
free of a U16 entry it would otherwise need just for this test.

Known divergence this table pins down: CUDA's __float2bfloat16 encodes every NaN
as 0x7FFF, torch 2.13 CPU as 0xFFFF. include/bf16.hpp special-cases NaN to match
torch; if a torch upgrade changes the encoding, the C++ test is what will say so.

Run:  python gen_bf16_table.py      (from python/, venv active)
"""
import os

import numpy as np
import torch
from safetensors.torch import save_file

OUT = "../dumps/bf16_table.safetensors"

# ── every bf16 pattern, widened ───────────────────────────────────────────────
patterns = np.arange(65536, dtype=np.uint16)
# bytearray: frombuffer needs a writable buffer, and numpy's tobytes() is not
widen = torch.frombuffer(bytearray(patterns.tobytes()), dtype=torch.bfloat16).float()

# ── narrowing: the cases that actually distinguish implementations ────────────
edge_bits = np.array([
    0x00000000,  # +0.0
    0x80000000,  # -0.0
    0x00000001,  # fp32 subnormal -> flushes to zero in bf16
    0x00800000,  # smallest normal fp32
    0x3F800000,  # 1.0
    0x3F808000,  # tie, surviving lsb 0 -> stays 0x3F80 (round half to EVEN)
    0x3F818000,  # tie, surviving lsb 1 -> rounds up to 0x3F82
    0x3F808001,  # just above a midpoint -> up
    0x3F807FFF,  # just below a midpoint -> down
    0xBF818000,  # negative tie
    0x3EAAAAAB,  # 1/3
    0x7F7FFFFF,  # max finite fp32 -> overflows to +inf
    0x7F800000,  # +inf
    0xFF800000,  # -inf
    0x7F7F0000,  # largest finite bf16, exactly representable
    0x7FC00000,  # quiet NaN
    0x7F800001,  # signalling NaN
    0x7F8F0000,  # NaN with payload in the surviving bits
    0xFFC00000,  # negative NaN
    0x71234567,  # large magnitude, arbitrary mantissa
    0x0DA24C00,  # small magnitude, arbitrary mantissa
], dtype=np.uint32)

rng = np.random.default_rng(0)
# ordinary activation magnitudes, plus a sweep either side of them
randoms = np.concatenate([
    rng.standard_normal(4096).astype(np.float32),
    (rng.standard_normal(4096) * 1e-20).astype(np.float32),
    (rng.standard_normal(4096) * 1e20).astype(np.float32),
])

cast_in = torch.cat([
    torch.frombuffer(bytearray(edge_bits.tobytes()), dtype=torch.float32),
    torch.from_numpy(randoms),
]).contiguous()
cast_out = cast_in.to(torch.bfloat16).float().contiguous()

os.makedirs(os.path.dirname(OUT), exist_ok=True)
save_file(
    {"widen": widen.contiguous(), "cast_in": cast_in, "cast_out": cast_out},
    OUT,
    metadata={"torch": torch.__version__, "note": "bf16 reference tables; torch is the oracle"},
)

nan_bits = torch.tensor([float("nan")]).to(torch.bfloat16).view(torch.uint16).item()
n_nan = int(torch.isnan(cast_in).sum())
print(f"wrote {OUT}")
print(f"  widen[{widen.numel()}], cast_in/cast_out[{cast_in.numel()}] ({n_nan} NaN inputs)")
print(f"  torch {torch.__version__}: NaN narrows to 0x{nan_bits:04X}")
