#pragma once

// Host-side launchers for the CUDA kernels. Plain C++ on purpose: tests/ and
// src/main.cpp are compiled by the host compiler, which cannot parse <<<>>>, so
// the __global__ kernels stay inside src/kernel.cu and only these declarations
// cross the boundary.
//
// Contract for every launcher here:
//   - argument shapes/dtypes are validated on the host (a kernel cannot throw)
//   - work is queued on ctx.stream(), never the default stream
//   - cudaGetLastError() is checked right after the launch, because a bad launch
//     configuration fails asynchronously and would otherwise be blamed on the
//     next unrelated CUDA call
//   - nothing here synchronises; compare() calls ctx.sync() when it reads back

#include "context.hpp"
#include "tensor.hpp"


// out[t, c] = table[ids[t], c]
//
//   table : BF16 [vocab, hidden]   (the weight slab, not the arena)
//   ids   : I64  [T]               (token ids, on the device)
//   out   : BF16 [T, hidden]       (arena)
//
// A pure 2-byte copy: no widening, no arithmetic, so the result must be
// bit-exact against the dump. Any difference is an indexing bug.
void embedGather(const Context& ctx, const Tensor& out, const Tensor& table, const Tensor& ids);


// Zero-centered RMSNorm over the last axis, mirroring Qwen3_5RMSNorm.forward
// (modeling_qwen3_5.py:726-734) exactly:
//
//   y   = x_fp32 * (1 / sqrt(mean(x_fp32^2) + eps))   // fp32
//   y   = y * (1.0 + w_fp32)                          // fp32, UNIT-OFFSET weight
//   out = bf16(y)                                     // one cast, at the very end
//
//   in     : BF16 [rows, cols]
//   weight : BF16 [cols]
//   out    : BF16 [rows, cols]
//
// unit_offset=false drops the `1.0 +` and is the negative control for decision 9:
// the wrong variant must FAIL the comparison, proving the check can see it.
void rmsNorm(const Context& ctx, const Tensor& out, const Tensor& in, const Tensor& weight,
             float eps, bool unit_offset = true);


void tiedHead(const Context& ctx, const Tensor& logits, const Tensor& h, 
            const Tensor& table);