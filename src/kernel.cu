#include "kernels.hpp"
#include "cuda_check.hpp"
#include "bf16.hpp"
#include "context.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <cuda_bf16.h>

namespace {
    __global__ void embedGatherKernel(uint16_t* out, const uint16_t* table, const int64_t* ids, int T, int hidden) {
        const int i = threadIdx.x + blockIdx.x * blockDim.x;
        if(i >= T*hidden) return;

        // t is the token index, c is the hidden index
        const int t = i / hidden;
        const int c = i % hidden;

        const int64_t row = ids[t];
        out[i] = table[row * hidden + c];
    }

    __global__ void rmsNormKernel(uint16_t* out, const uint16_t* in, const uint16_t* weight, int rows, int cols, float eps, bool unit_offset){
        const int row = blockIdx.x;
        if (row >= rows) return;
        const int lane = threadIdx.x % 32; // each warp has 32 threads
        const int warp = threadIdx.x / 32;

        const uint16_t* in_row = in + (int64_t)row * cols;
        uint16_t* out_row = out + (int64_t)row * cols;

        float sum = 0.0f;
        for (int c = threadIdx.x; c < cols; c += blockDim.x){
            const float x = bf16_to_f32(in_row[c]);
            sum += x * x;
        }

        for (int offset = 16; offset > 0; offset >>= 1){
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        }

        __shared__ float warp_sums[32];
        if (lane == 0) warp_sums[warp] = sum;
        __syncthreads();

        __shared__ float scale;
        if (threadIdx.x == 0) {
            float total = 0.0f;
            const int n_warps = (blockDim.x + 31) / 32;
            for (int w = 0; w < n_warps; ++w) total += warp_sums[w];
            scale = 1.0f / sqrtf(total/cols + eps);
        }
        __syncthreads();

        for (int c = threadIdx.x; c < cols; c += blockDim.x){
            const float x = bf16_to_f32(in_row[c]);
            const float w = bf16_to_f32(weight[c]);
            out_row[c] = f32_to_bf16(x * scale * (unit_offset ? (1.0f + w): w));
        }
    }
}

void embedGather(const Context &ctx, const Tensor &out, const Tensor &table, const Tensor &ids) {
    if (out.dtype != DType::BF16 || table.dtype != DType::BF16) {
        throw std::runtime_error("embedGather: out and table should be BF16");
    }
    if (ids.dtype != DType::I64) {
        throw std::runtime_error("embedGather: ids should be I64");
    }
    if (out.ndim != 2 || table.ndim != 2) {
        throw std::runtime_error("embedGather: ndim should be 2");
    }
    if (ids.ndim != 1) {
        throw std::runtime_error("embedGather: ids should be 1");
    }
    if (out.dim(0) != ids.numel() || out.dim(1) != table.dim(1)) {
        throw std::runtime_error("embedGather: out.dim[0] is not equal to ids.numel() or out.dim[1] is not equal to table.dim[1]");
    }

    auto* out_p = static_cast<uint16_t*>(out.data);
    auto* table_p = static_cast<const uint16_t*>(table.data);
    auto* ids_p = static_cast<const int64_t*>(ids.data);
    const int T = static_cast<int>(ids.dim(0));
    const int hidden = static_cast<int>(out.dim(1));

    const int n = T * hidden;
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads; // round up

    embedGatherKernel<<<blocks, threads, 0, ctx.stream()>>>(out_p, table_p, ids_p, T, hidden);
    CUDA_CHECK(cudaGetLastError());
}

void rmsNorm(const Context& ctx, const Tensor& out, const Tensor& in, const Tensor& weight, float eps, bool unit_offset) {
    if (out.dtype != DType::BF16 || in.dtype != DType::BF16 || weight.dtype != DType::BF16) {
        throw std::runtime_error("rmsNorm: out, in, and weight should be BF16");
    }
    if (out.ndim != 2 || in.ndim != 2 || weight.ndim != 1) {
        throw std::runtime_error("rmsNorm: ndim should be 2");
    }
    if (out.dim(0) != in.dim(0) || out.dim(1) != in.dim(1)) {
        throw std::runtime_error("rmsNorm: out.dim[0] is not equal to in.dim[0] or out.dim[1] is not equal to in.dim[1]");
    }
    if (weight.dim(0) != in.dim(1)) {
        throw std::runtime_error("rmsNorm: weight.dim[0] is not equal to in.dim[1]");
    }

    auto* out_p = static_cast<uint16_t*>(out.data);
    const auto* in_p = static_cast<const uint16_t*>(in.data);
    const auto* w_p = static_cast<const uint16_t*>(weight.data);
    const int rows = static_cast<int>(in.dim(0));
    const int cols = static_cast<int>(in.dim(1));

    const int threads = 256;
    const int blocks = rows;

    if (threads % 32 != 0) {
        throw std::runtime_error("rmsNorm: threads must be divisible by 32");
    }

    rmsNormKernel<<<blocks, threads, 0, ctx.stream()>>>(out_p, in_p, w_p, rows, cols, eps, unit_offset);
    CUDA_CHECK(cudaGetLastError());
}
