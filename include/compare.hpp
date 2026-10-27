#pragma once

// Device tensor vs HF dump, measured in fp64.
//
// Every stage check in the engine goes through here: copy the device tensor to
// the host, widen BOTH sides to double, and report max|d|, max|ref| and
// rel = max|d| / max|ref|. Relative, because bf16's error scales with magnitude
// (decision 13), and fp64, because comparing in fp32 would mix the comparison's
// own rounding into the number being calibrated.
//
// NaN and Inf are counted, never silently maxed: NaN > x is false, so a naive
// max treats a NaN as zero and the check passes for the wrong reason.

#include "bf16.hpp"
#include "context.hpp"
#include "cuda_check.hpp"
#include "tensor.hpp"
#include "tensor_view.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>


// One element as a double, by that side's own dtype. memcpy because the mmap
// blob makes no alignment promise at an arbitrary tensor offset.
inline double elemAsDouble(const uint8_t* base, DType dtype, size_t i) {
    switch (dtype) {
        case DType::BF16: {
            uint16_t b;
            std::memcpy(&b, base + i * 2, 2);
            return static_cast<double>(bf16_to_f32(b));
        }
        case DType::F32: {
            float f;
            std::memcpy(&f, base + i * 4, 4);
            return static_cast<double>(f);
        }
        default:
            throw std::runtime_error(std::string("compare: not a float dtype: ") + dtype_to_string(dtype));
    }
}

struct CompareResult {
    size_t n = 0;              // elements compared
    double max_abs_diff = 0.0; // max|mine - ref| over comparable elements
    double max_ref = 0.0;      // max|ref|, the scale rel is measured against
    double rel = 0.0;          // max_abs_diff / max_ref, or absolute if max_ref == 0
    bool rel_is_absolute = false;
    size_t worst = 0;          // index of max_abs_diff
    double worst_mine = 0.0;
    double worst_ref = 0.0;
    size_t nan_both = 0;       // NaN at the same index on both sides: counted, treated as equal
    size_t nan_one = 0;        // NaN on one side only: a hard mismatch, tolerance cannot excuse it
    size_t inf_mismatch = 0;   // Inf vs non-Inf, or opposite signs

    // A tolerance can only forgive magnitude, never a NaN/Inf disagreement.
    bool within(double rel_tol) const {
        return nan_one == 0 && inf_mismatch == 0 && rel <= rel_tol;
    }
};

// Copy `mine` to the host and measure it against `ref`. Both sides may have
// different dtypes (a bf16 kernel output vs the fp32 anchor dump is normal).
inline CompareResult compareToRef(const Context& ctx, const Tensor& mine, const TensorView& ref) {
    if (ref.host_ptr == nullptr) throw std::runtime_error("compare: ref has no host_ptr (dump not loaded)");

    const int64_t mine_n = mine.numel();
    const int64_t ref_n = numel_of(ref.shape, ref.ndim);
    if (mine_n != ref_n) {
        throw std::runtime_error("compare: element counts differ, mine=" + std::to_string(mine_n) +
                                 " ref=" + std::to_string(ref_n));
    }
    if (mine_n == 0) throw std::runtime_error("compare: nothing to compare");

    // The stream must be drained before the copy: our stream is not the default
    // one, so a kernel still in flight would otherwise be read half-written.
    ctx.sync();
    std::vector<uint8_t> host(mine.nbytes());
    CUDA_CHECK(cudaMemcpy(host.data(), mine.data, mine.nbytes(), cudaMemcpyDeviceToHost));

    CompareResult r;
    r.n = static_cast<size_t>(mine_n);
    for (size_t i = 0; i < r.n; ++i) {
        const double a = elemAsDouble(host.data(), mine.dtype, i);
        const double b = elemAsDouble(ref.host_ptr, ref.dtype, i);

        const bool a_nan = std::isnan(a), b_nan = std::isnan(b);
        if (a_nan || b_nan) {
            if (a_nan && b_nan) ++r.nan_both;
            else ++r.nan_one;
            continue;  // never let a NaN reach the max
        }
        const bool a_inf = std::isinf(a), b_inf = std::isinf(b);
        if (a_inf || b_inf) {
            if (!(a_inf && b_inf && std::signbit(a) == std::signbit(b))) ++r.inf_mismatch;
            continue;  // inf - inf is NaN even when both sides agree
        }

        const double d = std::fabs(a - b);
        if (d > r.max_abs_diff) {
            r.max_abs_diff = d;
            r.worst = i;
            r.worst_mine = a;
            r.worst_ref = b;
        }
        r.max_ref = std::fmax(r.max_ref, std::fabs(b));
    }

    // An all-zero reference has no scale to divide by; report the absolute
    // difference and flag it, so a printed `rel` is never secretly an atol.
    r.rel_is_absolute = (r.max_ref == 0.0);
    r.rel = r.rel_is_absolute ? r.max_abs_diff : r.max_abs_diff / r.max_ref;
    return r;
}

// ids and other integer tensors are not a tolerance question: exact or wrong.
// Returns the number of differing elements.
inline size_t compareExactI64(const Context& ctx, const Tensor& mine, const TensorView& ref) {
    if (mine.dtype != DType::I64 || ref.dtype != DType::I64) {
        throw std::runtime_error("compareExactI64: both sides must be I64");
    }
    if (mine.numel() != numel_of(ref.shape, ref.ndim)) throw std::runtime_error("compareExactI64: size mismatch");

    ctx.sync();
    std::vector<uint8_t> host(mine.nbytes());
    CUDA_CHECK(cudaMemcpy(host.data(), mine.data, mine.nbytes(), cudaMemcpyDeviceToHost));

    size_t bad = 0;
    for (size_t i = 0; i < static_cast<size_t>(mine.numel()); ++i) {
        int64_t a, b;
        std::memcpy(&a, host.data() + i * 8, 8);
        std::memcpy(&b, ref.host_ptr + i * 8, 8);
        bad += (a != b);
    }
    return bad;
}
