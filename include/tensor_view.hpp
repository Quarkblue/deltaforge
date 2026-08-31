#pragma once

#include <cstdint>
#include <array>

#include "dtype.hpp"

struct TensorView {
    const uint8_t* host_ptr;
    void* device_ptr;
    std::array<int64_t, MAX_DIM> shape;
    int ndim;
    DType dtype;
    size_t nbytes;
    size_t start_offset; // offset to the start of the tensor
    size_t end_offset; // offset to the end of the tensor
};