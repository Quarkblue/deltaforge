#pragma once

#include <stdexcept>
#include <string>


#define CUDA_CHECK(expr) do { \
    cudaError_t err = (expr); \
    if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)\
        + " at " __FILE__ ":" + std::to_string(__LINE__)); \
} while (0)