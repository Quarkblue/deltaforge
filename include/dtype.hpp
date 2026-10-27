#pragma once

#include <string>
#include <stdexcept>
#include <cstddef>

constexpr size_t MAX_DIM = 5;

enum class DType { BF16, F32, I64 };

inline DType string_to_dtype(const std::string& s) {
    if (s == "BF16") return DType::BF16;
    if (s == "F32") return DType::F32;
    if (s == "I64") return DType::I64;
    throw std::runtime_error("Invalid dtype: " + s);
}

inline const char* dtype_to_string(DType dtype) {
    switch (dtype) {
        case DType::BF16: return "BF16";
        case DType::F32:  return "F32";
        case DType::I64: return "I64";
        default: throw std::runtime_error("no string for DType: " + std::to_string(static_cast<int>(dtype)));
    }
}

inline size_t elem_size(DType dtype) {
    switch (dtype) {
        case DType::BF16: return 2;
        case DType::F32: return 4;
        case DType::I64: return 8;
        default: throw std::runtime_error("size not defined for DType: " + std::to_string(static_cast<int>(dtype)));
    }
}