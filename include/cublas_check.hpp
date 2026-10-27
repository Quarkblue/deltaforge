#pragma once

#include <cublas_v2.h>
#include <stdexcept>
#include <string>

#define CUBLAS_CHECK(expr) do { \
    cublasStatus_t st = (expr); \
    if (st != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string("CUBLAS error: ") + cublasGetStatusString(st) \
    + " at " + __FILE__ + " : " + std::to_string(__LINE__)); \
} while (0)