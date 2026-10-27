#pragma once

#include "cublas_check.hpp"
#include "cuda_check.hpp"
#include "arena.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>

class Context {

    private:
        Arena& arena_;
        cudaStream_t stream_ = nullptr;
        cublasHandle_t cublas_handle_ = nullptr;

        void cleanup() {
            if (cublas_handle_ != nullptr) {
                cublasDestroy(cublas_handle_);
                cublas_handle_ = nullptr;
            }
            if (stream_ != nullptr) {
                cudaStreamDestroy(stream_);
                stream_ = nullptr;
            }
        }


    public:
        explicit Context(Arena& arena) : arena_(arena) {
            try {
            CUDA_CHECK(cudaStreamCreate(&stream_));
            CUBLAS_CHECK(cublasCreate(&cublas_handle_));
            CUBLAS_CHECK(cublasSetStream(cublas_handle_, stream_));
            } catch (const std::exception& e) {
                cleanup();
                throw;
            }
        }

        Context(const Context&) = delete;
        Context& operator=(const Context&) = delete;

        ~Context() {
            cleanup();
        }

        cublasHandle_t cublas() const { return cublas_handle_; }
        cudaStream_t stream() const { return stream_; }
        Arena& arena() const { return arena_; }

        void sync() const {
            CUDA_CHECK(cudaStreamSynchronize(stream_));
        }
};
