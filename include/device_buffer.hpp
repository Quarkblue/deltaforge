#pragma once

#include "cuda_check.hpp"

#include <cuda_runtime.h>
#include <stdexcept>


class DeviceBuffer {
    private:
        void* device_ptr_ = nullptr;
        size_t size_ = 0;
        static inline size_t alloc_count_ = 0;
        static inline size_t free_count_ = 0;

    public:
        DeviceBuffer() = default;

        explicit DeviceBuffer(size_t bytes){
            if (bytes == 0) throw std::runtime_error("Device buffer size is 0");
            CUDA_CHECK(cudaMalloc(&device_ptr_, bytes));
            alloc_count_++;
            size_ = bytes;
        }

        DeviceBuffer(const DeviceBuffer&) = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;

        DeviceBuffer(DeviceBuffer&& other) noexcept
            : device_ptr_(std::exchange(other.device_ptr_, nullptr)),
            size_(std::exchange(other.size_, 0)){}

        DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
            if (this != &other) {
                if (device_ptr_){ 
                    cudaFree(device_ptr_);
                    free_count_++;
                }
                device_ptr_ = std::exchange(other.device_ptr_, nullptr);
                size_ = std::exchange(other.size_, 0);
            }
            return *this;
        }


        ~DeviceBuffer(){
            if (device_ptr_){ 
                cudaFree(device_ptr_);
                free_count_++;
            }
        }


        void upload(const void* host_ptr, size_t bytes){
            if (bytes > size_) throw std::runtime_error("Device buffer is too small to upload data");
            CUDA_CHECK(cudaMemcpy(device_ptr_, host_ptr, bytes, cudaMemcpyHostToDevice));
        }

        const void download(void* host_ptr, size_t bytes) const {
            if (bytes > size_) throw std::runtime_error("Device buffer is too small to download data");
            CUDA_CHECK(cudaMemcpy(host_ptr, device_ptr_, bytes, cudaMemcpyDeviceToHost));
        }
    
        void* getDevicePtr() const { return device_ptr_; }
        size_t getSize() const { return size_; }
        static size_t getAllocCount() { return alloc_count_; }
        static size_t getFreeCount() { return free_count_; }

        explicit operator bool() const { return device_ptr_ != nullptr; }
};