#pragma once

#include "device_buffer.hpp"
#include "tensor.hpp"

#include <stdexcept>
#include <algorithm>


class Arena {
    private:
        DeviceBuffer buffer_;
        size_t offset_ = 0;
        size_t highWater_ = 0;

    public:
        explicit Arena(size_t bytes) {
            if (bytes == 0) throw std::invalid_argument("Arena size must be greater than 0");
            buffer_ = DeviceBuffer(bytes);
        }

        Arena(const Arena&) = delete;
        Arena& operator=(const Arena&) = delete;

        void* alloc(size_t bytes, size_t align = 256) {
            if (align == 0 || (align & (align - 1)) != 0) throw std::invalid_argument("Alignment must be a power of 2");
            
            // align the offset and check for out of memory
            size_t aligned = (offset_ + align - 1) & ~(align - 1);
            size_t cap = buffer_.getSize();
            if (aligned > cap || bytes > cap - aligned) throw std::runtime_error("Arena out of memory" + std::to_string(aligned) + " " + std::to_string(cap) + " " + std::to_string(bytes));
            
            void* ptr = static_cast<char*>(buffer_.getDevicePtr()) + aligned;
            offset_ = aligned + bytes;
            highWater_ = std::max(highWater_, offset_);
            return ptr;
        }

        Tensor allocTensor(DType dtype, const std::initializer_list<int64_t> dims, size_t align = 256){
            auto [shape, ndim] = to_shape(dims);
            size_t bytes = nbytes_of(dtype, shape, ndim);
            void* ptr = alloc(bytes, align);
            return Tensor(ptr, dtype, shape, ndim);
        }
    
        void reset() { offset_ = 0; }

        const size_t used() { return offset_; }

        const size_t capacity() { return buffer_.getSize(); }
        

        const size_t highWater() { return highWater_; }

};


