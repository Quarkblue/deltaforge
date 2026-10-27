#pragma once

#include "dtype.hpp"
#include "tensor_view.hpp"

#include <array>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>

inline int64_t numel_of(const std::array<int64_t, MAX_DIM>& shape, int ndim) {
    int64_t n = 1;
    for(int i = 0; i < ndim; ++i) {
        if (shape[i] > INT64_MAX / n) throw std::runtime_error("numel overflow");
        n *= shape[i];
    }
    return n;
}

inline size_t nbytes_of(DType dtype, const std::array<int64_t, MAX_DIM>& shape, int ndim) {
    size_t elem = elem_size(dtype);
    size_t n = static_cast<size_t>(numel_of(shape, ndim));
    if (n > SIZE_MAX / elem) throw std::runtime_error("nbytes overflow");
    return n * elem;
}

inline std::pair<std::array<int64_t, MAX_DIM>, int> to_shape(std::initializer_list<int64_t> dims) {
    if (dims.size() == 0 || dims.size() > static_cast<size_t>(MAX_DIM)) {
        throw std::runtime_error("shape: ndim" + std::to_string(dims.size()) + " must be 1.." + std::to_string(MAX_DIM));
    }
    std::array<int64_t, MAX_DIM> shape{};
    int i = 0;
    for(int64_t d : dims) {
        if (d <= 0) throw std::runtime_error("shape: dim" + std::to_string(i) + " must be > 0");
        shape[i++] = d;
    };
    return {shape, static_cast<int>(dims.size())};
}

// A non-owning view of LIVE DEVICE memory (the arena owns the bytes).
//
// Not TensorView: that one describes bytes in a mapped file on the host.
// Tensor is what kernels take. Copying it is cheap (a pointer plus numbers),
// like std::string_view -- but the arena must outlive every Tensor cut from it,
// and Arena::reset() dangles all of them at once.
struct Tensor {
    private:
        void validate() const {
            if (data == nullptr) throw std::runtime_error("Tensor: data pointer is null");
            if (ndim < 1) throw std::runtime_error("Tensor: ndim must be >= 1");
            if (ndim > static_cast<int>(MAX_DIM)) {
                throw std::runtime_error("Tensor: ndim " + std::to_string(ndim) +
                                         " exceeds MAX_DIM " + std::to_string(MAX_DIM));
            }
            for (int i = 0; i < ndim; ++i) {
                if (shape[i] <= 0) {
                    throw std::runtime_error("Tensor: dim " + std::to_string(i) +
                                             " must be > 0, got " + std::to_string(shape[i]));
                }
            }
        }

    public:
        void* data = nullptr;
        std::array<int64_t, MAX_DIM> shape{};
        int ndim = 0;
        DType dtype = DType::F32;

        Tensor() = default;

        Tensor(void* data, DType dtype, const std::array<int64_t, MAX_DIM>& shape, int ndim)
            : data(data), shape(shape), ndim(ndim), dtype(dtype) {
            validate();
        }

        Tensor(void* data, DType dtype, std::initializer_list<int64_t> dims)
            : data(data), dtype(dtype) {
            auto [shape, ndim] = to_shape(dims);
            this->shape = shape;
            this->ndim = ndim;
            validate();
        }

        int64_t dim(int i) const {
            if (i < 0 || i >= ndim) {
                throw std::runtime_error("Tensor: dim index " + std::to_string(i) +
                                        " out of range for ndim " + std::to_string(ndim));
            }
            return shape[i];
        }

        // Product of the dims. Guarded: a bad shape must throw, not wrap around and
        // hand a kernel a too-small length.
        int64_t numel() const {
            return numel_of(shape, ndim);
        }

        size_t nbytes() const {
            return nbytes_of(dtype, shape, ndim);
        }
};


// Wrap an already-uploaded weight so kernels see one type. A null device_ptr
// means that weight was never uploaded -- caught here, not inside a kernel.
inline Tensor fromView(const TensorView& tv) {
    if (tv.device_ptr == nullptr) throw std::runtime_error("fromView: TensorView has no device_ptr");
    return Tensor(tv.device_ptr, tv.dtype, tv.shape, tv.ndim);
}