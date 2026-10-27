#pragma once

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>   // std::exchange


class MappedFile {
    private:
        HANDLE file_ = INVALID_HANDLE_VALUE;
        HANDLE mapping_ = nullptr;
        void* data_ = nullptr;
        size_t size_ = 0;

        // Release whatever this object currently owns and return it to the
        // empty state. Shared by the destructor and move-assignment so the
        // cleanup logic lives in exactly one place.
        void release() noexcept {
            if (data_) UnmapViewOfFile(data_);
            if (mapping_) CloseHandle(mapping_);
            if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
            file_ = INVALID_HANDLE_VALUE;
            mapping_ = nullptr;
            data_ = nullptr;
            size_ = 0;
        }

    public:
        explicit MappedFile(const std::wstring& path) {
            file_ = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);

            if (file_ == INVALID_HANDLE_VALUE) {
                throw std::runtime_error("CreateFileW failed");
            }

            LARGE_INTEGER file_size;
            if (!GetFileSizeEx(file_, &file_size)) {
                CloseHandle(file_);
                throw std::runtime_error("GetFileSizeEx failed");
            }
            size_ = static_cast<size_t>(file_size.QuadPart);

            mapping_ = CreateFileMappingW(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
            if (mapping_ == nullptr) {
                CloseHandle(file_);
                throw std::runtime_error("CreateFileMappingW failed");
            }

            data_ = MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0);
            if (data_ == nullptr) {
                CloseHandle(mapping_);
                CloseHandle(file_);
                throw std::runtime_error("MapViewOfFile failed");
            }            
        }

        // destructor
        ~MappedFile() {
            release();
        }

        const std::uint8_t* data() const { return static_cast<const std::uint8_t*>(data_); }
        size_t size() const { return size_; }

        // disable copying and moving
        MappedFile(const MappedFile&) = delete;
        MappedFile& operator=(const MappedFile&) = delete;

        // enable moving
        MappedFile(MappedFile&& other) noexcept
            : file_(std::exchange(other.file_, INVALID_HANDLE_VALUE)),
              mapping_(std::exchange(other.mapping_, nullptr)),
              data_(std::exchange(other.data_, nullptr)),
              size_(std::exchange(other.size_, 0)) {}

        // enable moving assignment
        MappedFile& operator=(MappedFile&& other) noexcept {
            if (this != &other) {
                release();
                file_    = std::exchange(other.file_, INVALID_HANDLE_VALUE);
                mapping_ = std::exchange(other.mapping_, nullptr);
                data_    = std::exchange(other.data_, nullptr);
                size_    = std::exchange(other.size_, 0);
            }
            return *this;
        }

};