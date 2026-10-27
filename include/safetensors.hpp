#pragma once

#ifndef JSON_USE_IMPLICIT_CONVERSIONS
#define JSON_USE_IMPLICIT_CONVERSIONS 0
#endif

#include "mapped_file.hpp"
#include "model_types.hpp"
#include "tensor_view.hpp"
#include "json.hpp"

#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cstring>
#include <iostream>

using json = nlohmann::json;

struct SafeTensorsHeader {
    json header_json; // json header of the model
    // offset to the data section of the model (8 + header_size)
    size_t data_offset;
};

const std::array<std::string, 1> METADATA_KEYS = {"__metadata__"};

class SafeTensors {
    private:

        std::wstring file_path_;
        MappedFile mapped_file_;
        SafeTensorsHeader header_;
        std::unordered_map<std::string, TensorView> tensors_;
        json metadata_;
        size_t data_size_ = 0;


        SafeTensorsHeader readFileHeader() {
            if (mapped_file_.size() < 8) {
                throw std::runtime_error("File is too small to contain a header");
            }

            uint64_t header_size = 0;
            memcpy(&header_size, mapped_file_.data(), 8);
            if (header_size > mapped_file_.size() - 8) {
                throw std::runtime_error("Header size is too large to fit in the file");
            }

            json header_json = json::parse(mapped_file_.data() + 8, mapped_file_.data() + 8 + header_size);

            std::cout << "Header size: " << header_size << std::endl;
            return SafeTensorsHeader{std::move(header_json), 8 + header_size};
        }

        
        void createTensorViews() {
            for(auto& [name, meta] : header_.header_json.items()) {
                if (std::find(METADATA_KEYS.begin(), METADATA_KEYS.end(), name) != METADATA_KEYS.end()) {
                    std::cout << "[Found Metadata]" << name << std::endl;
                    // store metadata for this safetensors file
                    metadata_ = meta;
                    std::cout << "Metadata: " << metadata_.dump(2, ' ', true) << std::endl;
                    continue;
                }

                TensorView tv;
                tv.dtype = string_to_dtype(meta.at("dtype").get<std::string>());

                std::vector<uint64_t> data_offsets = meta.at("data_offsets").get<std::vector<uint64_t>>();

                if (data_offsets.size() != 2) throw std::runtime_error("Invalid data offsets for tensor: " + name);

                tv.start_offset = data_offsets.front();
                tv.end_offset = data_offsets.back();

                if(header_.data_offset + tv.end_offset > mapped_file_.size()) throw std::runtime_error("Tensor data out of bounds: " + name);

                std::array<int64_t, MAX_DIM> shape{};
                int ndim = 0;
                int64_t num_el = 1;
                for(auto& d : meta.at("shape")) {
                    if (ndim >= MAX_DIM) throw std::runtime_error("Too many dimenssions for tensor: " + name);
                    int64_t dim = d.get<int64_t>();
                    if (dim <= 0) throw std::runtime_error("Invalid shape for tensor: " + name);
                    shape[ndim] = dim;
                    num_el *= dim;
                    ++ndim;
                }
                if (ndim < 1) throw std::runtime_error("Invalid shape: " + meta.at("shape").dump());
                tv.ndim = ndim;

                size_t nbytes_computed = num_el * elem_size(tv.dtype);
                size_t nbytes_ondisk = tv.end_offset - tv.start_offset;

                if (nbytes_computed != nbytes_ondisk) throw std::runtime_error("Mismatch in the number of bytes computed and on disk for tensor: " + name);
                tv.nbytes = nbytes_computed;

                tv.shape = shape;

                tv.host_ptr = mapped_file_.data() + header_.data_offset + tv.start_offset;
                tv.device_ptr = nullptr;

                // to get the total data size of the data section of the file
                // because start and end offsets are stored relative to after data in the file
                // so the start_offset for the first tensor will be 0, whereas the actual offset
                // in the file would be 8 + header_size + start_offset
                if (data_size_ < tv.end_offset) data_size_ = tv.end_offset;

                auto [it, inserted] = tensors_.emplace(name, tv);
                if (!inserted) throw std::runtime_error("Tensor already exists: " + name);
            }
            // header keys = tensors + the optional __metadata__ entry
            size_t expected = header_.header_json.size() - (metadata_.is_null() ? 0 : 1);
            if (tensors_.size() != expected) throw std::runtime_error("Mismatch in the number of tensors");
            std::cout << "Parsed and Created " << tensors_.size() << " tensor views" << std::endl;
        }

    public:
        explicit SafeTensors(const std::wstring& file_path): file_path_(file_path), mapped_file_(file_path) {
            header_ = readFileHeader();
        }

        SafeTensors(const SafeTensors&) = delete;
        SafeTensors& operator=(const SafeTensors&) = delete;

        const TensorView& at(const std::string& name) const {
            auto it = tensors_.find(name);
            if (it == tensors_.end()) throw std::runtime_error("Tensor not found: " + name);
            return it->second;
        }

        const TensorView* view(const std::string& name) const { return &at(name); }

        // unordered_map has no order and json iterates keys alphabetically, so anything
        // compared against another tool's output goes through this.
        std::vector<std::string> sortedNames() const {
            std::vector<std::string> names;
            names.reserve(tensors_.size());
            for (const auto& [name, tv] : tensors_) names.push_back(name);
            std::sort(names.begin(), names.end());
            return names;
        }

        void bindToDevice(void* base) {
            for(auto& [name, tv] : tensors_) {
                tv.device_ptr = static_cast<uint8_t*>(base) + tv.start_offset;
            }
        }



        // Getters
        const std::wstring& filePath() const { return file_path_; }
        const SafeTensorsHeader& header() const { return header_; }
        const std::unordered_map<std::string, TensorView>& getTensors() const { return tensors_; }
        size_t dataSize() const { return data_size_; }
        const uint8_t* dataPtr() const { return mapped_file_.data() + header_.data_offset; }
        // Empty json when the file has no __metadata__ key.
        const json& metadata() const { return metadata_; }
        const size_t dataOffset() const { return header_.data_offset; }


        bool isLoaded() const { return mapped_file_.size() > 0; }

        void loadTensorsToViews() {
            createTensorViews();
        }

        void uploadTensorsToGpu(void* base) {
            bindToDevice(base);
        }
};
