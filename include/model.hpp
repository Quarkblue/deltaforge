#pragma once

#define JSON_USE_IMPLICIT_CONVERSIONS 0

#include "model_types.hpp"
#include "mapped_file.hpp"
#include "json.hpp"
#include "safetensors.hpp"
#include "cuda_check.hpp"
#include "device_buffer.hpp"


#include <cuda_runtime.h>
#include <unordered_map>
#include <iostream>
#include <cstring>

using json = nlohmann::json;

constexpr size_t B_TO_MB = 1024 * 1024;
constexpr size_t B_TO_GB = 1024 * 1024 * 1024;

class Model {
    private:
        SafeTensors weights_;

        DeviceBuffer model_device_buffer_; // device buffer to the model
        
        json config_json_; // config json of the model

        std::vector<std::string> layer_types_; // layer types of the model sequentially
        LanguageModel language_model_;
        VisionModel vision_model_;
        MTP mtp_;
        
        std::wstring model_folder_path_;
        std::wstring model_file_path_;
        std::wstring config_file_path_;

        void setModelFolderPath(const std::wstring& model_folder_path) {
            if (model_folder_path.empty()) throw std::runtime_error("Model folder path is empty");
            model_folder_path_ = model_folder_path;
        }
        void setModelFileSuffix(const std::wstring& model_file_name) {
            if (model_file_name.empty()) throw std::runtime_error("Model file name is empty");
            model_file_path_ = model_folder_path_ + model_file_name;
        }
        void setConfigFileSuffix(const std::wstring& config_file_suffix) {
            if (config_file_suffix.empty()) throw std::runtime_error("Config file suffix is empty");
            config_file_path_ = model_folder_path_ + config_file_suffix;
        }


        void uploadWeightsToGpu() { 
            if (weights_.getTensors().empty()) return;

            // allocate total_model_size bytes on the GPU
            DeviceBuffer model_device_buffer = DeviceBuffer(weights_.dataSize());
            model_device_buffer.upload(weights_.dataPtr(), weights_.dataSize());

            weights_.bindToDevice(model_device_buffer.getDevicePtr());
            model_device_buffer_ = std::move(model_device_buffer);
            std::cout << "Model weights uploaded to GPU successfully" << std::endl;
        }

        void readLayerConfig(const json& config_json) {
            layer_types_ = config_json
                            .at("text_config")
                            .at("layer_types")
                            .get<std::vector<std::string>>();
            if (config_json.contains("vision_config")) {
                vision_model_.vision_model_depth = config_json.at("vision_config").at("depth").get<uint8_t>();
            }
            if (config_json.contains("mtp_num_hidden_layers")) {
                mtp_.mtp_num_hidden_layers = config_json.at("mtp_num_hidden_layers").get<uint8_t>();
            }
        }

        void buildLanguageModel(std::string language_model_prefix) {
            for (uint8_t i = 0; i < layer_types_.size(); ++i) {
                std::string language_model_prefix_i = language_model_prefix + "layers." + std::to_string(i) + ".";
                DecoderLayer layer;
                layer.input_layernorm.weight = view(language_model_prefix_i + "input_layernorm.weight");
                layer.post_attention_layernorm.weight = view(language_model_prefix_i + "post_attention_layernorm.weight");
                layer.mlp.gate_proj = view(language_model_prefix_i + "mlp.gate_proj.weight");
                layer.mlp.up_proj = view(language_model_prefix_i + "mlp.up_proj.weight");
                layer.mlp.down_proj = view(language_model_prefix_i + "mlp.down_proj.weight");
                if (layer_types_[i] == "full_attention") {
                    FullAttn full_attn;
                    full_attn.q_proj = view(language_model_prefix_i + "self_attn.q_proj.weight");
                    full_attn.k_proj = view(language_model_prefix_i + "self_attn.k_proj.weight");
                    full_attn.v_proj = view(language_model_prefix_i + "self_attn.v_proj.weight");
                    full_attn.o_proj = view(language_model_prefix_i + "self_attn.o_proj.weight");
                    full_attn.q_norm = view(language_model_prefix_i + "self_attn.q_norm.weight");
                    full_attn.k_norm = view(language_model_prefix_i + "self_attn.k_norm.weight");
                    layer.mixer = std::move(full_attn);
                }
                else if (layer_types_[i] == "linear_attention") {
                    LinearAttn linear_attn;
                    linear_attn.in_proj_qkv = view(language_model_prefix_i + "linear_attn.in_proj_qkv.weight");
                    linear_attn.conv1d = view(language_model_prefix_i + "linear_attn.conv1d.weight");
                    linear_attn.in_proj_a = view(language_model_prefix_i + "linear_attn.in_proj_a.weight");
                    linear_attn.in_proj_b = view(language_model_prefix_i + "linear_attn.in_proj_b.weight");
                    linear_attn.A_log = view(language_model_prefix_i + "linear_attn.A_log");
                    linear_attn.dt_bias = view(language_model_prefix_i + "linear_attn.dt_bias");
                    linear_attn.in_proj_z = view(language_model_prefix_i + "linear_attn.in_proj_z.weight");
                    linear_attn.norm = view(language_model_prefix_i + "linear_attn.norm.weight");
                    linear_attn.out_proj = view(language_model_prefix_i + "linear_attn.out_proj.weight");
                    layer.mixer = std::move(linear_attn);
                } else {
                    throw std::runtime_error("Invalid layer type: " + layer_types_[i]);
                }

                language_model_.layers.push_back(layer);
            }

            language_model_.embed_tokens = view(language_model_prefix + "embed_tokens.weight");
            language_model_.norm.weight = view(language_model_prefix + "norm.weight");

            // TODO: find what to do if tie_word_embeddings is false and assign lm_head accordingly.
            if (config_json_.at("text_config").at("tie_word_embeddings").get<bool>()) language_model_.lm_head = language_model_.embed_tokens;

        }

        void buildVisionModel(std::string vision_model_prefix){
            for (uint8_t i = 0; i < vision_model_.vision_model_depth; ++i) {
                std::string vision_model_prefix_i = vision_model_prefix + "blocks." + std::to_string(i) + ".";
                VisionBlock block;
                // Norm
                block.norm1.weight = view(vision_model_prefix_i + "norm1.weight");
                block.norm1.bias = view(vision_model_prefix_i + "norm1.bias");
                block.norm2.weight = view(vision_model_prefix_i + "norm2.weight");
                block.norm2.bias = view(vision_model_prefix_i + "norm2.bias");

                // mlp linear fc
                block.mlp_fc1.weight = view(vision_model_prefix_i + "mlp.linear_fc1.weight");
                block.mlp_fc1.bias = view(vision_model_prefix_i + "mlp.linear_fc1.bias");
                block.mlp_fc2.weight = view(vision_model_prefix_i + "mlp.linear_fc2.weight");
                block.mlp_fc2.bias = view(vision_model_prefix_i + "mlp.linear_fc2.bias");

                // attn
                block.attn_proj.weight = view(vision_model_prefix_i + "attn.proj.weight");
                block.attn_proj.bias = view(vision_model_prefix_i + "attn.proj.bias");
                block.attn_qkv.weight = view(vision_model_prefix_i + "attn.qkv.weight");
                block.attn_qkv.bias = view(vision_model_prefix_i + "attn.qkv.bias");
                
                vision_model_.blocks.push_back(block);
            }

            VisionMerger merger;
            merger.fc1.weight = view(vision_model_prefix + "merger.linear_fc1.weight");
            merger.fc1.bias = view(vision_model_prefix + "merger.linear_fc1.bias");
            merger.fc2.weight = view(vision_model_prefix + "merger.linear_fc2.weight");
            merger.fc2.bias = view(vision_model_prefix + "merger.linear_fc2.bias");

            merger.norm.weight = view(vision_model_prefix + "merger.norm.weight");
            merger.norm.bias = view(vision_model_prefix + "merger.norm.bias");

            vision_model_.merger = merger;

            LinearWB patch_embed;
            
            patch_embed.weight = view(vision_model_prefix + "patch_embed.proj.weight");
            patch_embed.bias = view(vision_model_prefix + "patch_embed.proj.bias");

            vision_model_.patch_embed = patch_embed;

            vision_model_.pos_embed = view(vision_model_prefix + "pos_embed.weight");
        };

        void buildMTP(std::string mtp_prefix) {
            for (uint8_t i = 0; i < mtp_.mtp_num_hidden_layers; ++i) {
                std::string mtp_prefix_i = mtp_prefix + "layers." + std::to_string(i) + ".";
                DecoderLayer layer;
                layer.input_layernorm.weight = view(mtp_prefix_i + "input_layernorm.weight");
                layer.post_attention_layernorm.weight = view(mtp_prefix_i + "post_attention_layernorm.weight");
                layer.mlp.gate_proj = view(mtp_prefix_i + "mlp.gate_proj.weight");
                layer.mlp.up_proj = view(mtp_prefix_i + "mlp.up_proj.weight");
                layer.mlp.down_proj = view(mtp_prefix_i + "mlp.down_proj.weight");
                // TODO: currently hardcoding the layer type to full_attention but need to figure out how to determing mtp layer type.
                FullAttn full_attn;
                full_attn.q_proj = view(mtp_prefix_i + "self_attn.q_proj.weight");
                full_attn.k_proj = view(mtp_prefix_i + "self_attn.k_proj.weight");
                full_attn.v_proj = view(mtp_prefix_i + "self_attn.v_proj.weight");
                full_attn.o_proj = view(mtp_prefix_i + "self_attn.o_proj.weight");
                full_attn.q_norm = view(mtp_prefix_i + "self_attn.q_norm.weight");
                full_attn.k_norm = view(mtp_prefix_i + "self_attn.k_norm.weight");

                layer.mixer = std::move(full_attn);

                mtp_.layers.push_back(layer);
            }

            mtp_.norm.weight = view(mtp_prefix + "norm.weight");
            mtp_.fc = view(mtp_prefix + "fc.weight");
            mtp_.pre_fc_norm_embedding.weight = view(mtp_prefix + "pre_fc_norm_embedding.weight");
            mtp_.pre_fc_norm_hidden.weight = view(mtp_prefix + "pre_fc_norm_hidden.weight");
        }

        void buildModel() {
            MappedFile config_file(config_file_path_);
            config_json_ = json::parse(config_file.data(), config_file.data() + config_file.size());
            readLayerConfig(config_json_);
    
            buildLanguageModel("model.language_model.");
            buildVisionModel("model.visual.");
            buildMTP("mtp.");
        }

    public:
        Model(const std::wstring& model_folder_path, const std::wstring& model_file_name, const std::wstring& config_file_suffix = L"config.json") : weights_(model_folder_path + model_file_name) {            
            setModelFolderPath(model_folder_path);
            setModelFileSuffix(model_file_name);
            setConfigFileSuffix(config_file_suffix);

            std::wcout << "Model folder path: " << model_folder_path_ << std::endl;
            std::wcout << "Model file path: " << model_file_path_ << std::endl;
            std::wcout << "Config file path: " << config_file_path_ << std::endl;

            if (!weights_.isLoaded()){
                throw std::runtime_error("Weights are not loaded");
            }
        
            weights_.loadTensorsToViews();

            std::cout << "Model loaded successfully with " << weights_.getTensors().size() << " tensors" << std::endl;

            size_t total = 0;
            for (auto& [name, tv] : weights_.getTensors()) total += tv.nbytes;

            std::cout << "Total model size: " << total  << " bytes | " << total / double(B_TO_GB) << " GB" << std::endl;

            uploadWeightsToGpu();
            buildModel();
        }

        Model(const Model&) = delete;
        Model& operator=(const Model&) = delete;

        const TensorView& at(const std::string& name) const {
            auto it = weights_.getTensors().find(name);
            if (it == weights_.getTensors().end()) throw std::runtime_error("Tensor not found: " + name);
            return it->second;
        }

        const TensorView* view(const std::string& name) { return &at(name); }

        // read-only access for code outside the class (model_checks.hpp, kernels, tests)
        const std::unordered_map<std::string, TensorView>& tensors() const { return weights_.getTensors(); }
        const std::vector<std::string>& layerTypes() const { return layer_types_; }
        const LanguageModel& languageModel() const { return language_model_; }
        const VisionModel& visionModel() const { return vision_model_; }
        const MTP& mtp() const { return mtp_; }
        const SafeTensors& getWeights() const { return weights_; }

};

