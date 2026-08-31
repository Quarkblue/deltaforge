#pragma once

#define JSON_USE_IMPLICIT_CONVERSIONS 0

#include "model_types.hpp"
#include "mapped_file.hpp"
#include "json.hpp"


#include <cuda_runtime.h>
#include <unordered_map>
#include <iostream>
#include <cstring>

using json = nlohmann::json;

#define CUDA_CHECK(expr) do { \
    cudaError_t err = (expr); \
    if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)\
        + " at " __FILE__ ":" + std::to_string(__LINE__)); \
} while (0)

constexpr size_t B_TO_MB = 1024 * 1024;
constexpr size_t B_TO_GB = 1024 * 1024 * 1024;

class Model {
    private:
        MappedFile mapped_; // mapped file of the model
        size_t data_offset_; // offset to the data section of the model
        size_t total_model_size = 0; // total size of the model
        std::unordered_map<std:: string, TensorView> tensors_; // map of tensor name to tensor view
        void* model_device_ptr = nullptr; // device pointer to the model
        
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


    static ModelHeader readModelHeader(const MappedFile& mapped_file){
        if (mapped_file.size() < 8) {
            throw std::runtime_error("Model file is too small to contain a header");
        }
        uint64_t header_size = 0;
        memcpy(&header_size, mapped_file.data(), 8);
        if (header_size > mapped_file.size() - 8) {
            throw std::runtime_error("Header size is too large to fit in the file");
        }
    
        std::cout << "Header size: " << header_size << std::endl;
        
        json header_json = json::parse(mapped_file.data() + 8, mapped_file.data() + 8 + header_size);
        
        // std::cout << "Header JSON: " << header_json.dump(2, ' ', true) << std::endl;
        std::cout << "Data offset: " << header_size + 8 << std::endl;

        return ModelHeader{std::move(header_json), header_size + 8};
    }

    void readModelWeights(const ModelHeader& header){
        for (auto& [name, meta] : header.header_json.items()){
            DType dtype = string_to_dtype(meta.at("dtype").get<std::string>());
            std::vector<uint64_t> data_offsets = meta.at("data_offsets").get<std::vector<uint64_t>>();
            size_t start_offset = data_offsets.front();
            size_t end_offset = data_offsets.back();
            std::array<int64_t, MAX_DIM> shape;
            int ndim = 0;
            int64_t numel = 1;
            for (auto& d : meta.at("shape")) {
                if (ndim >= MAX_DIM) {
                    throw std::runtime_error("Too many dimensions for tensor: " + name);
                }
                int64_t dim = d.get<int64_t>();
                shape[ndim] = dim;
                numel *= dim;
                ++ndim;
            }

            if (ndim < 1){
                throw std::runtime_error("Invalid shape: " + meta.at("shape").dump());
            }

            size_t nbytes_computed = numel * elem_size(dtype);
            size_t nbytes_ondisk = end_offset - start_offset;
            if (nbytes_computed != nbytes_ondisk) {
                throw std::runtime_error("Mismatch in the number of bytes computed and on disk for tensor: " + name);
            }

            const uint8_t* host_ptr = mapped_.data() + data_offset_ + start_offset;

            TensorView tv;
            tv.host_ptr = host_ptr;
            tv.device_ptr = nullptr;
            tv.ndim = ndim;
            tv.dtype = dtype;
            tv.nbytes = nbytes_computed;
            tv.shape = shape;
            tv.start_offset = start_offset;
            tv.end_offset = end_offset;

            auto [it, inserted] = tensors_.emplace(name, tv);
            if (!inserted) throw std::runtime_error("Tensor already exists: " + name);
            total_model_size += nbytes_computed;
        }
    }

    void uploadWeightsToGpu() {
        if (tensors_.empty()) return;

        // allocate total_model_size bytes on the GPU
        CUDA_CHECK(cudaMalloc(&model_device_ptr, total_model_size));
        CUDA_CHECK(cudaMemcpy(model_device_ptr, mapped_.data() + data_offset_, total_model_size, cudaMemcpyHostToDevice));

        for (auto& [name, tv] : tensors_) {
            tv.device_ptr = static_cast<char*>(model_device_ptr) + tv.start_offset;
        }
        std::cout << "Model weights uploaded to GPU successfully" << std::endl;
    }

    // ---- shared assertion helpers (used by all verify* methods) -------------
    static void requireNonNull(const TensorView* t, const std::string& who) {
        if (!t) throw std::runtime_error("null view: " + who);
    }
    static void requireShape(const TensorView* t, const std::string& who,
                             std::initializer_list<int64_t> dims, DType dt) {
        requireNonNull(t, who);
        if (t->ndim != (int)dims.size())
            throw std::runtime_error("ndim mismatch for " + who);
        int i = 0;
        for (int64_t d : dims)
            if (t->shape[i++] != d) throw std::runtime_error("shape mismatch for " + who);
        if (t->dtype != dt) throw std::runtime_error("dtype mismatch for " + who);
    }
    static void requireLayerNorm(const LayerNorm& n, const std::string& who) {
        requireNonNull(n.weight, who + ".weight");
        requireNonNull(n.bias, who + ".bias");
    }
    static void requireLinearWB(const LinearWB& l, const std::string& who) {
        requireNonNull(l.weight, who + ".weight");
        requireNonNull(l.bias, who + ".bias");
    }

    // Check 2: download a known F32 tensor back from the GPU and compare it
    // byte-for-byte against the host mmap. Proves offset math + copy direction +
    // byte fidelity end-to-end. A_log is the canary: F32 (readable as float),
    // tiny (64 B), sits at blob offset 0.
    void verifyUpload(const std::string& name) {
        const TensorView& tv = at(name);
        if (tv.dtype != DType::F32) throw std::runtime_error("verifyUpload expects F32: " + name);

        int64_t n = tv.nbytes / elem_size(tv.dtype);   // element count
        std::vector<float> dev(n);
        CUDA_CHECK(cudaMemcpy(dev.data(), tv.device_ptr, tv.nbytes, cudaMemcpyDeviceToHost));

        const float* host = reinterpret_cast<const float*>(tv.host_ptr);
        for (int64_t i = 0; i < n; ++i) {
            if (dev[i] != host[i]) {
                throw std::runtime_error("Round-trip mismatch in " + name +
                    " at element " + std::to_string(i));
            }
        }
        std::cout << "Round-trip OK: " << name << " (" << n << " floats, e.g. ["
                  << dev[0] << ", " << dev[1] << ", ...])" << std::endl;
    }
    
    // Check: the typed LM views resolved to the right tensors.
    //  (1) variant arm matches layer_types_ (FullAttn iff "full_attention"),
    //  (2) no field is null (belt-and-braces; view() already throws on a bad name),
    //  (3) a sampled full-attn + linear-attn layer match the reference shapes/dtypes,
    //  (4) lm_head aliases embed_tokens (same pointer, not a copy).
    void verifyLanguageModel() {
        if (language_model_.layers.size() != layer_types_.size())
            throw std::runtime_error("layer count mismatch");

        for (size_t i = 0; i < language_model_.layers.size(); ++i) {
            const DecoderLayer& L = language_model_.layers[i];
            const std::string p = "layer " + std::to_string(i);

            // (2) shared fields
            requireNonNull(L.input_layernorm.weight, p + " input_layernorm");
            requireNonNull(L.post_attention_layernorm.weight, p + " post_attn_norm");
            requireNonNull(L.mlp.gate_proj, p + " gate"); requireNonNull(L.mlp.up_proj, p + " up");
            requireNonNull(L.mlp.down_proj, p + " down");

            bool is_full = (layer_types_[i] == "full_attention");
            // (1) variant arm must agree with the config
            if (is_full != std::holds_alternative<FullAttn>(L.mixer))
                throw std::runtime_error("variant/layer_type mismatch at " + p);

            if (auto* fa = std::get_if<FullAttn>(&L.mixer)) {
                requireNonNull(fa->q_proj, p+" q"); requireNonNull(fa->k_proj, p+" k");
                requireNonNull(fa->v_proj, p+" v"); requireNonNull(fa->o_proj, p+" o");
                requireNonNull(fa->q_norm, p+" qn"); requireNonNull(fa->k_norm, p+" kn");
            } else {
                auto* la = std::get_if<LinearAttn>(&L.mixer);
                requireNonNull(la->in_proj_qkv, p+" qkv"); requireNonNull(la->conv1d, p+" conv");
                requireNonNull(la->in_proj_a, p+" a"); requireNonNull(la->in_proj_b, p+" b");
                requireNonNull(la->A_log, p+" A_log"); requireNonNull(la->dt_bias, p+" dt_bias");
                requireNonNull(la->in_proj_z, p+" z"); requireNonNull(la->norm, p+" norm");
                requireNonNull(la->out_proj, p+" out");
            }
        }

        // (3) sample one of each type against the reference table
        {
            const auto& fa = std::get<FullAttn>(language_model_.layers[3].mixer);
            requireShape(fa.q_proj, "L3 q_proj", {4096, 2048}, DType::BF16);
            requireShape(fa.k_proj, "L3 k_proj", {512, 2048}, DType::BF16);
            requireShape(fa.v_proj, "L3 v_proj", {512, 2048}, DType::BF16);
            requireShape(fa.o_proj, "L3 o_proj", {2048, 2048}, DType::BF16);
            requireShape(fa.q_norm, "L3 q_norm", {256}, DType::BF16);
            requireShape(fa.k_norm, "L3 k_norm", {256}, DType::BF16);

            const auto& la = std::get<LinearAttn>(language_model_.layers[0].mixer);
            requireShape(la.in_proj_qkv, "L0 in_proj_qkv", {6144, 2048}, DType::BF16);
            requireShape(la.conv1d, "L0 conv1d", {6144, 1, 4}, DType::BF16);
            requireShape(la.A_log, "L0 A_log", {16}, DType::F32);
            requireShape(la.dt_bias, "L0 dt_bias", {16}, DType::BF16);
            requireShape(la.norm, "L0 gdn_norm", {128}, DType::F32);
        }

        // top-level + the tie
        requireShape(language_model_.embed_tokens, "embed_tokens", {248320, 2048}, DType::BF16);
        requireShape(language_model_.norm.weight, "final norm", {2048}, DType::BF16);
        if (language_model_.lm_head != language_model_.embed_tokens)
            throw std::runtime_error("lm_head is not aliased to embed_tokens");

        std::cout << "LanguageModel OK: " << language_model_.layers.size()
                  << " layers, full-attn at {3,7,11,15,19,23}, lm_head tied" << std::endl;
    }

    // Spot-check vision + mtp views resolved to the right tensors (structure +
    // sampled shapes). Mirrors verifyLanguageModel.
    void verifyVisionAndMTP() {
        // --- vision: every field of every block non-null (bias included) ---
        if (vision_model_.blocks.size() != vision_model_.vision_model_depth)
            throw std::runtime_error("vision block count mismatch");
        for (size_t i = 0; i < vision_model_.blocks.size(); ++i) {
            const VisionBlock& b = vision_model_.blocks[i];
            const std::string p = "vblock " + std::to_string(i);
            requireLayerNorm(b.norm1, p+" norm1"); requireLayerNorm(b.norm2, p+" norm2");
            requireLinearWB(b.attn_qkv, p+" qkv"); requireLinearWB(b.attn_proj, p+" proj");
            requireLinearWB(b.mlp_fc1, p+" fc1"); requireLinearWB(b.mlp_fc2, p+" fc2");
        }
        // sampled shapes (block 0) — vision hidden = 1024
        const VisionBlock& b0 = vision_model_.blocks[0];
        requireShape(b0.attn_qkv.weight, "v0 qkv.w", {3072, 1024}, DType::BF16);
        requireShape(b0.attn_qkv.bias,   "v0 qkv.b", {3072},       DType::BF16);
        requireShape(b0.attn_proj.weight,"v0 proj.w",{1024, 1024}, DType::BF16);
        requireShape(b0.norm1.weight,    "v0 norm1", {1024},       DType::BF16);

        // patch_embed is the 5-D one; pos_embed is weight-only
        requireLinearWB(vision_model_.patch_embed, "patch_embed");
        requireShape(vision_model_.patch_embed.weight, "patch_embed.w",
                     {1024, 3, 2, 16, 16}, DType::BF16);
        requireShape(vision_model_.patch_embed.bias, "patch_embed.b", {1024}, DType::BF16);
        requireShape(vision_model_.pos_embed, "pos_embed", {2304, 1024}, DType::BF16);
        requireLinearWB(vision_model_.merger.fc1, "merger.fc1");
        requireLinearWB(vision_model_.merger.fc2, "merger.fc2");
        requireLayerNorm(vision_model_.merger.norm, "merger.norm");

        // --- mtp: one full-attn decoder layer + the extra norms + fc ---
        requireNonNull(mtp_.fc, "mtp.fc");
        requireNonNull(mtp_.pre_fc_norm_embedding.weight, "mtp.pre_fc_norm_embedding");
        requireNonNull(mtp_.pre_fc_norm_hidden.weight, "mtp.pre_fc_norm_hidden");
        requireNonNull(mtp_.norm.weight, "mtp.norm");
        if (mtp_.layers.size() != mtp_.mtp_num_hidden_layers)
            throw std::runtime_error("mtp layer count mismatch");
        requireShape(mtp_.fc, "mtp.fc", {2048, 4096}, DType::BF16);
        const DecoderLayer& m0 = mtp_.layers[0];
        if (!std::holds_alternative<FullAttn>(m0.mixer))
            throw std::runtime_error("mtp layer 0 is not FullAttn");
        const auto& mfa = std::get<FullAttn>(m0.mixer);
        requireShape(mfa.q_proj, "mtp q_proj", {4096, 2048}, DType::BF16);
        requireShape(mfa.o_proj, "mtp o_proj", {2048, 2048}, DType::BF16);

        std::cout << "Vision+MTP OK: " << vision_model_.blocks.size()
                  << " vision blocks, 1 mtp full-attn layer" << std::endl;
    }

    // Milestone: every one of the 632 tensors is referenced by exactly one typed
    // view, and nothing is left unclaimed. lm_head aliases embed_tokens (the tie)
    // and is deliberately NOT counted here — it shares a pointer, verified for
    // identity in verifyLanguageModel. So every claimed tensor must have count 1.
    void verifyAllTensorsClaimed() {
        std::unordered_map<const TensorView*, int> claims;
        auto add = [&](const TensorView* p) { if (p) claims[p]++; };
        auto addLN = [&](const LayerNorm& n) { add(n.weight); add(n.bias); };
        auto addWB = [&](const LinearWB& l) { add(l.weight); add(l.bias); };
        auto addLayer = [&](const DecoderLayer& L) {
            add(L.input_layernorm.weight); add(L.post_attention_layernorm.weight);
            add(L.mlp.gate_proj); add(L.mlp.up_proj); add(L.mlp.down_proj);
            if (auto* fa = std::get_if<FullAttn>(&L.mixer)) {
                add(fa->q_proj); add(fa->k_proj); add(fa->v_proj);
                add(fa->o_proj); add(fa->q_norm); add(fa->k_norm);
            } else if (auto* la = std::get_if<LinearAttn>(&L.mixer)) {
                add(la->in_proj_qkv); add(la->conv1d); add(la->in_proj_a);
                add(la->in_proj_b); add(la->A_log); add(la->dt_bias);
                add(la->in_proj_z); add(la->norm); add(la->out_proj);
            }
        };

        // language model (lm_head skipped — it is the alias)
        add(language_model_.embed_tokens);
        add(language_model_.norm.weight);
        for (const auto& L : language_model_.layers) addLayer(L);

        // vision
        addWB(vision_model_.patch_embed);
        add(vision_model_.pos_embed);
        for (const auto& b : vision_model_.blocks) {
            addLN(b.norm1); addLN(b.norm2);
            addWB(b.attn_qkv); addWB(b.attn_proj);
            addWB(b.mlp_fc1); addWB(b.mlp_fc2);
        }
        addWB(vision_model_.merger.fc1);
        addWB(vision_model_.merger.fc2);
        addLN(vision_model_.merger.norm);

        // mtp
        add(mtp_.fc);
        add(mtp_.pre_fc_norm_embedding.weight);
        add(mtp_.pre_fc_norm_hidden.weight);
        add(mtp_.norm.weight);
        for (const auto& L : mtp_.layers) addLayer(L);

        // every registry tensor must be claimed exactly once (nothing missed,
        // nothing grabbed twice by a name typo)
        for (const auto& [name, tv] : tensors_) {
            auto it = claims.find(&tv);
            if (it == claims.end())
                throw std::runtime_error("UNCLAIMED tensor: " + name);
            if (it->second != 1)
                throw std::runtime_error("tensor claimed " +
                    std::to_string(it->second) + "x: " + name);
        }
        // and no view pointed at something outside the registry
        if (claims.size() != tensors_.size())
            throw std::runtime_error("claimed " + std::to_string(claims.size()) +
                " distinct vs " + std::to_string(tensors_.size()) + " tensors");

        std::cout << "All " << claims.size()
                  << " tensors claimed by exactly one typed view (lm_head tie excluded)"
                  << std::endl;
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

    // ---- dump(): human-readable per-module print (task 7) --------------------
    // One tensor: "  label -> [d0, d1, ...] DTYPE". nullptr prints <null> so this
    // doubles as a debug tool for half-built views.
    static void printTensor(const std::string& label, const TensorView* t) {
        std::cout << "  " << label << " -> ";
        if (!t) { std::cout << "<null>\n"; return; }
        std::cout << "[";
        for (int i = 0; i < t->ndim; ++i)
            std::cout << (i ? ", " : "") << t->shape[i];
        std::cout << "] " << dtype_to_string(t->dtype) << "\n";
    }

    static void dumpDecoderLayer(const std::string& tag, const DecoderLayer& L) {
        std::cout << tag << ":\n";
        printTensor("input_layernorm", L.input_layernorm.weight);
        printTensor("post_attention_layernorm", L.post_attention_layernorm.weight);
        if (auto* fa = std::get_if<FullAttn>(&L.mixer)) {
            std::cout << " [mixer = FullAttn]\n";
            printTensor("q_proj", fa->q_proj); printTensor("k_proj", fa->k_proj);
            printTensor("v_proj", fa->v_proj); printTensor("o_proj", fa->o_proj);
            printTensor("q_norm", fa->q_norm); printTensor("k_norm", fa->k_norm);
        } else if (auto* la = std::get_if<LinearAttn>(&L.mixer)) {
            std::cout << " [mixer = LinearAttn]\n";
            printTensor("in_proj_qkv", la->in_proj_qkv); printTensor("conv1d", la->conv1d);
            printTensor("in_proj_a", la->in_proj_a);     printTensor("in_proj_b", la->in_proj_b);
            printTensor("A_log", la->A_log);             printTensor("dt_bias", la->dt_bias);
            printTensor("in_proj_z", la->in_proj_z);     printTensor("norm", la->norm);
            printTensor("out_proj", la->out_proj);
        }
        printTensor("mlp.gate_proj", L.mlp.gate_proj);
        printTensor("mlp.up_proj", L.mlp.up_proj);
        printTensor("mlp.down_proj", L.mlp.down_proj);
    }

    static void dumpVisionBlock(const std::string& tag, const VisionBlock& b) {
        std::cout << tag << " [VisionBlock]:\n";
        printTensor("norm1.weight", b.norm1.weight); printTensor("norm1.bias", b.norm1.bias);
        printTensor("attn.qkv.weight", b.attn_qkv.weight); printTensor("attn.qkv.bias", b.attn_qkv.bias);
        printTensor("attn.proj.weight", b.attn_proj.weight); printTensor("attn.proj.bias", b.attn_proj.bias);
        printTensor("norm2.weight", b.norm2.weight); printTensor("norm2.bias", b.norm2.bias);
        printTensor("mlp.fc1.weight", b.mlp_fc1.weight); printTensor("mlp.fc1.bias", b.mlp_fc1.bias);
        printTensor("mlp.fc2.weight", b.mlp_fc2.weight); printTensor("mlp.fc2.bias", b.mlp_fc2.bias);
    }

    public:
    // Print one of each module type (the milestone's "one of each"), not all 632.
    void dump() {
        std::cout << "\n===== MODEL DUMP =====\n";

        std::cout << "[LanguageModel top-level]\n";
        printTensor("embed_tokens", language_model_.embed_tokens);
        printTensor("norm", language_model_.norm.weight);
        printTensor("lm_head (tied)", language_model_.lm_head);

        dumpDecoderLayer("LM layer 0 (linear)", language_model_.layers[0]);
        dumpDecoderLayer("LM layer 3 (full)",   language_model_.layers[3]);

        std::cout << "\n[VisionModel top-level]\n";
        printTensor("patch_embed.weight", vision_model_.patch_embed.weight);
        printTensor("patch_embed.bias", vision_model_.patch_embed.bias);
        printTensor("pos_embed", vision_model_.pos_embed);
        printTensor("merger.fc1.weight", vision_model_.merger.fc1.weight);
        printTensor("merger.fc2.weight", vision_model_.merger.fc2.weight);
        printTensor("merger.norm.weight", vision_model_.merger.norm.weight);
        dumpVisionBlock("vision block 0", vision_model_.blocks[0]);

        std::cout << "\n[MTP top-level]\n";
        printTensor("fc", mtp_.fc);
        printTensor("pre_fc_norm_embedding", mtp_.pre_fc_norm_embedding.weight);
        printTensor("pre_fc_norm_hidden", mtp_.pre_fc_norm_hidden.weight);
        printTensor("norm", mtp_.norm.weight);
        dumpDecoderLayer("MTP layer 0", mtp_.layers[0]);

        std::cout << "======================\n\n";
    }

    // Run all registry/view self-checks. Debug scaffolding — call from main after
    // construction; not part of the build path.
    void selfCheck() {
        verifyLanguageModel();
        verifyVisionAndMTP();
        verifyAllTensorsClaimed();
        verifyUpload("model.language_model.layers.0.linear_attn.A_log");
    }

    private:
    void buildModel() {
        MappedFile config_file(config_file_path_);
        config_json_ = json::parse(config_file.data(), config_file.data() + config_file.size());
        readLayerConfig(config_json_);

        buildLanguageModel("model.language_model.");
        buildVisionModel("model.visual.");
        buildMTP("mtp.");
    }

    public:
        Model(const std::wstring& model_folder_path, const std::wstring& model_file_name, const std::wstring& config_file_suffix = L"config.json") : mapped_(model_folder_path + model_file_name) {
            setModelFolderPath(model_folder_path);
            setModelFileSuffix(model_file_name);
            setConfigFileSuffix(config_file_suffix);

            std::wcout << "Model folder path: " << model_folder_path_ << std::endl;
            std::wcout << "Model file path: " << model_file_path_ << std::endl;
            std::wcout << "Config file path: " << config_file_path_ << std::endl;


            ModelHeader header_ = readModelHeader(mapped_);
            data_offset_ = header_.data_offset;
            readModelWeights(header_);

            std::cout << "Model loaded successfully with " << tensors_.size() << " tensors" << std::endl;

            size_t total = 0;
            for (auto& [name, tv] : tensors_) total += tv.nbytes;

            std::cout << "Total model size: " << total  << " bytes | " << total / double(B_TO_GB) << " GB" << std::endl;

            uploadWeightsToGpu();
            buildModel();
        }


        ~Model() {
            if (model_device_ptr) {
                cudaFree(model_device_ptr);
            }
        }

        const TensorView& at(const std::string& name) const {
            auto it = tensors_.find(name);
            if (it == tensors_.end()) throw std::runtime_error("Tensor not found: " + name);
            return it->second;
        }

        const TensorView* view(const std::string& name) { return &at(name); }

};

