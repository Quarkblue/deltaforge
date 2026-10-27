#pragma once

#include <variant>
#include <vector>

#include "tensor_view.hpp"
#include "json.hpp"

using json = nlohmann::json;

struct RMSNorm { const TensorView* weight; };

struct MLP { const TensorView* gate_proj, *up_proj, *down_proj; };

struct FullAttn { const TensorView* q_proj, *k_proj, *v_proj, *o_proj, *q_norm, *k_norm; };

struct LinearAttn { const TensorView* in_proj_qkv, *conv1d, *in_proj_a, *in_proj_b, *A_log, *dt_bias, *in_proj_z, *norm, *out_proj; };

struct DecoderLayer {
    RMSNorm input_layernorm, post_attention_layernorm;
    std::variant<FullAttn, LinearAttn> mixer;
    MLP mlp;
};

struct LanguageModel {
    const TensorView* embed_tokens; 
    std::vector<DecoderLayer> layers;
    RMSNorm norm;
    const TensorView* lm_head;
};

struct LayerNorm { const TensorView* weight; const TensorView* bias; };

struct LinearWB { const TensorView* weight; const TensorView* bias; };

struct VisionBlock {
    LayerNorm norm1, norm2;
    LinearWB attn_qkv, attn_proj;
    LinearWB mlp_fc1, mlp_fc2;
};

struct VisionMerger { 
    LinearWB fc1, fc2;
    LayerNorm norm;
};

struct VisionModel {
    LinearWB patch_embed;
    const TensorView* pos_embed;
    std::vector<VisionBlock> blocks;
    VisionMerger merger;
    uint8_t vision_model_depth = 24;
};

struct MTP {
    const TensorView* fc;
    RMSNorm pre_fc_norm_embedding;
    RMSNorm pre_fc_norm_hidden;
    std::vector<DecoderLayer> layers;
    RMSNorm norm;
    uint8_t mtp_num_hidden_layers = 1;
};