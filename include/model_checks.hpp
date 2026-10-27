#pragma once

// Load-time self-checks for Model: registry, typed views, GPU upload.
// Kept outside Model so the class stays loading + typed views only.

#include "model.hpp"

#include <unordered_map>

namespace model_checks {

// ---- shared assertion helpers (used by all verify* methods) -------------
inline void requireNonNull(const TensorView* t, const std::string& who) {
    if (!t) throw std::runtime_error("null view: " + who);
}
inline void requireShape(const TensorView* t, const std::string& who,
                         std::initializer_list<int64_t> dims, DType dt) {
    requireNonNull(t, who);
    if (t->ndim != (int)dims.size())
        throw std::runtime_error("ndim mismatch for " + who);
    int i = 0;
    for (int64_t d : dims)
        if (t->shape[i++] != d) throw std::runtime_error("shape mismatch for " + who);
    if (t->dtype != dt) throw std::runtime_error("dtype mismatch for " + who);
}
inline void requireLayerNorm(const LayerNorm& n, const std::string& who) {
    requireNonNull(n.weight, who + ".weight");
    requireNonNull(n.bias, who + ".bias");
}
inline void requireLinearWB(const LinearWB& l, const std::string& who) {
    requireNonNull(l.weight, who + ".weight");
    requireNonNull(l.bias, who + ".bias");
}

// Check 2: download a known F32 tensor back from the GPU and compare it
// byte-for-byte against the host mmap. Proves offset math + copy direction +
// byte fidelity end-to-end. A_log is the canary: F32 (readable as float),
// tiny (64 B), sits at blob offset 0.
inline void verifyUpload(const Model& m, const std::string& name) {
    const TensorView& tv = m.at(name);
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
//  (1) variant arm matches m.layerTypes() (FullAttn iff "full_attention"),
//  (2) no field is null (belt-and-braces; view() already throws on a bad name),
//  (3) a sampled full-attn + linear-attn layer match the reference shapes/dtypes,
//  (4) lm_head aliases embed_tokens (same pointer, not a copy).
inline void verifyLanguageModel(const Model& m) {
    if (m.languageModel().layers.size() != m.layerTypes().size())
        throw std::runtime_error("layer count mismatch");

    for (size_t i = 0; i < m.languageModel().layers.size(); ++i) {
        const DecoderLayer& L = m.languageModel().layers[i];
        const std::string p = "layer " + std::to_string(i);

        // (2) shared fields
        requireNonNull(L.input_layernorm.weight, p + " input_layernorm");
        requireNonNull(L.post_attention_layernorm.weight, p + " post_attn_norm");
        requireNonNull(L.mlp.gate_proj, p + " gate"); requireNonNull(L.mlp.up_proj, p + " up");
        requireNonNull(L.mlp.down_proj, p + " down");

        bool is_full = (m.layerTypes()[i] == "full_attention");
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
        const auto& fa = std::get<FullAttn>(m.languageModel().layers[3].mixer);
        requireShape(fa.q_proj, "L3 q_proj", {4096, 2048}, DType::BF16);
        requireShape(fa.k_proj, "L3 k_proj", {512, 2048}, DType::BF16);
        requireShape(fa.v_proj, "L3 v_proj", {512, 2048}, DType::BF16);
        requireShape(fa.o_proj, "L3 o_proj", {2048, 2048}, DType::BF16);
        requireShape(fa.q_norm, "L3 q_norm", {256}, DType::BF16);
        requireShape(fa.k_norm, "L3 k_norm", {256}, DType::BF16);

        const auto& la = std::get<LinearAttn>(m.languageModel().layers[0].mixer);
        requireShape(la.in_proj_qkv, "L0 in_proj_qkv", {6144, 2048}, DType::BF16);
        requireShape(la.conv1d, "L0 conv1d", {6144, 1, 4}, DType::BF16);
        requireShape(la.A_log, "L0 A_log", {16}, DType::F32);
        requireShape(la.dt_bias, "L0 dt_bias", {16}, DType::BF16);
        requireShape(la.norm, "L0 gdn_norm", {128}, DType::F32);
    }

    // top-level + the tie
    requireShape(m.languageModel().embed_tokens, "embed_tokens", {248320, 2048}, DType::BF16);
    requireShape(m.languageModel().norm.weight, "final norm", {2048}, DType::BF16);
    if (m.languageModel().lm_head != m.languageModel().embed_tokens)
        throw std::runtime_error("lm_head is not aliased to embed_tokens");

    std::cout << "LanguageModel OK: " << m.languageModel().layers.size()
              << " layers, full-attn at {3,7,11,15,19,23}, lm_head tied" << std::endl;
}

// Spot-check vision + mtp views resolved to the right tensors (structure +
// sampled shapes). Mirrors verifyLanguageModel.
inline void verifyVisionAndMTP(const Model& m) {
    // --- vision: every field of every block non-null (bias included) ---
    if (m.visionModel().blocks.size() != m.visionModel().vision_model_depth)
        throw std::runtime_error("vision block count mismatch");
    for (size_t i = 0; i < m.visionModel().blocks.size(); ++i) {
        const VisionBlock& b = m.visionModel().blocks[i];
        const std::string p = "vblock " + std::to_string(i);
        requireLayerNorm(b.norm1, p+" norm1"); requireLayerNorm(b.norm2, p+" norm2");
        requireLinearWB(b.attn_qkv, p+" qkv"); requireLinearWB(b.attn_proj, p+" proj");
        requireLinearWB(b.mlp_fc1, p+" fc1"); requireLinearWB(b.mlp_fc2, p+" fc2");
    }
    // sampled shapes (block 0) — vision hidden = 1024
    const VisionBlock& b0 = m.visionModel().blocks[0];
    requireShape(b0.attn_qkv.weight, "v0 qkv.w", {3072, 1024}, DType::BF16);
    requireShape(b0.attn_qkv.bias,   "v0 qkv.b", {3072},       DType::BF16);
    requireShape(b0.attn_proj.weight,"v0 proj.w",{1024, 1024}, DType::BF16);
    requireShape(b0.norm1.weight,    "v0 norm1", {1024},       DType::BF16);

    // patch_embed is the 5-D one; pos_embed is weight-only
    requireLinearWB(m.visionModel().patch_embed, "patch_embed");
    requireShape(m.visionModel().patch_embed.weight, "patch_embed.w",
                 {1024, 3, 2, 16, 16}, DType::BF16);
    requireShape(m.visionModel().patch_embed.bias, "patch_embed.b", {1024}, DType::BF16);
    requireShape(m.visionModel().pos_embed, "pos_embed", {2304, 1024}, DType::BF16);
    requireLinearWB(m.visionModel().merger.fc1, "merger.fc1");
    requireLinearWB(m.visionModel().merger.fc2, "merger.fc2");
    requireLayerNorm(m.visionModel().merger.norm, "merger.norm");

    // --- mtp: one full-attn decoder layer + the extra norms + fc ---
    requireNonNull(m.mtp().fc, "mtp.fc");
    requireNonNull(m.mtp().pre_fc_norm_embedding.weight, "mtp.pre_fc_norm_embedding");
    requireNonNull(m.mtp().pre_fc_norm_hidden.weight, "mtp.pre_fc_norm_hidden");
    requireNonNull(m.mtp().norm.weight, "mtp.norm");
    if (m.mtp().layers.size() != m.mtp().mtp_num_hidden_layers)
        throw std::runtime_error("mtp layer count mismatch");
    requireShape(m.mtp().fc, "mtp.fc", {2048, 4096}, DType::BF16);
    const DecoderLayer& m0 = m.mtp().layers[0];
    if (!std::holds_alternative<FullAttn>(m0.mixer))
        throw std::runtime_error("mtp layer 0 is not FullAttn");
    const auto& mfa = std::get<FullAttn>(m0.mixer);
    requireShape(mfa.q_proj, "mtp q_proj", {4096, 2048}, DType::BF16);
    requireShape(mfa.o_proj, "mtp o_proj", {2048, 2048}, DType::BF16);

    std::cout << "Vision+MTP OK: " << m.visionModel().blocks.size()
              << " vision blocks, 1 mtp full-attn layer" << std::endl;
}

// Milestone: every one of the 632 tensors is referenced by exactly one typed
// view, and nothing is left unclaimed. lm_head aliases embed_tokens (the tie)
// and is deliberately NOT counted here — it shares a pointer, verified for
// identity in verifyLanguageModel. So every claimed tensor must have count 1.
inline void verifyAllTensorsClaimed(const Model& m) {
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
    add(m.languageModel().embed_tokens);
    add(m.languageModel().norm.weight);
    for (const auto& L : m.languageModel().layers) addLayer(L);

    // vision
    addWB(m.visionModel().patch_embed);
    add(m.visionModel().pos_embed);
    for (const auto& b : m.visionModel().blocks) {
        addLN(b.norm1); addLN(b.norm2);
        addWB(b.attn_qkv); addWB(b.attn_proj);
        addWB(b.mlp_fc1); addWB(b.mlp_fc2);
    }
    addWB(m.visionModel().merger.fc1);
    addWB(m.visionModel().merger.fc2);
    addLN(m.visionModel().merger.norm);

    // mtp
    add(m.mtp().fc);
    add(m.mtp().pre_fc_norm_embedding.weight);
    add(m.mtp().pre_fc_norm_hidden.weight);
    add(m.mtp().norm.weight);
    for (const auto& L : m.mtp().layers) addLayer(L);

    // every registry tensor must be claimed exactly once (nothing missed,
    // nothing grabbed twice by a name typo)
    for (const auto& [name, tv] : m.tensors()) {
        auto it = claims.find(&tv);
        if (it == claims.end())
            throw std::runtime_error("UNCLAIMED tensor: " + name);
        if (it->second != 1)
            throw std::runtime_error("tensor claimed " +
                std::to_string(it->second) + "x: " + name);
    }
    // and no view pointed at something outside the registry
    if (claims.size() != m.tensors().size())
        throw std::runtime_error("claimed " + std::to_string(claims.size()) +
            " distinct vs " + std::to_string(m.tensors().size()) + " tensors");

    std::cout << "All " << claims.size()
              << " tensors claimed by exactly one typed view (lm_head tie excluded)"
              << std::endl;
}

// Run all registry/view self-checks. Debug scaffolding: call from main after construction.
inline void selfCheck(const Model& m) {
    verifyLanguageModel(m);
    verifyVisionAndMTP(m);
    verifyAllTensorsClaimed(m);
    verifyUpload(m, "model.language_model.layers.0.linear_attn.A_log");
}

}  // namespace model_checks
