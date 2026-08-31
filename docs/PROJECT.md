# Inference Engine — Project Home Base

**Goal:** Hand-written CUDA inference engine (C++/nvcc) that runs **Qwen3.5-2B
in full** — it is a vision-language model, so the engine handles **text + vision
input, mrope, and the MTP head**, not just text. Target: single RTX 4060 Laptop
(8 GB, sm_89). Learning-driven: each component is researched, prototyped in NumPy
against HuggingFace, then ported to CUDA.

**Scope vs. order:** everything the model does is in scope. We *build* it
incrementally, and text-only generation is the **first bring-up milestone** (a
checkpoint), NOT the boundary of the project. Vision / mrope / MTP are later
phases, not cut features.

> This file is the source of truth. Every work session reads it first and
> appends to the Session Log + Decisions. Concept study happens in separate
> Claude Code sessions seeded by `docs/sessions/NN-*.md`; this file tracks
> the whole so the project never drifts out of scope.

---

## Target facts (do not re-derive)

- **Model:** Qwen3.5-2B (`model/Qwen3.5-2B/`), a **vision-language model** — full
  multimodal support is the goal; text is just the first path we light up.
- **Weights:** safetensors, 1 shard file (index still present), 632 tensors,
  **4548144832 B (4.236 GiB)**, mostly **BF16** — BUT some tensors are **F32**
  (`*.linear_attn.A_log`, `*.linear_attn.norm.weight`; 596 BF16 / 36 F32).
  Loader must read dtype **per tensor**. 632 = 320 language_model + 297 visual + 15 mtp.
- **Hardware:** RTX 4060 Laptop, compute capability **8.9**, ~8 GB VRAM.
- **Toolchain:** CUDA 13.3, MSVC 19.44, CMake (VS 2022 gen). Build flags live in
  `CMakeLists.txt` (`sm_89`, C++17, `/Zc:preprocessor`, links `CUDA::cublas`).

### Architecture — language model
- hidden **2048**, intermediate **6144** (SwiGLU), **24 layers**, vocab **248320**.
- Norm: **RMSNorm**, eps 1e-6. **`lm_head` is TIED** → not on disk, reuse `embed_tokens`.
- Layer types alternate: **full-attention at [3,7,11,15,19,23]** (6 layers),
  the other **18 are GatedDeltaNet** (linear-attention / SSM).
- Full-attn: GQA, **8 Q heads / 2 KV heads**, head_dim 256; **q_proj[4096,2048]**
  (gated — `attn_output_gate: true`, emits q + gate), k/v_proj[512,2048],
  o_proj[2048,2048]; per-head QK-RMSNorm over 256; RoPE
  (theta 1e7, partial_rotary 0.25).
- GatedDeltaNet (16 heads, head_dim 128): in_proj_qkv[6144,2048],
  conv1d[6144,1,4] (depthwise, causal), in_proj_a/b[16,2048], A_log[16]F32,
  dt_bias[16], in_proj_z[2048,2048], norm[128]F32 (gated), out_proj[2048,2048].
### Architecture — vision + extras (in scope, later phases)
- **Vision tower** `model.visual.*`: Conv3d patch_embed (3→1024, kernel 2×16×16),
  pos_embed Embedding(2304,1024), 24 ViT blocks (**LayerNorm w/ bias**, qkv+proj,
  GELU-tanh MLP), then a **merger** (LayerNorm → fc 4096→4096 → GELU → fc 4096→2048)
  that projects vision features into the 2048-dim text space. Runs only on image input.
- **mrope** — multimodal RoPE (`rope_parameters.mrope_interleaved`, sections
  [11,11,10]); needed once vision tokens share positions with text.
- **MTP** `mtp.*` — multi-token-prediction / speculative head. Not wired into HF's
  `ForConditionalGeneration` forward, so it's the **last** thing we add.

---

## Phase roadmap

Status: [ ] todo  [~] in progress  [x] done

- [~] **P0. Setup** — CUDA/nvcc/CMake build working. (done: vectorAdd runs)
- [x] **P1. Model loading** — mmap, safetensors header parse, weight registry
      (632 → typed LM/vision/MTP views), single-slab GPU upload. `dump()` + self-checks.
- [ ] **P2. Decoder backbone (NumPy ref first)** — embedding, RMSNorm, linear/GEMM,
      attention, GQA, RoPE, QK-norm, SwiGLU, final norm + tied head, KV cache, greedy sampling.
- [ ] **P3. GatedDeltaNet layers** — linear attention → SSM/Mamba2 → gated delta rule.
- [ ] **P4. Full LM NumPy forward matches HF logits (~1e-3).**  ← text bring-up milestone
- [ ] **P5. CUDA port of the LM** — kernels per op, cuBLAS for GEMMs. First runnable text gen.
- [ ] **P6. Vision tower** — patch embed, ViT blocks, merger → image tokens into text stream.
- [ ] **P7. mrope + multimodal fusion** — position handling for mixed image/text sequences.
- [ ] **P8. MTP / speculative decoding** — the multi-token head, for throughput.

(P6–P8 are the rest of the engine, not optional extras — text is done at P5.)

---

## Session log (append-only)

| Date | Session | Outcome / notes |
|------|---------|-----------------|
| 2026-08-26 | setup + loader | Build toolchain, RAII mmap wrapper, safetensors header parse, full weight catalogue (58 roles) derived. Started weight-mapping brief. |
| 2026-08-31 | 00 weight mapping | **P1 done.** `Model` class: registry (632 `TensorView`), single-slab GPU upload (4.236 GiB, ~564 KB overhead, round-trip exact), typed LM+vision+MTP views, `dump()`. All 632 tensors claimed exactly once; ctor-time self-checks pass. Corrected q_proj→[4096,2048], size→GiB. Next: P2 backbone (NumPy-vs-HF). |

---

## Key decisions

1. **Target Qwen3.5-2B directly** — not a plain transformer first. User wants to
   learn modern architectures (GatedDeltaNet) head-on. (2026-08-26)
2. **Keep weights BF16 on device; read dtype per-tensor** (F32 SSM params exist). (2026-08-26)
3. **Loading = mmap + single-slab upload** — one `cudaMalloc(total)` + one
   `cudaMemcpy` H2D straight from the mmap blob (no full-file heap read); each
   tensor's `device_ptr` is an interior pointer into the slab, freed once.
   (Supersedes the earlier "per-tensor cudaMemcpy" plan.) (2026-08-31)
4. **`lm_head` tied** → alias `embed_tokens`, don't look for it on disk. (2026-08-26)
5. **Correctness path = NumPy-vs-HF before any CUDA kernel.** (2026-08-26)

---

## Active session briefs
- `docs/sessions/00-weight-mapping.md` — registry + typed views (P1). **DONE 2026-08-31.**
- `docs/sessions/01-embed-rmsnorm.md` — NumPy-vs-HF harness + embed/RMSNorm/tied head (P2 slice 1). **NEXT.**
- (queued, not yet written) 02 — full-attention block (GEMM/GQA/RoPE/QK-norm); 03 — SwiGLU MLP + layer assembly.
- Note: P2 is split into 01→03 (one concept cluster per session); P3 GatedDeltaNet gets its own 2–3 briefs.
