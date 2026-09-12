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
- **Repo layout:** `D:\AI\Inferencing\deltaforge` (git). C++ in `include/*.hpp` +
  `src/`; Python reference/harness in `python/` (`.venv`, `reference.py`);
  weights in `model/Qwen3.5-2B/`; briefs in `docs/sessions/`.

### Architecture — language model
- hidden **2048**, intermediate **6144** (SwiGLU), **24 layers**, vocab **248320**.
- Norm: **RMSNorm**, eps 1e-6, **zero-centered: scale = `(1 + weight)`** (param stored as
  delta-from-1, Gemma-style). Applies to input/post-attn norms, q_norm, k_norm, final norm.
  **Exception:** `linear_attn.norm` is `RMSNormGated` — plain `weight` (init 1), F32, times
  `silu(gate)`. **`lm_head` is TIED** → not on disk, reuse `embed_tokens`.
- Layer types alternate: **full-attention at [3,7,11,15,19,23]** (6 layers),
  the other **18 are GatedDeltaNet** (linear-attention / SSM).
- Full-attn: GQA, **8 Q heads / 2 KV heads**, head_dim 256; **q_proj[4096,2048]**
  (gated — `attn_output_gate: true`; output viewed as [heads, 512] then chunked →
  **per-head** q[:256] + gate[256:], NOT first-half/second-half), k/v_proj[512,2048],
  o_proj[2048,2048]; per-head QK-RMSNorm over 256 (zero-centered); RoPE
  (theta 1e7, partial_rotary 0.25 → **rotary dim 64** of 256, rotate_half style,
  remaining 192 pass through; `inv_freq` has 32 entries, cos/sin duplicated to 64);
  scale 1/16; `attn_out * sigmoid(gate)` before o_proj. Merge is **head-major per token**,
  gate flattened identically. HF `self_attn.forward` needs a batch axis.
- GatedDeltaNet (16 heads, head_dim 128): in_proj_qkv[6144,2048],
  conv1d[6144,1,4] (depthwise, causal), in_proj_a/b[16,2048], A_log[16]F32,
  dt_bias[16], in_proj_z[2048,2048], norm[128]F32 (gated), out_proj[2048,2048].
### Architecture — vision + extras (in scope, later phases)
- **Vision tower** `model.visual.*`: Conv3d patch_embed (3→1024, kernel 2×16×16),
  pos_embed Embedding(2304,1024), 24 ViT blocks (**LayerNorm w/ bias**, qkv+proj,
  GELU-tanh MLP), then a **merger** (LayerNorm → fc 4096→4096 → GELU → fc 4096→2048)
  that projects vision features into the 2048-dim text space. Runs only on image input.
- **mrope** — multimodal RoPE (`rope_parameters.mrope_interleaved`, sections
  [11,11,10]); needed once vision tokens share positions with text. **P7 plug-in point =**
  `apply_interleaved_mrope` (`modeling_qwen3_5.py:148`): text-only, all 3 position rows
  are equal and it collapses to plain RoPE (verified 02).
- **MTP** `mtp.*` — multi-token-prediction / speculative head. Not wired into HF's
  `ForConditionalGeneration` forward, so it's the **last** thing we add.

---

## Phase roadmap

Status: [ ] todo  [~] in progress  [x] done

- [~] **P0. Setup** — CUDA/nvcc/CMake build working. (done: vectorAdd runs)
- [x] **P1. Model loading** — mmap, safetensors header parse, weight registry
      (632 → typed LM/vision/MTP views), single-slab GPU upload. `dump()` + self-checks.
- [~] **P2. Decoder backbone (NumPy ref first)** — embedding, RMSNorm, linear/GEMM,
      attention, GQA, RoPE, QK-norm, SwiGLU, final norm + tied head, KV cache, greedy sampling.
      (slices 1-2 done: embed + RMSNorm + tied head + harness; full-attention sub-block
      `attn()` at 4.9e-07 vs HF. next: 03 SwiGLU MLP + decoder-layer assembly.)
- [ ] **P3. GatedDeltaNet layers** — linear attention → SSM/Mamba2 → gated delta rule.
- [ ] **P4. Full LM NumPy forward matches HF logits (~1e-3).**  ← text bring-up milestone
- [ ] **P4.5. Tokenizer** — P4 uses HF `AutoTokenizer` (ids only); before P5 wiring, a
      C++ byte-level BPE from `tokenizer.json` (pre-tok regex, merges, byte↔unicode map,
      special tokens, chat template), validated id-for-id vs HF. Own brief. Vision special
      tokens (`<|vision_start|>`, `<|image_pad|>`, …) revisited at P7.
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
| 2026-09-01 | housekeeping | Repo renamed `start-inferencing` → **`deltaforge`**, `git init` + first commit (`2bf0182`). P1 code split out of `main.cpp` into `include/{mapped_file,tensor_view,dtype,model_types,model}.hpp` (+ vendored `json.hpp`); `src/main.cpp` is now just GPU probe + `Model` ctor/selfCheck/dump. `src/kernel.cu` empty placeholder. No P2 work yet. |
| 2026-09-03 | 01 embed/RMSNorm/head | **P2 slice 1 done.** `python/harness.py`: NumPy-vs-HF harness (`load_raw`/`load_hf`(cached fp32 CPU)/`compare`) + ops `embed`, `rmsnorm`, `tied_head`. All 5 tasks pass — embed & RMSNorm bit-exact, head 1.6e-6, tie proven by shared `data_ptr`. **Trap found:** `Qwen3_5RMSNorm` is zero-centered (`1.0 + weight`, init 0) — all plain norms; the lone exception is `linear_attn.norm` (`RMSNormGated`, plain `w` + silu-gate, F32). bf16→fp32 via torch `framework="pt"`. Next: 02 attention. |
| 2026-09-13 | 02 attention block | **P2 slice 2 done.** `python/harness_attn.py`: **`attn(x, pos, layer_idx)`** matches HF layer-3 `self_attn` at **4.92e-07** (layer 7: 6.41e-07), budget 1e-3. All 5 stages verified in isolation + 5 negative controls that must XFAIL (naive q/gate split 6.0e0, plain-w norm 7.0e0, rotate-all-256 7.9e0, np.tile KV 3.0e0). `python harness.py` runs 01+02 green. **Refinements:** `rotary_emb` returns `[b,seq,64]` not 3-row mrope — `apply_interleaved_mrope` (modeling:148) collapses it, and that line is the whole P7 hook; `inv_freq` has 32 entries not 64; `self_attn.forward` needs a batch axis; attn-output merge and gate are both head-major and must align. **Trap:** running `harness.py` as `__main__` made `harness_attn`'s `from harness import` load the model a second time (~17 GB) — fixed by aliasing `sys.modules`. Next: 03 MLP + layer assembly. |

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
6. **Oracle = full HF `AutoModelForImageTextToText` on CPU fp32** (the
   `ForConditionalGeneration` wrapper, so `.lm_head` and `.model.language_model` exist;
   attr path = disk key with one extra `.model`). Raw weights via
   `safe_open(framework="pt").get_tensor(k).float().numpy()` — numpy can't hold bf16. (2026-09-03)
7. **Tolerance budget:** fp32-native ops expect ~0 (HF also runs them fp32); bf16-weight
   GEMMs get atol 1e-3. A miss by ≫ that is a *semantics* bug, not rounding. (2026-09-03)
8. **Tokenizer deferred to P4.5, not P2.** Per-op NumPy checks are driven by ids /
   random hidden states, so a tokenizer adds no correctness signal until real prompts
   (P4) and is only *required* for the C++ engine (P5). Embedding lookup itself was
   done in 01. (2026-09-04)
9. **Every harness stage ships a negative control** (`check_differs` → `XFAIL`): the wrong
   variant (naive q/gate split, `np.tile` KV, plain-`w` norm, rotate-all-256) must *fail*.
   Same-shape bugs are invisible to shape checks; controls are the only proof the test
   can see them. (2026-09-13)
10. **One harness per concept file, one runner.** `harness_<x>.py` imports `harness.py`
   (never the reverse, except inside `harness.py`'s `__main__`); `python harness.py` runs
   everything and exits non-zero. Alias `sys.modules["harness"]` to `__main__` there or
   the 8.5 GB oracle loads twice. Run with `python/.venv/Scripts/python.exe` from `python/`. (2026-09-13)

---

## Active session briefs
- `docs/sessions/00-weight-mapping.md` — registry + typed views (P1). **DONE 2026-08-31.**
- `docs/sessions/01-embed-rmsnorm.md` — NumPy-vs-HF harness + embed/RMSNorm/tied head (P2 slice 1). **DONE 2026-09-03.**
- `docs/sessions/02-attention-block.md` — full-attention sub-block, layer 3 (P2 slice 2). **DONE 2026-09-13.**
- `docs/sessions/03-mlp-decoder-layer.md` — SwiGLU MLP + pre-norm decoder layer + KV-cache prefill/step (P2 slice 3). **NEXT.** Imports `attn()` from `harness_attn.py`.
- (queued, not yet written) P3 GatedDeltaNet in 2-3 briefs: 04 conv1d + projections + gated RMSNorm; 05 delta rule (recurrent form) vs HF `torch_recurrent_gated_delta_rule`; 06 chunked form + full 24-layer assembly (-> P4).
- Note: P2 is split into 01→03 (one concept cluster per session); P3 GatedDeltaNet gets its own 2–3 briefs.
