# Inference Engine — Project Home Base

**Goal:** Hand-written CUDA inference engine (C++/nvcc) that runs **Qwen3.5-2B
in full** — it is a vision-language model, so the engine handles **text + vision
input, mrope, and the MTP head**, not just text. Target: single RTX 4060 Laptop
(8 GB, sm_89). Learning-driven: each component is researched, then written **directly
in CUDA/C++** and validated stage-by-stage against **HuggingFace activation dumps**
(decision 11; the NumPy-first path ran for sessions 01-02 and was retired 2026-09-13).

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
  `src/`; Python in `python/` (`.venv`; HF oracle dumper from brief 04; legacy NumPy
  harness `harness.py`/`harness_attn.py` kept as record, not extended);
  weights in `model/Qwen3.5-2B/`; briefs in `docs/sessions/`.

### Architecture — language model
- hidden **2048**, intermediate **6144 = 3 x 2048** (SwiGLU: `down(silu(gate(x)) * up(x))`),
  **24 layers**, vocab **248320**. Decoder layer is pre-norm with two residual adds.
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
  **KV cache:** post-norm, post-RoPE k and v, `[2 kv_heads, T, 256]` each ->
  **24 KiB/token fp32, 12 KiB/token bf16** across the 6 attn layers. Causal mask for a
  multi-token chunk after a non-empty cache is `[S_q, S_k]` offset by cache length.
- GatedDeltaNet (16 heads, head_dim 128): in_proj_qkv[6144,2048],
  conv1d[6144,1,4] (depthwise, causal), in_proj_a/b[16,2048], A_log[16]F32,
  dt_bias[16], in_proj_z[2048,2048], norm[128]F32 (gated), out_proj[2048,2048].
  HF cache per GDN layer: `conv_states`, `recurrent_states` (`modeling_qwen3_5.py:455,504`).
- **HF dtype policy in bf16 mode (engine mirrors it, decision 13):** RMSNorm math in fp32
  then cast back; `RMSNormGated` fp32 (177-184); softmax fp32 (621); both delta-rule
  forms upcast q/k/v/beta/g to fp32 (266, 347); RoPE cos/sin computed fp32, cast to x dtype (140-147).
### Architecture — vision + extras (in scope, later phases)
- **Vision tower** `model.visual.*`: Conv3d patch_embed (3→1024, kernel 2×16×16),
  pos_embed Embedding(2304,1024), 24 ViT blocks (**LayerNorm w/ bias**, qkv+proj,
  GELU-tanh MLP), then a **merger** (LayerNorm → fc 4096→4096 → GELU → fc 4096→2048)
  that projects vision features into the 2048-dim text space. Runs only on image input.
- **mrope** — multimodal RoPE (`rope_parameters.mrope_interleaved`, sections
  [11,11,10]); needed once vision tokens share positions with text. **P9 plug-in point =**
  `apply_interleaved_mrope` (`modeling_qwen3_5.py:148`): text-only, all 3 position rows
  are equal and it collapses to plain RoPE (verified 02).
- **MTP** `mtp.*` — multi-token-prediction / speculative head. Not wired into HF's
  `ForConditionalGeneration` forward, so it's the **last** thing we add.

### Engine surface (built in 04, everything later builds on it)
- **Loader:** `SafeTensors(path)` — mmap + header parse + name→`TensorView` registry +
  `__metadata__`; `at/view/getTensors/metadata/sortedNames/dataPtr/dataSize/bindToDevice`.
  `Model` owns one, uploads the slab into a `DeviceBuffer`, builds typed views.
  `DType = {BF16, F32, I64}`. Verification lives in `model_checks.hpp`, printing in
  `manifest.hpp` (`printManifest(os, st, prefix)` — no per-module print code).
- **Runtime:** `DeviceBuffer` (RAII cudaMalloc/cudaFree, move-only, static alloc/free
  counters), `Arena` (bump allocator, 256-byte align, `allocTensor`, `reset`, `highWater`),
  `Tensor` (non-owning device view; `fromView` wraps an uploaded weight), `Context`
  (cuBLAS handle + own stream + arena, `sync()`), `cuda_check.hpp` / `cublas_check.hpp`.
- **Kernels:** `include/kernels.hpp` declares host launchers only (tests and `main` are
  host-compiled and cannot parse `<<<>>>`); `__global__` code stays in `src/kernel.cu`.
  Launchers validate on the host, queue on `ctx.stream()`, check `cudaGetLastError()`,
  never synchronise. Done: `embedGather`, `rmsNorm(unit_offset)`, `tiedHead`.
- **Checking:** `compare.hpp` → `compareToRef` (both sides widened to fp64; `max|d|`,
  `max|ref|`, `rel`, worst index, NaN/Inf counted not maxed), `compareExactI64`;
  `deltaforge_test` prints `OK/FAIL/XFAIL/BAD` and returns the failure count.
  Run: `cmake --build build --config Release` then `ctest --test-dir build -C Release`.
  **`DELTAFORGE_SLOW_TESTS=1`** additionally runs the 1 GiB embed-table checks.

### Measured tolerances (quote these, don't re-derive)
- **embed gather: bit-exact** (rel 0.0) · **RMSNorm: bit-exact** (rel 0.0) vs the bf16 dump.
- **bf16 GEMM (k=2048, fp32 accumulate): rel 2.75e-03** — the budget for every later GEMM.
- **bf16 vs fp32 oracle runs at `final_norm`: rel 4.11e-03** — end-to-end divergence of the
  two dumps; a kernel error approaching this is a semantics bug, not rounding.
- **VRAM at ctx 32768:** weights 4338 MiB + KV 384 MiB + GDN 18.8 MiB + arena 16 MiB +
  cuBLAS 32 MiB ≈ **4.68 GiB**, ~1.7 GiB headroom. Step scratch measured 1.26 MiB.

---

## Phase roadmap

Status: [ ] todo  [~] in progress  [x] done  (renumbered 2026-09-13, decision 11)

- [~] **P0. Setup** — CUDA/nvcc/CMake build working. (done: vectorAdd runs)
- [x] **P1. Model loading** — mmap, safetensors header parse, weight registry
      (632 → typed LM/vision/MTP views), single-slab GPU upload. `dump()` + self-checks.
- [x] **P2. NumPy reference (closed, superseded)** — 01 embed/RMSNorm/head and 02 attention
      done vs HF; 03 stopped before code. Findings live in Target facts.
- [x] **P3. Oracle + engine skeleton** — HF bf16 activation dumps (safetensors), CMake
      lib + `deltaforge_test`, Tensor/arena/cuBLAS context, bf16 helpers, first kernels:
      embed, RMSNorm, tied head. → brief 04 **DONE 2026-10-29**
- [ ] **P4. Full-attention decoder layer, step form** — projections, gated q, QK-norm,
      RoPE, GQA, KV cache, output gate, MLP, residuals. Layers 3 + 7 match. → brief 05
- [ ] **P5. GatedDeltaNet decoder layer, step form** — projections, conv1d state, gated
      norm (06); delta-rule step + recurrent state (07).
- [ ] **P6. Text bring-up** — 24 layers, prefill = loop of step, greedy decode id-for-id vs
      HF bf16 `generate`, per-layer drift table. → brief 08  ← **first text milestone**
- [ ] **P6.5. Tokenizer (C++ byte-level BPE)** — ids match HF on a test corpus. No
      dependencies; run any time before P6 closes. → brief 09
- [ ] **P7. Speed** — batched attention prefill + chunked GDN, validated vs the P6 step loop;
      measure tokens/s + VRAM (resume numbers). → brief 10
- [ ] **P8. Vision tower** — patch embed, ViT blocks, merger → image tokens into text stream.
- [ ] **P9. mrope + multimodal fusion** — position handling for mixed image/text sequences.
- [ ] **P10. MTP / speculative decoding** — the multi-token head, for throughput.

(P8–P10 are the rest of the engine, not optional extras — same pattern: HF dumps → CUDA.)

---

## Session log (append-only)

| Date | Session | Outcome / notes |
|------|---------|-----------------|
| 2026-08-26 | setup + loader | Build toolchain, RAII mmap wrapper, safetensors header parse, full weight catalogue (58 roles) derived. Started weight-mapping brief. |
| 2026-08-31 | 00 weight mapping | **P1 done.** `Model` class: registry (632 `TensorView`), single-slab GPU upload (4.236 GiB, ~564 KB overhead, round-trip exact), typed LM+vision+MTP views, `dump()`. All 632 tensors claimed exactly once; ctor-time self-checks pass. Corrected q_proj→[4096,2048], size→GiB. Next: P2 backbone (NumPy-vs-HF). |
| 2026-09-01 | housekeeping | Repo renamed `start-inferencing` → **`deltaforge`**, `git init` + first commit (`2bf0182`). P1 code split out of `main.cpp` into `include/{mapped_file,tensor_view,dtype,model_types,model}.hpp` (+ vendored `json.hpp`); `src/main.cpp` is now just GPU probe + `Model` ctor/selfCheck/dump. `src/kernel.cu` empty placeholder. No P2 work yet. |
| 2026-09-03 | 01 embed/RMSNorm/head | **P2 slice 1 done.** `python/harness.py`: NumPy-vs-HF harness (`load_raw`/`load_hf`(cached fp32 CPU)/`compare`) + ops `embed`, `rmsnorm`, `tied_head`. All 5 tasks pass — embed & RMSNorm bit-exact, head 1.6e-6, tie proven by shared `data_ptr`. **Trap found:** `Qwen3_5RMSNorm` is zero-centered (`1.0 + weight`, init 0) — all plain norms; the lone exception is `linear_attn.norm` (`RMSNormGated`, plain `w` + silu-gate, F32). bf16→fp32 via torch `framework="pt"`. Next: 02 attention. |
| 2026-09-13 | 02 attention block | **P2 slice 2 done.** `python/harness_attn.py`: **`attn(x, pos, layer_idx)`** matches HF layer-3 `self_attn` at **4.92e-07** (layer 7: 6.41e-07), budget 1e-3. All 5 stages verified in isolation + 5 negative controls that must XFAIL (naive q/gate split 6.0e0, plain-w norm 7.0e0, rotate-all-256 7.9e0, np.tile KV 3.0e0). `python harness.py` runs 01+02 green. **Refinements:** `rotary_emb` returns `[b,seq,64]` not 3-row mrope — `apply_interleaved_mrope` (modeling:148) collapses it, and that line is the whole P7 hook; `inv_freq` has 32 entries not 64; `self_attn.forward` needs a batch axis; attn-output merge and gate are both head-major and must align. **Trap:** running `harness.py` as `__main__` made `harness_attn`'s `from harness import` load the model a second time (~17 GB) — fixed by aliasing `sys.modules`. Next: 03 MLP + layer assembly. |
| 2026-10-29 | 04 oracle + skeleton | **P3 done.** `python/dump_oracle.py --dtype bf16\|fp32` → 196-tap dumps (49.1 / 80.2 MiB), all internal checks green, GDN = torch fallback (metadata `gdn_impl`). C++: `SafeTensors` (mmap + header + registry + `__metadata__`), `manifest.hpp` (generic printing), `DeviceBuffer`/`Arena`/`Tensor`/`Context`, `bf16.hpp`, `compare.hpp`, `model_checks.hpp`; CMake split into `deltaforge_core` + `infer` + `deltaforge_test` + CTest (red proven). **Kernels: embed and RMSNorm bit-exact vs the dump; tied head rel 2.747e-03 with argmax matching on all 30 rows.** Arena: 10k resets = 0 allocations, step high-water 1.26 MiB. ctx 32768 → 4.68 GiB total. Next: brief 05. |
| 2026-09-13 | 03 stopped → pivot | **NumPy-first retired** (decision 11): too slow for the value; the bugs were caught by per-stage HF comparison, not by the reimplementation. 03 stopped after pre-flight, which caught two brief errors (KV = 24 KiB/token fp32 not 12; 6144 = 3×2048 not 8/3×2048) and two harness traps (square causal mask only correct for S_q=1; shared failure list). Roadmap renumbered: P3 oracle+skeleton → P6 text bring-up (5 sessions + tokenizer). Next: brief 04. |

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
5. ~~Correctness path = NumPy-vs-HF before any CUDA kernel.~~ (2026-08-26) **Superseded by 11.**
6. **Oracle = full HF `AutoModelForImageTextToText` on CPU fp32** (the
   `ForConditionalGeneration` wrapper, so `.lm_head` and `.model.language_model` exist;
   attr path = disk key with one extra `.model`). Raw weights via
   `safe_open(framework="pt").get_tensor(k).float().numpy()` — numpy can't hold bf16. (2026-09-03)
7. *(Superseded by 13 for the CUDA path.)* **Tolerance budget:** fp32-native ops expect ~0 (HF also runs them fp32); bf16-weight
   GEMMs get atol 1e-3. A miss by ≫ that is a *semantics* bug, not rounding. (2026-09-03)
8. **Tokenizer deferred to P4.5, not P2.** Per-op NumPy checks are driven by ids /
   random hidden states, so a tokenizer adds no correctness signal until real prompts
   (P4) and is only *required* for the C++ engine (P5). Embedding lookup itself was
   done in 01. (2026-09-04) *Now brief 09 / P6.5, independent.*
9. **Every harness stage ships a negative control** (`check_differs` → `XFAIL`): the wrong
   variant (naive q/gate split, `np.tile` KV, plain-`w` norm, rotate-all-256) must *fail*.
   Same-shape bugs are invisible to shape checks; controls are the only proof the test
   can see them. (2026-09-13)
10. **One harness per concept file, one runner.** `harness_<x>.py` imports `harness.py`
   (never the reverse, except inside `harness.py`'s `__main__`); `python harness.py` runs
   everything and exits non-zero. Alias `sys.modules["harness"]` to `__main__` there or
   the 8.5 GB oracle loads twice. Run with `python/.venv/Scripts/python.exe` from `python/`. (2026-09-13)
   *(Legacy NumPy harness only.)*
11. **Direct CUDA, validated against HF activation dumps** (supersedes 5). HF runs once per
    dump on CPU with hooks; C++ compares each stage. No NumPy reimplementation. (2026-09-13)
12. **Bring-up is step form only.** Every layer is written as a one-token step with its cache;
    prefill = loop of steps. HF full-sequence row *t* equals step output *t*, so one dump
    covers all checks. Batched prefill + chunked GDN come in P7, validated vs the loop. (2026-09-13)
13. **Oracle dtype = engine dtype.** Dump HF in **bf16** (plus one fp32 anchor dump). Activations
    are bf16 on device; kernels upcast to fp32 exactly where HF does (see Target facts) and
    cast back with PyTorch's rounding. Tolerance is **relative** per stage (max|d| / max|ref|);
    bf16 budget calibrated in brief 04. (2026-09-13)
14. **Dumps are safetensors**, read by the existing C++ parser. Naming `ids`, `embed`,
    `L{ii}.{stage}`, `final_norm`, `logits`, `L{ii}.cache.{k,v,conv,recurrent}`. Dumps are
    gitignored and regenerated by one command. (2026-09-13)
15. **Engine skeleton fixed once (brief 04):** batch 1, fixed max context, one activation arena
    (no device malloc in the decode loop), one cuBLAS handle, cuBLAS for every GEMM, ~10
    custom kernels. `deltaforge_test` (CTest) re-runs every stage each session; negative
    controls carry over from decision 9. (2026-09-13)
16. **Briefs tag research as *gating* or *optional*;** only gating items block a milestone.
    Each brief carries all known traps forward. (2026-09-13)
17. **Rule of zero for every resource.** `MappedFile`, `DeviceBuffer` (and later the cuBLAS
    handle/stream) own one thing each and free it in their destructor; owners like `Model`
    hold them by value, declare no destructor, and delete copying. No `new`/`delete`, and
    never an explicit `~T()` call. (2026-10-03)
18. **Every kernel ships a negative control AND a hand-computable toy case.** The control
    (wrong variant must XFAIL) proves the check can see the bug; the toy case (e.g. the
    `2x3 @ (4x3)^T` head) proves the index/layout mapping independent of the model, so a
    GEMM failure separates "my lda is wrong" from "my weights are wrong". (2026-10-29)
19. **`compare` is relative and fp64.** Tolerances are `max|d| / max|ref|`; NaN/Inf
    disagreement can never be forgiven by a tolerance. Known gap to close in 05:
    `compareToRef` checks element *count*, not shape, so a transposed output
    (`[8,T,256]` vs `[T,8,256]`) would pass — add per-dim checking before the attention
    layout work. (2026-10-29)

---

## Active session briefs
- `docs/sessions/00-weight-mapping.md` — registry + typed views (P1). **DONE 2026-08-31.**
- `docs/sessions/01-embed-rmsnorm.md` — NumPy harness + embed/RMSNorm/tied head (P2). **DONE 2026-09-03.**
- `docs/sessions/02-attention-block.md` — NumPy full-attention sub-block (P2). **DONE 2026-09-13.**
- `docs/sessions/03-mlp-decoder-layer.md` — **STOPPED 2026-09-13** (pivot). Pre-flight notes harvested into facts + brief 05.
- `docs/sessions/04-oracle-skeleton.md` — HF bf16 dumps + engine skeleton + test runner + embed/RMSNorm/head kernels (P3). **DONE 2026-10-29.**
- (next) 05 full-attn layer step form (P4) — taps already in the dump: `input_norm`, `q_proj`, `q`, `gate`, `k`, `v`, `q_normed`, `k_normed`, `q_roped`, `k_roped`, `core_out`, `gated`, `mixer_out`, `resid1`, `post_norm`, `mlp_out`, `cache.k`, `cache.v` for layers 3/7/11/15/19/23.
- (queued) 06 GDN projections/conv/gated norm · 07 GDN delta-rule step (P5) · 08 text bring-up (P6) · 09 tokenizer (P6.5, any time) · 10 speed (P7).
