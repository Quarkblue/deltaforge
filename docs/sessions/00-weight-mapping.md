# Session 00 — Weight Mapping / Registry (Phase 1)

**Kickoff prompt (paste into a fresh `claude` in this repo):**
> Read `docs/sessions/00-weight-mapping.md` and `docs/PROJECT.md`, then help me
> build the weight registry in C++. Teaching style: pointers + done-when
> checkpoints, don't hand me whole files unless I ask.

**Goal:** Turn the safetensors tensors into a typed, GPU-resident model the
forward pass can index by role — not raw pointers.

**Prereqs already done:** mmap wrapper (`include/mapped_file.hpp`), safetensors
header parse (8-byte length → JSON), nlohmann/json wired, CMake build works.

---

## Facts this session needs (from PROJECT.md)

- 632 tensors, 1 shard file, mostly BF16, **some F32** (`A_log`, gated `norm.weight`).
- `lm_head` is tied → not on disk; alias `embed_tokens`.
- Layer types from `config.text_config.layer_types` — drive the union off this,
  do NOT guess from which tensors exist.

### Reference shapes (assert against these) — verified against the real header
| tensor | shape | dtype |
|---|---|---|
| `embed_tokens.weight` | [248320, 2048] | BF16 |
| `norm.weight` (final) | [2048] | BF16 |
| full-attn `q_proj` | **[4096, 2048]** | BF16 |
| full-attn `o_proj` | [2048, 2048] | BF16 |
| full-attn `k_proj` / `v_proj` | [512, 2048] | BF16 |
| full-attn `q_norm`/`k_norm` | [256] | BF16 |
| mlp `gate`/`up` | [6144, 2048] | BF16 |
| mlp `down` | [2048, 6144] | BF16 |
| gdn `in_proj_qkv` | [6144, 2048] | BF16 |
| gdn `conv1d` | [6144, 1, 4] | BF16 |
| gdn `in_proj_a` / `in_proj_b` | [16, 2048] | BF16 |
| gdn `A_log` | [16] | **F32** |
| gdn `dt_bias` | [16] | BF16 |
| gdn `in_proj_z` | [2048, 2048] | BF16 |
| gdn `norm.weight` | [128] | **F32** |
| gdn `out_proj` | [2048, 2048] | BF16 |

---

## Build tasks (ordered, each with done-when)

1. **DType enum + element size.** `{ BF16, F32 }`, `elem_size(dt)`.
   *Done when:* parsing the header "dtype" string maps to the enum, and
   `nbytes = numel * elem_size(dtype)`.
2. **`TensorView { void* dptr; int64_t shape[5]; int ndim; DType dtype; size_t nbytes; }`.**
   (`shape[5]` — vision `patch_embed.proj.weight` is 5-D `[1024,3,2,16,16]`.)
   *Done when:* one header entry → a filled TensorView (dptr still host at this stage).
3. **Registry = `unordered_map<string, TensorView>`; fill from every shard in the index.**
   *Done when:* all 632 names present; assert `numel*elem_size == offset span` for every one.
4. **GPU upload.** Per tensor `cudaMalloc` + `cudaMemcpy` H2D from mmap ptr; store device ptr.
   Add a `CUDA_CHECK` macro. *Done when:* summed nbytes == **4548144832 (4.236 GiB)** and
   `cudaMemGetInfo` shows VRAM dropped ~that much. (Doc's old "4.55 GB" was decimal GB;
   `cudaMemGetInfo` reports GiB — compare like for like.)
5. **Typed views — language model** (`RMSNorm`, `MLP`, `FullAttn`, `LinearAttn`,
   `DecoderLayer` with `variant<FullAttn,LinearAttn>`, `LanguageModel`). Resolve
   names → views; pick mixer from `layer_types[i]`. *Done when:* every field
   non-null for its layer type.
6. **Typed views — vision + MTP** (in scope; the goal is the full multimodal engine,
   so map these now even though they run in later phases). `LayerNorm` view (weight
   **and bias** — vision uses bias, text does not), `VisionBlock`, `VisionModel`
   (patch_embed, pos_embed, blocks, merger), `MTPHead`. *Done when:* all 632 tensors
   are referenced by exactly one typed view; nothing left unclaimed.
7. **`dump()`** — per module print type + each tensor's shape+dtype.
   *Done when:* full-attn layer shows q/k/v/o, linear layer shows A_log/conv1d, a
   vision block shows the biases, none mixed up.

**Session milestone:** `dump()` covers one full-attn layer, one linear-attn layer, and
one vision block, all matching real shapes; **every one of the 632 tensors is claimed by
a typed view** (assert count). Report back here.

---

## Research (light — mostly mechanical this session)
- safetensors dtype strings ("BF16","F32",...) → confirm the exact spellings in the header.
- `std::variant` basics if unfamiliar (tagged union for the two mixer types).

---

## Progress / notes (fill in during the session)
- [x] **task 1 — DType + helpers.** `enum class DType { BF16, F32 }`; `elem_size`
      (switch, throws on unknown) and `dtype_from_string` (throws with the bad
      string in the message). File has only "BF16"/"F32" — confirmed.
- [x] **task 2 — TensorView.** `{ const uint8_t* host_ptr; void* device_ptr;
      int64_t shape[5]; int ndim; DType dtype; size_t nbytes; }`. `shape[5]` (not 4)
      for the 5-D vision kernel. `MAX_DIM=5` constant; overflow guard on the shape walk.
- [x] **task 3 — Registry.** `Model` class owns `MappedFile` (declared first →
      destroyed last), parses header once (json dies as a local), builds
      `unordered_map<string,TensorView>`. Per-tensor cross-check
      `numel*elem_size == data_offsets[1]-data_offsets[0]` passes for all 632.
      `host_ptr = mapped.data() + data_offset(76656) + offsets[0]`. Public `at(name)`
      rethrows with the missing name. Verified: **632 tensors, Σnbytes = 4548144832
      (4.236 GiB), A_log = [16] F32, 64 B, at file offset 0.**
- [x] **task 4 — GPU upload.** Went with the **single-slab** approach (one
      `cudaMalloc(total)` + one `cudaMemcpy` H2D straight from the mmap blob, no
      full-file heap read — honors PROJECT.md decision #3). Each tensor's
      `device_ptr = slab + start_offset` (stored `start_offset`/`end_offset` on
      `TensorView`; `start_offset == data_offsets[0]`). `CUDA_CHECK` macro added.
      `model_device_ptr` is a member; destructor `cudaFree`s it once — per-tensor
      `device_ptr`s are non-owning views (interior pointers, never freed).
      **Verified: VRAM drop 4,548,722,688 B vs 4,548,144,832 payload = ~564 KB
      overhead (0.01%); A_log round-trip byte-exact.** Slab beats per-tensor here
      (632 mallocs would each round up to alloc granularity).
- [x] **task 5 — typed LM views.** `buildLanguageModel` loop resolves names →
      `RMSNorm`/`MLP`/`FullAttn`/`LinearAttn`/`DecoderLayer`; mixer arm picked off
      `layer_types_[i]` (`full_attention` → FullAttn, `linear_attention` → LinearAttn,
      else throw). Views store `const TensorView*` into `tensors_` (not copies) so
      `lm_head` **aliases** `embed_tokens` (same pointer; `tie_word_embeddings` true).
      `view(name)` helper = `&at(name)` (throws on a bad name → no silent null).
      **Verified `verifyLanguageModel`:** variant arm matches config for all 24;
      FullAttn exactly at {3,7,11,15,19,23}; sampled shapes/dtypes match the table;
      `lm_head == embed_tokens` pointer-identity holds.
- [x] **task 6 — typed vision + MTP views.** New leaves `LayerNorm{weight,bias}` +
      `LinearWB{weight,bias}` (vision has **biases** everywhere, text does not).
      `VisionBlock` (norm1/2, attn_qkv/proj, mlp_fc1/2), `VisionModel` (patch_embed
      5-D + pos_embed weight-only + 24 blocks + merger), `MTP` (fc, two pre_fc norms,
      one full-attn `DecoderLayer`, norm). Prefix gotcha: **mtp.* has no `model.`**.
      **Verified:** `verifyVisionAndMTP` (per-field non-null + sampled shapes:
      qkv[3072,1024], patch_embed[1024,3,2,16,16], pos_embed[2304,1024], mtp.fc[2048,4096]);
      `verifyAllTensorsClaimed` — **all 632 claimed exactly once**, lm_head tie
      excluded from the counter (it's the alias). Nothing unclaimed / double-claimed.
- [x] **task 7 — `dump()`.** `printTensor` (`label -> [dims] DTYPE`, `<null>` safe) +
      `dumpDecoderLayer`/`dumpVisionBlock`; `dump()` prints one of each (LM linear L0,
      LM full L3, vision block 0, MTP layer) + top-level singletons. Added
      `dtype_to_string` (inverse of `string_to_dtype`).
- decisions taken:
  - Registry is dtype-agnostic; upload is a raw byte-copy (`nbytes` only).
    `__nv_bfloat16` is a Phase-5 device concern, not a loader concern.
  - `DType` kept as pure identity; size/behavior hang off it via functions
    (so F16/quant types can be added without collisions).
  - Typed views hold **non-owning `const TensorView*`** into `tensors_`, never
    copies — light, identity-comparable, and lets the tie share one pointer.
  - Two leaf structs (`LayerNorm`, `LinearWB`) with identical layout but distinct
    names — role/readability over collapsing them.
- surprises / mismatches (corrections to the brief above):
  - **`q_proj` is `[4096, 2048]`**, not `[2048,2048]` — `attn_output_gate: true`
    means q_proj emits q **and** a gate (2 × 8 heads × 256). o_proj stays [2048,2048].
  - **`mtp.layers` has exactly ONE layer (index 0)**, full-attn shaped. Don't loop N.
  - Header keys come back **alphabetically sorted** (nlohmann default `json` uses
    `std::map`), NOT file order. Offsets are non-sequential in key order — always
    use each tensor's own `data_offsets[0]`, never an accumulator.
  - This shard has **no `__metadata__`** key (the sibling index.json does).
  - 632 = 320 language_model + 297 visual + 15 mtp. Dtypes: 596 BF16 / 36 F32.

## Handoff (copy up to PROJECT.md when done)
**Session 00 COMPLETE — all 7 tasks done & verified.** Built in `src/main.cpp`:
- `Model` class: mmap → safetensors header parse → `unordered_map<string,TensorView>`
  registry (632) → single-slab GPU upload → typed views (LM + vision + MTP) → `dump()`.
- Milestone met: `dump()` covers one full-attn / one linear-attn / one vision block,
  shapes match; **every one of the 632 tensors claimed by exactly one typed view**.
- Self-checks live in the ctor path (`verifyLanguageModel`, `verifyVisionAndMTP`,
  `verifyAllTensorsClaimed`) — keep them until the forward pass exists, then optional.

**Corrections to fold into PROJECT.md:**
- q_proj is **[4096,2048]** (gated), not [2048,2048] (PROJECT.md line 38 is stale).
- Weights total **4548144832 B (4.236 GiB)**; "4.55 GB" was decimal GB (line 25).
- Upload is **single-slab** (one malloc + one memcpy), not per-tensor — supersedes
  decision #3's "per-tensor cudaMemcpy" wording (still honors "no heap read").

**Next session to open:** P2 — decoder backbone, NumPy-vs-HF reference first
(embedding → RMSNorm → GEMM → attention/GQA/RoPE/QK-norm → SwiGLU → tied head).
Open a new `docs/sessions/01-*.md` brief.
