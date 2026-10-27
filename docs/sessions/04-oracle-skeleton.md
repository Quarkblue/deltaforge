# Session 04 — HF Oracle Dumps + Engine Skeleton + First Kernels (Phase 3)

**Kickoff prompt (paste into a fresh `claude` in this repo):**
> Read `docs/sessions/04-oracle-skeleton.md` and `docs/PROJECT.md` (decisions 11–16 are the
> new plan), then help me build the HF bf16 activation dumper, restructure the CMake build
> into a core library plus a `deltaforge_test` runner, set up the engine skeleton (Tensor,
> arena, cuBLAS context, bf16 helpers), and write the embed / RMSNorm / tied-head kernels
> validated against the dumps. Teaching style: research pointers + done-when checkpoints,
> don't hand me whole files unless I ask. Fill in this brief's Progress/Handoff as we go.

**Goal:** everything every later session stands on. After this, adding a component is
"tap it in the dumper → write the kernel → add a `compare` line", with the whole engine
re-verified by one command.

**Out of scope:** attention, MLP, GDN, decode loop, tokenizer. No layer kernels yet.

**If the session runs long:** stop cleanly after task 5 (infra done) and hand off. Tasks
6–7 (kernels) can open the next session. Infra correctness matters more than speed here.

---

## Facts this session needs (from PROJECT.md)

- **Existing C++:** `include/model.hpp` — `Model(folder, shard, config)` mmaps the shard,
  parses the safetensors header into `TensorView`s (`host_ptr`, `device_ptr`, `shape`,
  `ndim`, `dtype`, `nbytes`), uploads one slab, `at(name)` looks up a view. Typed structs in
  `model_types.hpp`. `DType` is only `{BF16, F32}` (`dtype.hpp`). CMake has one exe `infer`
  built from `src/main.cpp` + empty `src/kernel.cu`, links cuBLAS, sm_89, C++17.
- **Oracle (decisions 6, 13):** `AutoModelForImageTextToText`, but now `dtype=torch.bfloat16`,
  `attn_implementation="eager"` (so `eager_attention_forward` runs and can be tapped). CPU.
  Also produce one fp32 dump as the anchor.
- **HF dtype policy to mirror (Target facts):** norms compute in fp32 and cast back; softmax
  fp32; delta rule fp32; cos/sin fp32 then cast. Read `Qwen3_5RMSNorm.forward`
  (`modeling_qwen3_5.py:720-738`) for the *exact* cast order before writing the kernel.
- **Dump naming (decision 14):** `ids`, `embed`, `L{ii}.{stage}`, `final_norm`, `logits`,
  `L{ii}.cache.{k,v,conv,recurrent}`. Batch axis stripped. Gitignored under `dumps/`.
- **Known traps carried forward:** RMSNorm is `(1 + w)`; attr path is disk key with an extra
  `.model`; HF modules assume a batch axis; `lm_head` is tied and not on disk.
- **VRAM:** ~6.9 GB free at start, weights take 4.24 GiB. Budget everything else against ~2.5 GiB.

---

## Build tasks (ordered, each with done-when)

1. **Oracle dumper** — `python/dump_oracle.py [--dtype bf16|fp32] [--out dumps/...]`.
   Fixed plain-text prompt of ~16–32 tokens (store the string in safetensors metadata).
   Capture with forward hooks and forward *pre*-hooks, plus thin wrappers around
   `apply_rotary_pos_emb` and `eager_attention_forward`. For this session you need at least
   `ids`, `embed`, every `L{ii}.in` / `L{ii}.out`, `final_norm`, `logits`. Also tap now
   (cheap, needed by 05): per attn layer `input_norm`, `q`, `gate`, `k`, `v`, `q_normed`,
   `k_normed`, `q_roped`, `k_roped`, `core_out`, `gated`, `mixer_out`, `resid1`, `post_norm`,
   `mlp_out`; and the end-of-prompt cache (`use_cache=True`). Keep taps in one registry so
   06/07 add GDN internals by appending, not rewriting.
   *Done when:* both dumps exist; a printed manifest lists name/shape/dtype; reloading
   `logits` from the file equals the in-memory model output bit-for-bit; and you've logged
   **which GDN implementation ran** (`use_kernel_func_from_hub_with_fallback`, line 248 — it
   must be the torch fallback on CPU, or the dump isn't the math you'll reimplement).
2. **Reusable safetensors reader.** Pull mmap + header parse + `TensorView` registry out of
   `Model` into a `SafetensorsFile` class. `Model` keeps GPU upload and typed views on top.
   Extend `DType` with whatever the dump needs (ids: `I64` or `I32`).
   *Done when:* `infer` still passes `selfCheck()` unchanged, and a tiny test opens the dump
   file and prints its manifest from C++ matching the Python one.
3. **CMake split.** Static library `deltaforge_core` (headers, loader, `.cu` kernels),
   executables `infer` and `deltaforge_test`, `enable_testing()` + `add_test`.
   *Done when:* both build from a clean `build/`, and `ctest` runs `deltaforge_test` and
   reports its exit code (a failing test must make `ctest` fail — prove it once).
4. **Engine skeleton.** `Tensor` (device ptr, shape, dtype; views vs owning), `Arena` (one
   device allocation at startup, bump-allocate, `reset()` per step), `Context` (cuBLAS handle,
   stream, arena). Batch 1; pick a max context and **write the VRAM budget table**: weights,
   KV cache at max ctx (24 KiB/token fp32, 12 KiB bf16), GDN state per layer ×18 (compute it
   from the GDN shapes in Target facts), activation scratch for one step, cuBLAS workspace.
   *Done when:* the table is in Progress, the arena allocates and frees once, and a stress
   loop of 10k `reset()` cycles makes no device allocations (check with a counter, not faith).
5. **bf16 helpers + test runner.**
   - `bf16_to_f32` / `f32_to_bf16`, callable on host and device, matching PyTorch casting.
     *Done when:* **exhaustive** test over all 65,536 bf16 bit patterns → fp32 matches
     `torch` exactly (generate the reference table in Python once), and fp32 → bf16 matches
     `torch` on the dumped tensors (including rounding ties, inf, NaN).
   - `compare(name, device_tensor, ref_view, rel_tol)`: copy to host, widen to fp64, report
     `max|d|`, `max|ref|`, `rel`; `expect_differs(...)` for negative controls (`XFAIL`, or
     `BAD` if it unexpectedly matches); failure count → process exit code.
     *Done when:* self-compare is `OK`, a one-element perturbation is `FAIL`, a negative
     control shows `XFAIL`, and `ctest` goes red on the failure.
6. **Kernels: embed + RMSNorm.** Embedding gather from the bf16 table into the arena;
   zero-centered RMSNorm mirroring HF's cast order.
   *Done when:* `embed` vs dump `embed` is **bit-exact**; `rmsnorm(L23.out)` with the final
   norm weight vs dump `final_norm` at rel ≤ 1e-5 (record the actual; expect bit-exact if the
   cast order matches). Negative control: plain-`w` norm must `XFAIL`.
7. **Tied head via cuBLAS.** `logits = h @ embed.T` with `cublasGemmEx` on bf16.
   Work out the column-major mapping on paper first: which op flags and leading dimensions
   express a row-major `[out, in]` weight. *Done when:* vs dump `logits` at a relative
   tolerance you calibrate and **record** (this is the bf16 GEMM budget every later session
   uses); argmax of each row matches the dump's argmax. Negative control: the wrong op flag
   (transposed) must `XFAIL` or fail on shape.

**Session milestone:** one command (`ctest`) rebuilds nothing unnecessary and prints a green
table for embed, final norm, and head; both dumps regenerate with one Python command; VRAM
budget and bf16 GEMM tolerance are recorded as numbers.

---

## Research (gating = blocks the milestone; optional = read if curious)

- **[gating] cuBLAS `cublasGemmEx` + column-major convention** — official cuBLAS docs
  (NVIDIA docs, "cuBLAS Library → cublasGemmEx", supported type/compute combinations table).
  *Done when:* you can say which `Atype/Btype/Ctype` + `computeType` combos accept
  `CUDA_R_16BF` on compute capability 8.9, and write the call for `y = x Wᵀ` with every
  dimension argument justified.
- **[gating] bfloat16 bit layout + PyTorch cast semantics** — IEEE 754 binary32 vs Google
  Brain float (1 sign, 8 exponent, 7 mantissa). *Done when:* you can explain why bf16 → fp32
  is exact (same exponent width) and what fp32 → bf16 does to the dropped 16 mantissa bits.
- **[gating] CUDA kernel launch + indexing** — official CUDA C++ docs / NVIDIA "An Even
  Easier Introduction to CUDA". *Done when:* you can pick block/grid sizes for a
  `[seq, 2048]` elementwise kernel and state what happens to out-of-range threads.
- **[gating] PyTorch hooks** — `register_forward_hook`, `register_forward_pre_hook` (PyTorch
  docs, `torch.nn.Module`). *Done when:* you know how to capture a module's input, not only
  its output (that's how `resid1` and `gated` are captured).
- **[optional] CTest basics** — CMake docs `add_test`, `enable_testing`.
- **[optional] Arena / bump allocation** — why device allocation in a hot loop is slow
  (it can synchronize); how llama.cpp's `ggml` uses a fixed compute buffer.

---

## Progress / notes (fill in during the session)
- [x] task 1 dumper — prompt, token count, which GDN impl ran, dump sizes
  - **DONE 2026-09-15.** `python dump_oracle.py --dtype bf16|fp32` (from `python/`, venv).
    Script bf16 output **bit-identical** to the notebook dump. `oracle_bf16.safetensors`
    49.1 MiB (177 BF16 / 18 F32 / 1 I64); `oracle_fp32.safetensors` **80.2 MiB**
    (195 F32 / 1 I64). Both: 196 tensors, 0 failing checks.
  - Built as `python/dump_oracle.ipynb` for iteration (script form later). Prompt: "Hello,
    introduce yourself. by letting me know all the features that you have and can do. also
    let me know the best way to use you." → **30 tokens**. bf16 run: logits dtype **bf16**.
  - Step 2 done: 52 taps (`ids`, `embed`, `L{ii}.in/out`, `final_norm`, `logits`);
    `embed == L00.in` and `L{i}.out == L{i+1}.in` bit-exact; `L23.out != final_norm`.
  - Step 3 (2026-09-14): 12 attn taps × 6 layers → 124 tensors, 26.9 MiB bf16. Bug found:
    gate hook used `shape[:1]` instead of HF's `shape[:-1]` → stored `[61440]` not `[30,2048]`.
    Fixed (`[:-1]`); raw `L{ii}.q_proj` tap added for the wrong-split negative control.
  - Steps 4-5 written (not yet run): module-global patch of `apply_rotary_pos_emb`,
    `eager_attention_forward`, `torch_chunk_gated_delta_rule` (counter only), restored in
    `finally`; layer index for RoPE via a `self_attn` pre-hook. GDN impl read from the
    decorator closure's `implementation`. Cache taps from `out.past_key_values.layers[i]`
    (`.keys/.values` attn, `.conv_states[0]/.recurrent_states[0]` GDN). Expect 196 tensors.
    Checks cell: residual adds, module re-runs, gate = `core_out * sigmoid(gate)`,
    `cache.k == k_roped`, `cache.v == v` reshaped, naive split XFAIL, call counts 6/6/18.
  - Tap layouts (batch stripped, HF order, no reshaping): `q/q_normed [T,8,256]`,
    `k/k_normed [T,2,256]`, `v [T,512]`, `gate/gated [T,2048]`, `q_proj [T,4096]`,
    `q_roped [8,T,256]`, `k_roped [2,T,256]`, `core_out [T,8,256]`, `cache.k/v [2,T,256]`.
  - **Steps 3-5 green (2026-09-15):** 196 tensors, **49.1 MiB** bf16, 0 failing checks, calls
    6/6/18, naive gate split XFAIL on all 6 layers, reload bit-exact.
  - **GDN impl = torch fallback** for all four (`torch_chunk_gated_delta_rule`,
    `torch_recurrent_gated_delta_rule`, `causal_conv1d_fn`, `causal_conv1d_update` all resolve
    to `transformers.models.qwen3_5.modeling_qwen3_5.*`). Stored in file metadata `gdn_impl`.
  - Cache: attn `cache.k/v [2,30,256]` BF16; GDN `cache.conv [6144,4]` BF16,
    `cache.recurrent [16,128,128]` **F32**. Dump dtypes: 177 BF16, 18 F32, 1 I64.
  - `python/dump_oracle.py` written (2026-09-15; notebook kept as scratch): `--dtype bf16|fp32`,
    `--out` (default `dumps/oracle_<dtype>.safetensors`), `--compare <file>` (bit-identical
    check), `--quiet`. All checks run in-script, incl. tap count (196) and call counts; exits
    non-zero on any FAIL/BAD, reload mismatch or compare mismatch. Paths resolved from the
    script location. Syntax + `--help` verified; not yet run on the model.
- task 2 traps found in pre-read of `model.hpp` (2026-09-15):
  - `readModelWeights` iterates every header key; the dumps carry a `__metadata__` key
    (no `dtype`/`shape`) → `.at("dtype")` throws. Must skip it, and expose it (prompt, dtype).
  - nlohmann `json` iterates keys **sorted**, not file order → C++ manifest order ≠ Python
    insertion order; compare manifests as sets / sorted.
  - `ndim < 1` throws: fine for these dumps (no scalars), but safetensors allows shape `[]`.
- [~] task 2 SafetensorsFile
  - `include/safetensors.hpp`: `SafeTensors` class (mmap + header parse + name->TensorView +
    `__metadata__`), `at/view/tensors/metadata/sortedNames`. `DType` gained `I64`.
  - `include/manifest.hpp` (2026-09-20): generic printing, free functions in `namespace manifest`
    — `shapeToString`, `printTensor` (ref + null-safe pointer overload), `printManifest(os, st,
    prefix)`, `printSummary`, `printMetadata`. No per-module print code; prefix `"L03."` or
    `"model.language_model.layers.3."` replaces the old hand-written dump helpers.
  - **Manifest check green:** C++ manifest of `oracle_bf16` diffs clean against Python's
    (196 lines, `name	shape	DTYPE`, sorted). Summary: 196 tensors, 51493104 bytes,
    BF16 177 / F32 18 / I64 1. Metadata (prompt, gdn_impl, versions) reads back from C++.
  - `Model` verification moved out to `include/model_checks.hpp` (free functions taking
    `const Model&`; `Model` gained const getters). `Model` not yet switched to `SafeTensors`.
- [x] task 3 CMake split + red/green ctest proof — **DONE 2026-09-24**
  - `deltaforge_core` STATIC (src/kernel.cu + headers) carries includes, `CUDA::cublas`,
    sm_89, C++17, `/Zc:preprocessor` and `DELTAFORGE_ROOT=L"<src dir>"` as **PUBLIC**, so
    `infer` (src/main.cpp) and `deltaforge_test` (tests/test_main.cpp) just link it.
  - `enable_testing()` + `add_test`; run with `ctest --test-dir build -C Debug`.
    Exit code = number of failed checks. Project renamed `inferencing` -> `deltaforge`.
  - First tests (no GPU): both dumps open, 196 tensors, ids I64[30], logits shape/dtype per
    dtype, `L00.cache.recurrent` F32[16,128,128], metadata prompt/dtype/gdn_impl, device
    pointers null before `bindToDevice`, `sortedNames` sorted, unknown name throws,
    missing file throws, fp32 dump has zero BF16.
  - **Red proven:** expecting 195 tensors -> 2 FAIL lines, `0% tests passed`, ctest red;
    reverted -> green. `DeviceBuffer` + `cuda_check.hpp` added; `Model` is rule-of-zero.
- [x] task 4 skeleton + VRAM budget table — **DONE 2026-10-03**
  - **Max context chosen: 32768** (2026-09-26; briefly 16384 at a 250 MB KV budget, then
    raised). Model allows 262144, so the cap is memory only.
  - **VRAM budget (batch 1, bf16 activations, ctx 32768):**
    | Item | Size | Computed from |
    |---|---|---|
    | Weights (one slab) | 4338 MiB | measured, 4548144832 B |
    | KV cache | 384 MiB | 6 attn layers x 2 kv heads x 256 x 2 (k+v) x 2 B = 12 KiB/token x 32768 |
    | GDN recurrent states | 18 MiB | 18 x 16 x 128 x 128 x 4 B (F32) |
    | GDN conv states | 0.84 MiB | 18 x 6144 x 4 x 2 B |
    | Activation arena | 16 MiB reserved | **measured 1.26 MiB/step** (30 tokens) |
    | cuBLAS workspace | 32 MiB reserved | |
    | **Total** | **~4789 MiB (4.68 GiB)** | ~1.7 GiB headroom vs ~6.4 GiB free |
  - Per-step scratch: attn scores 8 x ctx fp32 = 1 MiB and logits 485 KiB (bf16) dominate;
    everything else (hidden/q/k/v/MLP/GDN temporaries) is tens of KiB. Only the KV cache and
    the attn-score buffer scale with context; GDN state is fixed-size by construction.
  - **Two lifetimes, two allocations:** arena = activations, `reset()` per step; KV cache and
    GDN states live in their own persistent buffers (resetting them would wipe decode state).
  - Code: `device_buffer.hpp` (RAII cudaMalloc/cudaFree, move-only, static alloc/free counters),
    `arena.hpp` (bump allocator over one DeviceBuffer, 256-byte align, `allocTensor`, `reset`,
    `highWater`), `tensor.hpp` (`Tensor` = non-owning device view + `numel_of`/`nbytes_of`/
    `to_shape`/`fromView`), `context.hpp` (cuBLAS handle + own stream + arena ref, `sync()`),
    `cuda_check.hpp`, `cublas_check.hpp`.
  - **Arena proof green:** 10k alloc+`reset()` cycles -> **zero** cudaMalloc calls, stable
    `highWater`, plus a negative control (a DeviceBuffer inside the loop) that XFAILs.
    Measured step high-water **1,319,168 B (1.26 MiB)** of 16 MiB for a 30-token step.
  - Counter checks also cover move semantics (one alloc / one free, no double free).
- [x] task 5 bf16 exhaustive test + compare/expect_differs — **DONE 2026-10-03**
  - `include/bf16.hpp`: `bf16_to_f32` / `f32_to_bf16` over CUDA's `__bfloat162float` /
    `__float2bfloat16` (RNE), host+device via a `DF_HD` macro; `python/gen_bf16_table.py`
    writes `dumps/bf16_table.safetensors` (widen[65536], cast_in/cast_out edge+random cases).
  - Green: widening matches torch on **all 65536** patterns; round-trip identity for every
    finite pattern; narrowing matches torch on every test value; control — truncation agrees
    on only 6234/12309 values (XFAIL), which proves the table can separate RNE from truncation.
  - **Trap:** CUDA's `__float2bfloat16` encodes NaN as 0x7FFF, torch 2.13 CPU as **0xFFFF**;
    `f32_to_bf16` special-cases NaN to match torch (the oracle), so host and device differ in
    NaN payload only.
  - `include/compare.hpp`: `compareToRef` -> `CompareResult{n, max|d|, max|ref|, rel, worst
    index/values, nan_both, nan_one, inf_mismatch}`, both sides widened to **fp64**, NaN/Inf
    counted not maxed, `within(tol)` refuses to forgive a NaN/Inf disagreement; absolute
    fallback flagged when max|ref| == 0. `compareExactI64` for ids. Test-side `compare` /
    `expect_differs` print the numbers on pass as well as fail.
  - Green: self-compare exact; one-element perturbation detected **and located**; negative
    control XFAIL; size mismatch and float-compare-on-I64 both throw.
  - **Measured: bf16 run vs fp32 anchor at `final_norm` = rel 4.106e-03** (max|d| 0.261,
    max|ref| 63.68). End-to-end divergence of the two oracle runs after 24 layers; kernels are
    compared against the bf16 dump, so anything approaching this is a semantics bug.
  - **Known gap:** `compareToRef` checks element *count*, not shape -> a transposed output
    (e.g. `[8,T,256]` vs `[T,8,256]`) would pass. Add per-dim checking + an explicit
    flat/reshape opt-in before the layout-heavy checks in 05.
- [x] task 6 embed gather + RMSNorm — **DONE 2026-10-12**
  - `include/kernels.hpp` declares the launchers in plain C++ (tests/ and `main.cpp` are host-
    compiled and cannot parse `<<<>>>`); the `__global__` kernels stay in `src/kernel.cu`. Every
    launcher validates dtype/ndim/shape on the host (a kernel cannot throw), queues on
    `ctx.stream()`, and checks `cudaGetLastError()` immediately — a bad launch config fails
    asynchronously and would otherwise be blamed on the next unrelated CUDA call.
  - `embedGather`: one thread per output element, `t = i / hidden`, `c = i % hidden`, so
    consecutive threads read consecutive table columns (coalesced). A pure 2-byte copy — no
    widening, no arithmetic — so the result must be bit-exact.
    **Measured: rel = 0.000e+00** against `dump["embed"]` on the real 248320x2048 table.
    Control: ids rotated by one → XFAIL, which proves the check is sensitive to *which* row
    each token asks for, not just to the shapes.
  - `rmsNorm`: one block per row, 256 threads, strided fp32 sum of squares → `__shfl_down_sync`
    within each warp → `__shared__ float warp_sums[32]` → thread 0 folds `n_warps` partials and
    broadcasts `scale = 1/sqrtf(total/cols + eps)` through shared memory. Weight applied as
    `(1.0f + w)` in fp32, with a **single** bf16 cast at the very end.
    **Measured: rel = 0.000e+00 — bit-exact**, all 61440 elements of
    `rmsnorm(L23.out, norm.weight, eps=1e-6)` vs `dump["final_norm"]` (max|ref| 63.75). The
    cast order matches `Qwen3_5RMSNorm.forward` exactly.
    Control: `unit_offset=false` (plain `w`, no `1.0 +`) → XFAIL at rel 3.201e-01.
  - **Trap found by the poison convention:** the control initially wrote into the *wrong* output
    tensor, leaving the second buffer at its `0xFF` poison. It still XFAILed — but with
    `NAN_ONE_SIDE=61440` and `max|ref|=0`, i.e. "a buffer of NaN differs from the dump", which is
    vacuous. `within()` refusing to forgive NaN is what made this visible in the printed numbers;
    an XFAIL line alone would have looked identical. Read the numbers, not just the verdict.
  - **Residual:** `rmsNormKernel`'s shuffle loop assumes a full warp (`0xffffffff`). The launcher
    hard-codes 256 threads and rejects any non-multiple of 32, so this is unreachable today; it
    must be revisited before any kernel here takes a dynamic block size.
  - `checkEmbedGather()` uploads the full 1017 MiB table, so it is **commented out** in `main`
    after one green run. Put it behind a `DELTAFORGE_SLOW` env gate rather than a comment.
- [ ] task 7 tied head — calibrated bf16 GEMM tolerance
- pre-flight (2026-09-13): transformers 5.15.1, torch 2.13.0+cu132. `fla`, `causal_conv1d`,
  `kernels` NOT installed → decorator (`integrations/hub_kernels.py:764`) should resolve to the
  torch fallback; still confirm at runtime. Line refs 248 / 554 / 604 / 720-738 verified.
- decisions taken:
- surprises / mismatches:
  - GDN impl is chosen at **import time** (decorator), and `functools.wraps` makes
    `__name__`/`__wrapped__` report the torch function whichever impl won → don't trust the name.
  - Notebook trap: hooks registered in one cell and removed after a forward that raised
    were never removed → every re-run stacked a duplicate set ("L00.in already exists").
    Register + forward + remove in one cell with `try/finally`.
  - **GDN recurrent state stays F32 in HF's cache even in a bf16 run** (delta rule upcasts,
    never casts the state back). Engine must keep it F32 → budget 16·128·128·4 B = **1 MiB**
    per GDN layer × 18 = **18 MiB** (conv state bf16: 6144·4·2 B = 48 KiB × 18 = 0.84 MiB).
    (An earlier note here said 256 KiB / 4.5 MiB, an arithmetic slip; the file sizes confirm 18 MiB.)
    Task 2's C++ `DType` needs `I64` for `ids`; `F32` already exists.
  - `self_attn` / `linear_attn` get `hidden_states` as a **kwarg** (decoder `forward:776`);
    decoder layers get it positionally (`TextModel:1199`). Pre-hooks need `with_kwargs=True`.

## Handoff (copy up to PROJECT.md when done)
- Dumper command + tap registry shape; C++ public surface (`SafetensorsFile`, `Tensor`,
  `Arena`, `Context`, `compare`, kernels); VRAM budget table; calibrated tolerances
  (fp32-internal stages vs bf16 GEMM); any facts contradicting PROJECT.md; what brief 05
  needs (which taps already exist, max context chosen).
