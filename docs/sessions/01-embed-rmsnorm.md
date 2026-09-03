# Session 01 — Embedding + RMSNorm + Tied Head (Phase 2, slice 1)

**Kickoff prompt (paste into a fresh `claude` in this repo):**
> Read `docs/sessions/01-embed-rmsnorm.md` and `docs/PROJECT.md`, then help me
> build the NumPy reference harness and the embedding / RMSNorm / tied-head ops,
> validated against HuggingFace. Teaching style: research pointers + done-when
> checkpoints, don't hand me whole files unless I ask.

**Goal:** Stand up the **NumPy-vs-HF correctness harness** and verify the three ops
that *bracket* the transformer stack — token embedding, RMSNorm, and the tied
`lm_head` projection. Once P2/P3 fill the middle (attention, MLP, GatedDeltaNet),
these are the top and bottom of the P4 end-to-end match.

**Why these three first:** they're the simplest ops, they exercise the whole
load→numpy→compare loop, and they let you confirm the tie (`lm_head` = `embed_tokens`)
that P1 asserted by pointer — now confirm it *numerically*.

**Prereqs already done (P1):** registry + typed views in `include/model.hpp`
(`model_types.hpp` for the typed structs, `tensor_view.hpp` for the view); weights on
GPU; `python/.venv`; `python/reference.py` holds the module repr. mmap/safetensors
understood. Python harness files go in `python/` (nothing there yet besides `reference.py`).

---

## Facts this session needs (from PROJECT.md)

- `embed_tokens.weight` **[248320, 2048]** BF16. `lm_head` **tied** → same tensor,
  no bias. Logits = `h @ embed_tokens.weight.T`.
- Final `norm` = **RMSNorm**, **eps 1e-6**, dim 2048, weight-only (no bias, no mean-centering).
- HF path to the modules (from `reference.py`):
  `model.model.language_model.embed_tokens`, `...language_model.norm`, `model.lm_head`.
- **The trap:** HF `Qwen3_5RMSNorm` **upcasts to float32** internally, normalizes,
  then casts back. Do the math in fp32 or you won't hit tolerance.

---

## Build tasks (ordered, each with done-when)

1. **Harness / weight access in Python.** Load the raw tensors two ways: (a) HF
   `state_dict()` from a CPU fp32 model, (b) `safetensors` numpy loader by name.
   *Done when:* `embed_tokens.weight` from both agrees (compare in fp32; expect exact
   after a single bf16→fp32 cast). Decide which source is your "ground truth" (HF)
   vs your "own impl input" (raw).
2. **Embedding lookup.** Gather rows by `input_ids`. Compare to
   `embed_tokens(input_ids)`. *Done when:* `np.allclose` in fp32 (it's a pure gather —
   should match to the cast).
3. **RMSNorm (fp32).** `y = x * rsqrt(mean(x², axis=-1) + eps) * weight`, computed in
   float32. Feed a random `[seq, 2048]` hidden state through both yours and the HF
   `norm` module. *Done when:* `allclose(atol=1e-5)`. Write down *why* it's rsqrt-of-
   mean-square (no subtract-mean, no bias) — that's the RMSNorm-vs-LayerNorm point.
4. **Tied head.** `logits = h @ embed_tokens.weight.T` (no bias). Compare to
   `model.lm_head(h)`. *Done when:* allclose AND you've confirmed
   `lm_head.weight is embed_tokens.weight` (or byte-equal) — the tie, numerically.
5. **End-to-end stub (plumbing only).** `ids → embed → final norm → head → logits`,
   shape `[seq, 248320]`, on a 1–2 token input. *Done when:* it runs and the shape is
   right. **NOT a correctness check** — no layers in between yet; note that explicitly
   so future-you doesn't think a mismatch here is a bug.

**Session milestone:** three ops each `allclose` against HF in isolation, tie confirmed
numerically, harness reusable for later slices (a `load_ref()` + `compare(name, mine, theirs)`
you'll keep calling in 02/03). Report back here.

---

## Research (do these; each has a done-when)
- **RMSNorm** — read the HF `Qwen3_5RMSNorm.forward` source (transformers
  `modeling_qwen3*.py`) and the RMSNorm paper's definition.
  *Done when:* you can state (a) why the upcast to fp32 matters, (b) why there's no
  mean-subtraction or bias.
- **`safetensors` numpy API** — official docs `safetensors.numpy.load_file` /
  `safe_open`. *Done when:* you load one tensor by name without the full file in RAM.
- **bf16 vs fp32 rounding** — *Done when:* you can explain why comparisons run in fp32
  and what `atol` is realistic for a bf16-weight matmul (~1e-2/1e-3), vs an fp32-native
  op like RMSNorm (~1e-5).

---

## Progress / notes (2026-09-03 — DONE, all 5 tasks pass)
- [x] task 1 — raw (safetensors) vs HF embedding weight agree in fp32. `max|d|=0` (exact).
- [x] task 2 — embedding gather `weight[ids]` vs HF `embed_tokens`. `max|d|=0`.
- [x] task 3 — RMSNorm (fp32) vs HF `norm`. `max|d|=0` after the fix below.
- [x] task 4 — tied head `h @ Wᵀ` vs HF `lm_head`. `max|d|=1.61e-6`; tie proven via
  `lm_head.weight.data_ptr() == embed_tokens.weight.data_ptr()` → `True`.
- [x] task 5 — end-to-end stub, shape `(2, 248320)`. Plumbing only (no layers yet).
- decisions taken:
  - Oracle = full HF model on CPU fp32 via `AutoModelForImageTextToText` (resolves
    `Qwen3_5Config` → the full `ForConditionalGeneration`, so `.lm_head` +
    `.model.language_model` exist). ~8.5 GB RAM, cached with `lru_cache`.
  - bf16→fp32 cast route = `safe_open(framework="pt").get_tensor(k).float().numpy()`.
    numpy's own loader can't materialize bf16 (`ml_dtypes` not installed, not needed).
  - Disk keys are `model.language_model.*` (no `lm_head` on disk — tied). Attr path on
    the loaded object is `hf.model.language_model.*` — one extra `.model` vs the key.
- surprises / mismatches:
  - **RMSNorm is zero-centered.** `Qwen3_5RMSNorm` scales by `(1.0 + weight)`, param
    init 0 (Gemma-style delta-from-1). First impl used `weight` → `max|d|=3.96`. Fix:
    `x * (1.0 + weight)`. Applies to input/post-attn layernorms, q_norm, k_norm, and
    the final norm — every plain RMSNorm in the model.
  - **One exception:** `linear_attn.norm` (inside GatedDeltaNet) is `Qwen3_5RMSNormGated`
    — NOT zero-centered (plain `w`, init 1) and adds a `silu(gate)` multiply. It's also
    the only norm stored in F32. Session 03 needs a *second* norm fn; don't reuse this one.
  - fp32-native ops came out bit-exact (0.00), not just within tol — HF does them in fp32
    too, so no divergent rounding. The head's 1.6e-6 is pure matmul reduction-order drift.

## Handoff (copy up to PROJECT.md when done)
- **Built:** `python/harness.py` — reusable harness + the 3 ops.
  - API: `load_raw(key)→fp32 np` · `load_hf()→cached HF model` · `raw_keys()` ·
    `compare(name, mine, theirs, atol)→bool` · ops `embed`, `rmsnorm`, `tied_head`.
  - Accessors `hf_embed/hf_norm/hf_head(model)` hide the attr-path vs disk-key gap.
- **Tolerances that held:** fp32-native ops (embed, RMSNorm) exact (atol 0); bf16-origin
  matmul (head) 1.6e-6 under a 1e-3 budget. Use ~1e-3 for GEMMs, ~1e-5 for fp32 ops in 02.
- **Next:** 02 — full-attention block (q/k/v proj, GQA 8Q/2KV, RoPE θ=1e7 partial 0.25,
  per-head QK-RMSNorm over 256 — reuse `rmsnorm`, it's zero-centered). Same
  load_raw/load_hf/compare loop, layer 3 (first full-attn layer) as the test target.
