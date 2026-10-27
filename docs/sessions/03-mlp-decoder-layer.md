# Session 03 — SwiGLU MLP + Decoder-Layer Assembly + KV-Cache Decode (Phase 2, slice 3)

> **STOPPED 2026-09-13 before any code — superseded by decision 11 (direct CUDA vs HF dumps).**
> Do not run this brief. Its pre-flight findings (bottom of file) were harvested into
> PROJECT.md Target facts and carry forward into brief 05 (full-attention layer, step form).

**Kickoff prompt (paste into a fresh `claude` in this repo):**
> Read `docs/sessions/03-mlp-decoder-layer.md` and `docs/PROJECT.md`, then help me build
> the NumPy reference for the SwiGLU MLP, assemble a full-attention decoder layer
> (pre-norm + residuals) around the `attn()` from session 02, and add an incremental
> KV-cache decode path that matches the full-sequence result. Validate against HF
> `layers[3]`. Teaching style: research pointers + done-when checkpoints, don't hand me
> whole files unless I ask. Fill in this brief's Progress/Handoff as we go.

**Goal:** `decoder_layer(h, pos, i) → h'` for any full-attention layer `i`, matching HF
`Qwen3_5DecoderLayer` to GEMM tolerance, plus `attn_step(x_t, pos_t, cache)` — the
one-token-at-a-time form that P5 will actually run. After this, the only missing LM
piece is GatedDeltaNet (P3); P4 wires embed → 24 layers → norm → head.

**Out of scope:** GatedDeltaNet layers (P3), the 24-layer forward and greedy loop (P4),
tokenizer (P4.5). Don't try to run a real prompt through — 18 of 24 layers don't exist yet.

**Prereqs (01/02):** `harness.py` (`load_raw`, `load_hf`, `compare`, `rmsnorm` —
zero-centered), `harness_attn.py` (`attn(x, pos, layer_idx)` black box, plus `linear`,
`project_q/kv`, `split_q_gate`, `qk_norm`, `to_head_major`, `rope_cos_sin`, `apply_rope`,
`repeat_kv_np`, `causal_mask`, `softmax`, `sigmoid`, `merge_heads`, `gqa_attend`,
`check`, `check_differs`). Follow decision 10: new file `harness_layer.py` imports
those; `harness.py`'s `__main__` gains one more import. Run from `python/` with the venv.

---

## Facts this session needs (from PROJECT.md — verified in HF source 09-13)

- **MLP** (`Qwen3_5MLP`, `modeling_qwen3_5.py:704-718`): `down_proj(silu(gate_proj(x)) * up_proj(x))`.
  `gate_proj`/`up_proj` **[6144, 2048]**, `down_proj` **[2048, 6144]**, no biases,
  `hidden_act = "silu"`. Same class on all 24 layers (GDN layers have an MLP too).
- **Decoder layer** (`Qwen3_5DecoderLayer`, 740-794), pre-norm, two residuals:
  ```
  r = h;  h = input_layernorm(h);  h = self_attn(h);            h = r + h
  r = h;  h = post_attention_layernorm(h);  h = mlp(h);         h = r + h
  ```
  Both norms are `Qwen3_5RMSNorm` [2048], zero-centered — reuse `rmsnorm`. `self_attn`
  returns a tuple `(hidden, weights)`; HF takes `[0]`.
- Disk keys: `model.language_model.layers.{i}.{mlp.gate_proj,mlp.up_proj,mlp.down_proj,input_layernorm,post_attention_layernorm}.weight`.
  HF attr: `hf.model.language_model.layers[i].{mlp,input_layernorm,post_attention_layernorm}`.
- Calling HF `layers[3](...)` needs the batch axis (02 refinement 3), `position_embeddings=(cos[None], sin[None])`,
  and the additive causal mask `[1,1,S,S]` — same as 02 task 5.
- **KV cache per full-attn layer:** k and v after QK-norm + RoPE, `[2 kv_heads, seq, 256]`
  each (RoPE is applied *before* caching, so cached k already carries its position).
  12 KiB per token across the 6 attn layers (fp32; halve for bf16 later).

---

## Build tasks (ordered, each with done-when)

1. **SwiGLU MLP.** `mlp(x, i)` with `silu(z) = z * sigmoid(z)` (reuse 02's overflow-safe
   `sigmoid`). *Done when:* matches `hf_layer(m, 3).mlp(x)` at atol 1e-3 on random
   `x [12, 2048]` (it's a 6144-wide reduction — expect ~1e-5, record it). Negative
   control: `gelu`, or `silu(up) * gate` (swapped), must XFAIL. Write down why SwiGLU has
   *two* up-projections and what the sigmoid gate does that a plain ReLU MLP doesn't.
2. **Pre-norm decoder layer.** `decoder_layer(h, pos, i)` per the fact block, using
   `attn()` as a black box. *Done when:* matches `hf_layer(m, 3)(h[None], position_embeddings=..., attention_mask=...)`
   at 1e-3 on `h [12, 2048]`; also layer 7. Negative control: post-norm order
   (norm *after* the residual add) must XFAIL. Note: HF's `layers[3].forward` may return a
   tensor or a tuple depending on version — check the source, not the type.
3. **Isolated stage checks.** Feed HF's own intermediates to confirm the residual points:
   `input_layernorm(h)` and `post_attention_layernorm(r)` both at 1e-5. *Done when:* green,
   and you can say why pre-norm (not post-norm) is the modern default (gradient path
   through the residual stream is identity).
4. **Incremental decode with a KV cache.** Split 02's `attn()` into
   `attn_prefill(x, pos, i, cache) → (y, cache)` and `attn_step(x_t, pos_t, i, cache) → (y_t, cache)`,
   where cache holds roped-and-normed `k`, `v` `[2, T, 256]` per layer. Decode step attends
   over `T+1` positions with **no mask needed** (the new token is last; everything cached
   is causal by construction). *Done when:* prefill on tokens `0..7` then 4 single-token
   steps reproduces `attn(x[0:12], pos[0:12])` **row-for-row** at 1e-5. Then the same
   through `decoder_layer`. Negative control: forgetting RoPE on the step's `q`/`k` (or
   using `pos=0` for every step) must XFAIL.
5. **Layer dispatcher stub + memory tally.** `forward_layer(h, pos, i, cache)` that calls
   `decoder_layer` when `is_full_attn(i)` and raises `NotImplementedError("P3")` otherwise.
   Compute and print: KV-cache bytes at 4096 and 32768 context for the 6 attn layers
   (fp32 and bf16). *Done when:* the numbers are in the Handoff; you can state why the 18
   GDN layers need a *fixed-size* state instead (their whole point).
6. **Regression hook.** `harness_layer.main()` → failure count; wire into `harness.py`
   `__main__`. *Done when:* `python harness.py` runs 01+02+03 green, one model load.

**Session milestone:** `decoder_layer` matches HF on layers 3 and 7 at ≤1e-3 (record actual);
`attn_step` prefill+decode equals full-sequence attention row-for-row; MLP/residual facts
and KV-cache sizing written down. Hand off the `harness_layer.py` public surface.

---

## Research (each has a done-when)
- **SwiGLU** — Shazeer 2020 "GLU Variants Improve Transformer" + HF `ACT2FN["silu"]`.
  *Done when:* you can explain the gate/up split and why intermediate is 6144 (≈ 8/3 × 2048
  to keep parameter count equal to a 4× ReLU MLP).
- **Pre-norm vs post-norm** — Xiong et al. 2020 "On Layer Normalization in the Transformer
  Architecture". *Done when:* one sentence on why pre-norm trains without warmup and why
  the final `norm` before the head exists at all under pre-norm.
- **KV cache** — HF `DynamicCache.update` (`cache_utils.py`), or the vLLM/PagedAttention
  paper §2. *Done when:* you can state (a) what is cached (post-RoPE k, post-norm v),
  (b) why decode needs no causal mask, (c) the bytes-per-token formula
  `2 · layers_attn · kv_heads · head_dim · dtype_bytes`, and (d) why GQA (2 KV heads not 8)
  is the reason this model's cache is 4× smaller than MHA would be.
- **Residual stream as the "bus"** — Elhage et al. 2021 "A Mathematical Framework for
  Transformer Circuits", residual-stream section. *Done when:* you can say why each
  sub-block *adds* to `h` rather than replacing it — this is why GDN and attention layers
  can be interleaved freely in P3/P4.

---

## Progress / notes (fill in during the session)
- [ ] task 1 SwiGLU MLP (layer 3) — max|d| = ?
- [ ] task 2 decoder_layer (layers 3, 7) + post-norm XFAIL — max|d| = ?
- [ ] task 3 isolated norm stages
- [ ] task 4 attn_prefill / attn_step row-for-row + RoPE XFAIL; then through decoder_layer
- [ ] task 5 forward_layer stub + KV tally
- [ ] task 6 main() wired into harness.py
- decisions taken:
- surprises / mismatches (pre-flight, 09-13, before any code):
  - **Task 2 note resolved:** installed `Qwen3_5DecoderLayer.forward` (modeling:753-793) returns a
    **tensor**, not a tuple. Trap: `out[0]` still "works" (drops the batch axis) — coincidence, not a tuple.
  - **Brief fact suspect — KV bytes/token:** formula `2·6·2·256·dtype_bytes` does not give
    "12 KiB fp32". Recompute in task 5 and correct the fact block.
  - **Brief fact suspect — 6144 = 8/3×2048:** 8/3×2048 ≈ 5461; 6144 = 3×2048. Param count of
    3-matrix SwiGLU at 6144 vs 2-matrix 4× MLP is *not* equal. Recompute in research item.
  - **`gqa_attend` hard-codes a square `causal_mask(S_q)`** (harness_attn:404). Step (S_q=1)
    gets a 1×1 zero mask that broadcasts — passes by accident. A multi-token chunk after a
    non-empty cache would be wrong. Mask must be `[S_q, S_k]` offset by cache length.
  - **`_failures` lives in `harness_attn`**: `harness_layer` must append to that same list
    (import it, don't rebind/clear a copy) or 03 failures won't count.

## Handoff (copy up to PROJECT.md when done)
- `harness_layer.py` public surface (`mlp`, `decoder_layer`, `attn_prefill`, `attn_step`,
  cache layout), per-stage max|d| on layers 3 + 7, KV-cache byte numbers, any facts that
  contradict PROJECT.md, and what P3's first GDN brief needs from this file.
