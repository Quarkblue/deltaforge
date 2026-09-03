# Session 02 — Full-Attention Block (Phase 2, slice 2)

**Kickoff prompt (paste into a fresh `claude` in this repo):**
> Read `docs/sessions/02-attention-block.md` and `docs/PROJECT.md`, then help me
> build the NumPy reference for one full-attention sub-block (layer 3) — projections,
> gated q, QK-RMSNorm, partial RoPE, GQA causal softmax, output gate, o_proj — validated
> against the HF module. Teaching style: research pointers + done-when checkpoints, don't
> hand me whole files unless I ask. Fill in this brief's Progress/Handoff as we go.

**Goal:** `attn(x, pos) → y` in NumPy that matches `hf.model.language_model.layers[3].self_attn`
to GEMM tolerance. This is the densest op cluster in the LM; everything here reappears in
the vision tower (P6) and mrope (P7), so learn it once, properly.

**Out of scope this session:** `input_layernorm` / `post_attention_layernorm`, residual
adds, the MLP, KV cache, multi-token generation. Those are 03. Test the *sub-block*
`self_attn` on its own — feed it a random hidden state, not real tokens.

**Prereqs (01):** `python/harness.py` — `load_raw`, `load_hf`, `compare`, and a
`rmsnorm` that is **already zero-centered `(1+w)`**. Reuse it for q_norm/k_norm.

---

## Facts this session needs (from PROJECT.md — verified in HF source 09-03)

- Layer 3 disk keys: `model.language_model.layers.3.self_attn.{q_proj,k_proj,v_proj,o_proj,q_norm,k_norm}.weight`.
  HF attr: `hf.model.language_model.layers[3].self_attn`. Linear weights are `[out, in]`
  (torch layout) → `y = x @ W.T`. No biases.
- Shapes: q_proj **[4096, 2048]**, k/v_proj **[512, 2048]**, o_proj **[2048, 2048]**,
  q_norm/k_norm **[256]**. 8 Q heads, 2 KV heads, head_dim 256 → GQA group = 4.
- **Gated q (the trap):** q_proj's 4096 outputs are viewed as `[heads=8, 512]` and
  `chunk(2, dim=-1)` → per head: `q = cols[0:256]`, `gate = cols[256:512]`. It is *not*
  "first 2048 = q, last 2048 = gate". The gate is flattened back to `[seq, 2048]` and
  applied as `attn_out * sigmoid(gate)` **after** attention, **before** o_proj.
- QK-norm: RMSNorm over the 256-dim head, per head, applied *before* RoPE. Zero-centered.
- RoPE: `rotary_dim = int(256 * 0.25) = 64`. Only `q[..., :64]` / `k[..., :64]` are
  rotated (`rotate_half` convention: pair `i` with `i+32`), the other 192 dims pass
  through untouched. `theta = 1e7`, `inv_freq[i] = theta^(-2i/64)`, i in [0,32).
- Text-only positions: `position_ids = arange(seq)`. HF builds a 3-row mrope position
  tensor (T/H/W) but for pure text all three rows are identical → mrope collapses to
  plain RoPE. **Don't** implement mrope sections now; note where it plugs in for P7.
- Scale `1/sqrt(256) = 1/16`. Causal mask (lower-triangular). Softmax in fp32.
- Order in HF forward: proj → split q/gate → q_norm/k_norm → transpose to
  `[heads, seq, 256]` → RoPE → repeat KV ×4 → SDPA → merge heads → `* sigmoid(gate)` → o_proj.

---

## Build tasks (ordered, each with done-when)

1. **Projections + split.** `x [seq, 2048]` → q `[seq, 8, 256]`, gate `[seq, 8, 256]`,
   k `[seq, 2, 256]`, v `[seq, 2, 256]`. *Done when:* your `q` and `gate` match a
   forward hook on HF's `q_proj` output re-split the same way (`view(seq, -1, 512)`), and
   the wrong split (halves) visibly *doesn't* match — prove the trap to yourself.
2. **QK-norm.** Apply `rmsnorm` per head with `q_norm.weight` / `k_norm.weight`.
   *Done when:* matches `self_attn.q_norm(q_view)` / `k_norm(k_view)` (atol 1e-5).
3. **RoPE, partial, rotate_half.** Build `cos/sin [seq, 64]` from `inv_freq` and
   `pos = arange(seq)`; rotate the first 64 dims of q and k. *Done when:* (a) matches
   HF `apply_rotary_pos_emb` on the same q/k (get `cos,sin` from
   `hf.model.language_model.rotary_emb(x, position_ids)` — check what shape it
   returns and whether it's already the 3-row mrope form), atol 1e-5; (b) you can state
   in one sentence why RoPE is applied *after* QK-norm and why partial rotary leaves
   192 dims position-blind.
4. **GQA causal attention.** Expand k/v from 2 → 8 heads (`repeat`, each KV head serves
   4 consecutive Q heads — verify the *grouping order*, it's `repeat_interleave`, not
   `tile`). Scores `q@k.T/16`, causal mask, fp32 softmax, `@v`. *Done when:* matches
   `torch.nn.functional.scaled_dot_product_attention(..., is_causal=True, enable_gqa=True)`
   on your own q/k/v tensors, atol 1e-5.
5. **Output gate + o_proj + full sub-block.** Merge heads → `[seq, 2048]`,
   `* sigmoid(gate)`, `@ o_proj.T`. Then `attn(x, pos)` end-to-end vs
   `self_attn(hidden_states=x, position_embeddings=(cos,sin), attention_mask=None)` —
   check the exact kwarg names in the source; you may need to pass a causal mask or set
   `attn_implementation="eager"` when loading. *Done when:* atol 1e-3 on `seq ≈ 8–16`.
   Record the actual max|d|.
6. **Regression hook.** Add these as tasks to `harness.py` (or a `harness_attn.py`
   that imports it) so 03 can call `attn()` as a black box. *Done when:* `python harness.py`
   runs 01 + 02 tasks green in one go.

**Session milestone:** `attn(x, pos, layer_idx)` matches HF layer-3 `self_attn` at 1e-3;
each of the 5 stages verified in isolation; gate split + KV grouping order + rotary dim
written down as facts. Bonus: run it on layer 7 too (same code, different weights).

---

## Research (each has a done-when)
- **RoPE** — RoFormer paper §3.4 + HF `rotate_half` / `apply_rotary_pos_emb` in
  `modeling_qwen3_5.py`. *Done when:* you can derive why rotating (q, k) by their
  positions makes `q·k` depend only on the *relative* offset, and why `rotate_half`
  pairing (i, i+d/2) is just a layout choice equivalent to adjacent-pair complex rotation.
- **Partial rotary + huge theta** — Qwen3 / Qwen3-Next model cards. *Done when:* one
  sentence each on what `partial_rotary_factor=0.25` and `theta=1e7` buy (long context).
- **GQA** — Ainslie et al. 2023 (GQA paper). *Done when:* you can say what it saves at
  inference (KV-cache size — compute the bytes for this model at 32k ctx) and why quality
  holds.
- **QK-norm** — Qwen3 tech report §2 / "Scaling ViT" (Dehghani 2023). *Done when:* you
  can state what it prevents (attention-logit growth) and why it must precede RoPE.
- **Attention output gating** — Qwen3-Next blog / "Gated Attention" (Qiu et al. 2025).
  *Done when:* you can say what the sigmoid gate on the attention output is for
  (attention-sink / massive-activation mitigation) — it's *why* q_proj is 4096 wide.
- **Numerics** — *Done when:* you know why softmax needs max-subtraction in fp32 and
  what `atol` a chain of 4 bf16-weight GEMMs realistically lands at (expect 1e-4–1e-3).

---

## Progress / notes (fill in during the session)
- [ ] task 1 …
- decisions taken:
- surprises / mismatches:

## Handoff (copy up to PROJECT.md when done)
- what got built (`attn()` signature, where it lives), per-stage max|d|, any facts that
  contradict PROJECT.md, and what 03 needs (MLP + layer assembly + causal loop).
