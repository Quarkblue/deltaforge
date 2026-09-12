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
- **RoPE** — RoFormer §3.4. **DONE.** Rotating q at position m by m*theta and k at n by
  n*theta makes their dot product a function of (m-n)*theta: absolute position in,
  relative distance out. `rotate_half`'s (i, i+32) pairing is the same rotation as
  adjacent-pair complex form under a permuted basis — chosen because it makes both
  halves contiguous slices instead of strided ones.
- **Partial rotary + huge theta** — **DONE.** `partial_rotary_factor=0.25` leaves 192 of
  256 dims position-blind, so each head keeps content-only matching capacity alongside its
  64 positional dims. `theta=1e7` (vs the usual 1e4) stretches the slow end of the
  frequency range so the longest wavelength still resolves distinct positions out to
  `max_position_embeddings=262144`.
- **GQA** — Ainslie et al. 2023 (GQA paper). **DONE.** KV cache at 32k ctx: only the 6
  full-attn layers cache (the 18 GatedDeltaNet layers hold constant-size state).
  `2 (k,v) x 2 kv_heads x 256 head_dim x 2 B` = 2048 B/token/layer x 6 = **12 KiB/token**
  -> **384 MiB at 32768 tokens**. MHA (8 kv heads) would be 4x = **1.5 GiB**; with 4.24 GiB
  of weights on an 8 GB card that is the difference between 32k fitting and not. Quality
  holds because the 4 Q heads in a group keep distinct query projections — only K and V
  are shared.
- **QK-norm** — ViT-22B §2.2 / Qwen3 report §2. **DONE.** Prevents attention-logit growth:
  unbounded |q||k| drives softmax into saturation where gradients vanish. Must precede
  RoPE because RMSNorm's *division* is rotation-invariant but its learned per-channel
  `(1+w)` scale is a diagonal matrix, and diagonals do not commute with rotations — so
  norm-then-rotate != rotate-then-norm, and the model was trained norm-first.
- **Attention output gating** — Qiu et al. 2025. **DONE.** Softmax rows must sum to 1, so a
  head with nothing useful to attend to still dumps its mass somewhere (usually position 0)
  -> attention sinks and massive activations that wreck quantization. A per-channel sigmoid
  gate lets the head scale its own output toward zero instead. The gate is computed from the
  same projection as q, which is exactly why q_proj is 4096 = 8 heads x (256 q + 256 gate).
- **Numerics** — **DONE.** exp() overflows fp32 above ~88; subtracting the row max is
  algebraically free (`exp(a-m)/sum exp(a-m)` == `exp(a)/sum exp(a)`) and caps every exponent
  at 0 so the largest term is exactly 1.0. Measured ladder this session: weights bit-exact
  (0.0) -> one 2048-term GEMM ~1e-6 -> per-head RMSNorm ~1e-6 -> RoPE 0.0 (same op order)
  -> attention vs SDPA ~5e-7. Semantics bugs all landed at 1e0-1e1, i.e. **6-7 orders above
  the noise floor** — there is no ambiguous middle ground, so never raise atol to pass.

---

## Progress / notes (fill in during the session)

**Source index** — `python/.venv/Lib/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py`
(the model dir has no remote code; HF ships `qwen3_5` natively, v4.57.0.dev0):
| lines | thing |
|---|---|
| 84–166 | `Qwen3_5TextRotaryEmbedding` (inv_freq + mrope forward) |
| 546–552 | `rotate_half` |
| 554–590 | `apply_rotary_pos_emb` (the mrope-section variant) |
| 592–602 | `repeat_kv` |
| 604–627 | `eager_attention_forward` |
| 629–702 | `Qwen3_5Attention` (`__init__` + `forward`) |
| 720–738 | `Qwen3_5RMSNorm` (zero-centered) |
| 740–794 | `Qwen3_5DecoderLayer` (session 03's job) |

- [x] task 0 scaffold — `python/harness_attn.py`. Config-derived constants
      (n_kv_groups 4, rotary_dim 64, scale 0.0625), `layer_key`/`hf_attn`/`is_full_attn`,
      fixtures (SEQ=12, seed 2, X[12,2048] fp32, POS=arange). `task0_shapes()`:
      all six layer-3 weights match expected shapes AND are **bit-exact raw-vs-HF**
      (max|d|=0.0). Module confirmed `Qwen3_5Attention`.
- [x] task 1 projections + gated-q split
  - [x] 1a `linear(x, w) = x.f32 @ w.f32.T` + `project_kv(x, i)`. k/v vs HF:
        **max|d| 9.5e-07 / 8.3e-07** at atol 1e-4.
  - [x] 1b split read from source (`modeling_qwen3_5.py:669-674`):
        `q_proj(x).view(*input_shape, -1, head_dim*2)` -> `[B,S,8,512]`, then
        `torch.chunk(.., 2, dim=-1)` -> q `[B,S,8,256]` (first), gate `[B,S,8,256]` (second);
        gate then `.reshape(*input_shape, -1)` -> `[B,S,2048]`, head-major.
        **Column map:** head h -> q `[h*512, h*512+256)`, gate `[h*512+256, (h+1)*512)`.
        Head 0's q = cols `[0:256)`.
        **Why the trap hides:** under the naive halves split head 0's q is ALSO cols
        `[0:256)` — identical. Only heads 1-7 diverge. Spot-checking head 0 proves nothing.
  - [x] 1c `split_q_gate` (reshape to `[-1, 8, 512]` FIRST, then slice `[..., :256]` /
        `[..., 256:]`) vs `split_naive` (slice `[:, :2048]` / `[:, 2048:]` once, reshape after).
        Both fed HF's own `q_proj` output, so only the split varies.
        **correct: q 2.09e-06, gate 2.17e-06** (atol 1e-4)
        **naive:   q 6.03e+00, gate 6.54e+00** — fails by ~6 orders of magnitude
        **naive head 0 alone: 0.00e+00** (bit-exact) — the trap confirmed to hide
        under any single-head spot check.
- [x] task 2 QK-norm — `qk_norm(q, k, i)` reuses `harness.rmsnorm` unchanged; weight is
      `[256]` and rmsnorm acts on the last axis, so `[seq, heads, 256]` broadcasts across
      heads for free (no loop). Fed HF's `q_norm`/`k_norm` OUR q/k to isolate the norm.
      **q_norm 9.5e-07, k_norm 9.5e-07** (atol 1e-5).
      Negative control `rmsnorm_plain_w` (w instead of 1+w): **7.02e+00** — confirms
      session 01's zero-centred finding for the per-head norms too.
- [x] task 3 partial RoPE — `rope_cos_sin` / `rotate_half_np` / `apply_rope`, plus
      `to_head_major`. Verified with a config-only `Qwen3_5TextRotaryEmbedding` (no weights):
      **cos/sin 5.96e-08**, **q/k rotated 0.00e+00 (bit-exact)** vs HF `apply_rotary_pos_emb`,
      192 tail dims bit-identical. Still to run in-harness: 3a vs the real model's
      `rotary_emb`: **cos/sin 5.96e-08**, `cos[:32]==cos[32:]` exact.
      Negative control (rotate all 256 dims): **7.91e+00** — partial rotary confirmed load-bearing.
      - `inv_freq` has **32** entries (`arange(0, 64, 2)`), one per (i, i+32) PAIR;
        cos/sin are 64 wide only because HF does `cat((freqs, freqs), -1)` (line 143),
        so `cos[..., i] == cos[..., i+32]`. That duplication is what makes
        `x*cos + rotate_half(x)*sin` express 32 independent 2-D rotations elementwise.
      - **Why RoPE after QK-norm:** RMSNorm's division IS rotation-invariant, but its
        learned per-channel `(1+w)` scale is a diagonal matrix and diagonals do not
        commute with rotations — so the two orders differ. Trained norm-first: norm
        bounds the vector, RoPE rotates without changing length, bound survives.
      - **Why 192 position-blind dims:** their q.k contribution depends on content only,
        giving each head position-independent matching capacity; theta=1e7 stretches the
        slow end of the 64 rotated dims to reach 262144 positions.
- [x] task 4 GQA causal attention — `repeat_kv_np` / `causal_mask` / `softmax` / `gqa_attend`.
      Weight-free check vs `F.scaled_dot_product_attention`: **4.77e-07** against both
      `enable_gqa=True` and hand-`repeat_interleave` (those two bit-identical).
      In-harness (real weights): same **4.77e-07**; negative control `np.tile` **2.98e+00**.
      - **Grouping order (derived, not assumed):** `repeat_kv` inserts the repeat axis
        AFTER the head axis then flattens the pair -> `[kv0,kv0,kv0,kv0,kv1,kv1,kv1,kv1]`.
        KV head 0 serves Q heads 0-3, KV head 1 serves 4-7. `np.repeat(x, 4, axis=0)` ✓,
        `np.tile` ✗ — **same shape either way**, so no shape check catches it.
      - Mask is **additive** (`-inf` above the diagonal), not boolean indexing (line 619).
      - Softmax forced fp32 (line 621) with max-subtraction: exp overflows fp32 near 88;
        subtracting the row max is algebraically free and caps every exponent at 0.
        All-masked rows -> NaN, but causal never makes one (row t keeps position t).
        Padding masks can — a real bug source once batching arrives.
- [x] task 5 gate + o_proj + end-to-end — `sigmoid` (overflow-safe, two-branch),
      `merge_heads`, and **`attn(x, pos, layer_idx)`** in `python/harness_attn.py`.
      Weight-free: sigmoid clean at +/-1000; `merge_heads` verified head-major
      (token 0 = head0 dims then head1 dims).
      **layer 3: max|d| 4.92e-07 / layer 7: 6.41e-07** at atol 1e-3 — three orders under budget. Because HF
      also runs this sub-block in fp32 and our op order matches theirs, error never
      compounds past single-GEMM noise. The 1e-3 budget is for bf16 weights meeting a
      *different* op order (the P5 CUDA port), not for this.
      - **Merge order is load-bearing:** HF's eager path ends `.transpose(1,2)` -> `[B,S,H,D]`
        then `.reshape(*input_shape, -1)`, so the 2048 is head-major per token. The gate was
        flattened from `[B,S,H,D]` identically, so they align elementwise. A different merge
        order multiplies head 3's output by head 5's gate — right shape, no error.
      - HF's forward **requires a batch axis** (`.transpose(1,2)` on 4-D): pass `X[None]`,
        `position_embeddings=(cos[None], sin[None])`, and an additive causal mask
        `[1,1,S,S]` (explicit mask keeps it correct under eager OR sdpa).
- [x] task 6 regression hook — `harness_attn.main()` runs all 9 session-02 tasks and
      returns a failure count; `harness.py`'s `__main__` imports it (import placed INSIDE
      `__main__` so the harness -> harness_attn dependency stays one-directional) and
      exits non-zero. `python harness.py` = 01 + 02 in one go, **all green**.
      - **Negative controls needed their own verdict.** 5 checks must FAIL by design; if
        they printed plain `[FAIL]` a real regression would be invisible. So `check` =
        must match, `check_differs` = must not (`XFAIL` when it correctly differs, `BAD`
        + recorded failure if it ever starts matching — i.e. someone "fixed" split_naive).

- decisions taken:
  - `python/harness_attn.py` imports `harness.py` (`load_raw`/`load_hf`/`compare`/`rmsnorm`)
    rather than growing one file; 03 imports `attn()` from it.
  - Expected shapes are *derived from config*, never typed: q_proj is
    `n_heads * head_dim * 2` — the `2` IS the gate. rotary_dim via
    `int(head_dim * partial_rotary_factor)` (`modeling_rope_utils.py:180`, note `int` not `round`).
- surprises / mismatches:
  - **Double model load via `__main__` aliasing.** `python harness.py` runs that file as
    module `__main__`; `harness_attn`'s `from harness import ...` then imports the SAME
    file again under the name `harness` — a distinct module object with its own
    `load_hf` lru_cache, so the 8.5 GB model loaded twice (~17 GB peak). Fixed with
    `sys.modules.setdefault("harness", sys.modules["__main__"])` at the top of
    `harness.py`'s `__main__`. Will bite again for any future `harness_*.py` sibling.
  - **`rotary_emb(x, position_ids)` returns `[batch, seq, 64]`, NOT a 3-row mrope tensor.**
    It builds `(3, bs, seq, 32)` internally, then `apply_interleaved_mrope`
    (`modeling_qwen3_5.py:148`) collapses the 3 rows into one by interleaving T/H/W
    frequencies per `mrope_section [11,11,10]`. Text-only -> all 3 rows identical -> the
    result equals row 0 and mrope is a no-op. **Line 148 is the entire P7 plug-in point.**
    `position_ids` goes in as `[batch, seq]`; the `ndim==2` branch does the 3-row expand.
  - **HF assumes a batch axis.** `Qwen3_5Attention.forward` does `.transpose(1, 2)` on a
    4-D `[B,S,H,D]`. Our `X` is `[seq, 2048]` (no batch) — calling `self_attn(X)` directly
    would transpose the wrong axes and silently return garbage. Stage 5 must pass `X[None]`.
    (`q_proj` alone is unaffected, so stages 1-4 can stay 2-D.)
  - **Tolerance is per-op, not global.** Weights compare at atol 0.0 (bit-exact, no
    arithmetic); a single 2048-term GEMM lands at ~1e-6 (numpy vs torch reduce in
    different orders, fp32 add isn't associative). Reusing atol=0.0 on a GEMM fails
    spuriously. Rule: never raise atol to make a test pass — >~1e-3 means semantics.
  - Must run with `python/.venv/Scripts/python.exe`, not the system/anaconda `python`
    (no torch there). And run from `python/` — `harness.MODEL_DIR` is relative.

## Handoff (copy up to PROJECT.md when done) — **DONE 2026-09-13**

**What got built.** `python/harness_attn.py` (imports `harness.py`, adds ~360 lines).
Public surface for session 03:

- **`attn(x, pos, layer_idx=3) -> y`** — `x [seq, 2048]`, `pos [seq]` int, `y [seq, 2048]`.
  One full-attention sub-block, layer-agnostic. Loads its own weights from disk per call
  (fine for a reference; 03 may want to hoist that out if the causal loop gets slow).
  **Excludes** input_layernorm / post_attention_layernorm / residual add — those are 03's.
- Reusable pieces: `linear`, `project_q`, `project_kv`, `split_q_gate`, `qk_norm`,
  `to_head_major`, `rope_cos_sin`, `rotate_half_np`, `apply_rope`, `repeat_kv_np`,
  `causal_mask`, `softmax`, `sigmoid`, `merge_heads`, `gqa_attend`.
- `main()` -> failure count; `check` / `check_differs` verdict helpers; `TASKS` list.
- Layout convention: `[seq, heads, head_dim]` out of the projections, then
  **`[heads, seq, head_dim]`** from `to_head_major` onward.

**Per-stage max|d| (seq 12, layer 3, fixed seed 2):**

| stage | check | max\|d\| | atol |
|---|---|---|---|
| 0 | six weights raw-vs-HF | 0.00e+00 | 0 |
| 1a | k_proj / v_proj | 9.5e-07 / 8.3e-07 | 1e-4 |
| 1c | q / gate correct split | 2.09e-06 / 2.17e-06 | 1e-4 |
| 1c | *control* naive split | 6.03e+00 / 6.54e+00 | XFAIL |
| 2 | q_norm / k_norm | 9.5e-07 / 9.5e-07 | 1e-5 |
| 2 | *control* plain-w | 7.02e+00 | XFAIL |
| 3a | cos / sin | 5.96e-08 | 1e-5 |
| 3b | q / k roped | 0.00e+00 | 1e-5 |
| 3b | *control* rotate all 256 | 7.91e+00 | XFAIL |
| 4 | vs SDPA (gqa + manual) | 4.77e-07 | 1e-5 |
| 4 | *control* np.tile KV | 2.98e+00 | XFAIL |
| 5 | **attn() layer 3** | **4.92e-07** | 1e-3 |
| 5 | **attn() layer 7** | **6.41e-07** | 1e-3 |

**Facts that contradict / refine PROJECT.md:** none contradict. Refinements:
1. `rotary_emb(x, position_ids)` returns `[batch, seq, 64]`, not a 3-row mrope tensor —
   `apply_interleaved_mrope` (`modeling_qwen3_5.py:148`) collapses the 3 rows. **That line
   is the whole P7 plug-in point.**
2. `inv_freq` has **32** entries, not 64 (one per rotated *pair*); cos/sin are 64 wide only
   because HF duplicates them (`cat((freqs, freqs), -1)`).
3. `Qwen3_5Attention.forward` **requires a batch axis** (4-D `.transpose(1,2)`).
4. The attention output merge is **head-major per token**, and the gate is flattened the
   same way — they must align or heads get cross-wired silently.
5. Tolerance in practice is far tighter than the 1e-3 budget (5e-7), because HF also runs
   this sub-block in fp32 with a near-identical op order. Expect 1e-3 to start mattering
   only at P5 (CUDA, different reduction order).

**What 03 needs:**
- `from harness_attn import attn` — treat as a black box, `attn(x, pos, layer_idx)`.
- Decoder layer assembly: `input_layernorm` -> `self_attn` -> residual add ->
  `post_attention_layernorm` -> MLP -> residual add. Source: `Qwen3_5DecoderLayer`
  (`modeling_qwen3_5.py:740-794`).
- SwiGLU MLP: `gate_proj`/`up_proj` `[6144, 2048]`, `down_proj` `[2048, 6144]`,
  `silu(gate_proj(x)) * up_proj(x)` -> `down_proj`. Source `Qwen3_5MLP` (704-718).
- Only 6 of 24 layers are full-attention ([3,7,11,15,19,23]); the other 18 are
  GatedDeltaNet and are P3's job — 03 should assemble a layer it can run, not all 24.
- KV cache: shape `[2, seq, 256]` per full-attn layer, 12 KiB/token total across the 6.
- Watch the double-load trap if a `harness_mlp.py` gets added (see surprises).
