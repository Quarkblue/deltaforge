import json
from harness import load_raw, load_hf, compare, rmsnorm
import numpy as np
import torch

# Compare against HF's own rotary helper rather than reimplementing it twice.
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    apply_rotary_pos_emb as hf_apply_rotary_pos_emb,
)

CONFIG_PATH = "../models/Qwen3.5-2B/config.json"

config = json.load(open(CONFIG_PATH))

text_config = config["text_config"]

text_hidden_size = text_config["hidden_size"]
head_dim = text_config["head_dim"]
num_attention_heads = text_config["num_attention_heads"]
num_kv_heads = text_config["num_key_value_heads"]
rms_norm_eps = text_config["rms_norm_eps"]

rope_params = text_config["rope_parameters"]
rope_theta = rope_params["rope_theta"]
partial_rotary_factor = rope_params["partial_rotary_factor"]


n_kv_groups = num_attention_heads // num_kv_heads
rotary_dim = int(head_dim * partial_rotary_factor)
scale = 1 / (head_dim ** 0.5)


print(f"n_kv_groups: {n_kv_groups}")
print(f"rotary_dim: {rotary_dim}")
print(f"scale: {scale}")


ATTN_PARTS = ["q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"]


# ── pass/fail bookkeeping ───────────────────────────────────────────────────
# Five checks in this file are negative controls — they MUST differ from HF.
# If those printed plain [FAIL], a real regression would be invisible in the
# noise. So: `check` = must match, `check_differs` = must not. Only genuine
# problems land in _failures, and main() exits non-zero on them.
_failures: list[str] = []


def check(name, mine, theirs, atol):
    """Must match. Records a failure if it does not."""
    if not compare(name, mine, theirs, atol):
        _failures.append(name)


def check_differs(name, mine, theirs, atol):
    """Negative control: must NOT match. Records a failure if it does.

    A control that starts passing means the thing it was proving has silently
    become true — e.g. someone "fixed" split_naive. That is a regression too.
    """
    matched = compare(f"(control) {name}", mine, theirs, atol)
    verdict = "BAD " if matched else "XFAIL"
    print(f"[{verdict}] {name:22s} expected to DIFFER -> {'MATCHED, control broken' if matched else 'differs, as designed'}")
    if matched:
        _failures.append(f"{name} (negative control matched)")


# ── addressing one layer, two ways ──────────────────────────────────────────
# The disk key and the HF attribute path are NOT the same string: the loaded
# object is a ForConditionalGeneration whose `.model` is the Qwen3_5Model, so
# the attr path carries one extra `.model` vs the key. Keep them as two
# functions so raw-vs-HF stays an honest comparison.
def layer_key(i: int, name: str) -> str:
    """Disk key for one self_attn weight of layer i."""
    return f"model.language_model.layers.{i}.self_attn.{name}.weight"


def hf_layer(model, i: int):
    return model.model.language_model.layers[i]


def hf_attn(model, i: int):
    return hf_layer(model, i).self_attn


def is_full_attn(i: int) -> bool:
    return text_config["layer_types"][i] == "full_attention"


# ── fixtures (fixed seed: every stage compares on the same input) ───────────
SEQ = 12
LAYER = 3
rng = np.random.default_rng(2)
X = rng.standard_normal((SEQ, text_hidden_size)).astype(np.float32)
POS = np.arange(SEQ)


# ── task 0: the six weights exist, agree raw-vs-HF, and have the shapes
#            PROJECT.md claims ────────────────────────────────────────────────
EXPECTED_SHAPES = {
    "q_proj": (num_attention_heads * head_dim * 2, text_hidden_size),  # gated -> 2x wide
    "k_proj": (num_kv_heads * head_dim, text_hidden_size),
    "v_proj": (num_kv_heads * head_dim, text_hidden_size),
    "o_proj": (text_hidden_size, text_hidden_size),
    "q_norm": (head_dim,),
    "k_norm": (head_dim,),
}


def task0_shapes(i: int = LAYER):
    print(f"\n== task 0: layer {i} self_attn weights ==")
    assert is_full_attn(i), f"layer {i} is {text_config['layer_types'][i]}, not full_attention"
    attn = hf_attn(load_hf(), i)
    print(f"       module: {type(attn).__name__}")
    for name in ATTN_PARTS:
        raw = load_raw(layer_key(i, name))
        hf_w = getattr(attn, name).weight.detach().numpy()
        want = EXPECTED_SHAPES[name]
        flag = "OK  " if (raw.shape == hf_w.shape == want) else "FAIL"
        print(f"[{flag}] {name:7s} raw={str(raw.shape):15s} hf={str(hf_w.shape):15s} want={want}")
        # same bf16 tensor widened to fp32 twice -> must be bit-exact
        check(f"{name} raw==hf", raw, hf_w, 0.0)

def linear(x, w) -> np.ndarray:
    x = x.astype(np.float32)
    w = w.astype(np.float32)
    return x @ w.T

def project_kv(x, i):
    k_proj = linear(x, load_raw(layer_key(i, "k_proj"))).reshape(-1, num_kv_heads, head_dim)
    v_proj = linear(x, load_raw(layer_key(i, "v_proj"))).reshape(-1, num_kv_heads, head_dim)
    return k_proj, v_proj

def task1a_kv(i = LAYER):
    attn = hf_attn(load_hf(), i)
    k_mine, v_mine = project_kv(X, i)
    xt = torch.from_numpy(X)
    k_hf = attn.k_proj(xt).detach().numpy().reshape(-1, num_kv_heads, head_dim)
    v_hf = attn.v_proj(xt).detach().numpy().reshape(-1, num_kv_heads, head_dim)
    check(f"k_proj raw==hf", k_mine, k_hf, 1e-4)
    check(f"v_proj raw==hf", v_mine, v_hf, 1e-4)

def split_q_gate(q_proj):
    q_proj = q_proj.reshape(-1, num_attention_heads, head_dim * 2)
    q = q_proj[..., :head_dim]
    gate = q_proj[..., head_dim:]
    return q, gate

def split_naive(q_proj):
    """The WRONG split, kept on purpose: 'first half is q, second half is gate'.

    Note the asymmetry vs split_q_gate: there we reshape into 512-wide per-head
    blocks FIRST and slice *within* each block. Here we slice ONCE across the
    full 4096-wide row and reshape after. Same shapes out, different numbers.
    """
    half = num_attention_heads * head_dim  # 2048
    q = q_proj[:, :half].reshape(-1, num_attention_heads, head_dim)
    gate = q_proj[:, half:].reshape(-1, num_attention_heads, head_dim)
    return q, gate

def project_q(x, i):
    q_proj = linear(x, load_raw(layer_key(i, "q_proj")))
    return split_q_gate(q_proj)

def task1c_split(i = LAYER):
    print("\n== task 1c: gated-q split (correct vs naive) ==")
    attn = hf_attn(load_hf(), i)
    q_mine, gate_mine = project_q(X, i)

    # HF's side: q_proj, then source lines 671-672 reproduced in torch.
    # (HF writes .view(*input_shape, -1, head_dim*2); with no batch axis, -1 for
    #  the seq dim and an explicit head count is the same reshape.)
    with torch.no_grad():
        xt = torch.from_numpy(X)
        qp_t = attn.q_proj(xt)
        q_hf_t, gate_hf_t = torch.chunk(
            qp_t.view(-1, num_attention_heads, head_dim * 2), 2, dim=-1
        )
        q_hf = q_hf_t.detach().numpy()
        gate_hf = gate_hf_t.detach().numpy()
        qp_np = qp_t.detach().numpy()

    # Feed the naive split HF's OWN projection output, not ours: that isolates
    # the split from the GEMM, so a failure can only be the split.
    q_bad, gate_bad = split_naive(qp_np)

    check("q correct", q_mine, q_hf, 1e-4)
    check("gate correct", gate_mine, gate_hf, 1e-4)
    check_differs("q NAIVE", q_bad, q_hf, 1e-4)
    check_differs("gate NAIVE", gate_bad, gate_hf, 1e-4)
    # ...but head 0 agrees under both splits (cols [0:256) either way), which is
    # exactly why spot-checking one head would have hidden the bug.
    check("q NAIVE head0", q_bad[:, 0], q_hf[:, 0], 1e-4)

# ── stage 2: per-head QK-RMSNorm ────────────────────────────────────────────
# Session 01's rmsnorm normalized over all 2048 hidden dims. This one normalizes
# over each head's 256 dims INDEPENDENTLY ("unlike olmo, only on the head dim!"
# — source line 653). Since rmsnorm works on the last axis and the weight is
# [256], feeding it [seq, heads, 256] broadcasts across heads for free: 8
# separate normalizations for q, 2 for k, no loop.
#
# Purpose: it bounds q·k before the softmax. Unnormalized, attention logits can
# grow until softmax saturates; per-head norm caps |q| and |k| so the logits
# stay in a usable range.
def qk_norm(q, k, i):
    q = rmsnorm(q, load_raw(layer_key(i, "q_norm")), rms_norm_eps)
    k = rmsnorm(k, load_raw(layer_key(i, "k_norm")), rms_norm_eps)
    return q, k


def rmsnorm_plain_w(x, weight, eps=rms_norm_eps):
    """Session 01's trap, rebuilt to prove it: `weight` instead of `1 + weight`.

    Qwen3_5RMSNorm stores the scale as a delta from 1.0 (init 0, Gemma-style),
    so using w directly is off by roughly the whole channel scale.
    """
    x = x.astype(np.float32)
    var = np.mean(x * x, axis=-1, keepdims=True)
    return x * (1.0 / np.sqrt(var + eps)) * weight.astype(np.float32)


def task2_qk_norm(i = LAYER):
    print("\n== task 2: per-head QK-RMSNorm ==")
    attn = hf_attn(load_hf(), i)
    q, _gate = project_q(X, i)
    k, _v = project_kv(X, i)

    q_mine, k_mine = qk_norm(q, k, i)

    # Feed HF's q_norm/k_norm OUR q and k (not HF's) so the only thing under
    # test is the norm itself — the projections were already verified in task 1.
    with torch.no_grad():
        q_hf = attn.q_norm(torch.from_numpy(q)).detach().numpy()
        k_hf = attn.k_norm(torch.from_numpy(k)).detach().numpy()

    # Same input, both sides fp32, no GEMM -> expect ~1e-7, far tighter than 1e-4.
    check("q_norm", q_mine, q_hf, 1e-5)
    check("k_norm", k_mine, k_hf, 1e-5)
    q_bad = rmsnorm_plain_w(q, load_raw(layer_key(i, "q_norm")))
    check_differs("q_norm plain-w", q_bad, q_hf, 1e-5)


# ── stage 3: partial RoPE, rotate_half convention ───────────────────────────
# From here on q/k live as [heads, seq, head_dim], mirroring HF's .transpose(1,2).
# That puts seq where cos/sin broadcast cleanly and makes stage 4's q @ k.T the
# natural expression.
def to_head_major(x):
    """[seq, heads, head_dim] -> [heads, seq, head_dim]."""
    return np.transpose(x, (1, 0, 2))


def rope_cos_sin(pos):
    """cos/sin tables for the rotary block only. Returns two [seq, rotary_dim].

    inv_freq has rotary_dim//2 == 32 entries, not 64: each entry is the rotation
    speed of one (i, i+32) PAIR. cos/sin are then 64 wide because HF does
    `cat((freqs, freqs), -1)` (source line 143) — so cos[..., i] == cos[..., i+32].
    That duplication is what lets `x*cos + rotate_half(x)*sin` express all 32
    independent 2-D rotations in one elementwise pass.

    theta = 1e7 (vs the usual 1e4) stretches the slow end of the frequency range
    far enough to address 262144 positions.
    """
    inv_freq = 1.0 / (rope_theta ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim))
    freqs = np.outer(pos.astype(np.float32), inv_freq)      # [seq, 32]
    emb = np.concatenate([freqs, freqs], axis=-1)           # [seq, 64]
    return np.cos(emb), np.sin(emb)


def rotate_half_np(x):
    """HF's rotate_half: pair dim i with dim i+32, not 2i with 2i+1.

    cat(-x2, x1) is the 2-D rotation matrix [[cos,-sin],[sin,cos]] applied to all
    32 pairs at once. The (i, i+d/2) pairing is a pure layout choice — same
    rotation as adjacent-pair complex form, but with contiguous slices.
    """
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rope(q, k, cos, sin):
    """Rotate only the first `rotary_dim` dims; pass the remaining 192 through.

    Those 192 untouched dims are position-BLIND: their contribution to q.k depends
    on content only, never on the offset. RoPE runs AFTER qk_norm because RMSNorm's
    division is rotation-invariant but its learned per-channel (1+w) scale is a
    diagonal matrix, and diagonals do not commute with rotations.
    """
    # q/k are [heads, seq, head_dim]; cos/sin are [seq, rot] -> add a head axis.
    cos = cos[None, :, :].astype(np.float32)
    sin = sin[None, :, :].astype(np.float32)
    rot = cos.shape[-1]

    def rot_one(x):
        x_rot, x_pass = x[..., :rot], x[..., rot:]
        out = (x_rot * cos) + (rotate_half_np(x_rot) * sin)
        return np.concatenate([out, x_pass], axis=-1)

    return rot_one(q), rot_one(k)


def task3a_cos_sin(i = LAYER):
    print("\n== task 3a: cos/sin tables ==")
    model = load_hf()
    cos_mine, sin_mine = rope_cos_sin(POS)
    print(f"       mine: cos{cos_mine.shape}  (rotary_dim wide, 32 unique angles doubled)")

    with torch.no_grad():
        # rotary_emb wants position_ids [batch, seq]; it internally expands to the
        # 3-row (T/H/W) mrope form, then apply_interleaved_mrope collapses it back.
        # For text-only input all 3 rows are identical -> mrope is a no-op here.
        pos_t = torch.from_numpy(POS)[None, :]
        cos_hf, sin_hf = model.model.language_model.rotary_emb(torch.from_numpy(X), pos_t)
    print(f"       HF:   cos{tuple(cos_hf.shape)}  <- [batch, seq, rotary_dim], NOT 3-row")

    check("cos", cos_mine[None], cos_hf.numpy(), 1e-5)
    check("sin", sin_mine[None], sin_hf.numpy(), 1e-5)
    # cat((freqs, freqs)) means the table is its own mirror across the halfway mark.
    half = rotary_dim // 2
    check("cos[:32] == cos[32:]", cos_mine[:, :half], cos_mine[:, half:], 0.0)


def task3b_rotate(i = LAYER):
    print("\n== task 3b: apply_rope (partial, rotate_half) ==")
    q, _gate = project_q(X, i)
    k, _v = project_kv(X, i)
    q, k = qk_norm(q, k, i)
    q, k = to_head_major(q), to_head_major(k)   # [heads, seq, 256]

    cos, sin = rope_cos_sin(POS)
    q_mine, k_mine = apply_rope(q, k, cos, sin)

    with torch.no_grad():
        # HF works on [batch, heads, seq, dim] and unsqueezes cos/sin at dim 1.
        qt = torch.from_numpy(np.ascontiguousarray(q))[None]
        kt = torch.from_numpy(np.ascontiguousarray(k))[None]
        cos_t = torch.from_numpy(cos)[None]
        sin_t = torch.from_numpy(sin)[None]
        q_hf, k_hf = hf_apply_rotary_pos_emb(qt, kt, cos_t, sin_t)
        q_hf = q_hf[0].numpy()
        k_hf = k_hf[0].numpy()

    check("q roped", q_mine, q_hf, 1e-5)
    check("k roped", k_mine, k_hf, 1e-5)

    # The 192 tail dims must come through bit-identical — nothing touched them.
    check("q tail untouched", q_mine[..., rotary_dim:], q[..., rotary_dim:], 0.0)
    cos_full = np.concatenate([cos, cos, cos, cos], axis=-1)   # 64 -> 256
    sin_full = np.concatenate([sin, sin, sin, sin], axis=-1)
    q_bad, _ = apply_rope(q, k, cos_full, sin_full)
    check_differs("q rotated-all-256", q_bad, q_hf, 1e-5)


# ── stage 4: GQA causal attention ───────────────────────────────────────────
def repeat_kv_np(x, n_rep):
    """2 KV heads -> 8, matching HF repeat_kv (source line 592).

    HF inserts the repeat axis AFTER the head axis and flattens the pair:
        x[:, :, None].expand(B, 2, 4, S, D).reshape(B, 8, S, D)
    which yields [kv0, kv0, kv0, kv0, kv1, kv1, kv1, kv1] — KV head 0 serves Q
    heads 0-3, KV head 1 serves Q heads 4-7. Consecutive blocks.

    np.repeat gives exactly that. np.tile gives [0,1,0,1,...] — WRONG, and the
    same shape, so no shape check catches it.
    """
    return np.repeat(x, n_rep, axis=0)


def causal_mask(seq):
    """Additive mask: 0 on/below the diagonal, -inf above.

    HF *adds* the mask to the logits (line 619), it does not index with a bool.

    An all-masked row would make softmax compute exp(-inf - -inf) = NaN. Causal
    masking never produces one: row t always keeps position t itself. Padding
    masks CAN, which is where that bug shows up once batching arrives.
    """
    m = np.zeros((seq, seq), dtype=np.float32)
    m[np.triu_indices(seq, k=1)] = -np.inf
    return m


def softmax(x):
    """fp32 softmax with max-subtraction.

    exp() of a large positive logit overflows fp32 (~88 is the ceiling).
    Subtracting the row max is algebraically free — exp(a-m)/sum(exp(a-m)) equals
    exp(a)/sum(exp(a)) — and bounds every exponent at 0, so the largest term is
    exactly 1.0 and nothing can overflow. HF forces dtype=float32 here (line 621)
    even when the model runs in bf16.
    """
    x = x.astype(np.float32)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def gqa_attend(q, k, v, n_rep=None):
    """q [8, S, 256], k/v [2, S, 256] -> [8, S, 256]. Causal, fp32 softmax."""
    n_rep = n_kv_groups if n_rep is None else n_rep
    k = repeat_kv_np(k, n_rep)
    v = repeat_kv_np(v, n_rep)
    scores = (q @ np.swapaxes(k, -1, -2)) * scale     # [8, S, S]
    scores = scores + causal_mask(q.shape[-2])
    return softmax(scores) @ v


def task4_gqa(i = LAYER):
    print("\n== task 4: GQA causal attention ==")
    # Grouping order, provable without any weights: label each KV head by index.
    probe = np.arange(num_kv_heads, dtype=np.float32)[:, None, None] * np.ones((1, 2, 2), np.float32)
    print(f"       np.repeat -> heads {repeat_kv_np(probe, n_kv_groups)[:, 0, 0].astype(int).tolist()}")
    print(f"       np.tile   -> heads {np.tile(probe, (n_kv_groups, 1, 1))[:, 0, 0].astype(int).tolist()}  <- WRONG")

    q, _gate = project_q(X, i)
    k, v = project_kv(X, i)
    q, k = qk_norm(q, k, i)
    q, k, v = to_head_major(q), to_head_major(k), to_head_major(v)
    cos, sin = rope_cos_sin(POS)
    q, k = apply_rope(q, k, cos, sin)

    mine = gqa_attend(q, k, v)

    with torch.no_grad():
        qt = torch.from_numpy(np.ascontiguousarray(q))[None]
        kt = torch.from_numpy(np.ascontiguousarray(k))[None]
        vt = torch.from_numpy(np.ascontiguousarray(v))[None]
        # enable_gqa=True: SDPA does the 2->8 expansion internally.
        sdpa_gqa = torch.nn.functional.scaled_dot_product_attention(
            qt, kt, vt, is_causal=True, enable_gqa=True, scale=scale
        )[0].numpy()
        # ...and again with the expansion done by hand, to prove they agree.
        sdpa_manual = torch.nn.functional.scaled_dot_product_attention(
            qt, kt.repeat_interleave(n_kv_groups, dim=1),
            vt.repeat_interleave(n_kv_groups, dim=1),
            is_causal=True, scale=scale
        )[0].numpy()

    check("attn vs SDPA(gqa)", mine, sdpa_gqa, 1e-5)
    check("attn vs SDPA(manual)", mine, sdpa_manual, 1e-5)
    check("SDPA gqa == manual", sdpa_gqa, sdpa_manual, 1e-6)
    k_bad = np.tile(k, (n_kv_groups, 1, 1))
    v_bad = np.tile(v, (n_kv_groups, 1, 1))
    scores = (q @ np.swapaxes(k_bad, -1, -2)) * scale + causal_mask(q.shape[-2])
    check_differs("attn TILED kv", softmax(scores) @ v_bad, sdpa_gqa, 1e-5)


# ── stage 5: output gate, o_proj, and the whole sub-block ────────────────────
def sigmoid(x):
    """Overflow-safe logistic. exp(-x) overflows for x very negative, so use the
    exp(x)/(1+exp(x)) form there — algebraically identical, numerically safe."""
    x = x.astype(np.float32)
    pos = x >= 0
    out = np.empty_like(x)
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def merge_heads(x):
    """[heads, seq, head_dim] -> [seq, heads * head_dim], head-major per token.

    Matches HF: eager_attention_forward ends with .transpose(1, 2) giving
    [B, S, H, D], then forward does .reshape(*input_shape, -1). The gate was
    flattened from [B, S, H, D] the same way, so the two align elementwise.
    Merge in any other order and sigmoid(gate) multiplies head 3's output by
    head 5's gate — right shape, wrong numbers, no error.
    """
    return np.transpose(x, (1, 0, 2)).reshape(x.shape[1], -1)


def attn(x, pos, i = LAYER):
    """One full-attention sub-block: x [seq, 2048] -> y [seq, 2048].

    Excludes input_layernorm / post_attention_layernorm and the residual add —
    those belong to the decoder layer (session 03), not to self_attn.
    """
    q, gate = project_q(x, i)                       # [seq, 8, 256] each
    k, v = project_kv(x, i)                         # [seq, 2, 256] each
    q, k = qk_norm(q, k, i)
    q, k, v = to_head_major(q), to_head_major(k), to_head_major(v)
    cos, sin = rope_cos_sin(pos)
    q, k = apply_rope(q, k, cos, sin)
    o = gqa_attend(q, k, v)                         # [8, seq, 256]
    o = merge_heads(o)                              # [seq, 2048]

    # The gate: attention rows must sum to 1, so a head with nothing useful to
    # attend to still dumps its mass somewhere (usually position 0) -> attention
    # sinks and massive activations. A per-channel sigmoid gate lets the head
    # scale its own output toward zero instead. This is WHY q_proj is 4096 wide.
    o = o * sigmoid(gate.reshape(gate.shape[0], -1))

    return linear(o, load_raw(layer_key(i, "o_proj")))


def task5_subblock(i = LAYER):
    print(f"\n== task 5: full self_attn sub-block, layer {i} ==")
    attn_mod = hf_attn(load_hf(), i)
    mine = attn(X, POS, i)

    with torch.no_grad():
        # HF's forward does .transpose(1, 2) on a 4-D tensor -> it REQUIRES a batch
        # axis. Passing our 2-D X would transpose the wrong axes and silently
        # return garbage. Hence X[None].
        xt = torch.from_numpy(X)[None]
        cos, sin = rope_cos_sin(POS)
        pe = (torch.from_numpy(cos)[None], torch.from_numpy(sin)[None])
        # Additive causal mask, broadcast over batch and heads. Supplying it
        # explicitly keeps this correct under either eager or sdpa.
        mask = torch.from_numpy(causal_mask(len(POS)))[None, None]
        theirs = attn_mod(hidden_states=xt, position_embeddings=pe, attention_mask=mask)[0]
        theirs = theirs[0].numpy()

    # Chain is 4 GEMMs (q/k/v -> scores -> @v -> o_proj) plus a softmax, all from
    # bf16-origin weights: 1e-3 is the honest budget (PROJECT.md decision 7).
    check(f"attn() layer {i}", mine, theirs, 1e-3)


# ── task 6: regression entry point ──────────────────────────────────────────
# Session 03 imports `attn` from here and treats it as a black box. `main()` is
# what harness.py calls so `python harness.py` runs 01 + 02 in one go.
TASKS = [
    task0_shapes,
    task1a_kv,
    task1c_split,
    task2_qk_norm,
    task3a_cos_sin,
    task3b_rotate,
    task4_gqa,
    task5_subblock,
    lambda: task5_subblock(7),   # same code, different weights (brief's bonus)
]


def main() -> int:
    """Run every session-02 task. Returns the number of genuine failures."""
    _failures.clear()
    for t in TASKS:
        t()
    print(f"\n== session 02 summary ==")
    if _failures:
        print(f"       {len(_failures)} FAILURE(S):")
        for f in _failures:
            print(f"         - {f}")
    else:
        print("       all checks green (negative controls correctly XFAIL)")
    return len(_failures)


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
