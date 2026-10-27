"""HF oracle activation dumper (brief 04, task 1).

Runs Qwen3.5-2B once on CPU with hooks + function wrappers, checks every tap for internal
consistency, and saves one safetensors file (naming: decision 14; batch axis stripped).

    python dump_oracle.py --dtype bf16
    python dump_oracle.py --dtype fp32
    python dump_oracle.py --dtype bf16 --out ../dumps/tmp.safetensors --compare ../dumps/oracle_bf16.safetensors

Exits non-zero if any check fails. Taps are stored exactly in HF's layout after stripping
the batch axis (see brief 04 Progress for the per-tap layout table).
"""

import argparse
import inspect
import os
import sys
from pathlib import Path

import torch
import transformers
from safetensors.torch import load_file, save_file
from transformers import AutoModelForImageTextToText, AutoTokenizer
import transformers.models.qwen3_5.modeling_qwen3_5 as mq

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "Qwen3.5-2B"
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
PROMPT = ("Hello, introduce yourself. by letting me know all the features that you have and can do. "
          "also let me know the best way to use you.")

# ---------------------------------------------------------------------------------------
# tap registry
# ---------------------------------------------------------------------------------------
TAPS = {}


def tap(name, t):
    if name in TAPS:
        raise ValueError(f"TAP {name} already exists")
    if isinstance(t, tuple):
        t = t[0]
    assert t.shape[0] == 1, f"{name}: expected batch 1, got {tuple(t.shape)}"
    TAPS[name] = t[0].detach().clone().contiguous()


def make_out_hook(name):
    def hook(module, args, output):
        tap(name, output)
    return hook


def make_in_hook(name):
    def hook(module, args):
        tap(name, args[0])
    return hook


def make_gate_hook(name, head_dim):
    # mirrors Qwen3_5Attention.forward lines 664-670: per-head split, second half = gate
    def hook(module, args, output):
        input_shape = output.shape[:-1]
        _, gate = torch.chunk(output.view(*input_shape, -1, head_dim * 2), 2, dim=-1)
        tap(name, gate.reshape(*input_shape, -1))
    return hook


# ---------------------------------------------------------------------------------------
# wrappers for module-level functions: forward() resolves these names in mq's globals at
# call time, so replacing mq.<name> is seen by the next forward; unpatch() restores them.
# ---------------------------------------------------------------------------------------
WRAPPED = ["apply_rotary_pos_emb", "eager_attention_forward", "torch_chunk_gated_delta_rule"]
ORIG = {n: getattr(mq, n) for n in WRAPPED}
CURRENT = {"layer": None}  # set by the self_attn pre-hook; RoPE has no layer_idx of its own
CALLS = {n: 0 for n in WRAPPED}


def make_layer_marker(i):
    def hook(module, args):
        CURRENT["layer"] = i
    return hook


def rope_wrapper(*args, **kwargs):
    q_out, k_out = ORIG["apply_rotary_pos_emb"](*args, **kwargs)
    CALLS["apply_rotary_pos_emb"] += 1
    p = f"L{CURRENT['layer']:02d}"
    tap(f"{p}.q_roped", q_out)  # [1, 8, T, 256] -> [8, T, 256]
    tap(f"{p}.k_roped", k_out)  # [1, 2, T, 256] -> [2, T, 256]
    return q_out, k_out


def eager_wrapper(module, *args, **kwargs):
    attn_output, attn_weights = ORIG["eager_attention_forward"](module, *args, **kwargs)
    CALLS["eager_attention_forward"] += 1
    tap(f"L{module.layer_idx:02d}.core_out", attn_output)  # [1, T, 8, 256] -> [T, 8, 256]
    return attn_output, attn_weights


def gdn_chunk_counter(*args, **kwargs):  # no taps yet (briefs 06/07 append GDN internals)
    CALLS["torch_chunk_gated_delta_rule"] += 1
    return ORIG["torch_chunk_gated_delta_rule"](*args, **kwargs)


def patch():
    mq.apply_rotary_pos_emb = rope_wrapper
    mq.eager_attention_forward = eager_wrapper
    mq.torch_chunk_gated_delta_rule = gdn_chunk_counter


def unpatch():
    for n, f in ORIG.items():
        setattr(mq, n, f)


def resolve_gdn_impl():
    """The hub_kernels decorator (integrations/hub_kernels.py:764) builds `wrapped`, which calls
    the closure variable `implementation`; functools.wraps makes __name__ lie, so read the closure."""
    impls = {}
    for n in ["torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule",
              "causal_conv1d_fn", "causal_conv1d_update"]:
        impl = inspect.getclosurevars(ORIG.get(n, getattr(mq, n))).nonlocals.get("implementation")
        assert impl is not None, f"{n}: no `implementation` in closure; decorator changed, inspect by hand"
        impls[n] = f"{impl.__module__}.{impl.__qualname__}"
        print(f"{n:34s} -> {impls[n]}")
        assert impl.__module__ == mq.__name__, f"{n} is NOT the torch fallback; dump is not the math we reimplement"
    return impls


# ---------------------------------------------------------------------------------------
# registration + run
# ---------------------------------------------------------------------------------------
def register_hooks(model):
    lm = model.model.language_model
    handles = [
        lm.embed_tokens.register_forward_hook(make_out_hook("embed")),
        lm.norm.register_forward_hook(make_out_hook("final_norm")),
    ]
    for i, layer in enumerate(lm.layers):
        p = f"L{i:02d}"
        handles.append(layer.register_forward_pre_hook(make_in_hook(f"{p}.in")))
        handles.append(layer.register_forward_hook(make_out_hook(f"{p}.out")))
        if not hasattr(layer, "self_attn"):
            continue
        a = layer.self_attn
        handles += [
            a.register_forward_pre_hook(make_layer_marker(i)),
            layer.input_layernorm.register_forward_hook(make_out_hook(f"{p}.input_norm")),
            a.q_proj.register_forward_hook(make_out_hook(f"{p}.q_proj")),
            a.q_proj.register_forward_hook(make_gate_hook(f"{p}.gate", a.head_dim)),
            a.q_norm.register_forward_pre_hook(make_in_hook(f"{p}.q")),
            a.q_norm.register_forward_hook(make_out_hook(f"{p}.q_normed")),
            a.k_norm.register_forward_pre_hook(make_in_hook(f"{p}.k")),
            a.k_norm.register_forward_hook(make_out_hook(f"{p}.k_normed")),
            a.v_proj.register_forward_hook(make_out_hook(f"{p}.v")),
            a.o_proj.register_forward_pre_hook(make_in_hook(f"{p}.gated")),
            a.register_forward_hook(make_out_hook(f"{p}.mixer_out")),
            layer.post_attention_layernorm.register_forward_pre_hook(make_in_hook(f"{p}.resid1")),
            layer.post_attention_layernorm.register_forward_hook(make_out_hook(f"{p}.post_norm")),
            layer.mlp.register_forward_hook(make_out_hook(f"{p}.mlp_out")),
        ]
    return handles


def tap_cache(model, cache):
    for i, lt in enumerate(model.model.language_model.config.layer_types):
        L, p = cache.layers[i], f"L{i:02d}.cache"
        if lt == "full_attention":
            tap(f"{p}.k", L.keys)    # [1, 2, T, 256] -> [2, T, 256], post-norm post-RoPE
            tap(f"{p}.v", L.values)
        else:
            tap(f"{p}.conv", L.conv_states[0])            # [6144, 4]
            tap(f"{p}.recurrent", L.recurrent_states[0])  # [16, 128, 128], stays F32


def run(model, ids):
    TAPS.clear()
    CURRENT["layer"] = None
    for n in CALLS:
        CALLS[n] = 0
    handles = register_hooks(model)
    try:
        patch()
        with torch.inference_mode():
            out = model(input_ids=ids, use_cache=True)
    finally:
        unpatch()
        for h in handles:
            h.remove()
    tap("ids", ids)
    tap("logits", out.logits)
    tap_cache(model, out.past_key_values)


# ---------------------------------------------------------------------------------------
# checks: OK / FAIL for expected-equal, XFAIL / BAD for negative controls (decision 9)
# ---------------------------------------------------------------------------------------
def run_checks(model):
    lm = model.model.language_model
    T = TAPS["ids"].shape[0]
    n_layers = len(lm.layers)
    fails = []

    def check(label, same, expect=True):
        status = ("OK" if same else "FAIL") if expect else ("BAD" if same else "XFAIL")
        if same != expect:
            fails.append(label)
            print(f"{status:5s} {label}")  # passing lines are counted, not printed
        return status

    n_attn = sum(hasattr(l, "self_attn") for l in lm.layers)
    check(f"tap count {len(TAPS)}", len(TAPS) == 4 + 2 * n_layers + 13 * n_attn + 3 * n_attn + 2 * n_layers)
    check(f"call counts {CALLS}", CALLS == {"apply_rotary_pos_emb": n_attn, "eager_attention_forward": n_attn,
                                            "torch_chunk_gated_delta_rule": n_layers - n_attn})
    check("embed == L00.in", torch.equal(TAPS["embed"], TAPS["L00.in"]))
    for i in range(n_layers - 1):
        check(f"L{i:02d}.out == L{i+1:02d}.in", torch.equal(TAPS[f"L{i:02d}.out"], TAPS[f"L{i+1:02d}.in"]))
    check(f"L{n_layers-1:02d}.out != final_norm",
          torch.equal(TAPS[f"L{n_layers-1:02d}.out"], TAPS["final_norm"]), expect=False)

    with torch.inference_mode():
        for i, layer in enumerate(lm.layers):
            p = f"L{i:02d}"
            t = lambda n: TAPS[f"{p}.{n}"]
            if not hasattr(layer, "self_attn"):
                check(f"{p} GDN cache present", f"{p}.cache.conv" in TAPS and f"{p}.cache.recurrent" in TAPS)
                continue
            a = layer.self_attn
            # step 3: residual adds + module re-runs on captured inputs ([None] = HF's batched shapes)
            check(f"{p} resid1 == in + mixer_out", torch.equal(t("resid1"), t("in") + t("mixer_out")))
            check(f"{p} out == resid1 + mlp_out", torch.equal(t("out"), t("resid1") + t("mlp_out")))
            check(f"{p} mixer_out == o_proj(gated)", torch.equal(t("mixer_out"), a.o_proj(t("gated")[None])[0]))
            check(f"{p} q_normed == q_norm(q)", torch.equal(t("q_normed"), a.q_norm(t("q")[None])[0]))
            check(f"{p} k_normed == k_norm(k)", torch.equal(t("k_normed"), a.k_norm(t("k")[None])[0]))
            check(f"{p} post_norm == post_attention_layernorm(resid1)",
                  torch.equal(t("post_norm"), layer.post_attention_layernorm(t("resid1")[None])[0]))
            check(f"{p} input_norm == input_layernorm(in)",
                  torch.equal(t("input_norm"), layer.input_layernorm(t("in")[None])[0]))
            check(f"{p} naive gate q_proj[..., 2048:] != gate",
                  torch.equal(t("q_proj")[..., 2048:], t("gate")), expect=False)
            # step 4
            check(f"{p} q_roped/k_roped/core_out shapes",
                  (tuple(t("q_roped").shape), tuple(t("k_roped").shape), tuple(t("core_out").shape))
                  == ((8, T, 256), (2, T, 256), (T, 8, 256)))
            check(f"{p} gated == core_out.flat * sigmoid(gate)",
                  torch.equal(t("gated"), (t("core_out").reshape(1, T, -1) * torch.sigmoid(t("gate")[None]))[0]))
            # step 5
            check(f"{p} cache.k == k_roped", torch.equal(t("cache.k"), t("k_roped")))
            check(f"{p} cache.v == v as [2, T, 256]",
                  torch.equal(t("cache.v"), t("v").view(T, 2, 256).transpose(0, 1)))
    return fails


def compare_files(a_path, b_path):
    a, b = load_file(a_path), load_file(b_path)
    if a.keys() != b.keys():
        return [f"key mismatch: {sorted(set(a) ^ set(b))}"]
    return [k for k in a if a[k].dtype != b[k].dtype or not torch.equal(a[k], b[k])]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dtype", choices=DTYPES, default="bf16")
    ap.add_argument("--out", type=Path, default=None, help="default: dumps/oracle_<dtype>.safetensors")
    ap.add_argument("--compare", type=Path, default=None, help="existing dump that must be bit-identical")
    ap.add_argument("--quiet", action="store_true", help="skip the manifest print")
    args = ap.parse_args()
    out_file = args.out or ROOT / "dumps" / f"oracle_{args.dtype}.safetensors"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    gdn_impl = resolve_gdn_impl()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
    print(f"prompt: {len(ids[0])} tokens, dtype {args.dtype}")

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_DIR, dtype=DTYPES[args.dtype], low_cpu_mem_usage=True, attn_implementation="eager")
    model.eval()
    assert model.config._attn_implementation == "eager", model.config._attn_implementation

    run(model, ids)
    fails = run_checks(model)
    print(f"checks: {len(fails)} failing")
    if fails:
        sys.exit(f"FAILED: {fails}")

    if not args.quiet:
        for k, v in TAPS.items():
            print(f"{k:20s} {str(tuple(v.shape)):18s} {v.dtype}")

    save_file(TAPS, str(out_file), metadata={
        "prompt": PROMPT,
        "dtype": args.dtype,
        "attn_implementation": model.config._attn_implementation,
        "gdn_impl": "; ".join(f"{k}={v}" for k, v in gdn_impl.items()),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
    })

    loaded = load_file(str(out_file))
    bad = ([f"key mismatch: {sorted(set(loaded) ^ set(TAPS))}"] if loaded.keys() != TAPS.keys() else
           [k for k in TAPS if loaded[k].dtype != TAPS[k].dtype or not torch.equal(loaded[k], TAPS[k])])
    if bad:
        sys.exit(f"reload mismatch: {bad}")
    print(f"reload OK: {len(loaded)} tensors, {os.path.getsize(out_file) / 2**20:.1f} MiB -> {out_file}")

    if args.compare:
        diff = compare_files(out_file, args.compare)
        if diff:
            sys.exit(f"differs from {args.compare}: {diff}")
        print(f"bit-identical to {args.compare}")


if __name__ == "__main__":
    main()
