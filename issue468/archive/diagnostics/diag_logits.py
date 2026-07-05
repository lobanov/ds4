#!/usr/bin/env python3
"""One-step instrumentation: does the drafter hidden x differ between q4k and f32,
and are the argmax margins large enough to absorb a 5% expert perturbation?

Replicates measure_bundle's exact step=1 math, but captures the hidden state fed
into forward_head and the logits top-5 + margin.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "dspark_oracle"))
from gguf_loader import load_gguf_dense_only
from expert_store import ExpertStore
from forward import layer_weights, forward_head, DIM, BLOCK, HC, NOISE_TOK
from build_main_hidden_from_captures import CANONICAL_INPUTS
from attention import precompute_rope, apply_rotary, ROPE_DIM, HEAD_DIM
from hc_primitives import rmsnorm, hc_pre, hc_post
from moe import moe
from measure_acceptance_bundle import dspark_attn, load_mh, build_model_ctx, build_drafter_ctx

WIN = 128
BUNDLE = HERE / "artifacts/exactness_small_bundles/code_histogram__t0p0"
MODEL = "/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
GGUFS = {"q4k": "/Users/lobanov/Projects/ds4/gguf/dspark.gguf",
         "f32": str(HERE / "artifacts/dspark_ceiling/dspark_f32.gguf")}


def forward_step1(dctx, mctx, bundle):
    T = dctx["T"]; layers = dctx["layers"]; stores = dctx["stores"]
    main_proj = dctx["main_proj"]; main_norm_w = dctx["main_norm_w"]
    embed_w = mctx["embed_w"]; lm_head = mctx["lm_head"]
    cos, sin = mctx["cos"], mctx["sin"]
    manifest = json.loads((bundle / "bundle_manifest.json").read_text())
    target_tokens = json.loads((bundle / "target_selected_tokens.json").read_text())
    pos0 = int(manifest["prompt_tokens"])
    win_kv = [np.zeros((WIN, HEAD_DIM), dtype=np.float32) for _ in range(3)]
    mh0 = load_mh(bundle, pos0)
    main_x0 = rmsnorm(mh0.reshape(1, 1, 3 * DIM) @ main_proj.T, main_norm_w)
    for s in range(3):
        mkv = rmsnorm(main_x0 @ layers[s]["kv"].T, layers[s]["kv_a_norm"])
        mkv[..., -ROPE_DIM:] = apply_rotary(mkv[..., -ROPE_DIM:], cos[0], sin[0])
        win_kv[s][0] = mkv[0, 0]
    # step=1
    step = 1
    pos = pos0 + step
    anchor = int(target_tokens[step])
    mh = load_mh(bundle, pos)
    main_x = rmsnorm(mh.reshape(1, 1, 3 * DIM) @ main_proj.T, main_norm_w)
    for s in range(3):
        mkv = rmsnorm(main_x @ layers[s]["kv"].T, layers[s]["kv_a_norm"])
        mkv[..., -ROPE_DIM:] = apply_rotary(mkv[..., -ROPE_DIM:], cos[step], sin[step])
        win_kv[s][step % WIN] = mkv[0, 0]
    n_real = 2
    draft_ids = np.full(BLOCK, NOISE_TOK, dtype=np.int64)
    draft_ids[0] = anchor
    x = embed_w[draft_ids][None]
    x = np.repeat(x[:, :, None, :], HC, axis=2)
    for s in range(3):
        res = x
        yd, post, comb = hc_pre(x, layers[s]["hc_attn_fn"], layers[s]["hc_attn_scale"], layers[s]["hc_attn_base"])
        yd = rmsnorm(yd, layers[s]["attn_norm"])
        ao = dspark_attn(yd, win_kv[s], n_real, layers[s], cos, sin, step)
        x = hc_post(ao, res, post, comb)
        res = x
        yd, post, comb = hc_pre(x, layers[s]["hc_ffn_fn"], layers[s]["hc_ffn_scale"], layers[s]["hc_ffn_base"])
        yd = rmsnorm(yd, layers[s]["ffn_norm"])
        fo = moe(yd, np.array([0]), layers[s], stores[s])
        x = hc_post(fo, res, post, comb)
    # x here is the drafter hidden into forward_head [1,block,hc,dim]
    from hc_primitives import hc_head
    h = hc_head(x, T["mtp.2.hc_head_fn.weight"][0], T["mtp.2.hc_head_scale.weight"][0], T["mtp.2.hc_head_base.weight"][0])
    hn = rmsnorm(h, T["mtp.2.norm.weight"][0])
    logits = hn[0] @ lm_head.T   # [block, vocab]
    return x, logits, anchor


def main():
    mctx = build_model_ctx(MODEL)
    xs, logits, anchor = {}, {}, None
    for label, path in GGUFS.items():
        dctx = build_drafter_ctx(path)
        x, lg, a = forward_step1(dctx, mctx, BUNDLE)
        xs[label] = x; logits[label] = lg; anchor = a
        for s in range(3): dctx["stores"][s].clear()

    print(f"step1 anchor (target_tokens[1]) = {anchor}")
    print(f"\n=== drafter hidden x into forward_head [1,5,4,4096] ===")
    for label in GGUFS:
        print(f"  {label}: ||x||={np.linalg.norm(xs[label]):.4f} mean|x|={np.abs(xs[label]).mean():.5f} std={xs[label].std():.5f}")
    dx = xs["f32"] - xs["q4k"]
    print(f"  ||x_f32 - x_q4k|| / ||x_q4k|| = {np.linalg.norm(dx)/np.linalg.norm(xs['q4k']):.4%}")
    print(f"  max|x_f32 - x_q4k| = {np.abs(dx).max():.5f}")

    print(f"\n=== logits[0] (position 0, the anchor slot) top-5 + margin ===")
    for label in GGUFS:
        lg = logits[label][0]
        top = np.argsort(-lg)[:5]
        print(f"  {label}: top5={top.tolist()} vals={np.round(lg[top],3).tolist()} margin(top1-top2)={lg[top[0]]-lg[top[1]]:.4f}")
    dlg = logits["f32"][0] - logits["q4k"][0]
    print(f"  ||logits_f32 - logits_q4k|| / ||logits_q4k|| = {np.linalg.norm(dlg)/np.linalg.norm(logits['q4k'][0]):.4%}")
    print(f"  max|logit diff| = {np.abs(dlg).max():.4f}  (vs typical margin above)")

    print(f"\n=== full-block draft tokens (argmax of logits[i]+markov) ===")
    for label in GGUFS:
        T = build_drafter_ctx(GGUFS[label])["T"]
        mw1 = T["mtp.2.markov_head.markov_w1.weight"][0]
        mw2 = T["mtp.2.markov_head.markov_w2.weight"][0]
        out = np.zeros(BLOCK + 1, dtype=np.int64); out[0] = anchor
        for i in range(BLOCK):
            bias = mw1[out[i]] @ mw2.T
            li = logits[label][i] + bias
            out[i + 1] = int(np.argmax(li))
        print(f"  {label}: draft={out[1:].tolist()}")


if __name__ == "__main__":
    main()
