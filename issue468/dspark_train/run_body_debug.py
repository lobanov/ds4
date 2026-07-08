#!/usr/bin/env python3
"""Activity 5 debug — localize the torch-body fidelity bug per layer.

Runs the numpy oracle forward (reusing dspark_oracle primitives verbatim) and the
torch body forward on one prompt's step 1, both capturing x AFTER each layer, and
prints the per-layer max abs diff to localize where the port diverges.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "dspark_oracle"))
import measure_acceptance_bundle as mab              # noqa
from measure_acceptance_bundle import dspark_attn    # noqa
import forward as fwd                                # noqa
from forward import BLOCK, HC, DIM, NOISE_TOK        # noqa
from hc_primitives import rmsnorm, hc_pre, hc_post   # noqa
from moe import moe                                  # noqa
from attention import apply_rotary, ROPE_DIM         # noqa
sys.path.insert(0, str(HERE / "dspark_train"))
from drafter_body import build_body                   # noqa

GGUF = HERE.parent.parent / "ds4" / "gguf"
MODEL = str((GGUF / "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf").resolve())
DSPARK = str((GGUF / "dspark.gguf").resolve())
BUNDLES = HERE / "artifacts" / "exactness_small_bundles"


def numpy_step1(bdir, mctx, dctx):
    T = dctx["T"]; layers = dctx["layers"]; stores = dctx["stores"]
    main_proj = dctx["main_proj"]; main_norm_w = dctx["main_norm_w"]
    embed_w = mctx["embed_w"]; cos, sin = mctx["cos"], mctx["sin"]
    man = json.loads((bdir / "bundle_manifest.json").read_text())
    pos0 = int(man["prompt_tokens"]); toks = json.loads((bdir / "target_selected_tokens.json").read_text())
    win_kv = [np.zeros((128, 512), np.float32) for _ in range(3)]
    mh0 = mab.load_mh(bdir, pos0)
    main_x0 = rmsnorm(mh0.reshape(1, 1, 3 * DIM) @ main_proj.T, main_norm_w)
    for s in range(3):
        mkv = rmsnorm(main_x0 @ layers[s]["kv"].T, layers[s]["kv_a_norm"])
        mkv[..., -ROPE_DIM:] = apply_rotary(mkv[..., -ROPE_DIM:], cos[0], sin[0])
        win_kv[s][0] = mkv[0, 0]
    n_real = 1
    # step 1
    anchor = int(toks[1]); mh = mab.load_mh(bdir, pos0 + 1)
    main_x = rmsnorm(mh.reshape(1, 1, 3 * DIM) @ main_proj.T, main_norm_w)
    for s in range(3):
        mkv = rmsnorm(main_x @ layers[s]["kv"].T, layers[s]["kv_a_norm"])
        mkv[..., -ROPE_DIM:] = apply_rotary(mkv[..., -ROPE_DIM:], cos[1], sin[1])
        win_kv[s][1] = mkv[0, 0]
    n_real = 2
    draft_ids = np.full(BLOCK, NOISE_TOK, dtype=np.int64); draft_ids[0] = anchor
    x = embed_w[draft_ids][None]; x = np.repeat(x[:, :, None, :], HC, axis=2)
    layer_xs = []
    for s in range(3):
        res = x
        yd, post, comb = hc_pre(x, layers[s]["hc_attn_fn"], layers[s]["hc_attn_scale"], layers[s]["hc_attn_base"])
        yd = rmsnorm(yd, layers[s]["attn_norm"])
        ao = dspark_attn(yd, win_kv[s], n_real, layers[s], cos, sin, 1)
        x = hc_post(ao, res, post, comb)
        res = x
        yd, post, comb = hc_pre(x, layers[s]["hc_ffn_fn"], layers[s]["hc_ffn_scale"], layers[s]["hc_ffn_base"])
        yd = rmsnorm(yd, layers[s]["ffn_norm"])
        fo = moe(yd, np.array([0]), layers[s], stores[s])
        x = hc_post(fo, res, post, comb)
        layer_xs.append(x[0].copy())
    return layer_xs  # [3, BLOCK, HC, DIM]


def main():
    p = "grounded_observatory"
    bdir = BUNDLES / f"{p}__t0p0"
    print("numpy mctx/dctx...", flush=True)
    mctx = mab.build_model_ctx(MODEL); dctx = mab.build_drafter_ctx(DSPARK)
    print("numpy step1...", flush=True)
    np_layers = numpy_step1(bdir, mctx, dctx)
    print("torch body (cache)...", flush=True)
    body = build_body(DSPARK, MODEL, device="cpu", dtype=torch.float32)
    man = json.loads((bdir / "bundle_manifest.json").read_text()); pos0 = int(man["prompt_tokens"])
    toks = json.loads((bdir / "target_selected_tokens.json").read_text())
    mh_seq = [np.asarray(mab.load_mh(bdir, pos0 + i), np.float32) for i in range(0, 2)]
    _, torch_layers = body.forward_prompt(mh_seq, [int(toks[1])], return_layers=True)
    torch_layers = torch_layers.numpy()  # [1,3,BLOCK,HC,DIM]
    print("\nper-layer max abs diff (torch vs numpy), step 1:")
    for s in range(3):
        d = np.abs(torch_layers[0, s] - np_layers[s])
        print(f"  layer {s}: max {d.max():.4f}  mean {d.mean():.6f}  (|x| mean {np.abs(np_layers[s]).mean():.3f})")


if __name__ == "__main__":
    main()
