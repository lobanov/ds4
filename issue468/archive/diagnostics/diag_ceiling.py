#!/usr/bin/env python3
"""One-bundle (temp=0) diagnostic: why are Q4_K / F16 / F32 acceptance identical?

Isolates whether the cause is:
  (A) the F32/F16 build didn't change expert/dense bytes (build bug),
  (B) the loaders return identical data (loader bug), or
  (C) the forward is genuinely insensitive to drafter weight noise.

Runs only code_histogram__t0p0. Prints: tensor types, expert-0 / main_proj value
diffs across GGUFs, per-step draft-token lists, and a zeroed-experts control.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "dspark_oracle"))
from gguf_loader import index_gguf, read_tensor, dequant_q4_k_expert, read_dense_expert, load_gguf_dense_only
from expert_store import ExpertStore
import moe as moe_mod
from measure_acceptance_bundle import build_model_ctx, build_drafter_ctx, measure_bundle

GGUFS = {
    "q4k": "/Users/lobanov/Projects/ds4/gguf/dspark.gguf",
    "f16": str(HERE / "artifacts/dspark_ceiling/dspark_f16.gguf"),
    "f32": str(HERE / "artifacts/dspark_ceiling/dspark_f32.gguf"),
}
BUNDLE = HERE / "artifacts/exactness_small_bundles/code_histogram__t0p0"
MODEL = "/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
TYPE_NAME = {0: "F32", 1: "F16", 30: "BF16", 8: "Q8_0", 12: "Q4_K"}


def expert0_gate(path):
    infos, doff = index_gguf(path)[1], index_gguf(path)[2]
    _, ti, _ = index_gguf(path)
    nm = "mtp.0.ffn_gate_exps.weight"
    dims, tt, off = infos[nm]
    if tt == 12:
        return dequant_q4_k_expert(path, infos, doff, nm, 0), tt
    return read_dense_expert(path, infos, doff, nm, 0), tt


def main():
    print("===== (A/B) do the weights actually differ between GGUFs? =====")
    exps, types = {}, {}
    for label, path in GGUFS.items():
        e, tt = expert0_gate(path)
        exps[label] = e; types[label] = tt
        print(f"  {label}: mtp.0.ffn_gate_exps type={TYPE_NAME.get(tt,tt)} expert0 shape={e.shape} mean={e.mean():.5f} std={e.std():.5f}")
    for a, b in [("q4k", "f16"), ("q4k", "f32"), ("f16", "f32")]:
        d = np.abs(exps[a] - exps[b])
        print(f"  |{a}-{b}| expert0 gate: mean={d.mean():.6f} max={d.max():.4f} rel={d.mean()/(np.abs(exps[a]).mean()+1e-9):.4%}")

    # dense tensor: main_proj (read lazily, single tensor)
    print("  --- dense tensor mtp.0.main_proj.weight ---")
    mp = {}
    for label, path in GGUFS.items():
        infos, doff = index_gguf(path)[1], index_gguf(path)[2]
        arr = read_tensor(path, infos, doff, "mtp.0.main_proj.weight")
        mp[label] = arr
        print(f"  {label}: main_proj shape={arr.shape} mean={arr.mean():.5f} std={arr.std():.5f} type={TYPE_NAME.get(infos['mtp.0.main_proj.weight'][1], '?')}")
    for a, b in [("q4k", "f16"), ("q4k", "f32"), ("f16", "f32")]:
        d = np.abs(mp[a] - mp[b])
        print(f"  |{a}-{b}| main_proj: mean={d.mean():.6f} max={d.max():.4f}")

    print("\n===== (C) per-step drafts for code_histogram__t0p0 =====")
    mctx = build_model_ctx(MODEL)
    summaries = {}
    for label, path in GGUFS.items():
        dctx = build_drafter_ctx(path)
        s = measure_bundle(BUNDLE, mctx, dctx, candidate=label)
        summaries[label] = s
        drafts = [r["draft"] for r in s["rows"]]
        print(f"  {label}: avg_prefix={s['average_prefix']:.3f}")
        for i, r in enumerate(s["rows"]):
            print(f"    step{i+1} draft={r['draft']} target={r['target']} prefix={r['prefix']}")
        print()

    print("===== are per-step drafts identical across GGUFs? =====")
    for i in range(len(summaries["q4k"]["rows"])):
        dq = summaries["q4k"]["rows"][i]["draft"]
        df16 = summaries["f16"]["rows"][i]["draft"]
        df32 = summaries["f32"]["rows"][i]["draft"]
        print(f"  step{i+1}: q4k==f16? {dq==df16}  q4k==f32? {dq==df32}  f16==f32? {df16==df32}")

    print("\n===== control: zero out experts — do drafts change? =====")
    orig_moe = moe_mod.moe
    dctx_q4k = build_drafter_ctx(GGUFS["q4k"])
    def zero_moe(x, input_ids, w, store, topk=moe_mod.TOPK):
        b, s, dim = x.shape
        return np.zeros((b, s, dim), dtype=np.float32)
    moe_mod.moe = zero_moe
    # measure_bundle imports moe by name into its module namespace, so patch there too
    import measure_acceptance_bundle as mab
    mab.moe = zero_moe
    s_zero = measure_bundle(BUNDLE, mctx, dctx_q4k, candidate="q4k_zeroexperts")
    moe_mod.moe = orig_moe
    mab.moe = orig_moe
    print(f"  q4k ZERO-experts: avg_prefix={s_zero['average_prefix']:.3f} (vs q4k normal {summaries['q4k']['average_prefix']:.3f})")
    same = all(summaries["q4k"]["rows"][i]["draft"] == s_zero["rows"][i]["draft"] for i in range(len(s_zero["rows"])))
    print(f"  drafts identical to normal q4k? {same}")
    for i, r in enumerate(s_zero["rows"]):
        print(f"    step{i+1} draft={r['draft']} target={r['target']} prefix={r['prefix']}")


if __name__ == "__main__":
    main()
