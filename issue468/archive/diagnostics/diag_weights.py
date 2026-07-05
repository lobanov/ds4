#!/usr/bin/env python3
"""Focused follow-up: expert-weight diffs across GGUFs, logit margins, and a
read_dense_expert vs read_tensor cross-check to rule out a silent loader bug."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "dspark_oracle"))
from gguf_loader import index_gguf, read_tensor, dequant_q4_k_expert, read_dense_expert

GGUFS = {
    "q4k": "/Users/lobanov/Projects/ds4/gguf/dspark.gguf",
    "f16": str(HERE / "artifacts/dspark_ceiling/dspark_f16.gguf"),
    "f32": str(HERE / "artifacts/dspark_ceiling/dspark_f32.gguf"),
}
TYPE_NAME = {0: "F32", 1: "F16", 30: "BF16", 8: "Q8_0", 12: "Q4_K"}


def get_expert(path, layer, part, e):
    nm = f"mtp.{layer}.ffn_{part}_exps.weight"
    kv, infos, doff = index_gguf(path)
    dims, tt, off = infos[nm]
    if tt == 12:
        return dequant_q4_k_expert(path, infos, doff, nm, e), tt
    return read_dense_expert(path, infos, doff, nm, e), tt


def main():
    print("===== expert weight diffs (the decisive number) =====")
    for layer, part in [(0, "gate"), (0, "down"), (2, "gate")]:
        print(f"--- mtp.{layer}.ffn_{part}_exps, expert 0 ---")
        arrs, types = {}, {}
        for label, path in GGUFS.items():
            a, tt = get_expert(path, layer, part, 0)
            arrs[label] = a; types[label] = tt
            print(f"  {label}: type={TYPE_NAME.get(tt,tt)} shape={a.shape} mean={a.mean():.5f} std={a.std():.5f} "
                  f"absmean={np.abs(a).mean():.5f}")
        base = np.abs(arrs["f32"])
        for a, b in [("q4k", "f32"), ("f16", "f32"), ("q4k", "f16")]:
            d = np.abs(arrs[a] - arrs[b])
            print(f"  |{a}-{b}|: mean={d.mean():.6f} max={d.max():.4f} rel_vs_absmean={d.mean()/(np.abs(arrs[a]).mean()+1e-9):.4%}")

    print("\n===== cross-check: read_dense_expert vs full-tensor read_tensor (F32) =====")
    path = GGUFS["f32"]
    kv, infos, doff = index_gguf(path)
    nm = "mtp.0.ffn_gate_exps.weight"
    full = read_tensor(path, infos, doff, nm)        # [n_exp, out, in]
    print(f"  full tensor shape={full.shape} dtype={full.dtype}")
    for e in (0, 1, 127, 255):
        sl = full[e]
        rd = read_dense_expert(path, infos, doff, nm, e)
        ok = np.array_equal(sl, rd)
        print(f"  expert {e}: slice==read_dense_expert? {ok}  (max diff {np.abs(sl-rd).max():.2e})")

    print("\n===== cross-check: Q4_K dequant vs read_tensor-style (sanity, q4k) =====")
    path = GGUFS["q4k"]
    kv, infos, doff = index_gguf(path)
    nm = "mtp.0.ffn_gate_exps.weight"
    dims, tt, off = infos[nm]
    print(f"  type={TYPE_NAME.get(tt,tt)} dims={dims}")
    # read_tensor would dequant the whole 3D Q4_K tensor (huge) — skip; just confirm type

    print("\n===== are F32 expert bytes actually F32 (not accidentally Q4_K data)? =====")
    path = GGUFS["f32"]
    kv, infos, doff = index_gguf(path)
    nm = "mtp.0.ffn_gate_exps.weight"
    dims, tt, off = infos[nm]
    in_dim, out_dim, n_exp = dims
    elems_per_exp = in_dim * out_dim
    import struct
    with open(path, "rb") as f:
        f.seek(doff + off)
        raw = f.read(elems_per_exp * 4)  # expert 0, F32 = 4 bytes/elem
    direct = np.frombuffer(raw, dtype=np.float32).copy()
    rd = read_dense_expert(path, infos, doff, nm, 0)
    # read_dense_expert applies _gguf_ne_to_torch(arr,[in,out]) -> reshapes reversed; flatten both for compare
    print(f"  type={TYPE_NAME.get(tt,tt)}; raw-F32 expert0 mean={direct.mean():.5f} std={direct.std():.5f}")
    print(f"  read_dense_expert expert0 mean={rd.mean():.5f} std={rd.std():.5f}")
    print(f"  same elements (as set, reordered by ne->torch)? {np.allclose(np.sort(direct), np.sort(rd.flatten()), atol=1e-5)}")


if __name__ == "__main__":
    main()
