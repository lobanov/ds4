#!/usr/bin/env python3
"""Activity 5 — torch-head fidelity gate.

Loads the numpy oracle's frozen pre-head features + numpy forward_head reference
(exactness_features.npz), runs the torch head (LoRA=0, float32 CPU for precision
parity with numpy), and checks: (1) argmax/output_ids parity, (2) full-score
base_logits max+mean abs diff, (3) top-128 set parity. The gate passes when argmax
parity is 100% and base_logits/top-128 are within tolerance.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent.parent            # issue468/
sys.path.insert(0, str(HERE / "dspark_train"))
from drafter_head import build_head, BLOCK, VOCAB        # noqa: E402

FEAT = HERE / "dspark_train" / "data" / "exactness_features.npz"


def main() -> int:
    gguf_dir = HERE.parent.parent / "ds4" / "gguf"
    dspark = str((gguf_dir / "dspark.gguf").resolve())
    target = str((gguf_dir / "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf").resolve())

    d = np.load(FEAT)
    x = torch.from_numpy(d["x"].astype(np.float32))           # [N,BLOCK,HC,DIM]
    anchor = torch.from_numpy(d["anchor"].astype(np.int64))   # [N]
    np_out = d["out"]                                          # [N,BLOCK+1]
    np_base = d["base_logits"]                                 # [N,BLOCK,VOCAB]
    n = x.shape[0]
    print(f"features: {n} anchors, x{x.shape} base{np_base.shape}", flush=True)

    print("building torch head (LoRA=0, float32 CPU)...", flush=True)
    head = build_head(dspark, target, device="cpu", lora_rank=0, dtype=torch.float32)
    head.eval()
    with torch.no_grad():
        torch_out, torch_base = [], []
        BS = 16
        for i in range(0, n, BS):
            o, b = head(x[i:i + BS], anchor[i:i + BS])        # o [b,BLOCK+1], b [b,BLOCK,VOCAB]
            torch_out.append(o); torch_base.append(b)
    torch_out = torch.cat(torch_out).numpy()                  # [N,BLOCK+1]
    torch_base = torch.cat(torch_base).numpy()                # [N,BLOCK,VOCAB]

    # (1) argmax / output_ids parity
    argmax_match = (torch_out == np_out)
    argmax_rate = argmax_match.mean()
    # p=1 specifically (output_ids[:,1] == np_out[:,1])
    p1_rate = (torch_out[:, 1] == np_out[:, 1]).mean()

    # (2) full-score base_logits diff
    diff = np.abs(torch_base - np_base)
    max_abs = float(diff.max()); mean_abs = float(diff.mean())

    # (3) top-128 set parity per (anchor, position)
    overlaps = []
    exact_top128 = 0
    npos = 0
    for a in range(n):
        for p in range(BLOCK):
            nv = np_base[a, p]; tv = torch_base[a, p]
            n_top = set(np.argpartition(-nv, 128)[:128].tolist())
            t_top = set(np.argpartition(-tv, 128)[:128].tolist())
            overlaps.append(len(n_top & t_top) / 128)
            exact_top128 += (n_top == t_top)
            npos += 1
    top128_overlap = float(np.mean(overlaps))
    top128_exact_rate = exact_top128 / npos

    print(f"\n=== FIDELITY (torch head LoRA=0 vs numpy forward_head) ===")
    print(f"  argmax output_ids parity: {argmax_rate*100:.2f}% (p=1: {p1_rate*100:.2f}%)")
    print(f"  base_logits max abs diff: {max_abs:.6g} | mean abs diff: {mean_abs:.6g}")
    print(f"  top-128 set overlap (mean): {top128_overlap*100:.4f}% | exact-top128 rate: {top128_exact_rate*100:.2f}%")

    ok = (argmax_rate == 1.0 and top128_overlap >= 0.999 and max_abs < 1e-2)
    print(f"\nFIDELITY GATE: {'PASS' if ok else 'FAIL'}")
    print(f"  (criteria: argmax 100%, top128 overlap >=99.9%, base max-abs <1e-2)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
