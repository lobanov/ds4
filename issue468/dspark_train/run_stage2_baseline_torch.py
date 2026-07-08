#!/usr/bin/env python3
"""Activity 5 — confirm torch head reproduces the numpy drafter's drafts on Stage 2 eval.

Loads eval_features.npz (x + numpy draft0 from run_stage2_baseline.py), runs the torch
head (LoRA=0), and checks torch draft0 == numpy draft0 (p=1 identical) — a direct
in-pipeline confirmation that numpy-body + torch-head == numpy-body + numpy-head.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent.parent            # issue468/
sys.path.insert(0, str(HERE / "dspark_train"))
from drafter_head import build_head                       # noqa: E402

FEAT = HERE / "dspark_train" / "data" / "eval_features.npz"


def main() -> int:
    gguf_dir = HERE.parent.parent / "ds4" / "gguf"
    d = np.load(FEAT)
    x = torch.from_numpy(d["x"].astype(np.float32))
    anchor = torch.from_numpy(d["anchor"].astype(np.int64))
    np_draft0 = d["numpy_draft0"]
    n = x.shape[0]
    print(f"eval anchors: {n}", flush=True)
    head = build_head(str((gguf_dir / "dspark.gguf").resolve()),
                      str((gguf_dir / "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf").resolve()),
                      device="cpu", lora_rank=0, dtype=torch.float32)
    head.eval()
    torch_draft0 = []
    with torch.no_grad():
        BS = 16
        for i in range(0, n, BS):
            out, _ = head(x[i:i + BS], anchor[i:i + BS])
            torch_draft0.append(out[:, 1])
    torch_draft0 = torch.cat(torch_draft0).numpy()
    match = (torch_draft0 == np_draft0)
    print(f"torch vs numpy draft0 (p=1) match: {match.sum()}/{n} = {match.mean()*100:.2f}%")
    print(f"-> numpy-body+torch-head p1 == numpy-body+numpy-head p1: {'CONFIRMED' if match.all() else 'MISMATCH'}")
    return 0 if match.all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
