#!/usr/bin/env python3
"""Activity 9 — reproducible Lce+Ltv McNemar for the best head-LoRA config.

Trains head-LoRA rank 32 seed 42 with the specified Lce+Ltv loss, then computes the
paired McNemar (frozen vs LoRA per-anchor p=1 correctness on eval) and writes
artifacts/stage2_results/mcnemar_lce_ltv.json. This makes the headline McNemar
reproducible from committed code (run_stage2_final_eval.py is the CE-only McNemar;
this is the Lce+Ltv one). NOTE: MPS matmul is mildly nondeterministic run-to-run, so
the exact discordants vary slightly; the verdict (significant harm, p<0.05) is robust.
"""
from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "dspark_train"))
from drafter_head import build_head                       # noqa
from run_stage2_head_lora_kloss import load, train, DEV   # noqa
GGUF = HERE.parent.parent / "ds4" / "gguf"
DSPARK = str((GGUF / "dspark.gguf").resolve())
TARGET = str((GGUF / "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf").resolve())
OUT = HERE / "artifacts" / "stage2_results" / "mcnemar_lce_ltv.json"


def preds(head, x, anchor, target_k, bs=128):
    n = x.shape[0]; out = []; head.eval()
    with torch.no_grad():
        for i in range(0, n, bs):
            out.append(head.p1_scores(x[i:i+bs, 0], anchor[i:i+bs].long()).argmax(-1).cpu())
    p = torch.cat(out)
    return (p == target_k[:, 0]).numpy().astype(np.int8)


def mcnemar(pre, post):
    b = int(((pre == 1) & (post == 0)).sum()); c = int(((pre == 0) & (post == 1)).sum())
    tot = b + c; k = min(b, c)
    p = (2 * sum(comb(tot, i) for i in range(k + 1)) * 0.5 ** tot) if tot else 1.0
    return b, c, min(1.0, p)


def main():
    tr = load("train"); ev = load("eval")
    frozen = build_head(DSPARK, TARGET, device=DEV, lora_rank=0, dtype=torch.float32).eval()
    pre = preds(frozen, ev["x"], ev["anchor"], ev["target_k"])
    torch.manual_seed(42)
    lora = build_head(DSPARK, TARGET, device=DEV, lora_rank=32, dtype=torch.float32)
    train(lora, tr, epochs=6)
    post = preds(lora, ev["x"], ev["anchor"], ev["target_k"])
    b, c, p = mcnemar(pre, post)
    res = {"config": "head-LoRA rank32 seed42, Lce+Ltv (exp weights gamma=5), 6 epochs",
           "pre_p1": float(pre.mean()), "post_p1": float(post.mean()),
           "delta_pp": (float(post.mean()) - float(pre.mean())) * 100,
           "mcnemar_pre_only": b, "mcnemar_post_only": c, "mcnemar_p": p, "n": int(len(pre))}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
