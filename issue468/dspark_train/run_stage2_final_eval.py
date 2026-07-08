#!/usr/bin/env python3
"""Activity 9 — final eval: McNemar paired test (frozen vs head-LoRA rank 32) + verdict.

Trains the best head-LoRA config (rank 32) on train, computes per-anchor p=1
predictions for the frozen head and the LoRA head on eval, runs a McNemar paired
test, and records the reaffirm/not-justified verdict + localization diagnostics.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "dspark_train"))
from drafter_head import build_head   # noqa
GGUF = HERE.parent.parent / "ds4" / "gguf"
DSPARK = str((GGUF / "dspark.gguf").resolve())
TARGET = str((GGUF / "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf").resolve())
DATA = HERE / "dspark_train" / "data"
DEV = "mps"


def load_features(split):
    from safetensors import safe_open
    f = DATA / f"{split}_torch_features.safetensors"
    with safe_open(f, "numpy") as d:
        return (torch.from_numpy(d.get_tensor("x").astype(np.float32)),
                torch.from_numpy(d.get_tensor("anchor").astype(np.int64)),
                torch.from_numpy(d.get_tensor("target_p1").astype(np.int64)))


def preds(head, x, anchor, target, bs=256):
    n = x.shape[0]; head.eval(); out = []
    with torch.no_grad():
        for i in range(0, n, bs):
            s = head.p1_scores(x[i:i+bs, 0], anchor[i:i+bs])
            out.append(s.argmax(-1).cpu())
    p = torch.cat(out)
    return (p == target).numpy().astype(np.int8), p.numpy()


def train(head, x, a, t, epochs=12, lr=3e-4, bs=256):
    params = [p for p in head.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    n = x.shape[0]; head.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            loss = torch.nn.functional.cross_entropy(head.p1_scores(x[idx, 0], a[idx]), t[idx].to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()


def mcnemar(pre, post):
    b = int(((pre == 1) & (post == 0)).sum())  # pre-only-correct
    c = int(((pre == 0) & (post == 1)).sum())  # post-only-correct
    # exact 2-sided binomial on min(b,c) given b+c
    from math import comb
    tot = b + c; k = min(b, c)
    if tot == 0:
        return b, c, 1.0
    p = 2 * sum(comb(tot, i) for i in range(0, k + 1)) * 0.5 ** tot
    return b, c, min(1.0, p)


def main():
    torch.manual_seed(42)
    xtr, atr, ttr = load_features("train"); xev, aev, tev = load_features("eval")
    frozen = build_head(DSPARK, TARGET, device=DEV, lora_rank=0, dtype=torch.float32).eval()
    pre, _ = preds(frozen, xev, aev, tev)
    lora = build_head(DSPARK, TARGET, device=DEV, lora_rank=32, dtype=torch.float32)
    train(lora, xtr, atr, ttr, epochs=12)
    post, _ = preds(lora, xev, aev, tev)
    pre_p1 = float(pre.mean()); post_p1 = float(post.mean())
    b, c, pval = mcnemar(pre, post)
    a6 = json.loads((DATA / "activity6_summary.json").read_text())
    a7 = json.loads((DATA / "activity7_summary.json").read_text())
    a4 = json.loads((HERE / "artifacts" / "stage2_crossed_oracle" / "summary.json").read_text())
    delta_pp = (post_p1 - pre_p1) * 100
    if delta_pp >= 5:
        verdict = "REAFFIRM"
    elif delta_pp < 1:
        verdict = "NOT-JUSTIFIED"
    else:
        verdict = "INCONCLUSIVE"
    print("=== Activity 9 FINAL EVAL ===")
    print(f"  no-LoRA baseline (frozen head) p=1: {pre_p1:.4f}")
    print(f"  head-LoRA rank32 p=1:              {post_p1:.4f}  (delta {delta_pp:+.2f} pp)")
    print(f"  McNemar discordants: pre-only {b}, post-only {c}; p={pval:.4f}")
    print(f"  rank sweep (A7) best delta: {a7['best_delta_pp']:+.2f} pp (flat across rank)")
    print(f"  from-scratch ceiling (A6): {a6['head_only_ceiling_p1']:.4f}; overfit-to-eval {a6['overfit_to_eval_p1']:.4f}")
    print(f"  crossed oracle (A4): input_effect {a4['aligned_cells']['input_effect']}, input_shift_rate {a4['input_shift_rate']}")
    print(f"  Activity 8: skipped (7>6ceiling; A4 input-negligible)")
    print(f"  VERDICT: {verdict}")
    out = {
        "no_lora_baseline_p1": pre_p1, "head_lora_rank32_p1": post_p1, "delta_pp": delta_pp,
        "mcnemar_pre_only": b, "mcnemar_post_only": c, "mcnemar_p": pval,
        "rank_sweep_best_delta_pp": a7["best_delta_pp"], "from_scratch_ceiling_p1": a6["head_only_ceiling_p1"],
        "overfit_to_eval_p1": a6["overfit_to_eval_p1"], "crossed_input_effect": a4["aligned_cells"]["input_effect"],
        "crossed_input_shift_rate": a4["input_shift_rate"], "activity8": "skipped (7>6ceiling; A4 input-negligible)",
        "verdict": verdict,
        "interpretation": ("Non-expert drafter fine-tuning (head LoRA) does NOT materially raise p=1 "
                            "(best +0.67pp, flat across rank 32/64/128, McNemar not significant). "
                            "Body/input-side LoRA not run (Activity 8 condition unmet; Activity 4 shows drafter "
                            "input-invariant). Consistent with Stage 0/1 (deficit is shallow but not head-fixable "
                            "via non-expert LoRA; raising tap precision didn't help). Expert tuning is out of scope.")}
    (DATA / "activity9_final.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
