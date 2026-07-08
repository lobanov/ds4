#!/usr/bin/env python3
"""Activity 6 — from-scratch head upper bound [GATE] (RESEARCH INSTRUMENTATION).

Trains a from-scratch head (hc_head + norm + markov_w1/w2 trainable; lm_head frozen)
on the frozen pre-head features x (torch MPS body) to predict the p=1 target, and
reports the held-out p=1 ceiling. Also reports (a) the frozen pretrained head's p=1
(the no-LoRA baseline / 'pre-') and (b) an overfit-to-eval check (train on eval ->
p=1 ~1.0 proves the head/impl can fit). Gate decision: high ceiling (~0.88-0.91) =>
h carries the info, proceed to 7; caps low (~0.83) => h lacks info, move toward 8.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "dspark_train"))
from drafter_head import build_head, HC, DIM, NORM_EPS, HC_EPS  # noqa
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


class FromScratchHead(nn.Module):
    def __init__(self, lm_head):
        super().__init__()
        self.register_buffer("lm_head", lm_head)
        self.hc_fn = nn.Parameter(torch.empty(HC, HC * DIM))
        self.hc_scale = nn.Parameter(torch.zeros(1))
        self.hc_base = nn.Parameter(torch.zeros(HC))
        self.norm_w = nn.Parameter(torch.ones(DIM))
        self.markov_w1 = nn.Parameter(torch.empty(129280, 256))
        self.markov_w2 = nn.Parameter(torch.empty(129280, 256))
        nn.init.normal_(self.hc_fn, std=0.02)
        nn.init.normal_(self.markov_w1, std=0.02)
        nn.init.normal_(self.markov_w2, std=0.02)

    def p1_scores(self, x0, anchor):  # x0 [N,HC,DIM], anchor [N] -> [N,VOCAB]
        flat = x0.reshape(x0.shape[0], HC * DIM)
        rsqrt = torch.rsqrt((flat * flat).mean(-1, keepdim=True) + NORM_EPS)
        mixes = (flat @ self.hc_fn.T) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_scale[0] + self.hc_base) + HC_EPS
        h = (pre.unsqueeze(-1) * x0).sum(1)
        h = h * torch.rsqrt((h * h).mean(-1, keepdim=True) + NORM_EPS) * self.norm_w
        base = h @ self.lm_head.T
        bias = self.markov_w1[anchor] @ self.markov_w2.T
        return base + bias


def p1_acc(head, x, anchor, target, bs=256):
    n = x.shape[0]; match = 0
    head.eval()
    with torch.no_grad():
        for i in range(0, n, bs):
            xb = x[i:i+bs].to(DEV); ab = anchor[i:i+bs].to(DEV)
            if hasattr(head, "p1_scores"):
                pred = head.p1_scores(xb[:, 0], ab).argmax(-1)
            else:  # DrafterHead (frozen) -> full rollout, take p=1
                out, _ = head(xb, ab); pred = out[:, 1]
            match += (pred.cpu() == target[i:i+bs]).sum().item()
    return match / n


def train_head(head, x, anchor, target, epochs=8, lr=1e-3, bs=256, tag=""):
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    n = x.shape[0]; head.train()
    for ep in range(epochs):
        perm = torch.randperm(n); tl = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            s = head.p1_scores(x[idx, 0].to(DEV), anchor[idx].to(DEV))
            loss = torch.nn.functional.cross_entropy(s, target[idx].to(DEV))
            opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item() * len(idx)
        print(f"  {tag}epoch {ep+1}: train CE {tl/n:.4f}", flush=True)


def main():
    print("loading features...", flush=True)
    xtr, atr, ttr = load_features("train"); xev, aev, tev = load_features("eval")
    print(f"train {xtr.shape[0]} eval {xev.shape[0]} anchors", flush=True)
    # frozen pretrained head baseline (the 'pre-')
    print("frozen pretrained head (no-LoRA baseline)...", flush=True)
    frozen = build_head(DSPARK, TARGET, device=DEV, lora_rank=0, dtype=torch.float32).eval()
    # reuse its lm_head for the from-scratch head
    lm_head = frozen.lm_head
    base_p1 = p1_acc(frozen, xev, aev, tev)
    print(f"  frozen-head p=1 on eval (no-LoRA baseline / pre-): {base_p1:.4f}", flush=True)

    # from-scratch head ceiling
    print("training from-scratch head on train...", flush=True)
    fs = FromScratchHead(lm_head.to(DEV)).to(DEV)
    train_head(fs, xtr, atr, ttr, epochs=25, lr=2e-3, tag="train ")
    ceil = p1_acc(fs, xev, aev, tev)
    print(f"  from-scratch head p=1 on eval (HEAD-ONLY CEILING): {ceil:.4f}", flush=True)

    # overfit-to-eval sanity check
    print("overfit-to-eval check (train on eval)...", flush=True)
    fs2 = FromScratchHead(lm_head.to(DEV)).to(DEV)
    train_head(fs2, xev, aev, tev, epochs=60, lr=2e-3, tag="overfit ")
    oe = p1_acc(fs2, xev, aev, tev)
    print(f"  overfit-to-eval p=1: {oe:.4f} (want ~1.0 -> head can fit)", flush=True)

    print(f"\n=== Activity 6 SUMMARY ===")
    print(f"  no-LoRA baseline (frozen head):   {base_p1:.4f}")
    print(f"  head-only ceiling (from-scratch): {ceil:.4f}")
    print(f"  overfit-to-eval:                  {oe:.4f}")
    gate = "proceed to 7 (LoRA)" if ceil >= base_p1 + 0.03 else "h lacks info -> consider 8"
    print(f"  gate: {gate}")
    import json
    (DATA / "activity6_summary.json").write_text(json.dumps({
        "no_lora_baseline_p1": base_p1, "head_only_ceiling_p1": ceil,
        "overfit_to_eval_p1": oe, "gate": gate}, indent=2) + "\n")


if __name__ == "__main__":
    main()
