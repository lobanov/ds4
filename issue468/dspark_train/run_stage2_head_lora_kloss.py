#!/usr/bin/env python3
"""Activity 7/9 (corrected) — head-LoRA with the SPECIFIED Lce+Ltv loss + seed sweep.

Trains head-LoRA with Lce + Ltv (total-variation) + exponential position weights
w_k=exp(-(k-1)/gamma) (DSpark §3.3), teacher-forced over K=5 positions, swept over
rank {32,64,128} x seed {42,7,123}. Evaluates held-out p=1 (position-0 argmax vs
target_k[:,0]). This closes the auditor gaps: specified loss + data/seed sweep.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "dspark_train"))
from drafter_head import build_head   # noqa
GGUF = HERE.parent.parent / "ds4" / "gguf"
DSPARK = str((GGUF / "dspark.gguf").resolve())
TARGET = str((GGUF / "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf").resolve())
DATA = HERE / "dspark_train" / "data"
DEV = "mps"; K = 5; GAMMA = 5.0


def load(split):
    from safetensors import safe_open
    f = DATA / f"{split}_kloss_features.safetensors"
    with safe_open(f, "numpy") as d:
        return {k: torch.from_numpy(d.get_tensor(k).astype(
            {"x": np.float32, "anchor": np.float32, "target_k": np.int64, "prev_tok": np.int64,
             "topk_k_ids": np.int64, "topk_k_logprobs": np.float32}[k])) for k in
            ["x", "anchor", "target_k", "prev_tok", "topk_k_ids", "topk_k_logprobs"]}


def lce_ltv(head, x, prev_tok, target_k, topk_ids, topk_lp):
    scores = head.k_scores(x, prev_tok)                     # [N,K,VOCAB]
    logp = F.log_softmax(scores, -1)
    ce = -logp.gather(2, target_k.unsqueeze(-1)).squeeze(-1)            # [N,K]
    p_d = logp.gather(2, topk_ids).exp()                                # [N,K,128]
    p_t = topk_lp.exp()                                                 # [N,K,128]
    term1 = (p_d - p_t).abs().sum(-1)                                   # [N,K]
    term2 = (p_t.sum(-1) - p_d.sum(-1)).abs()                           # [N,K]
    ltv = 0.5 * (term1 + term2)
    w = torch.exp(-torch.arange(K, device=scores.device) / GAMMA)       # [K]
    return (w * (ce + ltv)).sum(-1).mean()


def p1_eval(head, x, anchor, target_k, bs=128):
    n = x.shape[0]; m = 0; head.eval()
    with torch.no_grad():
        for i in range(0, n, bs):
            s = head.p1_scores(x[i:i+bs, 0], anchor[i:i+bs].long())
            m += (s.argmax(-1).cpu() == target_k[i:i+bs, 0]).sum().item()
    return m / n


def train(head, tr, epochs=10, lr=3e-4, bs=128):
    params = [p for p in head.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    n = tr["x"].shape[0]; head.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            loss = lce_ltv(head, tr["x"][idx], tr["prev_tok"][idx].to(DEV), tr["target_k"][idx].to(DEV),
                           tr["topk_k_ids"][idx].to(DEV), tr["topk_k_logprobs"][idx].to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", default="32,64,128")
    ap.add_argument("--seeds", default="42,7")
    ap.add_argument("--epochs", type=int, default=6)
    args = ap.parse_args()
    ranks = [int(x) for x in args.ranks.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    tr = load("train"); ev = load("eval")
    print(f"train {tr['x'].shape[0]} eval {ev['x'].shape[0]}", flush=True)
    base_head = build_head(DSPARK, TARGET, device=DEV, lora_rank=0, dtype=torch.float32).eval()
    base = p1_eval(base_head, ev["x"], ev["anchor"], ev["target_k"])
    print(f"no-LoRA baseline p=1: {base:.4f}", flush=True)
    results = {}
    for rank in ranks:
        run = []
        for seed in seeds:
            torch.manual_seed(seed)
            head = build_head(DSPARK, TARGET, device=DEV, lora_rank=rank, dtype=torch.float32)
            train(head, tr, epochs=args.epochs)
            lp = p1_eval(head, ev["x"], ev["anchor"], ev["target_k"])
            print(f"  rank {rank} seed {seed}: p1 {lp:.4f} (delta {(lp-base)*100:+.2f} pp)", flush=True)
            run.append(lp)
        results[rank] = {"p1_mean": float(np.mean(run)), "p1_std": float(np.std(run)),
                         "delta_mean_pp": (float(np.mean(run)) - base) * 100, "seeds": run}
    print("\n=== Activity 7/9 Lce+Ltv rank x seed sweep ===")
    print(f"  baseline: {base:.4f}")
    for r in ranks:
        rr = results[r]
        print(f"  rank {r}: p1 {rr['p1_mean']:.4f} ± {rr['p1_std']:.4f} (delta {rr['delta_mean_pp']:+.2f} pp) seeds {[round(s,4) for s in rr['seeds']]}")
    best_rank = max(results, key=lambda r: results[r]["p1_mean"])
    best = results[best_rank]["p1_mean"]; gate = (best - base) * 100
    verdict = ("REAFFIRM (>=+5pp)" if gate >= 5
               else f"NOT-JUSTIFIED (<+1pp after rank 32/64/128 + seed sweep); best {gate:+.2f}pp" if gate < 1
               else f"INCONCLUSIVE ({gate:+.2f}pp)")
    print(f"  best: rank {best_rank} = {best:.4f} ({gate:+.2f} pp) -> {verdict}")
    (DATA / "activity7_kloss_summary.json").write_text(json.dumps({
        "baseline_p1": base, "loss": "Lce+Ltv (exp position weights, gamma=5)",
        "rank_seed_sweep": {str(r): results[r] for r in ranks},
        "best_rank": best_rank, "best_p1": best, "best_delta_pp": gate, "verdict": verdict}, indent=2) + "\n")


if __name__ == "__main__":
    main()
