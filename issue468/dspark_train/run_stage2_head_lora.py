#!/usr/bin/env python3
"""Activity 7 — head-only LoRA rank sweep (32/64/128) on the pretrained head.

Builds the pretrained DrafterHead with LoRA adapters (zero-init -> starts == frozen),
trains the LoRA params on p=1 CE over train features, sweeps rank 32/64/128, reports
held-out p=1 vs the no-LoRA baseline. Decision rule: reaffirm >=+5pp; not-justified
<+1pp after the rank sweep. Activity 8 trigger: best LoRA < Activity 6 ceiling.
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


def p1(head, x, anchor, target, bs=256):
    n = x.shape[0]; m = 0; head.eval()
    with torch.no_grad():
        for i in range(0, n, bs):
            s = head.p1_scores(x[i:i+bs, 0], anchor[i:i+bs])
            m += (s.argmax(-1).cpu() == target[i:i+bs]).sum().item()
    return m / n


def train(head, x, a, t, epochs=12, lr=3e-4, bs=256, wd=0.01, tag=""):
    params = [p for p in head.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    n = x.shape[0]; head.train()
    for ep in range(epochs):
        perm = torch.randperm(n); tl = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            s = head.p1_scores(x[idx, 0], a[idx])
            loss = torch.nn.functional.cross_entropy(s, t[idx].to(DEV))
            opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item() * len(idx)
        if (ep + 1) % 6 == 0 or ep == 0:
            print(f"  {tag}epoch {ep+1}: CE {tl/n:.4f}", flush=True)


def main():
    torch.manual_seed(42)
    xtr, atr, ttr = load_features("train"); xev, aev, tev = load_features("eval")
    print(f"train {xtr.shape[0]} eval {xev.shape[0]}", flush=True)
    a6 = json.loads((DATA / "activity6_summary.json").read_text())
    results = {}
    for rank in [32, 64, 128]:
        head = build_head(DSPARK, TARGET, device=DEV, lora_rank=rank, dtype=torch.float32)
        base = p1(head, xev, aev, tev)
        train(head, xtr, atr, ttr, epochs=12, lr=3e-4, tag=f"r{rank} ")
        lp = p1(head, xev, aev, tev)
        print(f"rank {rank}: no-LoRA {base:.4f} -> LoRA {lp:.4f} (delta {(lp-base)*100:+.2f} pp)", flush=True)
        results[rank] = {"baseline": base, "lora": lp, "delta_pp": (lp - base) * 100}
    best_rank = max(results, key=lambda r: results[r]["lora"])
    best = results[best_rank]["lora"]; base0 = results[32]["baseline"]
    print("\n=== Activity 7 RANK SWEEP ===")
    print(f"  baseline (no-LoRA): {base0:.4f}")
    for r in [32, 64, 128]:
        print(f"  rank {r}: {results[r]['lora']:.4f} ({results[r]['delta_pp']:+.2f} pp)")
    print(f"  best: rank {best_rank} = {best:.4f} ({(best-base0)*100:+.2f} pp)")
    print(f"  from-scratch ceiling (A6): {a6['head_only_ceiling_p1']:.4f}")
    gate = (best - base0) * 100
    verdict = ("head path REAFFIRM (>=+5pp)" if gate >= 5
               else f"head path NOT-JUSTIFIED (<+1pp after rank 32/64/128 sweep); best +{gate:.2f}pp" if gate < 1
               else f"head path INCONCLUSIVE ({gate:+.2f}pp)")
    print(f"  verdict: {verdict}")
    print(f"  Activity 8 trigger (best<A6ceiling?): {'YES' if best < a6['head_only_ceiling_p1'] else 'NO'}")
    (DATA / "activity7_summary.json").write_text(json.dumps({
        "baseline_p1": base0, "rank_sweep": {str(r): results[r] for r in [32, 64, 128]},
        "best_rank": best_rank, "best_p1": best, "best_delta_pp": (best - base0) * 100,
        "from_scratch_ceiling_p1": a6["head_only_ceiling_p1"], "verdict": verdict}, indent=2) + "\n")


if __name__ == "__main__":
    main()
