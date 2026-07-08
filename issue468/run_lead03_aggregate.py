#!/usr/bin/env python3
"""Lead 03 — aggregate per-prompt acceptance checkpoints -> powered stats + power calc.

Reads a per_prompt/<pid>.json checkpoint dir (compact schema: E_a_5block, p1,
per_position_match, prefix_hist, drafts) produced by EITHER the numpy runner
(run_lead03_measure_stage2) OR the torch runner (run_lead03_torch_measure), and emits
the powered aggregate: E[a|K] K=1..5, S(K), per-position match, p1, the prompt-clustered
bootstrap CI on E[a|4], the empirical prompt-level sd, and the power-calc N for clustered
95% half-width 0.05 (floor) and 0.028 (stretch).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BLOCK = 5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-prompt-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260707)
    args = ap.parse_args()

    ppdir = Path(args.per_prompt_dir)
    files = sorted(ppdir.glob("*.json"))
    rows = [json.loads(f.read_text()) for f in files]
    n_prompts = len(rows)
    if n_prompts == 0:
        print("no checkpoints"); return 1
    # per-prompt E[a|4] from prefix_hist (E[a|4] = S(1)+S(2)+S(3)+S(4); S(j)=P(prefix>=j))
    def ea4_of(r):
        h = np.array(r["prefix_hist"], dtype=float)
        tot = h.sum()
        if tot == 0: return 0.0
        return sum(h[j:].sum() for j in range(1, 5)) / tot
    ea4 = np.array([ea4_of(r) for r in rows])
    ea5 = np.array([r["E_a_5block"] for r in rows])
    p1 = np.array([r["p1"] for r in rows])
    # aggregate cycle-level prefix hist -> S(k), E[a|k]
    agg_hist = np.sum([np.array(r["prefix_hist"]) for r in rows], axis=0)
    total_cycles = int(agg_hist.sum())
    S = np.array([agg_hist[j:].sum() / total_cycles for j in range(1, BLOCK + 1)])
    Eak = np.array([S[:k].sum() for k in range(BLOCK + 1)])
    # clustered bootstrap CI on E[a|4]
    rng = np.random.default_rng(args.seed)
    idx = np.arange(n_prompts)
    boot = np.array([ea4[rng.choice(idx, n_prompts, replace=True)].mean() for _ in range(args.n_boot)])
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    sd = float(ea4.std(ddof=1))
    hw = (ci[1] - ci[0]) / 2

    def n_for(t):
        return 0 if sd <= 0 else int(math.ceil((1.96 * sd / t) ** 2))

    agg = {
        "n_prompts": n_prompts, "total_cycles": total_cycles,
        "E_a_4": round(float(ea4.mean()), 5), "E_a_5": round(float(ea5.mean()), 5),
        "S_k": {f"S({k})": round(float(S[k-1]), 5) for k in range(1, BLOCK + 1)},
        "E_ak": {f"E[a|{k}]": round(float(Eak[k]), 5) for k in range(1, BLOCK + 1)},
        "p1_mean": round(float(p1.mean()), 5),
        "per_position_match_rate": [round(float(sum(r["per_position_match"][i] for r in rows) / total_cycles), 5)
                                    for i in range(BLOCK)],
        "prompt_sd_E_a_4": round(sd, 5),
        "prompt_clustered_ci95_E_a_4": [round(ci[0], 5), round(ci[1], 5)],
        "ci_half_width": round(hw, 5),
        "power_calc": {"N_for_hw_0.05": n_for(0.05), "N_for_hw_0.028": n_for(0.028),
                       "note": "N=(1.96*sd/hw)^2 from empirical prompt-level sd"},
        "sources": dict((s, sum(1 for r in rows if r.get("source") == s))
                        for s in set(r.get("source", "?") for r in rows)),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(agg, indent=2) + "\n")
    print(json.dumps(agg, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
