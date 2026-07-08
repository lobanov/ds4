#!/usr/bin/env python3
"""Lead 03 task-4 (CORRECTED) — realistic-trajectory acceptance via K-cycle-jump simulation.

The model's E[a|K] is PER-CYCLE accepted drafts, and real speculative decode advances by
(accepted+1) each cycle (not by 1). So the correct acceptance estimator is a CYCLE-JUMP
simulation from the stored drafts, NOT the sliding-position average (which overestimates
because it uniformly samples positions, under-weighting post-rejection correction cycles).

Per prompt: start at drafts index 0; each cycle compute prefix=min(match(drafts[s],
target[s+2:s+2+K]), K); accept=prefix; advance s += accept+1. Aggregate cycle-POOLED
(the model currency: cost is per-cycle) E[a|K], S(K)=P(accept==K), and the spec_speedup_model
speedup; also per-prompt mean + clustered CI + per-source. (Codex-GATE1 correction: the
sliding E[a|4]=2.345 was an overclaim; cycle-jump = 2.202, speedup ~0.98x.)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "dspark_oracle"))
BLOCK = 5
# model cost constants (spec_speedup_model, optimized/anchor-reuse verifier)
DECODE_MS, DRAFT_MS, VERIFY_MS = 26.0, 10.0, 65.8


def prefix_k(draft, tgt, k):
    p = 0
    for i in range(k):
        if i < len(draft) and i < len(tgt) and int(draft[i]) == int(tgt[i]):
            p += 1
        else:
            break
    return p


def simulate_prompt(drafts, tt, K):
    """Return list of per-cycle accepts for one prompt under K-cycle-jump."""
    s = 0; max_step = len(drafts); acc = []
    while s < max_step:
        tgt = [int(x) for x in tt[s + 2:s + 2 + K]]
        if len(tgt) < K:
            break
        p = prefix_k(drafts[s], tgt, K); a = min(p, K); acc.append(a); s += a + 1
    return acc


def speedup_of(Ea, SK, K):
    cost = DRAFT_MS + VERIFY_MS + DECODE_MS * SK
    return (Ea + 1) * DECODE_MS / cost


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-prompt-dir", required=True)
    ap.add_argument("--shards", nargs="+", default=[str(HERE / "dspark_train" / "data" / "shards")])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260707)
    args = ap.parse_args()
    from stage2_capture_store import Stage2CaptureStore
    store = Stage2CaptureStore(args.shards)
    rows = [json.loads(f.read_text()) for f in sorted(Path(args.per_prompt_dir).glob("*.json"))]

    out = {"K_values": {}, "model_constants": {"decode_ms": DECODE_MS, "draft_ms": DRAFT_MS, "verify_ms_k4": VERIFY_MS,
            "note": "optimized/anchor-reuse verifier cost: draft+verify+decode*S(K); speedup=(E[a|K]+1)*decode/cost"}}
    for K in (2, 3, 4, 5):
        per_prompt_ea = []; all_accept = []; per_source = {}
        for c in rows:
            pid = c["prompt_id"]; src = pid.rsplit("_", 1)[0]
            acc = simulate_prompt(c["drafts"], store.target_tokens(pid), K)
            if not acc:
                continue
            per_prompt_ea.append(float(np.mean(acc)))
            all_accept.extend(acc)
            per_source.setdefault(src, []).extend(acc)
        ea_pooled = float(np.mean(all_accept)) if all_accept else 0.0
        SK = float(np.mean([1 if a == K else 0 for a in all_accept])) if all_accept else 0.0
        ea_prompt = float(np.mean(per_prompt_ea)) if per_prompt_ea else 0.0
        ea_arr = np.array(per_prompt_ea)
        rng = np.random.default_rng(args.seed)
        idx = np.arange(len(ea_arr))
        boot = np.array([ea_arr[rng.choice(idx, len(ea_arr), replace=True)].mean()
                         for _ in range(args.n_boot)]) if len(ea_arr) else np.array([0.0])
        ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        out["K_values"][f"K{K}"] = {
            "n_cycles": len(all_accept), "n_prompts": len(per_prompt_ea),
            "E_a_K_pooled_cycle_weighted": round(ea_pooled, 5),
            "E_a_K_per_prompt_mean": round(ea_prompt, 5),
            "S_K_pooled": round(SK, 5),
            "speedup_optimized_verifier": round(speedup_of(ea_pooled, SK, K), 5),
            "clustered_ci95_per_prompt_E_a_K": [round(ci[0], 5), round(ci[1], 5)],
            "per_source_pooled": {s: {"E_a_K": round(float(np.mean(v)), 5),
                                      "S_K": round(float(np.mean([1 if a == K else 0 for a in v])), 5),
                                      "speedup": round(speedup_of(float(np.mean(v)), float(np.mean([1 if a == K else 0 for a in v])), K), 5)}
                                  for s, v in per_source.items()},
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    # headline print
    k4 = out["K_values"]["K4"]
    print(f"K=4 cycle-jump (realistic): E[a|4]={k4['E_a_K_pooled_cycle_weighted']} S(4)={k4['S_K_pooled']} "
          f"speedup={k4['speedup_optimized_verifier']}x  (sliding was E~2.345/~+1.3% — OVERCLAIM)")
    print(f"  per-source speedup: " + " ".join(f"{s}={v['speedup']}" for s, v in k4['per_source_pooled'].items()))
    print(f"  break-even E[a|4]=2.203; K=4 cycle-jump CI95={k4['clustered_ci95_per_prompt_E_a_K']}")
    print(json.dumps(out["K_values"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
