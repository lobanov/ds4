#!/usr/bin/env python3
"""Speculative-decode speedup model for the DSpark drafter on the ds4 target.

Grounded entirely in retained empirical data:
  - verifier timings verify_ms(K)  -> mtp_verifier_bench_long (cross-prompt median)
  - greedy prefix acceptance       -> exactness_small_bundles (temp=0 prefix_hist)
  - decode_ms                      -> plain baseline (1000 / 38.5 t/s)
  - draft_ms = 10                  -> DSPark drafter assumption (given)

Cycle model (matches the measured MTP cycle total = decode + draft + verify):
    cycle_cost(K) = decode_ms + draft_ms + verify_ms(K)
    tokens emitted per cycle = 1 (base/anchor) + a ,  a = accepted draft prefix (0..K)
    speedup(K) = (1 + E[a|K]) * decode_ms / cycle_cost(K)

Answers:
  Q1  acceptance needed to beat baseline and to clear +20%, per K
  Q2  does K=4 give a speedup at the current drafter acceptance?
  Q3  can a 4-node draft tree (verifier checks a tree) give a speedup?
      which tree construction strategies are best?
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "issue468/artifacts/spec_speedup_model"
OUT.mkdir(parents=True, exist_ok=True)

# ---- empirical inputs --------------------------------------------------------
DECODE_MS = 26.0                         # 1000 / 38.49 baseline t/s (mtp_verifier_bench)
DRAFT_MS = 10.0                          # DSPark drafter, given
# verify_ms(K): cross-prompt MEDIAN of mtp_verifier_bench_long (code/synth/grounded 8k).
# K=1 not directly measured; a 1-token suffix verify ~= one target forward = decode.
VERIFY_MS = {1: 26.0, 2: 43.6, 3: 59.7, 4: 65.8, 5: 74.5, 6: 79.6}

# greedy prefix histogram over a 5-token block, temp=0 (exactness_small_bundles summary).
HIST5 = {0: 15, 1: 13, 2: 18, 3: 11, 4: 10, 5: 13}
_N = sum(HIST5.values())
P = {a: HIST5[a] / _N for a in range(6)}
# survival S[a] = P(prefix >= a);  E[prefix|K] = sum_{a=1..K} S[a]
S = {0: 1.0}
for a in range(1, 6):
    S[a] = S[a - 1] - P[a - 1]
assert abs(S[5] - P[5]) < 1e-9 and abs(sum(S[a] for a in range(1, 6)) - 2.3375) < 1e-6


# ---- core model --------------------------------------------------------------
def e_accept(K: int) -> float:
    return float(sum(S[a] for a in range(1, K + 1)))


# Corrected cycle accounting: the verify forward produces the correction/bonus
# token at the rejection point, which IS the next anchor. A fresh decode is needed
# ONLY on full-block acceptance. So:
#   cost(reject)      = draft + verify
#   cost(full accept) = draft + verify + decode
#   E[cost]  = draft + verify_ms(K) + decode * P(full accept)
#            = draft + verify_ms(K) + decode * S(K)        (S(K)=P(first K all match))
#   E[tokens]= E[a|K] + 1   (the +1 = correction bonus on reject, fresh decode on full)
# NOTE: the shipped ds4 --mtp pays a separate anchor decode EVERY cycle
# (decode+draft+verify), which is why measured MTP t/s is worse than this
# (optimized-verifier) projection. This model is for a correctly-built verifier.


def p_full_accept(K: int) -> float:
    return S[K]


def cycle_cost(K: int) -> float:
    return DRAFT_MS + VERIFY_MS[K] + DECODE_MS * p_full_accept(K)


def speedup(K: int, e: float | None = None) -> float:
    if e is None:
        e = e_accept(K)
    return (1.0 + e) * DECODE_MS / cycle_cost(K)


def speedup_p(K: int, p: float) -> float:
    """speedup under a uniform per-position match prob p (geometric acceptance)."""
    e = sum(p ** a for a in range(1, K + 1))
    cost = DRAFT_MS + VERIFY_MS[K] + DECODE_MS * (p ** K)
    return (1.0 + e) * DECODE_MS / cost


def p_for_target(K: int, target_speedup: float) -> float:
    """uniform per-position p (geometric) achieving target_speedup; -1 if unreachable."""
    if speedup_p(K, 1.0 - 1e-6) < target_speedup:
        return -1.0
    lo, hi = 1e-4, 1.0 - 1e-6
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if speedup_p(K, mid) < target_speedup:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def required_e(K: int, target_speedup: float) -> float:
    """E[accepted|K] needed to reach target_speedup."""
    return target_speedup * cycle_cost(K) / DECODE_MS - 1.0


def uniform_p_for_e(K: int, e_req: float) -> float:
    """Per-position match prob p (geometric model, sum_{a=1..K} p^a = e_req)."""
    if e_req <= 0:
        return 0.0
    lo, hi = 1e-4, 1.0 - 1e-6
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        val = sum(mid ** a for a in range(1, K + 1))
        if val < e_req:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def partA_linear():
    print("=" * 78)
    print("PART A — linear draft speedup model (current drafter acceptance, temp=0)")
    print("=" * 78)
    rows = []
    hdr = f"{'K':>2} {'verify_ms':>9} {'P(full)':>7} {'cycle_ms':>8} {'E[a|K]':>7} {'speedup':>8} {'vs base':>8}"
    print(hdr)
    for K in range(1, 6):
        e = e_accept(K)
        sp = speedup(K, e)
        rows.append({"K": K, "verify_ms": VERIFY_MS[K], "P_full_accept": round(p_full_accept(K), 3),
                     "cycle_ms": round(cycle_cost(K), 1), "E_accept": round(e, 3),
                     "speedup": round(sp, 3), "vs_baseline_pct": round((sp - 1) * 100, 1)})
        print(f"{K:>2} {VERIFY_MS[K]:>9.1f} {p_full_accept(K):>7.3f} {cycle_cost(K):>8.1f} {e:>7.3f} "
              f"{sp:>8.3f} {(sp-1)*100:>7.1f}%")
    print("\nQ2: K=4 -> speedup {:.3f} ({:+.1f}% vs baseline); "
          "E[a|4]={:.3f}, P(full accept)={:.3f}".format(
              speedup(4), (speedup(4)-1)*100, e_accept(4), p_full_accept(4)))
    return rows


def partB_required_acceptance():
    print("\n" + "=" * 78)
    print("PART B / Q1 — acceptance REQUIRED to beat baseline and to clear +20%")
    print("=" * 78)
    print(f"{'K':>2} {'verify_ms':>9} {'E beat':>7} {'E +20%':>7} "
          f"{'p beat':>7} {'p +20%':>7} {'cur E':>6} {'cur p':>6}")
    rows = []
    for K in range(2, 6):
        e_beat = required_e(K, 1.0)
        e_20 = required_e(K, 1.2)
        p_beat = p_for_target(K, 1.0)
        p_20 = p_for_target(K, 1.2)
        cur_e = e_accept(K)
        cur_p = uniform_p_for_e(K, cur_e)
        rows.append({"K": K, "E_req_beat": round(e_beat, 3), "E_req_20": round(e_20, 3),
                     "p_req_beat": round(p_beat, 3), "p_req_20": round(p_20, 3),
                     "cur_E": round(cur_e, 3), "cur_p": round(cur_p, 3)})
        print(f"{K:>2} {VERIFY_MS[K]:>9.1f} {e_beat:>7.3f} {e_20:>7.3f} "
              f"{p_beat:>7.3f} {p_20:>7.3f} {cur_e:>6.3f} {cur_p:>6.3f}")
    print("\nReading: 'E beat' = accepted drafts/cycle for speedup 1.0 (uses actual S(K) for")
    print("the full-accept decode); 'p' columns = uniform per-position match prob (geometric)");
    print("achieving the target. -1 = unreachable (ceiling below target).")
    return rows


def partC_trees():
    print("\n" + "=" * 78)
    print("PART C / Q3 — 4-node draft tree: ceiling analysis")
    print("=" * 78)
    print("A verify of N nodes costs verify_ms(N) regardless of tree shape (bandwidth floor).")
    print("A branched tree with N nodes has max committed depth d < N; a linear chain has d = N.\n")
    print(f"{'N nodes':>7} {'verify_ms':>9} {'cycle_ms':>8} "
          f"{'linear d=N ceil':>16} {'best tree d=N-1':>17}")
    rows = []
    for N in range(3, 7):
        cyc = DECODE_MS + DRAFT_MS + VERIFY_MS[N]
        # linear chain: depth N -> max emit 1+N
        ceil_lin = (1 + N) * DECODE_MS / cyc
        # best branched tree: depth N-1 (spine N-1 + 1 repair) -> max emit 1+(N-1) = N
        ceil_tree = N * DECODE_MS / cyc
        rows.append({"N": N, "verify_ms": VERIFY_MS[N], "cycle_ms": round(cyc, 1),
                     "ceiling_linear_pct": round((ceil_lin - 1) * 100, 1),
                     "ceiling_tree_pct": round((ceil_tree - 1) * 100, 1)})
        print(f"{N:>7} {VERIFY_MS[N]:>9.1f} {cyc:>8.1f} "
              f"{(ceil_lin-1)*100:>15.1f}% {(ceil_tree-1)*100:>16.1f}%")
    print("\nCeiling = max possible speedup at PERFECT acceptance (drafter always right).")

    print("\n--- Q3: focus on 4-node structures (corrected cost model) ---")
    N = 4
    cyc_full = DECODE_MS + DRAFT_MS + VERIFY_MS[N]   # cost at full-block acceptance
    e_lin4 = e_accept(4)
    sp_lin4_now = speedup(4)
    # depth-3 tree (spine3 + 1 repair at pos3): E = S1+S2+S2*q3, P(full)=S2*q3
    p3 = S[3] / S[2]                      # cond top-1 match at position 3
    def tree_d3(q3):
        e = S[1] + S[2] + S[2] * q3
        cost = DRAFT_MS + VERIFY_MS[N] + DECODE_MS * (S[2] * q3)
        return e, (e + 1) * DECODE_MS / cost
    e_no, sp_no = tree_d3(p3)             # repair gives nothing (top-1 only)
    e_per, sp_per = tree_d3(1.0)          # perfect repair branch
    print(f"4-node verify cost = {VERIFY_MS[N]:.1f} ms (full-accept cycle = {cyc_full:.1f} ms)")
    print(f"  linear depth-4 : ceiling {(1+4)*DECODE_MS/cyc_full-1:+.1%} | current E={e_lin4:.3f} "
          f"-> {sp_lin4_now:.3f} ({(sp_lin4_now-1)*100:+.1f}%)")
    print(f"  tree depth-3   : ceiling {4*DECODE_MS/cyc_full-1:+.1%} (PERFECT acceptance, max 3 drafts)")
    print(f"      at current top-1 (no repair lift): E={e_no:.3f} -> {sp_no:.3f} ({(sp_no-1)*100:+.1f}%)")
    print(f"      with a PERFECT repair branch      : E={e_per:.3f} -> {sp_per:.3f} ({(sp_per-1)*100:+.1f}%)")
    print(f"  tree depth-2 (spine2 + 2 leaves): ceiling {3*DECODE_MS/cyc_full-1:+.1%}")
    print("  -> a 4-node tree's CEILING is +2.2% (below the +20% gate and barely above")
    print("     baseline); even with a perfect repair branch it is worse than linear-4.")

    # hedging-vs-spine: marginal value of the 4th node as spine ext vs as a repair
    print("--- hedging is dominated by spine extension (current drafter) ---")
    # value of extending spine to position 4 = S[4] (marginal E from one more greedy position)
    # value of using the 4th node to hedge position 3 (top-1 -> top-2) ~ S[2]*(q3 - p3)
    p3 = S[3] / S[2]          # cond top-1 match at position 3
    for lift in (0.10, 0.20, 0.30):
        q3 = min(p3 + lift, 0.999)
        hedge_gain = S[2] * (q3 - p3)      # extra E from hedging pos 3
        print(f"  spine-ext pos4 adds E={S[4]:.3f} | hedging pos3 (top1->top2, +{lift:.0%} lift) "
              f"adds E={hedge_gain:.3f}  -> spine ext {'wins' if S[4] > hedge_gain else 'loses'}")
    return rows


def main():
    a = partA_linear()
    b = partB_required_acceptance()
    c = partC_trees()
    (OUT / "model_inputs.json").write_text(json.dumps({
        "decode_ms": DECODE_MS, "draft_ms": DRAFT_MS, "verify_ms": VERIFY_MS,
        "prefix_hist_temp0": HIST5, "survival_S": {str(k): v for k, v in S.items()},
        "note": "verify_ms = cross-prompt median of mtp_verifier_bench_long; "
                "prefix_hist from exactness_small_bundles temp=0; decode from 38.49 t/s baseline",
    }, indent=2) + "\n")
    (OUT / "summary.json").write_text(json.dumps(
        {"partA_linear": a, "partB_required_acceptance": b, "partC_trees": c}, indent=2) + "\n")
    print(f"\nwrote {OUT}/summary.json and model_inputs.json")


if __name__ == "__main__":
    main()
