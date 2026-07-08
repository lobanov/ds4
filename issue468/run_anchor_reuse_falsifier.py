#!/usr/bin/env python3
"""Lead 01 — anchor-reuse falsifier (RESEARCH INSTRUMENTATION).

Offline test of whether the DSpark drafter's acceptance survives drafting from
the one-position-stale target hidden (last-accepted-position) with the
correction token entering only as embedding -- the realizability test for
anchor-reuse, the load-bearing assumption of spec_speedup_model's optimistic edge.

Pipeline (all offline, reusing dspark_oracle/measure_acceptance_bundle.py):
  1. FIDELITY GATE: re-run BASELINE (reuse=False) over the 10 temp=0 cells and
     confirm draft tokens reproduce the retained acceptance_summary.json
     bit-for-bit (Stage-0 self-check; catches any harness/indexing regression
     before the reuse numbers are trusted).
  2. REUSE MEASUREMENT: run reuse=True over the same 10 cells.
  3. PAIRED STATS: per (prompt, step) unit, compare reuse vs baseline:
       - E[a|5block] delta (per-step accepted-prefix length), paired bootstrap CI.
       - p=1 first-token match, McNemar test.
  4. TWO-TIER VERDICT (per goal decision rule):
       SURVIVES  : reuse E[a|5block] delta >= 0.
       COLLAPSES : delta < 0 AND paired bootstrap CI excludes 0 (McNemar corroborates).
       MARGINAL  : delta < 0 but CI includes 0 -- report magnitude.

Outputs compact json/csv to issue468/artifacts/anchor_reuse_falsifier/.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "dspark_oracle"))

from measure_acceptance_bundle import build_model_ctx, build_drafter_ctx, measure_bundle  # noqa: E402

BASE = HERE / "artifacts" / "exactness_small_bundles"
OUT = HERE / "artifacts" / "anchor_reuse_falsifier"

DEFAULT_MODEL = "/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
DEFAULT_DSPARK = "/Users/lobanov/Projects/ds4/gguf/dspark.gguf"

PROMPTS = [
    "code_histogram", "code_sort_pairs", "code_topk", "grounded_archive",
    "grounded_observatory", "grounded_repair", "mixed_exactness_smoke",
    "synthesis_incident_json", "synthesis_ops_json", "synthesis_timeline_json",
]
BLOCK = 5
N_BOOT = 10000
RNG_SEED = 20260707


def mcnemar(b: int, c: int) -> tuple[float, float]:
    """Exact McNemar test on discordant pairs b (base+ reuse-), c (base- reuse+).
    Returns (chi2_with_cc, exact_two_sided_p)."""
    n = b + c
    if n == 0:
        return 0.0, 1.0
    # exact two-sided binomial p: 2 * min(P(X<=min), P(X>=min)) on Binom(n, 0.5)
    from math import comb
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    exact_p = min(1.0, 2.0 * tail)
    chi2 = (abs(b - c) - 1) ** 2 / n if n > 0 else 0.0  # with continuity correction
    return float(chi2), float(exact_p)


def compute_stats(units, n_boot, seed):
    """units: list of (prompt, step, base_prefix, reuse_prefix, base_p1, reuse_p1).
    Returns aggregate dict with per-step AND per-prompt-clustered bootstrap CIs on
    the E[a|5block] delta, plus McNemar on p=1, and the two-tier verdict."""
    n = len(units)
    base_pref = np.array([u[2] for u in units], dtype=np.float64)
    reuse_pref = np.array([u[3] for u in units], dtype=np.float64)
    deltas = reuse_pref - base_pref
    mean_delta = float(deltas.mean())
    base_ea5 = float(base_pref.mean())
    reuse_ea5 = float(reuse_pref.mean())

    rng = np.random.default_rng(seed)
    idx_all = np.arange(n)
    boot_means = np.empty(n_boot, dtype=np.float64)
    for b_i in range(n_boot):
        samp = rng.choice(idx_all, size=n, replace=True)
        boot_means[b_i] = deltas[samp].mean()
    ci_lo, ci_hi = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))
    p_os = float((boot_means >= 0).mean())
    boot_p_two = min(1.0, 2.0 * min(p_os, 1.0 - p_os))

    # per-PROMPT-clustered bootstrap (steps within a prompt are not independent)
    prompts = sorted(set(u[0] for u in units))
    per_prompt_delta = np.array([
        np.mean([u[3] - u[2] for u in units if u[0] == p]) for p in prompts
    ], dtype=np.float64)
    rng2 = np.random.default_rng(seed + 1)
    nprompts = len(prompts)
    bootp = np.array([per_prompt_delta[rng2.choice(nprompts, nprompts, replace=True)].mean()
                      for _ in range(n_boot)])
    ci_lo_c, ci_hi_c = float(np.percentile(bootp, 2.5)), float(np.percentile(bootp, 97.5))

    # McNemar on p=1
    b_disc = sum(1 for u in units if u[4] == 1 and u[5] == 0)
    c_disc = sum(1 for u in units if u[4] == 0 and u[5] == 1)
    chi2, mcn_p = mcnemar(b_disc, c_disc)
    base_p1_rate = float(np.mean([u[4] for u in units]))
    reuse_p1_rate = float(np.mean([u[5] for u in units]))

    significant = (ci_lo > 0) or (ci_hi < 0)
    if mean_delta >= 0:
        verdict = "SURVIVES"
    elif mean_delta < 0 and ci_hi < 0:
        verdict = "COLLAPSES"
    else:
        verdict = "MARGINAL"
    return {
        "n_paired_units": n,
        "base_E_a_5block": round(base_ea5, 4),
        "reuse_E_a_5block": round(reuse_ea5, 4),
        "d_E_a_5block": round(mean_delta, 4),
        "bootstrap_ci95_per_step": [round(ci_lo, 4), round(ci_hi, 4)],
        "bootstrap_p_two_sided_per_step": round(boot_p_two, 5),
        "ci_excludes_zero_per_step": bool(significant),
        "per_prompt_clustered_ci95": [round(ci_lo_c, 4), round(ci_hi_c, 4)],
        "n_prompts_clustered": nprompts,
        "base_p1_match_rate": round(base_p1_rate, 4),
        "reuse_p1_match_rate": round(reuse_p1_rate, 4),
        "d_p1_match_rate": round(reuse_p1_rate - base_p1_rate, 4),
        "mcnemar_discordant_b_basepos_reuseneg": b_disc,
        "mcnemar_discordant_c_baseneg_reusepos": c_disc,
        "mcnemar_chi2_cc": round(chi2, 4),
        "mcnemar_exact_p_two_sided": round(mcn_p, 6),
        "verdict": verdict,
        "verdict_rule": ("SURVIVES if d(E[a|5block])>=0; COLLAPSES if d<0 and "
                         "per-step bootstrap CI95 excludes 0; else MARGINAL"),
    }


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dspark", default=DEFAULT_DSPARK)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    ap.add_argument("--temps", nargs="+", default=["t0p0"],
                    help="bundle temp suffixes to run (t0p0/t0p5/t1p0)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print("loading target + drafter contexts (once)...", flush=True)
    mctx = build_model_ctx(args.model)
    dctx = build_drafter_ctx(args.dspark)

    REUSE_MODES = ["lag", "backfill"]
    temps_out = {}
    fidelity_all_ok_global = True
    for temp in args.temps:
        print(f"===== TEMP {temp} =====", flush=True)
        # ---- run BASELINE: fidelity gate + reference rows ----
        base_rows_by_prompt = {}
        fidelity_details = []
        fid_ok_temp = True
        for p in PROMPTS:
            bdir = BASE / f"{p}__{temp}"
            retained = json.loads((bdir / "oracle" / "acceptance_summary.json").read_text())
            fresh_base = measure_bundle(bdir, mctx, dctx, reuse_mode="none", candidate="baseline")
            base_rows_by_prompt[p] = fresh_base
            fid_ok = (len(fresh_base["rows"]) == len(retained["rows"]))
            mm = [r1["step"] for r1, r2 in zip(retained["rows"], fresh_base["rows"])
                  if r1["draft"] != r2["draft"]]
            fid_ok = fid_ok and not mm
            fid_ok_temp = fid_ok_temp and fid_ok
            fidelity_details.append({"prompt": p, "fidelity_ok": fid_ok,
                                     "n_steps": len(fresh_base["rows"]), "mismatched_steps": mm})
            if not fid_ok:
                print(f"  [FIDELITY FAIL] {temp}/{p}: {len(mm)} steps differ", flush=True)
        fidelity_all_ok_global = fidelity_all_ok_global and fid_ok_temp
        # ---- run each reuse mode, pair vs baseline ----
        modes_out = {}
        all_per_prompt = {}
        for mode in REUSE_MODES:
            units = []
            per_prompt = []
            for p in PROMPTS:
                bdir = BASE / f"{p}__{temp}"
                fresh_base = base_rows_by_prompt[p]
                reuse = measure_bundle(bdir, mctx, dctx, reuse_mode=mode, candidate=f"anchor_reuse_{mode}")
                base_prefixes, reuse_prefixes, base_p1s, reuse_p1s = [], [], [], []
                for rb, rr in zip(fresh_base["rows"], reuse["rows"]):
                    assert rb["step"] == rr["step"] and rb["position"] == rr["position"], \
                        f"step alignment broken for {p}"
                    bp, rp = rb["prefix"], rr["prefix"]
                    b_p1 = 1 if (rb["draft"][0] == rb["target"][0]) else 0
                    r_p1 = 1 if (rr["draft"][0] == rr["target"][0]) else 0
                    base_prefixes.append(bp); reuse_prefixes.append(rp)
                    base_p1s.append(b_p1); reuse_p1s.append(r_p1)
                    units.append((p, rb["step"], bp, rp, b_p1, r_p1))
                bap = float(np.mean(base_prefixes)); rap = float(np.mean(reuse_prefixes))
                per_prompt.append({
                    "prompt": p, "n_steps": len(base_prefixes),
                    "base_avg_prefix": round(bap, 4), "reuse_avg_prefix": round(rap, 4),
                    "d_avg_prefix": round(rap - bap, 4),
                    "base_p1_match": round(float(np.mean(base_p1s)), 4),
                    "reuse_p1_match": round(float(np.mean(reuse_p1s)), 4),
                    "d_p1_match": round(float(np.mean(reuse_p1s)) - float(np.mean(base_p1s)), 4),
                    "base_match_pct": fresh_base["match_pct"], "reuse_match_pct": reuse["match_pct"],
                })
            agg = compute_stats(units, args.n_boot, args.seed)
            agg["n_prompts"] = len(PROMPTS)
            agg["fidelity_gate_passed"] = bool(fid_ok_temp)
            agg["per_prompt_positives"] = sum(1 for r in per_prompt if r["d_avg_prefix"] > 0)
            agg["per_prompt_negatives"] = sum(1 for r in per_prompt if r["d_avg_prefix"] < 0)
            agg["per_prompt_flat"] = sum(1 for r in per_prompt if r["d_avg_prefix"] == 0)
            modes_out[mode] = {"aggregate": agg, "per_prompt": per_prompt,
                               "units": [{"prompt": u[0], "step": u[1], "base_prefix": u[2],
                                           "reuse_prefix": u[3], "base_p1": u[4], "reuse_p1": u[5]}
                                          for u in units]}
            all_per_prompt[mode] = per_prompt
            print(f"--- {temp} reuse_mode={mode} ---", flush=True)
            print(json.dumps(agg, indent=2), flush=True)
        temps_out[temp] = {"fidelity_details": fidelity_details,
                           "fidelity_gate_passed": bool(fid_ok_temp),
                           "reuse_modes": modes_out, "all_per_prompt": all_per_prompt}

    # ---- write outputs ----
    out = {
        "fidelity_gate_passed_all_temps": bool(fidelity_all_ok_global),
        "temps": {t: {"fidelity_gate_passed": temps_out[t]["fidelity_gate_passed"],
                      "reuse_modes": temps_out[t]["reuse_modes"]}
                  for t in temps_out},
    }
    (OUT / "falsifier_result.json").write_text(json.dumps(out, indent=2) + "\n")
    for temp in args.temps:
        lag_pp = temps_out[temp]["all_per_prompt"]["lag"]
        with (OUT / f"per_prompt_{temp}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(lag_pp[0].keys())); w.writeheader(); w.writerows(lag_pp)
        for mode in REUSE_MODES:
            with (OUT / f"units_{temp}_{mode}.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["prompt", "step", "base_prefix", "reuse_prefix", "base_p1", "reuse_p1"])
                for u in temps_out[temp]["reuse_modes"][mode]["units"]:
                    w.writerow([u["prompt"], u["step"], u["base_prefix"], u["reuse_prefix"], u["base_p1"], u["reuse_p1"]])
    # clean up old single-temp csv names if present
    for old in ["per_prompt.csv", "units_lag.csv", "units_backfill.csv"]:
        op = OUT / old
        if op.exists():
            op.unlink()

    print("=== FINAL AGGREGATES BY TEMP ===")
    print(json.dumps({t: {m: temps_out[t]["reuse_modes"][m]["aggregate"] for m in REUSE_MODES}
                       for t in args.temps}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
