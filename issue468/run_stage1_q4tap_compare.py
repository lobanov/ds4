#!/usr/bin/env python3
"""Stage 1 — Q4-tap vs IQ2XXS drafter-acceptance comparison (RESEARCH INSTRUMENTATION).

Reads the retained IQ2XXS baseline bundles and the Stage-1 Q4-tap
(`Layers37-42Q4KExperts`) bundles (both temp=0, seed=2, same corpus) and emits a
head-to-head acceptance comparison: per-prompt and aggregate avg-prefix, overall
match%, and the clean p=1 (first draft token) match rate, plus token-divergence
and hidden-state-diff checks that the Q4-tap captures are genuinely fresh.

Outputs compact json/csv to issue468/artifacts/exactness_small_bundles_q4tap/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE / "artifacts" / "exactness_small_bundles"
Q4 = HERE / "artifacts" / "exactness_small_bundles_q4tap"
OUT = Q4 / "comparison_vs_baseline"
PROMPTS = [
    "code_histogram", "code_sort_pairs", "code_topk", "grounded_archive",
    "grounded_observatory", "grounded_repair", "mixed_exactness_smoke",
    "synthesis_incident_json", "synthesis_ops_json", "synthesis_timeline_json",
]


def p1_match(summary: dict) -> tuple[float, int]:
    rows = summary["rows"]
    m = sum(1 for r in rows if r["draft"][0] == r["target"][0])
    return m / len(rows), len(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    per_prompt = []
    b_ap_tot = q_ap_tot = b_p1_tot = q_p1_tot = 0.0
    b_p1_n = q_p1_n = tok_diff_tot = tok_tot = 0
    mh_maxdiff = mh_meandiff = 0.0
    mh_identical = True
    for p in PROMPTS:
        b = json.loads((BASE / f"{p}__t0p0" / "oracle" / "acceptance_summary.json").read_text())
        q = json.loads((Q4 / f"{p}__t0p0" / "oracle" / "acceptance_summary.json").read_text())
        b_ap, q_ap = b["average_prefix"], q["average_prefix"]
        b_p1, bnr = p1_match(b)
        q_p1, qnr = p1_match(q)
        bt = json.loads((BASE / f"{p}__t0p0" / "target_selected_tokens.json").read_text())
        qt = json.loads((Q4 / f"{p}__t0p0" / "target_selected_tokens.json").read_text())
        td = sum(1 for a, c in zip(bt, qt) if a != c)
        bm = np.load(BASE / f"{p}__t0p0" / "oracle" / "oracle_inputs.npz")["main_hidden"]
        qm = np.load(Q4 / f"{p}__t0p0" / "oracle" / "oracle_inputs.npz")["main_hidden"]
        mh_maxdiff = max(mh_maxdiff, float(np.abs(bm - qm).max()))
        mh_meandiff += float(np.abs(bm - qm).mean())
        mh_identical = mh_identical and np.allclose(bm, qm)
        b_ap_tot += b_ap; q_ap_tot += q_ap
        b_p1_tot += b_p1 * bnr; q_p1_tot += q_p1 * qnr
        b_p1_n += bnr; q_p1_n += qnr
        tok_diff_tot += td; tok_tot += len(bt)
        per_prompt.append({
            "prompt": p, "base_avg_prefix": b_ap, "q4_avg_prefix": q_ap,
            "d_avg_prefix": round(q_ap - b_ap, 4),
            "base_p1_match": round(b_p1, 4), "q4_p1_match": round(q_p1, 4),
            "d_p1_match": round(q_p1 - b_p1, 4),
            "tokens_differ": td, "tokens_total": len(bt),
            "base_match_pct": b["match_pct"], "q4_match_pct": q["match_pct"],
        })
    n = len(PROMPTS)
    agg = {
        "n_prompts": n,
        "baseline_E_a_5block": round(b_ap_tot / n, 4),
        "q4tap_E_a_5block": round(q_ap_tot / n, 4),
        "d_E_a_5block": round((q_ap_tot - b_ap_tot) / n, 4),
        "baseline_p1_match_rate": round(b_p1_tot / b_p1_n, 4),
        "q4tap_p1_match_rate": round(q_p1_tot / q_p1_n, 4),
        "d_p1_match_rate": round((q_p1_tot / q_p1_n) - (b_p1_tot / b_p1_n), 4),
        "baseline_p1_n": b_p1_n, "q4tap_p1_n": q_p1_n,
        "tokens_differ_total": tok_diff_tot, "tokens_total": tok_tot,
        "main_hidden_max_abs_diff": round(mh_maxdiff, 4),
        "main_hidden_mean_abs_diff": round(mh_meandiff / n, 5),
        "main_hidden_identical_to_baseline": bool(mh_identical),
        "per_prompt_positives": sum(1 for r in per_prompt if r["d_avg_prefix"] > 0),
        "per_prompt_negatives": sum(1 for r in per_prompt if r["d_avg_prefix"] < 0),
        "per_prompt_flat": sum(1 for r in per_prompt if r["d_avg_prefix"] == 0),
    }
    out = {"aggregate": agg, "per_prompt": per_prompt}
    (OUT / "comparison.json").write_text(json.dumps(out, indent=2) + "\n")
    # csv
    import csv
    fields = list(per_prompt[0].keys())
    with (OUT / "comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(per_prompt)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
