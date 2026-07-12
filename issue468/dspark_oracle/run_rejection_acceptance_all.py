#!/usr/bin/env python3
"""One-load driver: build the model/drafter context once, run
measure_acceptance_bundle over every q4tap bundle with --emit-draft-dist, then
run the rejection-sampling acceptance measurement. Avoids 10x model load.

Usage:
    .venv/bin/python dspark_oracle/run_rejection_acceptance_all.py \
        --bundles artifacts/exactness_small_bundles_q4tap/*__t0p0 \
        --model <target.gguf> --dspark <dspark.gguf> \
        --json-out artifacts/rejection_acceptance/summary.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from measure_acceptance_bundle import build_model_ctx, build_drafter_ctx, measure_bundle  # noqa: E402
from measure_rejection_acceptance import load_draft_dist, load_target_topk, measure_bundle as measure_rs  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", nargs="+", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dspark", required=True)
    ap.add_argument("--dist-topk", type=int, default=256)
    ap.add_argument("--json-out")
    args = ap.parse_args()

    print("loading model + drafter (once)...", flush=True)
    mctx = build_model_ctx(args.model)
    dctx = build_drafter_ctx(args.dspark)
    print("  ready.", flush=True)

    processed = []
    for b in args.bundles:
        bd = Path(b)
        tt = bd / "target_topk.json"
        if not tt.exists():
            print(f"[skip] {bd.name}: no target_topk.json")
            continue
        print(f"  drafting {bd.name} ...", flush=True)
        summary = measure_bundle(bd, mctx, dctx, emit_draft_dist=True, dist_topk=args.dist_topk)
        processed.append(bd)
        print(f"    avg_prefix(greedy)={summary['average_prefix']:.3f} "
              f"match%={summary['match_pct']:.1f}", flush=True)

    if not processed:
        print("no bundles processed")
        return 1
    print("\nrunning rejection-sampling measurement ...", flush=True)
    # reuse measure_rejection_acceptance by importing its main path inline
    import json
    all_tables, grand_pos = {}, {}
    g = {"n_blocks": 0, "greedy": 0.0, "rs_grdft": 0.0, "rs_samp": 0.0}
    for bd in processed:
        dd, tt = bd / "draft_dist.json", bd / "target_topk.json"
        res = measure_rs(load_draft_dist(dd), load_target_topk(tt))
        res["bundle"] = bd.name
        all_tables[bd.name] = res
        nb = res["n_blocks"] or 1
        g["n_blocks"] += res["n_blocks"]
        g["greedy"] += res["E_a_greasy"] * nb
        g["rs_grdft"] += res["E_a_rs_greedydraft"] * nb
        g["rs_samp"] += res["E_a_rs_sampled"] * nb
        for bp, d in res["per_block_pos"].items():
            gp = grand_pos.setdefault(bp, {"n": 0, "greedy": 0.0, "rs_sampled": 0.0,
                                           "rs_greedydraft": 0.0, "tv": 0.0,
                                           "p_mass": 0.0, "q_mass": 0.0})
            gp["n"] += d["n"]
            gp["greedy"] += d["greedy_match_rate"] * d["n"]
            gp["rs_sampled"] += d["rs_accept_sampled"] * d["n"]
            gp["rs_greedydraft"] += d["rs_accept_greedydraft"] * d["n"]
            gp["tv"] += d["mean_tv"] * d["n"]
            gp["p_mass"] += d["mean_p_mass_retained"] * d["n"]
            gp["q_mass"] += d["mean_q_mass_retained"] * d["n"]

    nb = g["n_blocks"] or 1
    print("\n=== pooled per-block-position ===")
    print(f"{'bpos':>4} {'n':>5} {'greedy':>8} {'rs_samp':>8} {'rs_grdft':>9} {'meanTV':>8} {'p_mass':>7} {'q_mass':>7}")
    pooled_pos = {}
    for bp in sorted(grand_pos):
        d = grand_pos[bp]
        n = d["n"] or 1
        row = {"n": d["n"], "greedy_match_rate": d["greedy"] / n,
               "rs_accept_sampled": d["rs_sampled"] / n,
               "rs_accept_greedydraft": d["rs_greedydraft"] / n,
               "mean_tv": d["tv"] / n, "mean_p_mass_retained": d["p_mass"] / n,
               "mean_q_mass_retained": d["q_mass"] / n}
        pooled_pos[bp] = row
        print(f"{bp:>4} {row['n']:>5} {row['greedy_match_rate']:>8.3f} {row['rs_accept_sampled']:>8.3f} "
              f"{row['rs_accept_greedydraft']:>9.3f} {row['mean_tv']:>8.3f} "
              f"{row['mean_p_mass_retained']:>7.3f} {row['mean_q_mass_retained']:>7.3f}")
    eg, erg, ers = g["greedy"] / nb, g["rs_grdft"] / nb, g["rs_samp"] / nb
    print(f"\nblocks={g['n_blocks']}  E[a|K] greedy={eg:.4f}  rs_greedydraft={erg:.4f} (Δ{erg-eg:+.4f})  rs_sampled={ers:.4f} (Δ{ers-eg:+.4f})")

    if args.json_out:
        op = Path(args.json_out)
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(json.dumps({
            "n_bundles": len(processed), "n_blocks": g["n_blocks"],
            "E_a_greasy": eg, "E_a_rs_greedydraft": erg, "E_a_rs_sampled": ers,
            "per_block_pos": pooled_pos, "per_bundle": all_tables,
        }, indent=2) + "\n")
        print(f"\nwrote {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
