#!/usr/bin/env python3
"""Live-vs-oracle drafter trace comparison (issue468 milestone 2, codex expt #4).

Aligns the LIVE DSpark runtime trace (anchor_id + draft_ids per cycle, from
ds4-spec-bench at a clean fixed K) to the OFFLINE numpy/torch oracle's drafts
(drafter.json rows: anchor/draft/target/match per spine step) on the SAME
prompts, and reports:

  * live vs oracle acceptance (p1, mean prefix) at matched anchors
  * draft-token AGREEMENT at matched anchors (does the live drafter produce the
    same drafts as the oracle?) -> separates "runtime drafter divergent
    (window-state pollution)" from "oracle overstates / target differs"

Usage:
    .venv/bin/python dspark_oracle/compare_live_vs_oracle.py \
        --live artifacts/rejection_acceptance/live_trace_vk5.jsonl \
        --bundles artifacts/exactness_small_bundles_q4tap/*__t0p0 \
        [--json-out artifacts/rejection_acceptance/live_vs_oracle.json]
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load_live(path: str) -> dict:
    """prompt_name -> list of (anchor_id, draft_ids[list], verified, drafted)."""
    out = {}
    for line in open(path):
        r = json.loads(line)
        pf = r.get("prompt_file") or r.get("chat_prompt_file") or r.get("id", "")
        name = Path(pf).stem if ("/" in pf or "\\" in pf) else pf
        cycs = []
        for c in r.get("dspark_cycles", []):
            if c.get("drafted", 0) == 0 and c.get("verify_n", 0) == 0:
                continue
            drafts = [d for d in c.get("draft_ids", []) if d is not None and d >= 0]
            cycs.append({"anchor": c.get("anchor_id", -1), "drafts": drafts,
                         "verified": c.get("verified", 0), "drafted": c.get("drafted", 0)})
        out[name] = cycs
    return out


def load_oracle(bundle_dir: Path) -> dict:
    """Reconstruct per-anchor oracle drafts/targets/prefix from draft_dist.json
    (written by measure_acceptance_bundle.py --emit-draft-dist). Each row is one
    (anchor_step, block_pos) with draft_id/target_id."""
    dd = json.loads((bundle_dir / "draft_dist.json").read_text())
    by_step = {}
    for r in dd["rows"]:
        s = by_step.setdefault(r["anchor_step"], {"step": r["anchor_step"],
                                                   "anchor": r["anchor_id"],
                                                   "drafts": [], "target": []})
        s["drafts"].append(r["draft_id"])
        s["target"].append(r["target_id"])
    rows = []
    for step in sorted(by_step):
        s = by_step[step]
        # prefix = consecutive draft==target from pos 0
        prefix = 0
        for d, t in zip(s["drafts"], s["target"]):
            if d == t:
                prefix += 1
            else:
                break
        s["prefix"] = prefix
        s["match"] = sum(1 for d, t in zip(s["drafts"], s["target"]) if d == t)
        rows.append(s)
    return {"rows": rows, "label": bundle_dir.name.replace("__t0p0", "")}


def align(live_cycs: list, oracle_rows: list) -> list:
    """Robust sequential anchor match. The live trace and the oracle share the same
    greedy spine (the oracle is built from the live's dumped hiddens), but there
    can be a small index offset (e.g. the live's first cycle anchor may be
    spine[1] while the dump records spine[0] first). Match each live cycle to the
    next oracle row (from a moving pointer) whose anchor token agrees; skip live
    cycles whose anchor is not found without exhausting the oracle."""
    pairs = []
    oi = 0
    for lc in live_cycs:
        found = -1
        for j in range(oi, len(oracle_rows)):
            if oracle_rows[j]["anchor"] == lc["anchor"]:
                found = j
                break
        if found < 0:
            continue
        pairs.append((lc, oracle_rows[found]))
        oi = found + 1
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", required=True)
    ap.add_argument("--bundles", nargs="+", required=True)
    ap.add_argument("--json-out")
    args = ap.parse_args()

    live = load_live(args.live)
    per_prompt = {}
    # accumulators
    g = {"pairs": 0, "live_p1": 0, "orc_p1": 0, "draft0_agree": 0,
         "live_prefix": 0.0, "orc_prefix": 0.0, "full_draft_agree": 0,
         "draft0_disagree_live_wrong": 0, "draft0_disagree_orc_wrong": 0,
         "draft0_disagree_both_wrong": 0}

    for b in args.bundles:
        bd = Path(b)
        name = bd.name.replace("__t0p0", "")
        if not (bd / "draft_dist.json").exists():
            continue
        orc = load_oracle(bd)
        lcycs = live.get(name, [])
        if not lcycs:
            continue
        pairs = align(lcycs, orc["rows"])
        pp = {"name": name, "n_pairs": len(pairs)}
        for lc, orc_row in pairs:
            g["pairs"] += 1
            l_d0 = lc["drafts"][0] if lc["drafts"] else -1
            o_d0 = orc_row["drafts"][0] if orc_row["drafts"] else -1
            tgt0 = orc_row["target"][0] if orc_row["target"] else -1
            live_p1 = 1 if l_d0 == tgt0 else 0
            orc_p1 = 1 if o_d0 == tgt0 else 0
            g["live_p1"] += live_p1
            g["orc_p1"] += orc_p1
            g["live_prefix"] += lc["verified"]
            g["orc_prefix"] += orc_row["prefix"]
            if l_d0 == o_d0:
                g["draft0_agree"] += 1
                # full block agreement (min length)
                k = min(len(lc["drafts"]), len(orc_row["drafts"]))
                if lc["drafts"][:k] == orc_row["drafts"][:k]:
                    g["full_draft_agree"] += 1
            else:
                # who is right?
                if live_p1 and not orc_p1:
                    g["draft0_disagree_live_wrong"] += 0  # live right
                elif orc_p1 and not live_p1:
                    g["draft0_disagree_live_wrong"] += 1
                else:
                    g["draft0_disagree_both_wrong"] += 1
        per_prompt[name] = pp

    n = g["pairs"] or 1
    res = {
        "n_pairs": g["pairs"],
        "live_p1": g["live_p1"] / n,
        "oracle_p1": g["orc_p1"] / n,
        "live_mean_verified": g["live_prefix"] / n,
        "oracle_mean_prefix": g["orc_prefix"] / n,
        "draft0_agreement": g["draft0_agree"] / n,
        "full_draft_agreement": g["full_draft_agree"] / n,
        "draft0_disagree_live_wrong": g["draft0_disagree_live_wrong"],
        "draft0_disagree_both_wrong": g["draft0_disagree_both_wrong"],
        "per_prompt": per_prompt,
    }
    print("=== live-vs-oracle drafter trace (matched anchors, same prompts) ===")
    print(f"matched pairs: {res['n_pairs']}")
    print(f"p1 (draft[0]==target):   live={res['live_p1']:.4f}  oracle={res['oracle_p1']:.4f}")
    print(f"mean accepted prefix:    live={res['live_mean_verified']:.4f}  oracle={res['oracle_mean_prefix']:.4f}")
    print(f"draft[0] AGREEMENT (live==oracle at matched anchor): {res['draft0_agreement']:.4f}")
    print(f"full-block draft AGREEMENT: {res['full_draft_agreement']:.4f}")
    print(f"when draft[0] disagrees: live-wrong={res['draft0_disagree_live_wrong']}, "
          f"both-wrong={res['draft0_disagree_both_wrong']} (of {res['n_pairs']})")
    print()
    print("Interpretation:")
    print(" - high draft AGREEMENT + similar p1 -> live drafter faithful; any gap is")
    print("   corpus/trajectory/target, not drafter divergence.")
    print(" - low draft AGREEMENT (live drafts differ from oracle) -> runtime drafter")
    print("   input diverges (window-state pollution / degraded hidden) -> fixable.")
    if args.json_out:
        op = Path(args.json_out)
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(json.dumps(res, indent=2) + "\n")
        print(f"\nwrote {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
