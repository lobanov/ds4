#!/usr/bin/env python3
"""Rejection-sampling acceptance probe (issue468 milestone 2, option A).

Reads the drafter distribution sidecar written by
``measure_acceptance_bundle.py --emit-draft-dist`` (q = softmax(base_logits +
markov_bias), top-K per draft position) and the retained target top-K
(``target_topk.json``, p = target softmax top-128 per generation step), and
computes per draft position and per draft block:

  * greedy-argmax acceptance:  ``argmax(q) == argmax(p)``  (the dossier currency)
  * rejection-sampling acceptance for a draft *sampled* from q:  ``1 - TV(p, q)``
  * rejection-sampling acceptance for the *greedy* draft token x*:
    ``min(1, p(x*) / q(x*))``

Aggregates into expected accepted prefix E[a|K]:
  * E[a|K]_greedy  — from the consecutive argmax-match prefix (the dossier metric)
  * E[a|K]_rs_greedydraft — expected prefix under independent per-position
    min(1,p/q) acceptance (drafter still proposes its argmax)
  * E[a|K]_rs_sampled — expected prefix under independent per-position 1-TV
    acceptance (drafter samples from q)

TV is computed over the union of the retained p top-128 and q top-K supports
(both distributions are concentrated; retained mass is reported so the bound
quality is visible). Decisive question: does rejection sampling raise E[a|K]
over greedy-argmax? If yes, the option-A reframing is viable on the acceptance
axis (the verifier-exactness axis is a separate measurement).

Usage:
    .venv/bin/python dspark_oracle/measure_rejection_acceptance.py \
        --bundles artifacts/exactness_small_bundles_q4tap/*__t0p0 \
        [--json-out artifacts/rejection_acceptance/summary.json]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_target_topk(path: Path) -> list[dict]:
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        entries = raw.get("steps", raw.get("rows", []))
    else:
        entries = raw
    out = []
    for e in entries:
        sel = e.get("selected", {})
        sel_id = sel.get("id") if isinstance(sel, dict) else sel
        p = {}
        for lp in e.get("top_logprobs", []):
            tok = lp.get("token", lp)
            tid = tok.get("id") if isinstance(tok, dict) else tok
            lpv = lp.get("logprob")
            p[int(tid)] = math.exp(lpv) if lpv is not None else float(lp.get("prob", 0.0))
        out.append({
            "step": int(e.get("step", len(out))),
            "selected_id": int(sel_id) if sel_id is not None else None,
            "p": p,
            "p_mass": sum(p.values()),
            "p_top1_id": max(p, key=p.get) if p else None,
        })
    return out


def load_draft_dist(path: Path) -> list[dict]:
    return json.loads(path.read_text())["rows"]


def best_alignment_offset(draft_rows: list[dict], target: list[dict]) -> int:
    tidx = {t["step"]: t for t in target}
    best_off, best_score = 0, -1
    for off in range(-3, 4):
        score = 0
        for r in draft_rows:
            t = tidx.get(r["spine_step"] + off)
            if t and t["p_top1_id"] is not None and t["p_top1_id"] == r["target_id"]:
                score += 1
        if score > best_score:
            best_off, best_score = off, score
    return best_off


def tv_union(p: dict, q: dict) -> tuple[float, float, float]:
    """TV over union support, conservative tail (unknown mass treated as
    disjoint -> upper bound on TV -> lower bound on 1-TV acceptance)."""
    p_mass = sum(p.values())
    q_mass = sum(q.values())
    p_tail = 1.0 - p_mass
    q_tail = 1.0 - q_mass
    union = set(p) | set(q)
    diff_known = sum(abs(p.get(x, 0.0) - q.get(x, 0.0)) for x in union)
    tv = 0.5 * (diff_known + p_tail + q_tail)
    return tv, p_mass, q_mass


def measure_bundle(draft_rows: list[dict], target: list[dict]) -> dict:
    tidx = {t["step"]: t for t in target}
    off = best_alignment_offset(draft_rows, target)
    per_pos = {}
    blocks = {}
    n_unmatched = 0
    for r in draft_rows:
        t = tidx.get(r["spine_step"] + off)
        if t is None:
            n_unmatched += 1
            continue
        p = t["p"]
        q = dict(zip(r["q_topk_ids"], r["q_topk_probs"]))
        draft_id, target_id = r["draft_id"], r["target_id"]
        greedy_match = 1 if draft_id == target_id else 0
        p_argmax_match = 1 if t["p_top1_id"] == target_id else 0
        tv, p_mass, q_mass = tv_union(p, q)
        rs_sampled = max(0.0, 1.0 - tv)
        p_draft = p.get(draft_id, 0.0)
        q_draft = q.get(draft_id, r.get("q_argmax_prob", 0.0))
        rs_greedydraft = min(1.0, (p_draft / q_draft) if q_draft > 0 else 0.0)
        bp = r["block_pos"]
        d = per_pos.setdefault(bp, {"n": 0, "greedy": 0.0, "p_argmax_match": 0.0,
                                    "rs_sampled": 0.0, "rs_greedydraft": 0.0,
                                    "tv": 0.0, "p_mass": 0.0, "q_mass": 0.0})
        d["n"] += 1
        d["greedy"] += greedy_match
        d["p_argmax_match"] += p_argmax_match
        d["rs_sampled"] += rs_sampled
        d["rs_greedydraft"] += rs_greedydraft
        d["tv"] += tv
        d["p_mass"] += p_mass
        d["q_mass"] += q_mass
        blocks.setdefault(r["anchor_step"], {})[bp] = {
            "greedy_match": greedy_match, "rs_greedydraft": rs_greedydraft,
            "rs_sampled": rs_sampled}

    table = {}
    for bp, d in sorted(per_pos.items()):
        n = d["n"] or 1
        table[bp] = {
            "n": d["n"],
            "greedy_match_rate": d["greedy"] / n,
            "p_argmax_match_rate": d["p_argmax_match"] / n,
            "rs_accept_sampled": d["rs_sampled"] / n,
            "rs_accept_greedydraft": d["rs_greedydraft"] / n,
            "mean_tv": d["tv"] / n,
            "mean_p_mass_retained": d["p_mass"] / n,
            "mean_q_mass_retained": d["q_mass"] / n,
        }

    BLOCK = max((max(b.keys()) for b in blocks.values()), default=-1) + 1
    greasy_prefix_sum = rs_grdft_prefix_sum = rs_samp_prefix_sum = 0.0
    n_blocks = 0
    greasy_hist = {k: 0 for k in range(BLOCK + 1)}
    for blk in blocks.values():
        n_blocks += 1
        gp = 0
        for i in range(BLOCK):
            if blk.get(i, {}).get("greedy_match"):
                gp = i + 1
            else:
                break
        greasy_hist[gp] += 1
        greasy_prefix_sum += gp
        for key, accum in (("rs_greedydraft", "rs_grdft"), ("rs_sampled", "rs_samp")):
            acc = 1.0
            ep = 0.0
            for i in range(BLOCK):
                acc *= blk.get(i, {}).get(key, 0.0)
                ep += acc
            if accum == "rs_grdft":
                rs_grdft_prefix_sum += ep
            else:
                rs_samp_prefix_sum += ep

    n_blocks = n_blocks or 1
    return {"alignment_offset": off, "n_unmatched": n_unmatched, "n_blocks": n_blocks,
            "block": BLOCK,
            "E_a_greasy": greasy_prefix_sum / n_blocks,
            "E_a_rs_greedydraft": rs_grdft_prefix_sum / n_blocks,
            "E_a_rs_sampled": rs_samp_prefix_sum / n_blocks,
            "greasy_prefix_hist": greasy_hist,
            "n_positions": sum(d["n"] for d in per_pos.values()),
            "per_block_pos": table}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", nargs="+", required=True)
    ap.add_argument("--json-out")
    args = ap.parse_args()

    all_tables = {}
    grand_pos = {}
    # pooled prefix sums (weighted by n_blocks)
    g = {"n_blocks": 0, "greasy": 0.0, "rs_grdft": 0.0, "rs_samp": 0.0}
    for b in args.bundles:
        bd = Path(b)
        dd, tt = bd / "draft_dist.json", bd / "target_topk.json"
        if not dd.exists():
            print(f"[skip] {bd.name}: no draft_dist.json")
            continue
        if not tt.exists():
            print(f"[skip] {bd.name}: no target_topk.json")
            continue
        res = measure_bundle(load_draft_dist(dd), load_target_topk(tt))
        res["bundle"] = bd.name
        all_tables[bd.name] = res
        nb = res["n_blocks"] or 1
        g["n_blocks"] += res["n_blocks"]
        g["greasy"] += res["E_a_greasy"] * nb
        g["rs_grdft"] += res["E_a_rs_greedydraft"] * nb
        g["rs_samp"] += res["E_a_rs_sampled"] * nb
        for bp, d in res["per_block_pos"].items():
            gp = grand_pos.setdefault(bp, {"n": 0, "greedy": 0.0, "rs_sampled": 0.0,
                                           "rs_greedydraft": 0.0, "tv": 0.0,
                                           "p_mass": 0.0, "q_mass": 0.0})
            gp["n"] += d["n"]
            for k in ("greedy", "rs_sampled", "rs_greedydraft", "tv", "p_mass", "q_mass"):
                gp[k] += d[{"greedy": "greedy_match_rate", "rs_sampled": "rs_accept_sampled",
                            "rs_greedydraft": "rs_accept_greedydraft", "tv": "mean_tv",
                            "p_mass": "mean_p_mass_retained",
                            "q_mass": "mean_q_mass_retained"}[k]] * d["n"]

    nb = g["n_blocks"] or 1
    pooled = {
        "n_blocks": g["n_blocks"],
        "E_a_greasy": g["greasy"] / nb,
        "E_a_rs_greedydraft": g["rs_grdft"] / nb,
        "E_a_rs_sampled": g["rs_samp"] / nb,
        "per_block_pos": {},
    }
    print("\n=== Rejection-sampling acceptance vs greedy-argmax (pooled) ===")
    print(f"{'bpos':>4} {'n':>5} {'greedy':>8} {'rs_samp':>8} {'rs_grdft':>9} {'meanTV':>8} {'p_mass':>7} {'q_mass':>7}")
    for bp in sorted(grand_pos):
        d = grand_pos[bp]
        n = d["n"] or 1
        row = {"n": d["n"], "greedy_match_rate": d["greedy"] / n,
               "rs_accept_sampled": d["rs_sampled"] / n,
               "rs_accept_greedydraft": d["rs_greedydraft"] / n,
               "mean_tv": d["tv"] / n, "mean_p_mass_retained": d["p_mass"] / n,
               "mean_q_mass_retained": d["q_mass"] / n}
        pooled["per_block_pos"][bp] = row
        print(f"{bp:>4} {row['n']:>5} {row['greedy_match_rate']:>8.3f} {row['rs_accept_sampled']:>8.3f} "
              f"{row['rs_accept_greedydraft']:>9.3f} {row['mean_tv']:>8.3f} "
              f"{row['mean_p_mass_retained']:>7.3f} {row['mean_q_mass_retained']:>7.3f}")
    print(f"\nE[a|K] greedy            = {pooled['E_a_greasy']:.4f}")
    print(f"E[a|K] rs (greedy draft) = {pooled['E_a_rs_greedydraft']:.4f}   "
          f"(Δ vs greedy {pooled['E_a_rs_greedydraft']-pooled['E_a_greasy']:+.4f})")
    print(f"E[a|K] rs (sampled)      = {pooled['E_a_rs_sampled']:.4f}   "
          f"(Δ vs greedy {pooled['E_a_rs_sampled']-pooled['E_a_greasy']:+.4f})")
    print("\nlegend: greedy=argmax(q)==argmax(p); rs_samp=1-TV; rs_grdft=min(1,p/q) on greedy draft; "
          "E[a|K] rs assumes independent per-position acceptance.")

    out = {"bundles": list(all_tables), "per_bundle": all_tables, "pooled": pooled}
    if args.json_out:
        op = Path(args.json_out)
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nwrote {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
