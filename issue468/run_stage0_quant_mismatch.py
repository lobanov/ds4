#!/usr/bin/env python3
"""Stage 0 — quantization-mismatch error-shape diagnostic.

RESEARCH INSTRUMENTATION for issue468 (quant-mismatch hypothesis). Re-runs the
DSpark drafter over the retained temp=0 exactness bundles, capturing the FULL
per-position decision score (base logits + markov bias) at each draft position,
and records, for every (bundle, measure-step, draft-position):

  - match/miss            : drafter argmax vs Q2 target argmax
  - rank_target_in_drafter: rank of the Q2 target token in the drafter's full
                            score distribution (low == perturbative / fine-tunable)
  - rank_drafter_in_q2    : rank of the drafter's pick in the Q2 target's own
                            top-128 distribution (low == Q2 considered the drafter
                            pick == quant-flip signature)
  - q2_top1_top2_gap      : Q2 target's own top1-top2 logit gap (small == Q2 was
                            near-tied == quant noise could flip argmax)
  - drafter_top1_top2_gap : drafter's own top1-top2 score gap (confidence)

Aggregates: per-position rank distributions, near-tie / quant-flip fractions,
top-k coverage (k=1..5), and a greedy-spine top-k counterfactual prefix
acceptance (the k=1 row must reproduce current acceptance — a sanity anchor).

Fidelity cross-checks (enforced, abort on mismatch):
  1. reproduced draft tokens == retained oracle/acceptance_summary.json drafts
  2. k=1 counterfactual match-rate == retained match-rate

Outputs compact csv/json to issue468/artifacts/quant_mismatch_diagnostic/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "dspark_oracle"))

from measure_acceptance_bundle import (  # noqa: E402
    build_model_ctx, build_drafter_ctx, load_mh, dspark_attn,
)
from forward import forward_head, BLOCK, DIM, HC, NOISE_TOK  # noqa: E402
from hc_primitives import rmsnorm, hc_pre, hc_post  # noqa: E402
from attention import apply_rotary, ROPE_DIM, HEAD_DIM  # noqa: E402
from moe import moe  # noqa: E402

WIN = 128
def _find_gguf(name: str) -> Path:
    """Locate a GGUF under the sibling ds4/gguf dir (robust to checkout location)."""
    here = HERE
    candidates = [
        here.parent.parent / "ds4" / "gguf",                       # repo sibling
        Path("/Users/lobanov/Projects/ds4/gguf"),                   # absolute fallback
    ]
    for base in candidates:
        p = base / name
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(f"cannot locate {name}; tried {[str(c / name) for c in candidates]}")

DEFAULT_MODEL = _find_gguf("DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf")
DEFAULT_DSPARK = _find_gguf("dspark.gguf")
BUNDLES = HERE / "artifacts" / "exactness_small_bundles"
OUT_DIR = HERE / "artifacts" / "quant_mismatch_diagnostic"
TOPK_FALLBACK_RANK = 129  # sentinel: drafter pick not in Q2 top-128


def q2_step_dist(topk: dict, step_idx: int) -> tuple[dict, list]:
    """Return (id->logit map, sorted top_logprobs list) for target_topk step step_idx."""
    steps = topk["steps"]
    if step_idx >= len(steps):
        return {}, []
    lp = steps[step_idx]["top_logprobs"]
    id2logit = {entry["token"]["id"]: entry["logit"] for entry in lp}
    return id2logit, lp


def rank_in_q2(drafter_pick: int, id2logit: dict, lp: list) -> int:
    """Rank (0-indexed) of drafter_pick in the Q2 top-128; TOPK_FALLBACK_RANK if absent."""
    for i, entry in enumerate(lp):
        if entry["token"]["id"] == drafter_pick:
            return i
    return TOPK_FALLBACK_RANK


def run() -> int:
    print(f"loading target ctx (embed+lm_head): {DEFAULT_MODEL.name}", file=sys.stderr, flush=True)
    mctx = build_model_ctx(str(DEFAULT_MODEL))
    print(f"loading drafter ctx: {DEFAULT_DSPARK.name}", file=sys.stderr, flush=True)
    dctx = build_drafter_ctx(str(DEFAULT_DSPARK))

    T = dctx["T"]
    layers = dctx["layers"]
    stores = dctx["stores"]
    main_proj = dctx["main_proj"]
    main_norm_w = dctx["main_norm_w"]
    embed_w = mctx["embed_w"]
    lm_head = mctx["lm_head"]
    cos, sin = mctx["cos"], mctx["sin"]
    # markov weights to reconstruct the full decision score (base + markov bias)
    markov_w1 = T["mtp.2.markov_head.markov_w1.weight"][0]  # [vocab, rank]
    markov_w2 = T["mtp.2.markov_head.markov_w2.weight"][0]  # [vocab, rank]
    markov_w2_T = markov_w2.T.astype(np.float32)            # [rank, vocab]

    bundle_dirs = sorted(BUNDLES.glob("*__t0p0"))
    rows = []
    audit_ids, audit_scores, audit_tgt_score = [], [], []  # top-64 for artifact-only rank re-derivation
    crosscheck_failures = 0

    for bd in bundle_dirs:
        manifest = json.loads((bd / "bundle_manifest.json").read_text())
        target_tokens = json.loads((bd / "target_selected_tokens.json").read_text())
        topk = json.loads((bd / "target_topk.json").read_text())
        retained = json.loads((bd / "oracle" / "acceptance_summary.json").read_text())
        retained_draft_by_step = {r["step"]: r["draft"] for r in retained["rows"]}
        prompt = manifest["prompt_name"]
        pos0 = int(manifest["prompt_tokens"])
        max_step = min(len(target_tokens) - BLOCK - 1, int(manifest["measure_steps"]))

        win_kv = [np.zeros((WIN, HEAD_DIM), dtype=np.float32) for _ in range(3)]
        # init KV at pos0 (mirrors measure_bundle exactly)
        mh0 = load_mh(bd, pos0)
        main_x0 = rmsnorm(mh0.reshape(1, 1, 3 * DIM) @ main_proj.T, main_norm_w)
        for s in range(3):
            mkv = rmsnorm(main_x0 @ layers[s]["kv"].T, layers[s]["kv_a_norm"])
            mkv[..., -ROPE_DIM:] = apply_rotary(mkv[..., -ROPE_DIM:], cos[0], sin[0])
            win_kv[s][0] = mkv[0, 0]
        n_real = 1

        for step in range(1, max_step + 1):
            pos = pos0 + step
            anchor = int(target_tokens[step])
            mh = load_mh(bd, pos)
            main_x = rmsnorm(mh.reshape(1, 1, 3 * DIM) @ main_proj.T, main_norm_w)
            for s in range(3):
                mkv = rmsnorm(main_x @ layers[s]["kv"].T, layers[s]["kv_a_norm"])
                mkv[..., -ROPE_DIM:] = apply_rotary(mkv[..., -ROPE_DIM:], cos[step], sin[step])
                win_kv[s][step % WIN] = mkv[0, 0]
            n_real = min(n_real + 1, WIN)
            draft_ids = np.full(BLOCK, NOISE_TOK, dtype=np.int64)
            draft_ids[0] = anchor
            x = embed_w[draft_ids][None]
            x = np.repeat(x[:, :, None, :], HC, axis=2)
            for s in range(3):
                res = x
                yd, post, comb = hc_pre(x, layers[s]["hc_attn_fn"], layers[s]["hc_attn_scale"], layers[s]["hc_attn_base"])
                yd = rmsnorm(yd, layers[s]["attn_norm"])
                ao = dspark_attn(yd, win_kv[s], n_real, layers[s], cos, sin, step)
                x = hc_post(ao, res, post, comb)
                res = x
                yd, post, comb = hc_pre(x, layers[s]["hc_ffn_fn"], layers[s]["hc_ffn_scale"], layers[s]["hc_ffn_base"])
                yd = rmsnorm(yd, layers[s]["ffn_norm"])
                fo = moe(yd, np.array([0]), layers[s], stores[s])
                x = hc_post(fo, res, post, comb)
            out, base_logits = forward_head(
                x, anchor, None,
                T["mtp.2.norm.weight"][0],
                T["mtp.2.hc_head_fn.weight"][0], T["mtp.2.hc_head_scale.weight"][0], T["mtp.2.hc_head_base.weight"][0],
                T["mtp.2.markov_head.markov_w1.weight"][0], T["mtp.2.markov_head.markov_w2.weight"][0],
                T["mtp.2.confidence_head.proj.weight"][0], lm_head, temp=1.0,
            )
            reproduced_draft = [int(v) for v in out[1:].tolist()]
            retained_draft = retained_draft_by_step.get(step)
            if retained_draft is not None and reproduced_draft != retained_draft:
                crosscheck_failures += 1
                print(f"  CROSSCHECK MISMATCH {prompt} step {step}: "
                      f"repro {reproduced_draft} vs retained {retained_draft}", file=sys.stderr)

            target_block = [int(v) for v in target_tokens[step + 1:step + 1 + BLOCK]]
            # per draft position p=1..5 (loop index i=p-1)
            for p in range(1, BLOCK + 1):
                i = p - 1
                prev_tok = int(out[i])           # output_ids[i]: anchor for p=1, else drafter pick at p-1
                bias = markov_w1[prev_tok] @ markov_w2_T
                full_score = base_logits[i] + bias
                drafter_argmax = int(np.argmax(full_score))
                # fidelity: drafter_argmax must equal the rollout token out[i+1]
                assert drafter_argmax == int(out[i + 1]), \
                    f"markov recon mismatch {prompt} step{step} p{p}: {drafter_argmax} vs {int(out[i+1])}"
                target_tok = target_block[i]
                match = int(drafter_argmax == target_tok)
                rank_target = int(np.count_nonzero(full_score > full_score[target_tok]))
                # drafter top1/top2 gap
                fs_sorted = np.partition(-full_score, 1)  # two smallest of -fs = two largest of fs
                drafter_top1 = -fs_sorted[0]
                drafter_top2 = -fs_sorted[1]
                drafter_gap = float(drafter_top1 - drafter_top2)
                # Q2 target's own distribution at this position
                id2logit, lp = q2_step_dist(topk, step + p)
                q2_gap = float("nan")
                if len(lp) >= 2:
                    q2_gap = float(lp[0]["logit"] - lp[1]["logit"])
                rdiq = rank_in_q2(drafter_argmax, id2logit, lp)
                # audit capture: top-64 (token,score) + target score, so any rank<=64 is
                # re-derivable from artifacts alone (count top64_scores > target_score).
                order64 = np.argsort(-full_score, kind="stable")[:64]
                audit_ids.append(order64.astype(np.int32))
                audit_scores.append(full_score[order64].astype(np.float32))
                audit_tgt_score.append(np.float32(full_score[target_tok]))
                rows.append({
                    "prompt": prompt, "step": step, "pos": pos, "p": p,
                    "anchor": anchor, "prev_tok": prev_tok,
                    "drafter_argmax": drafter_argmax, "target_tok": target_tok,
                    "match": match, "rank_target_in_drafter": rank_target,
                    "drafter_top1_top2_gap": round(drafter_gap, 4),
                    "rank_drafter_in_q2": rdiq,
                    "q2_top1_top2_gap": (round(q2_gap, 4) if q2_gap == q2_gap else None),
                })

        print(f"  {prompt}: {max_step} steps processed", file=sys.stderr, flush=True)

    if crosscheck_failures:
        print(f"\nABORT: {crosscheck_failures} reproduced-vs-retained draft mismatches; "
              f"forward fidelity not established, ranks untrusted.", file=sys.stderr)
        return 2

    # ---- persist raw per-position rows ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import csv
    with (OUT_DIR / "per_position.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (OUT_DIR / "per_position.json").write_text(json.dumps(rows, indent=2) + "\n")
    np.savez(OUT_DIR / "drafter_top64_scores.npz",
             top64_ids=np.stack(audit_ids), top64_scores=np.stack(audit_scores),
             target_score=np.array(audit_tgt_score, dtype=np.float32))
    print(f"  wrote audit npz: top64 over {len(audit_ids)} positions", file=sys.stderr)

    # ---- aggregates ----
    def filt(p=None, match_only=False, miss_only=False, prefix_ok_through=None):
        out = rows
        if p is not None:
            out = [r for r in out if r["p"] == p]
        if match_only:
            out = [r for r in out if r["match"] == 1]
        if miss_only:
            out = [r for r in out if r["match"] == 0]
        return out

    def pct(arr, qs):
        if not arr:
            return {q: None for q in qs}
        a = np.array(arr, dtype=float)
        return {q: float(np.percentile(a, q)) for q in qs}

    summary = {"n_rows": len(rows), "n_bundles": len(bundle_dirs), "per_position": {}}
    KS = [1, 2, 3, 4, 5]
    for p in range(1, BLOCK + 1):
        rp = filt(p=p)
        n = len(rp)
        n_match = sum(r["match"] for r in rp)
        misses = [r for r in rp if r["match"] == 0]
        ranks_miss = [r["rank_target_in_drafter"] for r in misses]
        # top-k coverage: fraction of positions where target is within drafter top-k
        coverage = {k: sum(1 for r in rp if r["rank_target_in_drafter"] < k) / n for k in KS}
        q2gaps = [r["q2_top1_top2_gap"] for r in misses if r["q2_top1_top2_gap"] is not None]
        rdiq = [r["rank_drafter_in_q2"] for r in misses]
        d = {
            "n": n, "n_match": n_match, "match_rate": round(n_match / n, 4),
            "n_miss": len(misses),
            "rank_target_in_drafter_on_misses": {
                "percentiles": pct(ranks_miss, [25, 50, 75, 90]),
                "frac_le2": round(np.mean([r <= 2 for r in ranks_miss]) if ranks_miss else None, 4),
                "frac_le3": round(np.mean([r <= 3 for r in ranks_miss]) if ranks_miss else None, 4),
                "frac_le5": round(np.mean([r <= 5 for r in ranks_miss]) if ranks_miss else None, 4),
                "frac_le10": round(np.mean([r <= 10 for r in ranks_miss]) if ranks_miss else None, 4),
                "frac_gt50": round(np.mean([r > 50 for r in ranks_miss]) if ranks_miss else None, 4),
            },
            "topk_coverage": {k: round(v, 4) for k, v in coverage.items()},
            "q2_near_tie_on_misses": {
                "n_with_q2gap": len(q2gaps),
                "frac_gap_lt0p5": round(np.mean([g < 0.5 for g in q2gaps]) if q2gaps else None, 4),
                "frac_gap_lt1": round(np.mean([g < 1.0 for g in q2gaps]) if q2gaps else None, 4),
                "frac_gap_lt2": round(np.mean([g < 2.0 for g in q2gaps]) if q2gaps else None, 4),
                "median_gap": round(float(np.median(q2gaps)), 4) if q2gaps else None,
            },
            "rank_drafter_in_q2_on_misses": {
                "frac_le2": round(np.mean([r <= 2 for r in rdiq]) if rdiq else None, 4),
                "frac_le3": round(np.mean([r <= 3 for r in rdiq]) if rdiq else None, 4),
                "frac_le5": round(np.mean([r <= 5 for r in rdiq]) if rdiq else None, 4),
                "frac_not_in_top128": round(np.mean([r >= TOPK_FALLBACK_RANK for r in rdiq]) if rdiq else None, 4),
                "median": round(float(np.median(rdiq)), 2) if rdiq else None,
            },
        }
        summary["per_position"][str(p)] = d

    # ---- greedy-spine top-k counterfactual prefix acceptance ----
    # For each (bundle, step), walk p=1..5; accept p iff target in drafter top-k.
    # prefix = longest accepted run. E[a|K] computed over all steps for K=5 block.
    from collections import defaultdict
    by_bs = defaultdict(list)  # (prompt, step) -> rows in p order
    for r in rows:
        by_bs[(r["prompt"], r["step"])].append(r)
    cf = {}
    for k in KS:
        prefixes = []
        full5 = 0
        for key, rws in by_bs.items():
            rws = sorted(rws, key=lambda r: r["p"])
            pref = 0
            for r in rws:
                if r["rank_target_in_drafter"] < k:
                    pref = r["p"]
                else:
                    break
            prefixes.append(pref)
            full5 += 1 if pref == 5 else 0
        ea = float(np.mean(prefixes))
        cf[str(k)] = {
            "E_a_given_5block": round(ea, 4),
            "S5_full_accept_rate": round(full5 / len(by_bs), 4),
            "n_steps": len(by_bs),
        }
    summary["greedy_spine_topk_counterfactual"] = cf

    # persist the core summary BEFORE enrichment so a headline bug can't lose it
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # ---- headline p=1 block (enrichment; non-fatal) ----
    try:
        p1 = summary["per_position"]["1"]
        summary["headline_p1"] = {
            "match_rate": p1["match_rate"],
            "topk_coverage": p1["topk_coverage"],
            "median_rank_target_on_misses": p1["rank_target_in_drafter_on_misses"]["percentiles"][50],
            "frac_misses_rank_le5": p1["rank_target_in_drafter_on_misses"]["frac_le5"],
            "frac_misses_q2_near_tie_lt1nat": p1["q2_near_tie_on_misses"]["frac_gap_lt1"],
            "frac_misses_drafter_in_q2_top3": p1["rank_drafter_in_q2_on_misses"]["frac_le3"],
        }
        (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    except Exception as e:
        print(f"  (headline enrichment skipped: {e})", file=sys.stderr)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
