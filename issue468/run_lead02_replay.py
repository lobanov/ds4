#!/usr/bin/env python3
"""Lead 02 — adaptive confidence-scheduled replay under local verifier accountings.

Consumes measured per-prompt confidence rows plus optional STS temperatures and
replays speculative decode on the realistic cycle-jump trajectory
(`next_step += accepted + 1`) with an adaptive verification length `l` chosen
per cycle from the confidence scores.

Policies included:
- fixed-K baselines
- expected-speedup argmax over l in {0..5}
- correction-harvest heuristic l = floor(E[a]) + 1, with skip fallback
- threshold sweep on cumulative survival probabilities
- per-cycle oracle upper bound (chooses l from the realized prefix)

Outputs pooled speedup, prompt-clustered bootstrap CI, P(speed < 1), per-source
breakouts, and policy diagnostics under both:
- shipped verifier accounting
- optimized / anchor-reuse verifier accounting
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

BLOCK = 5
DECODE_MS = 26.0
DRAFT_MS = 10.0
VERIFY_MS = {1: 26.0, 2: 43.6, 3: 59.7, 4: 65.8, 5: 74.5, 6: 79.6}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--conf-dir",
        default="issue468/artifacts/lead02_confidence_measure_eval",
        help="directory containing per_prompt/*.json from run_lead02_measure_confidence.py",
    )
    ap.add_argument(
        "--sts-file",
        default="issue468/artifacts/lead02_confidence_sts_eval/temperatures.json",
        help="temperatures.json from run_lead02_sts.py; omit or point elsewhere to change calibration",
    )
    ap.add_argument(
        "--out",
        default="issue468/artifacts/lead02_confidence_replay_eval/summary.json",
        help="output summary path",
    )
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260708)
    ap.add_argument("--threshold-min", type=float, default=0.0)
    ap.add_argument("--threshold-max", type=float, default=1.0)
    ap.add_argument("--threshold-step", type=float, default=0.01)
    ap.add_argument(
        "--selected-threshold",
        type=float,
        default=None,
        help="if set, also evaluate this externally chosen threshold as an out-of-sample policy",
    )
    return ap.parse_args()


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def load_temperatures(path: Path) -> np.ndarray:
    obj = json.loads(path.read_text())
    temps = np.asarray(obj["temperatures"], dtype=np.float64)
    if temps.shape != (BLOCK,):
        raise SystemExit(f"expected {BLOCK} temperatures in {path}, got shape {temps.shape}")
    return temps


def load_prompt_rows(conf_dir: Path, temperatures: np.ndarray) -> list[dict]:
    rows = []
    for p in sorted((conf_dir / "per_prompt").glob("*.json")):
        obj = json.loads(p.read_text())
        flat = []
        for row in obj["rows"]:
            raw_scores = np.asarray(row["confidence_scores"], dtype=np.float64)
            logits = np.asarray(row["confidence_logits"], dtype=np.float64)
            sts_scores = sigmoid(logits / temperatures)
            flat.append(
                {
                    "step": int(row["step"]),
                    "prefix": int(row["prefix"]),
                    "raw_scores": raw_scores,
                    "raw_cum": np.cumprod(raw_scores),
                    "sts_scores": sts_scores,
                    "sts_cum": np.cumprod(sts_scores),
                }
            )
        rows.append(
            {
                "prompt_id": obj["prompt_id"],
                "source": obj.get("source"),
                "split": obj.get("split"),
                "rows": flat,
            }
        )
    if not rows:
        raise SystemExit(f"no per_prompt rows found under {conf_dir}")
    return rows


def expected_speedup(cum_probs: np.ndarray, l: int, accounting: str) -> float:
    if l == 0:
        cost = DECODE_MS + DRAFT_MS
        emitted = 1.0
        return emitted * DECODE_MS / cost
    e_accept = float(np.sum(cum_probs[:l]))
    if accounting == "shipped":
        cost = DECODE_MS + DRAFT_MS + VERIFY_MS[l]
    elif accounting == "anchor_reuse":
        cost = DRAFT_MS + VERIFY_MS[l] + DECODE_MS * float(cum_probs[l - 1])
    else:
        raise ValueError(accounting)
    emitted = 1.0 + e_accept
    return emitted * DECODE_MS / cost


def realized_cycle(prefix: int, l: int, accounting: str) -> dict:
    accepted = min(prefix, l)
    emitted = accepted + 1
    if l == 0:
        cost = DECODE_MS + DRAFT_MS
        full_accept = False
    elif accounting == "shipped":
        cost = DECODE_MS + DRAFT_MS + VERIFY_MS[l]
        full_accept = accepted == l
    elif accounting == "anchor_reuse":
        full_accept = accepted == l
        cost = DRAFT_MS + VERIFY_MS[l] + (DECODE_MS if full_accept else 0.0)
    else:
        raise ValueError(accounting)
    return {
        "accepted": int(accepted),
        "emitted": int(emitted),
        "cost_ms": float(cost),
        "full_accept": bool(full_accept),
    }


def pick_fixed(_cum_probs: np.ndarray, _accounting: str, _prefix: int, l: int) -> int:
    return int(l)


def pick_expected_opt(cum_probs: np.ndarray, accounting: str, _prefix: int) -> int:
    best_l = 0
    best_sp = expected_speedup(cum_probs, 0, accounting)
    for l in range(1, BLOCK + 1):
        sp = expected_speedup(cum_probs, l, accounting)
        if (sp > best_sp + 1e-12) or (abs(sp - best_sp) <= 1e-12 and l < best_l):
            best_l = l
            best_sp = sp
    return int(best_l)


def pick_harvest(cum_probs: np.ndarray, accounting: str, _prefix: int) -> int:
    e_total = float(np.sum(cum_probs))
    l = max(0, min(BLOCK, int(math.floor(e_total)) + 1))
    skip_sp = expected_speedup(cum_probs, 0, accounting)
    use_sp = expected_speedup(cum_probs, l, accounting)
    return 0 if use_sp <= skip_sp else int(l)


def pick_threshold(cum_probs: np.ndarray, _accounting: str, _prefix: int, threshold: float) -> int:
    l = 0
    for j in range(1, BLOCK + 1):
        if float(cum_probs[j - 1]) >= threshold:
            l = j
        else:
            break
    return int(l)


def pick_oracle(_cum_probs: np.ndarray, accounting: str, prefix: int) -> int:
    best_l = 0
    best_sp = realized_cycle(prefix, 0, accounting)["emitted"] * DECODE_MS / realized_cycle(prefix, 0, accounting)["cost_ms"]
    for l in range(1, BLOCK + 1):
        c = realized_cycle(prefix, l, accounting)
        sp = c["emitted"] * DECODE_MS / c["cost_ms"]
        if (sp > best_sp + 1e-12) or (abs(sp - best_sp) <= 1e-12 and l < best_l):
            best_l = l
            best_sp = sp
    return int(best_l)


def simulate_prompt(prompt: dict, prob_key: str, accounting: str, picker, picker_arg=None) -> dict:
    rows = prompt["rows"]
    s = 0
    total_tokens = 0
    total_cost = 0.0
    accepts = []
    scheduled = []
    full_accepts = 0
    skipped = 0
    cycles = []
    while s < len(rows):
        row = rows[s]
        cum_probs = row[prob_key]
        prefix = int(row["prefix"])
        if picker_arg is None:
            l = picker(cum_probs, accounting, prefix)
        else:
            l = picker(cum_probs, accounting, prefix, picker_arg)
        cycle = realized_cycle(prefix, l, accounting)
        total_tokens += cycle["emitted"]
        total_cost += cycle["cost_ms"]
        accepts.append(cycle["accepted"])
        scheduled.append(l)
        full_accepts += int(cycle["full_accept"])
        skipped += int(l == 0)
        cycles.append({"start_index": int(s), "l": int(l), "prefix": prefix, **cycle})
        s += cycle["emitted"]
    return {
        "prompt_id": prompt["prompt_id"],
        "source": prompt.get("source"),
        "split": prompt.get("split"),
        "n_cycles": int(len(cycles)),
        "tokens_total": int(total_tokens),
        "cost_total_ms": float(total_cost),
        "speedup": float(total_tokens * DECODE_MS / total_cost) if total_cost > 0 else 0.0,
        "accept_mean": float(np.mean(accepts)) if accepts else 0.0,
        "scheduled_mean": float(np.mean(scheduled)) if scheduled else 0.0,
        "full_accept_rate": float(full_accepts / len(cycles)) if cycles else 0.0,
        "skip_rate": float(skipped / len(cycles)) if cycles else 0.0,
        "cycles": cycles,
    }


def bootstrap_speedup(per_prompt: list[dict], n_boot: int, seed: int) -> tuple[list[float], float]:
    rng = np.random.default_rng(seed)
    n = len(per_prompt)
    idx = np.arange(n)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(idx, n, replace=True)
        tokens = sum(per_prompt[i]["tokens_total"] for i in pick)
        cost = sum(per_prompt[i]["cost_total_ms"] for i in pick)
        vals.append(tokens * DECODE_MS / cost)
    arr = np.asarray(vals, dtype=np.float64)
    ci = [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]
    p_lt_1 = float(np.mean(arr < 1.0))
    return ci, p_lt_1


def summarize(per_prompt: list[dict], n_boot: int, seed: int) -> dict:
    tokens = sum(x["tokens_total"] for x in per_prompt)
    cost = sum(x["cost_total_ms"] for x in per_prompt)
    n_cycles = sum(x["n_cycles"] for x in per_prompt)
    accepts = [c["accepted"] for p in per_prompt for c in p["cycles"]]
    scheduled = [c["l"] for p in per_prompt for c in p["cycles"]]
    full_accepts = [c["full_accept"] for p in per_prompt for c in p["cycles"]]
    ci, p_lt_1 = bootstrap_speedup(per_prompt, n_boot=n_boot, seed=seed)
    by_source = {}
    for source in sorted({x.get("source") or "?" for x in per_prompt}):
        group = [x for x in per_prompt if (x.get("source") or "?") == source]
        if not group:
            continue
        g_tokens = sum(x["tokens_total"] for x in group)
        g_cost = sum(x["cost_total_ms"] for x in group)
        g_accepts = [c["accepted"] for p in group for c in p["cycles"]]
        g_sched = [c["l"] for p in group for c in p["cycles"]]
        g_full = [c["full_accept"] for p in group for c in p["cycles"]]
        by_source[source] = {
            "n_prompts": int(len(group)),
            "n_cycles": int(sum(x["n_cycles"] for x in group)),
            "speedup": round(float(g_tokens * DECODE_MS / g_cost), 6),
            "ms_per_token": round(float(g_cost / g_tokens), 6),
            "accept_mean": round(float(np.mean(g_accepts)) if g_accepts else 0.0, 6),
            "scheduled_mean": round(float(np.mean(g_sched)) if g_sched else 0.0, 6),
            "full_accept_rate": round(float(np.mean(g_full)) if g_full else 0.0, 6),
        }
    return {
        "n_prompts": int(len(per_prompt)),
        "n_cycles": int(n_cycles),
        "tokens_total": int(tokens),
        "cost_total_ms": round(float(cost), 6),
        "speedup": round(float(tokens * DECODE_MS / cost), 6),
        "ms_per_token": round(float(cost / tokens), 6),
        "accept_mean": round(float(np.mean(accepts)) if accepts else 0.0, 6),
        "scheduled_mean": round(float(np.mean(scheduled)) if scheduled else 0.0, 6),
        "full_accept_rate": round(float(np.mean(full_accepts)) if full_accepts else 0.0, 6),
        "skip_rate": round(float(np.mean([1 if l == 0 else 0 for l in scheduled])) if scheduled else 0.0, 6),
        "clustered_ci95_speedup": [round(ci[0], 6), round(ci[1], 6)],
        "p_speed_lt_1": round(float(p_lt_1), 6),
        "by_source": by_source,
        "per_prompt": [
            {
                "prompt_id": x["prompt_id"],
                "source": x.get("source"),
                "split": x.get("split"),
                "n_cycles": x["n_cycles"],
                "tokens_total": x["tokens_total"],
                "cost_total_ms": round(float(x["cost_total_ms"]), 6),
                "speedup": round(float(x["speedup"]), 6),
                "accept_mean": round(float(x["accept_mean"]), 6),
                "scheduled_mean": round(float(x["scheduled_mean"]), 6),
                "full_accept_rate": round(float(x["full_accept_rate"]), 6),
                "skip_rate": round(float(x["skip_rate"]), 6),
            }
            for x in per_prompt
        ],
    }


def run_policy_suite(prompts: list[dict], prob_key: str, accounting: str, thresholds: np.ndarray, n_boot: int, seed: int) -> dict:
    out = {}

    for l in range(2, BLOCK + 1):
        per_prompt = [simulate_prompt(p, prob_key, accounting, pick_fixed, l) for p in prompts]
        out[f"fixed_k{l}"] = summarize(per_prompt, n_boot=n_boot, seed=seed + l)

    per_prompt = [simulate_prompt(p, prob_key, accounting, pick_expected_opt) for p in prompts]
    out["expected_opt"] = summarize(per_prompt, n_boot=n_boot, seed=seed + 31)

    per_prompt = [simulate_prompt(p, prob_key, accounting, pick_harvest) for p in prompts]
    out["harvest"] = summarize(per_prompt, n_boot=n_boot, seed=seed + 41)

    per_prompt = [simulate_prompt(p, prob_key, accounting, pick_oracle) for p in prompts]
    out["oracle"] = summarize(per_prompt, n_boot=n_boot, seed=seed + 51)

    sweep = []
    best = None
    best_summary = None
    for i, thr in enumerate(thresholds):
        per_prompt = [simulate_prompt(p, prob_key, accounting, pick_threshold, float(thr)) for p in prompts]
        summ = summarize(per_prompt, n_boot=n_boot, seed=seed + 101 + i)
        sweep.append(
            {
                "threshold": round(float(thr), 6),
                "speedup": summ["speedup"],
                "scheduled_mean": summ["scheduled_mean"],
                "accept_mean": summ["accept_mean"],
                "skip_rate": summ["skip_rate"],
            }
        )
        if best is None or summ["speedup"] > best_summary["speedup"]:
            best = float(thr)
            best_summary = summ
    out["best_threshold"] = {
        "threshold": round(float(best), 6),
        "summary": best_summary,
        "note": "Threshold chosen on the same evaluation slice; reported CI / P(speed<1) condition on that chosen threshold and are optimistic.",
    }
    out["threshold_sweep"] = sweep
    return out


def add_selected_threshold(
    suite: dict,
    prompts: list[dict],
    prob_key: str,
    accounting: str,
    threshold: float,
    n_boot: int,
    seed: int,
) -> None:
    per_prompt = [simulate_prompt(p, prob_key, accounting, pick_threshold, float(threshold)) for p in prompts]
    suite["selected_threshold"] = {
        "threshold": round(float(threshold), 6),
        "summary": summarize(per_prompt, n_boot=n_boot, seed=seed),
        "note": "Threshold supplied externally; this is the relevant out-of-sample threshold policy when selection happened on a different slice.",
    }


def main() -> int:
    args = parse_args()
    conf_dir = Path(args.conf_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temperatures = load_temperatures(Path(args.sts_file))
    prompts = load_prompt_rows(conf_dir, temperatures)

    thresholds = np.arange(args.threshold_min, args.threshold_max + 0.5 * args.threshold_step, args.threshold_step)
    threshold_list = np.clip(thresholds, 0.0, 1.0)

    summary = {
        "input": {
            "conf_dir": str(conf_dir),
            "sts_file": str(args.sts_file),
            "n_prompts": int(len(prompts)),
            "n_rows": int(sum(len(p["rows"]) for p in prompts)),
        },
        "constants": {
            "decode_ms": DECODE_MS,
            "draft_ms": DRAFT_MS,
            "verify_ms": VERIFY_MS,
            "skip_policy_note": "l=0 is charged decode+draft (confidence required the drafter to run; skip saves only verify).",
            "cycle_note": "realistic trajectory: next_step += accepted + 1",
            "selection_note": "Temperatures come from the supplied sts_file. STS-policy results are out-of-sample only if that sts_file and any selected threshold were fit on different data from this replay slice.",
        },
        "temperatures": [round(float(t), 6) for t in temperatures.tolist()],
        "policies": {
            "raw": {
                "shipped": run_policy_suite(prompts, "raw_cum", "shipped", threshold_list, n_boot=args.n_boot, seed=args.seed),
                "anchor_reuse": run_policy_suite(prompts, "raw_cum", "anchor_reuse", threshold_list, n_boot=args.n_boot, seed=args.seed + 1000),
            },
            "sts": {
                "shipped": run_policy_suite(prompts, "sts_cum", "shipped", threshold_list, n_boot=args.n_boot, seed=args.seed + 2000),
                "anchor_reuse": run_policy_suite(prompts, "sts_cum", "anchor_reuse", threshold_list, n_boot=args.n_boot, seed=args.seed + 3000),
            },
        },
    }

    if args.selected_threshold is not None:
        for calib, prob_key in (("raw", "raw_cum"), ("sts", "sts_cum")):
            for accounting in ("shipped", "anchor_reuse"):
                add_selected_threshold(
                    summary["policies"][calib][accounting],
                    prompts,
                    prob_key,
                    accounting,
                    args.selected_threshold,
                    n_boot=args.n_boot,
                    seed=args.seed + (5000 if accounting == "shipped" else 6000) + (0 if calib == "raw" else 500),
                )

    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
