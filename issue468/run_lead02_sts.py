#!/usr/bin/env python3
"""Lead 02 — sequential temperature scaling for DSpark confidence rows.

Reads measured per-prompt confidence rows, fits DSpark-style left-to-right
sequential temperature scaling (STS) by minimizing cumulative-prefix ECE at each
draft position, and emits calibrated aggregate diagnostics for replay work.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

BLOCK = 5


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-dir",
        dest="in_dir",
        default="issue468/artifacts/lead02_confidence_measure_eval",
        help="input directory containing per_prompt/*.json from run_lead02_measure_confidence.py",
    )
    ap.add_argument(
        "--out-dir",
        dest="out_dir",
        default="issue468/artifacts/lead02_confidence_sts_eval",
        help="output directory for temperatures and calibrated summaries",
    )
    ap.add_argument("--ece-bins", type=int, default=15)
    ap.add_argument("--grid-min", type=float, default=0.25)
    ap.add_argument("--grid-max", type=float, default=4.0)
    ap.add_argument("--grid-points", type=int, default=151)
    return ap.parse_args()


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def auc_binary(y_true, y_score) -> float:
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(y_score, dtype=np.float64)
    pos = int(y.sum())
    neg = int((1 - y).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    s_sorted = s[order]
    i = 0
    while i < len(s_sorted):
        j = i + 1
        while j < len(s_sorted) and s_sorted[j] == s_sorted[i]:
            j += 1
        if j - i > 1:
            avg = (i + 1 + j) / 2.0
            ranks[order[i:j]] = avg
        i = j
    rank_sum_pos = float(ranks[y == 1].sum())
    return (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def ece_binary(y_true, y_prob, n_bins=15) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_prob, dtype=np.float64)
    if len(y) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        conf = float(p[mask].mean())
        acc = float(y[mask].mean())
        ece += (n / total) * abs(conf - acc)
    return ece


def load_prompt_rows(in_dir: Path) -> list[dict]:
    ppdir = in_dir / "per_prompt"
    rows = [json.loads(p.read_text()) for p in sorted(ppdir.glob("*.json"))]
    if not rows:
        raise SystemExit(f"no per-prompt confidence rows found under {ppdir}")
    return rows


def flatten_rows(rows: list[dict]) -> list[dict]:
    out = []
    for item in rows:
        for row in item["rows"]:
            out.append(
                {
                    "prompt_id": item["prompt_id"],
                    "source": item.get("source"),
                    "split": item.get("split"),
                    "prefix": int(row["prefix"]),
                    "logits": np.asarray(row["confidence_logits"], dtype=np.float64),
                    "scores": np.asarray(row["confidence_scores"], dtype=np.float64),
                }
            )
    return out


def fit_sts(flat_rows: list[dict], n_bins: int, grid: np.ndarray) -> dict:
    prefix_prev = np.ones(len(flat_rows), dtype=np.float64)
    temperatures = []
    per_position = {}
    for k in range(1, BLOCK + 1):
        logits_k = np.asarray([row["logits"][k - 1] for row in flat_rows], dtype=np.float64)
        labels_k = np.asarray([1 if row["prefix"] >= k else 0 for row in flat_rows], dtype=np.float64)
        raw_probs = sigmoid(logits_k)
        raw_cum = prefix_prev * raw_probs
        raw_ece = ece_binary(labels_k, raw_cum, n_bins=n_bins)

        best_t = None
        best_ece = None
        best_cum = None
        for t in grid:
            cand_probs = sigmoid(logits_k / float(t))
            cand_cum = prefix_prev * cand_probs
            cand_ece = ece_binary(labels_k, cand_cum, n_bins=n_bins)
            key = (cand_ece, abs(math.log(float(t))))
            if best_ece is None or key < (best_ece, abs(math.log(float(best_t)))):
                best_t = float(t)
                best_ece = float(cand_ece)
                best_cum = cand_cum

        temperatures.append(best_t)
        per_position[f"p{k}"] = {
            "temperature": round(best_t, 6),
            "raw_cum_ece": round(float(raw_ece), 6),
            "sts_cum_ece": round(float(best_ece), 6),
            "ece_delta": round(float(best_ece - raw_ece), 6),
            "positive_rate": round(float(labels_k.mean()), 6),
        }
        prefix_prev = best_cum
    return {
        "temperatures": temperatures,
        "by_position": per_position,
    }


def apply_sts_probs(logits: np.ndarray, temperatures: list[float]) -> np.ndarray:
    temps = np.asarray(temperatures, dtype=np.float64)
    return sigmoid(logits / temps)


def aggregate_group(flat_rows: list[dict], n_bins: int, temperatures: list[float]) -> dict:
    by_position = {}
    for k in range(1, BLOCK + 1):
        labels = np.asarray([1 if row["prefix"] >= k else 0 for row in flat_rows], dtype=np.float64)
        logits = np.stack([row["logits"] for row in flat_rows], axis=0)
        raw_scores = np.stack([row["scores"] for row in flat_rows], axis=0)
        sts_scores = apply_sts_probs(logits, temperatures)
        raw_pos = raw_scores[:, k - 1]
        sts_pos = sts_scores[:, k - 1]
        raw_cum = np.cumprod(raw_scores[:, :k], axis=1)[:, -1]
        sts_cum = np.cumprod(sts_scores[:, :k], axis=1)[:, -1]
        by_position[f"p{k}"] = {
            "n": int(len(labels)),
            "positive_rate": round(float(labels.mean()), 6),
            "raw_auc": round(float(auc_binary(labels, raw_pos)), 6),
            "raw_ece": round(float(ece_binary(labels, raw_pos, n_bins=n_bins)), 6),
            "cum_auc": round(float(auc_binary(labels, raw_cum)), 6),
            "cum_ece": round(float(ece_binary(labels, raw_cum, n_bins=n_bins)), 6),
            "sts_raw_auc": round(float(auc_binary(labels, sts_pos)), 6),
            "sts_raw_ece": round(float(ece_binary(labels, sts_pos, n_bins=n_bins)), 6),
            "sts_cum_auc": round(float(auc_binary(labels, sts_cum)), 6),
            "sts_cum_ece": round(float(ece_binary(labels, sts_cum, n_bins=n_bins)), 6),
            "raw_mean_score": round(float(raw_pos.mean()), 6),
            "cum_mean_score": round(float(raw_cum.mean()), 6),
            "sts_raw_mean_score": round(float(sts_pos.mean()), 6),
            "sts_cum_mean_score": round(float(sts_cum.mean()), 6),
            "temperature": round(float(temperatures[k - 1]), 6),
        }
    return {
        "n_steps_total": int(len(flat_rows)),
        "by_position": by_position,
    }


def group_rows(flat_rows: list[dict], key: str) -> dict[str, list[dict]]:
    groups = {}
    for row in flat_rows:
        value = row.get(key) or "?"
        groups.setdefault(value, []).append(row)
    return groups


def main() -> int:
    args = parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_prompt_rows(in_dir)
    flat_rows = flatten_rows(rows)
    grid = np.exp(np.linspace(math.log(args.grid_min), math.log(args.grid_max), args.grid_points))

    fit = fit_sts(flat_rows, n_bins=args.ece_bins, grid=grid)
    temperatures = fit["temperatures"]

    overall = aggregate_group(flat_rows, n_bins=args.ece_bins, temperatures=temperatures)
    by_source = {
        source: aggregate_group(group, n_bins=args.ece_bins, temperatures=temperatures)
        for source, group in sorted(group_rows(flat_rows, "source").items())
    }
    by_split = {
        split: aggregate_group(group, n_bins=args.ece_bins, temperatures=temperatures)
        for split, group in sorted(group_rows(flat_rows, "split").items())
    }

    summary = {
        "input_dir": str(in_dir),
        "n_prompts": int(len(rows)),
        "n_steps_total": int(len(flat_rows)),
        "ece_bins": int(args.ece_bins),
        "grid": {
            "min": float(args.grid_min),
            "max": float(args.grid_max),
            "points": int(args.grid_points),
            "space": "log",
        },
        "fit": fit,
        "overall": overall,
        "by_source": by_source,
        "by_split": by_split,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "temperatures.json").write_text(
        json.dumps(
            {
                "temperatures": [round(float(t), 6) for t in temperatures],
                "note": "STS temperatures for draft positions 1..5; apply as sigmoid(logit / T_k).",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
