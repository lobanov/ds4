#!/usr/bin/env python3
"""Build an imatrix overlay weighted by recoverable baseline-vs-oracle gaps.

Inputs:
- an existing sweep root containing ``ctx_#####`` bundle directories
- one baseline ``*.b2.json`` label per bundle
- either:
  - one oracle ``*.b2.json`` label per bundle
  - or one top-level oracle-envelope ``*.details.json`` file

Output:
- a symlink overlay root mirroring the bundle tree
- one ``imatrix_anchor_weights.txt`` per bundle
- one manifest JSON describing which anchor steps were treated as recoverable

The weighting rule is deliberately simple:
1. a step must be baseline-hard enough to matter
2. the oracle must beat baseline by at least ``min_gap``
3. weight rises with the recoverable accepted-token gap

This operationalizes Step 1 from
``issue468/59_oracle_only_quantization_search_goal.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-root", required=True)
    ap.add_argument("--baseline-label", required=True,
                    help="bundle-local label prefix, e.g. baseline-weighted4ctx_19t_default_256tr")
    ap.add_argument("--oracle-label",
                    help="bundle-local label prefix for the oracle-side scorer")
    ap.add_argument("--oracle-details-json",
                    help="top-level oracle-envelope details JSON with per-context per-step oracle values")
    ap.add_argument("--out-label", default="recoverable-gap")
    ap.add_argument("--steps-cap", type=int, default=0,
                    help="if >0, limit weighting to the first N anchor steps")
    ap.add_argument("--baseline-max", type=float, default=4.999,
                    help="steps with baseline accepted >= this are treated as already solved")
    ap.add_argument("--min-gap", type=float, default=0.05,
                    help="minimum oracle-baseline accepted gap to count as recoverable")
    ap.add_argument("--alpha", type=float, default=2.0,
                    help="multiplier applied to normalized recoverable gap")
    ap.add_argument("--floor", type=float, default=0.5)
    ap.add_argument("--ceil", type=float, default=2.0)
    return ap.parse_args()


def bundle_dirs(root: Path) -> list[Path]:
    dirs = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if not re.fullmatch(r"ctx_\d{5}", child.name):
            continue
        if (child / "target_topk.json").exists() and (child / "target_greedy.json").exists():
            dirs.append(child)
    return sorted(dirs)


def clamp(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


def read_b2(bundle: Path, label: str) -> dict:
    path = bundle / f"{label}.b2.json"
    if not path.exists():
        raise RuntimeError(f"missing {path}")
    return json.loads(path.read_text())


def load_oracle_details(path: Path) -> dict[int, list[float]]:
    details = json.loads(path.read_text())
    out: dict[int, list[float]] = {}
    for item in details:
        ctx = int(item["context"])
        out[ctx] = [float(v) for v in item["per_step_oracle_envelope"]]
    return out


def build_weights(per_step_base: list[float], per_step_oracle: list[float], *,
                  baseline_max: float, min_gap: float,
                  alpha: float, floor: float, ceil: float) -> tuple[list[float], list[dict]]:
    if len(per_step_base) != len(per_step_oracle):
        raise RuntimeError("baseline/oracle per-step lengths differ")

    raw = []
    details = []
    for idx, (base, oracle) in enumerate(zip(per_step_base, per_step_oracle), start=1):
        gap = oracle - base
        baseline_hard = base < baseline_max
        recoverable = baseline_hard and gap >= min_gap
        # Accepted lengths live on [0, 5]. Normalize the positive recoverable gap to [0, 1].
        gap_norm = clamp(gap / 5.0, 0.0, 1.0)
        value = 1.0 + alpha * gap_norm if recoverable else 1.0
        raw.append(value)
        details.append({
            "step": idx,
            "baseline_accepted": base,
            "oracle_accepted": oracle,
            "recoverable_gap": gap,
            "baseline_hard": baseline_hard,
            "recoverable": recoverable,
            "raw_weight": value,
        })

    mean_raw = sum(raw) / len(raw) if raw else 1.0
    weights = [clamp(v / mean_raw, floor, ceil) for v in raw]
    for item, weight in zip(details, weights):
        item["weight"] = weight
    return weights, details


def main() -> int:
    args = parse_args()
    if bool(args.oracle_label) == bool(args.oracle_details_json):
        raise RuntimeError("provide exactly one of --oracle-label or --oracle-details-json")
    sweep_root = Path(args.sweep_root).resolve()
    bundles = bundle_dirs(sweep_root)
    if not bundles:
        raise RuntimeError(f"no ctx_* bundles found under {sweep_root}")

    overlay_root = sweep_root / f".{args.out_label}.overlay"
    if overlay_root.exists():
        shutil.rmtree(overlay_root)
    overlay_root.mkdir(parents=True)

    oracle_details = (
        load_oracle_details(Path(args.oracle_details_json).resolve())
        if args.oracle_details_json else None
    )

    manifest = {
        "source": "recoverable-gap",
        "baseline_label": args.baseline_label,
        "oracle_label": args.oracle_label,
        "oracle_details_json": str(Path(args.oracle_details_json).resolve()) if args.oracle_details_json else None,
        "baseline_max": args.baseline_max,
        "min_gap": args.min_gap,
        "alpha": args.alpha,
        "floor": args.floor,
        "ceil": args.ceil,
        "bundles": {},
    }

    recoverable_steps = 0
    total_steps = 0
    mean_gap_num = 0.0
    mean_gap_den = 0

    for bundle in bundles:
        out_dir = overlay_root / bundle.name
        out_dir.mkdir()
        for child in bundle.iterdir():
            os.symlink(child, out_dir / child.name)

        base = read_b2(bundle, args.baseline_label)
        ctx = int(bundle.name.split("_")[1])
        oracle = read_b2(bundle, args.oracle_label) if args.oracle_label else None
        base_steps = [float(v) for v in base["per_step_accepted"]]
        oracle_steps = (
            [float(v) for v in oracle["per_step_accepted"]]
            if oracle is not None else oracle_details.get(ctx)
        )
        if oracle_steps is None:
            raise RuntimeError(f"missing oracle details for context {ctx}")
        if args.steps_cap > 0:
            base_steps = base_steps[:args.steps_cap]
            oracle_steps = oracle_steps[:args.steps_cap]

        weights, details = build_weights(
            base_steps,
            oracle_steps,
            baseline_max=args.baseline_max,
            min_gap=args.min_gap,
            alpha=args.alpha,
            floor=args.floor,
            ceil=args.ceil,
        )
        (out_dir / "imatrix_anchor_weights.txt").write_text(
            "\n".join(f"{w:.8f}" for w in weights) + "\n",
            encoding="utf-8",
        )

        bundle_recoverable = sum(1 for item in details if item["recoverable"])
        bundle_gap = [max(0.0, item["recoverable_gap"]) for item in details if item["recoverable"]]
        recoverable_steps += bundle_recoverable
        total_steps += len(details)
        mean_gap_num += sum(bundle_gap)
        mean_gap_den += len(bundle_gap)

        manifest["bundles"][bundle.name] = {
            "weights": weights,
            "mean": sum(weights) / len(weights) if weights else 1.0,
            "min": min(weights) if weights else 1.0,
            "max": max(weights) if weights else 1.0,
            "recoverable_steps": bundle_recoverable,
            "baseline_average_accepted": float(base["average_accepted"]),
            "oracle_average_accepted": (
                float(oracle["average_accepted"])
                if oracle is not None else (sum(oracle_steps) / len(oracle_steps) if oracle_steps else 0.0)
            ),
            "details": details,
        }

    manifest["summary"] = {
        "contexts": len(bundles),
        "total_steps": total_steps,
        "recoverable_steps": recoverable_steps,
        "recoverable_fraction": (
            recoverable_steps / total_steps if total_steps > 0 else 0.0
        ),
        "mean_positive_gap": (
            mean_gap_num / mean_gap_den if mean_gap_den > 0 else 0.0
        ),
        "overlay_root": str(overlay_root),
    }

    out_json = sweep_root / f"{args.out_label}.anchor_weights.json"
    out_json.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
