#!/usr/bin/env python3
"""Compare per-step complementarity across existing B2 candidate artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-root", required=True)
    ap.add_argument(
        "--bundle-label",
        action="append",
        default=[],
        metavar="NAME=FILESTEM",
        help="bundle-local B2 stem, e.g. layer2gateup=candidate-recoverablegap_boosted2ctx_layer2gateup",
    )
    ap.add_argument(
        "--external-b2",
        action="append",
        default=[],
        metavar="NAME=CTX:PATH",
        help="external B2 file for one context, e.g. ref_oracle=8192:/tmp/ref-oracle-fp8-ctx08192-19.b2.json",
    )
    ap.add_argument(
        "--recoverable-details-json",
        help="optional recoverable-gap manifest or details JSON for recoverable-step-only summaries",
    )
    ap.add_argument("--out-json", required=True)
    return ap.parse_args()


def ctx_dirs(root: Path) -> list[Path]:
    out = []
    for child in root.iterdir():
        if child.is_dir() and re.fullmatch(r"ctx_\d{5}", child.name):
            out.append(child)
    return sorted(out)


def parse_bundle_label(item: str) -> tuple[str, str]:
    if "=" not in item:
        raise RuntimeError(f"expected NAME=FILESTEM, got: {item}")
    name, stem = item.split("=", 1)
    return name, stem


def parse_external_b2(item: str) -> tuple[str, int, Path]:
    if "=" not in item or ":" not in item:
        raise RuntimeError(f"expected NAME=CTX:PATH, got: {item}")
    name, rest = item.split("=", 1)
    ctx_s, path_s = rest.split(":", 1)
    return name, int(ctx_s), Path(path_s).resolve()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_recoverable_steps(path: Path) -> dict[int, set[int]]:
    data = json.loads(path.read_text())
    out: dict[int, set[int]] = {}
    if isinstance(data, dict) and "bundles" in data:
        for bundle_name, item in data["bundles"].items():
            ctx = int(bundle_name.split("_")[1])
            out[ctx] = {int(d["step"]) for d in item["details"] if d.get("recoverable")}
        return out
    if isinstance(data, list):
        for item in data:
            ctx = int(item["context"])
            vals = item["per_step_oracle_envelope"]
            out[ctx] = set(range(1, len(vals) + 1))
        return out
    raise RuntimeError(f"unsupported recoverable details JSON format: {path}")


def main() -> int:
    args = parse_args()
    sweep_root = Path(args.sweep_root).resolve()
    bundles = ctx_dirs(sweep_root)
    if not bundles:
        raise RuntimeError(f"no ctx_* bundles found under {sweep_root}")

    bundle_specs = [parse_bundle_label(x) for x in args.bundle_label]
    external_specs = [parse_external_b2(x) for x in args.external_b2]
    recoverable = (
        load_recoverable_steps(Path(args.recoverable_details_json).resolve())
        if args.recoverable_details_json
        else {}
    )

    # candidate -> ctx -> payload
    candidates: dict[str, dict[int, dict]] = {}

    for name, stem in bundle_specs:
        per_ctx = {}
        for bundle in bundles:
            ctx = int(bundle.name.split("_")[1])
            path = bundle / f"{stem}.b2.json"
            if path.exists():
                per_ctx[ctx] = load_json(path)
        if not per_ctx:
            raise RuntimeError(f"bundle-local candidate {name} matched no files for stem {stem}")
        candidates[name] = per_ctx

    for name, ctx, path in external_specs:
        candidates.setdefault(name, {})
        candidates[name][ctx] = load_json(path)

    if "baseline" not in candidates:
        raise RuntimeError("baseline candidate is required")

    rows = []
    details = []
    winner_counts: dict[str, int] = {}
    recoverable_winner_counts: dict[str, int] = {}

    for bundle in bundles:
        ctx = int(bundle.name.split("_")[1])
        available = {
            name: per_ctx[ctx]
            for name, per_ctx in candidates.items()
            if ctx in per_ctx
        }
        if "baseline" not in available:
            raise RuntimeError(f"missing baseline for context {ctx}")

        n_steps = len(available["baseline"]["per_step_accepted"])
        best_all = []
        best_recoverable = []
        per_step_sources = []
        per_step_best = []
        baseline_steps = [float(v) for v in available["baseline"]["per_step_accepted"]]
        rec_steps = recoverable.get(ctx, set())

        for i in range(n_steps):
            step = i + 1
            best_label = None
            best_value = None
            for name, payload in available.items():
                vals = payload["per_step_accepted"]
                if len(vals) != n_steps:
                    raise RuntimeError(f"per-step length mismatch for {name} at ctx {ctx}")
                value = float(vals[i])
                if best_value is None or value > best_value:
                    best_label = name
                    best_value = value
            assert best_label is not None and best_value is not None
            per_step_sources.append(best_label)
            per_step_best.append(best_value)
            best_all.append(best_value)
            winner_counts[best_label] = winner_counts.get(best_label, 0) + 1
            if step in rec_steps:
                best_recoverable.append(best_value)
                recoverable_winner_counts[best_label] = recoverable_winner_counts.get(best_label, 0) + 1

        baseline_mean = sum(baseline_steps) / len(baseline_steps)
        best_mean = sum(best_all) / len(best_all)
        row = {
            "context": ctx,
            "baseline_accepted": baseline_mean,
            "best_available_accepted": best_mean,
            "best_delta_pct": 100.0 * (best_mean / max(baseline_mean, 1e-9) - 1.0),
            "recoverable_steps": len(rec_steps),
            "baseline_recoverable_accepted": (
                sum(baseline_steps[s - 1] for s in rec_steps) / len(rec_steps)
                if rec_steps else None
            ),
            "best_recoverable_accepted": (
                sum(best_recoverable) / len(best_recoverable)
                if best_recoverable else None
            ),
        }
        if row["baseline_recoverable_accepted"] is not None and row["best_recoverable_accepted"] is not None:
            row["best_recoverable_delta_pct"] = 100.0 * (
                row["best_recoverable_accepted"] / max(row["baseline_recoverable_accepted"], 1e-9) - 1.0
            )
        rows.append(row)
        details.append(
            {
                "context": ctx,
                "available_candidates": sorted(available),
                "per_step_sources": per_step_sources,
                "per_step_best_available": per_step_best,
                "recoverable_steps": sorted(rec_steps),
            }
        )

    mean_delta = sum(float(r["best_delta_pct"]) for r in rows) / len(rows)
    recoverable_rows = [r for r in rows if r.get("best_recoverable_delta_pct") is not None]
    mean_recoverable_delta = (
        sum(float(r["best_recoverable_delta_pct"]) for r in recoverable_rows) / len(recoverable_rows)
        if recoverable_rows else None
    )
    out = {
        "sweep_root": str(sweep_root),
        "bundle_labels": dict(bundle_specs),
        "external_b2": [
            {"name": name, "context": ctx, "path": str(path)}
            for name, ctx, path in external_specs
        ],
        "recoverable_details_json": str(Path(args.recoverable_details_json).resolve()) if args.recoverable_details_json else None,
        "rows": rows,
        "mean_best_delta_pct": mean_delta,
        "mean_best_recoverable_delta_pct": mean_recoverable_delta,
        "winner_counts": winner_counts,
        "recoverable_winner_counts": recoverable_winner_counts,
        "details": details,
    }
    out_path = Path(args.out_json).resolve()
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({
        "rows": rows,
        "mean_best_delta_pct": mean_delta,
        "mean_best_recoverable_delta_pct": mean_recoverable_delta,
        "winner_counts": winner_counts,
        "recoverable_winner_counts": recoverable_winner_counts,
        "out_json": str(out_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
