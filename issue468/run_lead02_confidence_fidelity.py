#!/usr/bin/env python3
"""Lead 02 fidelity gate: confidence extraction must not perturb oracle outputs.

Runs the retained numpy oracle on the exactness bundles twice:
  1. baseline path (no confidence fields requested)
  2. confidence path (include_confidence=True)

Gate condition:
  - draft rows, prefix histograms, match totals, and aggregate summaries must be
    bit-identical between the two runs after removing confidence-only fields.

This is the first trust gate for Lead 02. It proves the extraction path is
observationally inert with respect to previously trusted acceptance numerics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles-dir", default=str(HERE / "artifacts" / "exactness_small_bundles"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--dspark", default=None)
    ap.add_argument("--only-temp", type=float, default=0.0)
    ap.add_argument("--out", default=str(HERE / "artifacts" / "lead02_confidence_fidelity" / "summary.json"))
    return ap.parse_args()


def _default_model() -> str:
    return str((HERE.parent / ".." / "ds4" / "gguf" / "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf").resolve())


def _default_dspark() -> str:
    return str((HERE.parent / ".." / "ds4" / "gguf" / "dspark.gguf").resolve())


def _strip_conf(summary: dict) -> dict:
    clean = dict(summary)
    clean_rows = []
    for row in clean["rows"]:
        rr = dict(row)
        rr.pop("confidence_logits", None)
        rr.pop("confidence_scores", None)
        clean_rows.append(rr)
    clean["rows"] = clean_rows
    return clean


def main() -> int:
    args = parse_args()
    model = args.model or _default_model()
    dspark = args.dspark or _default_dspark()

    from dspark_oracle.measure_acceptance_bundle import build_model_ctx, build_drafter_ctx, measure_bundle

    bundles_dir = Path(args.bundles_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mctx = build_model_ctx(model)
    dctx = build_drafter_ctx(dspark)

    checked = []
    failures = []

    for bundle_dir in sorted(p for p in bundles_dir.iterdir() if p.is_dir()):
        manifest = json.loads((bundle_dir / "bundle_manifest.json").read_text())
        if float(manifest["temperature"]) != float(args.only_temp):
            continue
        base = measure_bundle(bundle_dir=bundle_dir, mctx=mctx, dctx=dctx, include_confidence=False)
        conf = measure_bundle(bundle_dir=bundle_dir, mctx=mctx, dctx=dctx, include_confidence=True)
        base_rows = base["rows"]
        conf_rows = conf["rows"]
        missing_conf = [
            row["step"] for row in conf_rows
            if "confidence_logits" not in row or "confidence_scores" not in row
        ]
        same = _strip_conf(base) == _strip_conf(conf)
        row_shapes_ok = all(
            len(row["confidence_logits"]) == 5 and len(row["confidence_scores"]) == 5
            for row in conf_rows
        )
        checked.append({
            "bundle": bundle_dir.name,
            "prompt_name": manifest["prompt_name"],
            "measure_steps": base["measure_steps"],
            "rows_equal_after_stripping_confidence": same,
            "confidence_present_all_rows": not missing_conf,
            "confidence_shape_5": row_shapes_ok,
        })
        if (not same) or missing_conf or (not row_shapes_ok):
            failures.append({
                "bundle": bundle_dir.name,
                "rows_equal_after_stripping_confidence": same,
                "missing_conf_steps": missing_conf,
                "confidence_shape_5": row_shapes_ok,
                "base_average_prefix": base["average_prefix"],
                "conf_average_prefix": conf["average_prefix"],
                "base_match_pct": base["match_pct"],
                "conf_match_pct": conf["match_pct"],
            })

    result = {
        "gate": "lead02_confidence_extraction_fidelity",
        "bundles_dir": str(bundles_dir),
        "model": model,
        "dspark": dspark,
        "temperature": args.only_temp,
        "n_bundles_checked": len(checked),
        "pass": len(failures) == 0 and len(checked) > 0,
        "checked": checked,
        "failures": failures,
        "criterion": "baseline summary must equal confidence-enabled summary after removing confidence-only fields",
    }
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "pass": result["pass"],
        "n_bundles_checked": result["n_bundles_checked"],
        "n_failures": len(failures),
        "out": str(out_path),
    }, indent=2))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
