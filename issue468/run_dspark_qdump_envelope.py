#!/usr/bin/env python3
"""Re-score existing DSpark q-dumps and compute an oracle envelope.

The envelope is intentionally impossible at runtime:

1. for each context and anchor step
2. re-score several existing q-dumps at high trial count
3. take the per-step maximum accepted value across all sources
4. compare that oracle chooser against a fixed baseline

This is a falsification / upper-bound tool for the assignment gate, not a
candidate-evaluation harness.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ISSUE468 = ROOT / "issue468"
DEFAULT_SWEEP_ROOT = Path("/tmp/dspark_sweep8")
DEFAULT_MEASURE = ISSUE468 / "baseline" / "dspark_capture" / "measure_metal_b2.py"
DEFAULT_MEASURE_PYTHON = ISSUE468 / ".venv" / "bin" / "python"
DEFAULT_BASELINE_DSPARK = (ROOT / ".." / "ds4" / "gguf" / "dspark.gguf").resolve()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-root", default=str(DEFAULT_SWEEP_ROOT))
    ap.add_argument("--measure-script", default=str(DEFAULT_MEASURE))
    ap.add_argument("--measure-python", default=str(DEFAULT_MEASURE_PYTHON))
    ap.add_argument("--baseline-dspark", default=str(DEFAULT_BASELINE_DSPARK))
    ap.add_argument("--trials", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=19)
    ap.add_argument("--baseline-label", default="baseline-weighted4ctx_19t_default_256tr")
    ap.add_argument(
        "--candidate-run-labels",
        nargs="+",
        default=[
            "weighted4ctx_19t_default_256tr",
            "weighted4ctx_19t_front_256tr",
            "weighted4ctx_19t_hard_256tr",
            "weighted4ctx_19t_tf_default_256tr",
        ],
    )
    ap.add_argument("--out-label", default="oracle-envelope-4096tr")
    ap.add_argument("--reuse-original-b2", action="store_true")
    return ap.parse_args()


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )


def plain_ms_for_context(ctx: int) -> float:
    anchors = {
        8192: 31.3,
        16384: 31.7,
        24576: 32.3,
        32768: 32.9,
        40960: 33.5,
        49152: 34.2,
        57344: 34.8,
        65536: 35.5,
    }
    return anchors.get(ctx, 35.5)


def ctx_dirs(root: Path) -> list[Path]:
    out = []
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith("ctx_") and (child / "target_topk.json").exists():
            out.append(child)
    return sorted(out)


def rescore(bundle: Path, *, q_path: Path, dspark: Path, label: str,
            pos0: int, trials: int, measure_python: str, measure_script: str,
            reuse_original_b2: bool) -> dict:
    if reuse_original_b2:
        return json.loads((bundle / f"{label}.b2.json").read_text())
    out_json = bundle / f"{label}.rescore{trials}.b2.json"
    if not out_json.exists():
        run([
            measure_python,
            measure_script,
            "--q-path", str(q_path),
            "--target-json", str(bundle / "target_topk.json"),
            "--greedy-json", str(bundle / "target_greedy.json"),
            "--dspark", str(dspark),
            "--pos0", str(pos0),
            "--trials", str(trials),
            "--label", label,
            "--plain-ms", str(plain_ms_for_context(pos0)),
            "--json-out", str(out_json),
        ])
    return json.loads(out_json.read_text())


def main() -> int:
    args = parse_args()
    sweep_root = Path(args.sweep_root).resolve()
    bundles = ctx_dirs(sweep_root)
    if not bundles:
        raise RuntimeError(f"no ctx_* bundles found under {sweep_root}")

    candidate_labels = list(args.candidate_run_labels)
    rows = []
    details = []
    baseline_path = Path(args.baseline_dspark).resolve()

    for bundle in bundles:
        ctx = int(bundle.name.split("_")[1])
        baseline_q = bundle / f"{args.baseline_label}.metal_base_logits_{args.steps}steps.bin"
        baseline = rescore(
            bundle,
            q_path=baseline_q,
            dspark=baseline_path,
            label=args.baseline_label,
            pos0=ctx,
            trials=args.trials,
            measure_python=args.measure_python,
            measure_script=str(Path(args.measure_script).resolve()),
            reuse_original_b2=args.reuse_original_b2,
        )

        source_results: dict[str, dict] = {
            "baseline": baseline,
        }
        for run_label in candidate_labels:
            q_path = bundle / f"candidate-{run_label}.metal_base_logits_{args.steps}steps.bin"
            dspark_path = sweep_root / f"{run_label}.gguf"
            source_results[run_label] = rescore(
                bundle,
                q_path=q_path,
                dspark=dspark_path,
                label=f"candidate-{run_label}",
                pos0=ctx,
                trials=args.trials,
                measure_python=args.measure_python,
                measure_script=str(Path(args.measure_script).resolve()),
                reuse_original_b2=args.reuse_original_b2,
            )

        per_step_sources = []
        best_per_step = []
        n_steps = len(baseline["per_step_accepted"])
        for i in range(n_steps):
            best_label = "baseline"
            best_value = float(source_results["baseline"]["per_step_accepted"][i])
            for run_label in candidate_labels:
                value = float(source_results[run_label]["per_step_accepted"][i])
                if value > best_value:
                    best_value = value
                    best_label = run_label
            per_step_sources.append(best_label)
            best_per_step.append(best_value)

        oracle_mean = sum(best_per_step) / len(best_per_step)
        row = {
            "context": ctx,
            "baseline_b2_accepted": float(baseline["average_accepted"]),
            "oracle_envelope_b2_accepted": oracle_mean,
            "envelope_delta_pct": 100.0 * (oracle_mean / max(float(baseline["average_accepted"]), 1e-9) - 1.0),
        }
        rows.append(row)
        details.append({
            "context": ctx,
            "per_step_sources": per_step_sources,
            "per_step_oracle_envelope": best_per_step,
        })
        print(json.dumps(row))

    mean_delta = sum(float(r["envelope_delta_pct"]) for r in rows) / len(rows)
    out_json = sweep_root / f"{args.out_label}.summary.json"
    out_tsv = sweep_root / f"{args.out_label}.summary.tsv"
    out_details = sweep_root / f"{args.out_label}.details.json"
    out_json.write_text(json.dumps({
        "trials": args.trials,
        "reuse_original_b2": args.reuse_original_b2,
        "baseline_label": args.baseline_label,
        "candidate_run_labels": candidate_labels,
        "rows": rows,
        "mean_envelope_delta_pct": mean_delta,
    }, indent=2) + "\n")
    out_details.write_text(json.dumps(details, indent=2) + "\n")
    header = "context\tbaseline_b2_accepted\toracle_envelope_b2_accepted\tenvelope_delta_pct"
    lines = [header]
    for row in rows:
        lines.append(
            f"{row['context']}\t{row['baseline_b2_accepted']}\t"
            f"{row['oracle_envelope_b2_accepted']}\t{row['envelope_delta_pct']}"
        )
    lines.append(f"MEAN\t\t\t{mean_delta}")
    out_tsv.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
