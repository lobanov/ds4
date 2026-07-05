#!/usr/bin/env python3
"""Re-run oracle acceptance on existing retained exactness bundles.

Use this when you want to compare a different DSpark GGUF against the same
already-captured target bundles, without re-running target-side capture.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ISSUE468 = ROOT / "issue468"
DEFAULT_BUNDLES = ISSUE468 / "artifacts" / "exactness_small_bundles"
DEFAULT_OUT = ISSUE468 / "artifacts" / "exactness_small_acceptance"
DEFAULT_MODEL = (ROOT / ".." / "ds4" / "gguf" / "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf").resolve()
DEFAULT_PY = sys.executable


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles-dir", default=str(DEFAULT_BUNDLES))
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--dspark", required=True)
    ap.add_argument("--label", required=True, help="output subdirectory label for this candidate run")
    ap.add_argument("--out-root", default=str(DEFAULT_OUT))
    ap.add_argument("--python", default=DEFAULT_PY)
    return ap.parse_args()


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, cwd=ROOT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def main() -> int:
    args = parse_args()
    bundles_dir = Path(args.bundles_dir)
    out_dir = Path(args.out_root) / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for bundle_dir in sorted(p for p in bundles_dir.iterdir() if p.is_dir()):
        out_json = out_dir / f"{bundle_dir.name}.json"
        run([
            args.python,
            str(ISSUE468 / "dspark_oracle" / "measure_acceptance_bundle.py"),
            "--bundle-dir", str(bundle_dir),
            "--model", args.model,
            "--dspark", args.dspark,
            "--json-out", str(out_json),
        ])
        acc = json.loads(out_json.read_text())
        rows.append({
            "bundle": bundle_dir.name,
            "prompt_name": acc["prompt_name"],
            "temperature": acc["temperature"],
            "reference_mode": acc["reference_mode"],
            "average_prefix": acc["average_prefix"],
            "match_pct": acc["match_pct"],
            "measure_steps": acc["measure_steps"],
            "json": str(out_json.relative_to(ROOT)),
        })

    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    header = ["bundle", "prompt_name", "temperature", "reference_mode", "average_prefix", "match_pct", "measure_steps", "json"]
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(row[key]) for key in header)
        )
    (out_dir / "summary.csv").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
