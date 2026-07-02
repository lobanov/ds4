#!/usr/bin/env python3
"""Build first-pass DSpark mixed routed Q2/Q4 frontier GGUF candidates."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "issue468"))

from plan_q2_q4_frontier_candidates import build_candidate_specs, routed_overrides  # noqa: E402


LAYERS = ("mtp.0", "mtp.1", "mtp.2")


def candidate_overrides(q4_layers: set[str]) -> list[str]:
    out: list[str] = []
    for layer in LAYERS:
        for override in routed_overrides(layer, layer in q4_layers):
            out.extend(["--tensor-type", override])
    return out


def build_command(*, quantizer: Path, hf: Path, template: Path, imatrix: Path, out_dir: Path, label: str, q4_layers: set[str]) -> list[str]:
    out_path = out_dir / f"dspark_frontier_{label}.gguf"
    cmd = [
        str(quantizer),
        "--hf", str(hf),
        "--template", str(template),
        "--out", str(out_path),
        "--overwrite",
        "--imatrix", str(imatrix),
    ]
    cmd.extend(candidate_overrides(q4_layers))
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantizer", type=Path, default=Path("gguf-tools/deepseek4-quantize"))
    ap.add_argument("--hf-dspark", type=Path, default=(ROOT / ".." / "ds4" / "hf-dspark").resolve())
    ap.add_argument("--template-dspark", type=Path, default=(ROOT / ".." / "ds4" / "gguf" / "dspark.gguf").resolve())
    ap.add_argument("--imatrix", type=Path, default=Path("/private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx.imatrix.dat"))
    ap.add_argument("--out-dir", type=Path, default=Path("/private/tmp/dspark_pareto_q2q4"))
    ap.add_argument("--label", action="append", help="candidate label to build; repeatable. Default: all")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    specs = build_candidate_specs()
    selected = {label for label, _ in specs} if not args.label else set(args.label)

    for label, q4_layers in specs:
        if label not in selected:
            continue
        cmd = build_command(
            quantizer=args.quantizer.resolve(),
            hf=args.hf_dspark.resolve(),
            template=args.template_dspark.resolve(),
            imatrix=args.imatrix.resolve(),
            out_dir=args.out_dir.resolve(),
            label=label,
            q4_layers=q4_layers,
        )
        if args.dry_run:
            print(" ".join(cmd))
            continue
        print(f"[build] {label}", flush=True)
        subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
