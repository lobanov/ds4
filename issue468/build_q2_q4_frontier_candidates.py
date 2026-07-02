#!/usr/bin/env python3
"""Build DSpark mixed routed Q2/Q4 frontier GGUF candidates."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "issue468"))

from plan_q2_q4_frontier_candidates import build_candidate_specs_for_suite, override_lines  # noqa: E402


def candidate_overrides(tensor_modes: dict[str, str]) -> list[str]:
    out: list[str] = []
    for override in override_lines(tensor_modes):
        out.extend(["--tensor-type", override])
    return out


def build_command(*, quantizer: Path, hf: Path, template: Path, imatrix: Path, out_dir: Path, label: str, tensor_modes: dict[str, str]) -> list[str]:
    out_path = out_dir / f"dspark_frontier_{label}.gguf"
    cmd = [
        str(quantizer),
        "--hf", str(hf),
        "--template", str(template),
        "--out", str(out_path),
        "--overwrite",
        "--imatrix", str(imatrix),
    ]
    cmd.extend(candidate_overrides(tensor_modes))
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantizer", type=Path, default=Path("gguf-tools/deepseek4-quantize"))
    ap.add_argument("--hf-dspark", type=Path, default=(ROOT / ".." / "ds4" / "hf-dspark").resolve())
    ap.add_argument("--template-dspark", type=Path, default=(ROOT / ".." / "ds4" / "gguf" / "dspark.gguf").resolve())
    ap.add_argument("--imatrix", type=Path, default=Path("/private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx.imatrix.dat"))
    ap.add_argument("--out-dir", type=Path, default=Path("/private/tmp/dspark_pareto_q2q4"))
    ap.add_argument("--suite", choices=["coarse", "focused"], default="coarse")
    ap.add_argument("--label", action="append", help="candidate label to build; repeatable. Default: all")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    specs = build_candidate_specs_for_suite(args.suite)
    selected = {label for label, _ in specs} if not args.label else set(args.label)

    for label, tensor_modes in specs:
        if label not in selected:
            continue
        cmd = build_command(
            quantizer=args.quantizer.resolve(),
            hf=args.hf_dspark.resolve(),
            template=args.template_dspark.resolve(),
            imatrix=args.imatrix.resolve(),
            out_dir=args.out_dir.resolve(),
            label=label,
            tensor_modes=tensor_modes,
        )
        if args.dry_run:
            print(" ".join(cmd))
            continue
        print(f"[build] {label}", flush=True)
        subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
