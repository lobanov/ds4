#!/usr/bin/env python3
"""Build recoverable-gap oracle-details JSON from per-context B2 outputs.

This converts one or more oracle-side ``*.b2.json`` files into the compact
``[{context, per_step_sources, per_step_oracle_envelope}, ...]`` format already
consumed by ``issue468/build_recoverable_gap_overlay.py``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--context-b2",
        action="append",
        default=[],
        metavar="CTX=PATH",
        help="context-to-B2 mapping, e.g. 8192=/tmp/ref-oracle-fp8-ctx08192-19.b2.json",
    )
    ap.add_argument(
        "--glob",
        action="append",
        default=[],
        help="optional glob(s) whose filenames contain ctx##### or ctx_#####",
    )
    ap.add_argument("--default-source", default="ref-oracle")
    ap.add_argument("--out-json", required=True)
    return ap.parse_args()


def parse_context_arg(item: str) -> tuple[int, Path]:
    if "=" not in item:
        raise RuntimeError(f"expected CTX=PATH, got: {item}")
    ctx_s, path_s = item.split("=", 1)
    return int(ctx_s), Path(path_s).resolve()


def infer_context_from_name(path: Path) -> int:
    m = re.search(r"ctx_?(\d{4,5})", path.name)
    if not m:
        raise RuntimeError(f"could not infer context from filename: {path}")
    return int(m.group(1))


def load_b2(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"missing oracle b2 file: {path}")
    data = json.loads(path.read_text())
    if "per_step_accepted" not in data:
        raise RuntimeError(f"missing per_step_accepted in {path}")
    return data


def main() -> int:
    args = parse_args()
    by_ctx: dict[int, Path] = {}

    for item in args.context_b2:
        ctx, path = parse_context_arg(item)
        by_ctx[ctx] = path

    for pattern in args.glob:
        for path in sorted(Path().glob(pattern)):
            by_ctx[infer_context_from_name(path)] = path.resolve()

    if not by_ctx:
        raise RuntimeError("no inputs provided")

    rows = []
    for ctx in sorted(by_ctx):
        data = load_b2(by_ctx[ctx])
        accepted = [float(v) for v in data["per_step_accepted"]]
        rows.append(
            {
                "context": ctx,
                "per_step_sources": [args.default_source] * len(accepted),
                "per_step_oracle_envelope": accepted,
            }
        )

    out = Path(args.out_json).resolve()
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({"contexts": len(rows), "out_json": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
