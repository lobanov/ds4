#!/usr/bin/env python3
"""Build a dense imatrix entry from recoverable-step DSpark activations.

The existing DSpark collector only emits routed-expert imatrix entries. This
script creates imatrix-compatible dense vectors directly from sweep bundles so
the dense legal Q8_0 lane can consume the same recoverable-gap weighting logic
without changing ds4 runtime code.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

from issue468.build_recoverable_gap_overlay import build_weights, bundle_dirs, load_oracle_details, read_b2


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-root", required=True)
    ap.add_argument("--baseline-label", required=True)
    ap.add_argument("--oracle-details-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tensor-name", default="mtp.0.main_proj.weight")
    ap.add_argument("--baseline-max", type=float, default=4.999)
    ap.add_argument("--min-gap", type=float, default=0.05)
    ap.add_argument("--mode", choices=["soft", "binary", "boosted"], default="boosted")
    ap.add_argument("--recoverable-boost", type=float, default=8.0)
    ap.add_argument("--nonrecoverable-weight", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--floor", type=float, default=0.5)
    ap.add_argument("--ceil", type=float, default=2.0)
    return ap.parse_args()


def load_main_hidden(bundle: Path, pos: int) -> np.ndarray:
    parts = []
    for layer_id in (40, 41, 42):
        path = bundle / f"hc_dspark_main_hc-{layer_id}_pos{pos}.bin"
        vec = np.fromfile(path, dtype=np.float32).reshape(4, -1).mean(axis=0)
        parts.append(vec)
    return np.concatenate(parts, axis=0).astype(np.float32)


def write_imatrix(path: Path, name: str, values: np.ndarray, ncall: int) -> None:
    if values.dtype != np.float32:
        values = values.astype(np.float32)
    with path.open("wb") as f:
        f.write(struct.pack("<i", 1))
        raw_name = name.encode("utf-8")
        f.write(struct.pack("<i", len(raw_name)))
        f.write(raw_name)
        f.write(struct.pack("<i", ncall))
        f.write(struct.pack("<i", int(values.size)))
        f.write(values.tobytes(order="C"))


def main() -> int:
    args = parse_args()
    sweep_root = Path(args.sweep_root).resolve()
    oracle_details = load_oracle_details(Path(args.oracle_details_json).resolve())
    bundles = bundle_dirs(sweep_root)

    accum: np.ndarray | None = None
    ncall = 0
    for bundle in bundles:
        ctx = int(bundle.name.split("_")[1])
        base = read_b2(bundle, args.baseline_label)
        oracle_steps = oracle_details.get(ctx)
        if oracle_steps is None:
            raise RuntimeError(f"missing oracle details for context {ctx}")
        _weights, details = build_weights(
            [float(v) for v in base["per_step_accepted"]],
            oracle_steps,
            mode=args.mode,
            recoverable_boost=args.recoverable_boost,
            nonrecoverable_weight=args.nonrecoverable_weight,
            baseline_max=args.baseline_max,
            min_gap=args.min_gap,
            alpha=args.alpha,
            floor=args.floor,
            ceil=args.ceil,
        )
        for item in details:
            if not item["recoverable"]:
                continue
            step = int(item["step"])
            pos_decode = ctx + step
            x = load_main_hidden(bundle, pos_decode)
            if accum is None:
                accum = np.zeros_like(x, dtype=np.float64)
            accum += float(item["weight"]) * np.square(x, dtype=np.float64)
            ncall += 1

    if accum is None or ncall == 0:
        raise RuntimeError("no recoverable steps found")

    out_path = Path(args.out).resolve()
    write_imatrix(out_path, args.tensor_name, accum.astype(np.float32), ncall)
    print(f"wrote {out_path}")
    print(f"tensor {args.tensor_name}")
    print(f"ncall {ncall}")
    print(f"nval {accum.size}")
    print(f"sum {float(accum.sum())}")
    print(f"max {float(accum.max())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
