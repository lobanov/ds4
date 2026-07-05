#!/usr/bin/env python3
"""Bulk ceiling measurement across multiple drafter GGUFs over the full bundle set.

Unlike run_exactness_small_acceptance_from_bundles.py (which spawns a subprocess
per bundle and reloads the 87 GB target embed/lm_head + drafter GGUF + RoPE each
time), this runner:

  - loads the target model once (embed_w, lm_head, RoPE)
  - loads each drafter GGUF once (dense tensors, layers, expert stores)
  - iterates every retained bundle in-process via measure_bundle()
  - clears the expert cache between bundles to bound RAM

Use this to compare candidates (e.g. q4k_baseline vs f16_ceiling vs f32_ceiling)
quickly. Per-bundle JSONs are written under <out-root>/<label>/ and a per-label
summary.json + a comparison table are produced.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # research worktree root
ISSUE468 = ROOT / "issue468"
sys.path.insert(0, str(ISSUE468 / "dspark_oracle"))

from measure_acceptance_bundle import build_model_ctx, build_drafter_ctx, measure_bundle  # noqa: E402

DEFAULT_BUNDLES = ISSUE468 / "artifacts" / "exactness_small_bundles"
DEFAULT_OUT = ISSUE468 / "artifacts" / "exactness_small_acceptance"
DEFAULT_MODEL = (ROOT / ".." / "ds4" / "gguf" /
                 "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf").resolve()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles-dir", default=str(DEFAULT_BUNDLES))
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--out-root", default=str(DEFAULT_OUT))
    ap.add_argument("--candidate", action="append", required=True,
                    help="LABEL=GGUF (repeatable; order preserved in the table)")
    ap.add_argument("--keep-cache", action="store_true",
                    help="do not clear expert cache between bundles (faster re-reads, more RAM)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    bundles_dir = Path(args.bundles_dir)
    bundles = sorted(p for p in bundles_dir.iterdir() if p.is_dir())
    candidates = []
    for spec in args.candidate:
        if "=" not in spec:
            raise SystemExit(f"bad --candidate {spec!r}; expected LABEL=GGUF")
        label, gguf = spec.split("=", 1)
        candidates.append((label, gguf))

    print(f"loading target model once: {args.model}", file=sys.stderr)
    mctx = build_model_ctx(args.model)
    print(f"  model ready ({len(bundles)} bundles, {len(candidates)} candidates)", file=sys.stderr)

    for label, gguf in candidates:
        print(f"\n=== drafter {label}: {gguf} ===", file=sys.stderr)
        dctx = build_drafter_ctx(gguf)
        out_dir = Path(args.out_root) / label
        out_dir.mkdir(parents=True, exist_ok=True)
        agg = []
        for i, b in enumerate(bundles):
            summary = measure_bundle(b, mctx, dctx, candidate=label)
            (out_dir / f"{b.name}.json").write_text(json.dumps(summary, indent=2) + "\n")
            agg.append(summary)
            print(f"  [{i + 1:2d}/{len(bundles)}] {b.name}: avg_prefix={summary['average_prefix']:.3f}",
                  file=sys.stderr)
            if not args.keep_cache:
                for s in range(3):
                    dctx["stores"][s].clear()

        allp = [s["average_prefix"] for s in agg]
        bytemp: dict[float, list[float]] = {}
        for s in agg:
            bytemp.setdefault(s["temperature"], []).append(s["average_prefix"])
        sl = {
            "label": label,
            "dspark": gguf,
            "n_bundles": len(agg),
            "mean_avg_prefix": statistics.mean(allp) if allp else 0.0,
            "by_temperature": {str(t): {"mean": statistics.mean(v), "n": len(v)}
                               for t, v in sorted(bytemp.items())},
        }
        (out_dir / "summary.json").write_text(json.dumps(sl, indent=2) + "\n")
        print(f"  -> mean avg_prefix = {sl['mean_avg_prefix']:.3f}", file=sys.stderr)

    print("\n===== COMPARISON (mean accepted prefix over 5-token block) =====")
    print(f"{'label':<16} {'mean':>7}   by temperature")
    for label, _ in candidates:
        sl = json.load(open(Path(args.out_root) / label / "summary.json"))
        bt = "  ".join(f"t{t}={v['mean']:.3f}" for t, v in sorted(sl["by_temperature"].items()))
        print(f"{label:<16} {sl['mean_avg_prefix']:>7.3f}   {bt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
