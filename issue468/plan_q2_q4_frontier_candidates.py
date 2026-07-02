#!/usr/bin/env python3
"""Emit first-pass DSpark mixed routed Q2/Q4 frontier candidate plans.

This helper does not build GGUFs. It prints:

- candidate label
- routed layers kept at Q4_K
- exact `deepseek4-quantize` `--tensor-type` overrides
- expected total size from `issue468/estimate_q2_q4_frontier_sizes.py`

The intent is to make the first Pareto build sweep reproducible and easy to
paste into shell commands or launch scripts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "issue468"))

from estimate_q2_q4_frontier_sizes import build_layer_inventory, gib, total_for_q4_layers  # noqa: E402


LAYERS = ("mtp.0", "mtp.1", "mtp.2")


def routed_overrides(layer: str, q4: bool) -> list[str]:
    if q4:
        gate_up = "q4_k"
        down = "q4_k"
    else:
        gate_up = "iq2_xxs"
        down = "q2_k"
    return [
        f"--tensor-type={layer}.ffn_gate_exps.weight={gate_up}",
        f"--tensor-type={layer}.ffn_up_exps.weight={gate_up}",
        f"--tensor-type={layer}.ffn_down_exps.weight={down}",
    ]


def build_candidate_specs() -> list[tuple[str, set[str]]]:
    return [
        ("all_q4", {"mtp.0", "mtp.1", "mtp.2"}),
        ("all_q2", set()),
        ("only_mtp2_q4", {"mtp.2"}),
        ("only_mtp1_q4", {"mtp.1"}),
        ("only_mtp0_q4", {"mtp.0"}),
        ("mtp01_q4", {"mtp.0", "mtp.1"}),
        ("mtp02_q4", {"mtp.0", "mtp.2"}),
        ("mtp12_q4", {"mtp.1", "mtp.2"}),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-gguf", type=Path, required=True)
    args = ap.parse_args()

    per_layer = build_layer_inventory(args.baseline_gguf.resolve())
    for label, q4_layers in build_candidate_specs():
        print(f"label: {label}")
        print(f"q4_layers: {','.join(sorted(q4_layers)) or '(none)'}")
        print(f"estimated_total_gib: {gib(total_for_q4_layers(per_layer, q4_layers)):.6f}")
        print("overrides:")
        for layer in LAYERS:
            for override in routed_overrides(layer, layer in q4_layers):
                print(f"  {override}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
