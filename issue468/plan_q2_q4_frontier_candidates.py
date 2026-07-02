#!/usr/bin/env python3
"""Emit DSpark mixed routed Q2/Q4 frontier candidate plans.

This helper does not build GGUFs. It prints:

- candidate label
- routed layers kept at Q4_K
- exact `deepseek4-quantize` `--tensor-type` overrides
- expected total size from `issue468/estimate_q2_q4_frontier_sizes.py`

The intent is to make the coarse Pareto sweep and the next targeted follow-on
around the measured `mtp0/mtp2` family reproducible and easy to paste into
shell commands or launch scripts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "issue468"))

from estimate_q2_q4_frontier_sizes import (
    EXPERT_NAMES,
    build_expert_tensor_inventory,
    build_layer_inventory,
    gib,
    total_for_q4_layers,
    total_for_tensor_modes,
)  # noqa: E402


LAYERS = ("mtp.0", "mtp.1", "mtp.2")


def routed_overrides(layer: str, q4: bool) -> list[str]:
    if q4:
        gate_up = "q4_k"
        down = "q4_k"
    else:
        gate_up = "iq2_xxs"
        down = "q2_k"
    return [
        f"{layer}.ffn_gate_exps.weight={gate_up}",
        f"{layer}.ffn_up_exps.weight={gate_up}",
        f"{layer}.ffn_down_exps.weight={down}",
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


def tensor_mode_map(q4_layers: set[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for layer in LAYERS:
        for leaf in EXPERT_NAMES:
            out[f"{layer}.{leaf}"] = "q4" if layer in q4_layers else "q2"
    return out


def focused_candidate_specs() -> list[tuple[str, dict[str, str]]]:
    specs = []

    base_mtp02 = tensor_mode_map({"mtp.0", "mtp.2"})
    specs.append(("mtp02_q4", base_mtp02))

    mtp0_full_mtp2_gate = tensor_mode_map({"mtp.0"})
    mtp0_full_mtp2_gate["mtp.2.ffn_gate_exps.weight"] = "q4"
    specs.append(("mtp0_full_mtp2_gate_q4", mtp0_full_mtp2_gate))

    mtp0_full_mtp2_up = tensor_mode_map({"mtp.0"})
    mtp0_full_mtp2_up["mtp.2.ffn_up_exps.weight"] = "q4"
    specs.append(("mtp0_full_mtp2_up_q4", mtp0_full_mtp2_up))

    mtp0_full_mtp2_gateup = tensor_mode_map({"mtp.0"})
    for leaf in ("ffn_gate_exps.weight", "ffn_up_exps.weight"):
        mtp0_full_mtp2_gateup[f"mtp.2.{leaf}"] = "q4"
    specs.append(("mtp0_full_mtp2_gateup_q4", mtp0_full_mtp2_gateup))

    mtp0_full_mtp2_gateup_mtp1_down = dict(mtp0_full_mtp2_gateup)
    mtp0_full_mtp2_gateup_mtp1_down["mtp.1.ffn_down_exps.weight"] = "q4"
    specs.append(("mtp0_full_mtp2_gateup_mtp1_down_q4", mtp0_full_mtp2_gateup_mtp1_down))

    mtp0_full_mtp2_down = tensor_mode_map({"mtp.0"})
    mtp0_full_mtp2_down["mtp.2.ffn_down_exps.weight"] = "q4"
    specs.append(("mtp0_full_mtp2_down_q4", mtp0_full_mtp2_down))

    mtp2_full_mtp0_gateup = tensor_mode_map({"mtp.2"})
    for leaf in ("ffn_gate_exps.weight", "ffn_up_exps.weight"):
        mtp2_full_mtp0_gateup[f"mtp.0.{leaf}"] = "q4"
    specs.append(("mtp2_full_mtp0_gateup_q4", mtp2_full_mtp0_gateup))

    mtp2_full_mtp0_down = tensor_mode_map({"mtp.2"})
    mtp2_full_mtp0_down["mtp.0.ffn_down_exps.weight"] = "q4"
    specs.append(("mtp2_full_mtp0_down_q4", mtp2_full_mtp0_down))

    return specs


def build_candidate_specs_for_suite(
    suite: str,
) -> list[tuple[str, dict[str, str]]]:
    if suite == "coarse":
        return [(label, tensor_mode_map(q4_layers)) for label, q4_layers in build_candidate_specs()]
    if suite == "focused":
        return focused_candidate_specs()
    raise ValueError(f"unexpected suite: {suite}")


def override_lines(tensor_modes: dict[str, str]) -> list[str]:
    out: list[str] = []
    for layer in LAYERS:
        for leaf in EXPERT_NAMES:
            mode = tensor_modes[f"{layer}.{leaf}"]
            if mode == "q4":
                if leaf == "ffn_down_exps.weight":
                    quant = "q4_k"
                else:
                    quant = "q4_k"
            else:
                quant = "q2_k" if leaf == "ffn_down_exps.weight" else "iq2_xxs"
            out.append(f"{layer}.{leaf}={quant}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-gguf", type=Path, required=True)
    ap.add_argument("--suite", choices=["coarse", "focused"], default="coarse")
    args = ap.parse_args()

    per_layer = build_layer_inventory(args.baseline_gguf.resolve())
    expert_inventory = build_expert_tensor_inventory(args.baseline_gguf.resolve())
    for label, tensor_modes in build_candidate_specs_for_suite(args.suite):
        print(f"label: {label}")
        q4_layers = sorted({name.split(".")[0] + "." + name.split(".")[1] for name, mode in tensor_modes.items() if mode == "q4"})
        print(f"q4_layers: {','.join(q4_layers) or '(none)'}")
        print(f"estimated_total_gib: {gib(total_for_tensor_modes(per_layer, expert_inventory, tensor_modes)):.6f}")
        print("overrides:")
        for override in override_lines(tensor_modes):
            print(f"  --tensor-type {override}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
