#!/usr/bin/env python3
"""Estimate DSpark drafter sizes for mixed routed Q2/Q4 recipes.

This helper starts from an existing DSpark drafter GGUF and:

- separates routed-expert bytes from non-expert bytes by MTP layer
- assumes a routed Q2 recipe of:
  - gate/up experts -> IQ2_XXS
  - down experts -> Q2_K
- estimates total artifact size for any subset of layers or expert tensor parts
  kept at Q4_K

It is a planning tool for the first-pass size/quality Pareto frontier, not a
GGUF writer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gguf-tools" / "mixed"))

from splice_mixed_expert_layers_gguf import GGML_QUANT_SIZES, parse_gguf  # noqa: E402


Q4_K = "Q4_K"
IQ2_XXS = "IQ2_XXS"
Q2_K = "Q2_K"

EXPERT_NAMES = (
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_down_exps.weight",
)


def gib(n: int) -> float:
    return n / (1024 ** 3)


def layer_of(name: str) -> str | None:
    parts = name.split(".")
    if len(parts) >= 3 and parts[0] == "mtp" and parts[1].isdigit():
        return f"mtp.{parts[1]}"
    return None


def quant_ratio(dst_name: str, src_name: str) -> tuple[int, int]:
    src = next(v for v in GGML_QUANT_SIZES.values() if v[2] == src_name)
    dst = next(v for v in GGML_QUANT_SIZES.values() if v[2] == dst_name)
    return dst[1], src[1]


def estimate_q2_bytes(q4_bytes: int, part: str) -> int:
    if part in {"ffn_gate_exps.weight", "ffn_up_exps.weight"}:
        dst_b, src_b = quant_ratio(IQ2_XXS, Q4_K)
    elif part == "ffn_down_exps.weight":
        dst_b, src_b = quant_ratio(Q2_K, Q4_K)
    else:
        raise ValueError(f"unexpected expert part: {part}")
    if q4_bytes % src_b != 0:
        raise ValueError(f"Q4 byte size {q4_bytes} not divisible by block bytes {src_b}")
    return q4_bytes // src_b * dst_b


def build_layer_inventory(path: Path) -> dict[str, dict[str, int]]:
    info = parse_gguf(path)
    per_layer: dict[str, dict[str, int]] = {}
    for tensor in info.tensors:
        layer = layer_of(tensor.name)
        if layer is None:
            continue
        slot = per_layer.setdefault(layer, {"total": 0, "nonexpert": 0, "q4_expert": 0, "q2_expert": 0})
        slot["total"] += tensor.n_bytes
        leaf = ".".join(tensor.name.split(".")[2:])
        if leaf in EXPERT_NAMES:
            slot["q4_expert"] += tensor.n_bytes
            slot["q2_expert"] += estimate_q2_bytes(tensor.n_bytes, leaf)
        else:
            slot["nonexpert"] += tensor.n_bytes
    return per_layer


def build_expert_tensor_inventory(path: Path) -> dict[str, dict[str, dict[str, int]]]:
    info = parse_gguf(path)
    per_layer: dict[str, dict[str, dict[str, int]]] = {}
    for tensor in info.tensors:
        layer = layer_of(tensor.name)
        if layer is None:
            continue
        leaf = ".".join(tensor.name.split(".")[2:])
        if leaf not in EXPERT_NAMES:
            continue
        per_layer.setdefault(layer, {})[leaf] = {
            "q4": tensor.n_bytes,
            "q2": estimate_q2_bytes(tensor.n_bytes, leaf),
        }
    return per_layer


def total_for_q4_layers(per_layer: dict[str, dict[str, int]], q4_layers: set[str]) -> int:
    total = 0
    for layer, inv in per_layer.items():
        total += inv["nonexpert"]
        total += inv["q4_expert"] if layer in q4_layers else inv["q2_expert"]
    return total


def total_for_tensor_modes(
    per_layer: dict[str, dict[str, int]],
    expert_inventory: dict[str, dict[str, dict[str, int]]],
    tensor_modes: dict[str, str],
) -> int:
    total = 0
    for layer, inv in per_layer.items():
        total += inv["nonexpert"]
        for leaf in EXPERT_NAMES:
            mode = tensor_modes.get(f"{layer}.{leaf}", "q2")
            if mode not in {"q2", "q4"}:
                raise ValueError(f"unexpected mode {mode} for {layer}.{leaf}")
            total += expert_inventory[layer][leaf][mode]
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-gguf", type=Path, required=True)
    args = ap.parse_args()

    per_layer = build_layer_inventory(args.baseline_gguf.resolve())
    layers = sorted(per_layer)

    print(f"baseline: {args.baseline_gguf.resolve()}")
    print()
    print("| layer | non-expert GiB | expert Q4_K GiB | expert Q2 GiB | all-Q4 total GiB | all-Q2 total GiB |")
    print("|---|---:|---:|---:|---:|---:|")
    for layer in layers:
        inv = per_layer[layer]
        q4_total = inv["nonexpert"] + inv["q4_expert"]
        q2_total = inv["nonexpert"] + inv["q2_expert"]
        print(
            f"| {layer} | {gib(inv['nonexpert']):.6f} | {gib(inv['q4_expert']):.6f} | "
            f"{gib(inv['q2_expert']):.6f} | {gib(q4_total):.6f} | {gib(q2_total):.6f} |"
        )

    print()
    print("| recipe | q4_layers | estimated total GiB |")
    print("|---|---|---:|")

    candidate_sets = [
        ("all_q4", set(layers)),
        ("all_q2", set()),
        ("only_mtp2_q4", {"mtp.2"}),
        ("only_mtp1_q4", {"mtp.1"}),
        ("only_mtp0_q4", {"mtp.0"}),
        ("mtp01_q4", {"mtp.0", "mtp.1"}),
        ("mtp02_q4", {"mtp.0", "mtp.2"}),
        ("mtp12_q4", {"mtp.1", "mtp.2"}),
    ]
    for label, q4_layers in candidate_sets:
        print(f"| {label} | {','.join(sorted(q4_layers)) or '(none)'} | {gib(total_for_q4_layers(per_layer, q4_layers)):.6f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
