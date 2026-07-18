#!/usr/bin/env python3
"""Validate and summarize the Lead 08 iteration-23 direct-replay capture."""

import argparse
import csv
import math
import random
import re
import statistics
from collections import Counter, defaultdict


SAMPLE_RE = re.compile(
    r"SAMPLE stratum=(\d+) round=(\d+) mapped_alu=(\d+) layers=(\d+) "
    r"gpu_ms=([0-9.]+) kernel_ms=([0-9.]+) wall_ms=([0-9.]+).*"
    r"gpu_layers=\[([^]]*)\]"
)
BEGIN_RE = re.compile(r"BEGIN stratum=(\d+) round=(\d+) mapped_alu=(\d+)")
FID_RE = re.compile(
    r"mapped_alu_fid: SUMMARY selector=(\d+) strata=2 cases=86 failures=0 result=PASS"
)
ARMS = (0, 2, 1, 32, 128)
ROLE = {
    0: "production",
    2: "production_duplicate",
    1: "companion_zero",
    32: "candidate_r32",
    128: "candidate_r128",
}
COMPARISON = {0: None, 2: 0, 1: 0, 32: 1, 128: 1}


def percentile(values, quantile):
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def bootstrap_median(values, seed, draws=20000):
    rng = random.Random(seed)
    count = len(values)
    medians = [
        statistics.median(values[rng.randrange(count)] for _ in range(count))
        for _ in range(draws)
    ]
    return percentile(medians, 0.025), percentile(medians, 0.975)


def parse_capture(path):
    rows = []
    order = defaultdict(list)
    fidelity = set()
    with open(path, encoding="utf-8") as capture:
        for line in capture:
            match = BEGIN_RE.search(line)
            if match:
                order[(int(match[1]), int(match[2]))].append(int(match[3]))
            match = SAMPLE_RE.search(line)
            if match:
                layers = [float(value) for value in match[8].split(",") if value]
                rows.append(
                    {
                        "stratum": int(match[1]),
                        "round": int(match[2]),
                        "selector": int(match[3]),
                        "layer_count": int(match[4]),
                        "gpu_ms": float(match[5]),
                        "kernel_ms": float(match[6]),
                        "wall_ms": float(match[7]),
                        "layers": layers,
                    }
                )
            match = FID_RE.search(line)
            if match:
                fidelity.add(int(match[1]))
    return rows, order, fidelity


def validate(rows, order, fidelity):
    if len(rows) != 300:
        raise ValueError(f"expected 300 measured samples, got {len(rows)}")
    if fidelity != {1, 2, 32, 128}:
        raise ValueError(f"incomplete fidelity summaries: {sorted(fidelity)}")
    sample_counts = Counter((row["stratum"], row["selector"]) for row in rows)
    if any(sample_counts[(stratum, arm)] != 30 for stratum in (0, 1) for arm in ARMS):
        raise ValueError(f"unbalanced sample counts: {sample_counts}")
    for row in rows:
        if row["layer_count"] != 43 or len(row["layers"]) != 43:
            raise ValueError(f"invalid layer count: {row}")
        if any(not math.isfinite(value) or value <= 0.0 for value in row["layers"]):
            raise ValueError(f"invalid GPU interval: {row}")
        if abs(sum(row["layers"]) - row["gpu_ms"]) > 0.000050:
            raise ValueError(f"GPU aggregate does not match retained intervals: {row}")
        if row["kernel_ms"] <= 0.0 or row["wall_ms"] <= 0.0:
            raise ValueError(f"invalid supporting interval: {row}")
    positions = Counter()
    for stratum in (0, 1):
        for round_number in range(30):
            arm_order = order[(stratum, round_number)]
            if sorted(arm_order) != sorted(ARMS):
                raise ValueError(f"invalid arm order: {(stratum, round_number, arm_order)}")
            for position, arm in enumerate(arm_order):
                positions[(stratum, arm, position)] += 1
    if any(count != 6 for count in positions.values()) or len(positions) != 50:
        raise ValueError(f"unbalanced order positions: {positions}")


def write_layers(path, rows, order):
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("stratum", "round", "order_position", "selector", "layer", "gpu_ms"))
        for row in rows:
            position = order[(row["stratum"], row["round"])].index(row["selector"])
            for layer, gpu_ms in enumerate(row["layers"]):
                writer.writerow(
                    (row["stratum"], row["round"], position, row["selector"], layer, f"{gpu_ms:.6f}")
                )


def write_samples(path, rows, order):
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            ("stratum", "round", "order_position", "selector", "layers", "gpu_ms", "kernel_ms", "wall_ms")
        )
        for row in rows:
            position = order[(row["stratum"], row["round"])].index(row["selector"])
            writer.writerow(
                (
                    row["stratum"],
                    row["round"],
                    position,
                    row["selector"],
                    row["layer_count"],
                    f'{row["gpu_ms"]:.6f}',
                    f'{row["kernel_ms"]:.6f}',
                    f'{row["wall_ms"]:.6f}',
                )
            )


def write_summary(path, rows):
    samples = {
        (row["stratum"], row["round"], row["selector"]): row["gpu_ms"] for row in rows
    }
    with open(path, "w", newline="", encoding="utf-8") as output:
        fields = (
            "stratum",
            "selector",
            "role",
            "comparison_selector",
            "n",
            "mean_gpu_ms",
            "median_gpu_ms",
            "mad_gpu_ms",
            "median_ci_low_ms",
            "median_ci_high_ms",
            "paired_relative_median_pct",
            "paired_ci_low_pct",
            "paired_ci_high_pct",
        )
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for stratum in (0, 1):
            for arm in ARMS:
                values = [samples[(stratum, round_number, arm)] for round_number in range(30)]
                median = statistics.median(values)
                mad = statistics.median(abs(value - median) for value in values)
                median_ci = bootstrap_median(values, 31000 + stratum * 1000 + arm)
                comparison = COMPARISON[arm]
                if comparison is None:
                    paired_median = paired_low = paired_high = 0.0
                else:
                    differences = [
                        100.0
                        * (samples[(stratum, round_number, arm)] - samples[(stratum, round_number, comparison)])
                        / samples[(stratum, round_number, comparison)]
                        for round_number in range(30)
                    ]
                    paired_median = statistics.median(differences)
                    paired_low, paired_high = bootstrap_median(
                        differences, 32000 + stratum * 1000 + arm
                    )
                writer.writerow(
                    {
                        "stratum": "overlap" if stratum == 0 else "dispersed",
                        "selector": arm,
                        "role": ROLE[arm],
                        "comparison_selector": "" if comparison is None else comparison,
                        "n": 30,
                        "mean_gpu_ms": f"{statistics.mean(values):.6f}",
                        "median_gpu_ms": f"{median:.6f}",
                        "mad_gpu_ms": f"{mad:.6f}",
                        "median_ci_low_ms": f"{median_ci[0]:.6f}",
                        "median_ci_high_ms": f"{median_ci[1]:.6f}",
                        "paired_relative_median_pct": f"{paired_median:.6f}",
                        "paired_ci_low_pct": f"{paired_low:.6f}",
                        "paired_ci_high_pct": f"{paired_high:.6f}",
                    }
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture")
    parser.add_argument("layers_csv")
    parser.add_argument("samples_csv")
    parser.add_argument("summary_csv")
    args = parser.parse_args()
    rows, order, fidelity = parse_capture(args.capture)
    validate(rows, order, fidelity)
    write_layers(args.layers_csv, rows, order)
    write_samples(args.samples_csv, rows, order)
    write_summary(args.summary_csv, rows)
    print("PASS: 300 samples, 12,900 positive finite GPU intervals, balanced 6x positions")


if __name__ == "__main__":
    main()
