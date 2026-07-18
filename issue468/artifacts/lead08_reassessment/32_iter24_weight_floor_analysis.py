#!/usr/bin/env python3
"""Validate and summarize the Lead 08 iteration-24 weight-floor capture."""

import argparse
import csv
import math
import random
import re
import statistics
from collections import Counter, defaultdict

SAMPLE = re.compile(r"SAMPLE stratum=(\d+) round=(\d+) mapped_alu=(\d+) layers=(\d+) gpu_ms=([0-9.]+) kernel_ms=([0-9.]+) wall_ms=([0-9.]+).*gpu_layers=\[([^]]*)\]")
BEGIN = re.compile(r"BEGIN stratum=(\d+) round=(\d+) mapped_alu=(\d+)")
FLOOR_FID = re.compile(r"floor_fid: stratum=(\d+) selector=(3|4) layer=(\d+) unwritten=0 duplicate_bits=0 hash=([0-9a-f]+) result=PASS")
PROD_FID = re.compile(r"mapped_alu_fid: SUMMARY selector=2 strata=2 cases=86 failures=0 result=PASS")
ARMS = (0, 2, 3, 4)
ROLES = {0: "production", 2: "production_duplicate", 3: "weight_floor", 4: "weight_floor_duplicate"}


def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def bootstrap_median(values, seed):
    rng = random.Random(seed)
    count = len(values)
    medians = [statistics.median(values[rng.randrange(count)] for _ in range(count)) for _ in range(20000)]
    return percentile(medians, 0.025), percentile(medians, 0.975)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture")
    parser.add_argument("samples_csv")
    parser.add_argument("layers_csv")
    parser.add_argument("summary_csv")
    args = parser.parse_args()
    rows = []
    orders = defaultdict(list)
    fidelity = []
    production_fidelity = False
    with open(args.capture, encoding="utf-8") as source:
        for line in source:
            match = BEGIN.search(line)
            if match:
                orders[(int(match[1]), int(match[2]))].append(int(match[3]))
            match = SAMPLE.search(line)
            if match:
                rows.append({
                    "stratum": int(match[1]), "round": int(match[2]), "selector": int(match[3]),
                    "count": int(match[4]), "gpu": float(match[5]), "kernel": float(match[6]),
                    "wall": float(match[7]), "layers": [float(value) for value in match[8].split(",")],
                })
            match = FLOOR_FID.search(line)
            if match:
                fidelity.append((int(match[1]), int(match[2]), int(match[3]), match[4]))
            if PROD_FID.search(line):
                production_fidelity = True
    if len(rows) != 256 or Counter((row["stratum"], row["selector"]) for row in rows) != Counter({(s, a): 32 for s in (0, 1) for a in ARMS}):
        raise ValueError("expected 32 samples for every stratum/arm")
    positions = Counter()
    for stratum in (0, 1):
        for round_number in range(32):
            order = orders[(stratum, round_number)]
            if sorted(order) != sorted(ARMS):
                raise ValueError("invalid arm schedule")
            for position, arm in enumerate(order):
                positions[(stratum, arm, position)] += 1
    if len(positions) != 32 or set(positions.values()) != {8}:
        raise ValueError("unbalanced order positions")
    for row in rows:
        if row["count"] != 43 or len(row["layers"]) != 43 or any(not math.isfinite(v) or v <= 0 for v in row["layers"]):
            raise ValueError("invalid layer intervals")
        if abs(sum(row["layers"]) - row["gpu"]) > 0.000050:
            raise ValueError("aggregate/layer mismatch")
    if len(fidelity) != 172 or len(set((s, a, layer) for s, a, layer, _ in fidelity)) != 172:
        raise ValueError("incomplete floor fidelity")
    if not production_fidelity:
        raise ValueError("missing production-duplicate fidelity summary")
    hashes = {(s, layer): value for s, a, layer, value in fidelity if a == 3}
    if len(set(hashes.values())) != 86 or any(hashes[(s, layer)] != value for s, a, layer, value in fidelity if a == 4):
        raise ValueError("floor hashes are constant or duplicate differs")

    with open(args.samples_csv, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("stratum", "round", "order_position", "selector", "layers", "gpu_ms", "kernel_ms", "wall_ms"))
        for row in rows:
            writer.writerow((row["stratum"], row["round"], orders[(row["stratum"], row["round"])].index(row["selector"]), row["selector"], 43, row["gpu"], row["kernel"], row["wall"]))
    with open(args.layers_csv, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("stratum", "round", "selector", "layer", "gpu_ms"))
        for row in rows:
            for layer, value in enumerate(row["layers"]):
                writer.writerow((row["stratum"], row["round"], row["selector"], layer, f"{value:.6f}"))
    values = {(row["stratum"], row["round"], row["selector"]): row["gpu"] for row in rows}
    saving_lowers = []
    for stratum in (0, 1):
        prod_relative = [100 * (values[(stratum, r, 2)] - values[(stratum, r, 0)]) / values[(stratum, r, 0)] for r in range(32)]
        floor_relative = [100 * (values[(stratum, r, 4)] - values[(stratum, r, 3)]) / values[(stratum, r, 3)] for r in range(32)]
        floor_absolute = [values[(stratum, r, 4)] - values[(stratum, r, 3)] for r in range(32)]
        saving = [values[(stratum, r, 0)] - values[(stratum, r, 3)] for r in range(32)]
        if max(abs(value) for value in bootstrap_median(prod_relative, 27000 + stratum * 100 + 2)) > 1.0:
            raise ValueError("production duplicate control failed")
        if max(abs(value) for value in bootstrap_median(floor_relative, 27000 + stratum * 100 + 4)) > 3.0:
            raise ValueError("floor relative duplicate control failed")
        if max(abs(value) for value in bootstrap_median(floor_absolute, 28000 + stratum * 100 + 4)) > 0.10:
            raise ValueError("floor absolute duplicate control failed")
        saving_lowers.append(bootstrap_median(saving, 28000 + stratum * 100 + 3)[0])
    with open(args.summary_csv, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("stratum", "selector", "role", "n", "mean_gpu_ms", "median_gpu_ms", "mad_gpu_ms", "relative_median_pct", "relative_ci_low_pct", "relative_ci_high_pct", "absolute_median_ms", "absolute_ci_low_ms", "absolute_ci_high_ms"))
        for stratum in (0, 1):
            for arm in ARMS:
                arm_values = [values[(stratum, round_number, arm)] for round_number in range(32)]
                median = statistics.median(arm_values)
                mad = statistics.median(abs(value - median) for value in arm_values)
                base = 0 if arm == 2 else 3 if arm == 4 else None
                relative = [100 * (values[(stratum, r, arm)] - values[(stratum, r, base)]) / values[(stratum, r, base)] for r in range(32)] if base is not None else [0.0]
                absolute = [values[(stratum, r, arm)] - values[(stratum, r, base)] for r in range(32)] if base is not None else ([values[(stratum, r, 0)] - values[(stratum, r, 3)] for r in range(32)] if arm == 3 else [0.0])
                relative_ci = bootstrap_median(relative, 27000 + stratum * 100 + arm) if base is not None else (0.0, 0.0)
                absolute_ci = bootstrap_median(absolute, 28000 + stratum * 100 + arm) if arm != 0 else (0.0, 0.0)
                writer.writerow(("overlap" if stratum == 0 else "dispersed", arm, ROLES[arm], 32, f"{statistics.mean(arm_values):.6f}", f"{median:.6f}", f"{mad:.6f}", f"{statistics.median(relative):.6f}", f"{relative_ci[0]:.6f}", f"{relative_ci[1]:.6f}", f"{statistics.median(absolute):.6f}", f"{absolute_ci[0]:.6f}", f"{absolute_ci[1]:.6f}"))
    decision = "GO optimistic bound" if all(value >= 8.7 for value in saving_lowers) else "STOP economics"
    print(f"PASS validity; {decision}: saving lower bounds {saving_lowers[0]:.6f}, {saving_lowers[1]:.6f} ms")


if __name__ == "__main__":
    main()
