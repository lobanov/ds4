#!/usr/bin/env python3
"""Validate and analyze the Lead 08 V15 B2b fixed-work timing capture."""

import argparse
import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict


PRIMARY_SEED = 0x46815
STABILITY_SEED_BASE = 0x468160
ORDER_SEED = 0x46818
BOOTSTRAP_DRAWS = 100_000
def percentile(values, quantile):
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def bootstrap_median(values, seed, quantiles=(0.025, 0.975)):
    rng = random.Random(seed)
    count = len(values)
    samples = [
        statistics.median(values[rng.randrange(count)] for _ in range(count))
        for _ in range(BOOTSTRAP_DRAWS)
    ]
    return tuple(percentile(samples, quantile) for quantile in quantiles)


def bootstrap_order_difference(left, right):
    rng = random.Random(ORDER_SEED)
    nl = len(left)
    nr = len(right)
    samples = []
    for _ in range(BOOTSTRAP_DRAWS):
        lmed = statistics.median(left[rng.randrange(nl)] for _ in range(nl))
        rmed = statistics.median(right[rng.randrange(nr)] for _ in range(nr))
        samples.append(lmed - rmed)
    return percentile(samples, 0.025), percentile(samples, 0.975)


def theil_sen(values):
    slopes = [
        (values[j] - values[i]) / (j - i)
        for i in range(len(values))
        for j in range(i + 1, len(values))
    ]
    return statistics.median(slopes)


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    expected_fields = {
        "phase", "block", "schedule", "slot", "arm", "occurrence",
        "dispatches", "status", "completed", "gpu_start_s", "gpu_end_s",
        "gpu_ms", "kernel_ms", "encode_ms", "commit_wait_ms", "wall_ms", "result",
    }
    if not rows or set(rows[0]) != expected_fields:
        raise ValueError("unexpected raw timing schema")
    return rows


def validate_rows(rows):
    phases = Counter(row["phase"] for row in rows)
    measured = [row for row in rows if row["phase"] == "measure"]
    if len(measured) not in (160, 320):
        raise ValueError(f"expected 160 or 320 measured rows, got {len(measured)}")
    blocks = len(measured) // 4
    if phases != Counter({"prime": 2, "warmup": 16, "measure": 4 * blocks, "replay": 2}):
        raise ValueError(f"unexpected phase cardinality: {phases}")

    for row in rows:
        numeric = ("gpu_start_s", "gpu_end_s", "gpu_ms", "kernel_ms",
                   "encode_ms", "commit_wait_ms", "wall_ms")
        values = {name: float(row[name]) for name in numeric}
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError(f"nonfinite timing row: {row}")
        if (row["dispatches"], row["status"], row["completed"], row["result"]) != (
                "344", "4", "1", "PASS"):
            raise ValueError(f"failed work/status census: {row}")
        if values["gpu_start_s"] <= 0 or values["gpu_end_s"] <= values["gpu_start_s"]:
            raise ValueError(f"invalid raw GPU timestamps: {row}")
        derived_gpu = (values["gpu_end_s"] - values["gpu_start_s"]) * 1000.0
        if abs(derived_gpu - values["gpu_ms"]) > 0.002:
            raise ValueError(f"GPU duration does not match raw timestamps: {row}")
        if values["gpu_ms"] <= 0 or values["kernel_ms"] <= 0:
            raise ValueError(f"nonpositive GPU/supporting interval: {row}")
        if values["encode_ms"] < 0 or values["commit_wait_ms"] <= 0 or values["wall_ms"] <= 0:
            raise ValueError(f"invalid host interval: {row}")
        if abs(values["encode_ms"] + values["commit_wait_ms"] - values["wall_ms"]) > 0.003:
            raise ValueError(f"host intervals do not compose: {row}")

    by_block = defaultdict(list)
    for row in measured:
        by_block[int(row["block"])].append(row)
    if set(by_block) != set(range(blocks)):
        raise ValueError("measured blocks are incomplete")
    for block, block_rows in by_block.items():
        schedule = "ECCE" if block % 2 == 0 else "CEEC"
        ordered = sorted(block_rows, key=lambda row: int(row["slot"]))
        if len(ordered) != 4 or "".join(row["arm"] for row in ordered) != schedule:
            raise ValueError(f"bad schedule at block {block}")
        if any(row["schedule"] != schedule for row in ordered):
            raise ValueError(f"schedule label mismatch at block {block}")
        occurrences = Counter()
        for row in ordered:
            occurrences[row["arm"]] += 1
            if int(row["occurrence"]) != occurrences[row["arm"]]:
                raise ValueError(f"occurrence mismatch at block {block}")
    if Counter(row["schedule"] for row in measured) != Counter(
            {"ECCE": 2 * blocks, "CEEC": 2 * blocks}):
        raise ValueError("orders are unbalanced")
    return measured, blocks


def analyze(measured, blocks):
    samples = {}
    for row in measured:
        samples[(int(row["block"]), row["arm"], int(row["occurrence"]))] = float(row["gpu_ms"])

    arm_means = {arm: [] for arm in ("E", "C")}
    deltas = []
    order_deltas = defaultdict(list)
    for block in range(blocks):
        means = {}
        for arm in ("E", "C"):
            means[arm] = statistics.mean(samples[(block, arm, occurrence)] for occurrence in (1, 2))
            arm_means[arm].append(means[arm])
        delta = means["C"] - means["E"]
        deltas.append(delta)
        order_deltas["ECCE" if block % 2 == 0 else "CEEC"].append(delta)

    primary_median = statistics.median(deltas)
    primary_upper = bootstrap_median(deltas, PRIMARY_SEED, (0.95,))[0]

    stability = []
    validity_failures = []
    for schedule_id, schedule in enumerate(("ECCE", "CEEC")):
        schedule_blocks = [block for block in range(blocks)
                           if ("ECCE" if block % 2 == 0 else "CEEC") == schedule]
        for arm_id, arm in enumerate(("E", "C")):
            absolute = [samples[(block, arm, 2)] - samples[(block, arm, 1)]
                        for block in schedule_blocks]
            relative = [100.0 * value / samples[(block, arm, 1)]
                        for block, value in zip(schedule_blocks, absolute)]
            ci = bootstrap_median(
                relative, STABILITY_SEED_BASE + schedule_id * 2 + arm_id)
            absolute_median = statistics.median(absolute)
            passed = ci[0] >= -1.0 and ci[1] <= 1.0 and abs(absolute_median) <= 0.10
            if not passed:
                validity_failures.append(f"stability:{schedule}:{arm}")
            stability.append({
                "schedule": schedule,
                "arm": arm,
                "n": len(absolute),
                "relative_median_pct": statistics.median(relative),
                "relative_ci_low_pct": ci[0],
                "relative_ci_high_pct": ci[1],
                "absolute_median_ms": absolute_median,
                "pass": passed,
            })

    order_point = statistics.median(order_deltas["ECCE"]) - statistics.median(order_deltas["CEEC"])
    order_ci = bootstrap_order_difference(order_deltas["ECCE"], order_deltas["CEEC"])
    order_pass = (abs(order_point) <= 0.25 and order_ci[0] >= -0.50 and order_ci[1] <= 0.50)
    if not order_pass:
        validity_failures.append("schedule_effect")

    slopes = {
        "E_ms_per_block": theil_sen(arm_means["E"]),
        "C_ms_per_block": theil_sen(arm_means["C"]),
        "delta_ms_per_block": theil_sen(deltas),
    }
    for name, slope in slopes.items():
        if abs(slope) > 0.01:
            validity_failures.append(f"drift:{name}")

    validity = "PASS" if not validity_failures else "AMBIGUOUS"
    if validity == "PASS":
        outcome = "PASS" if primary_upper <= 7.5 else "STOP"
    else:
        outcome = "AMBIGUOUS"
    return {
        "schema": "lead08-v15-b2b-timing-v1",
        "blocks": blocks,
        "measured_observations": 4 * blocks,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "primary_seed": PRIMARY_SEED,
        "candidate_minus_ext_median_ms": primary_median,
        "candidate_minus_ext_upper_95_ms": primary_upper,
        "economic_limit_ms": 7.5,
        "stability": stability,
        "schedule_effect": {
            "point_ms": order_point,
            "ci_low_ms": order_ci[0],
            "ci_high_ms": order_ci[1],
            "pass": order_pass,
        },
        "theil_sen": slopes,
        "validity_failures": validity_failures,
        "validity": validity,
        "outcome": outcome,
        "work_census": {
            "layers": 43,
            "sites": 7,
            "logical_dispatches_per_layer": 8,
            "logical_dispatches_per_observation": 344,
            "exposed_weight_mib_per_layer": 77.875,
            "common_output_a_weight_mib_per_layer": 34.0,
            "unique_mapped_weight_bytes": 5_044_305_920,
            "physical_dram_traffic_claimed": False,
        },
        "claim_scope": "fixed-work dense-carrier admission screen only",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_csv")
    parser.add_argument("summary_json")
    parser.add_argument("--prior-summary")
    args = parser.parse_args()
    rows = load_rows(args.raw_csv)
    measured, blocks = validate_rows(rows)
    summary = analyze(measured, blocks)
    if blocks == 80:
        if not args.prior_summary:
            raise ValueError("80-block repeat requires --prior-summary")
        with open(args.prior_summary, encoding="utf-8") as source:
            prior = json.load(source)
        if (prior.get("blocks"), prior.get("validity"), prior.get("outcome")) != (
                40, "AMBIGUOUS", "AMBIGUOUS"):
            raise ValueError("prior summary is not the required ambiguous 40-block run")
        summary["prior_summary"] = args.prior_summary
        summary["repeat_disposition"] = (
            "persistent ambiguity is STOP" if summary["validity"] == "AMBIGUOUS"
            else "repeat resolved validity"
        )
        if summary["validity"] == "AMBIGUOUS":
            summary["outcome"] = "STOP"
    elif args.prior_summary:
        raise ValueError("--prior-summary applies only to the 80-block repeat")
    with open(args.summary_json, "w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, sort_keys=True)
        output.write("\n")
    print(
        f"{summary['outcome']} validity={summary['validity']} blocks={blocks} "
        f"median_delta_ms={summary['candidate_minus_ext_median_ms']:.6f} "
        f"upper95_ms={summary['candidate_minus_ext_upper_95_ms']:.6f} "
        f"failures={summary['validity_failures']}"
    )


if __name__ == "__main__":
    main()
