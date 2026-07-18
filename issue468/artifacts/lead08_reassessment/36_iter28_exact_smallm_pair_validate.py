#!/usr/bin/env python3
"""Validate Lead 08 exact small-M Stage-A or Stage-B1 case matrices."""

import argparse
import csv
import re
import struct
from collections import Counter
from functools import lru_cache


SITES = {
    0: ("f16", 16384, 24),
    1: ("q8_0", 4096, 1024),
    2: ("q8_0", 4096, 512),
    3: ("q8_0", 1024, 32768),
    4: ("q8_0", 8192, 4096),
    5: ("f16", 16384, 24),
    6: ("f16", 4096, 256),
}
HOST_KERNEL = re.compile(r"^define void @kernel_lead08_exact_smallm_(q8_0|f16)_f32_m([2-8])\(")
ACCUMULATOR = re.compile(
    r"^define linkonce_odr void @_Z3[34]kernel_exact_smallm_(q8|f16)_accumulateILs([2-8])")
C4_WORDS = (
    0x00000001, 0x80000001, 0x00800000, 0x80800000, 0x33800000,
    0xB3800000, 0x3F7FFFFF, 0xBF7FFFFF, 0x3F800001, 0xBF800001,
)


def ds4_hash(data):
    value = 1469598103934665603
    for byte in data:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def input_words(corpus, layer, site, m=2, k=4096):
    words = [0] * (m * k)
    if corpus == 0:
        return [0x80000000 if j & 1 else 0 for j in range(m * k)]
    if corpus == 1:
        for token in range(m):
            for position, base in enumerate((0, 3, 31, 32, k - 1)):
                index = (base + token) % k
                words[token * k + index] = 0xBF800000 if (position + token) & 1 else 0x3F800000
        return words
    if corpus == 2:
        for token in range(m):
            for index in range(k):
                exponent = -12 + ((index + 7 * token) % 17)
                words[token * k + index] = (((index + token) & 1) << 31) | ((exponent + 127) << 23)
        return words
    if corpus == 3:
        state = 0x04681500 ^ (layer << 16) ^ (site << 8) ^ m
        for j in range(m * k):
            state = (1664525 * state + 1013904223) & 0xFFFFFFFF
            magnitude = ((state >> 8) & 0xFFFF) / 65536.0
            word = struct.unpack("<I", struct.pack("<f", magnitude))[0]
            words[j] = (word & 0x7FFFFFFF) | (state & 0x80000000)
        return words
    if corpus == 4:
        for token in range(m):
            for index in range(k):
                words[token * k + index] = C4_WORDS[(index + 3 * token) % len(C4_WORDS)]
        return words
    raise ValueError(f"unexpected corpus C{corpus}")


@lru_cache(maxsize=None)
def expected_hash(corpus, layer, site, m, k):
    words = input_words(corpus, layer, site, m, k)
    return ds4_hash(struct.pack(f"<{len(words)}I", *words))


def validate_air(path, map_path):
    with open(path, encoding="utf-8") as source:
        lines = source.readlines()
    hosts = {}
    accumulators = {}
    for index, line in enumerate(lines):
        match = HOST_KERNEL.match(line)
        if match:
            hosts[(match[1], int(match[2]))] = index + 1
        match = ACCUMULATOR.match(line)
        if match:
            format_name = "q8_0" if match[1] == "q8" else "f16"
            accumulators[(format_name, int(match[2]))] = index
    expected = {(format_name, m) for format_name in ("q8_0", "f16") for m in range(2, 9)}
    if set(hosts) != expected or set(accumulators) != expected:
        raise ValueError("AIR does not contain exactly 14 expected host/accumulator specializations")

    map_rows = []
    for key in sorted(expected):
        format_name, m = key
        start = accumulators[key]
        end = next((i for i in range(start + 1, len(lines))
                    if lines[i].startswith("define ")), len(lines))
        block = lines[start:end]
        if format_name == "q8_0":
            raw_offsets = [i for i, line in enumerate(block)
                           if "load i8, i8 addrspace(1)*" in line]
            scale_offsets = [i for i, line in enumerate(block)
                             if "load half, half addrspace(1)*" in line]
            if len(raw_offsets) != 1 or len(scale_offsets) != 1:
                raise ValueError(f"unexpected Q8 raw load sites for M={m}")
            raw_offset = raw_offsets[0]
            if scale_offsets[0] > raw_offset:
                raise ValueError(f"Q8 scale load is not before token arithmetic for M={m}")
        else:
            raw_offsets = [i for i, line in enumerate(block)
                           if "load <4 x half>, <4 x half> addrspace(1)*" in line]
            if len(raw_offsets) != 1:
                raise ValueError(f"unexpected F16 raw load sites for M={m}")
            raw_offset = raw_offsets[0]
            dot_offsets = [i for i, line in enumerate(block) if "@air.dot.v4f32" in line]
            if len(dot_offsets) != 1 or raw_offset > dot_offsets[0]:
                raise ValueError(f"F16 raw load is not before token dot loop for M={m}")

        axes = {int(match.group(1))
                for line in block
                for match in [re.search(r"extractelement <3 x i32> %4, i64 ([0-2])", line)]
                if match}
        if axes != {0}:
            raise ValueError(f"token grid axis consumed for {format_name} M={m}: {axes}")

        labels = {}
        for offset, line in enumerate(block):
            match = re.match(r"([0-9]+):", line)
            if match:
                labels[match.group(1)] = offset
        enclosing = []
        for offset, line in enumerate(block):
            for target in re.findall(r"label %([0-9]+)", line):
                if target in labels and labels[target] < raw_offset < offset:
                    enclosing.append((offset - labels[target], offset, target))
        if not enclosing:
            raise ValueError(f"no raw-load recurrence found for {format_name} M={m}")
        widest = max(enclosing)
        if sum(span == widest[0] for span, _, _ in enclosing) != 1:
            raise ValueError(f"ambiguous primary raw-load recurrence for {format_name} M={m}")
        map_rows.append((format_name, m, hosts[key], start + 1, start + raw_offset + 1,
                         start + widest[1] + 1, 1, "x_only", "PASS"))

    if map_path:
        with open(map_path, "w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(("format", "m", "host_line", "accumulator_line", "raw_load_line",
                             "primary_backedge_line", "raw_load_sites", "grid_axes", "result"))
            writer.writerows(map_rows)
    print("PASS air_topology=14/14 raw_load_sites=1 grid_axes=x_only primary_recurrence=1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cases_csv")
    parser.add_argument("--phase", choices=("pair", "direct"), default="pair")
    parser.add_argument("--air")
    parser.add_argument("--air-map")
    args = parser.parse_args()

    with open(args.cases_csv, newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    required = {
        "layer", "site", "format", "corpus", "m", "k", "n",
        "input_hash_ds4_64", "candidate_dispatches", "grid_y", "bit_diffs",
        "first_index", "first_ref_bits", "first_got_bits", "max_abs", "result",
    }
    if not rows or set(rows[0]) != required:
        raise ValueError("unexpected CSV schema")

    keys = Counter()
    hashes = {}
    for row in rows:
        layer = int(row["layer"])
        site = int(row["site"])
        m = int(row["m"])
        corpus = int(row["corpus"].removeprefix("C"))
        format_name = row["format"]
        if site not in SITES:
            raise ValueError(f"unexpected site {site}")
        expected_format, k, n = SITES[site]
        key = (layer, site, m, corpus)
        keys[key] += 1
        if not 0 <= layer < 43 or not 0 <= corpus < 5 or not 2 <= m <= 8:
            raise ValueError(f"out-of-range key {key}")
        if (format_name, int(row["k"]), int(row["n"])) != (expected_format, k, n):
            raise ValueError(f"shape/site mismatch for {key}")
        if (int(row["candidate_dispatches"]), int(row["grid_y"])) != (1, 1):
            raise ValueError(f"dispatch geometry mismatch for {key}")
        if (int(row["bit_diffs"]), row["first_index"], row["first_ref_bits"],
                row["first_got_bits"], float(row["max_abs"]), row["result"]) != (
                    0, "-1", "00000000", "00000000", 0.0, "PASS"):
            raise ValueError(f"fidelity failure for {key}")
        expected = expected_hash(corpus,
                                 layer if corpus == 3 else 0,
                                 site if corpus == 3 else 0,
                                 m,
                                 k)
        if row["input_hash_ds4_64"] != expected:
            raise ValueError(f"input hash mismatch for {key}")
        hashes[key] = expected

    expected_sites = range(7) if args.phase == "direct" else (1, 6)
    expected_m = range(2, 9) if args.phase == "direct" else (2,)
    expected_keys = Counter((layer, site, m, corpus)
                            for layer in range(43)
                            for site in expected_sites
                            for m in expected_m
                            for corpus in range(5))
    if keys != expected_keys:
        raise ValueError("case matrix is incomplete or duplicated")
    expected_count = 10535 if args.phase == "direct" else 430
    if len(rows) != expected_count or len(hashes) != expected_count:
        raise ValueError(f"expected exactly {expected_count} unique cases")
    counts = Counter(row["format"] for row in rows)
    site_counts = Counter(row["site"] for row in rows)
    m_counts = Counter(row["m"] for row in rows)
    corpus_counts = Counter(row["corpus"] for row in rows)
    print(f"PASS cases={len(rows)} q8_0={counts['q8_0']} f16={counts['f16']} "
          f"sites={dict(sorted(site_counts.items()))} m={dict(sorted(m_counts.items()))} "
          f"corpora={dict(sorted(corpus_counts.items()))} bit_diffs=0")
    if args.air:
        validate_air(args.air, args.air_map)
    elif args.air_map:
        raise ValueError("--air-map requires --air")


if __name__ == "__main__":
    main()
