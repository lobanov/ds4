#!/usr/bin/env python3
"""Validate the Lead 08 iteration-28 exact small-M Stage-A case matrix."""

import argparse
import csv
import struct
from collections import Counter


FORMATS = {"q8_0": (1, 1024), "f16": (6, 256)}
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


def expected_hash(corpus, layer, site):
    words = input_words(corpus, layer, site)
    return ds4_hash(struct.pack(f"<{len(words)}I", *words))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cases_csv")
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
        corpus = int(row["corpus"].removeprefix("C"))
        format_name = row["format"]
        if format_name not in FORMATS:
            raise ValueError(f"unexpected format {format_name}")
        site, n = FORMATS[format_name]
        key = (layer, format_name, corpus)
        keys[key] += 1
        if not 0 <= layer < 43 or not 0 <= corpus < 5:
            raise ValueError(f"out-of-range key {key}")
        if (int(row["site"]), int(row["m"]), int(row["k"]), int(row["n"])) != (site, 2, 4096, n):
            raise ValueError(f"shape/site mismatch for {key}")
        if (int(row["candidate_dispatches"]), int(row["grid_y"])) != (1, 1):
            raise ValueError(f"dispatch geometry mismatch for {key}")
        if (int(row["bit_diffs"]), row["first_index"], row["first_ref_bits"],
                row["first_got_bits"], float(row["max_abs"]), row["result"]) != (
                    0, "-1", "00000000", "00000000", 0.0, "PASS"):
            raise ValueError(f"fidelity failure for {key}")
        expected = expected_hash(corpus, layer, site)
        if row["input_hash_ds4_64"] != expected:
            raise ValueError(f"input hash mismatch for {key}")
        hashes[key] = expected

    expected_keys = Counter((layer, format_name, corpus)
                            for layer in range(43)
                            for format_name in FORMATS
                            for corpus in range(5))
    if keys != expected_keys:
        raise ValueError("case matrix is incomplete or duplicated")
    if len(rows) != 430 or len(hashes) != 430:
        raise ValueError("expected exactly 430 unique cases")
    counts = Counter(row["format"] for row in rows)
    corpus_counts = Counter(row["corpus"] for row in rows)
    print(f"PASS cases={len(rows)} q8_0={counts['q8_0']} f16={counts['f16']} "
          f"corpora={dict(sorted(corpus_counts.items()))} bit_diffs=0")


if __name__ == "__main__":
    main()
