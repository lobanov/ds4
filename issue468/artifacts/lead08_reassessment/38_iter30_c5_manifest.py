#!/usr/bin/env python3
"""Validate and manifest the external Lead 08 B2a C5 capture."""

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path


CAPTURES = {
    "Lead08C5HCAttnFlat": 16384,
    "Lead08C5AttnNorm": 4096,
    "Lead08C5QLoraNorm": 1024,
    "Lead08C5Heads": 32768,
    "Lead08C5Low": 8192,
    "Lead08C5HCFFNFlat": 16384,
    "Lead08C5FFNNorm": 4096,
}
M = 4
LAYERS = 43
POSITION = 103


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(32 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--prompt", required=True, type=Path)
    parser.add_argument("--capture-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    expected = {
        args.capture_dir / f"c5_b_{name}-{layer}_pos{POSITION}.bin": (name, layer, width)
        for name, width in CAPTURES.items()
        for layer in range(LAYERS)
    }
    actual = set(args.capture_dir.glob("*.bin"))
    if actual != set(expected):
        missing = sorted(str(path) for path in set(expected) - actual)
        extra = sorted(str(path) for path in actual - set(expected))
        raise ValueError(f"capture set mismatch missing={missing[:3]} extra={extra[:3]}")

    rows = []
    total_bytes = 0
    for path in sorted(expected, key=lambda item: item.name):
        name, layer, width = expected[path]
        data = path.read_bytes()
        expected_bytes = M * width * 4
        if len(data) != expected_bytes:
            raise ValueError(f"wrong byte count for {path}: {len(data)} != {expected_bytes}")
        if any(not math.isfinite(value) for (value,) in struct.iter_unpack("<f", data)):
            raise ValueError(f"nonfinite capture value in {path}")
        total_bytes += len(data)
        rows.append({
            "file": path.name,
            "tag": "b_",
            "position": POSITION,
            "layer": layer,
            "name": name,
            "shape": [M, width],
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        })

    if len(rows) != 301 or total_bytes != 57065472:
        raise ValueError(f"capture cardinality mismatch files={len(rows)} bytes={total_bytes}")

    capture_command = {
        "gate": "unset",
        "DS4_DSPARK_VERIFY_DIST_PROBE": "1",
        "DS4_DSPARK_VERIFY_K": "4",
        "DS4_DSPARK_VERIFY_BATCHED": "0",
        "DS4_DSPARK_ANCHOR_REUSE": "1",
        "DS4_DSPARK_VERIFY_PREFIX_CHECKPOINT": "1",
        "DS4_DSPARK_DRAFT_METAL": "1",
        "DS4_DSPARK_DRAFT_METAL_STS": "1",
        "DS4_METAL_GRAPH_DUMP_NAME": ",".join(CAPTURES),
        "DS4_METAL_GRAPH_DUMP_LAYER": "all",
        "DS4_METAL_GRAPH_DUMP_POS": str(POSITION),
        "DS4_METAL_GRAPH_DUMP_PREFIX": str(args.capture_dir / "c5"),
        "argv": [
            "./ds4-spec-bench", "--metal", "-m", str(args.model),
            "--dspark", "/Users/lobanov/Projects/ds4/gguf/dspark.gguf",
            "--bulk-config", str(args.config),
            "--jsonl-out", str(args.capture_json),
        ],
    }
    manifest = {
        "schema": "lead08-v15-b2a-c5-v1",
        "result": "PASS",
        "tag": "b_",
        "position": POSITION,
        "m": M,
        "layers": LAYERS,
        "files": len(rows),
        "total_bytes": total_bytes,
        "model": {
            "path": str(args.model),
            "bytes": args.model.stat().st_size,
            "sha256": sha256_file(args.model),
        },
        "config": {"path": str(args.config), "sha256": sha256_file(args.config)},
        "prompt": {"path": str(args.prompt), "sha256": sha256_file(args.prompt)},
        "capture_json": {
            "path": str(args.capture_json),
            "sha256": sha256_file(args.capture_json),
        },
        "capture_command": capture_command,
        "captures": rows,
    }
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"PASS files={len(rows)} bytes={total_bytes} position={POSITION} tag=b_ finite=all")


if __name__ == "__main__":
    main()
