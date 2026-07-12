#!/usr/bin/env python3
"""Convert live hidden dumps (from DS4_DSPARK_DUMP_HIDDEN) into the oracle-bundle
format that measure_acceptance_bundle.py consumes, so the offline oracle drafter
can be run on the LIVE greedy-spine hiddens (self-aligned to the live trajectory
-- no encoding/spine alignment needed).

Dump record format (little-endian): pos(int32), token(int32), hidden[3*4096](float32).
Output per prompt: <out>/<name>/oracle/oracle_inputs.npz {positions, main_hidden},
target_selected_tokens.json, bundle_manifest.json, prompt.txt.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

DUMP_DIR = Path(sys.argv[1])
OUT_DIR = Path(sys.argv[2])
HIDDEN_DIM = 3 * 4096
REC_BYTES = 4 + 4 + HIDDEN_DIM * 4

OUT_DIR.mkdir(parents=True, exist_ok=True)
n = 0
for dump in sorted(DUMP_DIR.glob("*.bin")):
    name = dump.stem
    data = dump.read_bytes()
    if len(data) % REC_BYTES != 0:
        print(f"[warn] {name}: dump size {len(data)} not a multiple of {REC_BYTES}")
    nrec = len(data) // REC_BYTES
    arr = np.frombuffer(data, dtype=np.dtype([("pos", "<i4"), ("tok", "<i4"), ("h", "<f4", HIDDEN_DIM)]))
    # sort by pos (should already be in order) and dedupe
    arr = np.sort(arr, order="pos")
    pos = arr["pos"].astype(np.int32)
    toks = arr["tok"].astype(np.int64).tolist()
    mh = np.ascontiguousarray(arr["h"])  # (N, 12288)
    bdir = OUT_DIR / name
    (bdir / "oracle").mkdir(parents=True, exist_ok=True)
    np.savez(bdir / "oracle" / "oracle_inputs.npz", positions=pos, main_hidden=mh)
    (bdir / "target_selected_tokens.json").write_text(json.dumps(toks) + "\n")
    (bdir / "bundle_manifest.json").write_text(json.dumps({
        "prompt_name": name, "prompt_file": "", "temperature": 0.0, "seed": 0,
        "ctx": 4096, "generated_tokens": len(toks), "block": 5,
        "measure_steps": max(0, len(toks) - 5 - 1), "prompt_tokens": 0,
        "reference_mode": "greedy",
    }, indent=2) + "\n")
    (bdir / "prompt.txt").write_text("")  # not needed; drafter uses captured hiddens
    n += 1
print(f"converted {n} dumps -> {OUT_DIR}")
