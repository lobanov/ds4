#!/usr/bin/env python3
"""Build a two-step DSpark reference-harness NPZ from a capture bundle.

Input bundle layout:
  - hc_dspark_main_hc-{40,41,42}_pos*.bin
  - greedy25_tokens.json OR target_greedy.json

The produced NPZ is the preferred input to dspark_ref_harness.py --validate:
  - main_hidden_prefill [1,1,12288]
  - input_ids_prefill   [1]
  - main_hidden_decode  [1,1,12288]
  - input_ids_decode    [1]
"""

import argparse
import json
import os
import re

import numpy as np


HC = 4
DIM = 4096
LAYERS = (40, 41, 42)
POS_RE = re.compile(r"hc_dspark_main_hc-\d+_pos(\d+)\.bin$")


def load_greedy_tokens(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "steps" in data:
        return [step["selected"]["id"] for step in data["steps"]]
    raise ValueError(f"unsupported greedy-token format: {path}")


def infer_pos0(capture_dir):
    positions = []
    for name in os.listdir(capture_dir):
        m = POS_RE.match(name)
        if m:
            positions.append(int(m.group(1)))
    if not positions:
        raise FileNotFoundError(f"no hc_dspark_main_hc-* files found in {capture_dir}")
    return min(positions)


def load_main_hidden(capture_dir, pos):
    parts = []
    for layer in LAYERS:
        path = os.path.join(capture_dir, f"hc_dspark_main_hc-{layer}_pos{pos}.bin")
        arr = np.fromfile(path, dtype=np.float32)
        if arr.size != HC * DIM:
            raise ValueError(f"unexpected size for {path}: got {arr.size}, want {HC * DIM}")
        parts.append(arr.reshape(HC, DIM).mean(axis=0))
    return np.concatenate(parts, axis=0).astype(np.float32).reshape(1, 1, len(LAYERS) * DIM)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", required=True)
    ap.add_argument("--greedy-json", help="defaults to greedy25_tokens.json or target_greedy.json in capture-dir")
    ap.add_argument("--prefill-pos", type=int, required=True)
    ap.add_argument("--decode-pos", type=int, required=True)
    ap.add_argument("--pos0", type=int, help="absolute position mapped to greedy token 0; defaults to min captured pos")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    capture_dir = os.path.abspath(args.capture_dir)
    greedy_json = args.greedy_json
    if greedy_json is None:
        for candidate in ("greedy25_tokens.json", "target_greedy.json"):
            path = os.path.join(capture_dir, candidate)
            if os.path.exists(path):
                greedy_json = path
                break
    if greedy_json is None:
        raise FileNotFoundError(
            "could not infer greedy token source; pass --greedy-json explicitly"
        )

    tokens = load_greedy_tokens(greedy_json)
    pos0 = args.pos0 if args.pos0 is not None else infer_pos0(capture_dir)

    prefill_idx = args.prefill_pos - pos0
    decode_idx = args.decode_pos - pos0
    if prefill_idx < 0 or decode_idx < 0:
        raise ValueError(
            f"prefill/decode positions {args.prefill_pos}/{args.decode_pos} are before pos0={pos0}"
        )
    if prefill_idx >= len(tokens) or decode_idx >= len(tokens):
        raise ValueError(
            f"greedy token list has len={len(tokens)} but needs indices {prefill_idx}/{decode_idx}"
        )

    np.savez(
        args.out,
        main_hidden_prefill=load_main_hidden(capture_dir, args.prefill_pos),
        input_ids_prefill=np.array([tokens[prefill_idx]], dtype=np.int64),
        main_hidden_decode=load_main_hidden(capture_dir, args.decode_pos),
        input_ids_decode=np.array([tokens[decode_idx]], dtype=np.int64),
        meta=np.array(
            [
                args.prefill_pos,
                args.decode_pos,
                pos0,
                tokens[prefill_idx],
                tokens[decode_idx],
            ],
            dtype=np.int64,
        ),
    )
    print(
        f"wrote {args.out} "
        f"(pos0={pos0}, prefill_pos={args.prefill_pos}, decode_pos={args.decode_pos}, "
        f"prefill_tok={tokens[prefill_idx]}, decode_tok={tokens[decode_idx]})"
    )


if __name__ == "__main__":
    main()
