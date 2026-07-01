#!/usr/bin/env python3
"""Summarize DSpark routed-expert imatrix hotspots by layer/expert/block.

The DS4 imatrix collector stores one flat importance vector per routed expert
tensor entry. For DSpark routed tensors, that vector is packed as:

- n_experts contiguous expert slices
- each slice contains `ncols` column weights

This helper turns that into:

- per-layer totals
- per-layer/per-tensor totals
- top expert slices
- top super-blocks (default 256 columns, matching Q4_K)

It is intended to support the next post-weighting oracle-only branch from
`issue468/63_independent_review_after_oracle_pivot.md`:

- acceptance-aware local routed-`Q4_K` search

The output is descriptive only; it does not mutate GGUFs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
from collections import defaultdict
from pathlib import Path


ENTRY_RE = re.compile(r"^(?:blk|mtp)\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imatrix", required=True, help="legacy DS4 .dat imatrix path")
    ap.add_argument("--n-experts", type=int, default=256)
    ap.add_argument("--q4-block", type=int, default=256,
                    help="column span used for routed Q4_K super-block summaries")
    ap.add_argument("--top-layers", type=int, default=8)
    ap.add_argument("--top-experts", type=int, default=16)
    ap.add_argument("--top-blocks", type=int, default=24)
    ap.add_argument("--json-out")
    return ap.parse_args()


def read_i32(src) -> int:
    data = src.read(4)
    if len(data) != 4:
        raise EOFError("short i32 read")
    return struct.unpack("<i", data)[0]


def load_imatrix(path: Path) -> list[dict]:
    entries: list[dict] = []
    with path.open("rb") as src:
        n_entries = read_i32(src)
        if n_entries <= 0:
            raise RuntimeError(f"imatrix has no entries: {path}")
        for _ in range(n_entries):
            name_len = read_i32(src)
            if name_len <= 0 or name_len > 4096:
                raise RuntimeError(f"bad name length {name_len}")
            name = src.read(name_len).decode("utf-8")
            ncall = read_i32(src)
            nval = read_i32(src)
            if nval <= 0:
                raise RuntimeError(f"bad value count for {name}: {nval}")
            raw = src.read(4 * nval)
            if len(raw) != 4 * nval:
                raise EOFError(f"short value read for {name}")
            vals = struct.unpack(f"<{nval}f", raw)
            entries.append({
                "name": name,
                "ncall": ncall,
                "nval": nval,
                "values": vals,
            })
        # Optional trailing chunk count + dataset len are ignored.
    return entries


def summarize(entries: list[dict], *, n_experts: int, q4_block: int,
              top_layers: int, top_experts: int, top_blocks: int) -> dict:
    layer_totals: dict[int, float] = defaultdict(float)
    tensor_totals: dict[tuple[int, str], float] = defaultdict(float)
    expert_rows: list[dict] = []
    block_rows: list[dict] = []

    for entry in entries:
        match = ENTRY_RE.match(entry["name"])
        if not match:
            continue
        layer = int(match.group(1))
        part = match.group(2)
        vals = entry["values"]
        if len(vals) % n_experts != 0:
            raise RuntimeError(
                f"{entry['name']} has {len(vals)} values, not divisible by n_experts={n_experts}"
            )
        ncols = len(vals) // n_experts
        if ncols <= 0:
            continue

        total = float(sum(vals))
        layer_totals[layer] += total
        tensor_totals[(layer, part)] += total

        for expert in range(n_experts):
            start = expert * ncols
            end = start + ncols
            row = vals[start:end]
            row_sum = float(sum(row))
            if row_sum == 0.0:
                continue
            expert_rows.append({
                "layer": layer,
                "tensor_part": part,
                "expert": expert,
                "ncols": ncols,
                "weight_sum": row_sum,
            })

            for block_idx in range(math.ceil(ncols / q4_block)):
                b0 = block_idx * q4_block
                b1 = min(ncols, b0 + q4_block)
                block_sum = float(sum(row[b0:b1]))
                if block_sum == 0.0:
                    continue
                block_rows.append({
                    "layer": layer,
                    "tensor_part": part,
                    "expert": expert,
                    "block_index": block_idx,
                    "col_start": b0,
                    "col_end": b1,
                    "weight_sum": block_sum,
                })

    layer_rows = [
        {"layer": layer, "weight_sum": weight}
        for layer, weight in sorted(layer_totals.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    tensor_rows = [
        {"layer": layer, "tensor_part": part, "weight_sum": weight}
        for (layer, part), weight in sorted(tensor_totals.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
    ]
    expert_rows.sort(key=lambda row: (-row["weight_sum"], row["layer"], row["tensor_part"], row["expert"]))
    block_rows.sort(key=lambda row: (-row["weight_sum"], row["layer"], row["tensor_part"], row["expert"], row["block_index"]))

    return {
        "n_routed_entries": sum(1 for e in entries if ENTRY_RE.match(e["name"])),
        "n_experts": n_experts,
        "q4_block": q4_block,
        "top_layers": layer_rows[:top_layers],
        "top_layer_tensor_parts": tensor_rows[: max(top_layers * 3, 12)],
        "top_expert_rows": expert_rows[:top_experts],
        "top_blocks": block_rows[:top_blocks],
    }


def main() -> int:
    args = parse_args()
    entries = load_imatrix(Path(args.imatrix).resolve())
    report = summarize(
        entries,
        n_experts=args.n_experts,
        q4_block=args.q4_block,
        top_layers=args.top_layers,
        top_experts=args.top_experts,
        top_blocks=args.top_blocks,
    )
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
