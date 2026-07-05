#!/usr/bin/env python3
"""Derive compact oracle inputs from retained layer capture dumps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HC = 4
DIM = 4096
LAYERS = (40, 41, 42)
CANONICAL_CAPTURE = "captures.npz"
CANONICAL_INPUTS = "oracle_inputs.npz"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--positions", nargs="*", type=int)
    ap.add_argument("--manifest-out")
    return ap.parse_args()


def capture_npz_path(captures_dir: Path) -> Path:
    if captures_dir.name == "captures":
        return captures_dir / CANONICAL_CAPTURE
    return captures_dir / "captures" / CANONICAL_CAPTURE


def detect_positions(captures_dir: Path) -> list[int]:
    npz = capture_npz_path(captures_dir)
    if npz.exists():
        data = np.load(npz)
        return [int(v) for v in data["positions"].tolist()]
    positions = set()
    for path in captures_dir.rglob("dump_hc_ffn_post-*_pos*.bin"):
        tail = path.stem.split("_pos")[-1]
        if tail.isdigit():
            positions.add(int(tail))
    return sorted(positions)


def load_or_build_capture_arrays(captures_dir: Path) -> tuple[list[int], dict[int, np.ndarray]]:
    npz = capture_npz_path(captures_dir)
    if npz.exists():
        data = np.load(npz)
        positions = [int(v) for v in data["positions"].tolist()]
        layer_arrays = {
            40: data["layer40"],
            41: data["layer41"],
            42: data["layer42"],
        }
        return positions, layer_arrays

    positions = [p for p in detect_positions(captures_dir) if p != 0]
    layer_lists: dict[int, list[np.ndarray]] = {layer: [] for layer in LAYERS}
    for pos in positions:
        for layer in LAYERS:
            matches = sorted(captures_dir.rglob(f"dump_hc_ffn_post-{layer}_pos{pos}.bin"))
            if len(matches) != 1:
                raise FileNotFoundError(f"expected exactly one capture for layer={layer} pos={pos}, found {len(matches)}")
            path = matches[0]
            arr = np.fromfile(path, dtype=np.float32)
            if arr.size != HC * DIM:
                raise ValueError(f"unexpected capture size for {path}: {arr.size} (wanted {HC * DIM})")
            layer_lists[layer].append(arr.reshape(HC, DIM))
    layer_arrays = {layer: np.stack(layer_lists[layer], axis=0) for layer in LAYERS}
    np.savez(
        npz,
        positions=np.array(positions, dtype=np.int32),
        layer40=layer_arrays[40],
        layer41=layer_arrays[41],
        layer42=layer_arrays[42],
    )
    return positions, layer_arrays


def main() -> int:
    args = parse_args()
    captures_dir = Path(args.captures_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    positions, layer_arrays = load_or_build_capture_arrays(captures_dir)
    requested = args.positions if args.positions else [p for p in positions if p != 0]
    pos_to_idx = {int(pos): idx for idx, pos in enumerate(positions)}
    rows = []
    inputs_positions = []
    inputs_main_hidden = []
    for pos in requested:
        idx = pos_to_idx[pos]
        parts = []
        for layer in LAYERS:
            arr = layer_arrays[layer][idx].mean(axis=0)
            parts.append(arr)
        main_hidden = np.concatenate(parts).astype(np.float32)
        out_path = out_dir / f"main_hidden_pos{pos}.npy"
        np.save(out_path, main_hidden)
        inputs_positions.append(pos)
        inputs_main_hidden.append(main_hidden)
        rows.append({
            "position": pos,
            "out": str(out_path),
            "shape": list(main_hidden.shape),
        })
    np.savez(
        out_dir / CANONICAL_INPUTS,
        positions=np.array(inputs_positions, dtype=np.int32),
        main_hidden=np.stack(inputs_main_hidden, axis=0) if inputs_main_hidden else np.zeros((0, 3 * DIM), dtype=np.float32),
    )
    if args.manifest_out:
        Path(args.manifest_out).write_text(json.dumps(rows, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
