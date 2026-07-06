#!/usr/bin/env python3
"""Minimal safetensors loader for the DFlash drafter (BF16 -> F32 numpy).

Reads only the 8-byte header length + JSON header, then mmaps each tensor's
byte range. BF16 is the only float dtype in the DFlash checkpoint; we dequant
to F32 for the numpy oracle (matching the DSpark oracle convention).
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np


def _bf16_to_f32(arr_u16: np.ndarray) -> np.ndarray:
    """BF16 (u16 bits) -> f32: BF16 is the top 16 bits of f32."""
    return (arr_u16.astype(np.uint32) << 16).view(np.float32).copy()


def index_safetensors(path: str | Path) -> tuple[dict, int]:
    """Return (header_dict, data_offset). header_dict maps name -> {dtype, shape, data_offsets}."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    data_offset = 8 + n
    return header, data_offset


def load_safetensors(path: str | Path) -> dict[str, np.ndarray]:
    """Load every tensor as F32 numpy (BF16 dequant; I64/BOOL kept as-is)."""
    header, data_offset = index_safetensors(path)
    out: dict[str, np.ndarray] = {}
    header.pop("__metadata__", {})  # format metadata, not a tensor
    with open(path, "rb") as f:
        for name, info in header.items():
            start, end = info["data_offsets"]
            dtype = info["dtype"]
            shape = info["shape"]
            f.seek(data_offset + start)
            raw = f.read(end - start)
            if dtype == "BF16":
                arr = _bf16_to_f32(np.frombuffer(raw, dtype=np.uint16).copy())
                arr = arr.reshape(shape)
            elif dtype == "F32":
                arr = np.frombuffer(raw, dtype=np.float32).copy().reshape(shape)
            elif dtype == "I64":
                arr = np.frombuffer(raw, dtype=np.int64).copy().reshape(shape)
            elif dtype == "BOOL":
                arr = np.frombuffer(raw, dtype=np.bool_).copy().reshape(shape)
            else:
                raise ValueError(f"unhandled dtype {dtype} for {name}")
            out[name] = arr
    return out


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "model.safetensors"
    w = load_safetensors(path)
    print(f"loaded {len(w)} tensors from {path}")
    from collections import Counter
    print("dtypes:", dict(Counter(str(v.dtype) for v in w.values())))
    for nm in ["fc.weight", "hidden_norm.weight", "layers.0.self_attn.q_proj.weight",
               "layers.0.self_attn.k_proj.weight", "layers.0.mlp.gate_proj.weight",
               "norm.weight", "lm_head.weight", "d2t", "t2d", "embed_tokens.weight"]:
        if nm in w:
            a = w[nm]
            print(f"  {nm}: shape={a.shape} dtype={a.dtype} mean={np.float32(a.mean()):.5f}")
