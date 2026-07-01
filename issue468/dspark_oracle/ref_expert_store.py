#!/usr/bin/env python3
"""Lazy routed-expert store for the converted reference checkpoint."""

from __future__ import annotations

import numpy as np

from ref_ckpt_loader import _dequant_fp4_tensor


class RefExpertStore:
    def __init__(self, path: str, stage: int):
        self.path = path
        self.stage = stage
        self._cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def _read(self, e: int, part: str):
        from safetensors import safe_open

        key = f"mtp.{self.stage}.ffn.experts.{e}.{part}.weight"
        scale_key = f"mtp.{self.stage}.ffn.experts.{e}.{part}.scale"
        with safe_open(self.path, framework="pt", device="cpu") as f:
            return _dequant_fp4_tensor(f.get_tensor(key), f.get_tensor(scale_key)).cpu().numpy().astype(np.float32)

    def expert(self, e: int):
        if e not in self._cache:
            self._cache[e] = (self._read(e, "w1"), self._read(e, "w3"), self._read(e, "w2"))
        return self._cache[e]

    def prefetch(self, expert_ids, n_workers=4):
        del n_workers
        for expert_id in set(int(i) for i in expert_ids):
            self.expert(expert_id)
