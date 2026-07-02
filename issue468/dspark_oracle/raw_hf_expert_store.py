#!/usr/bin/env python3
"""Lazy routed-expert store for the raw DSpark HF checkpoint."""

from __future__ import annotations

import numpy as np

from raw_hf_ckpt_loader import RawHFTensorStore, _dequant_fp4_weight


class RawHFExpertStore:
    def __init__(self, hf_dir: str, stage: int):
        self.stage = stage
        self.store = RawHFTensorStore(hf_dir)
        self._cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def _read(self, e: int, part: str):
        key = f"mtp.{self.stage}.ffn.experts.{e}.{part}.weight"
        scale_key = f"mtp.{self.stage}.ffn.experts.{e}.{part}.scale"
        meta, raw = self.store.raw_bytes(key)
        scale_meta, scale_raw = self.store.raw_bytes(scale_key)
        return _dequant_fp4_weight(raw, meta.shape, scale_raw, scale_meta.shape)

    def expert(self, e: int):
        if e not in self._cache:
            self._cache[e] = (self._read(e, "w1"), self._read(e, "w3"), self._read(e, "w2"))
        return self._cache[e]

    def prefetch(self, expert_ids, n_workers=4):
        del n_workers
        for expert_id in set(int(i) for i in expert_ids):
            self.expert(expert_id)
