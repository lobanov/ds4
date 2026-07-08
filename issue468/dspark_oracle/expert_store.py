#!/usr/bin/env python3
"""Lazy per-expert Q4_K dequant store for the DSpark numpy oracle.

Holds the raw packed Q4_K bytes for one layer's gate/up/down expert tensors
(same on-disk size, ~3.6GB total per layer) and dequants a single expert's
gate/up/down weights on demand, memoized. This mirrors how the ds4 Metal path
operates (experts stay quantized; dequanted per-call) and keeps the oracle's
working set small (~1GB for ~30 distinct experts touched per forward).

Optional prefetch(n_workers) dequants likely-needed experts in parallel to use
spare CPU. Default is on-demand only.
"""
import numpy as np
from collections import OrderedDict
from gguf_loader import dequant_q4_k_expert, read_dense_expert


class ExpertStore:
    """Per-layer lazy expert dequant for one DSparkBlock's MoE.

    `max_cache` (default None = unbounded) bounds the dequanted-expert cache as an
    LRU: once full, the least-recently-used expert is evicted. The drafter routes
    diversely (~215 unique experts/122-step prompt), so an unbounded cache can grow
    to ~77 GB (all 768 experts); Lead 03's powered measurement sets max_cache to
    bound RAM (experts are ~100 MB each). Hot experts recur across prompts, so a
    few-hundred cap retains most reuse after warmup."""

    def __init__(self, path, infos, data_off, stage, max_cache=None):
        self.path = path
        self.infos = infos
        self.data_off = data_off
        self.stage = stage
        self.max_cache = max_cache
        self._cache = OrderedDict()  # expert_id -> (gate, up, down)

    def _read(self, part, e):
        nm = f"mtp.{self.stage}.ffn_{part}_exps.weight"
        tt = self.infos[nm][1]
        if tt == 12:  # Q4_K
            return dequant_q4_k_expert(self.path, self.infos, self.data_off, nm, e)
        # dense expert (F32/F16/BF16) — used by unquantized ceiling GGUFs
        return read_dense_expert(self.path, self.infos, self.data_off, nm, e)

    def expert(self, e):
        """Return (gate_wg [inter,dim], up_wu [inter,dim], down_wd [dim,inter])."""
        if e in self._cache:
            self._cache.move_to_end(e)  # LRU: mark recently used
            return self._cache[e]
        val = (self._read("gate", e), self._read("up", e), self._read("down", e))
        self._cache[e] = val
        if self.max_cache is not None and len(self._cache) > self.max_cache:
            self._cache.popitem(last=False)  # evict least-recently-used
        return val

    def clear(self):
        """Drop cached dequanted experts to bound RAM between bundles in bulk mode."""
        self._cache.clear()

    def prefetch(self, expert_ids, n_workers=4):
        """Dequant a set of experts in parallel using a process pool (uses spare CPU).
        Each worker reads its own slice; results memoized into _cache."""
        from concurrent.futures import ProcessPoolExecutor
        todo = [e for e in set(int(i) for i in expert_ids) if e not in self._cache]
        if not todo:
            return
        args = [(self.path, self.infos, self.data_off, self.stage, e) for e in todo]
        # note: infos/data_off are plain dicts/ints -> picklable
        with ProcessPoolExecutor(max_workers=min(n_workers, len(todo))) as ex:
            results = list(ex.map(_prefetch_one, args))
        for e, res in zip(todo, results):
            self._cache[e] = res


def _prefetch_one(args):
    path, infos, data_off, stage, e = args
    def rd(part):
        nm = f"mtp.{stage}.ffn_{part}_exps.weight"
        tt = infos[nm][1]
        if tt == 12:  # Q4_K
            return dequant_q4_k_expert(path, infos, data_off, nm, e)
        return read_dense_expert(path, infos, data_off, nm, e)
    return (rd("gate"), rd("up"), rd("down"))
