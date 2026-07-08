#!/usr/bin/env python3
"""Activity 7/9 prep — enrich features with K=5 targets + top-128 distributions for Ltv.

Pairs the existing torch-body features x (per anchor, in prompt order) with the K=5
future target tokens + their captured top-128 logprob distributions (from the shards),
so the Lce+Ltv loss (DSpark §3.3) can be computed. No body re-run.

Alignment (per prompt, feature j is the anchor at within-prompt position s=j+1; drafter
position i predicts token at s+1+i):
  target_k[i]   = token_ids[j+2+i],  i=0..4            (the K=5 targets)
  topk_k[i]     = shard topk at rows j+2..j+6           (target distributions)
  prev_tok[i]   = token_ids[j+1+i], i=0..4              (teacher-forced markov prev)
Keeps features where j+6 <= G-1 (trims ~4 anchors/prompt).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "dspark_train" / "data"
SHARDS = DATA / "shards"
K = 5


def enrich(split):
    feats_path = DATA / f"{split}_torch_features.safetensors"
    with safe_open(feats_path, "numpy") as d:
        fx = d.get_tensor("x"); fa = d.get_tensor("anchor"); fpi = d.get_tensor("prompt_idx")
    # shard per-prompt ordered token_ids + topk
    tok_all, tid_all, tlp_all, spi_all, pidx_all = [], [], [], [], []
    for f in sorted(SHARDS.glob(f"{split}_*.safetensors")):
        with safe_open(f, "numpy") as d:
            tok_all.append(d.get_tensor("token_ids")); tid_all.append(d.get_tensor("topk_ids"))
            tlp_all.append(d.get_tensor("topk_logprobs")); spi_all.append(d.get_tensor("pos_in_prompt"))
            pidx_all.append(d.get_tensor("prompt_idx"))
    tok_all = np.concatenate(tok_all); tid_all = np.concatenate(tid_all)
    tlp_all = np.concatenate(tlp_all); spi_all = np.concatenate(spi_all); pidx_all = np.concatenate(pidx_all)
    # group shard rows by prompt_idx, sorted by pos -> per-prompt j=0..G-1 order
    shard_by_p = {}
    for i in range(len(pidx_all)):
        shard_by_p.setdefault(int(pidx_all[i]), []).append(i)
    for p in shard_by_p:
        shard_by_p[p].sort(key=lambda i: spi_all[i])

    # iterate features in order; track (P, j) per prompt
    xs, anc, tk, tk_ids, tk_lp, prevs = [], [], [], [], [], []
    j_counter: dict[int, int] = {}
    for n in range(len(fpi)):
        p = int(fpi[n])
        j = j_counter.get(p, 0); j_counter[p] = j + 1
        rows = shard_by_p[p]
        G = len(rows)
        if j + (K + 1) >= G:  # need rows j+2..j+6 -> max j+6 <= G-1 -> j <= G-7
            continue
        r = rows
        target_k = np.array([int(tok_all[r[j + 2 + i]]) for i in range(K)], dtype=np.int64)
        prev_tok = np.array([int(tok_all[r[j + 1 + i]]) for i in range(K)], dtype=np.int64)  # incl anchor at i=0
        topk_k_ids = np.stack([tid_all[r[j + 2 + i]] for i in range(K)]).astype(np.int32)      # [K,128]
        topk_k_lp = np.stack([tlp_all[r[j + 2 + i]] for i in range(K)]).astype(np.float16)     # [K,128]
        xs.append(fx[n]); anc.append(int(fa[n])); tk.append(target_k)
        tk_ids.append(topk_k_ids); tk_lp.append(topk_k_lp); prevs.append(prev_tok)
    out = {
        "x": np.stack(xs).astype(np.float16),
        "anchor": np.array(anc, dtype=np.int64),
        "target_k": np.stack(tk).astype(np.int64),            # [N,K]
        "prev_tok": np.stack(prevs).astype(np.int64),         # [N,K] teacher-forced markov prev
        "topk_k_ids": np.stack(tk_ids).astype(np.int32),      # [N,K,128]
        "topk_k_logprobs": np.stack(tk_lp).astype(np.float16),# [N,K,128]
    }
    fn = DATA / f"{split}_kloss_features.safetensors"
    save_file({k: np.ascontiguousarray(v) for k, v in out.items()}, str(fn))
    print(f"{split}: {out['x'].shape[0]} enriched anchors -> {fn.name} (x{out['x'].shape}, target_k{out['target_k'].shape})")


def main():
    for split in ["train", "eval"]:
        enrich(split)


if __name__ == "__main__":
    main()
