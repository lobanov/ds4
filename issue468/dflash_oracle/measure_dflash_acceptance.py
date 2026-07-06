#!/usr/bin/env python3
"""Measure DFlash accepted prefix on the exactness small corpus.

For each captured cell (issue468/artifacts/dflash_capture/<cell>/context.npz),
simulate the DFlash draft cycle offline: step through the target token stream
committing one real target token at a time, accumulating the drafter's
ContextOnlyDraftKVCache over prompt+committed context, draft 7 tokens per step,
and measure the accepted prefix vs the target continuation. Directly comparable
to the DSpark oracle's accepted-prefix metric.

Cycle mapping (mirrors dflash_mlx/runtime.py generate_dflash_once, committing
real tokens):
  step 0: append prompt context (positions 0..pos0-1); anchor=target_tokens[0]
          at pos0; draft predicts target_tokens[1..7].
  step k: append 1 committed position (pos0+k-1); anchor=target_tokens[k];
          draft predicts target_tokens[k+1..k+7].  (k = 0..max_step)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from safetensors_loader import load_safetensors
from forward import (build_drafter_weights, forward, precompute_rope,
                     ContextOnlyDraftKVCache, N_LAYERS, BLOCK_SIZE, N_DRAFT,
                     DRAFT_VOCAB, MASK_TOKEN_ID, HEAD_DIM, MAX_POS)

ROOT = HERE.parents[1]
ISSUE468 = ROOT / "issue468"
CAPTURE = ISSUE468 / "artifacts" / "dflash_capture"
DSBARK_BUNDLES = ISSUE468 / "artifacts" / "exactness_small_bundles"
OUT = ISSUE468 / "artifacts" / "dflash_acceptance"
DEFAULT_MODEL = (ROOT / ".." / "ds4" / "gguf" /
                 "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf").resolve()
DEFAULT_WEIGHTS = ISSUE468 / "dflash_drafter" / "model.safetensors"
TARGET_LAYERS = [3, 13, 23, 32, 42]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="target GGUF (for embed_w)")
    ap.add_argument("--capture-dir", default=str(CAPTURE))
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--smoke", action="store_true", help="one cell only, verbose, no summary write")
    return ap.parse_args()


def load_target_embed(model_path: str) -> np.ndarray:
    sys.path.insert(0, str(ISSUE468 / "dspark_oracle"))
    from gguf_loader import index_gguf, read_tensor
    _, ti, tdo = index_gguf(model_path)
    return read_tensor(model_path, ti, tdo, "token_embd.weight").astype(np.float32)


def build_context_feature(ctx: dict) -> np.ndarray:
    """Concatenate the 5 target layers' HC hidden -> [n_pos, 5*16384=81920], layer order 3,13,23,32,42."""
    parts = [ctx[f"layer{L}"] for L in TARGET_LAYERS]
    return np.concatenate(parts, axis=-1).astype(np.float32)


def longest_prefix(draft, target):
    n = 0
    for d, t in zip(draft, target):
        if d == t:
            n += 1
        else:
            break
    return n


def measure_cell(g, layers, embed_w, cos_tbl, sin_tbl, d2t, cell_dir, pos0, verbose=False):
    ctx = np.load(cell_dir / "context.npz")
    context_feature = build_context_feature(ctx)          # [n_pos, 81920]
    positions = ctx["positions"]
    pos_to_idx = {int(p): i for i, p in enumerate(positions)}
    target_tokens = json.loads((cell_dir / "target_selected_tokens.json").read_text())
    max_step = len(target_tokens) - N_DRAFT - 1           # need target[k+1..k+7]
    caches = [ContextOnlyDraftKVCache() for _ in range(N_LAYERS)]
    rows = []
    prefix_hist = {k: 0 for k in range(N_DRAFT + 1)}
    total_match = 0
    total_pos = 0
    for k in range(max_step + 1):
        # new committed context positions to append this step
        if k == 0:
            new_positions = list(range(0, pos0))          # prompt
        else:
            new_positions = [pos0 + k - 1]                # previous anchor's position
        ctx_offset = caches[0].offset
        new_idx = [pos_to_idx[p] for p in new_positions]
        target_hidden_new = context_feature[new_idx][None]   # [1, ctx_len, 81920]
        ctx_len = len(new_positions)
        q_offset = ctx_offset + ctx_len
        anchor = int(target_tokens[k])
        block_ids = np.full(BLOCK_SIZE, MASK_TOKEN_ID, dtype=np.int64)
        block_ids[0] = anchor
        noise_emb = embed_w[block_ids][None]                 # [1, 8, 4096]
        logits = forward(g, layers, noise_emb, target_hidden_new, caches,
                         cos_tbl, sin_tbl, q_offset, ctx_offset)
        draft_logits = logits[0, 1:, :]                      # [7, 32000]
        draft_idx = np.argmax(draft_logits, axis=-1)
        draft = [int(i + d2t[i]) for i in draft_idx]   # d2t is an OFFSET: target_id = draft_idx + d2t[draft_idx]
        target_cont = [int(t) for t in target_tokens[k + 1:k + 1 + N_DRAFT]]
        match = sum(1 for d, t in zip(draft, target_cont) if d == t)
        prefix = longest_prefix(draft, target_cont)
        prefix_hist[prefix] += 1
        total_match += match
        total_pos += N_DRAFT
        rows.append({"step": k, "anchor": anchor, "draft": draft,
                     "target": target_cont, "match": match, "prefix": prefix})
        if verbose:
            print(f"  step{k} anchor={anchor} draft={draft} target={target_cont} prefix={prefix}")
    n_rows = len(rows)
    avg_prefix = float(sum(r["prefix"] for r in rows) / n_rows) if n_rows else 0.0
    return {
        "average_prefix": avg_prefix,
        "match_pct": (100.0 * total_match / total_pos) if total_pos else 0.0,
        "prefix_hist": prefix_hist,
        "measure_steps": n_rows,
        "total_match": total_match,
        "total_positions": total_pos,
        "rows": rows,
    }


def main():
    args = parse_args()
    print(f"loading DFlash weights: {args.weights}", flush=True)
    w = load_safetensors(args.weights)
    g, layers = build_drafter_weights(w)
    d2t = g["d2t"]
    cos_tbl, sin_tbl = precompute_rope(HEAD_DIM, MAX_POS)
    print("loading target embed_w ...", flush=True)
    embed_w = load_target_embed(args.model)
    # sanity: DFlash embed vs target embed (should match; drafter freezes target embed)
    diff = np.abs(g["embed"] - embed_w).mean()
    print(f"  |dflash_embed - target_embed| mean = {diff:.6f}")

    cells = sorted(p for p in Path(args.capture_dir).iterdir() if (p / "context.npz").exists())
    if args.only:
        cells = [c for c in cells if any(n in c.name for n in args.only)]
    if args.smoke:
        cells = cells[:1]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_summary = []
    for cell_dir in cells:
        # pos0 (prompt_tokens) from the matching DSpark bundle manifest
        dspark_manifest = DSBARK_BUNDLES / cell_dir.name / "bundle_manifest.json"
        pos0 = json.loads(dspark_manifest.read_text())["prompt_tokens"]
        verbose = args.smoke
        res = measure_cell(g, layers, embed_w, cos_tbl, sin_tbl, d2t, cell_dir, pos0, verbose=verbose)
        res["cell"] = cell_dir.name
        # prompt_name + temperature from the cell name "<name>__t<temp>"
        name, ttag = cell_dir.name.rsplit("__t", 1)
        res["prompt_name"] = name
        res["temperature"] = float(ttag.replace("p", "."))
        if not args.smoke:
            (out_dir / f"{cell_dir.name}.json").write_text(json.dumps(res, indent=2) + "\n")
        rows_summary.append(res)
        print(f"[{cell_dir.name}] avg_prefix={res['average_prefix']:.3f} match_pct={res['match_pct']:.1f}",
              flush=True)

    if args.smoke:
        return
    # aggregate
    import statistics
    bytemp = {}
    for r in rows_summary:
        bytemp.setdefault(r["temperature"], []).append(r["average_prefix"])
    summary = {
        "n_cells": len(rows_summary),
        "mean_avg_prefix": statistics.mean([r["average_prefix"] for r in rows_summary]),
        "mean_match_pct": statistics.mean([r["match_pct"] for r in rows_summary]),
        "by_temperature": {str(t): {"mean": statistics.mean(v), "n": len(v)}
                           for t, v in sorted(bytemp.items())},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    # csv
    lines = ["cell,prompt,temperature,average_prefix,match_pct,measure_steps"]
    for r in rows_summary:
        lines.append(f"{r['cell']},{r['prompt_name']},{r['temperature']},{r['average_prefix']:.4f},{r['match_pct']:.2f},{r['measure_steps']}")
    (out_dir / "summary.csv").write_text("\n".join(lines) + "\n")
    print(f"\n=== DFlash summary (n={summary['n_cells']}) mean avg_prefix={summary['mean_avg_prefix']:.3f} ===")
    for t, v in sorted(summary["by_temperature"].items()):
        print(f"  temp={t}: {v['mean']:.3f} (n={v['n']})")


if __name__ == "__main__":
    raise SystemExit(main())
