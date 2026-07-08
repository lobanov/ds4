#!/usr/bin/env python3
"""Activity 3 — consolidate ds4 --capture-dataset output into sharded safetensors.

RESEARCH INSTRUMENTATION for Stage 2. Reads the per-prompt capture scratch
(<id>.topk.json + <id>_hc_ffn_post-{40,41,42}_pos{p}.bin) and writes sharded
safetensors per the stage2_finetune_protocol.md schema:

  main_hidden   [N, 12288] f16   -- drafter input = concat(mean_HC(layer40/41/42))
  token_ids     [N] i32          -- the token at position p (anchor + target via windowing)
  topk_ids      [N, 128] i32     -- target top-128 token ids over token[p]
  topk_logprobs [N, 128] f16     -- target top-128 logprobs over token[p] (for Ltv)
  prompt_idx    [N] i32          -- index into index.json prompts
  pos_in_prompt [N] i32          -- absolute position p (P..P+G-1)

Plus index.json (per-prompt: prompt_id, source, split, prompt_tokens, n_positions,
pos_range). pos0 (prefill artifact) is excluded. Train/eval split comes from the
Stage 2 manifest. K-step targets are derived by windowing token_ids at train time.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

HC, DIM = 4, 4096
LAYERS = (40, 41, 42)
TOPK = 128
SHARD_POS = 6000  # positions per shard


def read_layer_mean(capture_dir: Path, pid: str, layer: int, pos: int) -> np.ndarray | None:
    p = capture_dir / f"{pid}_hc_ffn_post-{layer}_pos{pos}.bin"
    if not p.exists():
        return None
    arr = np.fromfile(p, dtype=np.float32)
    if arr.size != HC * DIM:
        return None
    return arr.reshape(HC, DIM).mean(axis=0).astype(np.float16)  # [DIM]


def consolidate_prompt(capture_dir: Path, pid: str) -> tuple[list, list, list, list, list, int, int] | None:
    topk_path = capture_dir / f"{pid}.topk.json"
    if not topk_path.exists():
        return None
    tk = json.loads(topk_path.read_text())
    P = int(tk["prompt_tokens"])
    steps = tk["steps"]
    mh, tok, tids, tlp, pip = [], [], [], [], []
    p_min = p_max = -1
    for s in steps:
        p = P + int(s["step"])
        parts = [read_layer_mean(capture_dir, pid, L, p) for L in LAYERS]
        if any(x is None for x in parts):
            continue  # missing a layer dump for this position; skip
        mh.append(np.concatenate(parts).astype(np.float16))   # [12288]
        tok.append(int(s["selected"]["id"]))
        lp = s["top_logprobs"]
        ids = np.full(TOPK, 0, dtype=np.int32)
        lps = np.full(TOPK, -1000.0, dtype=np.float32)  # float16-safe; softmax->0
        for i, e in enumerate(lp[:TOPK]):
            ids[i] = int(e["token"]["id"])
            lps[i] = float(e["logprob"])
        tids.append(ids)
        tlp.append(lps.astype(np.float16))
        pip.append(p)
        p_min = p if p_min < 0 else p_min
        p_max = p
    if not mh:
        return None
    return mh, tok, tids, tlp, pip, p_min, p_max


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cap = Path(args.capture_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text())
    id2meta = {r["prompt_id"]: r for r in manifest["prompts"]}

    splits: dict[str, list] = {}  # any split key (train/eval/lead3/...); grouped per-split into shards
    index_prompts = []
    prompt_ids_seen = []
    # discover captured prompts from topk files
    for topk in sorted(cap.glob("*.topk.json")):
        pid = topk.name.removesuffix(".topk.json")
        meta = id2meta.get(pid)
        if meta is None:
            continue
        res = consolidate_prompt(cap, pid)
        if res is None:
            continue
        mh, tok, tids, tlp, pip, pmin, pmax = res
        idx = len(index_prompts)
        index_prompts.append({
            "prompt_id": pid, "source": meta["source"], "split": meta["split"],
            "prompt_tokens": json.loads(topk.read_text())["prompt_tokens"],
            "n_positions": len(mh), "pos_range": [pmin, pmax],
        })
        prompt_ids_seen.append(pid)
        splits.setdefault(meta["split"], []).append((idx, mh, tok, tids, tlp, pip))

    shard_files = []
    for split, recs in splits.items():
        if not recs:
            continue
        # flatten
        pidx, mh, tok, tids, tlp, pip = [], [], [], [], [], []
        for idx, m, t, ti, tl, pi in recs:
            n = len(m)
            pidx.extend([idx] * n); mh.extend(m); tok.extend(t)
            tids.extend(ti); tlp.extend(tl); pip.extend(pi)
        mh_a = np.stack(mh)                       # [N,12288] f16
        tok_a = np.asarray(tok, dtype=np.int32)
        tids_a = np.stack(tids).astype(np.int32)  # [N,128]
        tlp_a = np.stack(tlp).astype(np.float16)  # [N,128]
        pidx_a = np.asarray(pidx, dtype=np.int32)
        pip_a = np.asarray(pip, dtype=np.int32)
        n = mh_a.shape[0]
        n_shards = max(1, (n + SHARD_POS - 1) // SHARD_POS)
        for si in range(n_shards):
            a = si * SHARD_POS; b = min(n, (si + 1) * SHARD_POS)
            shard = {
                "main_hidden": mh_a[a:b].contiguous() if hasattr(mh_a[a:b], "contiguous") else np.ascontiguousarray(mh_a[a:b]),
                "token_ids": np.ascontiguousarray(tok_a[a:b]),
                "topk_ids": np.ascontiguousarray(tids_a[a:b]),
                "topk_logprobs": np.ascontiguousarray(tlp_a[a:b]),
                "prompt_idx": np.ascontiguousarray(pidx_a[a:b]),
                "pos_in_prompt": np.ascontiguousarray(pip_a[a:b]),
            }
            fn = out / f"{split}_{si:04d}.safetensors"
            save_file(shard, str(fn))
            shard_files.append(str(fn.relative_to(out)))
            print(f"  wrote {fn.name}: positions [{a},{b}) -> {b-a}", flush=True)
        print(f"{split}: {n} positions, {n_shards} shards")

    index = {
        "description": "Stage 2 captured dataset (Activity 3).",
        "schema": {"main_hidden": "[N,12288] f16", "token_ids": "[N] i32",
                   "topk_ids": "[N,128] i32", "topk_logprobs": "[N,128] f16",
                   "prompt_idx": "[N] i32", "pos_in_prompt": "[N] i32"},
        "note": "Per generated position p (P..P+G-1). Drafter input = main_hidden[p]; "
                "predicts token[p+1..p+K] by windowing token_ids (anchor = token_ids at p). "
                "topk_* are the target distribution over token[p]. pos0 prefill excluded.",
        "n_prompts": len(index_prompts),
        "shards": shard_files,
        "prompts": index_prompts,
    }
    (out / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"index.json: {len(index_prompts)} prompts, {len(shard_files)} shards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
