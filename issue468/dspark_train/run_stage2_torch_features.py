#!/usr/bin/env python3
"""Activity 6/7/9 prep — extract pre-head features x via the torch MPS drafter body.

Self-consistent feature source for head training/eval. For each prompt in a split,
runs DrafterBody.forward_prompt over its generated positions, collecting per anchor:
  x [BLOCK,HC,DIM] f16   (pre-head feature)
  anchor i64             (token at the anchor position)
  target_p1 i64          (the next token = the p=1 target)
Alignment: anchor = token_ids[s], target_p1 = token_ids[s+1], s=1..G-2 (n=G-2 anchors);
mh_seq = main_hidden[0:G-1]. Saves {split}_torch_features.safetensors.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "dspark_train"))
from drafter_body import build_body            # noqa
GGUF = HERE.parent.parent / "ds4" / "gguf"
DSPARK = str((GGUF / "dspark.gguf").resolve())
TARGET = str((GGUF / "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf").resolve())
SHARDS = HERE / "dspark_train" / "data" / "shards"
OUTDIR = HERE / "dspark_train" / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train,eval")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0, help="skip first N prompts of each split")
    ap.add_argument("--count", type=int, default=0, help="process at most C prompts (0=all); saves a _part file")
    args = ap.parse_args()
    print("building torch body (MPS)...", flush=True)
    body = build_body(DSPARK, TARGET, device="mps", dtype=torch.float32)
    idx = json.loads((SHARDS / "index.json").read_text())
    from safetensors import safe_open
    for split in args.splits.split(","):
        shards = sorted(SHARDS.glob(f"{split}_*.safetensors"))
        mh_all, tok_all, pidx_all, pip_all = [], [], [], []
        for f in shards:
            with safe_open(f, "numpy") as d:
                mh_all.append(d.get_tensor("main_hidden")); tok_all.append(d.get_tensor("token_ids"))
                pidx_all.append(d.get_tensor("prompt_idx")); pip_all.append(d.get_tensor("pos_in_prompt"))
        mh_all = np.concatenate(mh_all); tok_all = np.concatenate(tok_all)
        pidx_all = np.concatenate(pidx_all); pip_all = np.concatenate(pip_all)
        prompts_meta = [p for p in idx["prompts"] if p["split"] == split]
        if args.limit:
            prompts_meta = prompts_meta[: args.limit]
        prompts_meta = prompts_meta[args.offset: args.offset + args.count] if args.count else prompts_meta[args.offset:]
        part_tag = f"_part{args.offset}" if args.count else ""
        xs, anc, tp1, pidxv = [], [], [], []
        t0 = time.time(); n_done = 0
        for pm in prompts_meta:
            p_index = next(j for j, p in enumerate(idx["prompts"]) if p["prompt_id"] == pm["prompt_id"])
            rows = np.where(pidx_all == p_index)[0]
            rows = rows[np.argsort(pip_all[rows])]
            mh = mh_all[rows]; toks = tok_all[rows]
            G = len(toks)
            if G < 4:
                continue
            n = G - 2
            mh_seq = [np.asarray(mh[i], np.float32) for i in range(n + 1)]
            anchors = [int(toks[s]) for s in range(1, n + 1)]
            tgt = [int(toks[s + 1]) for s in range(1, n + 1)]
            with torch.no_grad():
                xv = body.forward_prompt(mh_seq, anchors).cpu().numpy().astype(np.float16)  # [n,BLOCK,HC,DIM]
            xs.append(xv); anc.append(np.array(anchors, np.int64)); tp1.append(np.array(tgt, np.int64))
            pidxv.append(np.full(n, p_index, np.int32))
            n_done += 1
            if n_done % 20 == 0:
                el = time.time() - t0
                print(f"  {split}: {n_done}/{len(prompts_meta)} prompts, {el:.0f}s ({el/n_done:.1f}s/prompt)", flush=True)
        out = {
            "x": np.concatenate(xs).astype(np.float16),
            "anchor": np.concatenate(anc).astype(np.int64),
            "target_p1": np.concatenate(tp1).astype(np.int64),
            "prompt_idx": np.concatenate(pidxv).astype(np.int32),
        }
        fn = OUTDIR / f"{split}_torch_features{part_tag}.safetensors"
        save_file({k: np.ascontiguousarray(v) for k, v in out.items()}, str(fn))
        print(f"{split}: {out['x'].shape[0]} anchors -> {fn.name} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
