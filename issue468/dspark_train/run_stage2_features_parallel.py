#!/usr/bin/env python3
"""Activity 5/6/7/9 — parallel numpy-drafter feature extraction (RESEARCH INSTRUMENTATION).

Extracts, for a split's prompts, the frozen drafter pre-head features x + draft0 + the
p=1 target, by running the VERIFIED numpy oracle (dspark_oracle) per prompt in parallel
(multiprocessing across prompts; each worker loads the model once via a pool initializer).
Reuses measure_acceptance_bundle unchanged per prompt (temp bundle dir) -> no forward drift.

Output: issue468/dspark_train/data/{split}_features.npz with
  x [N,BLOCK,HC,DIM] f32, anchor [N] i64, draft0 [N] i64 (numpy drafter p=1 pick),
  target_p1 [N] i64 (next token), prompt_idx [N] i32.
Also reports the frozen-drafter p=1 (the no-LoRA baseline for eval).

Usage:
  python3 run_stage2_features_parallel.py --split eval   --n-prompts 0 --workers 8
  python3 run_stage2_features_parallel.py --split train  --n-prompts 0 --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent          # issue468/
ORACLE = HERE / "dspark_oracle"
SHARDS = HERE / "dspark_train" / "data" / "shards"
OUTDIR = HERE / "dspark_train" / "data"
GGUF_DIR = HERE.parent.parent / "ds4" / "gguf"
MODEL = str((GGUF_DIR / "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf").resolve())
DSPARK = str((GGUF_DIR / "dspark.gguf").resolve())
BLOCK = 5

_G: dict = {}


def _init_worker(tmpbase: str):
    """Per-worker: load model once, monkeypatch forward_head to save x+draft0."""
    sys.path.insert(0, str(ORACLE))
    import measure_acceptance_bundle as mab
    import forward as fwd
    _G["mab"] = mab
    _G["mctx"] = mab.build_model_ctx(MODEL)
    _G["dctx"] = mab.build_drafter_ctx(DSPARK)
    _G["saved"] = []
    _G["tmpbase"] = Path(tmpbase)
    _orig = fwd.forward_head

    def _saving(h, anchor_tok, *a, **k):
        out, logits = _orig(h, anchor_tok, *a, **k)
        _G["saved"].append((np.ascontiguousarray(h[0]).astype(np.float32), int(anchor_tok), int(out[1])))
        return out, logits

    fwd.forward_head = _saving
    mab.forward_head = _saving


def _worker(args):
    """Process one prompt: write temp bundle, run measure_bundle, return features + p1."""
    pid, p_index, positions, tokens, mh = args
    mab = _G["mab"]; bdir = _G["tmpbase"] / f"{pid}_p{os.getpid()}"
    (bdir / "oracle").mkdir(parents=True, exist_ok=True)
    (bdir / "target_selected_tokens.json").write_text(json.dumps(tokens) + "\n")
    for pos, vec in zip(positions, mh):
        np.save(bdir / "oracle" / f"main_hidden_pos{int(pos)}.npy", vec.astype(np.float32))
    manifest = {"prompt_name": pid, "temperature": 0.0, "seed": 2, "ctx": 8192,
                "generated_tokens": len(tokens), "block": BLOCK,
                "measure_steps": len(tokens) - BLOCK - 1, "prompt_tokens": int(positions[0]),
                "reference_mode": "greedy"}
    (bdir / "bundle_manifest.json").write_text(json.dumps(manifest) + "\n")
    n0 = len(_G["saved"])
    s = mab.measure_bundle(bdir, _G["mctx"], _G["dctx"])
    shutil.rmtree(bdir)
    rows = s["rows"]
    feats = _G["saved"][n0:]
    # trim saved list to bound memory
    del _G["saved"][:]
    xs = np.stack([f[0] for f in feats]) if feats else np.zeros((0, BLOCK, 4, 4096), np.float32)
    anchors = np.array([f[1] for f in feats], dtype=np.int64)
    draft0 = np.array([f[2] for f in feats], dtype=np.int64)
    target_p1 = np.array([r["target"][0] for r in rows], dtype=np.int64)
    n_match = int((draft0 == target_p1).sum()) if len(draft0) else 0
    return pid, p_index, xs, anchors, draft0, target_p1, n_match, len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="eval", choices=["train", "eval"])
    ap.add_argument("--n-prompts", type=int, default=0, help="0 = all")
    ap.add_argument("--workers", type=int, default=min(8, max(1, (os.cpu_count() or 4) - 2)))
    args = ap.parse_args()

    from safetensors import safe_open
    idx = json.loads((SHARDS / "index.json").read_text())
    shards = sorted(SHARDS.glob(f"{args.split}_*.safetensors"))
    mh_all, tok_all, pidx_all, pip_all = [], [], [], []
    for f in shards:
        with safe_open(f, "numpy") as d:
            mh_all.append(d.get_tensor("main_hidden")); tok_all.append(d.get_tensor("token_ids"))
            pidx_all.append(d.get_tensor("prompt_idx")); pip_all.append(d.get_tensor("pos_in_prompt"))
    mh_all = np.concatenate(mh_all); tok_all = np.concatenate(tok_all)
    pidx_all = np.concatenate(pidx_all); pip_all = np.concatenate(pip_all)
    prompts_meta = [p for p in idx["prompts"] if p["split"] == args.split]
    if args.n_prompts:
        prompts_meta = prompts_meta[: args.n_prompts]
    print(f"split={args.split}: {len(prompts_meta)} prompts, {args.workers} workers", flush=True)

    # build per-prompt work items
    work = []
    for pm in prompts_meta:
        p_index = next(j for j, p in enumerate(idx["prompts"]) if p["prompt_id"] == pm["prompt_id"])
        rows = np.where(pidx_all == p_index)[0]
        if len(rows) < BLOCK + 2:
            continue
        rows = rows[np.argsort(pip_all[rows])]
        work.append((pm["prompt_id"], p_index, pip_all[rows], tok_all[rows].tolist(), mh_all[rows]))

    tmpbase = OUTDIR / f"_featmp_{args.split}"
    if tmpbase.exists():
        shutil.rmtree(tmpbase)
    tmpbase.mkdir(parents=True)

    import multiprocessing as mp
    results = []
    with mp.Pool(args.workers, initializer=_init_worker, initargs=(str(tmpbase),)) as pool:
        for i, res in enumerate(pool.imap_unordered(_worker, work, chunksize=1)):
            results.append(res)
            if (i + 1) % max(1, len(work) // 10) == 0:
                print(f"  {i+1}/{len(work)} prompts done", flush=True)
    shutil.rmtree(tmpbase)

    # order results by prompt_index and concatenate
    results.sort(key=lambda r: r[1])
    pmap = {pm["prompt_id"]: pm for pm in prompts_meta}
    xs, anc, d0, tp1, pidx_arr = [], [], [], [], []
    n_match = n_anc = 0
    for pid, p_index, x, a, d, t, m, n in results:
        if len(x):
            xs.append(x); anc.append(a); d0.append(d); tp1.append(t)
            pidx_arr.append(np.full(len(x), p_index, dtype=np.int32))
        n_match += m; n_anc += n
    p1 = n_match / max(1, n_anc)
    out = OUTDIR / f"{args.split}_features.npz"
    np.savez(out, x=np.concatenate(xs), anchor=np.concatenate(anc), draft0=np.concatenate(d0),
             target_p1=np.concatenate(tp1), prompt_idx=np.concatenate(pidx_arr))
    print(f"\n{args.split}: {n_anc} anchors, frozen-drafter p1 = {p1:.4f}  -> {out.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
