#!/usr/bin/env python3
"""Activity 5 — no-LoRA baseline: frozen drafter p=1 on the Stage 2 eval set.

Adapts the verified numpy oracle (dspark_oracle/measure_acceptance_bundle) to read the
Stage 2 eval shards by writing per-prompt temp bundle dirs (target_selected_tokens.json
+ oracle/main_hidden_pos*.npy + manifest), then running measure_bundle unchanged.
Monkeypatches forward_head to also save x + the numpy draft per anchor, for a direct
torch-head confirmation pass. Reports the frozen-drafter p=1 (the 'before' baseline).

Usage: python3 run_stage2_baseline.py [--n-prompts 15]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent          # issue468/
ORACLE = HERE / "dspark_oracle"
sys.path.insert(0, str(ORACLE))
import measure_acceptance_bundle as mab                # noqa: E402
import forward as fwd                                  # noqa: E402
from forward import BLOCK                              # noqa: E402

SHARDS = HERE / "dspark_train" / "data" / "shards"
TMP = HERE / "dspark_train" / "data" / "_baseline_tmp"
OUT = HERE / "dspark_train" / "data" / "eval_features.npz"

_orig_fh = fwd.forward_head
_saved: list[dict] = []


def _saving_fh(h, anchor_tok, w_head, norm_w, hc_fn, hc_s, hc_b, mw1, mw2, conf, lm_head, temp=1.0):
    out, logits = _orig_fh(h, anchor_tok, w_head, norm_w, hc_fn, hc_s, hc_b, mw1, mw2, conf, lm_head, temp)
    _saved.append({"x": np.ascontiguousarray(h[0]).astype(np.float32), "anchor": int(anchor_tok),
                   "draft0": int(out[1])})
    return out, logits


fwd.forward_head = _saving_fh
mab.forward_head = _saving_fh


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=15)
    args = ap.parse_args()
    gguf_dir = HERE.parent.parent / "ds4" / "gguf"
    model = str((gguf_dir / "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf").resolve())
    dspark = str((gguf_dir / "dspark.gguf").resolve())

    # load eval shard (first eval shard has the bulk; concatenate if multiple)
    from safetensors import safe_open
    idx = json.loads((SHARDS / "index.json").read_text())
    eval_shards = sorted(SHARDS.glob("eval_*.safetensors"))
    mh_all, tok_all, pidx_all, pip_all = [], [], [], []
    for f in eval_shards:
        with safe_open(f, "numpy") as d:
            mh_all.append(d.get_tensor("main_hidden")); tok_all.append(d.get_tensor("token_ids"))
            pidx_all.append(d.get_tensor("prompt_idx")); pip_all.append(d.get_tensor("pos_in_prompt"))
    mh_all = np.concatenate(mh_all); tok_all = np.concatenate(tok_all)
    pidx_all = np.concatenate(pidx_all); pip_all = np.concatenate(pip_all)
    prompts_meta = [p for p in idx["prompts"] if p["split"] == "eval"][: args.n_prompts]
    print(f"eval shard loaded; testing {len(prompts_meta)} eval prompts", flush=True)

    print("loading target + drafter ctx...", flush=True)
    mctx = mab.build_model_ctx(model)
    dctx = mab.build_drafter_ctx(dspark)
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)

    n_anchors = n_p1_match = 0
    for k, pm in enumerate(prompts_meta):
        pidx = pm["prompt_id"]
        p_index = next(j for j, p in enumerate(idx["prompts"]) if p["prompt_id"] == pidx)
        rows = np.where(pidx_all == p_index)[0]
        if len(rows) < BLOCK + 2:
            continue
        order = np.argsort(pip_all[rows])
        rows = rows[order]
        positions = pip_all[rows]
        tokens = tok_all[rows].tolist()
        mh = mh_all[rows]
        first_pos = int(positions[0])
        bdir = TMP / f"p{k}"
        (bdir / "oracle").mkdir(parents=True, exist_ok=True)
        (bdir / "target_selected_tokens.json").write_text(json.dumps(tokens) + "\n")
        for pos, vec in zip(positions, mh):
            np.save(bdir / "oracle" / f"main_hidden_pos{int(pos)}.npy", vec.astype(np.float32))
        manifest = {"prompt_name": pidx, "temperature": 0.0, "seed": 2, "ctx": 8192,
                    "generated_tokens": len(tokens), "block": BLOCK,
                    "measure_steps": len(tokens) - BLOCK - 1, "prompt_tokens": first_pos,
                    "reference_mode": "greedy"}
        (bdir / "bundle_manifest.json").write_text(json.dumps(manifest) + "\n")
        n0 = len(_saved)                       # capture index BEFORE measure_bundle appends
        s = mab.measure_bundle(bdir, mctx, dctx)
        # p=1 from saved drafts vs target next-token
        for j, r in enumerate(s["rows"]):
            d0 = _saved[n0 + j]["draft0"]
            t1 = r["target"][0]
            n_anchors += 1
            n_p1_match += int(d0 == t1)
        if (k + 1) % 5 == 0:
            print(f"  {k+1}/{len(prompts_meta)} prompts: p1 so far {n_p1_match}/{n_anchors} = {n_p1_match/max(1,n_anchors):.4f}", flush=True)

    p1 = n_p1_match / max(1, n_anchors)
    print(f"\nno-LoRA baseline (numpy drafter, Stage 2 eval subset): p1 = {p1:.4f} over {n_anchors} anchors")

    # save x + drafts for the torch-head confirmation pass
    xs = np.stack([r["x"] for r in _saved])
    anchors = np.array([r["anchor"] for r in _saved], dtype=np.int64)
    draft0 = np.array([r["draft0"] for r in _saved], dtype=np.int64)
    np.savez(OUT, x=xs, anchor=anchors, numpy_draft0=draft0)
    print(f"saved {len(_saved)} anchor features -> {OUT.name} for torch-head confirmation")
    shutil.rmtree(TMP)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
