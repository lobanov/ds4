#!/usr/bin/env python3
"""Activity 5 — extract frozen pre-head features x + the numpy forward_head reference.

RESEARCH INSTRUMENTATION. Runs the VERIFIED numpy drafter oracle (dspark_oracle) over
the temp=0 exactness bundles, monkeypatching forward_head so it saves, per anchor:
  x            [BLOCK, HC, DIM]   -- the pre-head feature (drafter body output)
  anchor       int                -- the anchor token
  out          [BLOCK+1]          -- numpy forward_head output_ids (argmax rollout)
  base_logits  [BLOCK, VOCAB]     -- numpy forward_head base logits (pre-markov)
  target_block [BLOCK]            -- the K target tokens (for p=1 / loss reference)
Saved to issue468/dspark_train/data/exactness_features.npz for the torch-head fidelity gate.

Monkeypatching (not re-implementing) the forward guarantees x + the numpy head output
come from the identical verified path used in Stage 0/1.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent   # issue468/
ORACLE = HERE / "dspark_oracle"
sys.path.insert(0, str(ORACLE))

import measure_acceptance_bundle as mab           # noqa: E402
import forward as fwd                             # noqa: E402
from forward import BLOCK                          # noqa: E402

BUNDLES = HERE / "artifacts" / "exactness_small_bundles"
OUT = HERE / "dspark_train" / "data" / "exactness_features.npz"

_orig_fh = fwd.forward_head
_saved: list[dict] = []


def _saving_fh(h, anchor_tok, w_head, norm_w, hc_fn, hc_s, hc_b, mw1, mw2, conf, lm_head, temp=1.0):
    out, logits = _orig_fh(h, anchor_tok, w_head, norm_w, hc_fn, hc_s, hc_b, mw1, mw2, conf, lm_head, temp)
    _saved.append({
        "x": np.ascontiguousarray(h[0]).astype(np.float32),     # [BLOCK,HC,DIM]
        "anchor": int(anchor_tok),
        "out": out.copy().astype(np.int64),                      # [BLOCK+1]
        "base_logits": logits.copy().astype(np.float32),        # [BLOCK,VOCAB]
    })
    return out, logits


# patch both the forward module and the name measure_acceptance_bundle bound
fwd.forward_head = _saving_fh
mab.forward_head = _saving_fh


def main() -> int:
    gguf_dir = HERE.parent.parent / "ds4" / "gguf"   # sibling ds4 engine repo
    model = str((gguf_dir / "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf").resolve())
    dspark = str((gguf_dir / "dspark.gguf").resolve())
    print("loading target + drafter ctx...", file=sys.stderr, flush=True)
    mctx = mab.build_model_ctx(model)
    dctx = mab.build_drafter_ctx(dspark)
    prompts = sorted(p.name.rsplit("__", 1)[0] for p in BUNDLES.glob("*__t0p0"))
    for p in prompts:
        s = mab.measure_bundle(BUNDLES / f"{p}__t0p0", mctx, dctx)
        # attach target_block per saved anchor (rows are in step order)
        rows = s["rows"]
        n_before = len(_saved) - len(rows)
        for j, r in enumerate(rows):
            _saved[n_before + j]["prompt"] = p
            _saved[n_before + j]["step"] = r["step"]
            _saved[n_before + j]["target_block"] = np.array(r["target"], dtype=np.int64)
        print(f"  {p}: {len(rows)} anchors", file=sys.stderr, flush=True)

    n = len(_saved)
    xs = np.stack([r["x"] for r in _saved])                       # [N,BLOCK,HC,DIM]
    anchors = np.array([r["anchor"] for r in _saved], dtype=np.int64)
    outs = np.stack([r["out"] for r in _saved])                   # [N,BLOCK+1]
    base = np.stack([r["base_logits"] for r in _saved])           # [N,BLOCK,VOCAB]
    targets = np.stack([r["target_block"] for r in _saved])       # [N,BLOCK]
    prompts_arr = np.array([r["prompt"] for r in _saved])
    steps = np.array([r["step"] for r in _saved], dtype=np.int64)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, x=xs, anchor=anchors, out=outs, base_logits=base, target_block=targets,
             prompt=prompts_arr, step=steps)
    print(f"saved {n} anchors to {OUT}: x{xs.shape} base_logits{base.shape}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
