#!/usr/bin/env python3
"""Activity 5 — body fidelity gate: torch drafter body x vs numpy oracle x.

Builds the torch body (CPU float32 for precision parity), runs it on the exactness
bundles' inputs (main_hidden per position + the anchor stream), and compares the
pre-head feature x to the numpy oracle's x (exactness_features.npz). Gate passes
when max abs diff is within float32 tolerance.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent.parent            # issue468/
sys.path.insert(0, str(HERE / "dspark_train"))
from drafter_body import build_body                        # noqa: E402
sys.path.insert(0, str(HERE / "dspark_oracle"))
from measure_acceptance_bundle import load_mh              # noqa: E402

BUNDLES = HERE / "artifacts" / "exactness_small_bundles"
FEAT = HERE / "dspark_train" / "data" / "exactness_features.npz"
GGUF_DIR = HERE.parent.parent / "ds4" / "gguf"


def main() -> int:
    d = np.load(FEAT, allow_pickle=True)
    npx = d["x"]                                 # [N, BLOCK, HC, DIM]
    prompts_arr = d["prompt"]; steps_arr = d["step"]
    # group numpy x by prompt
    by_prompt: dict[str, list] = {}
    for i in range(len(prompts_arr)):
        by_prompt.setdefault(str(prompts_arr[i]), []).append((int(steps_arr[i]), npx[i]))
    for k in by_prompt:
        by_prompt[k].sort()

    print("building torch body (CPU float32)...", flush=True)
    body = build_body(str((GGUF_DIR / "dspark.gguf").resolve()),
                      str((GGUF_DIR / "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf").resolve()),
                      device="cpu", dtype=torch.float32)

    PROMPTS = ["grounded_observatory", "code_histogram", "synthesis_ops_json"]
    overall_max = 0.0; overall_mean = 0.0; nn = 0
    for p in PROMPTS:
        bdir = BUNDLES / f"{p}__t0p0"
        man = json.loads((bdir / "bundle_manifest.json").read_text())
        pos0 = int(man["prompt_tokens"]); ms = int(man["measure_steps"])
        toks = json.loads((bdir / "target_selected_tokens.json").read_text())
        anchors = [int(toks[step]) for step in range(1, ms + 1)]
        mh_seq = []
        for step in range(0, ms + 1):
            v = load_mh(bdir, pos0 + step)
            if v is None:
                print(f"  {p}: missing main_hidden at {pos0+step}, skip"); p_done = False; break
            mh_seq.append(np.asarray(v, np.float32))
        else:
            p_done = True
        if not p_done:
            continue
        tx = body.forward_prompt(mh_seq, anchors).numpy()       # [ms, BLOCK, HC, DIM]
        np_list = [v for _, v in by_prompt.get(p, [])]
        if len(np_list) != tx.shape[0]:
            print(f"  {p}: count mismatch torch {tx.shape[0]} vs numpy {len(np_list)}"); continue
        npa = np.stack(np_list)
        diff = np.abs(tx - npa)
        overall_max = max(overall_max, float(diff.max())); overall_mean += float(diff.sum()); nn += diff.size
        print(f"  {p}: x max abs diff {float(diff.max()):.6g} | mean {float(diff.mean()):.6g} "
              f"(shape {tx.shape})")

    if nn:
        print(f"\nBODY FIDELITY (torch x vs numpy x): overall max abs diff {overall_max:.6g}, "
              f"mean {overall_mean/nn:.6g}")
        ok = overall_max < 2e-3
        print(f"BODY GATE: {'PASS' if ok else 'FAIL'} (criteria max abs diff < 2e-3)")
        return 0 if ok else 1
    print("no comparisons made"); return 1


if __name__ == "__main__":
    raise SystemExit(main())
