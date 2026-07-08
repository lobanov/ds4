#!/usr/bin/env python3
"""Activity 5 — torch-drafter acceptance check (the meaningful body gate).

Runs torch body (CPU float32) + torch head over the exactness corpus, measures p=1
(draft[0] == target next token), and compares to the numpy oracle's 0.8125. The body
is numerically chaotic vs numpy BLAS (x diverges in low bits), so the right gate is
acceptance-level parity, not bit-exact x. PASS if torch-drafter p=1 is within ~2 pp
of numpy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "dspark_train"))
from drafter_body import build_body            # noqa
from drafter_head import build_head            # noqa
sys.path.insert(0, str(HERE / "dspark_oracle"))
from measure_acceptance_bundle import load_mh  # noqa
from forward import BLOCK                      # noqa

GGUF = HERE.parent.parent / "ds4" / "gguf"
DSPARK = str((GGUF / "dspark.gguf").resolve())
TARGET = str((GGUF / "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf").resolve())
BUNDLES = HERE / "artifacts" / "exactness_small_bundles"
PROMPTS = sorted(p.name.rsplit("__", 1)[0] for p in BUNDLES.glob("*__t0p0"))


def main():
    body = build_body(DSPARK, TARGET, device="cpu", dtype=torch.float32)
    head = build_head(DSPARK, TARGET, device="cpu", lora_rank=0, dtype=torch.float32)
    head.eval()
    n = 0; match = 0
    with torch.no_grad():
        for p in PROMPTS:
            bdir = BUNDLES / f"{p}__t0p0"
            man = json.loads((bdir / "bundle_manifest.json").read_text())
            pos0 = int(man["prompt_tokens"]); ms = int(man["measure_steps"])
            toks = json.loads((bdir / "target_selected_tokens.json").read_text())
            anchors = [int(toks[step]) for step in range(1, ms + 1)]
            mh_seq = [np.asarray(load_mh(bdir, pos0 + i), np.float32) for i in range(0, ms + 1)]
            x = body.forward_prompt(mh_seq, anchors)            # [ms, BLOCK, HC, DIM]
            anc = torch.tensor(anchors, dtype=torch.long)
            out, _ = head(x, anc)                               # [ms, BLOCK+1]
            draft0 = out[:, 1].numpy()
            for j, step in enumerate(range(1, ms + 1)):
                n += 1; match += int(draft0[j] == int(toks[step + 1]))
    p1 = match / n
    print(f"torch-drafter (body+head, CPU f32) p=1 on exactness: {p1:.4f} over {n} anchors")
    print(f"numpy baseline p=1: 0.8125; delta {abs(p1 - 0.8125)*100:.2f} pp")
    ok = abs(p1 - 0.8125) < 0.02
    print(f"ACCEPTANCE GATE: {'PASS' if ok else 'FAIL'} (|delta| < 2 pp)")


if __name__ == "__main__":
    main()
