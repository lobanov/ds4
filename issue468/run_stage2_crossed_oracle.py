#!/usr/bin/env python3
"""Activity 4 — crossed hidden/label oracle diagnostic (RESEARCH INSTRUMENTATION).

Reuses the existing IQ2XXS (baseline) and Q4-tap exactness bundles (Stage 1) to
separate INPUT-shift (hidden-state/representation) from LABEL-shift (argmax drift)
at the drafter's p=1 prediction, via a 2x2 crossed acceptance:

   acc(input X, label Y) = P( drafter_draft_X[0] == target_Y[0] )   X,Y in {IQ2,Q4}

  - input main effect  ~ [acc(IQ2,*) - acc(Q4,*)]  -> representation/hidden-state side
  - label main effect  ~ [acc(*,IQ2) - acc(*,Q4)]  -> argmax-drift side

Computed on (a) all (prompt,step) cells and (b) prefix-aligned cells (IQ2 and Q4
generated the same token prefix through the anchor -> same context, so hidden
states differ ONLY by precision -> clean input vs label isolation). Alignment at
step s requires target_tokens_IQ2[0..s] == target_tokens_Q4[0..s].
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE / "artifacts" / "exactness_small_bundles"
Q4 = HERE / "artifacts" / "exactness_small_bundles_q4tap"
PROMPTS = ["code_histogram", "code_sort_pairs", "code_topk", "grounded_archive",
           "grounded_observatory", "grounded_repair", "mixed_exactness_smoke",
           "synthesis_incident_json", "synthesis_ops_json", "synthesis_timeline_json"]


def load(bundle_root: Path, p: str) -> tuple[dict, list[int]]:
    s = json.loads((bundle_root / f"{p}__t0p0" / "oracle" / "acceptance_summary.json").read_text())
    sel = json.loads((bundle_root / f"{p}__t0p0" / "target_selected_tokens.json").read_text())
    return s, sel


def main() -> int:
    cells = []  # (prompt, step, d_iq2, t_iq2, d_q4, t_q4, aligned)
    for p in PROMPTS:
        si, seli = load(BASE, p)
        sq, selq = load(Q4, p)
        ri = {r["step"]: r for r in si["rows"]}
        rq = {r["step"]: r for r in sq["rows"]}
        for step in ri:
            if step not in rq:
                continue
            d_iq2 = ri[step]["draft"][0]; t_iq2 = ri[step]["target"][0]
            d_q4 = rq[step]["draft"][0]; t_q4 = rq[step]["target"][0]
            # aligned iff generated prefix through the anchor (index step) matches
            aligned = (seli[: step + 1] == selq[: step + 1])
            cells.append((p, step, d_iq2, t_iq2, d_q4, t_q4, aligned))

    def accs(subset):
        n = len(subset)
        if not n:
            return None
        a = {  # acc(input, label)
            "II": sum(1 for c in subset if c[2] == c[3]) / n,  # drafter IQ2 hidden vs IQ2 tok
            "IQ": sum(1 for c in subset if c[2] == c[5]) / n,  # IQ2 hidden vs Q4 tok
            "QI": sum(1 for c in subset if c[4] == c[3]) / n,  # Q4 hidden vs IQ2 tok
            "QQ": sum(1 for c in subset if c[4] == c[5]) / n,  # Q4 hidden vs Q4 tok
        }
        # main effects (positive = that side helps acceptance)
        a["input_effect"] = 0.5 * ((a["II"] + a["IQ"]) - (a["QI"] + a["QQ"]))  # IQ2 input better than Q4?
        a["label_effect"] = 0.5 * ((a["II"] + a["QI"]) - (a["IQ"] + a["QQ"]))  # IQ2 label better than Q4?
        a["n"] = n
        return a

    aligned_cells = [c for c in cells if c[6]]
    res = {"all_cells": accs(cells), "aligned_cells": accs(aligned_cells),
           "n_aligned": len(aligned_cells), "n_total": len(cells)}

    # direct shift magnitudes on aligned cells
    if aligned_cells:
        na = len(aligned_cells)
        res["input_shift_rate"] = sum(1 for c in aligned_cells if c[2] != c[4]) / na  # drafter pred flips IQ2->Q4
        res["label_shift_rate"] = sum(1 for c in aligned_cells if c[3] != c[5]) / na  # target argmax flips

    print(json.dumps(res, indent=2))
    (HERE / "artifacts" / "stage2_crossed_oracle").mkdir(parents=True, exist_ok=True)
    (HERE / "artifacts" / "stage2_crossed_oracle" / "summary.json").write_text(json.dumps(res, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
