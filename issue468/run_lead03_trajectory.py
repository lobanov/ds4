#!/usr/bin/env python3
"""Lead 03 task-4 — realistic-trajectory partition from per-prompt checkpoints.

Reads the per_prompt checkpoints (drafts per step) + the Stage2CaptureStore (target
streams), and partitions cycles by anchor type:
  predicted-anchor : the prior step's draft[0] MATCHED this step's anchor (the drafter
                     had predicted the anchor token) -- the easy case.
  correction-anchor: the prior step's draft[0] MISSED -> this step's anchor is a
                     correction token the drafter just mispredicted (the realistic-
                     trajectory case the model flags as plausibly lower acceptance).

Reports E[a|K]/S(K) for both populations + the trajectory-weighted mixture (== overall).

Indexing (verified against measure_acceptance_bundle semantics): drafts[i] (0-indexed)
is measure step i+1; it predicts target_tokens[i+2..i+1+BLOCK]. The anchor at step i+1
is target_tokens[i+1]; the prior step (drafts[i-1]) predicted target_tokens[i+1] at its
position 0. So step i is correction-anchored iff (i>=1 and drafts[i-1][0] != target_tokens[i+1]).
Sanity: predicted fraction should ~= overall p1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BLOCK = 5


def match_prefix(draft, tgt):
    p = 0
    for d, t in zip(draft, tgt):
        if d == t:
            p += 1
        else:
            break
    return p


def pop_stats(prefixes):
    """prefixes: list of per-cycle prefix lengths (0..BLOCK). Return E[a|k], S(k), n."""
    n = len(prefixes)
    if n == 0:
        return {"n": 0, "E_ak": {}, "S_k": {}}
    h = np.zeros(BLOCK + 1, dtype=np.int64)
    for p in prefixes:
        h[p] += 1
    tot = h.sum()
    S = [int(h[j:].sum()) for j in range(1, BLOCK + 1)]  # S(k)=#(prefix>=k)
    Eak = {f"E[a|{k}]": round(sum(S[:k]) / tot, 5) for k in range(1, BLOCK + 1)}
    Sk = {f"S({k})": round(S[k-1] / tot, 5) for k in range(1, BLOCK + 1)}
    return {"n": n, "E_ak": Eak, "S_k": Sk}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-prompt-dir", required=True)
    ap.add_argument("--shards", default=str(HERE / "dspark_train" / "data" / "shards"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys_path = __import__("sys")
    sys_path.path.insert(0, str(HERE / "dspark_oracle"))
    from stage2_capture_store import Stage2CaptureStore
    store = Stage2CaptureStore(args.shards)

    ppdir = Path(args.per_prompt_dir)
    files = sorted(ppdir.glob("*.json"))
    predicted, correction = [], []
    n_first = 0  # step-1 cycles (no prior step) -- excluded from the partition
    overall = []
    pred_p1_check = 0
    n_steps_seen = 0
    for f in files:
        c = json.loads(f.read_text())
        pid = c["prompt_id"]
        drafts = c.get("drafts")
        if not drafts:
            continue
        tt = store.target_tokens(pid)
        max_step = len(drafts)
        for i in range(max_step):
            tgt = [int(x) for x in tt[i + 2:i + 2 + BLOCK]]
            if len(tgt) < BLOCK:
                continue
            prefix = match_prefix(drafts[i], tgt)
            overall.append(prefix)
            n_steps_seen += 1
            if i == 0:
                n_first += 1
                continue  # no prior step
            # prior step drafts[i-1][0] predicted target_tokens[(i-1)+2]=tt[i+1] (this step's anchor)
            prior_predicted_anchor = (int(drafts[i-1][0]) == int(tt[i + 1]))
            if prior_predicted_anchor:
                predicted.append(prefix)
                pred_p1_check += 1
            else:
                correction.append(prefix)

    out = {
        "n_prompts": len(files), "n_cycles_total": len(overall), "n_first_step_excluded": n_first,
        "predicted_anchor": pop_stats(predicted),
        "correction_anchor": pop_stats(correction),
        "overall_mixture": pop_stats(overall),
        "sanity_predicted_fraction_of_partitioned": (
            round(len(predicted) / max(1, len(predicted) + len(correction)), 4)),
        "note": "predicted_anchor = prior step's draft[0] matched this step's anchor; "
                "correction_anchor = prior step missed. Mixture == overall. Step-1 cycles "
                "(no prior) excluded from the partition but counted in the mixture. "
                "Scope: measures anchor-token difficulty, NOT drafter-state pollution from "
                "rejected drafts (Lead 05).",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
