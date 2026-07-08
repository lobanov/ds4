#!/usr/bin/env python3
"""Lead 02 torch confidence gate vs retained numpy oracle.

Purpose:
  Before using the powered 300-prompt corpus through the torch path for Lead 02,
  verify that the torch confidence extraction matches the retained numpy oracle
  closely enough to trust it as a carrier for confidence calibration.

Gate checks per prompt:
  - draft-token agreement must stay at 100% (re-assert the old gate locally)
  - confidence logits/scores from torch are compared against the numpy oracle
    on the same Stage2CaptureStore prompt set.

This script does not alter any canonical summaries. It emits a compact artifact
for the Lead 02 setup gate.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "dspark_train"))
sys.path.insert(0, str(HERE / "dspark_oracle"))
from drafter_body import build_body  # noqa: E402
from drafter_head import build_head  # noqa: E402
from stage2_capture_store import Stage2CaptureStore  # noqa: E402
from measure_acceptance_bundle import build_model_ctx, build_drafter_ctx, measure_bundle  # noqa: E402

DSPARK = "/Users/lobanov/Projects/ds4/gguf/dspark.gguf"
TARGET = "/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
BLOCK = 5


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default=str(HERE / "dspark_train" / "data" / "shards"))
    ap.add_argument("--extra-shards", nargs="*", default=None)
    ap.add_argument("--out", default=str(HERE / "artifacts" / "lead02_torch_confidence_gate" / "summary.json"))
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--pids", nargs="*", default=None)
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float32")
    ap.add_argument("--device", choices=["mps", "cpu"], default="mps")
    ap.add_argument("--max-logit-mae", type=float, default=1e-3)
    ap.add_argument("--max-score-mae", type=float, default=1e-4)
    return ap.parse_args()


def measure_prompt_torch_conf(pid, store, body, head, dev):
    meta = store.prompt_meta(pid)
    pos0 = int(meta["prompt_tokens"])
    tt = store.target_tokens(pid)
    n_pos = len(tt)
    max_step = n_pos - BLOCK - 1
    if max_step < 1:
        return []
    mh_seq = np.stack([store.main_hidden_at(pid, pos0 + s) for s in range(max_step + 1)])
    anchors = [int(tt[s]) for s in range(1, max_step + 1)]
    with torch.no_grad():
        xs = body.forward_prompt(mh_seq, anchors)
        anc_t = torch.tensor(anchors, device=dev, dtype=torch.long)
        rows = []
        CHUNK = 24
        for i in range(0, xs.shape[0], CHUNK):
            out, _base, conf_logits, conf_scores = head(xs[i:i + CHUNK], anc_t[i:i + CHUNK], return_conf=True)
            out = out[:, 1:].cpu().numpy().astype(np.int64)
            conf_logits = conf_logits.cpu().numpy().astype(np.float32)
            conf_scores = conf_scores.cpu().numpy().astype(np.float32)
            for j in range(out.shape[0]):
                step = i + j + 1
                tgt = [int(x) for x in tt[(i + j) + 2:(i + j) + 2 + BLOCK]]
                rows.append({
                    "step": step,
                    "draft": out[j].tolist(),
                    "target": tgt,
                    "confidence_logits": conf_logits[j].tolist(),
                    "confidence_scores": conf_scores[j].tolist(),
                })
    return rows


def mae(a, b):
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    return float(np.mean(np.abs(aa - bb)))


def max_abs(a, b):
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(aa - bb)))


def main() -> int:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shards = [args.shards] + (args.extra_shards or [])

    dev = args.device
    dt = torch.float16 if args.dtype == "float16" else torch.float32
    print(f"building torch body+head on {dev} ({args.dtype})...", flush=True)
    t0 = time.time()
    body = build_body(DSPARK, TARGET, dev, dtype=dt)
    head = build_head(DSPARK, TARGET, dev, lora_rank=0, dtype=dt)
    head.eval()
    print(f"  built in {time.time()-t0:.0f}s", flush=True)

    store = Stage2CaptureStore(shards)
    pids = args.pids or store.prompt_ids()[: args.limit]

    print("building numpy oracle ctx...", flush=True)
    mctx = build_model_ctx(TARGET)
    dctx = build_drafter_ctx(DSPARK)

    checked = []
    failures = []
    for pid in pids:
        trows = measure_prompt_torch_conf(pid, store, body, head, dev)
        nsum = measure_bundle(store=store, prompt_id=pid, mctx=mctx, dctx=dctx,
                              reuse_mode="none", include_confidence=True)
        nrows = nsum["rows"]
        if len(trows) != len(nrows):
            failures.append({"prompt_id": pid, "reason": "row_count_mismatch",
                             "torch_rows": len(trows), "numpy_rows": len(nrows)})
            continue
        step_agree = 0
        token_agree = 0
        ntok = 0
        logit_maes = []
        score_maes = []
        logit_maxes = []
        score_maxes = []
        for tr, nr in zip(trows, nrows):
            if tr["draft"] == nr["draft"]:
                step_agree += 1
            for a, b in zip(tr["draft"], nr["draft"]):
                token_agree += int(a == b)
                ntok += 1
            logit_maes.append(mae(tr["confidence_logits"], nr["confidence_logits"]))
            score_maes.append(mae(tr["confidence_scores"], nr["confidence_scores"]))
            logit_maxes.append(max_abs(tr["confidence_logits"], nr["confidence_logits"]))
            score_maxes.append(max_abs(tr["confidence_scores"], nr["confidence_scores"]))
        row = {
            "prompt_id": pid,
            "n_steps": len(trows),
            "step_draft_agree": round(step_agree / max(1, len(trows)), 6),
            "token_agree": round(token_agree / max(1, ntok), 6),
            "confidence_logit_mae": round(float(np.mean(logit_maes)), 8),
            "confidence_score_mae": round(float(np.mean(score_maes)), 8),
            "confidence_logit_max_abs": round(float(np.max(logit_maxes)), 8),
            "confidence_score_max_abs": round(float(np.max(score_maxes)), 8),
        }
        checked.append(row)
        if not (
            math.isclose(row["step_draft_agree"], 1.0)
            and row["confidence_logit_mae"] <= args.max_logit_mae
            and row["confidence_score_mae"] <= args.max_score_mae
        ):
            failures.append(row)
        print(
            f"  {pid}: step_agree={row['step_draft_agree']:.6f} "
            f"logit_mae={row['confidence_logit_mae']:.8f} score_mae={row['confidence_score_mae']:.8f}",
            flush=True,
        )

    result = {
        "gate": "lead02_torch_confidence_vs_numpy",
        "dtype": args.dtype,
        "device": args.device,
        "pids": pids,
        "thresholds": {
            "step_draft_agree": 1.0,
            "max_logit_mae": args.max_logit_mae,
            "max_score_mae": args.max_score_mae,
        },
        "pass": len(failures) == 0 and len(checked) > 0,
        "checked": checked,
        "failures": failures,
    }
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "pass": result["pass"],
        "n_prompts_checked": len(checked),
        "n_failures": len(failures),
        "out": str(out_path),
    }, indent=2))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
