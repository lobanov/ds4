#!/usr/bin/env python3
"""Lead 02 — confidence measurement harness on the powered Stage2/Lead3 corpus.

Collects per-step DSpark confidence outputs from the torch/MPS carrier and
computes compact discrimination/calibration summaries for the confidence head:

- per-position conditional labels y_k = 1[prefix >= k]
- per-position raw confidence AUC / ECE
- per-position cumulative-prefix AUC / ECE using prod_{i<=k} c_i

This is the first real measurement harness for Lead 02. It does not yet perform
STS; instead it emits the trusted per-prompt confidence rows plus aggregate
pre-calibration diagnostics that STS will consume.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE / "dspark_train"))
sys.path.insert(0, str(HERE / "dspark_oracle"))
from drafter_body import build_body  # noqa: E402
from drafter_head import build_head  # noqa: E402
from stage2_capture_store import Stage2CaptureStore  # noqa: E402

DSPARK = "/Users/lobanov/Projects/ds4/gguf/dspark.gguf"
TARGET = "/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
BLOCK = 5


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["measure", "aggregate"], required=True)
    ap.add_argument("--shards", default=str(HERE / "dspark_train" / "data" / "shards"))
    ap.add_argument("--extra-shards", nargs="*", default=None)
    ap.add_argument("--out", default=str(HERE / "artifacts" / "lead02_confidence_measure"))
    ap.add_argument("--limit", type=int, default=0, help="cap prompts (0 = all)")
    ap.add_argument("--pids", nargs="*", default=None)
    ap.add_argument("--only-split", default=None)
    ap.add_argument("--only-source", default=None)
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float32")
    ap.add_argument("--device", choices=["mps", "cpu"], default="mps")
    ap.add_argument("--ece-bins", type=int, default=15)
    return ap.parse_args()


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def auc_binary(y_true, y_score) -> float:
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(y_score, dtype=np.float64)
    pos = int(y.sum())
    neg = int((1 - y).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    s_sorted = s[order]
    i = 0
    while i < len(s_sorted):
        j = i + 1
        while j < len(s_sorted) and s_sorted[j] == s_sorted[i]:
            j += 1
        if j - i > 1:
            avg = (i + 1 + j) / 2.0
            ranks[order[i:j]] = avg
        i = j
    rank_sum_pos = float(ranks[y == 1].sum())
    return (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def ece_binary(y_true, y_prob, n_bins=15) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_prob, dtype=np.float64)
    if len(y) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        conf = float(p[mask].mean())
        acc = float(y[mask].mean())
        ece += (n / total) * abs(conf - acc)
    return ece


def measure_prompt_torch_conf(pid, store, body, head, dev):
    meta = store.prompt_meta(pid)
    pos0 = int(meta["prompt_tokens"])
    tt = store.target_tokens(pid)
    max_step = len(tt) - BLOCK - 1
    if max_step < 1:
        return {"prompt_id": pid, "source": meta.get("source"), "split": meta.get("split"), "n_steps": 0, "rows": []}
    mh_seq = np.stack([store.main_hidden_at(pid, pos0 + s) for s in range(max_step + 1)])
    anchors = [int(tt[s]) for s in range(1, max_step + 1)]
    rows = []
    with torch.no_grad():
        xs = body.forward_prompt(mh_seq, anchors)
        anc_t = torch.tensor(anchors, device=dev, dtype=torch.long)
        CHUNK = 24
        for i in range(0, xs.shape[0], CHUNK):
            out, _base, conf_logits, conf_scores = head(xs[i:i + CHUNK], anc_t[i:i + CHUNK], return_conf=True)
            out = out[:, 1:].cpu().numpy().astype(np.int64)
            conf_logits = conf_logits.cpu().numpy().astype(np.float32)
            conf_scores = conf_scores.cpu().numpy().astype(np.float32)
            for j in range(out.shape[0]):
                step = i + j + 1
                draft = out[j].tolist()
                target = [int(x) for x in tt[(i + j) + 2:(i + j) + 2 + BLOCK]]
                prefix = 0
                for d, t in zip(draft, target):
                    if d == t:
                        prefix += 1
                    else:
                        break
                rows.append({
                    "step": step,
                    "draft": draft,
                    "target": target,
                    "prefix": prefix,
                    "confidence_logits": [float(v) for v in conf_logits[j].tolist()],
                    "confidence_scores": [float(v) for v in conf_scores[j].tolist()],
                })
    return {
        "prompt_id": pid,
        "source": meta.get("source"),
        "split": meta.get("split"),
        "prompt_tokens": pos0,
        "n_steps": len(rows),
        "rows": rows,
    }


def aggregate_prompt_rows(rows: list[dict], n_bins: int) -> dict:
    pos_scores = {k: [] for k in range(1, BLOCK + 1)}
    pos_labels = {k: [] for k in range(1, BLOCK + 1)}
    cum_scores = {k: [] for k in range(1, BLOCK + 1)}
    cum_labels = {k: [] for k in range(1, BLOCK + 1)}
    source_counts = {}
    split_counts = {}
    per_prompt = []
    for item in rows:
        source_counts[item.get("source", "?")] = source_counts.get(item.get("source", "?"), 0) + 1
        split_counts[item.get("split", "?")] = split_counts.get(item.get("split", "?"), 0) + 1
        prompt_summary = {
            "prompt_id": item["prompt_id"],
            "source": item.get("source"),
            "split": item.get("split"),
            "n_steps": item["n_steps"],
        }
        per_prompt.append(prompt_summary)
        for row in item["rows"]:
            prefix = int(row["prefix"])
            cs = np.asarray(row["confidence_scores"], dtype=np.float64)
            cum = np.cumprod(cs)
            for k in range(1, BLOCK + 1):
                lab = 1 if prefix >= k else 0
                pos_scores[k].append(float(cs[k - 1]))
                pos_labels[k].append(lab)
                cum_scores[k].append(float(cum[k - 1]))
                cum_labels[k].append(lab)
    by_position = {}
    for k in range(1, BLOCK + 1):
        y = pos_labels[k]
        p = pos_scores[k]
        cy = cum_labels[k]
        cp = cum_scores[k]
        by_position[f"p{k}"] = {
            "n": len(y),
            "positive_rate": round(float(np.mean(y)) if y else float("nan"), 6),
            "raw_auc": round(float(auc_binary(y, p)), 6) if y else None,
            "raw_ece": round(float(ece_binary(y, p, n_bins=n_bins)), 6) if y else None,
            "cum_auc": round(float(auc_binary(cy, cp)), 6) if cy else None,
            "cum_ece": round(float(ece_binary(cy, cp, n_bins=n_bins)), 6) if cy else None,
            "raw_mean_score": round(float(np.mean(p)) if p else float("nan"), 6),
            "cum_mean_score": round(float(np.mean(cp)) if cp else float("nan"), 6),
        }
    return {
        "n_prompts": len(rows),
        "n_steps_total": int(sum(r["n_steps"] for r in rows)),
        "ece_bins": n_bins,
        "sources": source_counts,
        "splits": split_counts,
        "by_position": by_position,
        "per_prompt": per_prompt,
    }


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.mode == "aggregate":
        ppdir = out / "per_prompt"
        rows = [json.loads(p.read_text()) for p in sorted(ppdir.glob("*.json"))]
        if not rows:
            print("no per-prompt confidence rows found")
            return 1
        agg = aggregate_prompt_rows(rows, n_bins=args.ece_bins)
        (out / "aggregate.json").write_text(json.dumps(agg, indent=2) + "\n")
        print(json.dumps(agg, indent=2))
        return 0

    shards = [args.shards] + (args.extra_shards or [])
    store = Stage2CaptureStore(shards)
    pids = args.pids or store.prompt_ids()
    if args.only_split:
        pids = [pid for pid in pids if store.prompt_meta(pid).get("split") == args.only_split]
    if args.only_source:
        pids = [pid for pid in pids if store.prompt_meta(pid).get("source") == args.only_source]
    if args.limit:
        pids = pids[: args.limit]

    dev = args.device
    dt = torch.float16 if args.dtype == "float16" else torch.float32
    print(f"building torch body+head on {dev} ({args.dtype})...", flush=True)
    t0 = time.time()
    body = build_body(DSPARK, TARGET, dev, dtype=dt)
    head = build_head(DSPARK, TARGET, dev, lora_rank=0, dtype=dt)
    head.eval()
    print(f"  built in {time.time()-t0:.0f}s", flush=True)

    ppdir = out / "per_prompt"
    ppdir.mkdir(parents=True, exist_ok=True)
    todo = [pid for pid in pids if not (ppdir / f"{pid}.json").exists()]
    print(f"{len(pids)} prompts ({len(todo)} to do)", flush=True)
    t1 = time.time()
    for i, pid in enumerate(todo):
        res = measure_prompt_torch_conf(pid, store, body, head, dev)
        (ppdir / f"{pid}.json").write_text(json.dumps(res) + "\n")
        if (i + 1) % 5 == 0 or i == len(todo) - 1:
            el = time.time() - t1
            print(f"  [{i+1}/{len(todo)}] {pid} steps={res['n_steps']} ({el:.0f}s, {el/(i+1):.1f}s/prompt)", flush=True)

    rows = [json.loads((ppdir / f"{pid}.json").read_text()) for pid in pids if (ppdir / f"{pid}.json").exists()]
    agg = aggregate_prompt_rows(rows, n_bins=args.ece_bins)
    (out / "aggregate.json").write_text(json.dumps(agg, indent=2) + "\n")
    print(json.dumps({
        "n_prompts": agg["n_prompts"],
        "n_steps_total": agg["n_steps_total"],
        "out": str(out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
