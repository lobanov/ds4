#!/usr/bin/env python3
"""Lead 03 task-1c/4 — powered acceptance measurement over the Stage 2 corpus
(+ later the broader combined corpus) via the Stage2CaptureStore.

Parallel (ProcessPool), checkpointed (per-prompt JSON, skip-if-exists), memory-safe
(ExpertStore.clear() per prompt bounds the dequant cache). Produces per-prompt
E[a|K]/S(K)/p1 + per-position histogram, then aggregate E[a|4] with a prompt-clustered
bootstrap CI and (task-1c) the power-calc N for half-width 0.05 / 0.028.

Reusable for task-4 (full corpus) via --shards + --manifest pointing at the combined set.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "dspark_oracle"))

MODEL = "/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
DSPARK = "/Users/lobanov/Projects/ds4/gguf/dspark.gguf"
BLOCK = 5

_W = {}  # worker globals


def _init_worker(model, dspark, shards_dir, max_steps):
    import gc
    from measure_acceptance_bundle import build_model_ctx, build_drafter_ctx
    from stage2_capture_store import Stage2CaptureStore
    _W["mctx"] = build_model_ctx(model)
    _W["dctx"] = build_drafter_ctx(dspark)
    _W["store"] = Stage2CaptureStore(shards_dir)
    _W["max_steps"] = max_steps


def _measure_one(prompt_id):
    from measure_acceptance_bundle import measure_bundle
    mctx, dctx, store = _W["mctx"], _W["dctx"], _W["store"]
    # cap steps if requested (power-calc mode): truncate target_tokens via a wrapper
    s = measure_bundle(store=store, prompt_id=prompt_id, mctx=mctx, dctx=dctx, reuse_mode="none")
    # NB: do NOT clear expert caches between prompts -- the LRU cap (DS4_EXPERT_MAX_CACHE)
    # bounds RAM and hot experts recur across prompts (amortizes the Q4_K dequant).
    # compact summary (drop rows to keep checkpoints small, but keep per-position match + p1 + prefix)
    rows = s["rows"]
    per_pos = np.zeros(BLOCK, dtype=np.int64); n = len(rows)
    p1 = 0
    ea = 0
    sk = np.zeros(BLOCK + 1, dtype=np.int64)  # S(k)=P(first k match); prefix hist
    for r in rows:
        if r["draft"][0] == r["target"][0]:
            p1 += 1
        ea += r["prefix"]
        sk[r["prefix"]] += 1
        for i in range(BLOCK):
            if r["draft"][i] == r["target"][i]:
                per_pos[i] += 1
    return {
        "prompt_id": prompt_id, "n_steps": n,
        "E_a_5block": round(ea / n, 5) if n else 0.0,
        "p1": round(p1 / n, 5) if n else 0.0,
        "per_position_match": [int(x) for x in per_pos],
        "prefix_hist": [int(x) for x in sk],   # index k = count of cycles with prefix==k
        "source": store.prompt_meta(prompt_id).get("source"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default=str(HERE / "dspark_train" / "data" / "shards"))
    ap.add_argument("--out", required=True, help="output dir for checkpoints + aggregate")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="cap prompts (0=all)")
    ap.add_argument("--only-split", default=None, help="filter prompt source split (e.g. eval)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260707)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "per_prompt").mkdir(exist_ok=True)
    # get prompt list (read index directly to avoid loading store in parent)
    idx = json.loads((Path(args.shards) / "index.json").read_text())
    pids = [p["prompt_id"] for p in idx["prompts"]]
    if args.only_split:
        pids = [p["prompt_id"] for p in idx["prompts"] if p.get("split") == args.only_split]
    if args.limit:
        pids = pids[: args.limit]
    print(f"{len(pids)} prompts; workers={args.workers}; shards={args.shards}", flush=True)

    # skip already-measured
    todo = [p for p in pids if not (out / "per_prompt" / f"{p}.json").exists()]
    print(f"  {len(pids) - len(todo)} already measured; {len(todo)} to do", flush=True)
    t0 = time.time()
    if todo:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                                 initargs=(MODEL, DSPARK, args.shards, 0)) as ex:
            for i, res in enumerate(ex.map(_measure_one, todo, chunksize=1)):
                (out / "per_prompt" / f"{res['prompt_id']}.json").write_text(json.dumps(res) + "\n")
                if (i + 1) % 10 == 0 or i == len(todo) - 1:
                    el = time.time() - t0
                    print(f"  [{i+1}/{len(todo)}] {res['prompt_id']} E[a|5]={res['E_a_5block']} "
                          f"p1={res['p1']} ({el:.0f}s, {el/(i+1):.1f}s/prompt)", flush=True)

    # ---- aggregate from checkpoints ----
    rows = [json.loads((out / "per_prompt" / f"{p}.json").read_text()) for p in pids
            if (out / "per_prompt" / f"{p}.json").exists()]
    n_prompts = len(rows)
    ea5 = np.array([r["E_a_5block"] for r in rows])
    p1 = np.array([r["p1"] for r in rows])
    # E[a|k] for k=1..5: mean per-prompt prefix already = E_a_5block (block=5). Also compute
    # the per-position S(k) and E[a|k] from the aggregate prefix_hist (cycle-level).
    # E[a|k] (k<=5): from prefix distribution, E[a|k] = sum_{j=1..k} S(j) where S(j)=P(prefix>=j)
    agg_hist = np.sum([np.array(r["prefix_hist"]) for r in rows], axis=0)  # index 0..5
    total_cycles = agg_hist.sum()
    S = np.array([agg_hist[j:].sum() / total_cycles for j in range(1, BLOCK + 1)])  # S(k)=P(prefix>=k)
    Eak = np.array([S[:k].sum() if k > 0 else 0.0 for k in range(BLOCK + 1)])  # E[a|k], k=0..5
    # prompt-clustered bootstrap CI on E[a|4]
    ea4_per = ea5  # block=5 -> E[a|5block] per prompt; model uses E[a|4]; compute E[a|4] per prompt too
    # recompute per-prompt E[a|4] from prefix_hist (E[a|4] = S(1)+S(2)+S(3)+S(4) per prompt)
    ea4_per = np.array([
        (np.array(r["prefix_hist"]).astype(float).sum() and
         sum(np.array(r["prefix_hist"])[j:].sum() for j in range(1, 5)) / max(1, np.array(r["prefix_hist"]).sum()))
        for r in rows])
    rng = np.random.default_rng(args.seed)
    idx_arr = np.arange(n_prompts)
    boot = np.array([ea4_per[rng.choice(idx_arr, n_prompts, replace=True)].mean() for _ in range(args.n_boot)])
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    sd_prompt = float(ea4_per.std(ddof=1))
    mean_ea4 = float(ea4_per.mean())
    hw = (ci[1] - ci[0]) / 2

    def n_for(target_hw):
        if sd_prompt <= 0: return 0
        return int(math.ceil((1.96 * sd_prompt / target_hw) ** 2))

    agg = {
        "n_prompts": n_prompts, "total_cycles": int(total_cycles),
        "E_a_4": round(mean_ea4, 5), "E_a_5": round(float(ea5.mean()), 5),
        "S_k": {f"S({k})": round(float(S[k-1]), 5) for k in range(1, BLOCK + 1)},
        "E_ak": {f"E[a|{k}]": round(float(Eak[k]), 5) for k in range(1, BLOCK + 1)},
        "p1_mean": round(float(p1.mean()), 5),
        "per_position_match_rate": [round(float(sum(r["per_position_match"][i] for r in rows) / total_cycles), 5)
                                    for i in range(BLOCK)],
        "prompt_sd_E_a_4": round(sd_prompt, 5),
        "prompt_clustered_ci95_E_a_4": [round(ci[0], 5), round(ci[1], 5)],
        "ci_half_width": round(hw, 5),
        "power_calc": {
            "N_for_hw_0.05": n_for(0.05), "N_for_hw_0.028": n_for(0.028),
            "note": "N = (1.96*sd/hw)^2 from empirical prompt-level sd at the measured step count",
        },
        "sources": dict((s, sum(1 for r in rows if r["source"] == s)) for s in set(r["source"] for r in rows)),
    }
    (out / "aggregate.json").write_text(json.dumps(agg, indent=2) + "\n")
    print(json.dumps(agg, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
