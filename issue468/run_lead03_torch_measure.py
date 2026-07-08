#!/usr/bin/env python3
"""Lead 03 — torch/MPS acceptance measurement + precision gate (task-1c/1d/4).

Builds the Stage 2 torch drafter port (body+head, F16 MPS) and measures acceptance
(E[a|K]/S(K)/p1/per-position/draft-tokens-per-step) from the Stage2CaptureStore,
matching measure_acceptance_bundle's metrics.

Modes:
  gate    : measure N prompts via torch AND via the numpy oracle (single-process,
            DS4_EXPERT_MAX_CACHE), compare draft-token agreement. Required >=99% before
            the powered (measure) run is trusted.
  measure : torch-only measurement over prompts (the powered run); writes per-prompt
            checkpoints (same compact schema as run_lead03_measure_stage2) so the
            numpy and torch runners stay interchangeable.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "dspark_train"))
sys.path.insert(0, str(HERE / "dspark_oracle"))
from drafter_body import build_body, BLOCK, HC, DIM  # noqa: E402
from drafter_head import build_head  # noqa: E402
from stage2_capture_store import Stage2CaptureStore  # noqa: E402

DSPARK = "/Users/lobanov/Projects/ds4/gguf/dspark.gguf"
TARGET = "/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"


def measure_prompt_torch(pid, store, body, head, dev, max_step_cap=0):
    """Return drafts [max_step, BLOCK] (int) + compact acceptance rows for one prompt."""
    meta = store.prompt_meta(pid)
    pos0 = int(meta["prompt_tokens"])
    tt = store.target_tokens(pid)
    n_pos = len(tt)
    max_step = n_pos - BLOCK - 1
    if max_step_cap:
        max_step = min(max_step, max_step_cap)
    if max_step < 1:
        return [], np.zeros((0, BLOCK), dtype=np.int64)
    # mh_seq[step] = main_hidden at pos0+step, step=0..max_step  (step 0 = init/first anchor)
    mh_seq = np.stack([store.main_hidden_at(pid, pos0 + s) for s in range(max_step + 1)])
    # anchors[step-1] = target_tokens[step], step=1..max_step  (matches measure_acceptance_bundle)
    anchors = [int(tt[s]) for s in range(1, max_step + 1)]
    with torch.no_grad():
        xs = body.forward_prompt(mh_seq, anchors)          # [max_step, BLOCK, HC, DIM]
        anc_t = torch.tensor(anchors, device=dev, dtype=torch.long)
        drafts = []
        CHUNK = 24
        for i in range(0, xs.shape[0], CHUNK):
            out, _ = head(xs[i:i + CHUNK], anc_t[i:i + CHUNK])
            drafts.append(out[:, 1:].cpu().numpy())
    drafts = np.concatenate(drafts, axis=0).astype(np.int64)  # [max_step, BLOCK]
    rows = []
    for i in range(max_step):                              # i = xs index; measure step = i+1
        draft = drafts[i].tolist()
        tgt = [int(x) for x in tt[i + 2:i + 2 + BLOCK]]    # step i+1 predicts tt[(i+1)+1..]
        prefix = 0
        for d, t in zip(draft, tgt):
            if d == t:
                prefix += 1
            else:
                break
        rows.append({"step": i + 1, "draft": draft, "target": tgt, "prefix": prefix})
    return rows, drafts


def compact(pid, rows):
    n = len(rows)
    if n == 0:
        return {"prompt_id": pid, "n_steps": 0, "E_a_5block": 0.0, "p1": 0.0,
                "per_position_match": [0] * BLOCK, "prefix_hist": [0] * (BLOCK + 1)}
    ph = [0] * (BLOCK + 1)
    ppm = [0] * BLOCK
    p1 = 0
    ea = 0
    for r in rows:
        ph[r["prefix"]] += 1
        if r["draft"][0] == r["target"][0]:
            p1 += 1
        ea += r["prefix"]
        for i in range(BLOCK):
            if i < len(r["draft"]) and r["draft"][i] == r["target"][i]:
                ppm[i] += 1
    return {"prompt_id": pid, "n_steps": n, "drafts": [r["draft"] for r in rows],
            "E_a_5block": round(ea / n, 5), "p1": round(p1 / n, 5),
            "per_position_match": ppm, "prefix_hist": ph}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gate", "measure"], required=True)
    ap.add_argument("--shards", default=str(HERE / "dspark_train" / "data" / "shards"))
    ap.add_argument("--out", default=str(HERE / "artifacts" / "acceptance_powered" / "torch_measure"))
    ap.add_argument("--limit", type=int, default=0, help="cap prompts (0 = all)")
    ap.add_argument("--pids", nargs="*", default=None)
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--device", choices=["mps", "cpu"], default="mps")
    args = ap.parse_args()

    dev = args.device
    dt = torch.float16 if args.dtype == "float16" else torch.float32
    print(f"building torch body+head on {dev} ({args.dtype})...", flush=True)
    t0 = time.time()
    body = build_body(DSPARK, TARGET, dev, dtype=dt)
    head = build_head(DSPARK, TARGET, dev, lora_rank=0, dtype=dt)
    head.eval()
    print(f"  built in {time.time()-t0:.0f}s", flush=True)
    store = Stage2CaptureStore(args.shards)
    pids = args.pids or (store.prompt_ids()[: args.limit] if args.limit else store.prompt_ids())

    if args.mode == "gate":
        from measure_acceptance_bundle import build_model_ctx, build_drafter_ctx, measure_bundle
        print("building numpy oracle ctx (single-process, cap via env)...", flush=True)
        mctx = build_model_ctx(TARGET)
        dctx = build_drafter_ctx(DSPARK)
        res = []
        for pid in pids:
            trows, tdrafts = measure_prompt_torch(pid, store, body, head, dev)
            nsum = measure_bundle(store=store, prompt_id=pid, mctx=mctx, dctx=dctx, reuse_mode="none")
            nrows = nsum["rows"]
            # compare draft-token lists per step (torch vs numpy)
            agree_steps = 0; agree_tokens = 0; ntok = 0
            for tr, nr in zip(trows, nrows):
                if tr["draft"] == nr["draft"]:
                    agree_steps += 1
                for a, b in zip(tr["draft"], nr["draft"]):
                    agree_tokens += int(a == b); ntok += 1
            nsteps = min(len(trows), len(nrows))
            res.append({"prompt_id": pid,
                        "torch_E_a_5": round(sum(r["prefix"] for r in trows)/max(1,len(trows)),4),
                        "numpy_E_a_5": nsum["average_prefix"],
                        "step_draft_agree": round(agree_steps/max(1,nsteps),4),
                        "token_agree": round(agree_tokens/max(1,ntok),4),
                        "n_steps": nsteps})
            print(f"  {pid}: torch_E={res[-1]['torch_E_a_5']} numpy_E={res[-1]['numpy_E_a_5']} "
                  f"step_agree={res[-1]['step_draft_agree']} token_agree={res[-1]['token_agree']}", flush=True)
        agg = {"n_prompts": len(res),
               "mean_step_agree": round(float(np.mean([r["step_draft_agree"] for r in res])),5),
               "mean_token_agree": round(float(np.mean([r["token_agree"] for r in res])),5),
               "verdict_pass_99pct": bool(np.mean([r["step_draft_agree"] for r in res]) >= 0.99),
               "per_prompt": res}
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        (out / "torch_precision_gate.json").write_text(json.dumps(agg, indent=2) + "\n")
        print(json.dumps({k:v for k,v in agg.items() if k!="per_prompt"}, indent=2))
        return 0 if agg["verdict_pass_99pct"] else 2

    # measure mode
    out = Path(args.out); (out / "per_prompt").mkdir(parents=True, exist_ok=True)
    todo = [p for p in pids if not (out / "per_prompt" / f"{p}.json").exists()]
    print(f"{len(pids)} prompts ({len(todo)} to do)", flush=True)
    t0 = time.time()
    for i, pid in enumerate(todo):
        rows, _ = measure_prompt_torch(pid, store, body, head, dev)
        c = compact(pid, rows)
        (out / "per_prompt" / f"{pid}.json").write_text(json.dumps(c) + "\n")
        if (i + 1) % 10 == 0 or i == len(todo) - 1:
            el = time.time() - t0
            print(f"  [{i+1}/{len(todo)}] {pid} E[a|5]={c['E_a_5block']} p1={c['p1']} "
                  f"({el:.0f}s, {el/(i+1):.1f}s/prompt)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
