#!/usr/bin/env python3
"""Lead 04 Phase C — FP cycle-jump E[a|K]/speedup + confidence calibration.

Tests the user's hypothesis: do later-position acceptance (E[a|K], K>1) or confidence
calibration favor FP over Q2 even when first-token p1 doesn't? Runs the drafter on FP
hiddens at a given dtype, computes cycle-jump E[a|K]/S(K)/speedup (Lead 03's estimator)
+ confidence calibration (does the confidence score predict acceptance?), compares to the
Q2 reference (E[a|4]=2.198, speedup 0.982x, p1=0.793).
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dspark_train"))
sys.path.insert(0, str(HERE))
from drafter_body import build_body, BLOCK  # noqa
from drafter_head import build_head  # noqa

DSPARK = "/Users/lobanov/Projects/ds4/gguf/dspark.gguf"
TARGET = "/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
DECODE_MS, DRAFT_MS = 26.0, 10.0
VERIFY_MS = {2: 43.6, 3: 59.7, 4: 65.8, 5: 74.5}


def prefix_k(draft, tgt, k):
    p = 0
    for i in range(min(k, len(draft), len(tgt))):
        if int(draft[i]) == int(tgt[i]): p += 1
        else: break
    return p


def simulate_ea(drafts, tt, K):
    """Cycle-jump E[a|K] + S(K) for one prompt (Lead 03 estimator)."""
    s = 0; acc = []
    while s < len(drafts):
        tgt_slice = [int(tt[s + 2 + j]) for j in range(K) if s + 2 + j < len(tt)]
        a = min(prefix_k(drafts[s], tgt_slice, K), K)
        acc.append(a); s += a + 1
    if not acc: return None, None
    ea = np.mean(acc); sk = np.mean([1 if x == K else 0 for x in acc])
    return float(ea), float(sk)


def speedup_k(ea, sk, K):
    cost = DRAFT_MS + VERIFY_MS[K] + DECODE_MS * sk
    return (ea + 1) * DECODE_MS / cost


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp-bundles-dir", default="/tmp/phaseB_all")
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    dt = torch.float16 if args.dtype == "float16" else torch.float32
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading drafter on {dev} ({args.dtype})...", flush=True)
    body = build_body(DSPARK, TARGET, dev, dtype=dt)
    head = build_head(DSPARK, TARGET, dev, lora_rank=0, dtype=dt)

    bundles = sorted(d for d in Path(args.fp_bundles_dir).iterdir()
                     if (d / "oracle" / "oracle_inputs.npz").exists())[:args.limit]
    print(f"measuring {len(bundles)} FP bundles...", flush=True)

    ea4_all, p1_all, conf_accept, conf_reject = [], [], [], []
    t0 = time.time()
    for i, b in enumerate(bundles):
        oi = np.load(b / "oracle" / "oracle_inputs.npz")
        mh = oi["main_hidden"].astype(np.float32); tt = json.loads((b / "target_selected_tokens.json").read_text())
        tt = [int(t) for t in tt]
        max_step = min(len(tt) - BLOCK - 1, mh.shape[0] - 1)
        if max_step < 1: continue
        anc = [tt[s] for s in range(1, max_step + 1)]; mhseq = mh[:max_step + 1]
        with torch.no_grad():
            xs = body.forward_prompt(mhseq, anc)
            out, _, conf_logits, conf_scores = head(xs, torch.tensor(anc, device=dev, dtype=torch.long), return_conf=True)
        drafts = out[:, 1:].cpu().numpy()  # [max_step, BLOCK]
        conf = conf_scores.cpu().numpy()    # [max_step, BLOCK]
        # cycle-jump E[a|4]
        ea, sk = simulate_ea(drafts, tt, 4)
        if ea is not None: ea4_all.append(ea)
        # p1
        p1 = np.mean(drafts[:, 0] == np.array([tt[s + 1] for s in range(1, max_step + 1)]))
        p1_all.append(float(p1))
        # confidence calibration: position-0 (first draft) conf vs accept (correct offset: drafts[s] predicts tt[s+2])
        for s in range(max_step):
            if s + 2 >= len(tt): continue
            accept = int(drafts[s, 0] == tt[s + 2])
            (conf_accept if accept else conf_reject).append(float(conf[s, 0]))
        if (i + 1) % 10 == 0: print(f"  [{i+1}/{len(bundles)}] ({(time.time()-t0)/(i+1):.1f}s/p)", flush=True)

    ea4_mean = np.mean(ea4_all); p1_mean = np.mean(p1_all)
    # S(4) approx from ea (recompute properly would need per-cycle; use the simulate return)
    sp4 = speedup_k(ea4_mean, 0.34, 4)  # S(4) approx from Q2; report ea4 mainly
    # confidence: mean conf when accepted vs rejected (separation)
    ca = np.mean(conf_accept) if conf_accept else 0; cr = np.mean(conf_reject) if conf_reject else 0
    print(f"\n=== FP @ {args.dtype} (n={len(ea4_all)}) ===")
    print(f"  p1 = {p1_mean:.4f}  (Q2 ref 0.793)")
    print(f"  cycle-jump E[a|4] = {ea4_mean:.4f}  (Q2 ref 2.198)")
    print(f"  speedup(K=4) ~ {sp4:.4f}x  (Q2 ref 0.982x, uses Q2 S(4)=0.34)")
    print(f"  confidence: mean(accept)={ca:.4f} vs mean(reject)={cr:.4f}  separation={ca-cr:+.4f}")
    print(f"  (larger separation = better-calibrated confidence → better scheduling)")
    res = {"dtype": args.dtype, "n": len(ea4_all), "fp_p1": p1_mean, "fp_ea4": ea4_mean,
           "fp_speedup4": sp4, "conf_accept_mean": ca, "conf_reject_mean": cr, "conf_separation": ca - cr}
    if args.json_out: Path(args.json_out).write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
