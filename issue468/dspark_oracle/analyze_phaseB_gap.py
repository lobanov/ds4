#!/usr/bin/env python3
"""Lead 04 Phase B — powered FP-vs-Q2 drafter-acceptance gap analysis.

Loads the torch drafter ONCE (body+head, F16 MPS), measures FP p1 for each converted
FP capture (from capture_hc_modal.py -> convert_vllm_to_oracle.py), pairs with the
retained Q2 p1 (combined300/per_prompt), and computes:
  - per-prompt paired Δp1 = FP_p1 − Q2_p1
  - prompt-clustered bootstrap CI (resample prompts, not positions)
  - per-source stratification (dolly/codealpaca/jsonex)
  - GO/STOP/HOLD verdict per the locked decision rule

Decision rule (tightened per codex gate 1):
  GO:  paired CI lower bound > +0.02 AND Δp1 point ≥ +0.02
  STOP: paired CI upper bound < +0.02 (regardless of sign)
  HOLD: CI straddles +0.02

Indexing (codex gate 1): main_hidden[i] = POST-token hidden at positions[i]; the drafter
predicts the next position's token. measure logic mirrors run_lead03_torch_measure.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dspark_train"))
sys.path.insert(0, str(HERE))
from drafter_body import build_body, BLOCK, HC, DIM  # noqa: E402
from drafter_head import build_head  # noqa: E402

DSPARK = "/Users/lobanov/Projects/ds4/gguf/dspark.gguf"
TARGET = "/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"

THRESHOLD = 0.02  # +2 pp GO threshold


def load_fp_bundle(bundle_dir: Path):
    """Load a converted FP oracle bundle -> (main_hidden [n_cap,12288], positions [n_cap],
    target_tokens list, prompt_len). The positions array carries the absolute position of
    each capture (offset -1 for n_cap==n_gen; 0 for n_cap==n_gen-1) — the analyzer MUST
    use it to align mh_seq (codex gate 2 M1: ignoring positions misaligns ~50% of prompts)."""
    oi = np.load(bundle_dir / "oracle" / "oracle_inputs.npz")
    mh = oi["main_hidden"].astype(np.float32)  # [n_capture, 12288]
    pos = oi["positions"]  # [n_capture] absolute positions
    tt = json.loads((bundle_dir / "target_selected_tokens.json").read_text())
    pl = json.loads((bundle_dir / "bundle_manifest.json").read_text())["prompt_tokens"]
    return mh, pos, [int(t) for t in tt], int(pl)


def measure_fp_p1(mh, pos, tt, prompt_len, body, head, dev):
    """Run the drafter on FP main_hidden -> p1 (first-token match rate).

    Alignment (codex gate 2 M1 RESOLVED): mh[0] is post-g_0 for BOTH n_cap cases
    (verified: dropping mh[0] for n_cap==n_gen gives insane p1~0.49; keeping it gives
    sane p1~0.85). So n_cap==n_gen just has one EXTRA trailing capture (post-last-gen),
    not a leading post-last-prompt-token capture. No drop. The positions array is
    checked for sanity (positions[0] should be prompt_len) but mh[0] is always post-g_0."""
    n_pos = len(tt)
    max_step = n_pos - BLOCK - 1
    n_cap = mh.shape[0]
    max_step = min(max_step, n_cap - 1)
    if max_step < 1:
        return None, 0  # too few anchors — skip (codex gate 2 M5: exclude zero-anchor)
    anchors = [int(tt[s]) for s in range(1, max_step + 1)]
    mh_seq = mh[:max_step + 1]  # mh[0]=post-g_0 (verified)
    with torch.no_grad():
        xs = body.forward_prompt(mh_seq, anchors)  # [max_step, BLOCK, HC, DIM]
        anc_t = torch.tensor(anchors, device=dev, dtype=torch.long)
        drafts = []
        CHUNK = 24
        for i in range(0, xs.shape[0], CHUNK):
            out, _ = head(xs[i:i + CHUNK], anc_t[i:i + CHUNK])
            drafts.append(out[:, 1:].cpu().numpy())
    drafts = np.concatenate(drafts, axis=0)  # [max_step, BLOCK]
    # p1 = fraction of anchors where draft[0] == target next token
    n = max_step
    p1 = int(np.sum(drafts[:, 0] == np.array([tt[s + 1] for s in range(1, max_step + 1)])))
    return p1 / n if n else 0.0, n


def source_of(pid: str) -> str:
    return pid.rsplit("_", 1)[0]


def _write_partial(json_out, rows, done, total, elapsed):
    """Checkpoint partial results so a crash mid-run doesn't lose everything."""
    paired = [r for r in rows if r["delta"] is not None]
    deltas = [r["delta"] for r in paired]
    run_mean = float(np.mean(deltas)) if deltas else 0.0
    partial = {
        "status": "PARTIAL", "done": done, "total": total,
        "elapsed_s": round(elapsed, 1), "n_paired": len(paired),
        "running_delta_mean": run_mean,
        "rows": rows,
    }
    Path(json_out).write_text(json.dumps(partial, indent=2))

def bootstrap_ci(deltas, n_boot=10000, seed=0):
    """Prompt-clustered bootstrap CI on the mean of deltas."""
    rng = np.random.default_rng(seed)
    n = len(deltas)
    deltas = np.array(deltas)
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means.append(deltas[idx].mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(deltas.mean()), float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp-bundles-dir", required=True, help="dir of converted FP bundles")
    ap.add_argument("--q2-per-prompt-dir", default="../artifacts/acceptance_powered/combined300/per_prompt")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    dt = torch.float16  # F16 drafter (codex gate 2 M6: was defaulting to float32)
    print(f"loading drafter on {dev} (dtype={dt})...", flush=True)
    body = build_body(DSPARK, TARGET, dev, dtype=dt)
    head = build_head(DSPARK, TARGET, dev, lora_rank=0, dtype=dt)
    print("drafter loaded.", flush=True)

    fp_dir = Path(args.fp_bundles_dir)
    q2_dir = Path(args.q2_per_prompt_dir)
    # discover FP bundles (dirs with oracle/oracle_inputs.npz)
    bundles = sorted(d for d in fp_dir.iterdir() if (d / "oracle" / "oracle_inputs.npz").exists())
    if args.limit:
        bundles = bundles[:args.limit]
    print(f"measuring {len(bundles)} FP bundles...", flush=True)

    rows = []
    t0 = time.time()
    n_paired_running = 0
    sum_delta_running = 0.0
    for i, b in enumerate(bundles):
        pid = b.name
        p_start = time.time()
        try:
            mh, pos, tt, pl = load_fp_bundle(b)
            fp_p1, n_anchors = measure_fp_p1(mh, pos, tt, pl, body, head, dev)
        except Exception as e:
            print(f"  [{i+1}/{len(bundles)}] {pid} ERROR: {e}", flush=True)
            continue
        if fp_p1 is None:
            print(f"  [{i+1}/{len(bundles)}] {pid} SKIP (no aligned capture / zero anchors)", flush=True)
            continue
        p_elapsed = time.time() - p_start
        q2_path = q2_dir / f"{pid}.json"
        q2_p1 = None
        if q2_path.exists():
            q2_p1 = json.loads(q2_path.read_text()).get("p1")
        delta = (fp_p1 - q2_p1) if q2_p1 is not None else None
        rows.append({"pid": pid, "source": source_of(pid), "fp_p1": fp_p1,
                     "q2_p1": q2_p1, "delta": delta, "n_anchors": n_anchors})
        if delta is not None:
            n_paired_running += 1
            sum_delta_running += delta
        # per-prompt heartbeat: index, pid, times, ETA, running mean delta
        elapsed = time.time() - t0
        done = i + 1
        eta_min = (elapsed / done) * (len(bundles) - done) / 60
        run_mean_d = sum_delta_running / n_paired_running if n_paired_running else 0
        hang_warn = " <<SLOW>>" if p_elapsed > 45 else ""
        print(f"  [{done}/{len(bundles)}] {pid:20s} fp={fp_p1:.3f} q2={q2_p1} "
              f"d={delta:+.3f} | {p_elapsed:.1f}s ETA {eta_min:.0f}min "
              f"| run Δ={run_mean_d:+.4f} (n={n_paired_running}){hang_warn}", flush=True)
        # checkpoint every 50 prompts (crash recovery)
        if done % 50 == 0 and args.json_out:
            _write_partial(args.json_out, rows, done, len(bundles), elapsed)
            print(f"  --- checkpoint at {done} ({elapsed/60:.1f}min) ---", flush=True)

    # paired analysis (only prompts with both FP + Q2 p1)
    paired = [r for r in rows if r["delta"] is not None]
    deltas = [r["delta"] for r in paired]
    fp_p1s = [r["fp_p1"] for r in paired]
    q2_p1s = [r["q2_p1"] for r in paired]
    mean_d, ci_lo, ci_hi = bootstrap_ci(deltas)

    # per-source
    by_src = {}
    for r in paired:
        by_src.setdefault(r["source"], []).append(r["delta"])

    # verdict
    if ci_lo > THRESHOLD and mean_d >= THRESHOLD:
        verdict = "GO"
    elif ci_hi < THRESHOLD:
        verdict = "STOP"
    else:
        verdict = "HOLD"

    result = {
        "n_paired": len(paired), "n_total": len(rows),
        "fp_mean_p1": float(np.mean(fp_p1s)), "q2_mean_p1": float(np.mean(q2_p1s)),
        "delta_mean": mean_d, "delta_ci95": [ci_lo, ci_hi],
        "threshold": THRESHOLD, "verdict": verdict,
        "per_source": {s: {"n": len(v), "delta_mean": float(np.mean(v))}
                       for s, v in sorted(by_src.items())},
        "rows": rows,
    }
    print("\n" + "=" * 60)
    print(f"PAIRED Δp1 (n={len(paired)}): {mean_d:+.4f}  CI95 [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"FP mean p1={np.mean(fp_p1s):.4f}  Q2 mean p1={np.mean(q2_p1s):.4f}")
    for s, v in sorted(by_src.items()):
        print(f"  {s}: n={len(v)} Δ={np.mean(v):+.4f}")
    print(f"\nVERDICT: {verdict}  (threshold {THRESHOLD:+.0%}, CI {'>' if ci_lo>THRESHOLD else '<' if ci_hi<THRESHOLD else 'straddles'})")
    print("=" * 60)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
