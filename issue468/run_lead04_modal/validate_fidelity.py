#!/usr/bin/env python3
"""Post-capture fidelity gate for Lead 04 (research-lead skill step 3 + the gate).

Runs the lead's Phase A item-2 checklist on converted FP bundles and applies the
LOCKED go/no-go decision rule (see the worklog entry "Locked go/no-go decision
rule" in issue468/pending/lead_04_fp_ceiling_capture.md):

Checks per FP bundle:
  (a) oracle_inputs.npz: main_hidden [n_gen, 12288] + positions [n_gen]  (shapes)
  (b) D1 = FP-vs-IQ2XXS greedy flip rate (target_selected_tokens.json vs the
      retained Q2 bundle's) — the never-measured baseline; high agreement also
      implies tokenizer alignment (upstream vLLM == local ds4).
  (c) drafter-on-FP-hiddens sanity: run dspark_oracle/measure_acceptance_bundle.py
      on the FP bundle (local vendored drafter on FP hiddens vs FP greedy labels)
      -> p=1; compare to the retained Q2 reference p=1 = 0.8125.

Gate (representation mapping):
  GREEN  if 0.70 <= p=1 <= 0.90  AND  |p=1 - 0.8125| <= 0.08   (≈ [0.73, 0.89])
  RED    otherwise (drafter reads garbage / leakage / wrong representation -> STOP)
D1 is recorded as a deliverable, not a gate. The pilot (5 prompts) gives a noisy
p=1; bands detect a GROSS representation mismatch, not the FP-vs-Q2 gap.

embed/lm_head round-trip (item 2, last bullet) is a spot-check left to codex gate 1
+ manual verification (it needs the drafter's lm_head weights directly); the
drafter-on-FP-hiddens sanity above is the stronger representation test.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ISSUE468 = HERE.parent
ROOT = ISSUE468.parent.parent

ORACLE_SCRIPT = ISSUE468 / "dspark_oracle" / "measure_acceptance_bundle.py"
Q2_BUNDLES = ISSUE468 / "artifacts" / "exactness_small_bundles"

# Locked decision-rule reference (exactness_small corpus, quant_mismatch_diagnostic.md)
RETAINED_Q2_P1 = 0.8125
P1_BAND = (0.70, 0.90)          # sane vendored-drafter band
P1_TOL = 0.08                   # |p1 - retained| tolerance for GREEN
MAIN_HIDDEN_DIM = 3 * 4096      # 12288 (concat of layers 40/41/42)


def load_greedy(bundle_dir: Path) -> list[int]:
    """Greedy target tokens: prefer target_selected_tokens.json (flat ints)."""
    f = bundle_dir / "target_selected_tokens.json"
    if f.exists():
        return [int(x) for x in json.loads(f.read_text())]
    tk = json.loads((bundle_dir / "target_topk.json").read_text())
    return [int(s["selected"]["id"]) for s in tk["steps"]]


def check_oracle_inputs(bundle_dir: Path) -> dict:
    p = bundle_dir / "oracle" / "oracle_inputs.npz"
    if not p.exists():
        return {"status": "MISSING", "path": str(p)}
    d = np.load(str(p))
    if "main_hidden" not in d or "positions" not in d:
        return {"status": "MISSING_KEYS", "keys": list(d.keys())}
    mh = d["main_hidden"]
    ok = mh.ndim == 2 and mh.shape[1] == MAIN_HIDDEN_DIM and mh.dtype == np.float32
    return {
        "status": "PASS" if ok else "SHAPE_MISMATCH",
        "main_hidden_shape": list(mh.shape),
        "dtype": str(mh.dtype),
        "n_gen": int(mh.shape[0]),
        "positions_head": d["positions"][:3].tolist(),
        "expected_last_dim": MAIN_HIDDEN_DIM,
    }


def measure_d1(fp_bundle: Path, q2_bundle: Path) -> dict:
    fp = load_greedy(fp_bundle)
    q2 = load_greedy(q2_bundle)
    n = min(len(fp), len(q2))
    if n == 0:
        return {"status": "EMPTY"}
    disagree = sum(1 for i in range(n) if fp[i] != q2[i])
    rate = disagree / n
    agree = 1.0 - rate
    return {
        "status": "PASS",
        "n_positions": n,
        "n_disagree": disagree,
        "flip_rate_D1": round(rate, 4),
        "agreement": round(agree, 4),
    }


def run_drafter_sanity(bundle_dir: Path, json_out: Path) -> dict:
    """Run the oracle: local drafter on FP hiddens vs FP greedy labels -> p=1."""
    cmd = [sys.executable, str(ORACLE_SCRIPT),
           "--bundle-dir", str(bundle_dir), "--json-out", str(json_out)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT"}
    if proc.returncode != 0:
        return {"status": "FAIL", "returncode": proc.returncode,
                "stderr": proc.stderr[-800:]}
    if not json_out.exists():
        return {"status": "NO_OUTPUT"}
    res = json.loads(json_out.read_text())
    # acceptance_summary schema: match_pct, total_match, total_positions, average_prefix
    if "total_positions" in res and res["total_positions"]:
        p1 = res["total_match"] / res["total_positions"]
    else:
        p1 = res.get("match_pct", 0.0) / 100.0
    ea = res.get("average_prefix", res.get("E_a", 0.0))
    green = (P1_BAND[0] <= p1 <= P1_BAND[1]) and (abs(p1 - RETAINED_Q2_P1) <= P1_TOL)
    return {
        "status": "GREEN" if green else "RED",
        "p1": round(float(p1), 4),
        "E_a_5block": round(float(ea), 4),
        "total_positions": res.get("total_positions"),
        "retained_q2_p1": RETAINED_Q2_P1,
        "delta_vs_retained": round(float(p1) - RETAINED_Q2_P1, 4),
        "green_band": f"p1 in {P1_BAND} and |p1-{RETAINED_Q2_P1}|<={P1_TOL}",
    }


def validate_all(fp_root: Path, q2_root: Path, limit: int = 0) -> dict:
    rows, d1s, p1s, reds, greens = [], [], [], 0, 0
    bundles = sorted(b for b in fp_root.iterdir() if b.is_dir() and (b / "oracle").exists())
    if limit:
        bundles = bundles[:limit]
    if not bundles:
        return {"status": "NO_FP_BUNDLES", "fp_root": str(fp_root)}

    for b in bundles:
        name = b.name
        q2 = q2_root / f"{name}__t0p0"
        print(f"\n--- {name} ---")
        oi = check_oracle_inputs(b)
        print(f"  oracle_inputs: {oi['status']} {oi.get('main_hidden_shape')}")
        d1 = measure_d1(b, q2) if q2.exists() else {"status": "NO_Q2_BUNDLE"}
        if d1.get("status") == "PASS":
            d1s.append(d1["flip_rate_D1"])
            print(f"  D1 flip rate: {d1['flip_rate_D1']} (agree {d1['agreement']})")
        else:
            print(f"  D1: {d1['status']}")
        ds = run_drafter_sanity(b, b / "fidelity_oracle.json")
        if ds.get("status") in ("GREEN", "RED"):
            p1s.append(ds["p1"])
            greens += ds["status"] == "GREEN"
            reds += ds["status"] == "RED"
            print(f"  drafter-on-FP p=1: {ds['p1']} ({ds['status']}; "
                  f"retained {RETAINED_Q2_P1}, Δ {ds['delta_vs_retained']})")
        else:
            print(f"  drafter sanity: {ds['status']} {ds.get('stderr','')[-120:]}")
        rows.append({"prompt": name, "oracle_inputs": oi, "d1": d1, "drafter_sanity": ds})

    # Aggregate verdict on the representation gate.
    mean_p1 = round(float(np.mean(p1s)), 4) if p1s else None
    mean_d1 = round(float(np.mean(d1s)), 4) if d1s else None
    gate = ("GREEN" if p1s and reds == 0 and greens == len(p1s)
            and P1_BAND[0] <= (mean_p1 or 0) <= P1_BAND[1]
            and abs((mean_p1 or 0) - RETAINED_Q2_P1) <= P1_TOL
            else ("RED" if p1s else "INCONCLUSIVE"))
    return {
        "representation_gate": gate,
        "n_bundles": len(bundles),
        "n_green": greens, "n_red": reds,
        "mean_drafter_on_FP_p1": mean_p1,
        "retained_q2_p1": RETAINED_Q2_P1,
        "mean_D1_flip_rate": mean_d1,
        "per_prompt": rows,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp-bundles", required=True, help="root of converted FP bundles (with oracle/)")
    ap.add_argument("--q2-bundles", default=str(Q2_BUNDLES))
    ap.add_argument("--limit", type=int, default=0, help="cap #bundles (drafter sanity loads the 87GB model each)")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    result = validate_all(Path(args.fp_bundles), Path(args.q2_bundles), args.limit)
    print("\n" + "=" * 64)
    print(f"REPRESENTATION GATE: {result.get('representation_gate')}")
    print(f"  mean drafter-on-FP p=1 = {result.get('mean_drafter_on_FP_p1')} "
          f"(retained Q2 {RETAINED_Q2_P1}) | mean D1 = {result.get('mean_D1_flip_rate')}")
    print("=" * 64)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2, default=str) + "\n")
        print(f"\nfull report: {args.json_out}")
    else:
        print(json.dumps(result, indent=2, default=str))
