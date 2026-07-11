#!/usr/bin/env python3
"""Compare greedy output bytes on the retained exactness corpus.

Runs plain decode and speculative configs on each prompt in
`prompts/exactness_small_corpus/` and reports whether stdout is byte-identical
to baseline. Intended as the Lead 06 exactness gate for experimental verifier
paths.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "issue468/artifacts/mtp_exactness_compare"
OUT.mkdir(parents=True, exist_ok=True)
DS4 = str(ROOT / "ds4")
MODEL = "/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
MTP = "/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"
MANIFEST = ROOT / "issue468/prompts/exactness_small_corpus/manifest.json"
N = int(os.getenv("DS4_MTP_EXACTNESS_N", "128"))
CTX = 8192
TEMP = "0.0"
SEED = "1"
KS = [int(k) for k in os.getenv("DS4_MTP_EXACTNESS_KS", "4").split(",") if k.strip()]

TPS_RE = re.compile(r"prefill:\s*([0-9.]+) t/s, generation:\s*([0-9.]+) t/s")


def run_config(prompt_path: Path, label: str, extra_args: list[str], env_extra: dict[str, str]) -> dict:
    env = dict(os.environ)
    env.update(env_extra)
    cmd = [
        DS4, "--backend", "metal", "-m", MODEL, "-c", str(CTX), "-n", str(N),
        "--temp", TEMP, "--seed", SEED, "--prompt-file", str(prompt_path), *extra_args,
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    dt = time.time() - t0
    rec = {
        "label": label,
        "returncode": proc.returncode,
        "wall_s": round(dt, 2),
        "stdout": proc.stdout,
    }
    m = TPS_RE.search(proc.stderr)
    if m:
        rec["prefill_tps"] = float(m.group(1))
        rec["gen_tps"] = float(m.group(2))
    return rec


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    results = []
    for prompt_label, item in manifest.items():
        prompt = ROOT / "issue468" / item["file"]
        print(f"\n##### prompt={prompt_label} #####", flush=True)
        baseline = run_config(prompt, f"{prompt_label}__baseline", [], {})
        (OUT / f"{prompt_label}__baseline.stdout").write_text(baseline["stdout"])
        results.append({
            "prompt_label": prompt_label,
            "impl": "baseline",
            "K": "baseline",
            **{k: v for k, v in baseline.items() if k != "stdout"},
        })
        print(f"  baseline: rc={baseline['returncode']} gen={baseline.get('gen_tps')}", flush=True)
        for k in KS:
            env = {
                "DS4_MTP_TIMING": "1",
                "DS4_MTP_SPEC_LOG": "1",
                "DS4_MTP_ANCHOR_REUSE": "1",
            }
            rec = run_config(prompt, f"{prompt_label}__reuse_k{k}",
                             ["--mtp", MTP, "--mtp-draft", str(k)], env)
            identical = rec["stdout"] == baseline["stdout"]
            mismatch_at = None
            if not identical:
                n = min(len(rec["stdout"]), len(baseline["stdout"]))
                for i in range(n):
                    if rec["stdout"][i] != baseline["stdout"][i]:
                        mismatch_at = i
                        break
                if mismatch_at is None:
                    mismatch_at = n
            (OUT / f"{prompt_label}__reuse_k{k}.stdout").write_text(rec["stdout"])
            results.append({
                "prompt_label": prompt_label,
                "impl": "anchor_reuse_exact",
                "K": k,
                "matches_baseline": identical,
                "mismatch_at": mismatch_at,
                **{kk: vv for kk, vv in rec.items() if kk != "stdout"},
            })
            print(f"  reuse K={k}: rc={rec['returncode']} gen={rec.get('gen_tps')} match={identical}", flush=True)
    (OUT / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
