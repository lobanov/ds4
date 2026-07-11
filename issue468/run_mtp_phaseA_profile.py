#!/usr/bin/env python3
"""Focused Lead 08 Phase A profiling for the shipped MTP batch verifier.

Runs the retained long-prompt corpus with K in {3,4,5} under the new
`DS4_MTP_VERIFY_PROFILE` instrumentation and summarizes:
  - cycle-level `ds4: mtp timing ...` lines
  - initial verify profiles (`tokens == K`)
  - replay / partial-accept profiles (`tokens < K`)

Artifacts land in `issue468/artifacts/mtp_phaseA_profile/`.
"""
from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "issue468/artifacts/mtp_phaseA_profile"
OUT.mkdir(parents=True, exist_ok=True)
DS4 = str(ROOT / "ds4")
MODEL = "/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
MTP = "/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"
CORPUS = ROOT / "issue468/prompts/baseline_corpus"
N = 64
TEMP = "0.0"
SEED = "1"
KS = [3, 4, 5]
PROMPTS = [
    ("code_8k", CORPUS / "code_8k.txt", 16384),
    ("synthesis_8k", CORPUS / "synthesis_8k.txt", 16384),
    ("grounded_8k", CORPUS / "grounded_8k.txt", 16384),
]

TPS_RE = re.compile(r"prefill:\s*([0-9.]+) t/s, generation:\s*([0-9.]+) t/s")
TIMING_RE = re.compile(
    r"ds4: mtp timing (\S+) drafted=(\d+) (committed|verified)=(\d+).*?draft=([0-9.]+) ms.*?"
    r"verify=([0-9.]+) ms.*?total=([0-9.]+) ms")
PROFILE_RE = re.compile(
    r"ds4: mtp verify profile start=(\d+) tokens=(\d+) top_rows=(\d+) "
    r"upload=([0-9.]+) ms layer_encode=([0-9.]+) ms layer_execute=([0-9.]+) ms "
    r"head_encode=([0-9.]+) ms head_execute=([0-9.]+) ms top_read=([0-9.]+) ms "
    r"logits_read=([0-9.]+) ms selected=([0-9.]+) GiB full=([0-9.]+) GiB "
    r"avg_unique=([0-9.]+) min_unique=(\d+) max_unique=(\d+)")


def summarize(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def summarize_records(records: list[dict], fields: list[str]) -> dict[str, dict[str, float]] | None:
    if not records:
        return None
    out: dict[str, dict[str, float]] = {}
    for field in fields:
        stats = summarize([float(r[field]) for r in records])
        if stats:
            out[field] = stats
    return out


def run_config(label: str, prompt: Path, ctx: int, k: int) -> dict:
    env = dict(os.environ)
    env.update({
        "DS4_MTP_TIMING": "1",
        "DS4_MTP_SPEC_LOG": "1",
        "DS4_MTP_VERIFY_PROFILE": "1",
    })
    cmd = [
        DS4, "--backend", "metal", "-m", MODEL, "-c", str(ctx), "-n", str(N),
        "--temp", TEMP, "--seed", SEED, "--prompt-file", str(prompt),
        "--mtp", MTP, "--mtp-draft", str(k),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    dt = time.time() - t0
    (OUT / f"{label}.stderr").write_text(proc.stderr)
    (OUT / f"{label}.stdout").write_text(proc.stdout)

    result: dict = {
        "label": label,
        "prompt": str(prompt.relative_to(ROOT)),
        "ctx": ctx,
        "K": k,
        "returncode": proc.returncode,
        "wall_s": round(dt, 2),
        "n": N,
    }
    tps = TPS_RE.search(proc.stderr)
    if tps:
        result["prefill_tps"] = float(tps.group(1))
        result["gen_tps"] = float(tps.group(2))

    timings: list[dict] = []
    profiles: list[dict] = []
    for line in proc.stderr.splitlines():
        mt = TIMING_RE.search(line)
        if mt:
            timings.append({
                "kind": mt.group(1),
                "drafted": int(mt.group(2)),
                "count_kind": mt.group(3),
                "committed": int(mt.group(4)),
                "draft_ms": float(mt.group(5)),
                "verify_ms": float(mt.group(6)),
                "total_ms": float(mt.group(7)),
            })
            continue
        mp = PROFILE_RE.search(line)
        if mp:
            profiles.append({
                "start": int(mp.group(1)),
                "tokens": int(mp.group(2)),
                "top_rows": int(mp.group(3)),
                "upload_ms": float(mp.group(4)),
                "layer_encode_ms": float(mp.group(5)),
                "layer_execute_ms": float(mp.group(6)),
                "head_encode_ms": float(mp.group(7)),
                "head_execute_ms": float(mp.group(8)),
                "top_read_ms": float(mp.group(9)),
                "logits_read_ms": float(mp.group(10)),
                "selected_gib": float(mp.group(11)),
                "full_gib": float(mp.group(12)),
                "avg_unique": float(mp.group(13)),
                "min_unique": int(mp.group(14)),
                "max_unique": int(mp.group(15)),
            })

    result["spec_miss_first"] = proc.stderr.count("spec miss first draft")
    result["n_cycles"] = len(timings)
    result["timing_summary"] = summarize_records(
        timings, ["draft_ms", "verify_ms", "total_ms", "drafted", "committed"])

    initial_profiles = [p for p in profiles if p["tokens"] == k]
    replay_profiles = [p for p in profiles if p["tokens"] < k]
    profile_fields = [
        "upload_ms", "layer_encode_ms", "layer_execute_ms", "head_encode_ms",
        "head_execute_ms", "top_read_ms", "logits_read_ms", "selected_gib",
        "full_gib", "avg_unique",
    ]
    result["initial_profile_summary"] = summarize_records(initial_profiles, profile_fields)
    result["replay_profile_summary"] = summarize_records(replay_profiles, profile_fields)
    result["profile_counts"] = {
        "initial": len(initial_profiles),
        "replay": len(replay_profiles),
    }
    return result


def main() -> int:
    results = []
    for prompt_label, prompt_path, ctx in PROMPTS:
        print(f"\n##### prompt={prompt_label} #####", flush=True)
        for k in KS:
            label = f"{prompt_label}__k{k}"
            r = run_config(label, prompt_path, ctx, k)
            r["prompt_label"] = prompt_label
            results.append(r)
            print(
                f"  K={k}: rc={r['returncode']} gen={r.get('gen_tps')} "
                f"verify_med={r.get('timing_summary', {}).get('verify_ms', {}).get('median')} "
                f"initial_exec_med={r.get('initial_profile_summary', {}).get('layer_execute_ms', {}).get('median')} "
                f"replay_exec_med={r.get('replay_profile_summary', {}).get('layer_execute_ms', {}).get('median')}",
                flush=True,
            )
    (OUT / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
