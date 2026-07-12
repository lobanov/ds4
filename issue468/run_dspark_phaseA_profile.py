#!/usr/bin/env python3
"""Focused DSpark profiling on the retained long-prompt corpus.

Summarizes the `DS4_DSPARK_TIMING` / `DS4_DSPARK_SPEC_LOG` cycle lines for the
default scheduled path and optional fixed verify-K overrides.
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
OUT = ROOT / "issue468/artifacts/dspark_phaseA_profile"
OUT.mkdir(parents=True, exist_ok=True)
DS4 = str(ROOT / "ds4")
MODEL = "/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
DSPARK = os.getenv("DS4_TEST_DSPARK", "/Users/lobanov/Projects/ds4/gguf/dspark.gguf")
CORPUS = ROOT / "issue468/prompts/baseline_corpus"
N = int(os.getenv("DS4_DSPARK_PROFILE_N", "64"))
TEMP = "0.0"
SEED = "1"
VERIFY_KS = [v.strip() for v in os.getenv("DS4_DSPARK_PROFILE_VERIFY_KS", "sched,1").split(",") if v.strip()]
PROFILE_TAG = os.getenv("DS4_DSPARK_PROFILE_TAG", "").strip()
SCHEDULE_BATCHED = os.getenv("DS4_DSPARK_SCHEDULE_BATCHED", "").strip()
PROMPTS = [
    ("code_8k", CORPUS / "code_8k.txt", 16384),
    ("synthesis_8k", CORPUS / "synthesis_8k.txt", 16384),
    ("grounded_8k", CORPUS / "grounded_8k.txt", 16384),
]

TPS_RE = re.compile(r"prefill:\s*([0-9.]+) t/s, generation:\s*([0-9.]+) t/s")
TIMING_RE = re.compile(
    r"ds4: dspark timing drafted=(\d+) verify=(\d+) verified=(\d+) "
    r"decode=([0-9.]+) ms draft=([0-9.]+) ms verify=([0-9.]+) ms total=([0-9.]+) ms")
DETAIL_RE = re.compile(
    r"ds4: dspark timing detail pushes_init=(\d+) pushes_verify=(\d+) "
    r"push_init=([0-9.]+) ms push_verify=([0-9.]+) ms "
    r"verify_decode=([0-9.]+) ms logits_read=([0-9.]+) ms")
CONF_RE = re.compile(
    r"ds4: dspark drafted=(\d+) verify=(\d+) verified=(\d+) accepted=(\d+) "
    r"conf0=([0-9.eE+-]+) conf1=([0-9.eE+-]+) conf2=([0-9.eE+-]+)")


def summarize(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def run_config(label: str, prompt: Path, ctx: int, verify_k: str) -> dict:
    env = dict(os.environ)
    env.update({
        "DS4_DSPARK_TIMING": "1",
        "DS4_DSPARK_SPEC_LOG": "1",
    })
    if SCHEDULE_BATCHED:
        env["DS4_DSPARK_SCHEDULE_BATCHED"] = SCHEDULE_BATCHED
    if verify_k != "sched":
        env["DS4_DSPARK_VERIFY_K"] = verify_k
    cmd = [
        DS4, "--backend", "metal", "-m", MODEL, "--dspark", DSPARK,
        "-c", str(ctx), "-n", str(N), "--temp", TEMP, "--seed", SEED,
        "--prompt-file", str(prompt),
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
        "verify_k": verify_k,
        "returncode": proc.returncode,
        "wall_s": round(dt, 2),
        "n": N,
        "schedule_batched": bool(SCHEDULE_BATCHED and SCHEDULE_BATCHED not in ("0", "off", "OFF")),
    }
    tps = TPS_RE.search(proc.stderr)
    if tps:
        result["prefill_tps"] = float(tps.group(1))
        result["gen_tps"] = float(tps.group(2))

    timings = []
    details = []
    confs = []
    for line in proc.stderr.splitlines():
        mt = TIMING_RE.search(line)
        if mt:
            timings.append({
                "drafted": int(mt.group(1)),
                "verify_n": int(mt.group(2)),
                "verified": int(mt.group(3)),
                "decode_ms": float(mt.group(4)),
                "draft_ms": float(mt.group(5)),
                "verify_ms": float(mt.group(6)),
                "total_ms": float(mt.group(7)),
            })
            continue
        md = DETAIL_RE.search(line)
        if md:
            details.append({
                "pushes_init": int(md.group(1)),
                "pushes_verify": int(md.group(2)),
                "push_init_ms": float(md.group(3)),
                "push_verify_ms": float(md.group(4)),
                "verify_decode_ms": float(md.group(5)),
                "logits_read_ms": float(md.group(6)),
            })
            continue
        mc = CONF_RE.search(line)
        if mc:
            confs.append({
                "drafted": int(mc.group(1)),
                "verify_n": int(mc.group(2)),
                "verified": int(mc.group(3)),
                "accepted": int(mc.group(4)),
                "conf0": float(mc.group(5)),
                "conf1": float(mc.group(6)),
                "conf2": float(mc.group(7)),
            })
    result["n_cycles"] = len(timings)
    if timings:
        for key in ("drafted", "verify_n", "verified", "decode_ms", "draft_ms", "verify_ms", "total_ms"):
            stats = summarize([t[key] for t in timings])
            if stats:
                result[f"{key}_summary"] = stats
    if details:
        for key in ("pushes_init", "pushes_verify", "push_init_ms", "push_verify_ms",
                    "verify_decode_ms", "logits_read_ms"):
            stats = summarize([d[key] for d in details])
            if stats:
                result[f"{key}_summary"] = stats
    if confs:
        for key in ("conf0", "conf1", "conf2"):
            stats = summarize([c[key] for c in confs])
            if stats:
                result[f"{key}_summary"] = stats
    return result


def main() -> int:
    results = []
    for prompt_label, prompt_path, ctx in PROMPTS:
        print(f"\n##### prompt={prompt_label} #####", flush=True)
        for verify_k in VERIFY_KS:
            label = f"{prompt_label}__{verify_k}"
            if PROFILE_TAG:
                label = f"{label}__{PROFILE_TAG}"
            r = run_config(label, prompt_path, ctx, verify_k)
            r["prompt_label"] = prompt_label
            results.append(r)
            print(
                f"  verify={verify_k}: rc={r['returncode']} gen={r.get('gen_tps')} "
                f"verified_med={r.get('verified_summary', {}).get('median')} "
                f"total_med={r.get('total_ms_summary', {}).get('median')}",
                flush=True,
            )
    summary_name = "summary.json" if not PROFILE_TAG else f"summary__{PROFILE_TAG}.json"
    (OUT / summary_name).write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
