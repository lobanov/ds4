#!/usr/bin/env python3
"""Checkpointable corpus benchmark for the current DSpark path.

Benchmarks baseline, DSpark default scheduling, and optional fixed verify-K
over the combined Stage 2 + Lead 03 corpus.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "issue468/artifacts/dspark_corpus_bench"
OUT.mkdir(parents=True, exist_ok=True)
RUNS = OUT / "runs"
RUNS.mkdir(parents=True, exist_ok=True)
DS4 = str(ROOT / "ds4")
MODEL = "/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
DSPARK = os.getenv("DS4_TEST_DSPARK", "/Users/lobanov/Projects/ds4/gguf/dspark.gguf")
MANIFESTS = [
    ROOT / "issue468/prompts/stage2_corpus/manifest.json",
    ROOT / "issue468/prompts/lead3_corpus/manifest.json",
]
N = int(os.getenv("DS4_DSPARK_CORPUS_N", "128"))
TEMP = os.getenv("DS4_DSPARK_CORPUS_TEMP", "0.0")
SEED = os.getenv("DS4_DSPARK_CORPUS_SEED", "1")
LIMIT = int(os.getenv("DS4_DSPARK_CORPUS_LIMIT", "0"))
FIXED_VERIFY = [k.strip() for k in os.getenv("DS4_DSPARK_CORPUS_VERIFY_KS", "1").split(",") if k.strip()]

TPS_RE = re.compile(r"prefill:\s*([0-9.]+) t/s, generation:\s*([0-9.]+) t/s")
TIMING_RE = re.compile(
    r"ds4: dspark timing drafted=(\d+) verify=(\d+) verified=(\d+) "
    r"decode=([0-9.]+) ms draft=([0-9.]+) ms verify=([0-9.]+) ms total=([0-9.]+) ms")
DETAIL_RE = re.compile(
    r"ds4: dspark timing detail pushes_init=(\d+) pushes_verify=(\d+) "
    r"push_init=([0-9.]+) ms push_verify=([0-9.]+) ms "
    r"verify_decode=([0-9.]+) ms logits_read=([0-9.]+) ms")


def load_prompts() -> list[dict]:
    prompts = []
    for manifest_path in MANIFESTS:
        data = json.loads(manifest_path.read_text())
        prompts.extend(data["prompts"])
    return prompts[:LIMIT] if LIMIT > 0 else prompts


def run_config(prompt_path: Path, ctx: int, extra_args: list[str], env_extra: dict[str, str]) -> dict:
    env = dict(os.environ)
    env.update(env_extra)
    cmd = [
        DS4, "--backend", "metal", "-m", MODEL, "-c", str(ctx), "-n", str(N),
        "--temp", TEMP, "--seed", SEED, "--prompt-file", str(prompt_path), *extra_args,
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    dt = time.time() - t0
    rec = {
        "returncode": proc.returncode,
        "wall_s": round(dt, 2),
    }
    m = TPS_RE.search(proc.stderr)
    if m:
        rec["prefill_tps"] = float(m.group(1))
        rec["gen_tps"] = float(m.group(2))
    timings = []
    details = []
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
    rec["n_cycles"] = len(timings)
    if timings:
        for key in ("drafted", "verify_n", "verified", "decode_ms", "draft_ms", "verify_ms", "total_ms"):
            vals = [t[key] for t in timings]
            rec[f"mean_{key}"] = round(sum(vals) / len(vals), 3)
    if details:
        for key in ("pushes_init", "pushes_verify", "push_init_ms", "push_verify_ms",
                    "verify_decode_ms", "logits_read_ms"):
            vals = [d[key] for d in details]
            rec[f"mean_{key}"] = round(sum(vals) / len(vals), 3)
    return rec


def record_path(prompt_id: str, impl: str) -> Path:
    return RUNS / f"{prompt_id}__{impl}.json"


def ctx_for_prompt(item: dict) -> int:
    return 8192


def write_summary(results: list[dict]) -> None:
    (OUT / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


def main() -> int:
    prompts = load_prompts()
    results = []
    try:
        for idx, item in enumerate(prompts, start=1):
            prompt_id = item["prompt_id"]
            prompt_path = ROOT / "issue468" / item["file"]
            ctx = ctx_for_prompt(item)
            print(f"\n##### {idx}/{len(prompts)} {prompt_id} #####", flush=True)
            configs = [("baseline", [], {})]
            configs.append((
                "dspark_sched",
                ["--dspark", DSPARK],
                {"DS4_DSPARK_TIMING": "1", "DS4_DSPARK_SPEC_LOG": "1"},
            ))
            for k in FIXED_VERIFY:
                configs.append((
                    f"dspark_k{k}",
                    ["--dspark", DSPARK],
                    {
                        "DS4_DSPARK_TIMING": "1",
                        "DS4_DSPARK_SPEC_LOG": "1",
                        "DS4_DSPARK_VERIFY_K": k,
                    },
                ))
            for impl, args, env in configs:
                out_path = record_path(prompt_id, impl)
                if out_path.exists():
                    rec = json.loads(out_path.read_text())
                    results.append(rec)
                    print(f"  {impl}: skip rc={rec['returncode']} gen={rec.get('gen_tps')}", flush=True)
                    continue
                rec = {
                    "prompt_id": prompt_id,
                    "source": item.get("source"),
                    "split": item.get("split"),
                    "impl": impl,
                    "prompt_file": item["file"],
                    **run_config(prompt_path, ctx, args, env),
                }
                out_path.write_text(json.dumps(rec, indent=2) + "\n")
                results.append(rec)
                write_summary(results)
                print(
                    f"  {impl}: rc={rec['returncode']} gen={rec.get('gen_tps')} "
                    f"verified={rec.get('mean_verified')} total_ms={rec.get('mean_total_ms')}",
                    flush=True,
                )
    except KeyboardInterrupt:
        write_summary(results)
        print("\nInterrupted; partial summary.json updated from checkpointed results.", file=sys.stderr)
        return 130
    write_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
