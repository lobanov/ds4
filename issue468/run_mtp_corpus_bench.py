#!/usr/bin/env python3
"""Checkpointable corpus benchmark for the powered 300-prompt prompt set.

Benchmarks baseline, shipped MTP, and exact anchor-reuse MTP on the combined
Stage 2 (240) + Lead 03 (60) corpus. Each prompt is written as an individual
JSON record so the run is resumable.
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
OUT = ROOT / "issue468/artifacts/mtp_corpus_bench"
OUT.mkdir(parents=True, exist_ok=True)
RUNS = OUT / "runs"
RUNS.mkdir(parents=True, exist_ok=True)
DS4 = str(ROOT / "ds4")
MODEL = "/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
MTP = "/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"
MANIFESTS = [
    ROOT / "issue468/prompts/stage2_corpus/manifest.json",
    ROOT / "issue468/prompts/lead3_corpus/manifest.json",
]
N = int(os.getenv("DS4_MTP_CORPUS_N", "128"))
TEMP = os.getenv("DS4_MTP_CORPUS_TEMP", "0.0")
SEED = os.getenv("DS4_MTP_CORPUS_SEED", "1")
K = int(os.getenv("DS4_MTP_CORPUS_K", "4"))
LIMIT = int(os.getenv("DS4_MTP_CORPUS_LIMIT", "0"))

TPS_RE = re.compile(r"prefill:\s*([0-9.]+) t/s, generation:\s*([0-9.]+) t/s")


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
    return rec


def record_path(prompt_id: str, impl: str) -> Path:
    return RUNS / f"{prompt_id}__{impl}.json"


def ctx_for_prompt(item: dict) -> int:
    # The powered corpus is short-form capture-style prompting; 8k is ample.
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
            configs = [
                ("baseline", [], {}),
                ("shipped_k4", ["--mtp", MTP, "--mtp-draft", str(K)],
                 {"DS4_MTP_TIMING": "1", "DS4_MTP_SPEC_LOG": "1"}),
                ("anchor_reuse_exact_k4", ["--mtp", MTP, "--mtp-draft", str(K)],
                 {"DS4_MTP_TIMING": "1", "DS4_MTP_SPEC_LOG": "1", "DS4_MTP_ANCHOR_REUSE": "1"}),
            ]
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
                    "K": K if impl != "baseline" else "baseline",
                    "prompt_file": item["file"],
                    **run_config(prompt_path, ctx, args, env),
                }
                out_path.write_text(json.dumps(rec, indent=2) + "\n")
                results.append(rec)
                write_summary(results)
                print(f"  {impl}: rc={rec['returncode']} gen={rec.get('gen_tps')}", flush=True)
    except KeyboardInterrupt:
        write_summary(results)
        print("\nInterrupted; partial summary.json updated from checkpointed results.", file=sys.stderr)
        return 130
    write_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
