#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIR = ROOT / "issue468/prompts/baseline_corpus"
OUT_DIR = ROOT / "issue468/artifacts/plain_baseline_matrix"
LOG_DIR = OUT_DIR / "logs"
MODEL = "../ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
DEFAULT_CTX = 32768
DEFAULT_N = 640
DEFAULT_SEED = 1
DEFAULT_TEMPS = [0.0, 0.5, 1.0]
PREFILL_RE = re.compile(r"prefill:\s*([0-9.]+) t/s, generation:\s*([0-9.]+) t/s")
NAME_RE = re.compile(r"(code|synthesis|grounded)_(4k|8k|16k)\.txt$")


def prompt_files() -> list[Path]:
    files = []
    for p in sorted(PROMPT_DIR.glob("*.txt")):
        if NAME_RE.match(p.name):
            files.append(p)
    return files


def run(ctx: int = DEFAULT_CTX, n: int = DEFAULT_N, seed: int = DEFAULT_SEED, temps: list[float] | None = None) -> int:
    temps = DEFAULT_TEMPS if temps is None else temps
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for prompt_path in prompt_files():
        m = NAME_RE.match(prompt_path.name)
        assert m
        family, length_class = m.group(1), m.group(2)
        for temp in temps:
            tag = f"plain__{family}__{length_class}__t{temp:.1f}".replace('.', 'p')
            out_path = LOG_DIR / f"{tag}.out"
            log_path = LOG_DIR / f"{tag}.log"
            cmd = [
                "./ds4", "--backend", "metal", "-m", MODEL,
                "-c", str(ctx), "-n", str(n), "--temp", str(temp), "--seed", str(seed),
                "--prompt-file", str(prompt_path),
            ]
            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            dt = time.time() - t0
            out_path.write_text(proc.stdout)
            log_path.write_text(proc.stderr)
            row = {
                "kind": "plain",
                "prompt_file": str(prompt_path.as_posix()),
                "family": family,
                "length_class": length_class,
                "temperature": temp,
                "ctx": ctx,
                "n": n,
                "status": "ok" if proc.returncode == 0 else f"error_{proc.returncode}",
                "prefill_tps": None,
                "generation_tps": None,
                "wall_seconds": round(dt, 3),
                "stdout_bytes": len(proc.stdout.encode()),
                "log": str(log_path.as_posix()),
                "out": str(out_path.as_posix()),
            }
            if proc.returncode == 0:
                m2 = PREFILL_RE.search(proc.stderr)
                if m2:
                    row["prefill_tps"] = float(m2.group(1))
                    row["generation_tps"] = float(m2.group(2))
                else:
                    row["status"] = "parse_error"
            rows.append(row)
            print(f"plain {family} {length_class} temp={temp}: {row['status']} gen={row['generation_tps']}", flush=True)
    fieldnames = list(rows[0].keys())
    with (OUT_DIR / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    summary = {"rows": rows}
    oks = [r for r in rows if r["status"] == "ok"]
    if oks:
        summary["headline"] = {
            "avg_generation_tps": round(sum(r["generation_tps"] for r in oks if r["generation_tps"] is not None) / len(oks), 3),
            "best_cell": max(oks, key=lambda r: r["generation_tps"] or -1),
            "worst_cell": min(oks, key=lambda r: r["generation_tps"] or 1e9),
        }
    with (OUT_DIR / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
