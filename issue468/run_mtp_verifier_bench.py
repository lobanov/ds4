#!/usr/bin/env python3
"""Benchmark the MTP bulk draft verifier vs plain decode.

Runs three configs over the same short prompt:
  - baseline : no --mtp                 -> plain decode t/s (the decode-time reference)
  - mtp_k2   : --mtp --mtp-draft 2      -> decode2 verifier (DS4_MTP_TIMING draft/verify split)
  - mtp_k4   : --mtp --mtp-draft 4      -> micro verifier (DS4_MTP_TIMING draft/verify split)

Captures per-config: prefill/generation t/s, wall time, and per-cycle
draft/snapshot/verify/total ms + drafted/committed counts (mean verify ms and
its flatness across K is the bandwidth-floor signal). Results land in
issue468/artifacts/mtp_verifier_bench/.
"""
from __future__ import annotations
import json, os, re, statistics, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "issue468/artifacts/mtp_verifier_bench"
OUT.mkdir(parents=True, exist_ok=True)
DS4 = str(ROOT / "ds4")
MODEL = "/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
MTP = "/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"
PROMPT = str(ROOT / "issue468/prompts/baseline_corpus/code_4k.txt")
CTX = 8192
N = 96
TEMP = "0.0"
SEED = "1"

TPS_RE = re.compile(r"prefill:\s*([0-9.]+) t/s, generation:\s*([0-9.]+) t/s")
# matches: ds4: mtp timing <kind> drafted=D committed=C ... draft=X ms ... verify=Y ms ... total=Z ms
TIMING_RE = re.compile(
    r"ds4: mtp timing (\S+) drafted=(\d+) committed=(\d+).*?draft=([0-9.]+) ms.*?"
    r"verify=([0-9.]+) ms.*?total=([0-9.]+) ms")


def run_config(label: str, extra_args: list[str], env_extra: dict[str, str]) -> dict:
    env = dict(os.environ)
    env.update(env_extra)
    cmd = [DS4, "--backend", "metal", "-m", MODEL, "-c", str(CTX), "-n", str(N),
           "--temp", TEMP, "--seed", SEED, "--prompt-file", PROMPT, *extra_args]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    dt = time.time() - t0
    (OUT / f"{label}.stderr").write_text(proc.stderr)
    (OUT / f"{label}.stdout").write_text(proc.stdout)
    r: dict = {"label": label, "args": extra_args, "env": env_extra,
               "returncode": proc.returncode, "wall_s": round(dt, 2),
               "ctx": CTX, "n": N, "temp": TEMP, "seed": SEED,
               "prompt": PROMPT}
    m = TPS_RE.search(proc.stderr)
    r["prefill_tps"] = float(m.group(1)) if m else None
    r["gen_tps"] = float(m.group(2)) if m else None
    timings = []
    for line in proc.stderr.splitlines():
        mt = TIMING_RE.search(line)
        if mt:
            timings.append({"kind": mt.group(1), "drafted": int(mt.group(2)),
                            "committed": int(mt.group(3)), "draft_ms": float(mt.group(4)),
                            "verify_ms": float(mt.group(5)), "total_ms": float(mt.group(6))})
    r["n_cycles"] = len(timings)
    if timings:
        for k in ("draft_ms", "verify_ms", "total_ms"):
            v = [t[k] for t in timings]
            r[f"mean_{k}"] = round(statistics.mean(v), 3)
            r[f"median_{k}"] = round(statistics.median(v), 3)
        r["mean_drafted"] = round(statistics.mean(t["drafted"] for t in timings), 2)
        r["mean_committed"] = round(statistics.mean(t["committed"] for t in timings), 2)
    r["spec_miss_first"] = proc.stderr.count("spec miss first draft")
    return r


def main() -> int:
    env_t = {"DS4_MTP_TIMING": "1", "DS4_MTP_SPEC_LOG": "1"}
    configs = [
        ("baseline", [], {}),
        ("mtp_k2", ["--mtp", MTP, "--mtp-draft", "2"], env_t),
        ("mtp_k4", ["--mtp", MTP, "--mtp-draft", "4"], env_t),
        ("mtp_k8", ["--mtp", MTP, "--mtp-draft", "8"], env_t),
        ("mtp_k16", ["--mtp", MTP, "--mtp-draft", "16"], env_t),
    ]
    results = []
    for label, args, env in configs:
        print(f"=== {label} ===", flush=True)
        r = run_config(label, args, env)
        results.append(r)
        print(f"  rc={r['returncode']} wall={r['wall_s']}s prefill={r.get('prefill_tps')} "
              f"gen={r.get('gen_tps')} cycles={r.get('n_cycles')} "
              f"mean_draft={r.get('mean_draft_ms')} mean_verify={r.get('mean_verify_ms')} "
              f"mean_committed={r.get('mean_committed')}", flush=True)
        if r["returncode"] != 0:
            print("  FAILED; see stderr log", flush=True)
    (OUT / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    cols = ["label", "returncode", "wall_s", "prefill_tps", "gen_tps", "n_cycles",
            "mean_drafted", "mean_committed", "mean_draft_ms", "mean_verify_ms",
            "median_verify_ms", "mean_total_ms", "spec_miss_first"]
    lines = [",".join(cols)]
    for r in results:
        lines.append(",".join(str(r.get(c)) for c in cols))
    (OUT / "summary.csv").write_text("\n".join(lines) + "\n")
    print("\n=== TABLE ===")
    for ln in lines:
        print(ln)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
