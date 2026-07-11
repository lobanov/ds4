#!/usr/bin/env python3
"""MTP verifier benchmark on LONGER prompts, fine K=2..6 sweep.

Grid: 3 prompt families at the 8k length class x {baseline, K=2,3,4,5,6}.
Extends run_mtp_verifier_bench.py (code_4k, K in {2,4,8,16}) with longer prompts
and finer resolution around the K=2-6 optimum. Results in
issue468/artifacts/mtp_verifier_bench_long/.
"""
from __future__ import annotations
import json, os, re, statistics, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "issue468/artifacts/mtp_verifier_bench_long"
OUT.mkdir(parents=True, exist_ok=True)
DS4 = str(ROOT / "ds4")
MODEL = "/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
MTP = "/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"
CORPUS = ROOT / "issue468/prompts/baseline_corpus"
N = 128
TEMP = "0.0"
SEED = "1"
KS = [2, 3, 4, 5, 6]
INCLUDE_ANCHOR_REUSE = os.getenv("DS4_MTP_BENCH_INCLUDE_ANCHOR_REUSE") not in (None, "", "0")
# (label, prompt path, ctx). 8k prompts (~8.2k tokens) need ctx well above that.
PROMPTS = [
    ("code_8k", CORPUS / "code_8k.txt", 16384),
    ("synthesis_8k", CORPUS / "synthesis_8k.txt", 16384),
    ("grounded_8k", CORPUS / "grounded_8k.txt", 16384),
]

TPS_RE = re.compile(r"prefill:\s*([0-9.]+) t/s, generation:\s*([0-9.]+) t/s")
TIMING_RE = re.compile(
    r"ds4: mtp timing (\S+) drafted=(\d+) (committed|verified)=(\d+).*?draft=([0-9.]+) ms.*?"
    r"verify=([0-9.]+) ms.*?total=([0-9.]+) ms")


def run_config(label: str, prompt: Path, ctx: int, extra_args: list[str], env_extra: dict) -> dict:
    env = dict(os.environ); env.update(env_extra)
    cmd = [DS4, "--backend", "metal", "-m", MODEL, "-c", str(ctx), "-n", str(N),
           "--temp", TEMP, "--seed", SEED, "--prompt-file", str(prompt), *extra_args]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    dt = time.time() - t0
    (OUT / f"{label}.stderr").write_text(proc.stderr)
    (OUT / f"{label}.stdout").write_text(proc.stdout)
    r: dict = {"label": label, "args": extra_args, "returncode": proc.returncode,
               "wall_s": round(dt, 2), "ctx": ctx, "n": N, "temp": TEMP, "seed": SEED,
               "prompt": str(prompt.relative_to(ROOT))}
    m = TPS_RE.search(proc.stderr)
    r["prefill_tps"] = float(m.group(1)) if m else None
    r["gen_tps"] = float(m.group(2)) if m else None
    timings = []
    for line in proc.stderr.splitlines():
        mt = TIMING_RE.search(line)
        if mt:
            timings.append({"kind": mt.group(1), "drafted": int(mt.group(2)),
                            "count_kind": mt.group(3), "committed": int(mt.group(4)),
                            "draft_ms": float(mt.group(5)),
                            "verify_ms": float(mt.group(6)), "total_ms": float(mt.group(7))})
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
    env_reuse = dict(env_t)
    env_reuse["DS4_MTP_ANCHOR_REUSE"] = "1"
    results = []
    for plabel, ppath, ctx in PROMPTS:
        print(f"\n##### prompt={plabel} (ctx={ctx}) #####", flush=True)
        # baseline
        r = run_config(f"{plabel}__baseline", ppath, ctx, [], {})
        r["prompt_label"] = plabel; r["K"] = "baseline"; r["impl"] = "baseline"
        results.append(r)
        print(f"  baseline: rc={r['returncode']} wall={r['wall_s']}s gen={r.get('gen_tps')}", flush=True)
        for K in KS:
            r = run_config(f"{plabel}__k{K}", ppath, ctx, ["--mtp", MTP, "--mtp-draft", str(K)], env_t)
            r["prompt_label"] = plabel; r["K"] = K; r["impl"] = "shipped"
            results.append(r)
            print(f"  K={K}: rc={r['returncode']} wall={r['wall_s']}s gen={r.get('gen_tps')} "
                  f"verify_med={r.get('median_verify_ms')} committed={r.get('mean_committed')} "
                  f"miss={r.get('spec_miss_first')}/{r.get('n_cycles')}", flush=True)
            if r["returncode"] != 0:
                print("    FAILED; see stderr", flush=True)
            if not INCLUDE_ANCHOR_REUSE:
                continue
            rr = run_config(f"{plabel}__reuse_k{K}", ppath, ctx,
                            ["--mtp", MTP, "--mtp-draft", str(K)], env_reuse)
            rr["prompt_label"] = plabel; rr["K"] = K; rr["impl"] = "anchor_reuse_exact"
            results.append(rr)
            print(f"  reuse K={K}: rc={rr['returncode']} wall={rr['wall_s']}s gen={rr.get('gen_tps')} "
                  f"verify_med={rr.get('median_verify_ms')} committed={rr.get('mean_committed')} "
                  f"miss={rr.get('spec_miss_first')}/{rr.get('n_cycles')}", flush=True)
            if rr["returncode"] != 0:
                print("    FAILED; see stderr", flush=True)
    (OUT / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    cols = ["prompt_label", "impl", "K", "returncode", "wall_s", "prefill_tps", "gen_tps",
            "n_cycles", "mean_drafted", "mean_committed", "mean_draft_ms",
            "mean_verify_ms", "median_verify_ms", "mean_total_ms", "spec_miss_first"]
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
