#!/usr/bin/env python3
"""Capture target hidden states once, then compare DSpark drafters on them.

The sweep produces one bundle per prompt-token frontier:

- exact rendered prompt file
- target hidden-state dumps + top-k verifier distributions
- baseline probe log + q-dump + B2 summary
- candidate probe log + q-dump + B2 summary
- aggregate JSON/TSV summary across contexts

Primary use: compare baseline ``dspark.gguf`` vs an imatrix Q4_K candidate on
the same 8k..64k capture points.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

ISSUE468 = Path(__file__).resolve().parent
ROOT = ISSUE468.parent
DEFAULT_MODEL = (ROOT / ".." / "ds4" / "gguf" / "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf").resolve()
DEFAULT_BASELINE_DSPARK = (ROOT / ".." / "ds4" / "gguf" / "dspark.gguf").resolve()
DEFAULT_OUT_DIR = ISSUE468 / "baseline" / "dspark_imatrix_sweep"
DEFAULT_MEASURE_PYTHON = str(ISSUE468 / ".venv" / "bin" / "python") \
    if (ISSUE468 / ".venv" / "bin" / "python").exists() else sys.executable
DEFAULT_SYSTEM = (
    "You are DeepSeek V4 Flash running locally. Answer accurately, preserve "
    "technical details, and use tools only when the prompt asks for tool use."
)
BOS = "<｜begin▁of▁sentence｜>"
USER = "<｜User｜>"
ASSISTANT = "<｜Assistant｜>"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds4-bin", default="./ds4")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--baseline-dspark", default=str(DEFAULT_BASELINE_DSPARK))
    ap.add_argument("--candidate-dspark", required=True)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--contexts", nargs="+", type=int,
                    default=[8192, 16384, 24576, 32768, 40960, 49152, 57344, 65536])
    ap.add_argument("--steps", type=int, default=19)
    ap.add_argument("--top-k", type=int, default=128)
    ap.add_argument("--trials", type=int, default=128)
    ap.add_argument("--power", type=int, default=100)
    ap.add_argument("--ctx-headroom", type=int, default=96)
    ap.add_argument("--prompt-template", default=str(ISSUE468 / "prompts" / "code_humaneval.sh"))
    ap.add_argument("--measure-script", default=str(ISSUE468 / "baseline" / "dspark_capture" / "measure_metal_b2.py"))
    ap.add_argument("--measure-python", default=DEFAULT_MEASURE_PYTHON)
    ap.add_argument("--plain-ms", type=float, help="override decode ms/tok for speedup projection")
    ap.add_argument("--run-label", default="dspark-imatrix-q4k")
    ap.add_argument("--max-contexts", type=int, default=0, help="if >0, limit to the first N contexts")
    return ap.parse_args()


def run(cmd: list[str], *, env: dict[str, str] | None = None,
        stdout_path: Path | None = None, stderr_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    stdout = subprocess.PIPE if stdout_path is None else open(stdout_path, "w", encoding="utf-8")
    stderr = subprocess.PIPE if stderr_path is None else open(stderr_path, "w", encoding="utf-8")
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    finally:
        if stdout_path is not None:
            stdout.close()
        if stderr_path is not None:
            stderr.close()
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc


def load_code_template(path: str) -> str:
    proc = subprocess.run(
        ["bash", path],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return proc.stdout.strip()


def render_prompt(user_content: str) -> str:
    return f"{BOS}{DEFAULT_SYSTEM}{USER}{user_content}{ASSISTANT}</think>"


def token_count(ds4_bin: str, model: str, rendered_prompt: str) -> int:
    tmp = ROOT / ".tmp_dspark_acceptance_prompt.txt"
    tmp.write_text(rendered_prompt, encoding="utf-8")
    try:
        proc = run(
            [ds4_bin, "-m", model, "--dump-tokens", "--prompt-file", str(tmp)],
        )
    finally:
        tmp.unlink(missing_ok=True)
    first = proc.stdout.splitlines()[0]
    return len(ast.literal_eval(first))


def build_exact_prompt(ds4_bin: str, model: str, template_text: str, target_tokens: int) -> tuple[str, int]:
    intro = (
        "Read the following repeated coding tasks as one long context. "
        "Continue the final coding task and keep the continuation code-only.\n\n"
    )
    block = template_text.strip() + "\n"
    fillers = [
        "\n",
        "\n\n",
        " a",
        " the",
        " and",
        " code",
        " pass",
        " return",
        " x",
        " y",
        " z",
        " 0",
        " 1",
        ".",
        ":",
        ",",
        "\npass",
        "\nreturn 0",
        "\n# filler",
        "\n# note",
        "\n\npass",
        "\n\nreturn 0",
    ]
    cache: dict[str, int] = {}

    def count_for_body(body: str) -> int:
        rendered = render_prompt(body)
        if rendered not in cache:
            cache[rendered] = token_count(ds4_bin, model, rendered)
        return cache[rendered]

    lo, hi = 0, 1
    while count_for_body(intro + block * hi) < target_tokens:
        lo, hi = hi, hi * 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if count_for_body(intro + block * mid) <= target_tokens:
            lo = mid
        else:
            hi = mid

    body = intro + block * lo
    cur = count_for_body(body)
    if cur > target_tokens:
        raise RuntimeError(f"prompt builder overshot {target_tokens} with base count {cur}")

    for _ in range(512):
        if cur == target_tokens:
            rendered = render_prompt(body)
            return rendered, cur
        remaining = target_tokens - cur
        best_body = None
        best_count = None
        best_delta = 0
        for filler in fillers:
            trial_body = body + filler
            trial_count = count_for_body(trial_body)
            delta = trial_count - cur
            if 0 < delta <= remaining and delta > best_delta:
                best_body = trial_body
                best_count = trial_count
                best_delta = delta
        if best_body is None:
            break
        body = best_body
        cur = best_count

    raise RuntimeError(f"failed to hit exact token target {target_tokens}; reached {cur}")


def write_greedy_json(target_json: Path, greedy_json: Path) -> list[int]:
    src = json.loads(target_json.read_text())
    greedy = [int(step["selected"]["id"]) for step in src["steps"]]
    greedy_json.write_text(json.dumps(greedy) + "\n")
    return greedy


def parse_probe_summary(log_path: Path) -> dict[str, float | int | list[int]]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"SUMMARY: greedy match (\d+)/(\d+) \(([\d.]+)%\), avg prefix ([\d.]+)/5, hist \[([0-9 ]+)\]",
        text,
    )
    if not match:
        raise RuntimeError(f"failed to parse probe summary from {log_path}")
    hist = [int(part) for part in match.group(5).split()]
    return {
        "greedy_match": int(match.group(1)),
        "greedy_total": int(match.group(2)),
        "greedy_match_pct": float(match.group(3)),
        "avg_prefix": float(match.group(4)),
        "prefix_hist": hist,
    }


def plain_ms_for_context(target_json: Path, override: float | None) -> float:
    if override is not None:
        return override
    data = json.loads(target_json.read_text())
    prompt_tokens = int(data["prompt_tokens"])
    anchors = {
        8192: 31.3,
        16384: 31.7,
        24576: 32.3,
        32768: 32.9,
        40960: 33.5,
        49152: 34.2,
        57344: 34.8,
        65536: 35.5,
    }
    if prompt_tokens in anchors:
        return anchors[prompt_tokens]
    return 35.5


def measure_probe(ds4_bin: str, model: str, dspark: str, prompt_path: Path, ctx_size: int,
                  pos0: int, cap_dir: Path, steps: int, power: int,
                  label: str, plain_ms: float, measure_script: str,
                  measure_python: str) -> dict:
    env = os.environ.copy()
    env.update({
        "DS4_DSPARK_PROBE_ACCEPT": "1",
        "DS4_DSPARK_PROBE_ONLY": "1",
        "DS4_DSPARK_PROBE_DUMP_Q": "1",
        "DS4_DSPARK_PROBE_CAPDIR": str(cap_dir),
        "DS4_DSPARK_PROBE_GREEDY": str(cap_dir / "target_greedy.json"),
        "DS4_DSPARK_PROBE_POS": str(pos0),
        "DS4_DSPARK_PROBE_ACCEPT_STEPS": str(steps),
    })
    stdout_path = cap_dir / f"{label}.probe.stdout"
    stderr_path = cap_dir / f"{label}.probe.stderr"
    run(
        [
            ds4_bin,
            "--metal",
            "-m", model,
            "--dspark", dspark,
            "--prompt-file", str(prompt_path),
            "--verifier-curve-test",
            "--ctx", str(ctx_size),
            "--power", str(power),
        ],
        env=env,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )

    q_src = cap_dir / f"metal_base_logits_{steps}steps.bin"
    q_dst = cap_dir / f"{label}.metal_base_logits_{steps}steps.bin"
    shutil.move(q_src, q_dst)
    summary = parse_probe_summary(stderr_path)
    b2_json = cap_dir / f"{label}.b2.json"
    run(
        [
            measure_python,
            measure_script,
            "--q-path", str(q_dst),
            "--target-json", str(cap_dir / "target_topk.json"),
            "--greedy-json", str(cap_dir / "target_greedy.json"),
            "--dspark", dspark,
            "--pos0", str(pos0),
            "--trials", str(args.trials),
            "--label", label,
            "--plain-ms", str(plain_ms),
            "--json-out", str(b2_json),
        ],
    )
    summary["b2"] = json.loads(b2_json.read_text())
    summary["q_path"] = str(q_dst)
    return summary


def collect_target_bundle(ds4_bin: str, model: str, prompt_path: Path, ctx_size: int, steps: int, top_k: int,
                          power: int, cap_dir: Path) -> dict:
    env = os.environ.copy()
    env.update({
        "DS4_METAL_GRAPH_DUMP_PREFIX": str(cap_dir / "hc"),
        "DS4_METAL_GRAPH_DUMP_NAME": "dspark_main_hc",
    })
    stdout_path = cap_dir / "target.dump.stdout"
    stderr_path = cap_dir / "target.dump.stderr"
    target_json = cap_dir / "target_topk.json"
    run(
        [
            ds4_bin,
            "--metal",
            "-m", model,
            "--prompt-file", str(prompt_path),
            "--dump-logprobs", str(target_json),
            "--logprobs-top-k", str(top_k),
            "--tokens", str(steps + 8),
            "--ctx", str(ctx_size),
            "--temp", "0",
            "--power", str(power),
        ],
        env=env,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    greedy = write_greedy_json(target_json, cap_dir / "target_greedy.json")
    target = json.loads(target_json.read_text())
    prompt_tokens = int(target["prompt_tokens"])
    return {
        "prompt_tokens": prompt_tokens,
        "greedy_len": len(greedy),
        "target_json": str(target_json),
    }


def write_summary(out_dir: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    header = [
        "context",
        "prompt_tokens",
        "baseline_avg_prefix",
        "candidate_avg_prefix",
        "avg_prefix_delta_pct",
        "baseline_b2_accepted",
        "candidate_b2_accepted",
        "b2_accepted_delta_pct",
        "baseline_b2_committed",
        "candidate_b2_committed",
        "b2_committed_delta_pct",
    ]
    lines = ["\t".join(header)]
    for row in rows:
        lines.append("\t".join(str(row[key]) for key in header))
    (out_dir / "summary.tsv").write_text("\n".join(lines) + "\n")


def main() -> int:
    global args
    args = parse_args()
    ds4_bin = str(Path(args.ds4_bin).resolve()) if "/" in args.ds4_bin else args.ds4_bin
    model = str(Path(args.model).resolve())
    baseline = str(Path(args.baseline_dspark).resolve())
    candidate = str(Path(args.candidate_dspark).resolve())
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    template_text = load_code_template(args.prompt_template)
    contexts = list(args.contexts)
    if args.max_contexts > 0:
        contexts = contexts[:args.max_contexts]

    rows = []
    for ctx in contexts:
        ctx_dir = out_dir / f"ctx_{ctx:05d}"
        ctx_dir.mkdir(parents=True, exist_ok=True)
        rendered, prompt_tokens = build_exact_prompt(ds4_bin, model, template_text, ctx)
        prompt_path = ctx_dir / "prompt_rendered.txt"
        prompt_path.write_text(rendered, encoding="utf-8")
        ctx_size = prompt_tokens + args.ctx_headroom
        bundle = collect_target_bundle(
            ds4_bin=ds4_bin,
            model=model,
            prompt_path=prompt_path,
            ctx_size=ctx_size,
            steps=args.steps,
            top_k=args.top_k,
            power=args.power,
            cap_dir=ctx_dir,
        )
        if bundle["prompt_tokens"] != ctx:
            raise RuntimeError(
                f"target bundle prompt_tokens={bundle['prompt_tokens']} but expected {ctx}"
            )
        plain_ms = plain_ms_for_context(ctx_dir / "target_topk.json", args.plain_ms)
        baseline_summary = measure_probe(
            ds4_bin=ds4_bin,
            model=model,
            dspark=baseline,
            prompt_path=prompt_path,
            ctx_size=ctx_size,
            pos0=ctx,
            cap_dir=ctx_dir,
            steps=args.steps,
            power=args.power,
            label="baseline",
            plain_ms=plain_ms,
            measure_script=args.measure_script,
            measure_python=args.measure_python,
        )
        candidate_summary = measure_probe(
            ds4_bin=ds4_bin,
            model=model,
            dspark=candidate,
            prompt_path=prompt_path,
            ctx_size=ctx_size,
            pos0=ctx,
            cap_dir=ctx_dir,
            steps=args.steps,
            power=args.power,
            label="candidate",
            plain_ms=plain_ms,
            measure_script=args.measure_script,
            measure_python=args.measure_python,
        )
        row = {
            "context": ctx,
            "prompt_tokens": ctx,
            "baseline_avg_prefix": round(float(baseline_summary["avg_prefix"]), 4),
            "candidate_avg_prefix": round(float(candidate_summary["avg_prefix"]), 4),
            "avg_prefix_delta_pct": round(
                100.0 * (
                    float(candidate_summary["avg_prefix"]) /
                    max(float(baseline_summary["avg_prefix"]), 1e-9) - 1.0
                ),
                3,
            ),
            "baseline_b2_accepted": round(float(baseline_summary["b2"]["average_accepted"]), 4),
            "candidate_b2_accepted": round(float(candidate_summary["b2"]["average_accepted"]), 4),
            "b2_accepted_delta_pct": round(
                100.0 * (
                    float(candidate_summary["b2"]["average_accepted"]) /
                    max(float(baseline_summary["b2"]["average_accepted"]), 1e-9) - 1.0
                ),
                3,
            ),
            "baseline_b2_committed": round(float(baseline_summary["b2"]["average_committed"]), 4),
            "candidate_b2_committed": round(float(candidate_summary["b2"]["average_committed"]), 4),
            "b2_committed_delta_pct": round(
                100.0 * (
                    float(candidate_summary["b2"]["average_committed"]) /
                    max(float(baseline_summary["b2"]["average_committed"]), 1e-9) - 1.0
                ),
                3,
            ),
        }
        rows.append(row)
        write_summary(out_dir, rows)
        print(
            f"ctx={ctx} prefix {row['baseline_avg_prefix']:.3f}->{row['candidate_avg_prefix']:.3f} "
            f"({row['avg_prefix_delta_pct']:+.2f}%) "
            f"B2 accepted {row['baseline_b2_accepted']:.3f}->{row['candidate_b2_accepted']:.3f} "
            f"({row['b2_accepted_delta_pct']:+.2f}%)"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
