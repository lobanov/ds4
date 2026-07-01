#!/usr/bin/env python3
"""Build and evaluate a weighted DSpark candidate from an existing sweep root.

Inputs:
- a sweep root containing ``ctx_#####`` bundle directories with:
  - ``prompt_rendered.txt``
  - ``target_topk.json``
  - ``target_greedy.json``
  - ``hc_dspark_main_hc-*``

Pipeline:
1. collect one aggregated weighted DSpark imatrix from the sweep root
2. quantize one candidate DSpark GGUF from that imatrix
3. direct-reprobe every context bundle against baseline and candidate
4. write aggregate JSON/TSV summary
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ISSUE468 = ROOT / "issue468"
DEFAULT_MODEL = (ROOT / ".." / "ds4" / "gguf" / "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf").resolve()
DEFAULT_BASELINE_DSPARK = (ROOT / ".." / "ds4" / "gguf" / "dspark.gguf").resolve()
DEFAULT_HF_DSPARK = (ROOT / ".." / "ds4" / "hf-dspark").resolve()
DEFAULT_MEASURE = ISSUE468 / "baseline" / "dspark_capture" / "measure_metal_b2.py"
DEFAULT_MEASURE_PYTHON = ISSUE468 / ".venv" / "bin" / "python"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-root", required=True)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--baseline-dspark", default=str(DEFAULT_BASELINE_DSPARK))
    ap.add_argument("--hf-dspark", default=str(DEFAULT_HF_DSPARK))
    ap.add_argument("--template-dspark", default=str(DEFAULT_BASELINE_DSPARK))
    ap.add_argument("--measure-script", default=str(DEFAULT_MEASURE))
    ap.add_argument("--measure-python", default=str(DEFAULT_MEASURE_PYTHON))
    ap.add_argument("--draft-pos-weights", default="1,0.75,0.5,0.33,0.2")
    ap.add_argument("--collector-max-tokens", type=int, default=0)
    ap.add_argument("--ctx-size", type=int, default=4096)
    ap.add_argument("--power", type=int, default=100)
    ap.add_argument("--steps", type=int, default=19)
    ap.add_argument("--trials", type=int, default=128)
    ap.add_argument("--run-label", default="weighted-root")
    return ap.parse_args()


def run(cmd: list[str], *, env: dict[str, str] | None = None,
        stdout_path: Path | None = None, stderr_path: Path | None = None) -> None:
    stdout = subprocess.PIPE if stdout_path is None else open(stdout_path, "w", encoding="utf-8")
    stderr = subprocess.PIPE if stderr_path is None else open(stderr_path, "w", encoding="utf-8")
    try:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=stdout, stderr=stderr, check=False)
    finally:
        if stdout_path is not None:
            stdout.close()
        if stderr_path is not None:
            stderr.close()
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def plain_ms_for_context(target_json: Path) -> float:
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
    return anchors.get(prompt_tokens, 35.5)


def bundle_dirs(root: Path) -> list[Path]:
    dirs = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if not re.fullmatch(r"ctx_\d{5}", child.name):
            continue
        if (child / "target_topk.json").exists() and (child / "target_greedy.json").exists():
            dirs.append(child)
    return sorted(dirs)


def reprobe_bundle(bundle: Path, *, model: str, dspark: str, steps: int, power: int,
                   trials: int, measure_python: str, measure_script: str, label: str) -> dict:
    target = json.loads((bundle / "target_topk.json").read_text())
    pos0 = int(target["prompt_tokens"])
    ctx_size = pos0 + 96
    plain_ms = plain_ms_for_context(bundle / "target_topk.json")
    env = os.environ.copy()
    env.update({
        "DS4_DSPARK_PROBE_ACCEPT": "1",
        "DS4_DSPARK_PROBE_ONLY": "1",
        "DS4_DSPARK_PROBE_DUMP_Q": "1",
        "DS4_DSPARK_PROBE_CAPDIR": str(bundle),
        "DS4_DSPARK_PROBE_GREEDY": str(bundle / "target_greedy.json"),
        "DS4_DSPARK_PROBE_POS": str(pos0),
        "DS4_DSPARK_PROBE_ACCEPT_STEPS": str(steps),
    })
    stdout_path = bundle / f"{label}.probe.stdout"
    stderr_path = bundle / f"{label}.probe.stderr"
    run(
        [
            "./ds4",
            "--metal",
            "-m", model,
            "--dspark", dspark,
            "--prompt-file", str(bundle / "prompt_rendered.txt"),
            "--verifier-curve-test",
            "--ctx", str(ctx_size),
            "--power", str(power),
        ],
        env=env,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    q_src = bundle / f"metal_base_logits_{steps}steps.bin"
    q_dst = bundle / f"{label}.metal_base_logits_{steps}steps.bin"
    shutil.move(q_src, q_dst)
    b2_json = bundle / f"{label}.b2.json"
    run(
        [
            measure_python,
            measure_script,
            "--q-path", str(q_dst),
            "--target-json", str(bundle / "target_topk.json"),
            "--greedy-json", str(bundle / "target_greedy.json"),
            "--dspark", dspark,
            "--pos0", str(pos0),
            "--trials", str(trials),
            "--label", label,
            "--plain-ms", str(plain_ms),
            "--json-out", str(b2_json),
        ]
    )
    return json.loads(b2_json.read_text())


def main() -> int:
    args = parse_args()
    sweep_root = Path(args.sweep_root).resolve()
    dirs = bundle_dirs(sweep_root)
    if not dirs:
        raise RuntimeError(f"no ctx_* bundles found under {sweep_root}")

    imatrix_out = sweep_root / f"{args.run_label}.imatrix.dat"
    candidate_out = sweep_root / f"{args.run_label}.gguf"

    collect_cmd = [
        "./ds4",
        "--metal",
        "-m", str(Path(args.model).resolve()),
        "--dspark", str(Path(args.baseline_dspark).resolve()),
        "--imatrix-dataset", str(sweep_root),
        "--imatrix-out", str(imatrix_out),
        "--imatrix-draft-pos-weights", args.draft_pos_weights,
        "--ctx", str(args.ctx_size),
    ]
    if args.collector_max_tokens > 0:
        collect_cmd.extend(["--imatrix-max-tokens", str(args.collector_max_tokens)])
    run(collect_cmd)

    run([
        "gguf-tools/deepseek4-quantize",
        "--hf", str(Path(args.hf_dspark).resolve()),
        "--template", str(Path(args.template_dspark).resolve()),
        "--out", str(candidate_out),
        "--overwrite",
        "--imatrix", str(imatrix_out),
    ])

    rows = []
    baseline_path = str(Path(args.baseline_dspark).resolve())
    candidate_path = str(candidate_out.resolve())
    for bundle in dirs:
        ctx = int(bundle.name.split("_")[1])
        base = reprobe_bundle(
            bundle,
            model=str(Path(args.model).resolve()),
            dspark=baseline_path,
            steps=args.steps,
            power=args.power,
            trials=args.trials,
            measure_python=args.measure_python,
            measure_script=str(Path(args.measure_script).resolve()),
            label=f"baseline-{args.run_label}",
        )
        cand = reprobe_bundle(
            bundle,
            model=str(Path(args.model).resolve()),
            dspark=candidate_path,
            steps=args.steps,
            power=args.power,
            trials=args.trials,
            measure_python=args.measure_python,
            measure_script=str(Path(args.measure_script).resolve()),
            label=f"candidate-{args.run_label}",
        )
        row = {
            "context": ctx,
            "baseline_b2_accepted": base["average_accepted"],
            "candidate_b2_accepted": cand["average_accepted"],
            "b2_accepted_delta_pct": 100.0 * (cand["average_accepted"] / max(base["average_accepted"], 1e-9) - 1.0),
            "baseline_b2_committed": base["average_committed"],
            "candidate_b2_committed": cand["average_committed"],
            "b2_committed_delta_pct": 100.0 * (cand["average_committed"] / max(base["average_committed"], 1e-9) - 1.0),
        }
        rows.append(row)
        print(json.dumps(row))

    summary_path = sweep_root / f"{args.run_label}.summary.json"
    summary_tsv = sweep_root / f"{args.run_label}.summary.tsv"
    summary_path.write_text(json.dumps(rows, indent=2) + "\n")
    lines = ["context\tbaseline_b2_accepted\tcandidate_b2_accepted\tb2_accepted_delta_pct\tbaseline_b2_committed\tcandidate_b2_committed\tb2_committed_delta_pct"]
    for row in rows:
        lines.append(
            f"{row['context']}\t{row['baseline_b2_accepted']}\t{row['candidate_b2_accepted']}\t"
            f"{row['b2_accepted_delta_pct']}\t{row['baseline_b2_committed']}\t"
            f"{row['candidate_b2_committed']}\t{row['b2_committed_delta_pct']}"
        )
    summary_tsv.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
