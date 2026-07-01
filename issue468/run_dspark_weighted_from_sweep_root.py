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
DEFAULT_RECOVERABLE_GAP = ISSUE468 / "build_recoverable_gap_overlay.py"
DEFAULT_QUANTIZER = ROOT / "gguf-tools" / "deepseek4-quantize"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds4-bin", default="./ds4")
    ap.add_argument("--backend", default="metal")
    ap.add_argument("--sweep-root", required=True)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--baseline-dspark", default=str(DEFAULT_BASELINE_DSPARK))
    ap.add_argument("--hf-dspark", default=str(DEFAULT_HF_DSPARK))
    ap.add_argument("--template-dspark", default=str(DEFAULT_BASELINE_DSPARK))
    ap.add_argument("--quantizer-bin", default=str(DEFAULT_QUANTIZER))
    ap.add_argument("--imatrix-in",
                    help="use an existing imatrix.dat and skip collection")
    ap.add_argument("--candidate-gguf",
                    help="use an existing candidate GGUF and skip quantization")
    ap.add_argument("--skip-reprobe", action="store_true",
                    help="stop after imatrix collection / quantization")
    ap.add_argument("--measure-script", default=str(DEFAULT_MEASURE))
    ap.add_argument("--measure-python", default=str(DEFAULT_MEASURE_PYTHON))
    ap.add_argument("--draft-pos-weights", default="1,0.75,0.5,0.33,0.2")
    ap.add_argument("--collector-max-tokens", type=int, default=0)
    ap.add_argument("--ctx-size", type=int, default=4096)
    ap.add_argument("--power", type=int, default=100)
    ap.add_argument("--steps", type=int, default=19)
    ap.add_argument("--trials", type=int, default=128)
    ap.add_argument("--anchor-weight-source", choices=["none", "baseline-hardness", "recoverable-gap"], default="none")
    ap.add_argument("--anchor-weight-b2-label", default="baseline-weighted4ctx_19t_default_256tr")
    ap.add_argument("--oracle-b2-label",
                    help="bundle-local oracle label for recoverable-gap weighting")
    ap.add_argument("--oracle-details-json",
                    help="top-level oracle-envelope details JSON for recoverable-gap weighting")
    ap.add_argument("--anchor-weight-floor", type=float, default=0.5)
    ap.add_argument("--anchor-weight-ceil", type=float, default=2.0)
    ap.add_argument("--anchor-weight-alpha", type=float, default=1.5)
    ap.add_argument("--anchor-weight-min-gap", type=float, default=0.05)
    ap.add_argument("--anchor-weight-baseline-max", type=float, default=4.999)
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


def available_steps(bundle: Path, steps_cap: int) -> int:
    greedy = json.loads((bundle / "target_greedy.json").read_text())
    if isinstance(greedy, dict):
        vals = [int(step["selected"]["id"]) for step in greedy.get("steps", [])]
    else:
        vals = [int(v) for v in greedy]
    count = len(vals) - 1 - 5
    if count <= 0:
        raise RuntimeError(f"bundle {bundle} has too few greedy steps")
    if steps_cap > 0:
        count = min(count, steps_cap)
    return count


def clamp(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


def bundle_anchor_weights(bundle: Path, *, source: str, b2_label: str,
                          floor: float, ceil: float, alpha: float,
                          steps_cap: int) -> list[float]:
    n_steps = available_steps(bundle, steps_cap)
    if source == "none":
        return [1.0] * n_steps
    if source != "baseline-hardness":
        raise RuntimeError(f"unsupported anchor weight source: {source}")

    b2_path = bundle / f"{b2_label}.b2.json"
    data = json.loads(b2_path.read_text())
    per_step = [float(v) for v in data["per_step_accepted"][:n_steps]]
    if len(per_step) != n_steps:
        raise RuntimeError(f"{b2_path} has {len(per_step)} steps but need {n_steps}")
    raw = []
    for accepted in per_step:
        hardness = clamp((5.0 - accepted) / 5.0, 0.0, 1.0)
        raw.append(1.0 + alpha * hardness)
    mean_raw = sum(raw) / len(raw)
    return [clamp(v / mean_raw, floor, ceil) for v in raw]


def build_weighted_overlay(sweep_root: Path, dirs: list[Path], *, run_label: str,
                           source: str, b2_label: str, floor: float,
                           ceil: float, alpha: float, steps_cap: int) -> tuple[Path, dict]:
    overlay_root = sweep_root / f".{run_label}.overlay"
    if overlay_root.exists():
        shutil.rmtree(overlay_root)
    overlay_root.mkdir(parents=True)
    manifest = {"source": source, "bundles": {}}
    for bundle in dirs:
        out_dir = overlay_root / bundle.name
        out_dir.mkdir()
        for child in bundle.iterdir():
            os.symlink(child, out_dir / child.name)
        weights = bundle_anchor_weights(
            bundle,
            source=source,
            b2_label=b2_label,
            floor=floor,
            ceil=ceil,
            alpha=alpha,
            steps_cap=steps_cap,
        )
        (out_dir / "imatrix_anchor_weights.txt").write_text(
            "\n".join(f"{w:.8f}" for w in weights) + "\n",
            encoding="utf-8",
        )
        manifest["bundles"][bundle.name] = {
            "weights": weights,
            "mean": sum(weights) / len(weights),
            "min": min(weights),
            "max": max(weights),
        }
    return overlay_root, manifest


def build_recoverable_gap_overlay(sweep_root: Path, *, run_label: str,
                                  baseline_label: str, oracle_b2_label: str | None,
                                  oracle_details_json: str | None,
                                  floor: float, ceil: float, alpha: float,
                                  min_gap: float, baseline_max: float,
                                  steps_cap: int) -> tuple[Path, dict]:
    cmd = [
        sys.executable,
        str(DEFAULT_RECOVERABLE_GAP),
        "--sweep-root", str(sweep_root),
        "--baseline-label", baseline_label,
        "--out-label", run_label,
        "--floor", str(floor),
        "--ceil", str(ceil),
        "--alpha", str(alpha),
        "--min-gap", str(min_gap),
        "--baseline-max", str(baseline_max),
    ]
    if steps_cap > 0:
        cmd.extend(["--steps-cap", str(steps_cap)])
    if oracle_b2_label:
        cmd.extend(["--oracle-label", oracle_b2_label])
    elif oracle_details_json:
        cmd.extend(["--oracle-details-json", str(Path(oracle_details_json).resolve())])
    else:
        raise RuntimeError("recoverable-gap requires --oracle-b2-label or --oracle-details-json")
    run(cmd)
    overlay_root = sweep_root / f".{run_label}.overlay"
    manifest_path = sweep_root / f"{run_label}.anchor_weights.json"
    return overlay_root, json.loads(manifest_path.read_text())


def reprobe_bundle(bundle: Path, *, backend: str, model: str, dspark: str, steps: int, power: int,
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
            args.ds4_bin,
            "--backend", backend,
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
    collect_root = sweep_root
    overlay_root: Path | None = None
    imatrix_path = Path(args.imatrix_in).resolve() if args.imatrix_in else imatrix_out
    candidate_path = Path(args.candidate_gguf).resolve() if args.candidate_gguf else candidate_out

    if args.anchor_weight_source != "none":
        if args.anchor_weight_source == "baseline-hardness":
            overlay_root, manifest = build_weighted_overlay(
                sweep_root,
                dirs,
                run_label=args.run_label,
                source=args.anchor_weight_source,
                b2_label=args.anchor_weight_b2_label,
                floor=args.anchor_weight_floor,
                ceil=args.anchor_weight_ceil,
                alpha=args.anchor_weight_alpha,
                steps_cap=args.collector_max_tokens,
            )
        else:
            overlay_root, manifest = build_recoverable_gap_overlay(
                sweep_root,
                run_label=args.run_label,
                baseline_label=args.anchor_weight_b2_label,
                oracle_b2_label=args.oracle_b2_label,
                oracle_details_json=args.oracle_details_json,
                floor=args.anchor_weight_floor,
                ceil=args.anchor_weight_ceil,
                alpha=args.anchor_weight_alpha,
                min_gap=args.anchor_weight_min_gap,
                baseline_max=args.anchor_weight_baseline_max,
                steps_cap=args.collector_max_tokens,
            )
        collect_root = overlay_root
        manifest_path = sweep_root / f"{args.run_label}.anchor_weights.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    try:
        if args.imatrix_in:
            if not imatrix_path.exists():
                raise RuntimeError(f"--imatrix-in not found: {imatrix_path}")
        else:
            collect_cmd = [
                args.ds4_bin,
                "--backend", args.backend,
                "-m", str(Path(args.model).resolve()),
                "--dspark", str(Path(args.baseline_dspark).resolve()),
                "--imatrix-dataset", str(collect_root),
                "--imatrix-out", str(imatrix_path),
                "--imatrix-draft-pos-weights", args.draft_pos_weights,
                "--ctx", str(args.ctx_size),
            ]
            if args.collector_max_tokens > 0:
                collect_cmd.extend(["--imatrix-max-tokens", str(args.collector_max_tokens)])
            run(collect_cmd)

        if args.candidate_gguf:
            if not candidate_path.exists():
                raise RuntimeError(f"--candidate-gguf not found: {candidate_path}")
        else:
            run([
                str(Path(args.quantizer_bin).resolve()),
                "--hf", str(Path(args.hf_dspark).resolve()),
                "--template", str(Path(args.template_dspark).resolve()),
                "--out", str(candidate_path),
                "--overwrite",
                "--imatrix", str(imatrix_path),
            ])

        if args.skip_reprobe:
            print(json.dumps({
                "run_label": args.run_label,
                "collect_root": str(collect_root),
                "imatrix_path": str(imatrix_path),
                "candidate_path": str(candidate_path),
                "reprobe_skipped": True,
            }))
            return 0

        rows = []
        baseline_path = str(Path(args.baseline_dspark).resolve())
        candidate_path_str = str(candidate_path.resolve())
        for bundle in dirs:
            ctx = int(bundle.name.split("_")[1])
            base = reprobe_bundle(
                bundle,
                backend=args.backend,
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
                backend=args.backend,
                model=str(Path(args.model).resolve()),
                dspark=candidate_path_str,
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
    finally:
        if overlay_root and overlay_root.exists():
            shutil.rmtree(overlay_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
