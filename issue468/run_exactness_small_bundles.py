#!/usr/bin/env python3
"""Capture compact exactness bundles for the retained small corpus.

For each prompt/temperature cell this script retains:
- target top-k/logprob dump
- selected target token stream (greedy at temp=0, sampled stream at temp>0)
- layer-wise captures for target layers 40/41/42
- derived oracle inputs and oracle acceptance summary
- oracle_ref.npz for temp=0
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ISSUE468 = ROOT / "issue468"
CORPUS = ISSUE468 / "prompts" / "exactness_small_corpus"
DEFAULT_MODEL = (ROOT / ".." / "ds4" / "gguf" / "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf").resolve()
DEFAULT_DSPARK = (ROOT / ".." / "ds4" / "gguf" / "dspark.gguf").resolve()
DEFAULT_OUT = ISSUE468 / "artifacts" / "exactness_small_bundles"
DEFAULT_PY = sys.executable


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--dspark", default=str(DEFAULT_DSPARK))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--temps", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--tokens", type=int, default=14, help="generated target tokens to retain")
    ap.add_argument("--block", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=128)
    ap.add_argument("--power", type=int, default=100)
    ap.add_argument("--python", default=DEFAULT_PY)
    ap.add_argument("--force-recapture", action="store_true",
                    help="re-run target-side capture even if compact bundle files already exist")
    return ap.parse_args()


def run(cmd: list[str], *, env: dict[str, str] | None = None,
        stdout_path: Path | None = None, stderr_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run one command from the repo root and raise on failure."""
    stdout = subprocess.PIPE if stdout_path is None else open(stdout_path, "w", encoding="utf-8")
    stderr = subprocess.PIPE if stderr_path is None else open(stderr_path, "w", encoding="utf-8")
    try:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=stdout, stderr=stderr)
    finally:
        if stdout_path is not None:
            stdout.close()
        if stderr_path is not None:
            stderr.close()
    if proc.returncode != 0:
        if stdout_path is None and proc.stdout:
            print(proc.stdout)
        if stderr_path is None and proc.stderr:
            print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc


def selected_tokens_from_topk(path: Path) -> tuple[list[int], int]:
    """Read the retained selected token stream and prompt length from one top-k dump."""
    data = json.loads(path.read_text())
    toks = [int(step["selected"]["id"]) for step in data["steps"]]
    return toks, int(data["prompt_tokens"])


def derive_main_hidden(python_bin: str, bundle_dir: Path) -> None:
    """Derive oracle main_hidden_pos*.npy inputs from retained layer captures."""
    run([
        python_bin,
        str(ISSUE468 / "dspark_oracle" / "build_main_hidden_from_captures.py"),
        "--captures-dir", str(bundle_dir),
        "--out-dir", str(bundle_dir / "oracle"),
        "--manifest-out", str(bundle_dir / "oracle" / "main_hidden_manifest.json"),
    ])


def measure_acceptance(python_bin: str, model: str, dspark: str, bundle_dir: Path) -> dict:
    """Run the retained numpy oracle over one bundle and return its acceptance summary."""
    out = bundle_dir / "oracle" / "acceptance_summary.json"
    run([
        python_bin,
        str(ISSUE468 / "dspark_oracle" / "measure_acceptance_bundle.py"),
        "--bundle-dir", str(bundle_dir),
        "--model", model,
        "--dspark", dspark,
        "--json-out", str(out),
    ])
    return json.loads(out.read_text())


def compact_outputs_exist(bundle_dir: Path, temp: float) -> bool:
    captures_npz = bundle_dir / "captures" / "captures.npz"
    oracle_inputs_npz = bundle_dir / "oracle" / "oracle_inputs.npz"
    target_topk = bundle_dir / "target_topk.json"
    selected = bundle_dir / "target_selected_tokens.json"
    greedy_ok = True if temp != 0.0 else (bundle_dir / "target_greedy.json").exists()
    return captures_npz.exists() and oracle_inputs_npz.exists() and target_topk.exists() and selected.exists() and greedy_ok


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((CORPUS / "manifest.json").read_text())
    rows = []
    for name, entry in manifest.items():
        prompt_path = ISSUE468 / entry["file"]
        for temp in args.temps:
            temp_tag = str(temp).replace('.', 'p')
            bundle_dir = out_dir / f"{name}__t{temp_tag}"
            (bundle_dir / "captures").mkdir(parents=True, exist_ok=True)
            (bundle_dir / "oracle").mkdir(parents=True, exist_ok=True)
            prompt_copy = bundle_dir / "prompt.txt"
            prompt_copy.write_text(prompt_path.read_text(), encoding="utf-8")
            topk_path = bundle_dir / "target_topk.json"
            if args.force_recapture or not compact_outputs_exist(bundle_dir, temp):
                base_env = os.environ.copy()
                base_env.update({
                    "DS4_METAL_GRAPH_DUMP_PREFIX": str(bundle_dir / "captures" / "dump"),
                    "DS4_METAL_GRAPH_DUMP_NAME": "hc_ffn_post",
                })
                # Capture the three DSpark target-input layers independently so each
                # retained dump is the compact [HC, DIM] tensor for that one layer.
                first = True
                moved = 0
                for layer in (40, 41, 42):
                    env = dict(base_env)
                    env["DS4_METAL_GRAPH_DUMP_LAYER"] = str(layer)
                    # The first run writes the canonical target_topk.json; later
                    # per-layer runs keep auxiliary top-k/stdout/stderr only so the
                    # bundle preserves one main target reference plus per-layer captures.
                    run([
                        "./ds4",
                        "--metal",
                        "-m", args.model,
                        "--prompt-file", str(prompt_path),
                        "--dump-logprobs", str(topk_path if first else (bundle_dir / f'target_topk.layer{layer}.json')),
                        "--logprobs-top-k", str(args.top_k),
                        "--tokens", str(args.tokens),
                        "--ctx", str(args.ctx),
                        "--temp", str(temp),
                        "--seed", str(args.seed),
                        "--power", str(args.power),
                    ], env=env,
                       stdout_path=bundle_dir / ("target.dump.stdout" if first else f"target.dump.layer{layer}.stdout"),
                       stderr_path=bundle_dir / ("target.dump.stderr" if first else f"target.dump.layer{layer}.stderr"))
                    first = False
                dump_prefix = bundle_dir / "captures" / "dump"
                for dump in dump_prefix.parent.glob(f"{dump_prefix.name}_hc_ffn_post-*_pos*.bin"):
                    dump.rename(bundle_dir / "captures" / dump.name)
                    moved += 1
                if moved == 0:
                    raise RuntimeError(f"no hidden-state captures were produced for {bundle_dir}")
                selected, prompt_tokens = selected_tokens_from_topk(topk_path)
                (bundle_dir / "target_selected_tokens.json").write_text(json.dumps(selected) + "\n")
                if temp == 0.0:
                    (bundle_dir / "target_greedy.json").write_text(json.dumps(selected) + "\n")
            else:
                selected, prompt_tokens = selected_tokens_from_topk(topk_path)
            bundle_manifest = {
                "prompt_name": name,
                "prompt_file": str(prompt_path.relative_to(ROOT)),
                "temperature": temp,
                "seed": args.seed,
                "ctx": args.ctx,
                "generated_tokens": len(selected),
                "block": args.block,
                "measure_steps": max(0, len(selected) - args.block - 1),
                "prompt_tokens": prompt_tokens,
                "reference_mode": "greedy" if temp == 0.0 else "sampled-stream",
            }
            (bundle_dir / "bundle_manifest.json").write_text(json.dumps(bundle_manifest, indent=2) + "\n")
            # build_main_hidden_from_captures.py also writes compact packed files:
            #   captures/captures.npz and oracle/oracle_inputs.npz
            derive_main_hidden(args.python, bundle_dir)
            acceptance = measure_acceptance(args.python, args.model, args.dspark, bundle_dir)
            if float(temp) == 0.0:
                run([
                    args.python,
                    str(ISSUE468 / "dspark_oracle" / "forward.py"),
                    "--validate",
                    "--capture-dir", str(bundle_dir / "oracle"),
                    "--prefill-pos", str(prompt_tokens),
                    "--decode-pos", str(prompt_tokens + 1),
                    "--out", str(bundle_dir / "oracle" / "oracle_ref.npz"),
                ])
            rows.append({
                "prompt": name,
                "temperature": temp,
                "reference_mode": bundle_manifest["reference_mode"],
                "prompt_tokens": prompt_tokens,
                "average_prefix": acceptance["average_prefix"],
                "prefix_hist": acceptance["prefix_hist"],
                "match_pct": acceptance["match_pct"],
                "measure_steps": acceptance["measure_steps"],
                "bundle_dir": str(bundle_dir.relative_to(ROOT)),
            })
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    header = ["prompt", "temperature", "reference_mode", "prompt_tokens", "average_prefix", "match_pct", "measure_steps", "bundle_dir"]
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(row[key]) for key in header))
    (out_dir / "summary.csv").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
