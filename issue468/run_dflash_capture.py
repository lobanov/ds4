#!/usr/bin/env python3
"""Capture target HC hidden at DFlash layers (3,13,23,32,42) for the exactness
small corpus.

Mirrors run_exactness_small_bundles.py's ds4 dump flow but for the 5 DFlash
target layers and keeping prompt+generated positions (full context). Each layer
is captured in its own ds4 run (DS4_METAL_GRAPH_DUMP_LAYER takes one layer;
DUMP_LAYER=all would GPU-sync ~600x per run). Output per cell: a compact
context.npz with layer{3,13,23,32,42} arrays [n_pos, 16384] + positions, plus the
target token stream copied from the matching DSpark bundle (same prompt/temp/seed
=> identical stream).

Cell-idempotent: skips any cell whose context.npz already exists, so the run can
be chunked across multiple invocations.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ISSUE468 = ROOT / "issue468"
CORPUS = ISSUE468 / "prompts" / "exactness_small_corpus"
DSBARK_BUNDLES = ISSUE468 / "artifacts" / "exactness_small_bundles"
OUT = ISSUE468 / "artifacts" / "dflash_capture"
DEFAULT_MODEL = (ROOT / ".." / "ds4" / "gguf" /
                 "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf").resolve()
LAYERS = (3, 13, 23, 32, 42)
HC_DIM = 4 * 4096   # 16384 floats per (layer, position)
DS4 = ROOT / "ds4"
HC = 4
DIM = 4096


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--temps", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--tokens", type=int, default=14)
    ap.add_argument("--power", type=int, default=100)
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these prompt names")
    return ap.parse_args()


def run_ds4_dump(model, prompt_path, layer, temp, seed, ctx, tokens, power, dump_prefix):
    """One ds4 run dumping hc_ffn_post for `layer` (prefill batch + decode positions)."""
    env = os.environ.copy()
    env.update({
        "DS4_METAL_GRAPH_DUMP_PREFIX": str(dump_prefix),
        "DS4_METAL_GRAPH_DUMP_NAME": "hc_ffn_post",
        "DS4_METAL_GRAPH_DUMP_LAYER": str(layer),
    })
    log = dump_prefix.parent / f"run.layer{layer}.log"
    cmd = [str(DS4), "--metal", "-m", str(model), "--prompt-file", str(prompt_path),
           "--tokens", str(tokens), "--ctx", str(ctx), "--temp", str(temp),
           "--seed", str(seed), "--power", str(power)]
    with open(log, "w") as lf:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=lf, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"ds4 failed (layer {layer}): see {log}")


def parse_layer_dumps(captures_dir, dump_prefix_name):
    """Return (positions list, array [n_pos, HC_DIM]) from one layer's dump files.

    pos0 file holds the prefill batch [n_prompt, HC_DIM]; posN (N>=n_prompt) files
    hold single generated positions [HC_DIM]."""
    files = sorted(captures_dir.glob(f"{dump_prefix_name}_hc_ffn_post-*.bin"))
    if not files:
        raise RuntimeError(f"no dump files found in {captures_dir}")
    by_pos = {}
    n_prompt = None
    for f in files:
        arr = np.fromfile(f, dtype=np.float32)
        stem_pos = int(f.stem.split("_pos")[-1])
        if arr.size == HC_DIM:
            by_pos[stem_pos] = arr.reshape(1, HC_DIM)
        elif arr.size % HC_DIM == 0:
            # prefill batch: positions stem_pos .. stem_pos + (arr.size/HC_DIM) - 1
            nrows = arr.size // HC_DIM
            rows = arr.reshape(nrows, HC_DIM)
            n_prompt = nrows
            for i in range(nrows):
                by_pos[stem_pos + i] = rows[i:i + 1]
        else:
            raise RuntimeError(f"unexpected dump size {arr.size} for {f}")
    positions = sorted(by_pos)
    arr = np.concatenate([by_pos[p] for p in positions], axis=0).astype(np.float32)
    return positions, arr


def capture_cell(model, name, entry, temp, args, cell_dir):
    cell_dir.mkdir(parents=True, exist_ok=True)
    captures_dir = cell_dir / "captures"
    captures_dir.mkdir(parents=True, exist_ok=True)
    context = {}
    positions_all = None
    for layer in LAYERS:
        dump_prefix = captures_dir / f"dump"
        run_ds4_dump(model, ISSUE468 / entry["file"], layer, temp, args.seed,
                     args.ctx, args.tokens, args.power, dump_prefix)
        positions, arr = parse_layer_dumps(captures_dir, "dump")
        context[f"layer{layer}"] = arr
        if positions_all is None:
            positions_all = positions
        elif positions != positions_all:
            # keep the intersection-consistent record; layers should match
            print(f"  warn: layer {layer} positions differ", file=sys.stderr)
        # remove this layer's raw dumps to keep the dir small
        for f in captures_dir.glob("dump_hc_ffn_post-*.bin"):
            f.unlink()
    np.savez(cell_dir / "context.npz", positions=np.array(positions_all, dtype=np.int32), **context)
    # target token stream: reuse the DSpark bundle (same prompt/temp/seed => identical)
    dspark_cell = DSBARK_BUNDLES / f"{name}__t{str(temp).replace('.', 'p')}"
    src = dspark_cell / "target_selected_tokens.json"
    if not src.exists():
        raise FileNotFoundError(f"missing DSpark token stream {src}")
    shutil.copy(src, cell_dir / "target_selected_tokens.json")
    (cell_dir / "cell_manifest.json").write_text(json.dumps({
        "prompt_name": name, "temperature": temp, "seed": args.seed, "ctx": args.ctx,
        "tokens": args.tokens, "layers": list(LAYERS), "n_positions": len(positions_all),
        "n_prompt_positions_est": int(context["layer3"].shape[0] - (len(positions_all) - context["layer3"].shape[0])),
    }, indent=2) + "\n")
    return len(positions_all)


def main():
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((CORPUS / "manifest.json").read_text())
    names = [n for n in manifest if (args.only is None or n in args.only)]
    total_cells = 0
    for name in names:
        entry = manifest[name]
        for temp in args.temps:
            tag = str(temp).replace('.', 'p')
            cell_dir = OUT / f"{name}__t{tag}"
            if (cell_dir / "context.npz").exists():
                print(f"[skip] {cell_dir.name} (context.npz exists)")
                continue
            print(f"[capture] {cell_dir.name} ...", flush=True)
            npos = capture_cell(args.model, name, entry, temp, args, cell_dir)
            total_cells += 1
            print(f"  -> {npos} positions x {len(LAYERS)} layers")
    print(f"\ncaptured {total_cells} new cells this run; output under {OUT}")


if __name__ == "__main__":
    raise SystemExit(main())
