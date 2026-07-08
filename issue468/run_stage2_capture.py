#!/usr/bin/env python3
"""Activity 3 orchestrator — capture the Stage 2 corpus + consolidate to shards.

RESEARCH INSTRUMENTATION. Generates the ds4 --capture-dataset list from the Stage 2
manifest, runs ds4 (model loaded once, multi-layer 40/41/42 dump in one pass) from
the engine cwd (ds4 loads metal/*.metal relative to cwd), then consolidates the
scratch into sharded safetensors via run_stage2_consolidate.py.

Usage:
  python3 run_stage2_capture.py --limit 6           # rate test
  python3 run_stage2_capture.py                     # full 240-prompt capture
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent  # issue468/
ENGINE = Path("/Users/lobanov/Projects/ds4")
DS4 = ENGINE / "ds4"
MODEL = ENGINE / "gguf" / "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
MANIFEST = HERE / "prompts" / "stage2_corpus" / "manifest.json"
SCRATCH = HERE / "dspark_train" / "data" / "capture_scratch"
SHARDS = HERE / "dspark_train" / "data" / "shards"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap prompts (0 = all)")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()

    import json
    man = json.loads(MANIFEST.read_text())
    recs = man["prompts"]
    if args.limit:
        recs = recs[: args.limit]
    SCRATCH.mkdir(parents=True, exist_ok=True)
    list_file = SCRATCH.parent / "capture_list.txt"
    with list_file.open("w") as f:
        for r in recs:
            f.write(f"{r['prompt_id']}\t{(HERE / r['file']).resolve()}\n")
    print(f"capture list: {len(recs)} prompts -> {list_file}", flush=True)

    cmd = [
        str(DS4), "--metal", "-m", str(MODEL),
        "--capture-dataset", str(list_file), "--capture-out", str(SCRATCH),
        "--capture-layers", "40,41,42",
        "--tokens", str(args.tokens), "--ctx", str(args.ctx),
        "--temp", "0", "--seed", str(args.seed), "--logprobs-top-k", "128",
    ]
    print("running (cwd=engine for metal/): " + " ".join(cmd[:6]) + " ...", flush=True)
    env = dict(os.environ)
    env["DS4_METAL_GRAPH_DUMP_NAME"] = "hc_ffn_post"
    env["DS4_METAL_GRAPH_DUMP_LAYER"] = "40,41,42"
    rc = subprocess.run(cmd, cwd=str(ENGINE), env=env).returncode
    print(f"ds4 capture rc={rc}", flush=True)
    if rc != 0:
        return rc

    # consolidate
    consolidator = HERE / "run_stage2_consolidate.py"
    py = sys.executable
    rc2 = subprocess.run([py, str(consolidator), "--capture-dir", str(SCRATCH),
                          "--manifest", str(MANIFEST), "--out", str(SHARDS)]).returncode
    print(f"consolidate rc={rc2}", flush=True)
    return rc2


if __name__ == "__main__":
    raise SystemExit(main())
