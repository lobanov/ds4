#!/usr/bin/env python3
"""Lead 03 task-1b — F16-vs-F32 fidelity gate.

Re-captures 2-3 Stage 2 prompts at TRUE F32 via the exactness path (./ds4 +
DS4_METAL_GRAPH_DUMP, 3 per-layer runs), derives F32 main_hidden, measures
acceptance, and compares to the F16-stored Stage 2 capture for the SAME prompt
(via Stage2CaptureStore). Quantifies whether F16 storage materially shifts
drafter acceptance (draft tokens / E[a|5block] / p1) vs true F32.

If F16-vs-F32 is within tolerance -> Stage 2 F16 captures usable as-is. If
material -> re-capture/consolidate at F32 (task-3 would use F32 shards).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "dspark_oracle"))

from measure_acceptance_bundle import build_model_ctx, build_drafter_ctx, measure_bundle  # noqa: E402
from stage2_capture_store import Stage2CaptureStore  # noqa: E402

ENGINE = Path("/Users/lobanov/Projects/ds4")
DS4 = ENGINE / "ds4"
MODEL = ENGINE / "gguf" / "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
DSPARK = ENGINE / "gguf" / "dspark.gguf"
SHARDS = HERE / "dspark_train" / "data" / "shards"
OUT = HERE / "artifacts" / "acceptance_powered" / "f32_fidelity"
LAYERS = (40, 41, 42)
TOKENS = 128
PROMPT_IDS = ["codealpaca_0000", "dolly_0000"]  # diverse sources


def run_capture(prompt_id: str, prompt_file: Path, cap_dir: Path) -> list[int]:
    """Run ds4 3x (per layer) producing F32 dumps; return captured positions."""
    cap_dir.mkdir(parents=True, exist_ok=True)
    (cap_dir / "captures").mkdir(exist_ok=True)
    base_env = os.environ.copy()
    base_env.update({"DS4_METAL_GRAPH_DUMP_PREFIX": str(cap_dir / "captures" / "dump"),
                     "DS4_METAL_GRAPH_DUMP_NAME": "hc_ffn_post"})
    topk_path = cap_dir / "target_topk.json"
    first = True
    for layer in LAYERS:
        env = dict(base_env); env["DS4_METAL_GRAPH_DUMP_LAYER"] = str(layer)
        subprocess.run([
            str(DS4), "--metal", "-m", str(MODEL),
            "--prompt-file", str(prompt_file),
            "--dump-logprobs", str(topk_path if first else (cap_dir / f"target_topk.layer{layer}.json")),
            "--logprobs-top-k", "128", "--tokens", str(TOKENS),
            "--ctx", "8192", "--temp", "0", "--seed", "2", "--power", "100",
        ], cwd=str(ENGINE), env=env, check=True,
            stdout=open(cap_dir / ("target.dump.stdout" if first else f"target.dump.layer{layer}.stdout"), "w"),
            stderr=open(cap_dir / ("target.dump.stderr" if first else f"target.dump.layer{layer}.stderr"), "w"))
        first = False
    # move dumps flat (run_exactness_small_bundles convention)
    prefix = cap_dir / "captures" / "dump"
    moved = 0
    for d in prefix.parent.glob(f"{prefix.name}_hc_ffn_post-*_pos*.bin"):
        d.rename(cap_dir / "captures" / d.name); moved += 1
    assert moved > 0, f"no dumps produced for {prompt_id}"
    return moved


def selected_from_topk(p: Path):
    data = json.loads(p.read_text())
    return [int(s["selected"]["id"]) for s in data["steps"]], int(data["prompt_tokens"])


def derive_main_hidden(cap_dir: Path):
    """build_main_hidden_from_captures -> oracle/oracle_inputs.npz (F32)."""
    subprocess.run([sys.executable,
                    str(HERE / "dspark_oracle" / "build_main_hidden_from_captures.py"),
                    "--captures-dir", str(cap_dir), "--out-dir", str(cap_dir / "oracle"),
                    "--manifest-out", str(cap_dir / "oracle" / "main_hidden_manifest.json")],
                   check=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    st2man = json.loads((HERE / "prompts" / "stage2_corpus" / "manifest.json").read_text())
    pid_to_file = {p["prompt_id"]: HERE / p["file"] for p in st2man["prompts"]}
    print("loading store + model/drafter ctx...", flush=True)
    store = Stage2CaptureStore(SHARDS)
    mctx = build_model_ctx(str(MODEL)); dctx = build_drafter_ctx(str(DSPARK))

    results = []
    for pid in PROMPT_IDS:
        pf = pid_to_file[pid]
        cap_dir = OUT / pid
        print(f"\n=== {pid} ({pf.name}) ===", flush=True)
        run_capture(pid, pf, cap_dir)
        derive_main_hidden(cap_dir)
        sel, pos0 = selected_from_topk(cap_dir / "target_topk.json")
        (cap_dir / "target_selected_tokens.json").write_text(json.dumps(sel) + "\n")
        (cap_dir / "bundle_manifest.json").write_text(json.dumps({
            "prompt_name": pid, "temperature": 0.0, "seed": 2, "ctx": 8192,
            "generated_tokens": len(sel), "block": 5,
            "measure_steps": max(0, len(sel) - 5 - 1), "prompt_tokens": pos0,
            "reference_mode": "greedy"}) + "\n")
        # F32 acceptance (bundle-dir, true F32 capture)
        f32 = measure_bundle(cap_dir, mctx, dctx, reuse_mode="none", candidate="f32")
        # F16 acceptance (store, Stage 2 capture)
        f16 = measure_bundle(store=store, prompt_id=pid, mctx=mctx, dctx=dctx, reuse_mode="none", candidate="f16")
        # per-position draft-token agreement between F32 and F16
        agree = sum(1 for r32, r16 in zip(f32["rows"], f16["rows"])
                    if r32["draft"] == r16["draft"])
        n = min(len(f32["rows"]), len(f16["rows"]))
        # main_hidden F32-vs-F16 magnitude diff (first step)
        mh32 = np.load(cap_dir / "oracle" / "oracle_inputs.npz")["main_hidden"]
        mh16 = store._mh[store.id_to_idx[pid]][:mh32.shape[0]]
        rel = float(np.abs(mh32 - mh16).max() / (np.abs(mh32).max() + 1e-9))
        res = {"prompt_id": pid, "pos0": pos0, "n_steps_f32": len(f32["rows"]),
               "f32_avg_prefix": f32["average_prefix"], "f16_avg_prefix": f16["average_prefix"],
               "d_avg_prefix": round(f16["average_prefix"] - f32["average_prefix"], 4),
               "f32_p1": round(sum(1 for r in f32["rows"] if r["draft"][0] == r["target"][0]) / len(f32["rows"]), 4),
               "f16_p1": round(sum(1 for r in f16["rows"] if r["draft"][0] == r["target"][0]) / len(f16["rows"]), 4),
               "draft_token_agree_f32_vs_f16": round(agree / n, 4) if n else 0,
               "main_hidden_max_rel_diff_f16_vs_f32": round(rel, 5)}
        print(json.dumps(res, indent=2), flush=True)
        results.append(res)
    out = {"prompts": results,
           "tolerance_note": "F16 usable if d_avg_prefix within ~0.03 and draft-token agreement >~0.95 per prompt",
           "verdict_f16_usable": all(r["draft_token_agree_f32_vs_f16"] > 0.95 for r in results)}
    (OUT / "f32_fidelity.json").write_text(json.dumps(out, indent=2) + "\n")
    print("\n=== AGGREGATE ===")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
