#!/usr/bin/env python3
"""Convert the MTP-hook surgery capture -> oracle ``main_hidden`` (drafter input).

Input: the .npz from capture_hc_residual.py, containing per generated position:
    layer{40,41,42} : [n_gen, hc_mult=4, hidden=4096]   # the post-FFN HC residual
    greedy_tokens   : [n_gen], prompt_tokens : int

Target (what the drafter / oracle consumes — dspark_oracle/forward.py `forward_embed`
+ build_main_hidden_from_captures.py):
    main_hidden[k] = concat( mean(layer40[k], axis=hc_mult),       # [4096]
                             mean(layer41[k], axis=hc_mult),       # [4096]
                             mean(layer42[k], axis=hc_mult) )      # -> [3*4096 = 12288]
  i.e. each layer's [4,4096] HC residual is mean-reduced over the HC=4 axis to [4096]
  (the SAME reduction build_main_hidden applies to the ds4 hc_ffn_post captures), then
  the 3 layers are concatenated. The HC=4 is internal drafter state (np.repeat of the
  anchor embedding in forward_embed); the capture's mean(hc) is the drafter's input.

REPRESENTATION-MATCH CHECK: this converter produces main_hidden in the EXACT format
the oracle expects. The fidelity gate (validate_fidelity drafter-on-FP-hiddens sanity,
or a direct compare to the retained ds4 Q2 hc_ffn_post-derived main_hidden) validates
that vLLM's per-layer HC residual == ds4's after_ffn_hc within quantized-kernel tolerance.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LAYERS = (40, 41, 42)
DIM = 4096
HC = 4
MAIN_HIDDEN_DIM = len(LAYERS) * DIM  # 12288


def surgery_to_main_hidden(capture_npz: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """capture .npz -> (main_hidden [n_gen,12288], positions [n_gen], greedy [n_gen])."""
    d = np.load(str(capture_npz))
    n_gen = int(d["n_gen"])
    parts = []
    for layer in LAYERS:
        arr = d[f"layer{layer}"]            # [n_gen, HC=4, DIM]
        assert arr.ndim == 3 and arr.shape[1:] == (HC, DIM), (
            f"layer{layer} expected [n_gen,{HC},{DIM}], got {arr.shape}")
        parts.append(arr.astype(np.float32).mean(axis=1))   # [n_gen, DIM]  (mean over hc_mult)
    main_hidden = np.concatenate(parts, axis=1).astype(np.float32)  # [n_gen, 12288]
    prompt_len = int(d["prompt_tokens"])
    positions = np.arange(prompt_len, prompt_len + n_gen, dtype=np.int32)
    greedy = d["greedy_tokens"][:n_gen] if "greedy_tokens" in d else np.zeros(n_gen, dtype=np.int64)
    return main_hidden, positions, greedy


def convert(capture_npz: Path, output_dir: Path, prompt_name: str = "") -> dict:
    main_hidden, positions, greedy = surgery_to_main_hidden(capture_npz)
    out = output_dir / "oracle"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out / "oracle_inputs.npz"),
                        main_hidden=main_hidden, positions=positions)
    # Write a full oracle bundle so measure_acceptance_bundle.py runs directly:
    #   bundle_manifest.json + target_selected_tokens.json (the greedy trajectory)
    #   + oracle/oracle_inputs.npz (main_hidden). prompt_tokens from the capture.
    d = np.load(str(capture_npz))
    prompt_len = int(d["prompt_tokens"])
    n_gen = int(main_hidden.shape[0])
    (output_dir / "bundle_manifest.json").write_text(json.dumps({
        "prompt_name": prompt_name, "prompt_file": f"prompts/{prompt_name}.txt",
        "temperature": 0.0, "seed": 0, "ctx": 4096, "block": 5,
        "generated_tokens": n_gen, "measure_steps": n_gen,
        "prompt_tokens": prompt_len, "reference_mode": "greedy",
    }, indent=2))
    (output_dir / "target_selected_tokens.json").write_text(
        json.dumps([int(t) for t in greedy.tolist()], indent=2))
    return {
        "prompt_name": prompt_name,
        "n_gen": n_gen,
        "main_hidden_shape": list(main_hidden.shape),
        "positions_head": positions[:3].tolist(),
        "oracle_inputs": str(out / "oracle_inputs.npz"),
        "bundle_dir": str(output_dir),
        "per_layer_pre_mean": {f"layer{L}": float(d[f'layer{L}'].mean()) for L in LAYERS},
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True, help="capture_hc_residual.py output .npz")
    ap.add_argument("--output-dir", required=True, help="bundle dir to write oracle/")
    ap.add_argument("--prompt-name", default="")
    args = ap.parse_args()
    res = convert(Path(args.capture), Path(args.output_dir), args.prompt_name)
    print(json.dumps(res, indent=2))
