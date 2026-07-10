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


def surgery_to_main_hidden(capture_npz: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """capture .npz -> (main_hidden [n_capture,12288], positions [n_capture],
    greedy [n_generated], prompt_len, n_capture).

    Indexing (verified codex gate 1): the chunked-prefill first token has no decode-step
    capture, so n_capture = n_generated - 1. main_hidden[i] is POST-token hidden for the
    i-th generated token (anchor g_i); the drafter predicts g_{i+1}. positions covers the
    n_capture anchor positions; greedy is the FULL trajectory for target indexing.
    """
    d = np.load(str(capture_npz))
    n_generated = int(d["n_generated"]) if "n_generated" in d.files else int(d["n_gen"])
    # n_capture = decode-step hidden count (layer array length); = n_generated-1 due to
    # the chunked-prefill first token lacking a decode-step capture.
    n_capture = int(d["n_capture"]) if "n_capture" in d.files else d["layer40"].shape[0]
    assert n_capture == n_generated - 1, (
        f"expected n_capture=n_generated-1 (chunked-prefill off-by-one), "
        f"got n_capture={n_capture} vs n_generated={n_generated}")
    parts = []
    for layer in LAYERS:
        arr = d[f"layer{layer}"]            # [n_capture, HC=4, DIM]
        assert arr.ndim == 3 and arr.shape[1:] == (HC, DIM), (
            f"layer{layer} expected [n_capture,{HC},{DIM}], got {arr.shape}")
        parts.append(arr.astype(np.float32).mean(axis=1))   # [n_capture, DIM]
    main_hidden = np.concatenate(parts, axis=1).astype(np.float32)  # [n_capture, 12288]
    prompt_len = int(d["prompt_tokens"])
    positions = np.arange(prompt_len, prompt_len + n_capture, dtype=np.int32)
    greedy = d["greedy_tokens"][:n_generated]  # full trajectory for target indexing
    return main_hidden, positions, greedy, prompt_len, n_capture


def convert(capture_npz: Path, output_dir: Path, prompt_name: str = "") -> dict:
    main_hidden, positions, greedy, prompt_len, n_capture = surgery_to_main_hidden(capture_npz)
    n_generated = len(greedy)
    out = output_dir / "oracle"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out / "oracle_inputs.npz"),
                        main_hidden=main_hidden, positions=positions)
    # Write a full oracle bundle so measure_acceptance_bundle.py runs directly:
    #   bundle_manifest.json + target_selected_tokens.json (the greedy trajectory)
    #   + oracle/oracle_inputs.npz (main_hidden). prompt_tokens from the capture.
    (output_dir / "bundle_manifest.json").write_text(json.dumps({
        "prompt_name": prompt_name, "prompt_file": f"prompts/{prompt_name}.txt",
        "temperature": 0.0, "seed": 0, "ctx": 4096, "block": 5,
        "generated_tokens": n_generated, "n_capture": n_capture,
        "measure_steps": n_capture,
        "prompt_tokens": prompt_len, "reference_mode": "greedy",
    }, indent=2))
    (output_dir / "target_selected_tokens.json").write_text(
        json.dumps([int(t) for t in greedy.tolist()], indent=2))
    d = np.load(str(capture_npz))
    return {
        "prompt_name": prompt_name,
        "n_generated": n_generated,
        "n_capture": n_capture,
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
