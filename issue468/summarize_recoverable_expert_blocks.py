#!/usr/bin/env python3
"""Summarize recoverable-step expert-column block usage for routed Q4_K tensors."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
ORACLE_ROOT = HERE / "dspark_oracle"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ORACLE_ROOT))

from issue468.build_recoverable_gap_overlay import build_weights, bundle_dirs, load_oracle_details, read_b2
from issue468.dspark_oracle.attention import dspark_attention, dspark_attention_prefill, precompute_rope
from issue468.dspark_oracle.expert_store import ExpertStore
from issue468.dspark_oracle.forward import DIM, DSPARK_GGUF, MAX_POS, TARGET_GGUF, forward_embed, layer_weights
from issue468.dspark_oracle.gguf_loader import index_gguf, load_gguf_dense_only, read_tensor
from issue468.dspark_oracle.hc_primitives import hc_post, hc_pre, rmsnorm
from issue468.dspark_oracle.moe import gate


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-root", required=True)
    ap.add_argument("--baseline-label", required=True)
    ap.add_argument("--oracle-details-json", required=True)
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--q4-block", type=int, default=256)
    ap.add_argument("--top-cohorts", type=int, default=64)
    ap.add_argument("--json-out")
    ap.add_argument("--baseline-max", type=float, default=4.999)
    ap.add_argument("--min-gap", type=float, default=0.05)
    ap.add_argument("--mode", choices=["soft", "binary", "boosted"], default="boosted")
    ap.add_argument("--recoverable-boost", type=float, default=8.0)
    ap.add_argument("--nonrecoverable-weight", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--floor", type=float, default=0.5)
    ap.add_argument("--ceil", type=float, default=2.0)
    return ap.parse_args()


def load_main_hidden(bundle: Path, pos: int) -> np.ndarray:
    parts = []
    for layer_id in (40, 41, 42):
        path = bundle / f"hc_dspark_main_hc-{layer_id}_pos{pos}.bin"
        vec = np.fromfile(path, dtype=np.float32).reshape(4, DIM).mean(axis=0)
        parts.append(vec)
    return np.concatenate(parts, axis=0).astype(np.float32)


def forward_selected_experts_and_inputs(
    T: dict,
    infos: dict,
    data_off: int,
    embed_w: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    main_hidden_prefill: np.ndarray,
    main_hidden_decode: np.ndarray,
    anchor_prefill: int,
    anchor_decode: int,
    target_layer: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    main_proj = T["mtp.0.main_proj.weight"][0]
    main_norm_w = T["mtp.0.main_norm.weight"][0]
    _, main_x_p = forward_embed(main_hidden_prefill, anchor_prefill, main_proj, main_norm_w, embed_w)
    layers = [layer_weights(T, s) for s in range(3)]
    stores = [ExpertStore(DSPARK_GGUF, infos, data_off, s) for s in range(3)]
    slot0_per_layer = [dspark_attention_prefill(main_x_p, layers[s], cos, sin) for s in range(3)]
    x, main_x_d = forward_embed(main_hidden_decode, anchor_decode, main_proj, main_norm_w, embed_w)
    input_ids = np.array([anchor_decode], dtype=np.int64)

    for s in range(3):
        w = layers[s]
        residual = x
        yd, post, comb = hc_pre(x, w["hc_attn_fn"], w["hc_attn_scale"], w["hc_attn_base"])
        yd = rmsnorm(yd, w["attn_norm"])
        w_with_cache = dict(w)
        w_with_cache["_win_slot0_kv"] = slot0_per_layer[s][None]
        attn_out, _slot1 = dspark_attention(yd, main_x_d, w_with_cache, 1, cos, sin)
        x = hc_post(attn_out, residual, post, comb)

        residual = x
        yd, post, comb = hc_pre(x, w["hc_ffn_fn"], w["hc_ffn_scale"], w["hc_ffn_base"])
        yd = rmsnorm(yd, w["ffn_norm"])
        flat = yd.reshape(-1, yd.shape[-1]).astype(np.float32)
        weights, idx = gate(flat, w["ffn_gate_inp"], w["ffn_exp_probs_b"], 6)
        if s == target_layer:
            return flat.copy(), weights.copy(), idx.copy()

        from issue468.dspark_oracle.moe import moe

        ffn_out = moe(yd, input_ids, w, stores[s])
        x = hc_post(ffn_out, residual, post, comb)

    raise RuntimeError(f"failed to capture target layer {target_layer}")


def main() -> int:
    args = parse_args()
    if args.q4_block <= 0:
        raise RuntimeError("q4-block must be > 0")

    sweep_root = Path(args.sweep_root).resolve()
    oracle_details = load_oracle_details(Path(args.oracle_details_json).resolve())
    bundles = bundle_dirs(sweep_root)
    _kv, T, infos, data_off, _skipped = load_gguf_dense_only(DSPARK_GGUF)
    _t_kv, target_infos, target_data_off = index_gguf(TARGET_GGUF)
    embed_w = read_tensor(TARGET_GGUF, target_infos, target_data_off, "token_embd.weight").astype(np.float32)
    cos, sin = precompute_rope(64, MAX_POS)

    cohort_mass: dict[tuple[int, int], float] = defaultdict(float)
    per_context = {}
    ncols_seen: int | None = None

    for bundle in bundles:
        ctx = int(bundle.name.split("_")[1])
        base = read_b2(bundle, args.baseline_label)
        oracle_steps = oracle_details.get(ctx)
        if oracle_steps is None:
            raise RuntimeError(f"missing oracle details for context {ctx}")
        _weights, details = build_weights(
            [float(v) for v in base["per_step_accepted"]],
            oracle_steps,
            mode=args.mode,
            recoverable_boost=args.recoverable_boost,
            nonrecoverable_weight=args.nonrecoverable_weight,
            baseline_max=args.baseline_max,
            min_gap=args.min_gap,
            alpha=args.alpha,
            floor=args.floor,
            ceil=args.ceil,
        )
        greedy = json.loads((bundle / "target_greedy.json").read_text())
        recoverable_rows = []
        for item in details:
            if not item["recoverable"]:
                continue
            step = int(item["step"])
            pos_prefill = ctx + step - 1
            pos_decode = ctx + step
            mh_p = load_main_hidden(bundle, pos_prefill)
            mh_d = load_main_hidden(bundle, pos_decode)
            anchor_prefill = int(greedy[step - 1])
            anchor_decode = int(greedy[step])
            flat, gate_weights, gate_idx = forward_selected_experts_and_inputs(
                T, infos, data_off, embed_w, cos, sin, mh_p, mh_d, anchor_prefill, anchor_decode, args.layer
            )
            ncols = flat.shape[1]
            if ncols_seen is None:
                ncols_seen = ncols
            elif ncols_seen != ncols:
                raise RuntimeError(f"inconsistent ncols: {ncols_seen} vs {ncols}")
            nblocks = (ncols + args.q4_block - 1) // args.q4_block

            sample_record = {
                "step": step,
                "weight": float(item["weight"]),
                "top_pairs": [],
            }
            for draft_pos in range(flat.shape[0]):
                x2 = np.square(flat[draft_pos], dtype=np.float64)
                block_sums = [
                    float(x2[b0:b1].sum())
                    for b0, b1 in (
                        (block_idx * args.q4_block, min(ncols, (block_idx + 1) * args.q4_block))
                        for block_idx in range(nblocks)
                    )
                ]
                for slot in range(gate_idx.shape[1]):
                    expert = int(gate_idx[draft_pos, slot])
                    gate_weight = float(gate_weights[draft_pos, slot])
                    for block_idx, block_mass in enumerate(block_sums):
                        mass = float(item["weight"]) * gate_weight * block_mass
                        cohort_mass[(expert, block_idx)] += mass
                if draft_pos == 0:
                    sample_record["top_pairs"] = [
                        {
                            "expert": int(gate_idx[draft_pos, slot]),
                            "gate_weight": float(gate_weights[draft_pos, slot]),
                        }
                        for slot in range(gate_idx.shape[1])
                    ]
            recoverable_rows.append(sample_record)
        per_context[bundle.name] = recoverable_rows

    top_rows = [
        {"expert": expert, "block_index": block_idx, "mass": mass}
        for (expert, block_idx), mass in sorted(cohort_mass.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))[
            : args.top_cohorts
        ]
    ]
    report = {
        "layer": args.layer,
        "q4_block": args.q4_block,
        "ncols": ncols_seen,
        "top_expert_blocks": top_rows,
        "contexts": per_context,
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
