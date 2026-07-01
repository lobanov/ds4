#!/usr/bin/env python3
"""Summarize recoverable-step routed expert usage from sweep bundles.

This uses the numpy oracle path to recover which experts are selected in one
DSpark layer on recoverable acceptance-bundle steps. It is intended to support
expert-coupled local splices after the first sparse block-delta heuristic
proved uninformative.
"""

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
from issue468.dspark_oracle.forward import (
    BLOCK,
    DIM,
    DSPARK_GGUF,
    MAX_POS,
    TARGET_GGUF,
    forward_embed,
    layer_weights,
)
from issue468.dspark_oracle.gguf_loader import index_gguf, load_gguf_dense_only, read_tensor
from issue468.dspark_oracle.hc_primitives import hc_post, hc_pre, rmsnorm
from issue468.dspark_oracle.moe import gate


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-root", required=True)
    ap.add_argument("--baseline-label", required=True)
    ap.add_argument("--oracle-details-json", required=True)
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--top-experts", type=int, default=32)
    ap.add_argument("--top-cohorts", type=int, default=48)
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


def forward_selected_experts(
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
) -> list[tuple[np.ndarray, np.ndarray]]:
    main_proj = T["mtp.0.main_proj.weight"][0]
    main_norm_w = T["mtp.0.main_norm.weight"][0]
    _, main_x_p = forward_embed(main_hidden_prefill, anchor_prefill, main_proj, main_norm_w, embed_w)
    layers = [layer_weights(T, s) for s in range(3)]
    stores = [ExpertStore(DSPARK_GGUF, infos, data_off, s) for s in range(3)]
    slot0_per_layer = []
    for s in range(3):
        slot0_per_layer.append(dspark_attention_prefill(main_x_p, layers[s], cos, sin))
    x, main_x_d = forward_embed(main_hidden_decode, anchor_decode, main_proj, main_norm_w, embed_w)
    input_ids = np.array([anchor_decode], dtype=np.int64)
    selected: list[tuple[np.ndarray, np.ndarray]] = []
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
            selected.append((weights.copy(), idx.copy()))
        _ = stores[s]  # keep parity with forward path even if not used here
        # We do not need the full expert output for later layers because target_layer=2 today.
        if s != target_layer:
            # conservative fallback: keep the tensor moving through the true moe path
            from issue468.dspark_oracle.moe import moe

            ffn_out = moe(yd, input_ids, w, stores[s])
            x = hc_post(ffn_out, residual, post, comb)
    return selected


def main() -> int:
    args = parse_args()
    sweep_root = Path(args.sweep_root).resolve()
    oracle_details = load_oracle_details(Path(args.oracle_details_json).resolve())
    bundles = bundle_dirs(sweep_root)
    kv, T, infos, data_off, _skipped = load_gguf_dense_only(DSPARK_GGUF)
    _t_kv, target_infos, target_data_off = index_gguf(TARGET_GGUF)
    embed_w = read_tensor(TARGET_GGUF, target_infos, target_data_off, "token_embd.weight").astype(np.float32)
    cos, sin = precompute_rope(64, MAX_POS)

    expert_mass: dict[int, float] = defaultdict(float)
    cohort_mass: dict[tuple[int, int], float] = defaultdict(float)
    per_context = {}

    for bundle in bundles:
        ctx = int(bundle.name.split("_")[1])
        base = read_b2(bundle, args.baseline_label)
        oracle_steps = oracle_details.get(ctx)
        if oracle_steps is None:
            raise RuntimeError(f"missing oracle details for context {ctx}")
        weights, details = build_weights(
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
            selected = forward_selected_experts(
                T, infos, data_off, embed_w, cos, sin, mh_p, mh_d, anchor_prefill, anchor_decode, args.layer
            )
            if len(selected) != 1:
                raise RuntimeError(f"expected one selection block for layer {args.layer}")
            gate_weights, gate_idx = selected[0]
            recoverable_rows.append({
                "step": step,
                "weight": float(item["weight"]),
                "selected": gate_idx.tolist(),
                "gate_weights": gate_weights.tolist(),
            })
            for draft_pos in range(gate_idx.shape[0]):
                sample_weight = float(item["weight"])
                for slot in range(gate_idx.shape[1]):
                    expert = int(gate_idx[draft_pos, slot])
                    gate_weight = float(gate_weights[draft_pos, slot])
                    mass = sample_weight * gate_weight
                    expert_mass[expert] += mass
                    cohort_mass[(expert, draft_pos)] += mass
        per_context[bundle.name] = recoverable_rows

    top_experts = [
        {"expert": expert, "mass": mass}
        for expert, mass in sorted(expert_mass.items(), key=lambda kv: (-kv[1], kv[0]))[: args.top_experts]
    ]
    top_cohorts = [
        {"expert": expert, "draft_pos": draft_pos, "mass": mass}
        for (expert, draft_pos), mass in sorted(cohort_mass.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))[
            : args.top_cohorts
        ]
    ]
    report = {
        "layer": args.layer,
        "top_experts": top_experts,
        "top_cohorts": top_cohorts,
        "contexts": per_context,
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
