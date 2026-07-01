#!/usr/bin/env python3
"""Build dense imatrix entries from recoverable-step DSpark activations.

The runtime DSpark collector only emits routed-expert tensors today. This
script synthesizes imatrix-compatible dense vectors directly from sweep bundles
so dense legal Q8_0 probes can reuse the same recoverable-gap weighting logic
without changing frozen inference/runtime code.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
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
    DIM,
    DSPARK_GGUF,
    MAX_POS,
    TARGET_GGUF,
    forward_embed,
    layer_weights,
)
from issue468.dspark_oracle.gguf_loader import index_gguf, load_gguf_dense_only, read_tensor
from issue468.dspark_oracle.hc_primitives import hc_post, hc_pre, rmsnorm
from issue468.dspark_oracle.moe import SWIGLU_LIMIT, moe


MAIN_PROJ = "mtp.0.main_proj.weight"
SUPPORTED = {
    MAIN_PROJ,
    "mtp.2.ffn_gate_shexp.weight",
    "mtp.2.ffn_up_shexp.weight",
    "mtp.2.ffn_down_shexp.weight",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-root", required=True)
    ap.add_argument("--baseline-label", required=True)
    ap.add_argument("--oracle-details-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tensor-name", action="append")
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
        vec = np.fromfile(path, dtype=np.float32).reshape(4, -1).mean(axis=0)
        parts.append(vec)
    return np.concatenate(parts, axis=0).astype(np.float32)


def build_tensor_sizes(names: list[str]) -> dict[str, int]:
    _kv, infos, data_off = index_gguf(DSPARK_GGUF)
    sizes = {}
    for name in names:
        info = infos.get(name)
        if info is None:
            raise RuntimeError(f"tensor not found in dspark.gguf: {name}")
        dims, _tt, _off = info
        sizes[name] = int(dims[0])
    return sizes


def write_imatrix(path: Path, entries: list[tuple[str, np.ndarray, int]]) -> None:
    with path.open("wb") as f:
        f.write(struct.pack("<i", len(entries)))
        for name, values, ncall in entries:
            raw_name = name.encode("utf-8")
            arr = values.astype(np.float32, copy=False)
            f.write(struct.pack("<i", len(raw_name)))
            f.write(raw_name)
            f.write(struct.pack("<i", ncall))
            f.write(struct.pack("<i", int(arr.size)))
            f.write(arr.tobytes(order="C"))


def load_layer2_ffn_state(
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
) -> tuple[np.ndarray, np.ndarray]:
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
        if s == 2:
            x2 = yd[0].astype(np.float32)
            g = x2 @ w["ffn_gate_shexp"].T
            u = x2 @ w["ffn_up_shexp"].T
            if SWIGLU_LIMIT > 0:
                u = np.clip(u, -SWIGLU_LIMIT, SWIGLU_LIMIT)
                g = np.clip(g, None, SWIGLU_LIMIT)
            h = g * (1.0 / (1.0 + np.exp(-g)))
            inter = (h * u).astype(np.float32)
            return x2, inter
        ffn_out = moe(yd, input_ids, w, stores[s])
        x = hc_post(ffn_out, residual, post, comb)

    raise RuntimeError("failed to reach layer 2 FFN state")


def main() -> int:
    args = parse_args()
    requested = args.tensor_name or [MAIN_PROJ]
    tensor_names = []
    for name in requested:
        if name not in SUPPORTED:
            raise RuntimeError(f"unsupported tensor-name {name!r}; expected one of {sorted(SUPPORTED)}")
        if name not in tensor_names:
            tensor_names.append(name)

    sweep_root = Path(args.sweep_root).resolve()
    oracle_details = load_oracle_details(Path(args.oracle_details_json).resolve())
    bundles = bundle_dirs(sweep_root)
    sizes = build_tensor_sizes(tensor_names)

    need_layer2 = any(name != MAIN_PROJ for name in tensor_names)
    T = infos = data_off = embed_w = cos = sin = None
    if need_layer2:
        _kv, T, infos, data_off, _skipped = load_gguf_dense_only(DSPARK_GGUF)
        _t_kv, target_infos, target_data_off = index_gguf(TARGET_GGUF)
        embed_w = read_tensor(TARGET_GGUF, target_infos, target_data_off, "token_embd.weight").astype(np.float32)
        cos, sin = precompute_rope(64, MAX_POS)

    accum = {name: np.zeros(sizes[name], dtype=np.float64) for name in tensor_names}
    ncall = {name: 0 for name in tensor_names}

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
        greedy = None
        if need_layer2:
            greedy = json.loads((bundle / "target_greedy.json").read_text())
        for item in details:
            if not item["recoverable"]:
                continue
            weight = float(item["weight"])
            step = int(item["step"])
            pos_decode = ctx + step

            if MAIN_PROJ in accum:
                x = load_main_hidden(bundle, pos_decode)
                accum[MAIN_PROJ] += weight * np.square(x, dtype=np.float64)
                ncall[MAIN_PROJ] += 1

            if need_layer2:
                assert T is not None and infos is not None and data_off is not None
                assert embed_w is not None and cos is not None and sin is not None and greedy is not None
                pos_prefill = ctx + step - 1
                mh_p = load_main_hidden(bundle, pos_prefill)
                mh_d = load_main_hidden(bundle, pos_decode)
                anchor_prefill = int(greedy[step - 1])
                anchor_decode = int(greedy[step])
                x2, inter = load_layer2_ffn_state(
                    T,
                    infos,
                    data_off,
                    embed_w,
                    cos,
                    sin,
                    mh_p,
                    mh_d,
                    anchor_prefill,
                    anchor_decode,
                )
                if "mtp.2.ffn_gate_shexp.weight" in accum:
                    accum["mtp.2.ffn_gate_shexp.weight"] += weight * np.square(x2, dtype=np.float64).sum(axis=0)
                    ncall["mtp.2.ffn_gate_shexp.weight"] += x2.shape[0]
                if "mtp.2.ffn_up_shexp.weight" in accum:
                    accum["mtp.2.ffn_up_shexp.weight"] += weight * np.square(x2, dtype=np.float64).sum(axis=0)
                    ncall["mtp.2.ffn_up_shexp.weight"] += x2.shape[0]
                if "mtp.2.ffn_down_shexp.weight" in accum:
                    accum["mtp.2.ffn_down_shexp.weight"] += weight * np.square(inter, dtype=np.float64).sum(axis=0)
                    ncall["mtp.2.ffn_down_shexp.weight"] += inter.shape[0]

    entries = []
    for name in tensor_names:
        if ncall[name] == 0:
            raise RuntimeError(f"no recoverable steps accumulated for {name}")
        arr = accum[name].astype(np.float32)
        if arr.size != sizes[name]:
            raise RuntimeError(f"size mismatch for {name}: {arr.size} vs {sizes[name]}")
        entries.append((name, arr, ncall[name]))

    out_path = Path(args.out).resolve()
    write_imatrix(out_path, entries)
    print(f"wrote {out_path}")
    for name, arr, nc in entries:
        print(f"tensor {name}")
        print(f"ncall {nc}")
        print(f"nval {arr.size}")
        print(f"sum {float(arr.sum())}")
        print(f"max {float(arr.max())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
