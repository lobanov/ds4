#!/usr/bin/env python3
"""Measure oracle draft-prefix agreement against one retained target bundle.

For temp=0 this is greedy-prefix agreement against the deterministic target
stream. For temp>0 it is sampled-stream prefix agreement against the retained
selected target continuation from the same seeded run.

This module exposes three pieces so a bulk runner can amortize the expensive
one-time loads (target embed/lm_head, drafter dense tensors, RoPE) across many
bundles and many drafter GGUFs without re-spawning a process per bundle:

  - build_model_ctx(model_path)   : load target embed_w + lm_head + RoPE once
  - build_drafter_ctx(dspark_path): load drafter dense tensors + layers + stores
  - measure_bundle(bundle_dir, mctx, dctx) : run one bundle, return summary dict

The CLI main() builds both contexts from args and calls measure_bundle, so the
single-bundle path is behaviour-identical to the bulk path (single source of
truth for the measurement numerics).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gguf_loader import load_gguf_dense_only, index_gguf, read_tensor
from expert_store import ExpertStore
from forward import layer_weights, forward_head, DIM, BLOCK, HC, NOISE_TOK
from build_main_hidden_from_captures import CANONICAL_INPUTS
from attention import precompute_rope, apply_rotary, sparse_attn, ROPE_DIM, HEAD_DIM
from hc_primitives import rmsnorm, hc_pre, hc_post
from moe import moe

WIN = 128
N_HEADS = 64
N_GROUPS = 8
O_LORA = 1024
DEFAULT_MODEL = (HERE.parents[3] / "ds4" / "gguf" / "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf").resolve()
DEFAULT_DSPARK = (HERE.parents[3] / "ds4" / "gguf" / "dspark.gguf").resolve()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-dir", required=True)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--dspark", default=str(DEFAULT_DSPARK))
    ap.add_argument("--json-out")
    # Anchor-reuse falsifier (issue468 Lead 01): at each step draft from the
    # one-position-stale target hidden (main_hidden[pos-1]) instead of the true
    # main_hidden[pos], keeping the anchor token unchanged. Tests whether the
    # drafter's acceptance survives drafting from the last-accepted-position
    # hidden with the correction token entering only as embedding.
    ap.add_argument("--reuse-mode", choices=["none", "lag", "backfill"],
                    default="none",
                    help="anchor-reuse mode: none (baseline) / lag (stale, consistent) / backfill (stale current, true prior)")
    ap.add_argument("--reuse", choices=["lag"], default=None,
                    help="deprecated shorthand for --reuse-mode=lag")
    return ap.parse_args()


def load_mh(bundle_dir: Path, pos: int) -> np.ndarray | None:
    packed = bundle_dir / "oracle" / CANONICAL_INPUTS
    if packed.exists():
        data = np.load(packed)
        positions = data["positions"]
        hits = np.where(positions == pos)[0]
        if hits.size:
            return data["main_hidden"][int(hits[0])]
    path = bundle_dir / "oracle" / f"main_hidden_pos{pos}.npy"
    if not path.exists():
        return None
    return np.load(path)


def dspark_attn(x_draft: np.ndarray, win_kv: np.ndarray, n_real: int,
                w: dict, cos: np.ndarray, sin: np.ndarray, start_pos: int) -> np.ndarray:
    bs = 1
    block = x_draft.shape[1]
    scale = HEAD_DIM ** -0.5
    qr = rmsnorm(x_draft @ w["q_a"].T, w["q_a_norm"])
    q = (qr @ w["q_b"].T).reshape(bs, block, N_HEADS, HEAD_DIM)
    q = q * (1.0 / np.sqrt(np.mean(q * q, axis=-1, keepdims=True) + 1e-6))
    cs = cos[start_pos + 1:start_pos + 1 + block]
    ss = sin[start_pos + 1:start_pos + 1 + block]
    q[..., -ROPE_DIM:] = apply_rotary(q[..., -ROPE_DIM:], cs[None, :, None, :], ss[None, :, None, :])
    kv = rmsnorm(x_draft @ w["kv"].T, w["kv_a_norm"])
    kv[..., -ROPE_DIM:] = apply_rotary(kv[..., -ROPE_DIM:], cs[None, :, :], ss[None, :, :])
    kv_all = np.concatenate([win_kv[None], kv], axis=1)
    idx = np.concatenate([np.arange(n_real), WIN + np.arange(block)])
    kv_g = np.broadcast_to(kv_all[:, idx], (1, block, idx.size, HEAD_DIM))
    o = sparse_attn(q, kv_g, w["attn_sinks"], scale)
    o[..., -ROPE_DIM:] = apply_rotary(o[..., -ROPE_DIM:], cs[None, :, None, :], -ss[None, :, None, :])
    gd = HEAD_DIM * N_HEADS // N_GROUPS
    o_g = o.reshape(bs, block, N_GROUPS, gd)
    wo_a = w["output_a"].reshape(N_GROUPS, O_LORA, gd)
    o_lor = np.einsum("bsgd,grd->bsgr", o_g, wo_a)
    return (o_lor.reshape(bs, block, N_GROUPS * O_LORA) @ w["output_b"].T).astype(np.float32)


def build_model_ctx(model_path: str) -> dict:
    """Load target-model-derived tensors that are independent of the drafter GGUF.
    Load once, reuse across every drafter and every bundle."""
    _, ti, tdo = index_gguf(model_path)
    embed_w = read_tensor(model_path, ti, tdo, "token_embd.weight").astype(np.float32)
    lm_head = read_tensor(model_path, ti, tdo, "output.weight").astype(np.float32)
    cos, sin = precompute_rope(64, 4096)
    return {"embed_w": embed_w, "lm_head": lm_head, "cos": cos, "sin": sin}


def build_drafter_ctx(dspark_path: str) -> dict:
    """Load drafter-specific dense tensors + per-layer weight dicts + expert stores.
    Load once per drafter GGUF, reuse across every bundle."""
    _, T, infos, doff, _ = load_gguf_dense_only(dspark_path)
    return {
        "T": T,
        "infos": infos,
        "doff": doff,
        "main_proj": T["mtp.0.main_proj.weight"][0],
        "main_norm_w": T["mtp.0.main_norm.weight"][0],
        "layers": [layer_weights(T, s) for s in range(3)],
        "stores": [ExpertStore(dspark_path, infos, doff, s) for s in range(3)],
    }


def measure_bundle(bundle_dir: Path, mctx: dict, dctx: dict, *, candidate: str | None = None,
                    reuse_mode: str = "none") -> dict:
    """Run the drafter forward over one retained bundle and return the acceptance
    summary dict. Fresh per-bundle KV window; does NOT clear expert caches (the
    caller decides memory policy).

    Anchor-reuse modes (issue468 Lead 01 falsifier):
    -----------------------------------------------
    main_hidden[pos] is the POST-token hidden at position pos (captured from
    dump_hc_ffn_post; target_tokens[k] sits at position pos0+k, pos0=prompt_tokens).
    main_hidden enters the draft ONLY via the per-step KV-window entry
    win_kv[s][step] = mkv(main_x); the residual stream is seeded from embeddings.
    So the reuse substitution is entirely about which hidden sources main_x.

      reuse_mode="none"   : main_x from main_hidden[pos] (true post-token hidden at
                            the anchor's own position). BASELINE.
      reuse_mode="lag"    : main_x from main_hidden[pos-1] (the last-accepted-
                            position hidden; correction token enters as embedding).
                            Applied every step, so win_kv is a consistent one-
                            position lag of baseline. Init mh0 at pos0 (the first
                            GENERATED token's post-token hidden = the first anchor)
                            is unchanged; the first cycle's anchor is a real,
                            processed token, so there is no reuse at step 1 yet.
      reuse_mode="backfill": current step drafts with main_x from main_hidden[pos-1]
                            (stale), but AFTER drafting the slot is backfilled with
                            main_x from main_hidden[pos] (true) so subsequent steps
                            see a true prior context. Models a folded verifier that
                            commits accepted positions with true hiddens and only
                            the just-rejected cycle's anchor slot is stale. This is
                            the codex-GATE1-requested sensitivity to the KV-window
                            modeling choice.

    Both reuse modes feed the same correction token (target_tokens[step]) as the
    anchor; only the hidden sourcing differs."""
    bundle_dir = Path(bundle_dir)
    T = dctx["T"]
    layers = dctx["layers"]
    stores = dctx["stores"]
    main_proj = dctx["main_proj"]
    main_norm_w = dctx["main_norm_w"]
    embed_w = mctx["embed_w"]
    lm_head = mctx["lm_head"]
    cos, sin = mctx["cos"], mctx["sin"]

    manifest = json.loads((bundle_dir / "bundle_manifest.json").read_text())
    target_tokens = json.loads((bundle_dir / "target_selected_tokens.json").read_text())
    pos0 = int(manifest["prompt_tokens"])
    label = f"{manifest['prompt_name']}@temp={manifest['temperature']}"

    win_kv = [np.zeros((WIN, HEAD_DIM), dtype=np.float32) for _ in range(3)]

    mh0 = load_mh(bundle_dir, pos0)
    if mh0 is None:
        raise FileNotFoundError(f"missing oracle main_hidden for pos {pos0}")
    main_x0 = rmsnorm(mh0.reshape(1, 1, 3 * DIM) @ main_proj.T, main_norm_w)
    for s in range(3):
        mkv = rmsnorm(main_x0 @ layers[s]["kv"].T, layers[s]["kv_a_norm"])
        mkv[..., -ROPE_DIM:] = apply_rotary(mkv[..., -ROPE_DIM:], cos[0], sin[0])
        win_kv[s][0] = mkv[0, 0]
    n_real = 1

    rows = []
    prefix_hist = {k: 0 for k in range(BLOCK + 1)}
    total_match = 0
    total_pos = 0
    max_step = min(len(target_tokens) - BLOCK - 1, int(manifest["measure_steps"]))

    for step in range(1, max_step + 1):
        pos = pos0 + step
        anchor = int(target_tokens[step])
        # reuse modes: which hidden sources main_x for the current draft.
        #   none      -> main_hidden[pos]      (baseline, true anchor hidden)
        #   lag       -> main_hidden[pos-1]    (stale; kept in win_kv -> consistent lag)
        #   backfill  -> main_hidden[pos-1] for the draft, main_hidden[pos] backfilled after
        reuse = reuse_mode in ("lag", "backfill")
        mh = load_mh(bundle_dir, pos - 1 if reuse else pos)
        if mh is None:
            break
        main_x = rmsnorm(mh.reshape(1, 1, 3 * DIM) @ main_proj.T, main_norm_w)
        if reuse_mode == "backfill":
            mh_true = load_mh(bundle_dir, pos)
            if mh_true is None:
                break
            main_x_true = rmsnorm(mh_true.reshape(1, 1, 3 * DIM) @ main_proj.T, main_norm_w)
        for s in range(3):
            mkv = rmsnorm(main_x @ layers[s]["kv"].T, layers[s]["kv_a_norm"])
            mkv[..., -ROPE_DIM:] = apply_rotary(mkv[..., -ROPE_DIM:], cos[step], sin[step])
            win_kv[s][step % WIN] = mkv[0, 0]
        n_real = min(n_real + 1, WIN)
        draft_ids = np.full(BLOCK, NOISE_TOK, dtype=np.int64)
        draft_ids[0] = anchor
        x = embed_w[draft_ids][None]
        x = np.repeat(x[:, :, None, :], HC, axis=2)
        for s in range(3):
            res = x
            yd, post, comb = hc_pre(x, layers[s]["hc_attn_fn"], layers[s]["hc_attn_scale"], layers[s]["hc_attn_base"])
            yd = rmsnorm(yd, layers[s]["attn_norm"])
            ao = dspark_attn(yd, win_kv[s], n_real, layers[s], cos, sin, step)
            x = hc_post(ao, res, post, comb)
            res = x
            yd, post, comb = hc_pre(x, layers[s]["hc_ffn_fn"], layers[s]["hc_ffn_scale"], layers[s]["hc_ffn_base"])
            yd = rmsnorm(yd, layers[s]["ffn_norm"])
            fo = moe(yd, np.array([0]), layers[s], stores[s])
            x = hc_post(fo, res, post, comb)
        # backfill: overwrite the current slot with the TRUE hidden so future
        # steps see a true prior context (only the just-drafted cycle was stale).
        if reuse_mode == "backfill":
            for s in range(3):
                mkv_t = rmsnorm(main_x_true @ layers[s]["kv"].T, layers[s]["kv_a_norm"])
                mkv_t[..., -ROPE_DIM:] = apply_rotary(mkv_t[..., -ROPE_DIM:], cos[step], sin[step])
                win_kv[s][step % WIN] = mkv_t[0, 0]
        out, _ = forward_head(
            x,
            anchor,
            None,
            T["mtp.2.norm.weight"][0],
            T["mtp.2.hc_head_fn.weight"][0],
            T["mtp.2.hc_head_scale.weight"][0],
            T["mtp.2.hc_head_base.weight"][0],
            T["mtp.2.markov_head.markov_w1.weight"][0],
            T["mtp.2.markov_head.markov_w2.weight"][0],
            T["mtp.2.confidence_head.proj.weight"][0],
            lm_head,
            temp=1.0,
        )
        draft = [int(v) for v in out[1:].tolist()]
        target = [int(v) for v in target_tokens[step + 1:step + 1 + BLOCK]]
        match = sum(1 for d, t in zip(draft, target) if d == t)
        prefix = 0
        for i, (d, t) in enumerate(zip(draft, target)):
            if d == t:
                prefix = i + 1
            else:
                break
        prefix_hist[prefix] += 1
        total_match += match
        total_pos += BLOCK
        rows.append({
            "step": step,
            "position": pos,
            "anchor": anchor,
            "draft": draft,
            "target": target,
            "match": match,
            "prefix": prefix,
        })

    n_rows = len(rows)
    avg_prefix = float(sum(row["prefix"] for row in rows) / n_rows) if n_rows else 0.0
    summary = {
        "label": label,
        "candidate": candidate,
        "reuse_mode": reuse_mode,
        "bundle_dir": str(bundle_dir),
        "prompt_name": manifest["prompt_name"],
        "temperature": manifest["temperature"],
        "seed": manifest["seed"],
        "prompt_tokens": manifest["prompt_tokens"],
        "measure_steps": n_rows,
        "total_positions": total_pos,
        "total_match": total_match,
        "match_pct": (100.0 * total_match / total_pos) if total_pos else 0.0,
        "average_prefix": avg_prefix,
        "prefix_hist": prefix_hist,
        "rows": rows,
        "reference_mode": "greedy" if float(manifest["temperature"]) == 0.0 else "sampled-stream",
    }
    return summary


def main() -> int:
    args = parse_args()
    mctx = build_model_ctx(args.model)
    dctx = build_drafter_ctx(args.dspark)
    summary = measure_bundle(Path(args.bundle_dir), mctx, dctx,
                             reuse_mode=(args.reuse if args.reuse else args.reuse_mode))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2) + "\n")
    else:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
