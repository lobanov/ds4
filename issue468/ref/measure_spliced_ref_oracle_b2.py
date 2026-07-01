#!/usr/bin/env python3
"""Measure baseline-oracle B2 after splicing selected dense tensors from ref-ckpt."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ORACLE = os.path.join(os.path.dirname(__file__), "..", "dspark_oracle")
sys.path.insert(0, os.path.normpath(ORACLE))

from attention import precompute_rope, apply_rotary, dspark_topk_idxs, sparse_attn, dspark_attention_prefill, ROPE_DIM, HEAD_DIM
from expert_store import ExpertStore
from forward import layer_weights, DIM, BLOCK, HC, NOISE_TOK
from gguf_loader import load_gguf_dense_only, index_gguf, read_tensor
from hc_primitives import rmsnorm, hc_pre, hc_post, hc_head
from moe import moe
from ref_ckpt_loader import load_ref_ckpt_dense


ROOT = Path(__file__).resolve().parents[2]
DSPARK_GGUF = os.path.expanduser("~/ds4/gguf/dspark.gguf")
TARGET_GGUF = os.path.expanduser("~/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf")
REF_CKPT = os.path.expanduser("~/ds4/ref-ckpt/model0-mp1.safetensors")
CAP = ROOT / "issue468" / "baseline" / "dspark_capture"
WIN = 128
N_HEADS = 64
N_GROUPS = 8
O_LORA = 1024
RNG = np.random.default_rng(20260701)


def load_mh(capture_dir: Path, pos: int):
    if (capture_dir / f"main_hidden_pos{pos}.npy").exists():
        return np.load(capture_dir / f"main_hidden_pos{pos}.npy").astype(np.float32)
    parts = []
    for layer in [40, 41, 42]:
        path = capture_dir / f"hc_dspark_main_hc-{layer}_pos{pos}.bin"
        if not path.exists():
            return None
        parts.append(np.fromfile(path, dtype=np.float32).reshape(HC, DIM).mean(0))
    return np.concatenate(parts).astype(np.float32)


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def dspark_attn(x_draft, main_x, w, win_kv, start_pos, cos, sin):
    bs = 1
    block = x_draft.shape[1]
    scale = HEAD_DIM ** -0.5
    n_heads = N_HEADS
    head_dim = HEAD_DIM
    n_groups = N_GROUPS
    o_lora = O_LORA
    main_kv = rmsnorm(main_x @ w["kv"].T, w["kv_a_norm"])
    fc = cos[start_pos], sin[start_pos]
    main_kv[..., -ROPE_DIM:] = apply_rotary(main_kv[..., -ROPE_DIM:], fc[0], fc[1])
    win_kv[start_pos % WIN] = main_kv[0, 0]
    qr = rmsnorm(x_draft @ w["q_a"].T, w["q_a_norm"])
    q = (qr @ w["q_b"].T).reshape(bs, block, n_heads, head_dim)
    q = q * (1.0 / np.sqrt(np.mean(q * q, axis=-1, keepdims=True) + 1e-6))
    cs = cos[start_pos + 1 : start_pos + 1 + block]
    ss = sin[start_pos + 1 : start_pos + 1 + block]
    cs_q = cs[None, :, None, :]
    ss_q = ss[None, :, None, :]
    cs_kv = cs[None, :, :]
    ss_kv = ss[None, :, :]
    q[..., -ROPE_DIM:] = apply_rotary(q[..., -ROPE_DIM:], cs_q, ss_q)
    kv = rmsnorm(x_draft @ w["kv"].T, w["kv_a_norm"])
    kv[..., -ROPE_DIM:] = apply_rotary(kv[..., -ROPE_DIM:], cs_kv, ss_kv)
    kv_all = np.concatenate([win_kv[None], kv], axis=1)
    idx = dspark_topk_idxs(WIN, block, start_pos)
    kv_g = np.broadcast_to(kv_all[:, idx, :], (bs, block, idx.size, head_dim))
    o = sparse_attn(q, kv_g, w["attn_sinks"], scale)
    o[..., -ROPE_DIM:] = apply_rotary(o[..., -ROPE_DIM:], cs_q, -ss_q)
    gd = head_dim * n_heads // n_groups
    o_g = o.reshape(bs, block, n_groups, gd)
    wo_a = w["output_a"].reshape(n_groups, o_lora, gd)
    o_lor = np.einsum("bsgd,grd->bsgr", o_g, wo_a)
    o_flat = o_lor.reshape(bs, block, n_groups * o_lora)
    return (o_flat @ w["output_b"].T).astype(np.float32)


def run_blocks(x, main_x, step, layers, stores, win_kv_list, cos, sin):
    for s in range(3):
        res = x
        yd, post, comb = hc_pre(x, layers[s]["hc_attn_fn"], layers[s]["hc_attn_scale"], layers[s]["hc_attn_base"])
        yd = rmsnorm(yd, layers[s]["attn_norm"])
        ao = dspark_attn(yd, main_x, layers[s], win_kv_list[s], step, cos, sin)
        x = hc_post(ao, res, post, comb)
        res = x
        yd, post, comb = hc_pre(x, layers[s]["hc_ffn_fn"], layers[s]["hc_ffn_scale"], layers[s]["hc_ffn_base"])
        yd = rmsnorm(yd, layers[s]["ffn_norm"])
        fo = moe(yd, np.array([0]), layers[s], stores[s])
        x = hc_post(fo, res, post, comb)
    return x


def head_base_logits(h, T, lm_head):
    x = hc_head(h, T["mtp.2.hc_head_fn.weight"][0], T["mtp.2.hc_head_scale.weight"][0], T["mtp.2.hc_head_base.weight"][0])
    x = rmsnorm(x, T["mtp.2.norm.weight"][0])
    return (x[0] @ lm_head.T).astype(np.float32)


def markov_bias(prev_tok, markov_w1, markov_w2):
    return (markov_w1[prev_tok] @ markov_w2.T).astype(np.float32)


def load_baseline_oracle():
    _, T, infos, data_off, skipped = load_gguf_dense_only(DSPARK_GGUF)
    _, ti, tdo = index_gguf(TARGET_GGUF)
    embed_w = read_tensor(TARGET_GGUF, ti, tdo, "token_embd.weight").astype(np.float32)
    lm_head = read_tensor(TARGET_GGUF, ti, tdo, "output.weight").astype(np.float32)
    return T, infos, data_off, embed_w, lm_head, skipped


def splice_dense_tensors(T_base, T_ref, names: list[str]):
    T = dict(T_base)
    for name in names:
        if name not in T_ref:
            raise KeyError(f"reference checkpoint does not provide tensor {name}")
        T[name] = (T_ref[name][0].copy(),)
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", default=str(CAP))
    ap.add_argument("--target-json", default=str(CAP / "target_topk200.json"))
    ap.add_argument("--greedy-json", default=str(CAP / "greedy25_tokens.json"))
    ap.add_argument("--pos0", type=int, default=152)
    ap.add_argument("--trials", type=int, default=128)
    ap.add_argument("--steps-cap", type=int, default=19)
    ap.add_argument("--replace", action="append", required=True, help="oracle tensor name to splice from ref-ckpt; repeatable")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--label", default="spliced-ref-oracle")
    args = ap.parse_args()

    capture_dir = Path(args.capture_dir)
    T_base, infos, data_off, embed_w, lm_head, _ = load_baseline_oracle()
    _, T_ref, _, _, _, _ = load_ref_ckpt_dense(REF_CKPT)
    T = splice_dense_tensors(T_base, T_ref, args.replace)

    cos, sin = precompute_rope(64, 4096)
    greedy = json.loads(Path(args.greedy_json).read_text())
    tgt_steps = json.loads(Path(args.target_json).read_text())["steps"]
    main_proj = T["mtp.0.main_proj.weight"][0]
    main_norm_w = T["mtp.0.main_norm.weight"][0]
    layers = [layer_weights(T, s) for s in range(3)]
    stores = [ExpertStore(DSPARK_GGUF, infos, data_off, s) for s in range(3)]
    mw1 = T["mtp.2.markov_head.markov_w1.weight"][0]
    mw2 = T["mtp.2.markov_head.markov_w2.weight"][0]

    def target_p(step_idx):
        entries = tgt_steps[step_idx]["top_logprobs"]
        toks = np.array([e["token"]["id"] for e in entries], dtype=np.int64)
        probs = np.exp(np.array([e["logprob"] for e in entries], dtype=np.float64))
        probs /= probs.sum()
        return toks, probs

    win_kv = [np.zeros((WIN, HEAD_DIM), dtype=np.float32) for _ in range(3)]
    mh0 = load_mh(capture_dir, args.pos0)
    main_x0 = rmsnorm(mh0.reshape(1, 1, 3 * DIM) @ main_proj.T, main_norm_w)
    for s in range(3):
        slot0 = dspark_attention_prefill(main_x0, layers[s], cos, sin)
        win_kv[s][0] = slot0

    max_step = min(len(greedy) - 6, len(tgt_steps) - 6)
    if args.steps_cap > 0:
        max_step = min(max_step, args.steps_cap)
    base_logits_cache = {}
    for step in range(1, max_step + 1):
        pos = args.pos0 + step
        mh = load_mh(capture_dir, pos)
        if mh is None:
            break
        main_x = rmsnorm(mh.reshape(1, 1, 3 * DIM) @ main_proj.T, main_norm_w)
        anchor = greedy[step]
        draft_ids = np.full(BLOCK, NOISE_TOK, dtype=np.int64)
        draft_ids[0] = anchor
        x = embed_w[draft_ids][None]
        x = np.repeat(x[:, :, None, :], HC, axis=2)
        h = run_blocks(x, main_x, step, layers, stores, win_kv, cos, sin)
        base_logits_cache[step] = head_base_logits(h, T, lm_head)

    committed_hist = {}
    pos_accept = [0, 0, 0, 0, 0]
    pos_total = [0, 0, 0, 0, 0]
    per_step_committed = []
    per_step_accepted = []
    analytical = [[] for _ in range(BLOCK)]
    greedy_hist = {}

    for step in base_logits_cache:
        base = base_logits_cache[step]
        prev = greedy[step]
        prefix = 0
        for j in range(BLOCK):
            li = base[j] + markov_bias(prev, mw1, mw2)
            dtok = int(np.argmax(li))
            ttok = greedy[step + 1 + j]
            if dtok == ttok:
                prefix = j + 1
                prev = dtok
            else:
                break
        greedy_hist[prefix] = greedy_hist.get(prefix, 0) + 1

        prev_truth = greedy[step]
        for j in range(BLOCK):
            q = softmax(base[j] + markov_bias(prev_truth, mw1, mw2))
            ptoks, pprobs = target_p(step + 1 + j)
            p = np.zeros_like(q)
            p[ptoks] = pprobs
            analytical[j].append(1.0 - 0.5 * np.abs(p - q).sum())
            prev_truth = greedy[step + 1 + j]

        step_committed = 0.0
        step_accepted = 0.0
        for _ in range(args.trials):
            prev = greedy[step]
            n_committed = 0
            n_accepted = 0
            for j in range(BLOCK):
                q = softmax(base[j] + markov_bias(prev, mw1, mw2))
                x = int(RNG.choice(len(q), p=q))
                ptoks, pprobs = target_p(step + 1 + j)
                pmap = dict(zip(ptoks.tolist(), pprobs.tolist()))
                px = pmap.get(x, 0.0)
                qx = q[x]
                accept_p = min(1.0, px / qx) if qx > 0 else 0.0
                pos_total[j] += 1
                if RNG.random() < accept_p:
                    n_accepted += 1
                    n_committed += 1
                    pos_accept[j] += 1
                    prev = x
                else:
                    n_committed += 1
                    break
            step_committed += n_committed
            step_accepted += n_accepted
            committed_hist[n_committed] = committed_hist.get(n_committed, 0) + 1
        per_step_committed.append(step_committed / args.trials)
        per_step_accepted.append(step_accepted / args.trials)

    analytical_means = [float(np.mean(v)) if v else 0.0 for v in analytical]
    analytical_prefix = 0.0
    reach = 1.0
    for v in analytical_means:
        analytical_prefix += reach * v
        reach *= v

    out = {
        "label": args.label,
        "target_json": args.target_json,
        "greedy_json": args.greedy_json,
        "pos0": args.pos0,
        "n_steps": len(per_step_committed),
        "trials": args.trials,
        "replace": args.replace,
        "block": BLOCK,
        "vocab": int(lm_head.shape[0]),
        "average_committed": float(np.mean(per_step_committed)),
        "average_accepted": float(np.mean(per_step_accepted)),
        "per_step_committed": per_step_committed,
        "per_step_accepted": per_step_accepted,
        "per_position_accept_rate": [pos_accept[i] / max(pos_total[i], 1) for i in range(BLOCK)],
        "committed_hist": {str(k): v for k, v in sorted(committed_hist.items())},
        "greedy_avg_prefix": float(sum(k * v for k, v in greedy_hist.items()) / max(sum(greedy_hist.values()), 1)),
        "greedy_hist": {str(k): v for k, v in sorted(greedy_hist.items())},
        "analytical_accept_upper_bound_by_pos": analytical_means,
        "analytical_committed_upper_bound": float(analytical_prefix + 1.0),
    }
    Path(args.out_json).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out_json}")
    print(
        f"avg_committed={out['average_committed']:.4f} "
        f"avg_accepted={out['average_accepted']:.4f} "
        f"greedy_avg_prefix={out['greedy_avg_prefix']:.4f} "
        f"analytic_upper={out['analytical_committed_upper_bound']:.4f}"
    )


if __name__ == "__main__":
    main()
