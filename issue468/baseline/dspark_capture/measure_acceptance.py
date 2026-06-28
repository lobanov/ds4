#!/usr/bin/env python3
"""Measure DSpark greedy acceptance: multi-step drafter with persistent KV cache.

Runs the numpy drafter for N decode steps (building up the window KV cache from
captured main_hidden), and at each step compares the 5 drafted tokens against
the target's true greedy continuation. This is the load-bearing number for the
Phase-6 speedup equation.

Usage: python measure_acceptance.py
(reads main_hidden dumps + greedy tokens from issue468/baseline/dspark_capture/)
"""
import os, sys, json
import numpy as np

ORACLE = os.path.join(os.path.dirname(__file__), "..", "..", "dspark_oracle")
sys.path.insert(0, os.path.normpath(ORACLE))
from gguf_loader import load_gguf_dense_only, index_gguf, read_tensor
from expert_store import ExpertStore
from forward import layer_weights, forward_head, DIM, BLOCK, HC, NOISE_TOK
from attention import (precompute_rope, apply_rotary, dspark_topk_idxs, sparse_attn,
                       ROPE_DIM, HEAD_DIM)
from hc_primitives import rmsnorm, hc_pre, hc_post, hc_head
from moe import moe

# Models live in a sibling repo: <project>/../ds4/gguf/
_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DSPARK = os.path.join(_ROOT, "..", "ds4", "gguf", "dspark.gguf")
TARGET = os.path.join(_ROOT, "..", "ds4", "gguf",
    "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf")
CAP = os.path.dirname(os.path.abspath(__file__))

WIN = 128; N_HEADS = 64; HEAD_DIM_C = 512; N_GROUPS = 8; O_LORA = 1024


def load_mh(pos):
    parts = []
    for L in [40, 41, 42]:
        f = os.path.join(CAP, f"hc_dspark_main_hc-{L}_pos{pos}.bin")
        if not os.path.exists(f): return None
        parts.append(np.fromfile(f, dtype=np.float32).reshape(HC, DIM).mean(0))
    return np.concatenate(parts).astype(np.float32)


def dspark_attn(x_draft, win_kv, n_real, w, cos, sin, start_pos):
    """DSpark decode attention with a persistent window KV cache.
    x_draft [1,block,dim], win_kv [win,512] (first n_real slots populated)."""
    bs = 1; block = x_draft.shape[1]; scale = HEAD_DIM_C ** -0.5
    # q
    qr = rmsnorm(x_draft @ w["q_a"], w["q_a_norm"])
    q = (qr @ w["q_b"]).reshape(bs, block, N_HEADS, HEAD_DIM_C)
    q = q * (1.0 / np.sqrt(np.mean(q*q, axis=-1, keepdims=True) + 1e-6))
    cs = cos[start_pos+1:start_pos+1+block]; ss = sin[start_pos+1:start_pos+1+block]
    q[..., -ROPE_DIM:] = apply_rotary(q[..., -ROPE_DIM:], cs[None,:,None,:], ss[None,:,None,:])
    # draft kv
    kv = rmsnorm(x_draft @ w["kv"], w["kv_a_norm"])
    kv[..., -ROPE_DIM:] = apply_rotary(kv[..., -ROPE_DIM:], cs[None,:,:], ss[None,:,:])
    kv_all = np.concatenate([win_kv[None], kv], axis=1)  # [1, win+block, 512]
    # topk: [0..n_real-1] ++ [win..win+block-1]
    idx = np.concatenate([np.arange(n_real), WIN + np.arange(block)])
    kv_g = np.broadcast_to(kv_all[:, idx], (1, block, idx.size, HEAD_DIM_C))
    o = sparse_attn(q, kv_g, w["attn_sinks"], scale)
    o[..., -ROPE_DIM:] = apply_rotary(o[..., -ROPE_DIM:], cs[None,:,None,:], -ss[None,:,None,:])
    # output proj
    gd = HEAD_DIM_C * N_HEADS // N_GROUPS
    o_g = o.reshape(bs, block, N_GROUPS, gd)
    wo_a = w["output_a"].T.reshape(N_GROUPS, O_LORA, gd)
    o_lor = np.einsum("bsgd,grd->bsgr", o_g, wo_a)
    return (o_lor.reshape(bs, block, N_GROUPS*O_LORA) @ w["output_b"]).astype(np.float32)


def main():
    print("=== load weights ===", flush=True)
    _, T, infos, doff, _ = load_gguf_dense_only(DSPARK)
    _, ti, tdo = index_gguf(TARGET)
    embed_w = read_tensor(TARGET, ti, tdo, "token_embd.weight").astype(np.float32)
    lm_head = read_tensor(TARGET, ti, tdo, "output.weight").astype(np.float32)
    cos, sin = precompute_rope(64, 4096)
    greedy = json.load(open(os.path.join(CAP, "greedy25_tokens.json")))
    main_proj = T["mtp.0.main_proj.weight"][0]; main_norm_w = T["mtp.0.main_norm.weight"][0]
    layers = [layer_weights(T, s) for s in range(3)]
    stores = [ExpertStore(DSPARK, infos, doff, s) for s in range(3)]
    win_kv = [np.zeros((WIN, HEAD_DIM_C), dtype=np.float32) for _ in range(3)]

    POS0 = 152  # first generated position
    # prefill (start_pos=0): cache anchor KV at slot 0
    mh0 = load_mh(POS0)
    main_x0 = rmsnorm(mh0.reshape(1,1,3*DIM) @ main_proj, main_norm_w)
    for s in range(3):
        mkv = rmsnorm(main_x0 @ layers[s]["kv"], layers[s]["kv_a_norm"])
        mkv[...,-ROPE_DIM:] = apply_rotary(mkv[...,-ROPE_DIM:], cos[0], sin[0])
        win_kv[s][0] = mkv[0,0]
    n_real = 1

    print(f"\nprefill done (slot 0 cached). Measuring acceptance at decode steps 1-{len(greedy)-6}...")
    print(f"{'step':>4} {'pos':>4} {'KV':>3} {'draft':>40} {'target':>40} {'match':>5} {'prefix':>6}")
    total_match = 0; total_pos = 0; prefix_hist = {}

    for step in range(1, len(greedy) - 5):  # need 5 future tokens to compare
        pos = POS0 + step
        anchor = greedy[step]  # token generated at this pos
        mh = load_mh(pos)
        if mh is None: break
        main_x = rmsnorm(mh.reshape(1,1,3*DIM) @ main_proj, main_norm_w)
        # cache anchor KV
        for s in range(3):
            mkv = rmsnorm(main_x @ layers[s]["kv"], layers[s]["kv_a_norm"])
            mkv[...,-ROPE_DIM:] = apply_rotary(mkv[...,-ROPE_DIM:], cos[step], sin[step])
            win_kv[s][step % WIN] = mkv[0,0]
        n_real = min(n_real + 1, WIN)
        # embed noise block
        draft_ids = np.full(BLOCK, NOISE_TOK, dtype=np.int64); draft_ids[0] = anchor
        x = embed_w[:, draft_ids].T[None]; x = np.repeat(x[:,:,None,:], HC, axis=2)
        # 3 blocks
        for s in range(3):
            res = x
            yd,post,comb = hc_pre(x, layers[s]["hc_attn_fn"], layers[s]["hc_attn_scale"], layers[s]["hc_attn_base"])
            yd = rmsnorm(yd, layers[s]["attn_norm"])
            ao = dspark_attn(yd, win_kv[s], n_real, layers[s], cos, sin, step)
            x = hc_post(ao, res, post, comb)
            res = x
            yd,post,comb = hc_pre(x, layers[s]["hc_ffn_fn"], layers[s]["hc_ffn_scale"], layers[s]["hc_ffn_base"])
            yd = rmsnorm(yd, layers[s]["ffn_norm"])
            fo = moe(yd, np.array([0]), layers[s], stores[s])
            x = hc_post(fo, res, post, comb)
        # head
        out, logits = forward_head(x, anchor, None, T["mtp.2.norm.weight"][0],
            T["mtp.2.hc_head_fn.weight"][0], T["mtp.2.hc_head_scale.weight"][0],
            T["mtp.2.hc_head_base.weight"][0], T["mtp.2.markov_head.markov_w1.weight"][0],
            T["mtp.2.markov_head.markov_w2.weight"][0], T["mtp.2.confidence_head.proj.weight"][0],
            lm_head, temp=1.0)
        draft = out[1:].tolist()
        target = greedy[step+1:step+6]
        match = sum(1 for d,t in zip(draft,target) if d==t)
        prefix = 0
        for i,(d,t) in enumerate(zip(draft,target)):
            if d==t: prefix=i+1
            else: break
        total_match += match; total_pos += 5
        prefix_hist[prefix] = prefix_hist.get(prefix, 0) + 1
        d_str = str(draft[:5])[:38]; t_str = str(target[:5])[:38]
        print(f"{step:>4} {pos:>4} {n_real:>3} {d_str:>40} {t_str:>40} {match:>5} {prefix:>6}", flush=True)

    print(f"\n=== SUMMARY ({total_pos//5} decode steps) ===")
    print(f"  Total token positions: {total_pos}")
    print(f"  Greedy matches: {total_match}/{total_pos} = {100*total_match/max(total_pos,1):.1f}%")
    print(f"  Accepted prefix distribution: {dict(sorted(prefix_hist.items()))}")
    if total_pos > 0:
        avg_prefix = sum(k*v for k,v in prefix_hist.items()) / sum(prefix_hist.values())
        print(f"  Average accepted prefix: {avg_prefix:.2f}/5")
        print(f"\n  Speedup estimate (preliminary, greedy upper bound):")
        print(f"    With {avg_prefix:.1f} avg accepted tokens per {BLOCK}-token draft,")
        print(f"    speedup requires: (draft_cost + verify_cost) / {avg_prefix:.1f} < plain_decode_cost")
        print(f"    (Phase-1 verify(L) curve + measured draft cost needed for final answer)")


if __name__ == "__main__":
    main()
