#!/usr/bin/env python3
"""DSpark MLA attention primitives for the numpy oracle.

Faithful ports of model.py apply_rotary_emb + DSparkAttention.forward (window-
only MLA, compress_ratio==0) and kernel.py sparse_attn (with the per-head
attention-sink bias added to the softmax denominator only). See
issue468/13_phase4_forward_design.md §4.

All arrays F32, GGUF ne order (ne[0] innermost). Matmuls use the convention
that a stored weight [out, in] (GGUF ne=[in,out]) is applied as x @ W.T.
"""
import numpy as np

ROPE_DIM = 64          # qk_rope_head_dim (rope applied to last 64 of head_dim=512)
HEAD_DIM = 512
NORM_EPS = 1e-6


def precompute_rope(rope_dim, seqlen, theta=10000.0):
    """Standard RoPE freqs (no YaRN; original_seq_len=0 for compress_ratio==0).
    Returns cos[seqlen, rope_dim/2], sin[seqlen, rope_dim/2]."""
    half = rope_dim // 2
    freqs = 1.0 / (theta ** (np.arange(0, rope_dim, 2, dtype=np.float64) / rope_dim))  # [half]
    t = np.arange(seqlen, dtype=np.float64)
    ang = np.outer(t, freqs)            # [seqlen, half]
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def apply_rotary(x_last, cos, sin):
    """Rotate the last rope_dim of x. cos/sin must already be broadcast-shaped to
    match x_last[..., 0::2] (caller builds the shape: e.g. [1,s,1,half] for q,
    [1,s,half] for kv, or [half] to share across all positions). Inverse rotation
    is obtained by passing -sin. Returns rotated [..., rope_dim]."""
    x = x_last.astype(np.float32)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    cos_b = np.broadcast_to(cos, x_even.shape)
    sin_b = np.broadcast_to(sin, x_odd.shape)
    out_even = x_even * cos_b - x_odd * sin_b
    out_odd = x_even * sin_b + x_odd * cos_b
    out = np.empty_like(x)
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd
    return out


def sparse_attn(q, kv_gathered, attn_sink, scale):
    """Sparse attention with a per-head sink. q [b,s,h,d], kv_gathered [b,s,topk,d],
    attn_sink [h]. For each (b,s,h): scores = (q·kv)*scale [topk]; softmax denom
    includes an extra exp(attn_sink[h]-m) term (sink absorbs mass, 0 to numerator).
    Returns o [b,s,h,d]."""
    b, s, h, d = q.shape
    topk = kv_gathered.shape[2]
    # scores [b,s,h,topk]
    scores = np.einsum("bshd,bstd->bsht", q.astype(np.float32), kv_gathered.astype(np.float32)) * scale
    m = scores.max(axis=-1)                       # [b,s,h]
    sink_term = np.maximum(m, attn_sink[None, None, :])  # stable max incl sink
    numer = np.exp(scores - sink_term[..., None])  # [b,s,h,topk]
    denom = numer.sum(axis=-1) + np.exp(attn_sink[None, None, :] - sink_term)  # [b,s,h]
    o = np.einsum("bsht,bstd->bshd", numer, kv_gathered.astype(np.float32)) / denom[..., None]
    return o.astype(np.float32)


def dspark_topk_idxs(window_size, block_size, start_pos):
    """get_dspark_topk_idxs: [0..min(win,start+1)-1] ++ [win..win+block-1].
    Each draft position attends to the window's real anchors + the draft block.
    Returns [block_size, topk_k] (broadcast across batch; per-position identical)."""
    n_win = min(window_size, start_pos + 1)
    win_idx = np.arange(n_win, dtype=np.int32)
    block_idx = (window_size + np.arange(block_size, dtype=np.int32))
    return np.concatenate([win_idx, block_idx])  # [topk_k]; same for all draft positions


def rmsnorm(x, weight, eps=NORM_EPS):
    xf = x.astype(np.float32)
    return xf * (1.0 / np.sqrt(np.mean(xf * xf, axis=-1, keepdims=True) + eps)) * weight


def dspark_attention(x_draft, main_x, w, start_pos, cos_full, sin_full):
    """DSparkAttention decode (start_pos>0). x_draft [1,block,dim], main_x [1,1,dim].
    w: dict of layer attn weights (q_a,q_a_norm,q_b,kv,kv_a_norm,attn_sinks,
    output_a,output_b). cos/sin_full: precomputed RoPE over enough positions.
    Maintains a window KV cache (passed/returned). Returns (o [1,block,dim],
    kv_cache updated).

    Reuses the kernel semantics of model.py DSparkAttention.forward exactly,
    EXCEPT skipping act_quant (FP8 activation sim) — see design doc §4.
    """
    block_size = x_draft.shape[1]
    win = 128
    n_heads = 64
    head_dim = HEAD_DIM
    n_groups = 8
    o_lora = 1024
    scale = head_dim ** -0.5

    # 1. anchor KV from main_x (target hidden), cached at slot start_pos%win
    main_kv = rmsnorm(main_x @ w["kv"].T, w["kv_a_norm"])           # [1,1,512]
    fc = cos_full[start_pos], sin_full[start_pos]   # [32] each (1D)
    main_kv[..., -ROPE_DIM:] = apply_rotary(main_kv[..., -ROPE_DIM:], fc[0], fc[1])

    # build window: slot0=prefill anchor (start_pos=0 earlier), slot start_pos%win=this anchor
    # For the oracle's single-decode scenario we materialize only the slots topk will read.
    # (start_pos=1: real slots {0,1}; zeros elsewhere, masked out by topk.)
    n_win_real = min(win, start_pos + 1)
    win_kv = np.zeros((1, win, head_dim), dtype=np.float32)
    # slot 0 is the prefill anchor; caller must have run prefill to set it.
    # We pass it in via w["_win_slot0_kv"]. Slot start_pos%win = main_kv.
    if "_win_slot0_kv" in w:
        win_kv[:, 0:1] = w["_win_slot0_kv"]
    win_kv[:, start_pos % win:start_pos % win + 1] = main_kv

    # 2. draft-block q,kv
    qr = rmsnorm(x_draft @ w["q_a"].T, w["q_a_norm"])              # [1,block,1024]
    q = (qr @ w["q_b"].T).reshape(1, block_size, n_heads, head_dim)  # [1,block,64,512]
    q = q * (1.0 / np.sqrt(np.mean(q * q, axis=-1, keepdims=True) + NORM_EPS))  # per-head RMSNorm
    # per-position cos/sin for the block positions, shaped to broadcast over heads.
    cs = cos_full[start_pos + 1:start_pos + 1 + block_size]   # [block, 32]
    ss = sin_full[start_pos + 1:start_pos + 1 + block_size]   # [block, 32]
    cs_q = cs[None, :, None, :]   # [1, block, 1, 32] for q [1,block,heads,rope]
    ss_q = ss[None, :, None, :]
    cs_kv = cs[None, :, :]        # [1, block, 32] for kv [1,block,rope]
    ss_kv = ss[None, :, :]
    q[..., -ROPE_DIM:] = apply_rotary(q[..., -ROPE_DIM:], cs_q, ss_q)
    kv = rmsnorm(x_draft @ w["kv"].T, w["kv_a_norm"])              # [1,block,512]
    kv[..., -ROPE_DIM:] = apply_rotary(kv[..., -ROPE_DIM:], cs_kv, ss_kv)
    kv_all = np.concatenate([win_kv, kv], axis=1)                   # [1, win+block, 512]

    # 3. sparse attention over the DSpark topk pattern (identical for all draft positions)
    idx = dspark_topk_idxs(win, block_size, start_pos)             # [topk_k]
    kv_gathered = kv_all[:, idx, :]                                # [1,block,topk,d] (broadcast block)
    kv_gathered = np.broadcast_to(kv_gathered, (1, block_size, idx.size, head_dim))
    o = sparse_attn(q, kv_gathered, w["attn_sinks"], scale)        # [1,block,64,512]
    # inverse rotary on the rope dims of o (same per-position cos/sin; inverse = conj)
    o[..., -ROPE_DIM:] = apply_rotary(o[..., -ROPE_DIM:], cs_q, -ss_q)

    # 4. grouped low-rank output projection: wo_a [8,1024,..] einsum then wo_b
    # o reshaped to [1,block,n_groups, head_dim//n_groups*..] — see model.py:
    #   o = o.view(b,s,n_groups,-1); wo_a viewed [n_groups, o_lora, -1];
    #   o = einsum("bsgd,grd->bsgr", o, wo_a); x = wo_b(o.flatten(2))
    gd = head_dim * n_heads // n_groups   # 512*64/8 = 4096
    o_g = o.reshape(1, block_size, n_groups, gd)                    # [1,block,8,4096]
    # wo_a GGUF ne: [out_low_dim, head_dim*(heads/groups)] = [8192, 4096]; HF/torch [8,1024,512]
    # model.py: wo_a.view(n_groups, o_lora, -1) -> [8, 1024, 512]; einsum bsgd,grd->bsgr
    # but our stored output_a is the fused [out_low, in] = [8192, 4096]. Reshape to [8,1024,4096]? No.
    # Re-derive: HF wo_a weight shape is [n_groups*o_lora, in_per_group]. in_per_group = head_dim*(n_heads/n_groups)
    # = 512*8 = 4096. So wo_a is [8*1024, 4096] = [8192,4096]. view(8,1024,-1) -> [8,1024,4096]?? that's wrong dim.
    # Correct: per group r, the slice is [o_lora, in_per_group]=[1024,4096]. n_groups slices -> [8,1024,4096]? No,
    # total rows = n_groups*o_lora = 8192, cols = in_per_group = 4096. view(n_groups,o_lora,in_per_group) needs
    # 8*1024*4096 elements = way more than 8192*4096. So the per-group input is SMALLER.
    # Actual: in_per_group = head_dim * (n_heads/n_groups) = 512 * (64/8) = 512*8 = 4096. Hmm same.
    # The math: total params = out_low * in = 8192 * 4096. view(8, o_lora=1024, head_dim=512) = 8*1024*512=4.19M,
    # but we have 8192*4096 = 33.5M. So view is (n_groups, o_lora, head_dim) = (8,1024,512)?? That's 4.19M, mismatch.
    # The reference uses einsum "bsgd,grd->bsgr" with o [b,s,8,gd] and wo_a [g=8, r=1024, d=head_dim=512].
    # So gd = head_dim = 512, NOT 4096! o is reshaped to [b,s,8,512]?? But o is [b,s,64,512] (n_heads=64).
    # o.view(b,s,n_groups,-1): 64*512/8 = 4096 per group -> [b,s,8,4096]. Then einsum bsgd,grd needs d match:
    # o[b,s,8,4096] @ wo_a[8,1024,512]? d mismatch (4096 vs 512). So gd in einsum is NOT head_dim.
    # Resolving: wo_a.view(n_groups, o_lora, -1) on a [8192, 4096] tensor -> [8, 1024, 4096]? elements 8*1024*4096=33.5M != 33.5M YES.
    # So wo_a viewed [8,1024,4096], einsum bsgd(4096),grd(4096)->bsgr(1024). d=4096. OK matches!
    wo_a = w["output_a"].reshape(n_groups, o_lora, gd)              # [8,1024,4096] (already HF order)
    # einsum bsgd,grd->bsgr: o_g[b,s,g,d] wo_a[g,r,d] -> [b,s,g,r]
    o_lor = np.einsum("bsgd,grd->bsgr", o_g, wo_a)                # [1,block,8,1024]
    o_flat = o_lor.reshape(1, block_size, n_groups * o_lora)       # [1,block,8192]
    out = o_flat @ w["output_b"].T                                   # [1,block,4096] (output_b HF [out=4096,in=8192])
    return out.astype(np.float32), main_kv[:, 0]   # return slot-1 KV for caller (prefill of next)


def dspark_attention_prefill(main_x, w, cos_full, sin_full):
    """DSparkAttention prefill (start_pos==0): cache the anchor KV at slot 0, no
    attention output (returns x unchanged per model.py). Returns the cached
    main_kv [1,512] for slot 0."""
    main_kv = rmsnorm(main_x @ w["kv"].T, w["kv_a_norm"])          # [1,1,512]
    c = cos_full[0], sin_full[0]   # [32] each (1D -> broadcasts in apply_rotary)
    main_kv[..., -ROPE_DIM:] = apply_rotary(main_kv[..., -ROPE_DIM:], c[0], c[1])
    return main_kv[:, 0]   # [512] slot-0 KV


if __name__ == "__main__":
    # Self-test 1: RoPE norm preservation (rotating preserves L2 norm).
    rng = np.random.default_rng(1)
    cos, sin = precompute_rope(64, 16)
    x = rng.standard_normal((2, 5, 64)).astype(np.float32)
    xr = apply_rotary(x, cos[3], sin[3])
    n_in = np.linalg.norm(x, axis=-1)
    n_out = np.linalg.norm(xr, axis=-1)
    print(f"RoPE norm preservation: max abs diff = {np.abs(n_in-n_out).max():.2e}")
    assert np.allclose(n_in, n_out, atol=1e-4), "RoPE broke norm"

    # Self-test 2: sparse_attn sink — with a large positive sink, output -> 0
    # (all mass absorbed); with very negative sink, normal softmax.
    q = rng.standard_normal((1, 1, 4, 8)).astype(np.float32)
    kvg = rng.standard_normal((1, 1, 3, 8)).astype(np.float32)
    sink_big = np.array([1e9, 1e9, 1e9, 1e9], dtype=np.float32)
    o_sink = sparse_attn(q, kvg, sink_big, 0.1)
    print(f"sparse_attn big-sink output norm (want ~0): {np.linalg.norm(o_sink):.2e}")
    assert np.linalg.norm(o_sink) < 1e-3, "big sink should zero output"
    sink_small = np.array([-1e9, -1e9, -1e9, -1e9], dtype=np.float32)
    o_normal = sparse_attn(q, kvg, sink_small, 0.1)
    # compare to plain softmax-weighted sum
    scores = (q[0,0] @ kvg[0,0].T) * 0.1
    sm = np.exp(scores - scores.max(-1, keepdims=True)); sm /= sm.sum(-1, keepdims=True)
    o_ref = sm @ kvg[0,0]
    print(f"sparse_attn no-sink vs plain softmax: max abs diff = {np.abs(o_normal[0,0]-o_ref).max():.2e}")
    assert np.allclose(o_normal[0, 0], o_ref, atol=1e-4), "no-sink mismatch"

    # Self-test 3: dspark_topk_idxs at start_pos=1
    idx = dspark_topk_idxs(128, 5, 1)
    print(f"dspark_topk_idxs(start_pos=1): {idx.tolist()} (want [0,1,128,129,130,131,132])")
    assert idx.tolist() == [0, 1, 128, 129, 130, 131, 132]
    print("\nattention primitives: RoPE norm + sink + topk invariants OK")
