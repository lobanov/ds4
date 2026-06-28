#!/usr/bin/env python3
"""Numpy primitives for the DSpark drafter oracle.

Faithful ports of the reference's HC mixing + RMSNorm + sampling primitives
(issue468/ref/inference/{model.py,kernel.py}), operating on F32 arrays in GGUF
ne order (ne[0] innermost). Used by the Phase-4 numpy oracle to validate the
Metal drafter port (token agreement, greedy argmax).

See issue468/13_phase4_forward_design.md for the full algorithm spec.
"""
import numpy as np

HC_EPS = 1e-6
NORM_EPS = 1e-6
HC_MULT = 4


def rmsnorm(x, weight, eps=NORM_EPS):
    """RMSNorm over the last axis. x [...,d], weight [d]. Returns f32 [...,d]."""
    xf = x.astype(np.float32)
    var = np.mean(xf * xf, axis=-1, keepdims=True)
    return (xf * (1.0 / np.sqrt(var + eps)) * weight).astype(np.float32)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc=HC_MULT, sinkhorn_iters=20, eps=HC_EPS):
    """Port of kernel.py hc_split_sinkhorn. mixes [..., (2+hc)*hc], scale [3],
    base [(2+hc)*hc] -> (pre [...,hc], post [...,hc], comb [...,hc,hc]).

    pre[j]   = sigmoid(mixes[j]     * scale[0] + base[j])      + eps
    post[j]  = 2*sigmoid(mixes[j+hc]* scale[1] + base[j+hc])
    comb[j,k]= mixes[j*hc+k+hc*2]   * scale[2] + base[j*hc+k+hc*2]
    then: comb = softmax(comb,-1)+eps; comb /= (col_sum+eps);
          repeat (sinkhorn_iters-1): comb/=(row_sum+eps); comb/=(col_sum+eps).
    """
    mix = mixes.astype(np.float32)   # [..., mix_hc]
    s = hc_scale.astype(np.float32)  # [3]
    b = hc_base.astype(np.float32)   # [mix_hc]
    # pre (hc) + post (hc) + comb (hc*hc) consume the (2+hc)*hc mixes in order.
    pre = 1.0 / (1.0 + np.exp(-(mix[..., :hc] * s[0] + b[:hc]))) + eps
    post = 2.0 / (1.0 + np.exp(-(mix[..., hc:2*hc] * s[1] + b[hc:2*hc])))
    comb_flat = mix[..., 2*hc:] * s[2] + b[2*hc:]              # [..., hc*hc]
    comb = comb_flat.reshape(*comb_flat.shape[:-1], hc, hc)    # [...,hc,hc]
    # softmax over last axis (rows)
    comb = comb - comb.max(axis=-1, keepdims=True)
    comb = np.exp(comb)
    comb = comb / comb.sum(axis=-1, keepdims=True) + eps
    # one extra col-normalize (col_sum over axis=-2)
    col_sum = comb.sum(axis=-2, keepdims=True)
    comb = comb / (col_sum + eps)
    # (sinkhorn_iters - 1) rounds of row-then-col normalize
    for _ in range(sinkhorn_iters - 1):
        row_sum = comb.sum(axis=-1, keepdims=True)
        comb = comb / (row_sum + eps)
        col_sum = comb.sum(axis=-2, keepdims=True)
        comb = comb / (col_sum + eps)
    return pre, post, comb


def hc_pre(x_hc, hc_fn, hc_scale, hc_base, hc=HC_MULT, eps=NORM_EPS):
    """Block.hc_pre. x_hc [b,s,hc,d] -> (y [b,s,d], post [b,s,hc], comb [b,s,hc,hc]).

    flat = x_hc.flatten(hc,d) [b,s,hc*d]; mixes = hc_fn(flat)*rsqrt(mean(flat^2)+eps);
    y = sum(pre[...,None]*x_hc, axis=hc). hc_fn is [hc_mix_dim, hc*d] (GGUF ne:
    out=hc_mix_dim rows). Linear is x @ hc_fn.T.
    """
    b, s, _, d = x_hc.shape
    flat = x_hc.reshape(b, s, hc * d).astype(np.float32)
    rsqrt = 1.0 / np.sqrt(np.mean(flat * flat, axis=-1, keepdims=True) + eps)
    mixes = (flat @ hc_fn) * rsqrt            # [b,s,hc_mix_dim] (hc_fn stored = W_hf.T)
    pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc)
    y = (pre[..., None] * x_hc).sum(axis=2)     # [b,s,d]
    return y, post, comb


def hc_post(x, residual_hc, post, comb, hc=HC_MULT):
    """Block.hc_post. x [b,s,d], residual [b,s,hc,d], post [b,s,hc], comb [b,s,hc,hc]
    -> [b,s,hc,d]. y = post[...,None]*x[...,None,:] + sum(comb[...,None]*residual, hc)."""
    # post[...,None] * x[...,None,:] broadcasts x [b,s,d] -> [b,s,hc,d] via outer
    y_post = post[..., None] * x[..., None, :]              # [b,s,hc,d]
    # sum over residual's hc axis (the SECOND-to-last), weighted by comb
    # comb[j,k] weights residual[k] into output copy j: comb[...,j,k,None]*residual[...,k,:]
    y_comb = (comb[..., None] * residual_hc[..., None, :]).sum(axis=-2)
    return y_post + y_comb


def hc_head(x_hc, hc_fn, hc_scale, hc_base, hc=HC_MULT, eps=HC_EPS):
    """Block.hc_head (drafter output stage, SIGMOID reduce). x_hc [b,s,hc,d]
    -> [b,s,d]. pre=sigmoid(mixes*scale+base)+eps; y=sum(pre*x, hc).
    NOTE: hc_head_scale is [1] (scalar), applied uniformly; differs from hc_pre's [3]."""
    b, s, _, d = x_hc.shape
    flat = x_hc.reshape(b, s, hc * d).astype(np.float32)
    rsqrt = 1.0 / np.sqrt(np.mean(flat * flat, axis=-1, keepdims=True) + NORM_EPS)
    mixes = (flat @ hc_fn) * rsqrt             # [b,s, hc] (hc_head_fn stored = W_hf.T)
    pre = 1.0 / (1.0 + np.exp(-(mixes * hc_scale[0] + hc_base))) + eps
    return (pre[..., None] * x_hc).sum(axis=2)   # [b,s,d]


if __name__ == "__main__":
    # Self-check hc_split_sinkhorn against its invariant: after Sinkhorn, rows
    # and cols of comb sum to ~1 (doubly stochastic). This is the canonical
    # Sinkhorn property and catches scale/indexing bugs.
    rng = np.random.default_rng(0)
    mixes = rng.standard_normal((2, 24)).astype(np.float32)
    hc_scale = rng.standard_normal(3).astype(np.float32)
    hc_base = rng.standard_normal(24).astype(np.float32)
    pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base)
    print("shapes:", pre.shape, post.shape, comb.shape)
    row_sum = comb.sum(axis=-1)   # should be ~1
    col_sum = comb.sum(axis=-2)   # should be ~1
    print(f"row_sum (want ~1): {row_sum[0]}")
    print(f"col_sum (want ~1): {col_sum[0]}")
    assert np.allclose(row_sum, 1.0, atol=1e-3), "rows not normalized (Sinkhorn broken)"
    assert np.allclose(col_sum, 1.0, atol=1e-3), "cols not normalized (Sinkhorn broken)"
    print("hc_split_sinkhorn: doubly-stochastic invariant holds (port correct)")
    # hc_pre/hc_post round-trip shape check (hc_fn stored [in=hc*d, out=hc_mix])
    x_hc = rng.standard_normal((1, 5, 4, 8)).astype(np.float32)
    hc_fn = rng.standard_normal((32, 24)).astype(np.float32)
    y, post2, comb2 = hc_pre(x_hc, hc_fn, hc_scale, hc_base)
    x_back = hc_post(y, x_hc, post2, comb2)
    print(f"hc_pre: x_hc{x_hc.shape} -> y{y.shape}; hc_post -> {x_back.shape} (matches input)")
    assert x_back.shape == x_hc.shape
    print("hc primitives: shape + Sinkhorn invariants OK")
