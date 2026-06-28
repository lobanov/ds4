#!/usr/bin/env python3
"""DSpark MoE for the numpy oracle.

Faithful port of model.py Gate + MoE + Expert (SwiGLU), F32. Drafter layers
(43+, n_hash_layers=3 targets layers 0-2) use SCORE-BASED routing (no tid2eid).
See issue468/13_phase4_forward_design.md §5.

score_func = "sqrtsoftplus": scores = softplus(x@gate).sqrt(); bias shifts topk
selection only (not weights); weights normalized to sum 1 then *route_scale=1.5.
n_activated_experts=6, n_routed=256, n_shared=1, moe_inter_dim=2048, dim=4096.
"""
import numpy as np

NORM_EPS = 1e-6
ROUTE_SCALE = 1.5
TOPK = 6


def softplus(x):
    # numerically stable
    return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20))))


def gate(x, gate_inp, exp_probs_b, topk=TOPK):
    """x [...,dim] -> (weights [...,topk], indices [...,topk]). sqrtsoftplus,
    bias shifts topk but not weights; weights gathered from ORIGINAL scores,
    normalized, *route_scale. (model.py Gate.forward)."""
    scores = softplus((x @ gate_inp.T).astype(np.float32))     # [...,256] (gate_inp HF [out=256,in=dim])
    scores = np.sqrt(scores)
    sel_scores = scores + exp_probs_b                           # bias for topk only
    # topk indices along last axis
    idx = np.argpartition(sel_scores, -topk, axis=-1)[..., -topk:]  # unsorted topk
    # sort the topk by score descending for determinism
    order = np.argsort(-np.take_along_axis(sel_scores, idx, axis=-1), axis=-1)
    idx = np.take_along_axis(idx, order, axis=-1)               # [...,topk] sorted desc
    weights = np.take_along_axis(scores, idx, axis=-1)          # original (no bias)
    weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-12)
    weights = weights * ROUTE_SCALE
    return weights.astype(np.float32), idx


def swiglu_expert(x, w_gate, w_up, w_down):
    """Single expert SwiGLU: down(silu(gate(x)) * up(x)). x [...,dim] -> [...,dim].
    w_gate/w_up/w_down are expert slices in HF [out,in] order (standard linear).
    Apply as x @ W.T."""
    g = x @ w_gate.T
    u = x @ w_up.T
    # silu(g) = g * sigmoid(g)
    h = g * (1.0 / (1.0 + np.exp(-g)))
    inter = h * u
    return inter @ w_down.T


def moe(x, input_ids, w, expert_store, topk=TOPK):
    """x [1,block,dim], input_ids (unused for score routing). w has the dense
    tensors (gate, shared, norms). expert_store yields per-expert [inter,dim]
    gate/up/down lazily: expert_store.expert(e) -> (wg,wu,wd).
    Returns [1,block,dim]."""
    b, s, dim = x.shape
    xf = x.reshape(b * s, dim).astype(np.float32)
    weights, idx = gate(xf, w["ffn_gate_inp"], w["ffn_exp_probs_b"], topk)  # [bs, topk]
    y = np.zeros_like(xf)
    cache = {}  # per-forward memo: (e) -> (wg,wu,wd)
    for t in range(xf.shape[0]):
        for k in range(topk):
            e = int(idx[t, k])
            if e not in cache:
                cache[e] = expert_store.expert(e)
            wg, wu, wd = cache[e]
            y[t] += weights[t, k] * swiglu_expert(xf[t], wg, wu, wd)
    y += swiglu_expert(xf, w["ffn_gate_shexp"], w["ffn_up_shexp"], w["ffn_down_shexp"])
    return y.reshape(b, s, dim).astype(np.float32)


if __name__ == "__main__":
    rng = np.random.default_rng(2)
    # gate: weights sum to route_scale, indices unique (gate_inp stored [in=dim,out=256])
    x = rng.standard_normal((2, 4096)).astype(np.float32)
    gi = rng.standard_normal((4096, 256)).astype(np.float32) * 0.01
    eb = rng.standard_normal(256).astype(np.float32) * 0.01
    w, idx = gate(x, gi, eb)
    print(f"gate: weights shape {w.shape}, sum={w.sum(-1)} (want ~{ROUTE_SCALE})")
    print(f"  indices unique per row: {[len(np.unique(idx[i])) for i in range(2)]} (want 6)")
    assert np.allclose(w.sum(-1), ROUTE_SCALE, atol=1e-3)
    assert all(len(np.unique(idx[i])) == 6 for i in range(2))
    # swiglu + moe shape + determinism (weights in GGUF-ne [in,out] order)
    dim, inter = 4096, 2048
    wg = rng.standard_normal((dim, inter)).astype(np.float32) * 0.01
    wu = rng.standard_normal((dim, inter)).astype(np.float32) * 0.01
    wd = rng.standard_normal((inter, dim)).astype(np.float32) * 0.01
    xe = rng.standard_normal(dim).astype(np.float32)
    y1 = swiglu_expert(xe, wg, wu, wd)
    y2 = swiglu_expert(xe, wg, wu, wd)
    print(f"swiglu: deterministic {np.array_equal(y1, y2)}, output shape {y1.shape}")
    assert np.array_equal(y1, y2) and y1.shape == (dim,)
    print("moe primitives: gate invariants + swiglu OK")
