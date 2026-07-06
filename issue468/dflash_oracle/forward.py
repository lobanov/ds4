#!/usr/bin/env python3
"""DFlash drafter forward — numpy oracle.

Faithful port of dflash_mlx/model.py (DFlashDraftModel + DFlashAttention +
DFlashDecoderLayer) + the draft-cycle mechanics from dflash_mlx/runtime.py
(generate_dflash_once), operating on F32 numpy.

DFlash is a 5-layer dense Llama-style drafter (GQA 64 Q / 1 KV, head_dim 256,
SwiGLU MLP, plain residuals). Its attention is non-causal "cross + noise":
queries come from the [anchor + MASK*7] noise block; K/V come from the projected
5-layer target context (appended to a ContextOnlyDraftKVCache) concatenated with
the noise block's own K/V. Output is the shared target lm_head over a 32000-token
draft vocabulary, mapped to target ids via `d2t`.

Usage:
  python forward.py --smoke            # random target context; validate shapes/values
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from safetensors_loader import load_safetensors

DSPARK_TARGET_GGUF_DEFAULT = (HERE.parents[3] / "ds4" / "gguf" /
                              "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf").resolve()

# ---- architecture constants (from config.json) ----
N_LAYERS = 5
HIDDEN = 4096
INTER = 2048
N_HEADS = 64
N_KV = 1
HEAD_DIM = 256
ROPE_THETA = 10000.0
RMS_EPS = 1e-6
BLOCK_SIZE = 8                 # [anchor + 7 draft]
N_DRAFT = BLOCK_SIZE - 1       # 7 speculative tokens
DRAFT_VOCAB = 32000            # lm_head output dim
MASK_TOKEN_ID = 1
TARGET_LAYER_IDS = [3, 13, 23, 32, 42]
HC = 4                         # target hyper-connection mult
# drafter context KV cache windowing (from runtime.py defaults)
DRAFT_SINK = 64
DRAFT_WINDOW = 1024
MAX_POS = 8192                 # rope table size


def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = RMS_EPS) -> np.ndarray:
    xf = x.astype(np.float32)
    var = np.mean(xf * xf, axis=-1, keepdims=True)
    return (xf * (1.0 / np.sqrt(var + eps)) * weight).astype(np.float32)


# ---- rope (rotate-half, mlx traditional=False / Llama style) ----
def precompute_rope(head_dim: int, max_pos: int, base: float = ROPE_THETA):
    inv_freq = base ** (-(np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))  # [D/2]
    t = np.arange(max_pos, dtype=np.float32)
    freqs = np.outer(t, inv_freq)                 # [max_pos, D/2]
    return np.cos(freqs).astype(np.float32), np.sin(freqs).astype(np.float32)


def _rotate_half(x: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rope(x: np.ndarray, cos_half: np.ndarray, sin_half: np.ndarray) -> np.ndarray:
    """x [..., seq, head_dim]; cos_half/sin_half [seq, head_dim/2] (duplicated internally)."""
    cos_full = np.concatenate([cos_half, cos_half], axis=-1)   # [seq, head_dim]
    sin_full = np.concatenate([sin_half, sin_half], axis=-1)
    return x * cos_full + _rotate_half(x) * sin_full


class ContextOnlyDraftKVCache:
    """Mirrors dflash_mlx.model.ContextOnlyDraftKVCache (sink + sliding window)."""
    def __init__(self, sink_size: int = DRAFT_SINK, window_size: int = DRAFT_WINDOW):
        self.sink_size = sink_size
        self.window_size = window_size
        self.keys: np.ndarray | None = None     # [b, n_kv, len, hd]
        self.values: np.ndarray | None = None
        self.offset = 0

    def append_context(self, keys: np.ndarray, values: np.ndarray, num_positions: int) -> None:
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = np.concatenate([self.keys, keys], axis=2)
            self.values = np.concatenate([self.values, values], axis=2)
        self.offset += int(num_positions)
        self._apply_window()

    def _apply_window(self) -> None:
        cache_len = self.keys.shape[2]
        max_len = self.sink_size + self.window_size
        if cache_len <= max_len:
            return
        self.keys = np.concatenate([self.keys[:, :, :self.sink_size, :],
                                    self.keys[:, :, -self.window_size:, :]], axis=2)
        self.values = np.concatenate([self.values[:, :, :self.sink_size, :],
                                      self.values[:, :, -self.window_size:, :]], axis=2)

    def fetch(self):
        return self.keys, self.values


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def dflash_attention(hidden_states: np.ndarray, target_hidden: np.ndarray, w: dict,
                     cache: ContextOnlyDraftKVCache, cos_tbl, sin_tbl,
                     q_offset: int, ctx_offset: int) -> np.ndarray:
    """Non-causal cross+noise GQA attention.
    hidden_states [b, block, HIDDEN] (noise block); target_hidden [b, ctx, HIDDEN] (projected context).
    Returns [b, block, HIDDEN]."""
    b, block, _ = hidden_states.shape
    ctx = target_hidden.shape[1]
    scale = HEAD_DIM ** -0.5

    q = (hidden_states @ w["q_proj"].T).reshape(b, block, N_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    q = rmsnorm(q, w["q_norm"])                                             # over head_dim
    ctx_k = (target_hidden @ w["k_proj"].T).reshape(b, ctx, N_KV, HEAD_DIM).transpose(0, 2, 1, 3)
    ctx_k = rmsnorm(ctx_k, w["k_norm"])
    ctx_v = (target_hidden @ w["v_proj"].T).reshape(b, ctx, N_KV, HEAD_DIM).transpose(0, 2, 1, 3)
    noise_k = (hidden_states @ w["k_proj"].T).reshape(b, block, N_KV, HEAD_DIM).transpose(0, 2, 1, 3)
    noise_k = rmsnorm(noise_k, w["k_norm"])
    noise_v = (hidden_states @ w["v_proj"].T).reshape(b, block, N_KV, HEAD_DIM).transpose(0, 2, 1, 3)

    q = apply_rope(q, cos_tbl[q_offset:q_offset + block], sin_tbl[q_offset:q_offset + block])
    ctx_k = apply_rope(ctx_k, cos_tbl[ctx_offset:ctx_offset + ctx], sin_tbl[ctx_offset:ctx_offset + ctx])
    noise_k = apply_rope(noise_k, cos_tbl[q_offset:q_offset + block], sin_tbl[q_offset:q_offset + block])

    cache.append_context(ctx_k, ctx_v, ctx)
    cached_k, cached_v = cache.fetch()
    keys = np.concatenate([cached_k, noise_k], axis=2)                      # [b, n_kv, L, hd]
    values = np.concatenate([cached_v, noise_v], axis=2)
    # GQA: replicate the single KV head to all Q heads
    keys = np.broadcast_to(keys, (b, N_HEADS, keys.shape[2], HEAD_DIM))
    values = np.broadcast_to(values, (b, N_HEADS, values.shape[2], HEAD_DIM))

    scores = np.einsum("bhqd,bhkd->bhqk", q, keys) * scale                  # [b, h, block, L]
    probs = _softmax(scores, axis=-1)
    out = np.einsum("bhqk,bhkd->bhqd", probs, values)                       # [b, h, block, hd]
    out = out.transpose(0, 2, 1, 3).reshape(b, block, N_HEADS * HEAD_DIM)
    return out @ w["o_proj"].T


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def swiglu_mlp(x: np.ndarray, w: dict) -> np.ndarray:
    gate = x @ w["gate_proj"].T
    up = x @ w["up_proj"].T
    inter = (gate * _sigmoid(gate)) * up                     # silu(gate)*up
    return inter @ w["down_proj"].T


def decoder_layer(noise_emb: np.ndarray, projected_ctx: np.ndarray, w: dict,
                  cache: ContextOnlyDraftKVCache, cos_tbl, sin_tbl,
                  q_offset: int, ctx_offset: int) -> np.ndarray:
    residual = noise_emb
    h = rmsnorm(noise_emb, w["input_layernorm"])
    h = dflash_attention(h, projected_ctx, w["self_attn"], cache, cos_tbl, sin_tbl, q_offset, ctx_offset)
    noise_emb = residual + h
    residual = noise_emb
    h = rmsnorm(noise_emb, w["post_attention_layernorm"])
    h = swiglu_mlp(h, w["mlp"])
    return residual + h


def build_drafter_weights(w: dict):
    """Split the flat safetensors dict into (globals, layers).
    globals: {fc, hidden_norm, norm, lm_head, d2t, t2d, embed}
    layers[i]: {input_layernorm, post_attention_layernorm, self_attn:{q/k/v/o_proj,q/k_norm}, mlp:{gate/up/down_proj}}."""
    layers = []
    for i in range(N_LAYERS):
        p = f"layers.{i}."
        layers.append({
            "input_layernorm": w[p + "input_layernorm.weight"],
            "post_attention_layernorm": w[p + "post_attention_layernorm.weight"],
            "self_attn": {
                "q_proj": w[p + "self_attn.q_proj.weight"],
                "k_proj": w[p + "self_attn.k_proj.weight"],
                "v_proj": w[p + "self_attn.v_proj.weight"],
                "o_proj": w[p + "self_attn.o_proj.weight"],
                "q_norm": w[p + "self_attn.q_norm.weight"],
                "k_norm": w[p + "self_attn.k_norm.weight"],
            },
            "mlp": {
                "gate_proj": w[p + "mlp.gate_proj.weight"],
                "up_proj": w[p + "mlp.up_proj.weight"],
                "down_proj": w[p + "mlp.down_proj.weight"],
            },
        })
    g = {
        "fc": w["fc.weight"],
        "hidden_norm": w["hidden_norm.weight"],
        "norm": w["norm.weight"],
        "lm_head": w["lm_head.weight"],
        "d2t": w["d2t"],
        "t2d": w["t2d"],
        "embed": w["embed_tokens.weight"],
    }
    return g, layers


def project_target_context(target_hidden_5hc: np.ndarray, g: dict) -> np.ndarray:
    """target_hidden_5hc [b, ctx, 5*HC*HIDDEN=81920] -> projected [b, ctx, HIDDEN]."""
    return rmsnorm(target_hidden_5hc @ g["fc"].T, g["hidden_norm"])


def forward(g: dict, layers: list, noise_embedding: np.ndarray, target_hidden_5hc: np.ndarray,
            caches: list[ContextOnlyDraftKVCache], cos_tbl, sin_tbl,
            q_offset: int, ctx_offset: int) -> np.ndarray:
    """Run the 5-layer drafter. Returns draft logits [b, block, DRAFT_VOCAB]."""
    projected = project_target_context(target_hidden_5hc, g)
    h = noise_embedding.astype(np.float32)
    for i in range(N_LAYERS):
        h = decoder_layer(h, projected, layers[i], caches[i], cos_tbl, sin_tbl, q_offset, ctx_offset)
    h = rmsnorm(h, g["norm"])
    logits = h @ g["lm_head"].T                                # [b, block, 32000]
    return logits.astype(np.float32)


def draft_tokens_from_logits(draft_logits: np.ndarray, d2t: np.ndarray) -> list[int]:
    """draft_logits [block-1, DRAFT_VOCAB] (positions 1..block-1) -> target token ids.
    d2t is an OFFSET: target_id = draft_idx + d2t[draft_idx] (verified:
    np.nonzero(t2d)[0] == d2t + arange(32000))."""
    idx = np.argmax(draft_logits, axis=-1)                                  # [block-1]
    return [int(i + d2t[i]) for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--weights", default=str(HERE / "model.safetensors"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not args.smoke:
        ap.error("currently --smoke only")

    print(f"loading DFlash weights: {args.weights}", flush=True)
    w = load_safetensors(args.weights)
    print(f"  {len(w)} tensors", flush=True)
    g, layers = build_drafter_weights(w)
    cos_tbl, sin_tbl = precompute_rope(HEAD_DIM, MAX_POS, ROPE_THETA)
    d2t = g["d2t"]
    rng = np.random.default_rng(args.seed)

    # smoke: random target context (5 layers x HC x HIDDEN) for 8 committed positions,
    # random noise embedding block [anchor + 7 MASK]. Validate shapes + non-degenerate logits.
    ctx_len = 8
    block = BLOCK_SIZE
    target_5hc = (rng.standard_normal((1, ctx_len, len(TARGET_LAYER_IDS) * HC * HIDDEN)) * 0.02).astype(np.float32)
    noise_emb = (rng.standard_normal((1, block, HIDDEN)) * 0.02).astype(np.float32)

    caches = [ContextOnlyDraftKVCache() for _ in range(N_LAYERS)]
    q_off, ctx_off = ctx_len, 0
    logits = forward(g, layers, noise_emb, target_5hc, caches, cos_tbl, sin_tbl, q_off, ctx_off)
    print(f"forward logits shape: {logits.shape} (want (1, {block}, {DRAFT_VOCAB}))")
    print(f"  logits range [{logits.min():.3f}, {logits.max():.3f}] std {logits.std():.4f}")
    draft_logits = logits[0, 1:, :]                          # positions 1..7
    toks = draft_tokens_from_logits(draft_logits, d2t)
    print(f"  draft target-token ids (smoke): {toks}")
    assert logits.shape == (1, block, DRAFT_VOCAB), "logits shape mismatch"
    assert np.isfinite(logits).all(), "non-finite logits"
    print("DFlash forward smoke: OK")


if __name__ == "__main__":
    main()
