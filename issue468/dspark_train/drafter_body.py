#!/usr/bin/env python3
"""Activity 5 — DSpark drafter BODY in torch (MPS-capable), RESEARCH INSTRUMENTATION.

Faithful port of dspark_oracle/measure_acceptance_bundle's forward loop (main_proj +
3 DSpark blocks: hc_pre -> MLA attention -> hc_post -> hc_pre -> MoE -> hc_post),
producing the pre-head feature x [BLOCK, HC, DIM] per anchor with the window KV cache.
Routed experts dequanted (Q4_K) to F16 and stacked per layer for vectorized topk
dispatch. Frozen (Activity 5 fidelity); Activity 8 adds body LoRA on top.

Body fidelity gate (run_body_fidelity.py): torch x vs numpy x on the exactness corpus.
The head module (drafter_head.py) is separate and already fidelity-verified.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent.parent            # issue468/
sys.path.insert(0, str(HERE / "dspark_oracle"))
from gguf_loader import load_gguf_dense_only, index_gguf, read_tensor, dequant_q4_k_expert  # noqa: E402
from measure_acceptance_bundle import layer_weights       # noqa: E402  (exact key mapping)

HC, DIM, BLOCK = 4, 4096, 5
HEAD_DIM, ROPE_DIM, N_HEADS, N_GROUPS, O_LORA = 512, 64, 64, 8, 1024
WIN, TOPK, ROUTE_SCALE, SWIGLU_LIMIT = 128, 6, 1.5, 10.0
NORM_EPS, HC_EPS = 1e-6, 1e-6
NOISE_TOK = 128799


def rmsnorm(x, w, eps=NORM_EPS):
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * w


def apply_rotary(x_last, cos, sin):
    xe = x_last[..., 0::2]; xo = x_last[..., 1::2]
    out = torch.empty_like(x_last)
    out[..., 0::2] = xe * cos - xo * sin
    out[..., 1::2] = xe * sin + xo * cos
    return out


def hc_split_sinkhorn(mixes, scale, base, hc=HC, iters=20, eps=HC_EPS):
    pre = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2.0 * torch.sigmoid(mixes[..., hc:2 * hc] * scale[1] + base[hc:2 * hc])
    comb = (mixes[..., 2 * hc:] * scale[2] + base[2 * hc:]).reshape(*mixes.shape[:-1], hc, hc)
    comb = comb - comb.amax(-1, keepdim=True); comb = torch.exp(comb)
    comb = comb / comb.sum(-1, keepdim=True) + eps
    col = comb.sum(-2, keepdim=True); comb = comb / (col + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def hc_pre(x_hc, fn, scale, base):
    b, s, hc, d = x_hc.shape
    flat = x_hc.reshape(b, s, hc * d)
    rsqrt = torch.rsqrt((flat * flat).mean(-1, keepdim=True) + NORM_EPS)
    pre, post, comb = hc_split_sinkhorn((flat @ fn.T) * rsqrt, scale, base, hc)
    return (pre.unsqueeze(-1) * x_hc).sum(2), post, comb


def hc_post(x, residual, post, comb):
    return post.unsqueeze(-1) * x.unsqueeze(-2) + (comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(-2)


def sparse_attn(q, kv_g, sink, scale):
    scores = torch.einsum("bshd,bstd->bsht", q, kv_g) * scale
    m = scores.amax(-1)                                     # [b,s,h]
    sink_t = torch.maximum(m, sink.view(1, 1, -1))          # [b,s,h]
    numer = torch.exp(scores - sink_t.unsqueeze(-1))        # [b,s,h,t]
    denom = numer.sum(-1) + torch.exp(sink.view(1, 1, -1) - sink_t)  # [b,s,h]
    return torch.einsum("bsht,bstd->bshd", numer, kv_g) / denom.unsqueeze(-1)


def softplus(x):
    return torch.where(x > 20, x, torch.log1p(torch.exp(torch.clamp(x, max=20))))


class DrafterBody:
    """Frozen drafter body (plain tensors). forward_prompt returns x per anchor."""
    def __init__(self, p, dev, dt):
        self.dev, self.dt = dev, dt
        g = lambda k: torch.as_tensor(np.ascontiguousarray(p[k]), device=dev, dtype=dt)
        self.main_proj = g("main_proj"); self.main_norm = g("main_norm")
        self.embed_w = g("embed_w"); self.cos = g("cos"); self.sin = g("sin")
        self.layers = [{k: g(f"L{s}_{k}") for k in p["layer_keys"]} for s in range(3)]
        self.exp_gate = g("exp_gate"); self.exp_up = g("exp_up"); self.exp_down = g("exp_down")

    def _moe(self, x, s):
        w = self.layers[s]
        xf = x.reshape(-1, DIM)
        scores = torch.sqrt(softplus(xf @ w["ffn_gate_inp"].T))
        sel = scores + w["ffn_exp_probs_b"]
        idx = sel.topk(TOPK, dim=-1).indices
        wg = torch.gather(scores, -1, idx); wg = wg / (wg.sum(-1, keepdim=True) + 1e-12) * ROUTE_SCALE
        y = torch.zeros_like(xf)
        for k in range(TOPK):
            ids = idx[:, k]
            gg = torch.bmm(xf.unsqueeze(1), self.exp_gate[s][ids].transpose(1, 2)).squeeze(1)
            uu = torch.bmm(xf.unsqueeze(1), self.exp_up[s][ids].transpose(1, 2)).squeeze(1)
            gg = torch.clamp(gg, max=SWIGLU_LIMIT); uu = torch.clamp(uu, -SWIGLU_LIMIT, SWIGLU_LIMIT)
            inter = (gg * torch.sigmoid(gg)) * uu
            out = torch.bmm(inter.unsqueeze(1), self.exp_down[s][ids].transpose(1, 2)).squeeze(1)
            y = y + wg[:, k:k + 1] * out
        g0 = xf @ w["ffn_gate_shexp"].T; u0 = xf @ w["ffn_up_shexp"].T
        g0 = torch.clamp(g0, max=SWIGLU_LIMIT); u0 = torch.clamp(u0, -SWIGLU_LIMIT, SWIGLU_LIMIT)
        y = y + (g0 * torch.sigmoid(g0)) * u0 @ w["ffn_down_shexp"].T
        return y.reshape(1, x.shape[1], DIM)

    def _attn(self, x_draft, win_kv, n_real, s, step):
        w = self.layers[s]; block = x_draft.shape[1]; scale = HEAD_DIM ** -0.5
        qr = rmsnorm(x_draft @ w["q_a"].T, w["q_a_norm"])
        q = (qr @ w["q_b"].T).reshape(1, block, N_HEADS, HEAD_DIM)
        q = q * torch.rsqrt((q * q).mean(-1, keepdim=True) + NORM_EPS)
        cs = self.cos[step + 1:step + 1 + block][None, :, None, :]   # [1,block,1,32] for q/o
        ss = self.sin[step + 1:step + 1 + block][None, :, None, :]
        cs_kv = self.cos[step + 1:step + 1 + block][None, :, :]        # [1,block,32] for kv
        ss_kv = self.sin[step + 1:step + 1 + block][None, :, :]
        q[..., -ROPE_DIM:] = apply_rotary(q[..., -ROPE_DIM:], cs, ss)
        kv = rmsnorm(x_draft @ w["kv"].T, w["kv_a_norm"])
        kv[..., -ROPE_DIM:] = apply_rotary(kv[..., -ROPE_DIM:], cs_kv, ss_kv)
        kv_all = torch.cat([win_kv.unsqueeze(0), kv], dim=1)
        idx = torch.cat([torch.arange(n_real, device=self.dev),
                         torch.arange(WIN, WIN + block, device=self.dev)])
        kv_g = kv_all[:, idx, :].unsqueeze(1).expand(1, block, idx.size(0), HEAD_DIM)
        o = sparse_attn(q, kv_g, w["attn_sinks"], scale)
        o[..., -ROPE_DIM:] = apply_rotary(o[..., -ROPE_DIM:], cs, -ss)
        gd = HEAD_DIM * N_HEADS // N_GROUPS
        wo_a = w["output_a"].reshape(N_GROUPS, O_LORA, gd)
        o_lor = torch.einsum("bsgd,grd->bsgr", o.reshape(1, block, N_GROUPS, gd), wo_a)
        return (o_lor.reshape(1, block, N_GROUPS * O_LORA) @ w["output_b"].T).to(self.dt)

    def forward_prompt(self, main_hidden_seq, anchors, return_layers=False):
        dev, dt = self.dev, self.dt
        mh = torch.as_tensor(np.asarray(main_hidden_seq, np.float32), device=dev, dtype=dt)
        win_kv = [torch.zeros(WIN, HEAD_DIM, device=dev, dtype=dt) for _ in range(3)]
        main_x0 = rmsnorm(mh[0:1].reshape(1, 1, 3 * DIM) @ self.main_proj.T, self.main_norm)
        for s in range(3):
            mkv = rmsnorm(main_x0 @ self.layers[s]["kv"].T, self.layers[s]["kv_a_norm"])
            mkv[..., -ROPE_DIM:] = apply_rotary(mkv[..., -ROPE_DIM:], self.cos[0], self.sin[0])
            win_kv[s][0] = mkv[0, 0]
        n_real = 1; xs = []; layer_xs = []
        for step in range(1, len(anchors) + 1):
            main_x = rmsnorm(mh[step:step + 1].reshape(1, 1, 3 * DIM) @ self.main_proj.T, self.main_norm)
            for s in range(3):
                mkv = rmsnorm(main_x @ self.layers[s]["kv"].T, self.layers[s]["kv_a_norm"])
                mkv[..., -ROPE_DIM:] = apply_rotary(mkv[..., -ROPE_DIM:], self.cos[step], self.sin[step])
                win_kv[s][step % WIN] = mkv[0, 0]
            n_real = min(n_real + 1, WIN)
            draft_ids = torch.full((BLOCK,), NOISE_TOK, dtype=torch.long, device=dev); draft_ids[0] = int(anchors[step - 1])
            x = self.embed_w[draft_ids][None, :, None, :].expand(1, BLOCK, HC, DIM).clone()
            step_layers = []
            for s in range(3):
                w = self.layers[s]
                res = x
                yd, post, comb = hc_pre(x, w["hc_attn_fn"], w["hc_attn_scale"], w["hc_attn_base"])
                x = hc_post(self._attn(rmsnorm(yd, w["attn_norm"]), win_kv[s], n_real, s, step), res, post, comb)
                res = x
                yd, post, comb = hc_pre(x, w["hc_ffn_fn"], w["hc_ffn_scale"], w["hc_ffn_base"])
                x = hc_post(self._moe(rmsnorm(yd, w["ffn_norm"]), s), res, post, comb)
                if return_layers: step_layers.append(x[0].clone())
            xs.append(x[0])
            if return_layers: layer_xs.append(torch.stack(step_layers))  # [3,BLOCK,HC,DIM]
        if return_layers:
            return torch.stack(xs), torch.stack(layer_xs)  # [n,..], [n,3,..]
        return torch.stack(xs)


def build_body(dspark_path, target_path, device, dtype=torch.float32):
    cache = HERE / "dspark_train" / "data" / "drafter_body_weights.npz"
    if cache.exists():
        # mmap so the ~81 GB F32 expert cache stays on disk and is read per-array on
        # demand (Lead 03: avoids an ~80 GB CPU-RAM spike when moving to MPS as F16).
        lo = np.load(cache, allow_pickle=True, mmap_mode='r')
        p = {k: lo[k] for k in lo.files}
        p["layer_keys"] = ["hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm",
                  "q_a", "q_a_norm", "q_b", "kv", "kv_a_norm", "attn_sinks",
                  "output_a", "output_b", "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base",
                  "ffn_norm", "ffn_gate_inp", "ffn_exp_probs_b", "ffn_gate_shexp",
                  "ffn_up_shexp", "ffn_down_shexp"]
        return DrafterBody(p, device, dtype)
    _, T, infos, doff, _ = load_gguf_dense_only(dspark_path)
    layer_keys = ["hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm",
                  "q_a", "q_a_norm", "q_b", "kv", "kv_a_norm", "attn_sinks",
                  "output_a", "output_b", "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base",
                  "ffn_norm", "ffn_gate_inp", "ffn_exp_probs_b", "ffn_gate_shexp",
                  "ffn_up_shexp", "ffn_down_shexp"]
    p = {"main_proj": T["mtp.0.main_proj.weight"][0], "main_norm": T["mtp.0.main_norm.weight"][0],
         "layer_keys": layer_keys}
    for s in range(3):
        lw = layer_weights(T, s)
        for k in layer_keys:
            p[f"L{s}_{k}"] = lw[k]
    eg = np.stack([np.stack([dequant_q4_k_expert(dspark_path, infos, doff, f"mtp.{s}.ffn_gate_exps.weight", e)
                             for e in range(256)]) for s in range(3)])
    eu = np.stack([np.stack([dequant_q4_k_expert(dspark_path, infos, doff, f"mtp.{s}.ffn_up_exps.weight", e)
                             for e in range(256)]) for s in range(3)])
    ed = np.stack([np.stack([dequant_q4_k_expert(dspark_path, infos, doff, f"mtp.{s}.ffn_down_exps.weight", e)
                             for e in range(256)]) for s in range(3)])
    p["exp_gate"] = eg; p["exp_up"] = eu; p["exp_down"] = ed
    _, tinfos, tdoff = index_gguf(target_path)
    p["embed_w"] = read_tensor(target_path, tinfos, tdoff, "token_embd.weight").astype(np.float32)
    half = ROPE_DIM // 2
    freqs = 1.0 / (10000.0 ** (np.arange(0, ROPE_DIM, 2, dtype=np.float64) / ROPE_DIM))
    ang = np.outer(np.arange(4096, dtype=np.float64), freqs)
    p["cos"] = np.cos(ang).astype(np.float32); p["sin"] = np.sin(ang).astype(np.float32)
    save = {k: v for k, v in p.items() if isinstance(v, np.ndarray)}
    np.savez(cache, **save)
    return DrafterBody(p, device, dtype)
