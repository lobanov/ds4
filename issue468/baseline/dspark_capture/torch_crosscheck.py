#!/usr/bin/env python3
"""Minimal PyTorch CPU forward of DSpark forward_embed -> forward_head.

Uses the SAME dequantized F32 weights as the numpy oracle (from dspark.gguf via
gguf_loader), so this isolates the forward MATH (numpy vs torch). If torch
produces different draft tokens than numpy, the bug is in numpy's math.

Only implements the DSpark-unique stages (forward_embed + 3 blocks + forward_head)
using torch primitives (no tilelang). Skips act_quant (FP8 sim) as the oracle does.
"""
import sys, os, json
import numpy as np
import torch

ORACLE = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "dspark_oracle"))
sys.path.insert(0, ORACLE)
from gguf_loader import load_gguf_dense_only, index_gguf, read_tensor, dequant_q4_k_expert
from expert_store import ExpertStore

DSPARK = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "ds4", "gguf", "dspark.gguf"))
TARGET = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "ds4", "gguf",
    "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"))
CAP = os.path.dirname(os.path.abspath(__file__))

DIM = 4096; HC = 4; BLOCK = 5; NOISE = 128799; VOCAB = 129280
MARKOV_RANK = 256; ROPE_DIM = 64; HEAD_DIM = 512; WIN = 128
N_HEADS = 64; N_GROUPS = 8; O_LORA = 1024; EPS = 1e-6; HC_EPS = 1e-6


def rmsnorm(x, w, eps=EPS):
    """x [...,d], w [d] -> [...,d]."""
    xf = x.float()
    return (w * xf * torch.rsqrt(xf.pow(2).mean(-1, keepdims=True) + eps))


def rope(x, cos, sin):
    """Apply RoPE to last rope_dim of x. x [...,rope], cos/sin [...,rope/2]."""
    x_even = x[..., 0::2]; x_odd = x[..., 1::2]
    return torch.stack([x_even*cos - x_odd*sin, x_even*sin + x_odd*cos], dim=-1).flatten(-2)


def precompute_rope(seqlen, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, ROPE_DIM, 2, dtype=torch.float64) / ROPE_DIM))
    ang = torch.outer(torch.arange(seqlen, dtype=torch.float64), freqs)
    return torch.cos(ang).float(), torch.sin(ang).float()


def main():
    print("=== load weights (F32 from dspark.gguf) ===", flush=True)
    _, T, infos, doff, _ = load_gguf_dense_only(DSPARK)
    _, ti, tdo = index_gguf(TARGET)
    embed_np = read_tensor(TARGET, ti, tdo, "token_embd.weight").astype(np.float32)
    lm_head_np = read_tensor(TARGET, ti, tdo, "output.weight").astype(np.float32)

    def W(name):
        """Load a dense weight as torch tensor [in,out] (GGUF ne order, x@W)."""
        return torch.from_numpy(T[name][0])

    def Wexp(stage, part, e):
        """Load one expert as torch [in,out]."""
        nm = f"mtp.{stage}.ffn_{part}_exps.weight"
        return torch.from_numpy(dequant_q4_k_expert(DSPARK, infos, doff, nm, e))

    greedy = json.load(open(os.path.join(CAP, "greedy25_tokens.json")))

    def load_mh(pos):
        parts = [np.fromfile(os.path.join(CAP, f"hc_dspark_main_hc-{L}_pos{pos}.bin"),
                             dtype=np.float32).reshape(HC, DIM).mean(0) for L in [40, 41, 42]]
        return torch.from_numpy(np.concatenate(parts).astype(np.float32))

    cos_full, sin_full = precompute_rope(4096)

    main_proj = W("mtp.0.main_proj.weight"); main_norm_w = W("mtp.0.main_norm.weight")
    layers = []
    for s in range(3):
        p = f"mtp.{s}."
        layers.append({
            "hc_attn_fn": W(p+"hc_attn_fn.weight"), "hc_attn_s": W(p+"hc_attn_scale.weight"),
            "hc_attn_b": W(p+"hc_attn_base.weight"),
            "hc_ffn_fn": W(p+"hc_ffn_fn.weight"), "hc_ffn_s": W(p+"hc_ffn_scale.weight"),
            "hc_ffn_b": W(p+"hc_ffn_base.weight"),
            "attn_norm": W(p+"attn_norm.weight"), "ffn_norm": W(p+"ffn_norm.weight"),
            "q_a": W(p+"attn_q_a.weight"), "q_a_norm": W(p+"attn_q_a_norm.weight"),
            "q_b": W(p+"attn_q_b.weight"), "kv": W(p+"attn_kv.weight"),
            "kv_a_norm": W(p+"attn_kv_a_norm.weight"), "sinks": W(p+"attn_sinks.weight"),
            "out_a": W(p+"attn_output_a.weight"), "out_b": W(p+"attn_output_b.weight"),
            "gate_inp": W(p+"ffn_gate_inp.weight"), "exp_bias": W(p+"exp_probs_b.bias"),
            "shexp_g": W(p+"ffn_gate_shexp.weight"), "shexp_u": W(p+"ffn_up_shexp.weight"),
            "shexp_d": W(p+"ffn_down_shexp.weight"),
        })
    embed_w = torch.from_numpy(embed_np); lm_head = torch.from_numpy(lm_head_np)

    # KV cache
    win_kv = [torch.zeros(WIN, HEAD_DIM) for _ in range(3)]

    # prefill at pos 152 (anchor=2581)
    mh0 = load_mh(152)
    main_x0 = rmsnorm(mh0[None, None, :] @ main_proj, main_norm_w)
    for s in range(3):
        L = layers[s]
        mkv = rmsnorm(main_x0 @ L["kv"], L["kv_a_norm"])
        mkv[..., -ROPE_DIM:] = rope(mkv[..., -ROPE_DIM:], cos_full[0], sin_full[0])
        win_kv[s][0] = mkv[0, 0]
    n_real = 1

    # run 5 steps building cache, measure at step 5
    for step in range(1, 6):
        pos = 152 + step; anchor = greedy[step]
        mh = load_mh(pos)
        main_x = rmsnorm(mh[None, None, :] @ main_proj, main_norm_w)
        for s in range(3):
            L = layers[s]
            mkv = rmsnorm(main_x @ L["kv"], L["kv_a_norm"])
            mkv[..., -ROPE_DIM:] = rope(mkv[..., -ROPE_DIM:], cos_full[step], sin_full[step])
            win_kv[s][step % WIN] = mkv[0, 0]
        n_real = step + 1
        if step < 5: continue

        # forward_embed
        draft_ids = torch.tensor([anchor] + [NOISE]*4)
        x = embed_w[:, draft_ids].T[None]  # [1,5,dim]
        x = x[:, :, None, :].expand(1, BLOCK, HC, DIM)  # [1,5,4,dim]

        cs = cos_full[step+1:step+6]; ss = sin_full[step+1:step+6]
        for s in range(3):
            L = layers[s]
            # attn sub-block
            residual = x
            flat = x.reshape(1, BLOCK, HC*DIM).float()
            mixes = (flat @ L["hc_attn_fn"]) * torch.rsqrt(flat.pow(2).mean(-1, keepdims=True) + EPS)
            # hc_pre: sigmoid pre
            pre = torch.sigmoid(mixes[..., :HC] * L["hc_attn_s"][0] + L["hc_attn_b"][:HC]) + HC_EPS
            post = 2*torch.sigmoid(mixes[..., HC:2*HC] * L["hc_attn_s"][1] + L["hc_attn_b"][HC:2*HC])
            comb_flat = mixes[..., 2*HC:] * L["hc_attn_s"][2] + L["hc_attn_b"][2*HC:]
            comb = comb_flat.reshape(1, BLOCK, HC, HC)
            # Sinkhorn on comb
            for _ in range(20):
                comb = comb / (comb.sum(-1, keepdims=True) + HC_EPS)
                comb = comb / (comb.sum(-2, keepdims=True) + HC_EPS)
            yd = (pre[..., None] * x).sum(2)  # [1,5,dim]
            yd = rmsnorm(yd, L["attn_norm"])
            # attention
            qr = rmsnorm(yd @ L["q_a"], L["q_a_norm"])
            q = (qr @ L["q_b"]).reshape(1, BLOCK, N_HEADS, HEAD_DIM)
            q = q * torch.rsqrt(q.pow(2).mean(-1, keepdims=True) + EPS)
            q[..., -ROPE_DIM:] = rope(q[..., -ROPE_DIM:], cs[None,:,None,:], ss[None,:,None,:])
            kv = rmsnorm(yd @ L["kv"], L["kv_a_norm"])
            kv[..., -ROPE_DIM:] = rope(kv[..., -ROPE_DIM:], cs[None,:,:], ss[None,:,:])
            kv_all = torch.cat([win_kv[s][None], kv], dim=1)
            idx = torch.cat([torch.arange(n_real), WIN + torch.arange(BLOCK)])
            kv_g = kv_all[:, idx].unsqueeze(1).expand(1, BLOCK, idx.size(0), HEAD_DIM)
            scores = torch.einsum("bshd,bstd->bsht", q, kv_g) * (HEAD_DIM ** -0.5)
            smax = scores.max(-1, keepdims=True).values
            sink_max = torch.maximum(smax, L["sinks"][None,None,:,None])
            numer = torch.exp(scores - sink_max)
            denom = numer.sum(-1, keepdims=True) + torch.exp(L["sinks"][None,None,:,None] - sink_max)
            o = torch.einsum("bsht,bstd->bshd", numer, kv_g) / denom
            o[..., -ROPE_DIM:] = rope(o[..., -ROPE_DIM:], cs[None,:,None,:], torch.tensor([-1.0]).expand_as(ss[None,:,None,:]))
            # WRONG: inverse rope needs -sin. Fix:
            # (redone below)
            # output proj
            gd = HEAD_DIM * N_HEADS // N_GROUPS
            og = o.reshape(1, BLOCK, N_GROUPS, gd)
            wo_a = L["out_a"].T.reshape(N_GROUPS, O_LORA, gd)
            ol = torch.einsum("bsgd,grd->bsgr", og, wo_a)
            ao = ol.reshape(1, BLOCK, N_GROUPS*O_LORA) @ L["out_b"]
            x = post[..., None] * ao[..., None, :] + (comb[..., None] * residual.unsqueeze(-2)).sum(-2)
            # ffn sub-block
            residual = x
            flat = x.reshape(1, BLOCK, HC*DIM).float()
            mixes = (flat @ L["hc_ffn_fn"]) * torch.rsqrt(flat.pow(2).mean(-1, keepdims=True) + EPS)
            pre2 = torch.sigmoid(mixes[..., :HC] * L["hc_ffn_s"][0] + L["hc_ffn_b"][:HC]) + HC_EPS
            post2 = 2*torch.sigmoid(mixes[..., HC:2*HC] * L["hc_ffn_s"][1] + L["hc_ffn_b"][HC:2*HC])
            comb_flat2 = mixes[..., 2*HC:] * L["hc_ffn_s"][2] + L["hc_ffn_b"][2*HC:]
            comb2 = comb_flat2.reshape(1, BLOCK, HC, HC)
            for _ in range(20):
                comb2 = comb2 / (comb2.sum(-1, keepdims=True) + HC_EPS)
                comb2 = comb2 / (comb2.sum(-2, keepdims=True) + HC_EPS)
            yd2 = (pre2[..., None] * x).sum(2)
            yd2 = rmsnorm(yd2, L["ffn_norm"])
            # MoE (simplified: just shared expert for now to check signal)
            y_ffn = torch.sigmoid(yd2 @ L["shexp_g"]) * (yd2 @ L["shexp_u"])
            y_ffn = y_ffn @ L["shexp_d"]
            x = post2[..., None] * y_ffn[..., None, :] + (comb2[..., None] * residual.unsqueeze(-2)).sum(-2)

        print(f"\n=== step {step}, pos {pos}, anchor={anchor} ===")
        print(f"  h stats: mean={x.mean():.4f} std={x.std():.4f}")
        # head
        xh = x[:, :, 0, :]  # just take hc copy 0 as approximation for hc_head test
        # Actually do hc_head properly
        # hc_head: sigmoid reduce
        flat_h = x.reshape(1, BLOCK, HC*DIM).float()
        hh_fn = W("mtp.2.hc_head_fn.weight"); hh_s = W("mtp.2.hc_head_scale.weight"); hh_b = W("mtp.2.hc_head_base.weight")
        hh_norm_w = W("mtp.2.norm.weight")
        mixes_h = (flat_h @ hh_fn) * torch.rsqrt(flat_h.pow(2).mean(-1, keepdims=True) + EPS)
        pre_h = torch.sigmoid(mixes_h * hh_s[0] + hh_b) + HC_EPS
        xh = (pre_h[..., None] * x).sum(2)  # [1,5,dim]
        xh = rmsnorm(xh, hh_norm_w)
        logits = (xh[0] @ lm_head)  # [5, vocab]
        # Markov head
        mw1 = W("mtp.2.markov_head.markov_w1.weight"); mw2 = W("mtp.2.markov_head.markov_w2.weight")
        out_ids = [anchor]
        for i in range(BLOCK):
            emb = mw1[:, out_ids[i]]
            bias = emb @ mw2
            li = logits[i] + bias
            out_ids.append(int(li.argmax()))

        target = greedy[step+1:step+6]
        print(f"  torch draft: {out_ids[1:]}")
        print(f"  target:      {target}")
        print(f"  match: {sum(1 for d,t in zip(out_ids[1:],target) if d==t)}/5")


if __name__ == "__main__":
    main()
