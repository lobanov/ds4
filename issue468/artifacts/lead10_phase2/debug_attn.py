#!/usr/bin/env python3
"""Debug the torch port's _attn for rec0 stage 0: print magnitudes at each step to find the gap."""
import sys, numpy as np, torch
sys.path.insert(0, 'issue468/dspark_train')
from drafter_body import build_body, BLOCK, HC, DIM, HEAD_DIM, ROPE_DIM, N_HEADS, N_GROUPS, O_LORA, WIN, NOISE_TOK, rmsnorm, hc_pre, hc_post, apply_rotary, sparse_attn

DSPARK = '/Users/lobanov/Projects/ds4/gguf/dspark.gguf'
TARGET = '/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf'
dev = 'cpu'  # cpu for easy introspection
body = build_body(DSPARK, TARGET, dev, dtype=torch.float32)
s = 0; w = body.layers[s]
pos = 37

# draft block input (rec0: anchor 9544)
draft_ids = torch.full((BLOCK,), NOISE_TOK, dtype=torch.long); draft_ids[0] = 9544
x = body.embed_w[draft_ids][None, :, None, :].expand(1, BLOCK, HC, DIM).clone()
# hc_pre (attention)
yd, post, comb = hc_pre(x, w["hc_attn_fn"], w["hc_attn_scale"], w["hc_attn_base"])
x_draft = rmsnorm(yd, w["attn_norm"])
print(f'hc_pre out (yd): mean|v|={yd.abs().mean():.4f} max={yd.abs().max():.4f}')
print(f'attn_norm input: mean|v|={x_draft.abs().mean():.4f} max={x_draft.abs().max():.4f}')
print(f'post: {post.flatten()[:8].numpy()}')
print(f'comb diag: {comb[0,0].diagonal().numpy()}')

# q projection
qr = rmsnorm(x_draft @ w["q_a"].T, w["q_a_norm"])
print(f'\nqr (q_a out): mean|v|={qr.abs().mean():.4f} max={qr.abs().max():.4f}')
q = (qr @ w["q_b"].T).reshape(1, BLOCK, N_HEADS, HEAD_DIM)
print(f'q (q_b out, pre-norm): mean|v|={q.abs().mean():.4f} max={q.abs().max():.4f}')
q = q * torch.rsqrt((q*q).mean(-1, keepdim=True) + 1e-6)
cs = body.cos[pos+1:pos+1+BLOCK][None,:,None,:]; ss = body.sin[pos+1:pos+1+BLOCK][None,:,None,:]
q[...,-ROPE_DIM:] = apply_rotary(q[...,-ROPE_DIM:], cs, ss)
print(f'q (after head_rms_norm + rope): mean|v|={q.abs().mean():.4f} max={q.abs().max():.4f}')

# kv projection
kv = rmsnorm(x_draft @ w["kv"].T, w["kv_a_norm"])
print(f'\nkv (pre-rope): mean|v|={kv.abs().mean():.4f} max={kv.abs().max():.4f} shape={list(kv.shape)}')
cs_kv = body.cos[pos+1:pos+1+BLOCK][None,:,:]; ss_kv = body.sin[pos+1:pos+1+BLOCK][None,:,:]
kv[...,-ROPE_DIM:] = apply_rotary(kv[...,-ROPE_DIM:], cs_kv, ss_kv)
print(f'kv (after rope): mean|v|={kv.abs().mean():.4f} max={kv.abs().max():.4f}')

# attention (rec0: empty win_kv, n_real=0)
n_real = 0
win_kv = torch.zeros(WIN, HEAD_DIM)
kv_all = torch.cat([win_kv.unsqueeze(0), kv], dim=1)
idx = torch.cat([torch.arange(n_real), torch.arange(WIN, WIN+BLOCK)])
kv_g = kv_all[:, idx, :].unsqueeze(1).expand(1, BLOCK, idx.size(0), HEAD_DIM)
print(f'\nkv_g (attended keys): mean|v|={kv_g.abs().mean():.4f} max={kv_g.abs().max():.4f}')
scale = HEAD_DIM ** -0.5
o = sparse_attn(q, kv_g, w["attn_sinks"], scale)
print(f'attn sinks: {w["attn_sinks"].numpy()}')
print(f'attention output o (pre-rope): mean|v|={o.abs().mean():.4f} max={o.abs().max():.4f}')
o[...,-ROPE_DIM:] = apply_rotary(o[...,-ROPE_DIM:], cs, -ss)
print(f'o (after inverse rope): mean|v|={o.abs().mean():.4f} max={o.abs().max():.4f}')

# output projection
gd = HEAD_DIM * N_HEADS // N_GROUPS
wo_a = w["output_a"].reshape(N_GROUPS, O_LORA, gd)
print(f'\noutput_a: mean|v|={wo_a.abs().mean():.4f} max={wo_a.abs().max():.4f} shape={list(wo_a.shape)}')
print(f'output_b: mean|v|={w["output_b"].abs().mean():.4f} max={w["output_b"].abs().max():.4f} shape={list(w["output_b"].shape)}')
o_lor = torch.einsum("bsgd,grd->bsgr", o.reshape(1, BLOCK, N_GROUPS, gd), wo_a)
print(f'o_lor (after output_a): mean|v|={o_lor.abs().mean():.4f} max={o_lor.abs().max():.4f}')
attn_out = (o_lor.reshape(1, BLOCK, N_GROUPS * O_LORA) @ w["output_b"].T)
print(f'attn_out (after output_b): mean|v|={attn_out.abs().mean():.4f} max={attn_out.abs().max():.4f}')

# hc_post_attn
res = x
post_attn = hc_post(attn_out, res, post, comb)
print(f'\nPOST-ATTENTION (hc_post): mean|v|={post_attn.abs().mean():.4f} max={post_attn.abs().max():.4f}')
print(f'(live rec0 post-attn was mean|v|=986.5 max=65961)')
