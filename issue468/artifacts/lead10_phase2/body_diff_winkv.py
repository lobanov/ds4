#!/usr/bin/env python3
"""Decisive bisection: feed the LIVE win_kv into the torch port's body forward.

Bypasses both (a) the capture mismatch (the H from a different run) and (b) the torch
port's own win_kv computation. If the torch body output then matches the live body
output, the body forward proper (attn/MoE/hc-gating) is faithful and the bug was in the
win_kv derivation. If it still diverges, the body forward proper has a bug.
"""
import sys, struct, numpy as np, torch
sys.path.insert(0, 'issue468/dspark_train')
from drafter_body import build_body, BLOCK, HC, DIM, HEAD_DIM, WIN, NOISE_TOK, rmsnorm, hc_pre, hc_post, apply_rotary, ROPE_DIM

DSPARK = '/Users/lobanov/Projects/ds4/gguf/dspark.gguf'
TARGET = '/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf'
WINKV_DUMP = '/tmp/live_winkv.bin'
BODY_DUMP = '/tmp/live_body3.bin'
dev = 'mps'
HC_DIM = HC * DIM
N_SWA = 128
KV_ROWS = N_SWA + BLOCK  # 133

# parse win_kv dump
wbuf = open(WINKV_DUMP, 'rb').read()
per_wrec = 12 + 3 * KV_ROWS * HEAD_DIM * 4
wrecs = []
off = 0
while off + per_wrec <= len(wbuf):
    a, p, n = struct.unpack_from('<iii', wbuf, off); off += 12
    wkv = np.frombuffer(wbuf, dtype='<f4', count=3*KV_ROWS*HEAD_DIM, offset=off).reshape(3, KV_ROWS, HEAD_DIM).copy()
    off += 3*KV_ROWS*HEAD_DIM*4
    wrecs.append((a, p, n, wkv))
print(f'parsed {len(wrecs)} win_kv records', flush=True)

# parse body dump
bbuf = open(BODY_DUMP, 'rb').read()
brecs = []
off = 0
while off < len(bbuf):
    if off + 12 > len(bbuf): break
    a, p, bs = struct.unpack_from('<iii', bbuf, off); off += 12
    body = np.frombuffer(bbuf, dtype='<f4', count=bs*HC_DIM, offset=off).reshape(bs, HC_DIM).copy(); off += bs*HC_DIM*4
    drafts = np.frombuffer(bbuf, dtype='<i4', count=bs, offset=off); off += bs*4
    brecs.append((a, p, bs, body, drafts))
print(f'parsed {len(brecs)} body records', flush=True)
assert len(wrecs) == len(brecs), f'{len(wrecs)} vs {len(brecs)}'

body = build_body(DSPARK, TARGET, dev, dtype=torch.float32)

def custom_forward(anchor, pos, n_real, live_wkv):
    """Run the 3-stage body forward with the LIVE win_kv. Returns [BLOCK, HC, DIM]."""
    draft_ids = torch.full((BLOCK,), NOISE_TOK, dtype=torch.long, device=dev); draft_ids[0] = int(anchor)
    x = body.embed_w[draft_ids][None, :, None, :].expand(1, BLOCK, HC, DIM).clone()  # [1, BLOCK, HC, DIM]
    for s in range(3):
        w = body.layers[s]
        # prepare the live win_kv for this stage: first n_real rows, padded to [WIN, HEAD_DIM]
        lwk = torch.zeros(WIN, HEAD_DIM, device=dev, dtype=torch.float32)
        nr = min(n_real, N_SWA)
        lwk[:nr] = torch.as_tensor(live_wkv[s][:nr], device=dev, dtype=torch.float32)
        # attention sub-layer
        res = x
        yd, post, comb = hc_pre(x, w["hc_attn_fn"], w["hc_attn_scale"], w["hc_attn_base"])
        x = hc_post(body._attn(rmsnorm(yd, w["attn_norm"]), lwk, nr, s, pos), res, post, comb)
        # ffn sub-layer
        res = x
        yd, post, comb = hc_pre(x, w["hc_ffn_fn"], w["hc_ffn_scale"], w["hc_ffn_base"])
        x = hc_post(body._moe(rmsnorm(yd, w["ffn_norm"]), s), res, post, comb)
    return x[0]  # [BLOCK, HC, DIM]

print(f'\n{"i":>3s} {"anchor":>7s} {"pos":>4s} {"n_real":>6s} {"max|d|":>10s} {"mean|d|":>10s} {"rel|d|":>8s}', flush=True)
sum_d = 0.0; sum_abs = 0.0; n_tot = 0; max_d = 0.0; n_match = 0
with torch.no_grad():
    for i, ((a, p, n, wkv), (ba, bp, bs, live_body, drafts)) in enumerate(zip(wrecs, brecs)):
        assert a == ba and p == bp, f'record mismatch {a}/{ba} {p}/{bp}'
        torch_body = custom_forward(a, p, n, wkv).cpu().numpy().reshape(BLOCK, HC_DIM)[:bs]
        live_body_cmp = live_body[:bs]
        d = np.abs(torch_body - live_body_cmp)
        this_max = float(d.max()); this_mean = float(d.mean())
        sum_d += d.sum(); sum_abs += np.abs(live_body_cmp).sum(); n_tot += live_body_cmp.size
        max_d = max(max_d, this_max); n_match += 1
        if i < 12 or i % 10 == 0:
            print(f'{i:3d} {a:7d} {p:4d} {n:6d} {this_max:10.4f} {this_mean:10.5f} {d.sum()/max(np.abs(live_body).sum(),1e-9):8.5f}', flush=True)
print(f'\noverall: n_match={n_match} max|d|={max_d:.4f} mean|d|={sum_d/max(n_tot,1):.5f} rel|d|={sum_d/max(sum_abs,1e-9):.5f}', flush=True)
print('(rel|d| near 0 => body forward proper is faithful; large => body bug)', flush=True)
