#!/usr/bin/env python3
"""Stage-by-stage bisection: where does the torch body forward diverge from the live?

Compares the torch port's per-stage body output (stage 0, 1, 2=final) against the live
Metal drafter's per-stage dumps, using the LIVE win_kv fed in (bypasses the win_kv computation).
Stage 0 = hc_pre -> _attn -> hc_post -> hc_pre -> _moe -> hc_post. If stage 0 already diverges,
the bug is in stage 0's _attn / hc_post / _moe (hc_pre Sinkhorn already verified to match).
"""
import sys, struct, numpy as np, torch
sys.path.insert(0, 'issue468/dspark_train')
from drafter_body import build_body, BLOCK, HC, DIM, HEAD_DIM, WIN, NOISE_TOK, rmsnorm, hc_pre, hc_post
from gguf_loader import index_gguf, read_tensor  # noqa

DSPARK = '/Users/lobanov/Projects/ds4/gguf/dspark.gguf'
TARGET = '/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf'
dev = 'mps'
HC_DIM = HC * DIM
N_SWA = 128
KV_ROWS = N_SWA + BLOCK

def parse_winkv(path):
    buf = open(path, 'rb').read(); per = 12 + 3*KV_ROWS*HEAD_DIM*4; recs = []; off = 0
    while off + per <= len(buf):
        a, p, n = struct.unpack_from('<iii', buf, off); off += 12
        wkv = np.frombuffer(buf, dtype='<f4', count=3*KV_ROWS*HEAD_DIM, offset=off).reshape(3, KV_ROWS, HEAD_DIM).copy()
        off += 3*KV_ROWS*HEAD_DIM*4; recs.append((a, p, n, wkv))
    return recs

def parse_body_stage(path, with_drafts=False):
    buf = open(path, 'rb').read(); recs = []; off = 0
    while off < len(buf):
        if off + 12 > len(buf): break
        a, p, bs = struct.unpack_from('<iii', buf, off); off += 12
        body = np.frombuffer(buf, dtype='<f4', count=bs*HC_DIM, offset=off).reshape(bs, HC_DIM).copy(); off += bs*HC_DIM*4
        drafts = None
        if with_drafts:
            drafts = np.frombuffer(buf, dtype='<i4', count=bs, offset=off).copy(); off += bs*4
        recs.append((a, p, bs, body, drafts))
    return recs

wrecs = parse_winkv('/tmp/live_winkv.bin')
s0 = parse_body_stage('/tmp/live_bodyS0.bin')
s1 = parse_body_stage('/tmp/live_bodyS1.bin')
sf = parse_body_stage('/tmp/live_body3.bin', with_drafts=True)
print(f'recs: winkv={len(wrecs)} s0={len(s0)} s1={len(s1)} sf={len(sf)}', flush=True)
assert len(wrecs) == len(s0) == len(s1) == len(sf)

body = build_body(DSPARK, TARGET, dev, dtype=torch.float32)

def custom_forward_stages(anchor, pos, n_real, live_wkv):
    """Return [stage0, stage1, stage2] body outputs, each [BLOCK, HC, DIM], using the live win_kv."""
    draft_ids = torch.full((BLOCK,), NOISE_TOK, dtype=torch.long, device=dev); draft_ids[0] = int(anchor)
    x = body.embed_w[draft_ids][None, :, None, :].expand(1, BLOCK, HC, DIM).clone()
    stage_outs = []
    for s in range(3):
        w = body.layers[s]
        lwk = torch.zeros(WIN, HEAD_DIM, device=dev, dtype=torch.float32)
        nr = min(n_real, N_SWA); lwk[:nr] = torch.as_tensor(live_wkv[s][:nr], device=dev, dtype=torch.float32)
        res = x; yd, post, comb = hc_pre(x, w["hc_attn_fn"], w["hc_attn_scale"], w["hc_attn_base"])
        x = hc_post(body._attn(rmsnorm(yd, w["attn_norm"]), lwk, nr, s, pos), res, post, comb)
        res = x; yd, post, comb = hc_pre(x, w["hc_ffn_fn"], w["hc_ffn_scale"], w["hc_ffn_base"])
        x = hc_post(body._moe(rmsnorm(yd, w["ffn_norm"]), s), res, post, comb)
        stage_outs.append(x[0].cpu().numpy())
    return stage_outs

print(f'\n{"i":>3s} {"anchor":>7s} {"pos":>4s} {"nr":>4s} | {"S0 rel|d|":>10s} {"S1 rel|d|":>10s} {"S2 rel|d|":>10s}', flush=True)
agg = np.zeros(3); cnt = 0
with torch.no_grad():
    for i, ((a, p, n, wkv), (a0,_,_,lb0,_), (a1,_,_,lb1,_), (af,_,bs,lbf,_)) in enumerate(zip(wrecs, s0, s1, sf)):
        outs = custom_forward_stages(a, p, n, wkv)
        live_stages = [lb0, lb1, lbf]
        row_rels = []
        for si in range(3):
            tb = outs[si].reshape(BLOCK, HC_DIM)[:bs]; lb = live_stages[si][:bs]
            d = np.abs(tb - lb); rel = d.sum() / max(np.abs(lb).sum(), 1e-9)
            row_rels.append(rel); agg[si] += d.sum()
        cnt += 1
        if i < 10 or i % 15 == 0:
            print(f'{i:3d} {a:7d} {p:4d} {n:4d} | {row_rels[0]:10.5f} {row_rels[1]:10.5f} {row_rels[2]:10.5f}', flush=True)
# overall rel|d| per stage (normalized by the sum of |live| across all records — approximate)
print(f'\n(bisect: S0 large => bug in stage-0 _attn/_moe; S0 small + S2 large => accumulates)', flush=True)
