#!/usr/bin/env python3
"""Debug rec0: print torch vs live magnitudes per stage to understand the divergence nature."""
import sys, struct, numpy as np, torch
sys.path.insert(0, 'issue468/dspark_train')
from drafter_body import build_body, BLOCK, HC, DIM, HEAD_DIM, WIN, NOISE_TOK, rmsnorm, hc_pre, hc_post

DSPARK = '/Users/lobanov/Projects/ds4/gguf/dspark.gguf'
TARGET = '/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf'
dev = 'mps'; HC_DIM = HC*DIM; N_SWA=128; KV_ROWS=N_SWA+BLOCK

def parse_body_stage(path, with_drafts=False):
    buf=open(path,'rb').read(); recs=[]; off=0
    while off<len(buf):
        if off+12>len(buf): break
        a,p,bs=struct.unpack_from('<iii',buf,off); off+=12
        body=np.frombuffer(buf,dtype='<f4',count=bs*HC_DIM,offset=off).reshape(bs,HC_DIM).copy(); off+=bs*HC_DIM*4
        if with_drafts: off+=bs*4
        recs.append((a,p,bs,body))
    return recs

s0=parse_body_stage('/tmp/live_bodyS0.bin')[0][3]  # rec0 stage-0 live [5, HC_DIM]
s1=parse_body_stage('/tmp/live_bodyS1.bin')[0][3]
sf=parse_body_stage('/tmp/live_body3.bin', with_drafts=True)[0][3]
print('LIVE rec0 magnitudes (mean|max|std):')
for nm, lb in [('S0',s0),('S1',s1),('S2',sf)]:
    print(f'  live {nm}: mean|v|={np.abs(lb).mean():.4f} max|v|={np.abs(lb).max():.4f} std={lb.std():.4f}')

body = build_body(DSPARK, TARGET, dev, dtype=torch.float32)
draft_ids = torch.full((BLOCK,), NOISE_TOK, dtype=torch.long, device=dev); draft_ids[0]=9544
x = body.embed_w[draft_ids][None, :, None, :].expand(1, BLOCK, HC, DIM).clone()
print(f'\nDRAFT BLOCK INPUT (embed): mean|v|={x.abs().mean().item():.4f} max|v|={x.abs().max().item():.4f}')
# also print embed_w[9544] vs embed_w[NOISE]
print(f'  embed_w[9544]: first5={body.embed_w[9544][:5].cpu().numpy()}')
print(f'  embed_w[NOISE={NOISE_TOK}]: first5={body.embed_w[NOISE_TOK][:5].cpu().numpy()}')

stage_outs=[]
with torch.no_grad():
    for s in range(3):
        w=body.layers[s]
        lwk=torch.zeros(WIN,HEAD_DIM,device=dev,dtype=torch.float32)  # rec0: empty win_kv
        res=x; yd,post,comb=hc_pre(x,w["hc_attn_fn"],w["hc_attn_scale"],w["hc_attn_base"])
        x_post_attn = hc_post(body._attn(rmsnorm(yd,w["attn_norm"]),lwk,0,s,37),res,post,comb)
        res=x_post_attn; yd,post,comb=hc_pre(x_post_attn,w["hc_ffn_fn"],w["hc_ffn_scale"],w["hc_ffn_base"])
        x = hc_post(body._moe(rmsnorm(yd,w["ffn_norm"]),s),res,post,comb)
        stage_outs.append((x_post_attn[0].cpu().numpy(), x[0].cpu().numpy()))
        print(f'\nTORCH stage {s}: post-attn mean|v|={np.abs(stage_outs[-1][0]).mean():.4f} max={np.abs(stage_outs[-1][0]).max():.4f} | post-moe mean|v|={np.abs(stage_outs[-1][1]).mean():.4f} max={np.abs(stage_outs[-1][1]).max():.4f}')

print('\n=== rec0 stage-0 (post-moe) comparison: torch vs live, first 20 elements of pos0 ===')
t0 = stage_outs[0][1].reshape(BLOCK, HC_DIM)[0]  # [HC_DIM]
print('torch S0 pos0[:20]:', np.array2string(t0[:20], precision=3, max_line_width=120))
print('live  S0 pos0[:20]:', np.array2string(s0[0][:20], precision=3, max_line_width=120))
print(f'torch S0 pos0: mean={t0.mean():.4f} std={t0.std():.4f}')
print(f'live  S0 pos0: mean={s0[0].mean():.4f} std={s0[0].std():.4f}')
# correlation
corr = np.corrcoef(t0, s0[0])[0,1]
print(f'correlation(torch S0 pos0, live S0 pos0) = {corr:.4f}')
