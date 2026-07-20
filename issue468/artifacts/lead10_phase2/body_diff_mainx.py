#!/usr/bin/env python3
"""Test the main_x hypothesis: feed win_kv[0..n_real] (win_kv + main_x at slot n_real)
to the torch port's body forward, using the POST-attention kv_cache dump (fresh)."""
import sys, struct, numpy as np, torch
sys.path.insert(0, 'issue468/dspark_train')
from drafter_body import build_body, BLOCK, HC, DIM, HEAD_DIM, WIN, NOISE_TOK, rmsnorm, hc_pre, hc_post

DSPARK = '/Users/lobanov/Projects/ds4/gguf/dspark.gguf'
TARGET = '/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf'
dev = 'mps'; HC_DIM = HC*DIM; N_SWA=128; KV_ROWS=N_SWA+BLOCK

def parse_winkv(path):
    buf = open(path,'rb').read(); per=12+3*KV_ROWS*HEAD_DIM*4; recs=[]; off=0
    while off+per<=len(buf):
        a,p,n=struct.unpack_from('<iii',buf,off); off+=12
        wkv=np.frombuffer(buf,dtype='<f4',count=3*KV_ROWS*HEAD_DIM,offset=off).reshape(3,KV_ROWS,HEAD_DIM).copy()
        off+=3*KV_ROWS*HEAD_DIM*4; recs.append((a,p,n,wkv))
    return recs
def parse_body(path, with_drafts=False):
    buf=open(path,'rb').read(); recs=[]; off=0
    while off<len(buf):
        if off+12>len(buf): break
        a,p,bs=struct.unpack_from('<iii',buf,off); off+=12
        body=np.frombuffer(buf,dtype='<f4',count=bs*HC_DIM,offset=off).reshape(bs,HC_DIM).copy(); off+=bs*HC_DIM*4
        drafts=None
        if with_drafts:
            drafts=np.frombuffer(buf,dtype='<i4',count=bs,offset=off).copy(); off+=bs*4
        recs.append((a,p,bs,body,drafts))
    return recs

wrecs = parse_winkv('/tmp/live_winkv_final.bin')   # POST-LOOP (all 3 stages' main_x valid)
brecs = parse_body('/tmp/live_body3.bin', with_drafts=True)
print(f'recs: winkv_post={len(wrecs)} body={len(brecs)}', flush=True)
body = build_body(DSPARK, TARGET, dev, dtype=torch.float32)
from drafter_head import build_head
head = build_head(DSPARK, TARGET, dev, lora_rank=0, dtype=torch.float32)
head.eval()

def custom_forward(anchor, pos, n_real, live_wkv):
    draft_ids = torch.full((BLOCK,), NOISE_TOK, dtype=torch.long, device=dev); draft_ids[0]=int(anchor)
    x = body.embed_w[draft_ids][None,:,None,:].expand(1,BLOCK,HC,DIM).clone()
    for s in range(3):
        w = body.layers[s]
        lwk = torch.zeros(WIN, HEAD_DIM, device=dev, dtype=torch.float32)
        nr = min(n_real, N_SWA)
        # KEY CHANGE: include slot n_real (the main_x) -> nr+1 context tokens
        lwk[:nr+1] = torch.as_tensor(live_wkv[s][:nr+1], device=dev, dtype=torch.float32)
        res=x; yd,post,comb=hc_pre(x,w["hc_attn_fn"],w["hc_attn_scale"],w["hc_attn_base"])
        x = hc_post(body._attn(rmsnorm(yd,w["attn_norm"]), lwk, nr+1, s, pos), res, post, comb)
        res=x; yd,post,comb=hc_pre(x,w["hc_ffn_fn"],w["hc_ffn_scale"],w["hc_ffn_base"])
        x = hc_post(body._moe(rmsnorm(yd,w["ffn_norm"]), s), res, post, comb)
    return x[0]

print(f'\n=== WITH main_x (slots 0..n_real) ===')
print(f'{"i":>3s} {"anchor":>7s} {"pos":>4s} {"nr":>4s} {"max|d|":>10s} {"mean|d|":>10s} {"rel|d|":>8s}', flush=True)
sum_d=0.0; sum_abs=0.0; n_tot=0; max_d=0.0
with torch.no_grad():
    for i,((a,p,n,wkv),(ba,bp,bs,lb,ld)) in enumerate(zip(wrecs,brecs)):
        tb = custom_forward(a,p,n,wkv).cpu().numpy().reshape(BLOCK,HC_DIM)[:bs]
        d = np.abs(tb-lb); this_max=float(d.max()); this_mean=float(d.mean())
        sum_d+=d.sum(); sum_abs+=np.abs(lb).sum(); n_tot+=lb.size; max_d=max(max_d,this_max)
        if i<10 or i%15==0:
            print(f'{i:3d} {a:7d} {p:4d} {n:4d} {this_max:10.4f} {this_mean:10.5f} {d.sum()/max(np.abs(lb).sum(),1e-9):8.5f}', flush=True)
print(f'\noverall WITH main_x: max|d|={max_d:.4f} mean|d|={sum_d/max(n_tot,1):.5f} rel|d|={sum_d/max(sum_abs,1e-9):.5f}', flush=True)
print('(was rel|d|=0.60 WITHOUT main_x; if this drops near 0, main_x was the bug)', flush=True)

# DRAFT AGREEMENT: do the torch body outputs produce the same drafts as the live?
print(f'\n=== draft agreement (torch drafts from body_diff_mainx body vs live drafts) ===', flush=True)
per_pos_agree = np.zeros(BLOCK, dtype=np.int64); per_pos_tot = 0
with torch.no_grad():
    for (a,p,n,wkv),(ba,bp,bs,lb,live_drafts) in zip(wrecs,brecs):
        if live_drafts is None: continue
        tb = custom_forward(a,p,n,wkv)  # [BLOCK, HC, DIM]
        out,_ = head.forward(tb.unsqueeze(0).to(dev), torch.tensor([a]))
        torch_drafts = out[0,1:].cpu().numpy()
        k = min(len(torch_drafts), len(live_drafts))
        for pos in range(k):
            if torch_drafts[pos]==live_drafts[pos]: per_pos_agree[pos]+=1
        per_pos_tot += 1
print('per-position draft agreement (torch vs live):')
for pos in range(BLOCK):
    print(f'  pos {pos+1}: {per_pos_agree[pos]}/{per_pos_tot} = {per_pos_agree[pos]/max(per_pos_tot,1):.3f}')
print(f'  (the live ACCEPTANCE is [0.996, 0.835, 0.673, 0.516, 0.390]; draft-vs-live agreement near 1.0 => faithful)', flush=True)
