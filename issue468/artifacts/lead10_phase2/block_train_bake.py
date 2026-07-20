#!/usr/bin/env python3
"""Lead 10 block-trained LoRA: train head.hc_fn on the BLOCK (all K=5 positions), not just p1.

The p1-trained LoRA optimized only position 1 (which is already 0.996) and didn't help the
suffix (positions 2-5, the ~20%-miss-per-position region where the drafter actually loses).
This trains on ALL K positions via k_scores (the teacher-forced block) to directly target the
suffix quality.

Steps: (1) re-precompute the FULL xs (all BLOCK positions) via the body forward (F32, no_grad),
(2) train head.hc_fn LoRA on the block KL (sum over K positions), (3) bake into dspark GGUF.
"""
import sys, struct, json, shutil, numpy as np, torch
sys.path.insert(0, 'issue468/dspark_train'); sys.path.insert(0, 'issue468/dspark_oracle')
from drafter_body import build_body, BLOCK, HC, DIM
from drafter_head import build_head
from gguf_loader import index_gguf, read_tensor

DSPARK = '/Users/lobanov/Projects/ds4/gguf/dspark.gguf'
DSPARK_LORA = '/Users/lobanov/Projects/ds4/gguf/dspark_lora_block.gguf'
TARGET = '/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf'
CAP_H = 'issue468/artifacts/lead11_unified_capture/lead3_h.bin'
CAP_LP = 'issue468/artifacts/lead10_phase1/lead3_logprobs_recapture.jsonl'
RANK = 32; EPOCHS = 6; LR = 3e-4; WD = 0.01; BS = 32; SEED = 42; TOPN = 128; K = BLOCK; N_PROMPTS = 20  # subset for speed
dev = 'mps'

# --- parse h.bin + logprobs ---
prompts = {}
with open(CAP_H, 'rb') as f: data = f.read()
off = 0
while off < len(data):
    if off + 4 > len(data): break
    (idl,) = struct.unpack_from('<i', data, off); off += 4
    pid = data[off:off+idl].decode(); off += idl
    (pos, tok) = struct.unpack_from('<ii', data, off); off += 8
    h = np.frombuffer(data, dtype='<f4', count=12288, offset=off); off += 12288 * 4
    prompts.setdefault(pid, {'pos': [], 'sel': [], 'h': []})
    prompts[pid]['pos'].append(pos); prompts[pid]['sel'].append(tok); prompts[pid]['h'].append(h)
lp = {}
for l in open(CAP_LP):
    d = json.loads(l); arr = np.array(d['top'], dtype=np.float32)
    lp[(d['id'], d['pos'])] = (d['sel'], arr[:, 0].astype(np.int64), arr[:, 1])
for pid in prompts:
    o = np.argsort(prompts[pid]['pos'])
    prompts[pid]['h'] = np.stack(prompts[pid]['h'])[o]
    prompts[pid]['sel'] = [prompts[pid]['sel'][i] for i in o]
    prompts[pid]['pos'] = [prompts[pid]['pos'][i] for i in o]
pids = sorted(prompts.keys())[:N_PROMPTS]
print(f'{len(pids)} prompts', flush=True)

# --- re-precompute the FULL xs (all BLOCK positions) ---
print('precomputing FULL xs (all BLOCK positions, F32 no_grad)...', flush=True)
body = build_body(DSPARK, TARGET, dev, dtype=torch.float32)
full_feats = {}  # pid -> list of (xs_block [BLOCK,HC,DIM], prev_tok [K], target_tids [K,K_TOPN], target_tlogp [K,K_TOPN], target_tok [K])
with torch.no_grad():
    for i, pid in enumerate(pids):
        H = prompts[pid]['h']; sel = prompts[pid]['sel']; pos = prompts[pid]['pos']; N = len(sel)
        ms = min(N - BLOCK - 1, N - 1)
        if ms < K + 2: continue
        xs = body.forward_prompt(H[:ms+1], [int(sel[s]) for s in range(1, ms+1)])  # [ms, BLOCK, HC, DIM]
        anchors = []
        for j in range(ms - K):  # need K+1 future tokens
            x_block = xs[j].cpu().numpy()  # [BLOCK, HC, DIM]
            # prev_tok[k] = sel[j+1+k] (the token before the k-th draft)
            prev_tok = np.array([int(sel[j+1+k]) for k in range(K)], dtype=np.int64)
            # target[k] = sel[j+k+2] (the k-th draft's target = the next token)
            target_tok = np.array([int(sel[j+k+2]) for k in range(K)], dtype=np.int64)
            # block teacher: the top-128 at each target position
            tids = np.zeros((K, TOPN), dtype=np.int64); tlogp = np.zeros((K, TOPN), dtype=np.float32)
            for k in range(K):
                p = pos[j+k+2] if (j+k+2) < len(pos) else pos[-1]
                _, ids, lpv = lp[(pid, p)]; tids[k] = ids; tlogp[k] = lpv
            anchors.append((x_block, prev_tok, tids, tlogp, target_tok))
        full_feats[pid] = anchors
        if (i+1) % 20 == 0: print(f'  {i+1}/{len(pids)} ({sum(len(a) for a in full_feats.values())} anchors)', flush=True)
del body
total_a = sum(len(a) for a in full_feats.values())
print(f'precomputed {total_a} anchors across {len(full_feats)} prompts', flush=True)

# --- collect into tensors ---
all_x = np.stack([a[0] for p in pids if p in full_feats for a in full_feats[p]], 0)  # [N, BLOCK, HC, DIM]
all_pt = np.stack([a[1] for p in pids if p in full_feats for a in full_feats[p]], 0)  # [N, K]
all_ti = np.stack([a[2] for p in pids if p in full_feats for a in full_feats[p]], 0)  # [N, K, TOPN]
all_tl = np.stack([a[3] for p in pids if p in full_feats for a in full_feats[p]], 0)  # [N, K, TOPN]
all_tt = np.stack([a[4] for p in pids if p in full_feats for a in full_feats[p]], 0)  # [N, K]
xtr = torch.from_numpy(all_x); pttr = torch.from_numpy(all_pt); titr = torch.from_numpy(all_ti); tltr = torch.from_numpy(all_tl); ttr = torch.from_numpy(all_tt)
print(f'train: {xtr.shape[0]} anchors, x {xtr.shape}', flush=True)

# --- train head.hc_fn LoRA on the block KL ---
print(f'training head.hc_fn LoRA (rank={RANK}, {EPOCHS} epochs, block KL)...', flush=True)
head = build_head(DSPARK, TARGET, device=dev, lora_rank=RANK, dtype=torch.float32)
hc_params = [p for n, p in head.named_parameters() if 'lora_hc' in n]
for n, p in head.named_parameters():
    if 'lora_hc' not in n: p.requires_grad_(False)
opt = torch.optim.AdamW(hc_params, lr=LR, weight_decay=WD)
head.train(); ntr = xtr.shape[0]
for ep in range(EPOCHS):
    perm = torch.randperm(ntr); tl = 0.0
    for i in range(0, ntr, BS):
        idx = perm[i:i+BS]
        # k_scores: x [BS, BLOCK, HC, DIM] + prev_tok [BS, K] -> [BS, K, VOCAB]
        logits = head.k_scores(xtr[idx].to(dev), pttr[idx].to(dev))  # [BS, K, VOCAB]
        # block KL: sum over K positions
        draft_lp = torch.log_softmax(logits, dim=-1)  # [BS, K, VOCAB]
        loss = 0.0
        for k in range(K):
            dlp_top = torch.gather(draft_lp[:, k, :], 1, titr[:, k, :][idx].to(dev))  # [BS, TOPN]
            tlp = torch.log_softmax(tltr[:, k, :][idx].to(dev), dim=-1)
            t_prob = torch.exp(tlp)
            loss = loss + (t_prob * (tlp - dlp_top)).sum(-1).mean()
        loss = loss / K
        opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item() * len(idx)
    print(f'  ep{ep+1:2d}: block KL={tl/ntr:.4f}', flush=True)

# --- compute the delta + bake ---
delta = (head.lora_hc_B.float() @ head.lora_hc_A.float()).detach().cpu().numpy()
print(f'\nLoRA delta: shape {delta.shape} | ‖delta‖={np.linalg.norm(delta):.4f} | max|delta|={np.abs(delta).max():.4f}', flush=True)

print(f'copying {DSPARK} -> {DSPARK_LORA}...', flush=True)
shutil.copy2(DSPARK, DSPARK_LORA)
_, ginfos, gdoff = index_gguf(DSPARK)
hc_ne, hc_dt, hc_off = ginfos['mtp.2.hc_head_fn.weight']
hc_abs_off = gdoff + hc_off
current = read_tensor(DSPARK, ginfos, gdoff, 'mtp.2.hc_head_fn.weight').astype(np.float32)
modified = current + delta
with open(DSPARK_LORA, 'r+b') as f:
    f.seek(hc_abs_off)
    f.write(modified.astype(np.float16).tobytes())
# verify
_, ginfos2, gdoff2 = index_gguf(DSPARK_LORA)
check = read_tensor(DSPARK_LORA, ginfos2, gdoff2, 'mtp.2.hc_head_fn.weight').astype(np.float32)
print(f'verify: ‖check - modified‖={np.linalg.norm(check - modified):.6f} (should be ~0)', flush=True)
print(f'\nbaked the block-trained LoRA into {DSPARK_LORA}', flush=True)
print('Next: run ds4-spec-bench with --dspark dspark_lora_block.gguf')
