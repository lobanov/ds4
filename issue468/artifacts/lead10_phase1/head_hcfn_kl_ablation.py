#!/usr/bin/env python3
"""Lead 10 Phase-1 (codex gate A experiment #2): head.hc_fn-only LoRA, KL vs the IQ2 top-128.

Confirms the hard-label ablation (+2.41 pp) holds under the goal's specified loss (KL vs the
IQ2 top-128). Same setup as head_hcfn_ablation.py except the loss: KL(teacher_top128 ‖ draft)
instead of CE vs sel. Eval metric unchanged (held-out p1, first-draft argmax vs sel) so the
result is directly comparable to the +2.41 pp CE result.
"""
import sys, struct, json, numpy as np, torch
sys.path.insert(0, 'issue468/dspark_train'); sys.path.insert(0, 'issue468/dspark_oracle')
from drafter_body import build_body, BLOCK, HC, DIM
from drafter_head import build_head

DSPARK = '/Users/lobanov/Projects/ds4/gguf/dspark.gguf'
TARGET = '/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf'
CAP_H = 'issue468/artifacts/lead11_unified_capture/lead3_h.bin'
CAP_LP = 'issue468/artifacts/lead10_phase1/lead3_logprobs_recapture.jsonl'
RANK = 32; EPOCHS = 12; LR = 3e-4; WD = 0.01; BS = 256; SEED = 42; TOPN = 128

# --- parse h.bin -> per-prompt (H, sel); parse logprobs -> per (id,pos) (sel, top) ---
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
lp = {}  # (id,pos) -> (sel, top_ids[NP32], top_logp[NP32])
for l in open(CAP_LP):
    d = json.loads(l); arr = np.array(d['top'], dtype=np.float32)
    lp[(d['id'], d['pos'])] = (d['sel'], arr[:, 0].astype(np.int64), arr[:, 1])
# alignment check: logprobs sel == h.bin tok
n_mis = 0
for pid in prompts:
    o = np.argsort(prompts[pid]['pos'])
    prompts[pid]['h'] = np.stack(prompts[pid]['h'])[o]
    prompts[pid]['sel'] = [prompts[pid]['sel'][i] for i in o]
    prompts[pid]['pos'] = [prompts[pid]['pos'][i] for i in o]
    for p, s in zip(prompts[pid]['pos'], prompts[pid]['sel']):
        if (pid, p) in lp and lp[(pid, p)][0] != s: n_mis += 1
pids = sorted(prompts.keys())
print(f'{len(pids)} prompts; logprobs sel vs h.bin tok mismatches: {n_mis}', flush=True)

# --- precompute body features + the KL target (top-128 at pos i+2) per anchor ---
dev = 'mps'
print('loading drafter body F32 (no_grad)...', flush=True)
body = build_body(DSPARK, TARGET, dev, dtype=torch.float32)
feats = {}
with torch.no_grad():
    for i, pid in enumerate(pids):
        H = prompts[pid]['h']; sel = prompts[pid]['sel']; pos = prompts[pid]['pos']; N = len(sel)
        ms = min(N - BLOCK - 1, N - 1)
        if ms < 1: continue
        xs = body.forward_prompt(H[:ms+1], [int(sel[s]) for s in range(1, ms+1)])
        x0 = xs[:, 0, :, :].float().cpu().numpy()
        anchor = np.array([int(sel[s]) for s in range(1, ms+1)], dtype=np.int64)
        # KL target = the top-128 at pos (i+2) for i=0..ms-1; the greedy eval target = sel[i+2]
        tids = np.zeros((ms, TOPN), dtype=np.int64); tlogp = np.zeros((ms, TOPN), dtype=np.float32)
        tgt = np.zeros(ms, dtype=np.int64)
        for k in range(ms):
            p = pos[k+2] if k+2 < len(pos) else pos[-1]
            _, ids, lpv = lp[(pid, p)]
            tids[k] = ids; tlogp[k] = lpv; tgt[k] = int(sel[k+2]) if k+2 < N else int(sel[-1])
        feats[pid] = (x0, anchor, tids, tlogp, tgt)
        if (i+1) % 20 == 0: print(f'  {i+1}/{len(pids)} precomputed', flush=True)
del body
print(f'precomputed {len(feats)} prompts', flush=True)

rng = np.random.default_rng(SEED); rng.shuffle(pids)
train_pids = pids[:40]; eval_pids = pids[40:60]
def stack(split):
    return ([torch.from_numpy(np.concatenate([feats[p][i] for p in split], 0)) for i in range(5)])
xtr, atr, tids_tr, tlogp_tr, tgt_tr = stack(train_pids)
xev, aev, _, _, tev = stack(eval_pids)
print(f'train {xtr.shape[0]} | eval {xev.shape[0]} anchors', flush=True)

def p1_eval(head, x, a, t, bs=512):
    head.eval(); n = x.shape[0]; hit = 0
    with torch.no_grad():
        for i in range(0, n, bs):
            s = head.p1_scores(x[i:i+bs].to(dev), a[i:i+bs].to(dev))
            hit += int((s.argmax(-1).cpu() == t[i:i+bs]).sum())
    return hit / n

head0 = build_head(DSPARK, TARGET, device=dev, lora_rank=0, dtype=torch.float32)
base_p1 = p1_eval(head0, xev, aev, tev)
print(f'baseline (no LoRA) held-out p1 = {base_p1:.4f}', flush=True)
del head0

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
        logits = head.p1_scores(xtr[idx].to(dev), atr[idx].to(dev))
        draft_lp = torch.log_softmax(logits, dim=-1)
        tids_b = tids_tr[idx].to(dev); tlogp_b = tlogp_tr[idx].to(dev)
        draft_lp_top = torch.gather(draft_lp, 1, tids_b)               # [BS, TOPN]
        t_logp = torch.log_softmax(tlogp_b, dim=-1)                     # renormalize teacher over top-128
        t_prob = torch.exp(t_logp)
        kl = (t_prob * (t_logp - draft_lp_top)).sum(-1)                 # KL(teacher || draft)
        loss = kl.mean()
        opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item() * len(idx)
    if (ep+1) % 3 == 0 or ep == 0:
        tp = p1_eval(head, xev, aev, tev)
        print(f'  ep{ep+1:2d}: train KL={tl/ntr:.4f} | held-out p1={tp:.4f}', flush=True)
        head.train()

trained_p1 = p1_eval(head, xev, aev, tev)
delta = trained_p1 - base_p1
print('\n' + '=' * 60)
print(f'head.hc_fn-only KL ablation (rank={RANK}, {EPOCHS} epochs, KL vs IQ2 top-128):')
print(f'  baseline p1 = {base_p1:.4f} | trained p1 = {trained_p1:.4f} | delta = {delta:+.4f} ({delta*100:+.2f} pp)')
print(f'  (CE result was +2.41 pp; KL is the goal-specified loss)')
print('=' * 60)
json.dump({'baseline_p1': base_p1, 'trained_p1': trained_p1, 'delta': delta, 'loss': 'KL vs IQ2 top-128',
           'rank': RANK, 'epochs': EPOCHS, 'ce_reference_delta': 0.0241},
          open('issue468/artifacts/lead10_phase1/head_hcfn_kl_ablation_result.json', 'w'), indent=2)
print('wrote issue468/artifacts/lead10_phase1/head_hcfn_kl_ablation_result.json')
