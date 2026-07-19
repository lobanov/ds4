#!/usr/bin/env python3
"""Lead 10 Phase-1: head.hc_fn-only learning curve + fidelity gate.

(1) Precompute + cache the lead3 body features (F32 no_grad) + the KL target (IQ2 top-128).
(2) Learning curve: train head.hc_fn KL on nested train subsets (10/20/40 prompts) -> held-out
    p1 on the 20-eval. Sets the train-set size (data-limited vs plateau).
(3) Fidelity gate: (a) the baseline (LoRA disabled) reproduces the IQ2-native p1; (b) the
    trained head at F16 vs F32 agrees on Q2 (the dtype-invariance survives fine-tuning).
"""
import sys, struct, json, os, numpy as np, torch
sys.path.insert(0, 'issue468/dspark_train'); sys.path.insert(0, 'issue468/dspark_oracle')
from drafter_body import build_body, BLOCK
from drafter_head import build_head

DSPARK = '/Users/lobanov/Projects/ds4/gguf/dspark.gguf'
TARGET = '/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf'
CAP_H = 'issue468/artifacts/lead11_unified_capture/lead3_h.bin'
CAP_LP = 'issue468/artifacts/lead10_phase1/lead3_logprobs_recapture.jsonl'
FEAT_CACHE = 'issue468/artifacts/lead10_phase1/lead3_features.npz'
RANK = 32; EPOCHS = 12; LR = 3e-4; WD = 0.01; BS = 256; SEED = 42; TOPN = 128
dev = 'mps'

# --- features: parse h.bin + logprobs, precompute (or load cache) ---
def precompute():
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
    print('precomputing body features F32 (no_grad)...', flush=True)
    body = build_body(DSPARK, TARGET, dev, dtype=torch.float32)
    feats = {}
    with torch.no_grad():
        for i, pid in enumerate(sorted(prompts.keys())):
            H = prompts[pid]['h']; sel = prompts[pid]['sel']; pos = prompts[pid]['pos']; N = len(sel)
            ms = min(N - BLOCK - 1, N - 1)
            if ms < 1: continue
            xs = body.forward_prompt(H[:ms+1], [int(sel[s]) for s in range(1, ms+1)])
            x0 = xs[:, 0, :, :].float().cpu().numpy()
            anchor = np.array([int(sel[s]) for s in range(1, ms+1)], dtype=np.int64)
            tids = np.zeros((ms, TOPN), dtype=np.int64); tlogp = np.zeros((ms, TOPN), dtype=np.float32)
            tgt = np.zeros(ms, dtype=np.int64)
            for k in range(ms):
                p = pos[k+2] if k+2 < len(pos) else pos[-1]
                _, ids, lpv = lp[(pid, p)]; tids[k] = ids; tlogp[k] = lpv
                tgt[k] = int(sel[k+2]) if k+2 < N else int(sel[-1])
            feats[pid] = (x0, anchor, tids, tlogp, tgt)
            if (i+1) % 20 == 0: print(f'  {i+1}/60', flush=True)
    del body
    np.savez(FEAT_CACHE, pids=np.array(sorted(feats.keys())),
             **{f'{pid}__x0': feats[pid][0] for pid in feats},
             **{f'{pid}__a': feats[pid][1] for pid in feats},
             **{f'{pid}__tids': feats[pid][2] for pid in feats},
             **{f'{pid}__tlogp': feats[pid][3] for pid in feats},
             **{f'{pid}__tgt': feats[pid][4] for pid in feats})
    print(f'cached features -> {FEAT_CACHE}', flush=True)
    return feats

if os.path.exists(FEAT_CACHE):
    print(f'loading cached features {FEAT_CACHE}', flush=True)
    cz = np.load(FEAT_CACHE, allow_pickle=True)
    pids_all = list(cz['pids'])
    feats = {pid: (cz[f'{pid}__x0'], cz[f'{pid}__a'], cz[f'{pid}__tids'], cz[f'{pid}__tlogp'], cz[f'{pid}__tgt']) for pid in pids_all}
else:
    feats = precompute(); pids_all = sorted(feats.keys())

rng = np.random.default_rng(SEED); pids = list(pids_all); rng.shuffle(pids)
train_pids = pids[:40]; eval_pids = pids[40:60]
def stack(split):
    return [torch.from_numpy(np.concatenate([feats[p][i] for p in split], 0)) for i in range(5)]
xev, aev, _, _, tev = stack(eval_pids)

def p1_eval(head, x, a, t, bs=512):
    head.eval(); n = x.shape[0]; hit = 0; hdt = head.hc_fn.dtype
    with torch.no_grad():
        for i in range(0, n, bs):
            s = head.p1_scores(x[i:i+bs].to(dev).to(hdt), a[i:i+bs].to(dev))
            hit += int((s.argmax(-1).cpu() == t[i:i+bs]).sum())
    return hit / n

def train_kl(train_subset, tag):
    xtr, atr, tids_tr, tlogp_tr, _ = stack(train_subset)
    head = build_head(DSPARK, TARGET, device=dev, lora_rank=RANK, dtype=torch.float32)
    hc_params = [p for n, p in head.named_parameters() if 'lora_hc' in n]
    for n, p in head.named_parameters():
        if 'lora_hc' not in n: p.requires_grad_(False)
    opt = torch.optim.AdamW(hc_params, lr=LR, weight_decay=WD)
    head.train(); ntr = xtr.shape[0]
    for ep in range(EPOCHS):
        perm = torch.randperm(ntr)
        for i in range(0, ntr, BS):
            idx = perm[i:i+BS]
            logits = head.p1_scores(xtr[idx].to(dev), atr[idx].to(dev))
            draft_lp = torch.log_softmax(logits, dim=-1)
            draft_lp_top = torch.gather(draft_lp, 1, tids_tr[idx].to(dev))
            t_logp = torch.log_softmax(tlogp_tr[idx].to(dev), dim=-1); t_prob = torch.exp(t_logp)
            loss = (t_prob * (t_logp - draft_lp_top)).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    tp = p1_eval(head, xev, aev, tev)
    print(f'  [{tag}] train={ntr} anchors -> held-out p1 = {tp:.4f}', flush=True)
    return head, tp

# baseline
head0 = build_head(DSPARK, TARGET, device=dev, lora_rank=0, dtype=torch.float32)
base_p1 = p1_eval(head0, xev, aev, tev); del head0
print(f'baseline (no LoRA) held-out p1 = {base_p1:.4f}', flush=True)

# learning curve
print('\n=== learning curve (head.hc_fn KL, nested train subsets) ===', flush=True)
curve = {}
for n_train in [10, 20, 40]:
    head_n, tp_n = train_kl(train_pids[:n_train], f'n={n_train}')
    curve[n_train] = tp_n
    if n_train == 40: head_40 = head_n  # keep the full-train head for the fidelity gate

# fidelity gate: F16 vs F32 on the trained (40-train) head
print('\n=== fidelity gate (F16 vs F32 on the trained head) ===', flush=True)
p1_f32 = p1_eval(head_40, xev, aev, tev)
head_f16 = build_head(DSPARK, TARGET, device=dev, lora_rank=RANK, dtype=torch.float16)
with torch.no_grad():
    head_f16.lora_hc_A.copy_(head_40.lora_hc_A.half()); head_f16.lora_hc_B.copy_(head_40.lora_hc_B.half())
p1_f16 = p1_eval(head_f16, xev, aev, tev)
print(f'  trained head: F32 p1 = {p1_f32:.4f} | F16 p1 = {p1_f16:.4f} | |diff| = {abs(p1_f32-p1_f16):.5f}', flush=True)
f16_ok = abs(p1_f32 - p1_f16) < 0.005

print('\n' + '=' * 64)
print('LEARNING CURVE (held-out p1 vs train-set size):')
for n, tp in curve.items():
    print(f'  n_train={n:2d} prompts -> held-out p1 = {tp:.4f} (Δ={tp-base_p1:+.4f})')
climb = curve[40] - curve[20]
print(f'\n  20->40 gain: {climb:+.4f} ({"data-limited (scale to distill_corpus)" if climb > 0.003 else "plateau (no need to scale)"})')
print(f'\nFIDELITY GATE:')
print(f'  baseline (no LoRA) p1 = {base_p1:.4f} (matches combined300 per-prompt ~0.85; the 0.79 is the full-59 mean)')
print(f'  F16==F32: |diff| = {abs(p1_f32-p1_f16):.5f} -> {"PASS" if f16_ok else "FAIL"} (dtype-invariance survives fine-tuning)')
print('=' * 64)
json.dump({'baseline_p1': base_p1, 'curve': curve, 'climb_20_to_40': climb,
           'f16_p1': p1_f16, 'f32_p1': p1_f32, 'f16_f32_ok': f16_ok},
          open('issue468/artifacts/lead10_phase1/learning_curve_result.json', 'w'), indent=2)
print('wrote issue468/artifacts/lead10_phase1/learning_curve_result.json')
