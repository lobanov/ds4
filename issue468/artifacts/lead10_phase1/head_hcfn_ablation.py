#!/usr/bin/env python3
"""Lead 10 Phase-1 step 1 (codex gate A's decisive test): head.hc_fn-only LoRA ablation.

Does the #1 gradient target (head.hc_fn) actually GENERALIZE to a held-out p1 gain, or is the
gradient spike non-actionable? Precompute the body features (no_grad -> sidesteps the MoE
activation-retention issue), train ONLY the hc_fn LoRA (freeze norm/markov LoRA), hard-label CE
vs sel (soft labels still missing), measure held-out p1 vs the baseline.

If held-out p1 improves -> keep head.hc_fn #1 + proceed to the full multi-target LoRA.
If it harms / only reduces train loss -> demote the spike + re-select under KL/top-r.
"""
import sys, struct, json, numpy as np, torch
sys.path.insert(0, 'issue468/dspark_train'); sys.path.insert(0, 'issue468/dspark_oracle')
from drafter_body import build_body, BLOCK, HC, DIM
from drafter_head import build_head

DSPARK = '/Users/lobanov/Projects/ds4/gguf/dspark.gguf'
TARGET = '/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf'
CAP_H = 'issue468/artifacts/lead11_unified_capture/lead3_h.bin'
RANK = 32; EPOCHS = 12; LR = 3e-4; WD = 0.01; BS = 256; SEED = 42

# --- parse h.bin -> per-prompt (H [N,12288], sel [N]) pos-sorted ---
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
for pid in prompts:
    o = np.argsort(prompts[pid]['pos'])
    prompts[pid]['h'] = np.stack(prompts[pid]['h'])[o]
    prompts[pid]['sel'] = [prompts[pid]['sel'][i] for i in o]
pids = sorted(prompts.keys())
print(f'{len(pids)} prompts', flush=True)

# --- precompute body features (no_grad, F32 — F16 overflows in the MoE for some positions,
#     contaminating x0 with inf -> NaN training; F32 no_grad fits: experts 77GB + ~9GB/step,
#     no retention) ---
dev = 'mps'
print('loading drafter body F32 (MPS, no_grad)...', flush=True)
body = build_body(DSPARK, TARGET, dev, dtype=torch.float32)
print('body loaded; precomputing features...', flush=True)
feats = {}  # pid -> (x0 [ms,HC,DIM] f32 cpu, anchor [ms], target [ms])
n_inf = 0
with torch.no_grad():
    for i, pid in enumerate(pids):
        H = prompts[pid]['h']; sel = prompts[pid]['sel']; N = H.shape[0]
        ms = min(N - BLOCK - 1, N - 1)
        if ms < 1: continue
        mh_seq = H[:ms+1]; anchors = [int(sel[s]) for s in range(1, ms+1)]
        xs = body.forward_prompt(mh_seq, anchors)            # [ms,BLOCK,HC,DIM] F32
        x0 = xs[:, 0, :, :].float().cpu().numpy()            # [ms,HC,DIM] f32
        if not np.isfinite(x0).all():
            n_inf += 1  # shouldn't happen at F32; flag if it does
        anchor = np.array(anchors, dtype=np.int64)           # sel[1..ms]
        target = np.array([int(sel[s+1]) for s in range(1, ms+1)], dtype=np.int64)  # sel[2..ms+1]
        feats[pid] = (x0, anchor, target)
        if (i+1) % 20 == 0: print(f'  {i+1}/{len(pids)} precomputed (inf-prompts so far: {n_inf})', flush=True)
print(f'precomputed {len(feats)} prompts; features in CPU RAM; inf-prompts: {n_inf}', flush=True)
del body  # free the experts
torch.mps.empty_cache() if hasattr(torch.mps, 'empty_cache') else None
print(f'precomputed {len(feats)} prompts; features in CPU RAM', flush=True)

# --- train/eval split (40/20, seed) ---
rng = np.random.default_rng(SEED); rng.shuffle(pids)
train_pids = pids[:40]; eval_pids = pids[40:60]
def stack(split):
    xs = np.concatenate([feats[p][0] for p in split], 0)
    a = np.concatenate([feats[p][1] for p in split], 0)
    t = np.concatenate([feats[p][2] for p in split], 0)
    return torch.from_numpy(xs), torch.from_numpy(a), torch.from_numpy(t)
xtr, atr, ttr = stack(train_pids); xev, aev, tev = stack(eval_pids)
print(f'train {xtr.shape[0]} anchors | eval {xev.shape[0]} anchors', flush=True)

dev = 'mps'
def p1_eval(head, x, a, t, bs=512):
    head.eval(); n = x.shape[0]; hit = 0
    with torch.no_grad():
        for i in range(0, n, bs):
            s = head.p1_scores(x[i:i+bs].to(dev), a[i:i+bs].to(dev))
            hit += int((s.argmax(-1).cpu() == t[i:i+bs]).sum())
    return hit / n

# --- baseline (lora_rank=0) ---
print('building baseline head (lora_rank=0)...', flush=True)
head0 = build_head(DSPARK, TARGET, device=dev, lora_rank=0, dtype=torch.float32)
base_p1 = p1_eval(head0, xev, aev, tev)
print(f'baseline (no LoRA) held-out p1 = {base_p1:.4f}  (fidelity: should be ~0.79 IQ2-native)', flush=True)
del head0; torch.mps.empty_cache() if hasattr(torch.mps, 'empty_cache') else None

# --- head.hc_fn-only LoRA ---
print(f'building head lora_rank={RANK}, freezing non-hc_fn LoRA...', flush=True)
head = build_head(DSPARK, TARGET, device=dev, lora_rank=RANK, dtype=torch.float32)
hc_params = []; frozen = []
for name, p in head.named_parameters():
    if 'lora_hc' in name:
        hc_params.append(p)
    else:
        p.requires_grad_(False); frozen.append(name)
print(f'  trainable (hc_fn LoRA): {[p.shape for p in hc_params]} | frozen: {frozen}', flush=True)
opt = torch.optim.AdamW(hc_params, lr=LR, weight_decay=WD)
head.train()
ntr = xtr.shape[0]
for ep in range(EPOCHS):
    perm = torch.randperm(ntr); tl = 0.0
    for i in range(0, ntr, BS):
        idx = perm[i:i+BS]
        s = head.p1_scores(xtr[idx].to(dev), atr[idx].to(dev))
        loss = torch.nn.functional.cross_entropy(s, ttr[idx].to(dev))
        opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item() * len(idx)
    if (ep+1) % 3 == 0 or ep == 0:
        tp = p1_eval(head, xev, aev, tev)
        print(f'  ep{ep+1:2d}: train CE={tl/ntr:.4f} | held-out p1={tp:.4f}', flush=True)
        head.train()

trained_p1 = p1_eval(head, xev, aev, tev)
delta = trained_p1 - base_p1
print('\n' + '=' * 60)
print(f'head.hc_fn-only ablation (rank={RANK}, {EPOCHS} epochs, hard-label CE):')
print(f'  baseline (no LoRA) p1 = {base_p1:.4f}')
print(f'  trained (hc_fn LoRA) p1 = {trained_p1:.4f}')
print(f'  delta = {delta:+.4f} ({delta*100:+.2f} pp)')
verdict = ('GENERALIZES (keep head.hc_fn #1)' if delta > 0.005 else
           ('MARGINAL' if delta > -0.005 else 'NO GENERALIZATION (demote the spike)'))
print(f'  -> {verdict}')
print('=' * 60)
json.dump({'baseline_p1': base_p1, 'trained_p1': trained_p1, 'delta': delta,
           'rank': RANK, 'epochs': EPOCHS, 'loss': 'CE vs sel (hard-label)',
           'verdict': verdict, 'train_anchors': int(ntr), 'eval_anchors': int(xev.shape[0])},
          open('issue468/artifacts/lead10_phase1/head_hcfn_ablation_result.json', 'w'), indent=2)
print('wrote issue468/artifacts/lead10_phase1/head_hcfn_ablation_result.json')
