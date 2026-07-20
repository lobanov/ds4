#!/usr/bin/env python3
"""Lead 10 Phase-2: held-out validation via 3-fold cross-validation on the lead3 60 prompts.

Each prompt is held-out once (3 folds × 20 eval) → 60 held-out eval points. The head.hc_fn
LoRA (KL vs IQ2 top-128, rank=32, 12 epochs) is trained on 40, eval'd on 20 per fold.
Per-prompt p1 (fraction correct) → the Δp1 vs the baseline (untrained head) → prompt-clustered
bootstrap CI + per-source breakdown → the PROCEED/MARGINAL/STOP decision.
"""
import sys, json, numpy as np, torch
sys.path.insert(0, 'issue468/dspark_train'); sys.path.insert(0, 'issue468/dspark_oracle')
from drafter_head import build_head

DSPARK = '/Users/lobanov/Projects/ds4/gguf/dspark.gguf'
TARGET = '/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf'
FEAT = 'issue468/artifacts/lead10_phase1/lead3_features.npz'
RANK = 32; EPOCHS = 12; LR = 3e-4; WD = 0.01; BS = 256; SEED = 42
dev = 'mps'

cz = np.load(FEAT, allow_pickle=True)
pids = list(cz['pids'])
feats = {pid: (cz[f'{pid}__x0'], cz[f'{pid}__a'], cz[f'{pid}__tids'], cz[f'{pid}__tlogp'], cz[f'{pid}__tgt']) for pid in pids}
print(f'{len(pids)} prompts loaded from cache', flush=True)

def per_prompt_p1(head, eval_pids):
    """Return {pid: p1} — the fraction of correct first-drafts per prompt."""
    head.eval()
    result = {}
    with torch.no_grad():
        for pid in eval_pids:
            x0, a, _, _, t = feats[pid]
            xt = torch.from_numpy(x0).to(dev).to(head.hc_fn.dtype)
            at = torch.from_numpy(a).to(dev)
            tt = torch.from_numpy(t)
            s = head.p1_scores(xt, at)
            hit = int((s.argmax(-1).cpu() == tt).sum())
            result[pid] = hit / len(tt)
    return result

def train_kl(train_pids):
    xtr = torch.from_numpy(np.concatenate([feats[p][0] for p in train_pids], 0))
    atr = torch.from_numpy(np.concatenate([feats[p][1] for p in train_pids], 0))
    tids = torch.from_numpy(np.concatenate([feats[p][2] for p in train_pids], 0))
    tlogp = torch.from_numpy(np.concatenate([feats[p][3] for p in train_pids], 0))
    head = build_head(DSPARK, TARGET, device=dev, lora_rank=RANK, dtype=torch.float32)
    hc = [p for n, p in head.named_parameters() if 'lora_hc' in n]
    for n, p in head.named_parameters():
        if 'lora_hc' not in n: p.requires_grad_(False)
    opt = torch.optim.AdamW(hc, lr=LR, weight_decay=WD)
    head.train(); ntr = xtr.shape[0]
    for ep in range(EPOCHS):
        perm = torch.randperm(ntr)
        for i in range(0, ntr, BS):
            idx = perm[i:i+BS]
            logits = head.p1_scores(xtr[idx].to(dev), atr[idx].to(dev))
            dlp = torch.log_softmax(logits, -1)
            dlpt = torch.gather(dlp, 1, tids[idx].to(dev))
            tlp = torch.log_softmax(tlogp[idx].to(dev), -1)
            loss = (torch.exp(tlp) * (tlp - dlpt)).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return head

# baseline (untrained head) per-prompt p1 on all 60
h0 = build_head(DSPARK, TARGET, device=dev, lora_rank=0, dtype=torch.float32)
base_pp = per_prompt_p1(h0, pids)
del h0

# 3-fold CV
rng = np.random.default_rng(SEED); shuffled = list(pids); rng.shuffle(shuffled)
folds = [shuffled[i::3] for i in range(3)]  # 3 folds, ~20 each
trained_pp = {}
for fi, eval_fold in enumerate(folds):
    train_fold = [p for p in shuffled if p not in eval_fold]
    print(f'fold {fi+1}/3: train {len(train_fold)} | eval {len(eval_fold)}', flush=True)
    head = train_kl(train_fold)
    pp = per_prompt_p1(head, eval_fold)
    trained_pp.update(pp)
    del head

# aggregate + CI
base_arr = np.array([base_pp[p] for p in pids])
trained_arr = np.array([trained_pp[p] for p in pids])
delta_arr = trained_arr - base_arr

def boot_ci(vals, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed); n = len(vals); vals = np.array(vals)
    means = [vals[rng.integers(0, n, size=n)].mean() for _ in range(n_boot)]
    return float(vals.mean()), *np.percentile(means, [2.5, 97.5])

base_m, base_lo, base_hi = boot_ci(base_arr)
trained_m, trained_lo, trained_hi = boot_ci(trained_arr)
delta_m, delta_lo, delta_hi = boot_ci(delta_arr)

# per-source
sources = {}
for pid in pids:
    src = pid.rsplit('_', 1)[0]
    sources.setdefault(src, []).append(delta_arr[pids.index(pid)])

print('\n' + '=' * 64)
print(f'PHASE-2 HELD-OUT VALIDATION (3-fold CV, 60 held-out points, head.hc_fn KL LoRA):')
print(f'  baseline (untrained) p1 = {base_m:.4f} CI[{base_lo:.4f},{base_hi:.4f}]')
print(f'  trained (head.hc_fn)  p1 = {trained_m:.4f} CI[{trained_lo:.4f},{trained_hi:.4f}]')
print(f'  Δp1 = {delta_m:+.4f} CI[{delta_lo:+.4f},{delta_hi:+.4f}]')
print(f'\n  per-source Δp1:')
for s, v in sorted(sources.items()):
    sm, slo, shi = boot_ci(v)
    print(f'    {s:12s} (n={len(v):2d}): Δ={sm:+.4f} CI[{slo:+.4f},{shi:+.4f}]')
print()
if delta_m >= 0.02 and delta_lo > 0:
    verdict = 'PROCEED'
elif delta_m >= 0.01:
    verdict = 'MARGINAL'
else:
    verdict = 'STOP'
print(f'DECISION: {verdict}')
print(f'  (rule: PROCEED ≥+2pp CI>0; MARGINAL +1-2pp; STOP ≤+1pp or harm)')
print('=' * 64)

result = {
    'n_eval': len(pids), 'method': '3-fold CV (head.hc_fn KL LoRA, rank=32, 12 epochs)',
    'baseline_p1': base_m, 'baseline_ci': [base_lo, base_hi],
    'trained_p1': trained_m, 'trained_ci': [trained_lo, trained_hi],
    'delta_p1': delta_m, 'delta_ci': [delta_lo, delta_hi],
    'per_source': {s: {'n': len(v), 'delta': float(np.mean(v)), 'ci': list(boot_ci(v))} for s, v in sources.items()},
    'verdict': verdict,
    'per_prompt': {p: {'base': float(base_pp[p]), 'trained': float(trained_pp[p]), 'delta': float(trained_pp[p] - base_pp[p])} for p in pids},
}
import os; os.makedirs('issue468/artifacts/lead10_phase2', exist_ok=True)
json.dump(result, open('issue468/artifacts/lead10_phase2/phase2_validation_result.json', 'w'), indent=2)
print(f'\nwrote issue468/artifacts/lead10_phase2/phase2_validation_result.json')
