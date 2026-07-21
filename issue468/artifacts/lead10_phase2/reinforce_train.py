#!/usr/bin/env python3
"""Task-5: REINFORCE LoRA on head.hc_fn. Reward = #accepted (the longest draft prefix
matching the sel), positions coupled via the autoregressive rollout. Trains on the
captured LIVE body (bit-exact) + the sel; held-out ΔE[a|K] with CI."""
import sys, struct, numpy as np, torch
sys.path.insert(0, 'issue468/dspark_train')
from drafter_head import build_head, BLOCK, HC, DIM, VOCAB

DSPARK = '/Users/lobanov/Projects/ds4/gguf/dspark.gguf'
TARGET = '/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf'
CAP_H = 'issue468/artifacts/lead11_unified_capture/lead3_h.bin'
BODY_DUMP = '/tmp/live_body20.bin'
dev = 'mps'; HC_DIM = HC * DIM
RANK = 32; LR = 1e-3; EPOCHS = 8; BS = 16; SAMPLES_PER = 2  # REINFORCE samples per anchor
TEMP = 0.7  # sampling temperature (explore; eval is greedy)

# --- parse sel + body (same as grad_probe_suffix) ---
sel_by_pid = {}
with open(CAP_H, 'rb') as f: data = f.read()
off = 0
while off < len(data):
    if off + 4 > len(data): break
    (idl,) = struct.unpack_from('<i', data, off); off += 4
    pid = data[off:off+idl].decode(); off += idl
    (pos, tok) = struct.unpack_from('<ii', data, off); off += 8; off += 12288 * 4
    sel_by_pid.setdefault(pid, {'pos': [], 'sel': []})
    sel_by_pid[pid]['pos'].append(pos); sel_by_pid[pid]['sel'].append(tok)
for pid in sel_by_pid:
    o = np.argsort(sel_by_pid[pid]['pos'])
    sel_by_pid[pid]['sel'] = [sel_by_pid[pid]['sel'][i] for i in o]

body_recs = []
with open(BODY_DUMP, 'rb') as f: buf = f.read()
off = 0
while off < len(buf):
    if off + 12 > len(buf): break
    a, p, bs = struct.unpack_from('<iii', buf, off); off += 12
    body = np.frombuffer(buf, dtype='<f4', count=bs*HC_DIM, offset=off).reshape(bs, HC_DIM).copy(); off += bs*HC_DIM*4
    drafts = np.frombuffer(buf, dtype='<i4', count=bs, offset=off).copy(); off += bs*4
    body_recs.append((a, p, bs, body, drafts))

# group by the bench jsonl cycle counts (robust; the pos-reset heuristic found 19 not 20)
import json
bench_counts = []
for l in open('/tmp/bench20.jsonl'):
    r = json.loads(l); bench_counts.append(len(r.get('dspark_cycles', [])))
groups = []; idx = 0
for cnt in bench_counts:
    groups.append(body_recs[idx:idx+cnt]); idx += cnt
pids = sorted(sel_by_pid.keys())[:20]
print(f'grouped into {len(groups)} prompts via bench cycle counts', flush=True)

# build the sample list: (body[BLOCK,HC,DIM], anchor, sel_targets[BLOCK])
samples = []
for grp, pid in zip(groups, pids):
    sel = sel_by_pid[pid]['sel']; prompt_len = grp[0][1]
    for (anchor, abs_pos, bs, body, _) in grp:
        if bs < BLOCK: continue
        h_pos = abs_pos - prompt_len
        tgts = [sel[h_pos+1+k] if (h_pos+1+k) < len(sel) else -1 for k in range(BLOCK)]
        if any(t < 0 for t in tgts): continue
        samples.append((torch.as_tensor(body[:BLOCK], dtype=torch.float32).reshape(BLOCK, HC, DIM), anchor, tgts))
print(f'{len(samples)} samples', flush=True)

# split train/held-out (by prompt: 16 train, 3 held-out, ~20%)
np.random.seed(42); n_hold = 4
hold_idx = set(np.random.choice(len(groups), n_hold, replace=False))
train_samples = []; hold_samples = []
for grp_i, (grp, pid) in enumerate(zip(groups, pids)):
    sel = sel_by_pid[pid]['sel']; prompt_len = grp[0][1]
    for (anchor, abs_pos, bs, body, _) in grp:
        if bs < BLOCK: continue
        h_pos = abs_pos - prompt_len
        tgts = [sel[h_pos+1+k] if (h_pos+1+k) < len(sel) else -1 for k in range(BLOCK)]
        if any(t < 0 for t in tgts): continue
        s = (torch.as_tensor(body[:BLOCK], dtype=torch.float32).reshape(BLOCK, HC, DIM), anchor, tgts)
        (hold_samples if grp_i in hold_idx else train_samples).append(s)
print(f'train={len(train_samples)} held-out={len(hold_samples)}', flush=True)

head = build_head(DSPARK, TARGET, dev, lora_rank=RANK, dtype=torch.float32)
head.train()
# only train the hc_fn LoRA (the chosen target); freeze the markov + norm LoRA
for nm in ['lora_norm', 'lora_mw1_A', 'lora_mw1_B', 'lora_mw2_A', 'lora_mw2_B']:
    getattr(head, nm).requires_grad_(False)
opt = torch.optim.AdamW([head.lora_hc_A, head.lora_hc_B], lr=LR, weight_decay=0.01)
mw2_T = head._mw2_T(torch.float32).detach()  # constant (markov LoRA frozen) -> no graph
lm_head = head.lm_head  # [VOCAB, DIM]

def rollout(body, anchor, sample=True, temp=TEMP):
    """Autoregressive rollout. Returns drafts [BLOCK], logpi sum, accepted count (vs None here)."""
    x = body.unsqueeze(0).to(dev)  # [1, BLOCK, HC, DIM]
    h = head.hc_head(x)[0]  # [BLOCK, DIM]
    hf = h.float(); hf = hf * torch.rsqrt((hf*hf).mean(-1, keepdim=True) + 1e-6) * head._norm_w().to(dev, torch.float32)
    base = hf.to(torch.float32) @ lm_head.T  # [BLOCK, VOCAB]
    prev = torch.tensor([anchor], device=dev)
    drafts = []; logpi = 0.0
    for k in range(BLOCK):
        emb = head._mw1(prev).to(dev, torch.float32)  # [1, rank]
        logits = (base[k] + emb @ mw2_T) / temp  # [VOCAB]
        logp = torch.log_softmax(logits, -1)
        if sample:
            d = torch.multinomial(torch.exp(logp), 1).squeeze(0)  # [1]
        else:
            d = logits.argmax(-1)
        logpi = logpi + logp[0, d]
        drafts.append(int(d.item())); prev = d
    return drafts, logpi

def accepted_count(drafts, tgts):
    n = 0
    for k in range(BLOCK):
        if drafts[k] == tgts[k]: n += 1
        else: break
    return n

# baseline (EMA of the reward)
baseline = None; BETA = 0.9

print(f'\n=== REINFORCE training (rank={RANK}, {EPOCHS} epochs, temp={TEMP}) ===', flush=True)
for ep in range(EPOCHS):
    perm = np.random.permutation(len(train_samples))
    ep_reward = 0.0; nb = 0
    for i in range(0, len(perm), BS):
        batch_idx = perm[i:i+BS]
        loss = torch.zeros(1, device=dev); batch_R = []
        logpis = []; Rs = []
        for bi in batch_idx:
            body, anchor, tgts = train_samples[bi]
            for _ in range(SAMPLES_PER):
                drafts, logpi = rollout(body, anchor, sample=True)
                R = accepted_count(drafts, tgts)
                logpis.append(logpi); Rs.append(R)
        Rs_t = torch.tensor(Rs, dtype=torch.float32, device=dev)
        if baseline is None: baseline = Rs_t.mean().item()
        adv = Rs_t - baseline  # [S]
        logpis_t = torch.stack(logpis)  # [S]
        loss = -(adv * logpis_t).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        batch_R.extend(Rs)
        baseline = BETA * baseline + (1-BETA) * np.mean(batch_R)
        ep_reward += np.mean(batch_R); nb += 1
    print(f'  ep{ep+1}: mean reward (train, sampled) = {ep_reward/max(nb,1):.3f}  baseline={baseline:.3f}', flush=True)

# held-out eval: greedy E[a|K] with LoRA vs without
trained_B = head.lora_hc_B.data.clone()
trained_A = head.lora_hc_A.data.clone()
def eval_greedy(samples, with_lora):
    if with_lora:
        head.lora_hc_B.data.copy_(trained_B); head.lora_hc_A.data.copy_(trained_A)
    else:
        head.lora_hc_B.data.zero_()
    counts = []
    for body, anchor, tgts in samples:
        with torch.no_grad():
            drafts, _ = rollout(body, anchor, sample=False)
        counts.append(accepted_count(drafts, tgts))
    return np.array(counts)

print(f'\n=== held-out evaluation (greedy, {len(hold_samples)} samples) ===', flush=True)
baseline_counts = eval_greedy(hold_samples, with_lora=False)
lora_counts = eval_greedy(hold_samples, with_lora=True)
print(f'baseline E[a|K] (no LoRA) = {baseline_counts.mean():.4f}', flush=True)
print(f'LoRA      E[a|K]          = {lora_counts.mean():.4f}', flush=True)
delta = lora_counts - baseline_counts
print(f'ΔE[a|K] = {delta.mean():.4f}  (paired)', flush=True)
# bootstrap CI
rng = np.random.default_rng(42); B = 2000; boots = []
for _ in range(B):
    idx = rng.integers(0, len(delta), len(delta))
    boots.append(delta[idx].mean())
lo, hi = np.percentile(boots, [2.5, 97.5])
print(f'95% CI = [{lo:.4f}, {hi:.4f}]  (excludes 0: {not (lo <= 0 <= hi)})', flush=True)
print(f'\nPROCEED gate: CI excludes 0 = {not (lo <= 0 <= hi)}', flush=True)

# save the LoRA delta for baking
if not (lo <= 0 <= hi):
    delta_w = (head.lora_hc_B.float() @ head.lora_hc_A.float()).detach().cpu().numpy()
    np.save('/tmp/reinforce_lora_delta.npy', delta_w)
    print(f'saved LoRA delta to /tmp/reinforce_lora_delta.npy (shape {delta_w.shape})', flush=True)
