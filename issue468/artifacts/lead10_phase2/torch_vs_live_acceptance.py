#!/usr/bin/env python3
"""Measure the torch port's per-position acceptance on the lead3 captures + compare with live.

The codex gate B found the torch port overestimates acceptance (~0.844 vs live ~0.7945).
This localizes WHERE: per-position, torch vs live. If the overestimate is in the suffix
(positions 2-5), it confirms the divergence is in the drafter's suffix quality.
"""
import sys, struct, numpy as np, torch
sys.path.insert(0, 'issue468/dspark_train')
from drafter_body import build_body, BLOCK, HC, DIM
from drafter_head import build_head

DSPARK = '/Users/lobanov/Projects/ds4/gguf/dspark.gguf'
TARGET = '/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf'
CAP_H = 'issue468/artifacts/lead11_unified_capture/lead3_h.bin'
dev = 'mps'

# parse h.bin
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
    prompts[pid]['pos'] = [prompts[pid]['pos'][i] for i in o]
pids = sorted(prompts.keys())[:20]
print(f'{len(pids)} prompts', flush=True)

body = build_body(DSPARK, TARGET, dev, dtype=torch.float32)
head = build_head(DSPARK, TARGET, dev, lora_rank=0, dtype=torch.float32)
head.eval()

# per-position match counts (over all anchors, all prompts)
match = np.zeros(BLOCK, dtype=np.int64)       # match[k] = # anchors where draft[k]==sel
condmatch = np.zeros(BLOCK, dtype=np.int64)   # condmatch[k] = # anchors where ALL of draft[0..k]==sel
n_anchors = 0
with torch.no_grad():
    for pid in pids:
        H = prompts[pid]['h']; sel = prompts[pid]['sel']; N = len(sel)
        ms = min(N - BLOCK - 1, N - 1)
        if ms < BLOCK + 2: continue
        xs = body.forward_prompt(H[:ms+1], [int(sel[s]) for s in range(1, ms+1)])  # [ms, BLOCK, HC, DIM]
        # head.forward per anchor: xs[j] is anchor sel[j+1], proposes sel[j+2..j+6]
        for j in range(ms - BLOCK):
            out, _ = head.forward(xs[j:j+1], torch.tensor([int(sel[j+1])]))  # out [1, BLOCK+1]
            drafts = out[0, 1:].cpu().numpy()  # [BLOCK]
            ok = np.array([drafts[k] == int(sel[j+2+k]) for k in range(BLOCK)])
            for k in range(BLOCK):
                if ok[:k+1].all(): condmatch[k] += 1
                if ok[k]: match[k] += 1
            n_anchors += 1
        print(f'  {pid}: {ms-BLOCK} anchors', flush=True)

print(f'\n{n_anchors} anchors total', flush=True)
print(f'\n=== per-position acceptance (torch port, F32) ===')
prev = n_anchors
for k in range(BLOCK):
    marg = condmatch[k] / n_anchors  # P(all of 0..k match)
    cond = condmatch[k] / condmatch[k-1] if k > 0 else condmatch[0] / n_anchors
    print(f'  pos {k+1}: marginal={marg:.4f}  conditional={cond:.4f}  (match[k]={match[k]})')

print(f'\n=== live Metal drafter (for comparison) ===')
print('  pos 1: marginal=0.996  conditional=0.996')
print('  pos 2: marginal=0.835  conditional=0.839')
print('  pos 3: marginal=0.673  conditional=0.806')
print('  pos 4: marginal=0.516  conditional=0.767')
print('  pos 5: marginal=0.390  conditional=0.755')
live_marg = [0.996, 0.835, 0.673, 0.516, 0.390]
print(f'\n=== overestimate (torch marginal - live marginal) ===')
for k in range(BLOCK):
    m = condmatch[k] / n_anchors
    print(f'  pos {k+1}: +{m - live_marg[k]:.4f}')
