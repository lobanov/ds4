#!/usr/bin/env python3
"""Task-4: gradient probe for the LoRA target. Ranks head.hc_fn vs markov-w1/w2 vs the
dense-body by the RMS gradient of the SUFFIX cross-entropy (positions 2-5) on the faithful
port. Uses the captured LIVE body (bit-exact) + the sel (the greedy targets)."""
import sys, struct, numpy as np, torch
sys.path.insert(0, 'issue468/dspark_train')
from drafter_head import build_head, BLOCK, HC, DIM, VOCAB

DSPARK = '/Users/lobanov/Projects/ds4/gguf/dspark.gguf'
TARGET = '/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf'
CAP_H = 'issue468/artifacts/lead11_unified_capture/lead3_h.bin'   # the sel (greedy)
BODY_DUMP = '/tmp/live_body20.bin'                                  # the captured live body (20 prompts)
dev = 'cpu'  # gradient probe on CPU (no MPS F8 needed; the body is captured)
HC_DIM = HC * DIM

# --- parse the sel (the greedy trajectory) from lead3_h.bin ---
sel_by_pid = {}
with open(CAP_H, 'rb') as f: data = f.read()
off = 0
while off < len(data):
    if off + 4 > len(data): break
    (idl,) = struct.unpack_from('<i', data, off); off += 4
    pid = data[off:off+idl].decode(); off += idl
    (pos, tok) = struct.unpack_from('<ii', data, off); off += 8
    off += 12288 * 4
    sel_by_pid.setdefault(pid, {'pos': [], 'sel': []})
    sel_by_pid[pid]['pos'].append(pos); sel_by_pid[pid]['sel'].append(tok)
for pid in sel_by_pid:
    o = np.argsort(sel_by_pid[pid]['pos'])
    sel_by_pid[pid]['sel'] = [sel_by_pid[pid]['sel'][i] for i in o]
    sel_by_pid[pid]['pos'] = [sel_by_pid[pid]['pos'][i] for i in o]

# --- parse the body dump (the 20 prompts, in order) ---
body_recs = []
with open(BODY_DUMP, 'rb') as f: buf = f.read()
off = 0
while off < len(buf):
    if off + 12 > len(buf): break
    a, p, bs = struct.unpack_from('<iii', buf, off); off += 12
    body = np.frombuffer(buf, dtype='<f4', count=bs*HC_DIM, offset=off).reshape(bs, HC_DIM).copy(); off += bs*HC_DIM*4
    drafts = np.frombuffer(buf, dtype='<i4', count=bs, offset=off).copy(); off += bs*4
    body_recs.append((a, p, bs, body, drafts))
print(f'parsed {len(body_recs)} body records', flush=True)

# group by prompt (pos reset) + match to sel_by_pid (the first 20 prompts of lead3)
pids = sorted(sel_by_pid.keys())[:20]
groups = []; prev_p = -1; g = []
for rec in body_recs:
    if prev_p >= 0 and rec[1] < prev_p:
        groups.append(g); g = []
    g.append(rec); prev_p = rec[1]
if g: groups.append(g)
print(f'grouped into {len(groups)} prompts; matching to {len(pids)} sels', flush=True)

# --- build the gradient probe: for each body record, the suffix CE (positions 2-5) ---
head = build_head(DSPARK, TARGET, dev, lora_rank=0, dtype=torch.float32)
head.eval()
# enable grad on the candidate target weights
for nm in ['hc_fn', 'markov_w1', 'markov_w2']:
    getattr(head, nm).requires_grad_(True)

# accumulate the gradient RMS per component (suffix only, positions 2-5 => k=1..4)
grad_accum = {nm: 0.0 for nm in ['hc_fn', 'markov_w1', 'markov_w2']}
n_samples = 0
for pi, (grp, pid) in enumerate(zip(groups, pids)):
    sel = sel_by_pid[pid]['sel']; pos_list = sel_by_pid[pid]['pos']
    # the H-capture pos starts at 0 = first generated; body-dump pos is absolute
    prompt_len = grp[0][1]   # the first anchor's absolute pos == prompt length
    for (anchor, abs_pos, bs, body, live_drafts) in grp:
        if bs < BLOCK: continue
        h_pos = abs_pos - prompt_len   # H-capture index of the anchor
        # the targets: draft[k] predicts sel[h_pos+1+k]  (position k+1)
        targets = []
        for k in range(BLOCK):
            idx = h_pos + 1 + k
            targets.append(sel[idx] if 0 <= idx < len(sel) else -1)
        if any(t < 0 for t in targets[1:5]):  # need the suffix targets
            continue
        # the teacher-forced prev_tok for k_scores: [anchor, sel[h_pos+1], sel[h_pos+2], sel[h_pos+3], sel[h_pos+4]]
        prev_tok = [anchor] + [sel[h_pos+1+i] if (h_pos+1+i) < len(sel) else 0 for i in range(BLOCK-1)]
        x = torch.as_tensor(body, dtype=torch.float32).reshape(bs, HC, DIM).unsqueeze(0)  # [1, bs, HC, DIM]
        logits = head.k_scores(x, torch.tensor([prev_tok]))  # [1, BLOCK, VOCAB]
        # suffix CE: positions 2-5 (k=1..4)
        loss = 0.0
        for k in range(1, BLOCK):
            tgt = torch.tensor([targets[k]])
            loss = loss + torch.nn.functional.cross_entropy(logits[0, k:k+1], tgt)
        loss = loss / (BLOCK - 1)
        for nm in grad_accum:
            getattr(head, nm).grad = None
        loss.backward(retain_graph=False)
        for nm in grad_accum:
            g = getattr(head, nm).grad
            if g is not None:
                grad_accum[nm] += (g.float() ** 2).mean().item()
        n_samples += 1
    if (pi+1) % 5 == 0: print(f'  {pi+1}/{len(groups)} prompts ({n_samples} samples)', flush=True)

print(f'\n=== suffix-CE gradient RMS (positions 2-5) over {n_samples} samples ===')
ranked = sorted(grad_accum.items(), key=lambda kv: -kv[1] / max(n_samples, 1))
for nm, acc in ranked:
    print(f'  {nm}: RMS-grad = {acc / max(n_samples, 1):.6e}')
print(f'\nrecommended LoRA target: {ranked[0][0]} (highest suffix-loss gradient)')
print('(note: the dense-body is structural — the parallel-block noise-placeholder — so a body')
print(' LoRA is unlikely to help the suffix; the head targets are the lever.)')
