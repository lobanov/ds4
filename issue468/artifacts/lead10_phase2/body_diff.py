#!/usr/bin/env python3
"""Bit-exact diff: live Metal drafter body output vs torch port body output.

Parses /tmp/live_body.bin (the live dump) + runs the torch port on the same prompts
(from lead3_h.bin), aligns by position, and reports the per-element diff of the body
output (the 3-stage output batch_cur_hc). Also verifies the draft-token alignment.
"""
import sys, struct, numpy as np, torch
sys.path.insert(0, 'issue468/dspark_train')
from drafter_body import build_body, BLOCK, HC, DIM
from drafter_head import build_head

DSPARK = '/Users/lobanov/Projects/ds4/gguf/dspark.gguf'
TARGET = '/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf'
CAP_H = 'issue468/artifacts/lead11_unified_capture/lead3_h.bin'
LIVE_DUMP = '/tmp/live_body.bin'
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

# parse live body dump
HC_DIM = HC * DIM  # 16384
records = []  # list of (anchor, pos, block_size, body[block_size, HC_DIM], drafts[block_size])
with open(LIVE_DUMP, 'rb') as f:
    buf = f.read()
off = 0
while off < len(buf):
    if off + 12 > len(buf): break
    anchor, pos, bs = struct.unpack_from('<iii', buf, off); off += 12
    body_n = bs * HC_DIM
    body = np.frombuffer(buf, dtype='<f4', count=body_n, offset=off); off += body_n * 4
    drafts = np.frombuffer(buf, dtype='<i4', count=bs, offset=off); off += bs * 4
    records.append((anchor, pos, bs, body.reshape(bs, HC_DIM).copy(), drafts.copy()))
print(f'parsed {len(records)} live dump records', flush=True)

# the 3 prompts (codealpaca_0080/81/82); separate records by pos reset (pos drops at prompt boundary)
pids = ['codealpaca_0080', 'codealpaca_0081', 'codealpaca_0082']
rec_by_pid = {}
group = []
prev_pos = -1
groups = []
for rec in records:
    _, pos, _, _, _ = rec
    if prev_pos >= 0 and pos < prev_pos:
        groups.append(group); group = []
    group.append(rec); prev_pos = pos
if group: groups.append(group)
for pid, g in zip(pids, groups):
    rec_by_pid[pid] = g
print(f'groups: {[len(g) for g in groups]} records', flush=True)

body = build_body(DSPARK, TARGET, dev, dtype=torch.float32)
head = build_head(DSPARK, TARGET, dev, lora_rank=0, dtype=torch.float32)
head.eval()

print(f'\n{"prompt":16s} {"n":>3s} {"matched":>7s} {"max|d|":>10s} {"mean|d|":>10s} {"rel|d|":>8s} {"draft_agree":>11s}', flush=True)
overall_max = 0.0
with torch.no_grad():
    for pid in pids:
        H = prompts[pid]['h']; sel = prompts[pid]['sel']; pos_list = prompts[pid]['sel']
        N = len(sel); ms = min(N - BLOCK - 1, N - 1)
        xs = body.forward_prompt(H[:ms+1], [int(sel[s]) for s in range(1, ms+1)])  # [ms, BLOCK, HC, DIM]
        xs_cpu = xs.cpu().numpy()  # [ms, BLOCK, HC, DIM]
        recs = rec_by_pid[pid]
        prompt_len = recs[0][1]  # first anchor's absolute pos == prompt length
        n_match = 0; n_tot = 0; max_d = 0.0; sum_d = 0.0; sum_abs = 0.0; n_draft_agree = 0; n_draft_tot = 0
        early_diffs = []; late_diffs = []; anchor_ok = 0; anchor_bad = 0
        pos1_agree = 0; pos1_tot = 0
        for (anchor, pos, bs, live_body, live_drafts) in recs:
            h_pos = pos - prompt_len     # H-capture index of this anchor
            j = h_pos - 1                # torch xs[j] = anchor sel[j+1] at H pos j+1
            if j < 0 or j >= xs_cpu.shape[0]:
                continue
            # alignment check: torch anchor sel[h_pos] must equal the live anchor
            if int(sel[h_pos]) == anchor: anchor_ok += 1
            else: anchor_bad += 1
            torch_body = xs_cpu[j].reshape(bs, HC_DIM)  # [BLOCK, HC, DIM] -> [bs, HC_DIM]
            if torch_body.shape != live_body.shape:
                continue
            n_match += 1
            d = np.abs(torch_body - live_body)
            this_max = float(d.max()); this_mean = float(d.mean())
            max_d = max(max_d, this_max)
            sum_d += d.sum(); sum_abs += np.abs(live_body).sum()
            n_tot += live_body.size
            if h_pos < 20: early_diffs.append(this_mean)
            else: late_diffs.append(this_mean)
            # draft agreement: torch drafts at xs[j]
            out, _ = head.forward(xs[j:j+1], torch.tensor([int(sel[h_pos])]))
            torch_drafts = out[0, 1:].cpu().numpy()
            n_draft_agree += int((torch_drafts == live_drafts).sum())
            n_draft_tot += len(live_drafts)
            if int(torch_drafts[0]) == int(live_drafts[0]): pos1_agree += 1
            pos1_tot += 1
        mean_d = sum_d / max(n_tot, 1)
        rel_d = sum_d / max(sum_abs, 1e-9)
        overall_max = max(overall_max, max_d)
        em = np.mean(early_diffs) if early_diffs else float('nan')
        lm_ = np.mean(late_diffs) if late_diffs else float('nan')
        print(f'{pid:16s} {len(recs):3d} {n_match:7d} {max_d:10.4f} {mean_d:10.5f} {rel_d:8.5f} '
              f'draft={n_draft_agree}/{n_draft_tot} pos1={pos1_agree}/{pos1_tot} '
              f'anchor_ok={anchor_ok}/{anchor_ok+anchor_bad} early={em:.3f} late={lm_:.3f}', flush=True)

print(f'\noverall max|diff| = {overall_max:.4f}', flush=True)
print('(max|diff| near 0 => bit-exact; large => body forward diverges)', flush=True)
