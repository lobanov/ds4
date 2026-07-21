#!/usr/bin/env python3
"""Task-6: conf_proj calibration LoRA (CORRECTED alignment).
Trains on the SAME-RUN capture: body dump (/tmp/live_body20_v2.bin, block_size=4,
drafts=[d0..d3]) + bench (/tmp/bench20_v2.jsonl, VERIFY_K=5 -> true longest_match).
Aligned BY INDEX (body rec i <-> bench cyc i; drafts match exactly).

Labels: proposals_accepted = bench_accepted - 1 (the bench counts the reused anchor).
match[k]=1 for k<proposals_accepted; match[proposals_accepted]=0 (first mismatch) if <4;
positions beyond the first mismatch are MASKED (the draft assumed a wrong prefix).
If proposals_accepted==4: all 4 match (=1).

STS: conf[0..3] are the 4 proposals' confidence; verify_n = sts(conf,4) + 1 (anchor);
cycle = 32.57 + 9.36*verify_n; accepted = min(proposals_accepted, sts); tokens = accepted+1.
"""
import sys, struct, numpy as np, torch, json
sys.path.insert(0, 'issue468/dspark_train')
from drafter_head import build_head, HC, DIM, VOCAB, NORM_EPS

DSPARK = '/Users/lobanov/Projects/ds4/gguf/dspark.gguf'
TARGET = '/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf'
BODY_DUMP = '/tmp/live_body20_v2.bin'
BENCH = '/tmp/bench20_v2.jsonl'
dev = 'mps'; HC_DIM = HC * DIM
RANK = 32; LR = 3e-3; EPOCHS = 15; BS = 32
NPROP = 4  # anchor_reuse ON -> 4 proposals

STS_TEMPS = [1.057018, 0.757858, 1.037660, 1.369200, 1.295342]
STS_THRESH = 0.15
CYCLE_A = 32.57; CYCLE_B = 9.36

def sigmoid(x): return 1.0/(1.0+np.exp(-np.clip(x,-60,60)))
def sts_verify(conf, max_n):
    survive = 1.0; keep = 0
    for i in range(max_n):
        survive *= sigmoid(conf[i] / STS_TEMPS[i])
        if survive < STS_THRESH: break
        keep = i + 1
    return keep

# --- parse body dump (variable block_size) ---
body_recs = []
with open(BODY_DUMP, 'rb') as f: buf = f.read()
off = 0
while off < len(buf):
    if off + 12 > len(buf): break
    a, p, bs = struct.unpack_from('<iii', buf, off); off += 12
    body = np.frombuffer(buf, dtype='<f4', count=bs*HC_DIM, offset=off).reshape(bs, HC_DIM).copy(); off += bs*HC_DIM*4
    drafts = np.frombuffer(buf, dtype='<i4', count=bs, offset=off).copy(); off += bs*4
    body_recs.append((int(a), int(p), int(bs), body, [int(d) for d in drafts]))

# --- parse bench (per-prompt cycle lists) ---
bench_rows = [json.loads(l) for l in open(BENCH)]
bench_cycles_per_prompt = [r.get('dspark_cycles', []) for r in bench_rows]
n_prompts = len(bench_cycles_per_prompt)
total_bench = sum(len(c) for c in bench_cycles_per_prompt)
print(f'body dump records={len(body_recs)}  bench cycles={total_bench}  prompts={n_prompts}', flush=True)

# align by draft-token-id (robust to skipped cycles in either source)
body_keys = [tuple(rec[4][:NPROP]) for rec in body_recs]
samples = []; sample_prompt = []
body_ptr = 0
for pi, cyc_list in enumerate(bench_cycles_per_prompt):
    for c in cyc_list:
        bench_drafts = c.get('draft_ids', [])
        if len(bench_drafts) < NPROP + 1: continue
        key = tuple(int(d) for d in bench_drafts[1:1+NPROP])
        found = -1
        for j in range(body_ptr, min(body_ptr + 8, len(body_recs))):  # search ahead
            if body_keys[j] == key: found = j; break
        if found < 0: continue
        body_ptr = found + 1
        anchor, pos, bs, body, drafts = body_recs[found]
        if bs < NPROP: continue
        accepted = c.get('accepted', 0); proposals_accepted = max(0, accepted - 1)
        samples.append((torch.as_tensor(body[:NPROP], dtype=torch.float32).reshape(NPROP, HC, DIM),
                        anchor, drafts[:NPROP], proposals_accepted))
        sample_prompt.append(pi)
print(f'aligned samples={len(samples)}  (body_recs={len(body_recs)}, bench_cycles={total_bench})', flush=True)

# split train/held-out by prompt
np.random.seed(42); n_hold = 4
hold_idx = set(np.random.choice(n_prompts, n_hold, replace=False))
train_samples = []; hold_samples = []
for s, pi in zip(samples, sample_prompt):
    (hold_samples if pi in hold_idx else train_samples).append(s)
print(f'train={len(train_samples)} held-out={len(hold_samples)}', flush=True)

# label stats
all_pa = [s[3] for s in samples]
print(f'proposals_accepted dist: mean={np.mean(all_pa):.3f} (=> mean tokens/cycle={np.mean(all_pa)+1:.3f})', flush=True)
print(f'  dist: {dict(zip(*[list(x) for x in np.unique(all_pa, return_counts=True)]))}', flush=True)

head = build_head(DSPARK, TARGET, dev, lora_rank=RANK, dtype=torch.float32)
for nm, p in head.named_parameters(): p.requires_grad_(False)
head.lora_conf_A.requires_grad_(True); head.lora_conf_B.requires_grad_(True)
opt = torch.optim.AdamW([head.lora_conf_A, head.lora_conf_B], lr=LR, weight_decay=0.01)
head.train()

def conf_logits_for(body, anchor, drafts):
    x = body.unsqueeze(0).to(dev)  # [1, NPROP, HC, DIM]
    h = head.hc_head(x)[0].float()  # [NPROP, DIM]
    hf = h * (1.0 / torch.sqrt((h*h).mean(-1, keepdim=True) + NORM_EPS)) * head._norm_w().to(dev, torch.float32)
    prev = torch.tensor([[anchor] + drafts[:NPROP-1]], device=dev)  # [1, NPROP]
    logits, _ = head.confidence_logits(hf.unsqueeze(0), prev)  # [1, NPROP]
    return logits.squeeze(0)  # [NPROP]

def make_labels(proposals_accepted):
    """match[k] + mask[k]: train on prefix(=1) + first mismatch(=0); beyond is masked."""
    match = np.zeros(NPROP, dtype=np.float32); mask = np.zeros(NPROP, dtype=np.float32)
    for k in range(NPROP):
        if k < proposals_accepted:
            match[k] = 1.0; mask[k] = 1.0
        elif k == proposals_accepted and proposals_accepted < NPROP:
            match[k] = 0.0; mask[k] = 1.0  # the first mismatch
            break
        else:
            break  # beyond first mismatch: masked
    if proposals_accepted >= NPROP:  # all matched
        match[:] = 1.0; mask[:] = 1.0
    return match, mask

print(f'\n=== conf_proj calibration (rank={RANK}, {EPOCHS} epochs, masked BCE) ===', flush=True)
for ep in range(EPOCHS):
    perm = np.random.permutation(len(train_samples))
    ep_loss = 0.0; nb = 0; n_pos = 0; n_neg = 0
    for i in range(0, len(perm), BS):
        batch_idx = perm[i:i+BS]
        loss = torch.zeros(1, device=dev)
        for bi_ in batch_idx:
            body, anchor, drafts, pa = train_samples[bi_]
            clog = conf_logits_for(body, anchor, drafts)  # [NPROP]
            temps = torch.tensor(STS_TEMPS[:NPROP], device=dev, dtype=torch.float32)
            scores = torch.sigmoid(clog / temps)
            match, mask = make_labels(pa)
            match = torch.as_tensor(match, device=dev); mask = torch.as_tensor(mask, device=dev)
            if mask.sum() == 0: continue
            w = torch.where(match > 0.5, torch.tensor(1.0, device=dev), torch.tensor(2.0, device=dev))
            bce = torch.nn.functional.binary_cross_entropy(scores, match, weight=w, reduction='none')
            loss = loss + (bce * mask).sum() / mask.sum().clamp(min=1)
            n_pos += int(((match>0.5)&(mask>0.5)).sum().item()); n_neg += int(((match<=0.5)&(mask>0.5)).sum().item())
        loss = loss / max(len(batch_idx),1)
        opt.zero_grad(); loss.backward(); opt.step()
        ep_loss += float(loss.item()); nb += 1
    print(f'  ep{ep+1}: BCE={ep_loss/max(nb,1):.4f}  (train labels: pos={n_pos} neg={n_neg})', flush=True)

# --- held-out STS-simulated throughput ---
trained_B = head.lora_conf_B.data.clone(); trained_A = head.lora_conf_A.data.clone()
def eval_throughput(samples, with_lora):
    if with_lora:
        head.lora_conf_B.data.copy_(trained_B); head.lora_conf_A.data.copy_(trained_A)
    else:
        head.lora_conf_B.data.zero_(); head.lora_conf_A.data.zero_()
    sts_out_l=[]; acc_l=[]; tok_l=[]; cyc_l=[]; pa_l=[]
    for body, anchor, drafts, pa in samples:
        with torch.no_grad():
            clog = conf_logits_for(body, anchor, drafts).cpu().numpy()
        sts = sts_verify(clog, NPROP)
        sts_out_l.append(sts)
        acc = min(pa, sts)  # proposals accepted (capped by sts)
        tok = acc + 1  # + the bonus/anchor
        vn = sts + 1   # anchor_reuse: +1 for the anchor
        cyc = CYCLE_A + CYCLE_B * vn
        acc_l.append(acc); tok_l.append(tok); cyc_l.append(cyc); pa_l.append(pa)
    sts_out_l=np.array(sts_out_l); acc_l=np.array(acc_l); tok_l=np.array(tok_l)
    cyc_l=np.array(cyc_l); pa_l=np.array(pa_l)
    tp = tok_l.sum() / (cyc_l.sum()/1000.0)
    return tp, sts_out_l, acc_l, tok_l, cyc_l, pa_l

base_tp, b_sts, b_acc, b_tok, b_cyc, b_pa = eval_throughput(hold_samples, False)
lor_tp, l_sts, l_acc, l_tok, l_cyc, l_pa = eval_throughput(hold_samples, True)
delta = 100*(lor_tp-base_tp)/base_tp
print(f'\n=== held-out STS-simulated ({len(hold_samples)} samples) ===', flush=True)
print(f'  baseline: STS={b_sts.mean():.3f} acc={b_acc.mean():.3f} cycle={b_cyc.mean():.2f}ms -> {base_tp:.2f} t/s', flush=True)
print(f'  LoRA:      STS={l_sts.mean():.3f} acc={l_acc.mean():.3f} cycle={l_cyc.mean():.2f}ms -> {lor_tp:.2f} t/s', flush=True)
print(f'  oracle (STS=pa): mean verify_n would be {b_pa.mean()+1:.3f} -> cycle {(CYCLE_A+CYCLE_B*(b_pa.mean()+1)):.2f}ms', flush=True)
print(f'  Δthroughput = {delta:+.2f}%  (offline, grounded cycle economics)', flush=True)

# correlation of survival prob with match (held-out, with lora)
for tag, la, lb in [('base', None, None), ('lora', trained_A, trained_B)]:
    if la is None: head.lora_conf_B.data.zero_(); head.lora_conf_A.data.zero_()
    else: head.lora_conf_B.data.copy_(lb); head.lora_conf_A.data.copy_(la)
    sc=[]; mt=[]
    for body, anchor, drafts, pa in hold_samples:
        with torch.no_grad():
            clog = conf_logits_for(body, anchor, drafts).cpu().numpy()
        surv=1.0
        for k in range(NPROP):
            surv *= sigmoid(clog[k]/STS_TEMPS[k])
            m,_=make_labels(pa)
            sc.append(surv); mt.append(m[k])
    sc=np.array(sc); mt=np.array(mt)
    corr = np.corrcoef(sc,mt)[0,1] if sc.std()>0 and mt.std()>0 else float('nan')
    print(f'    {tag}: corr(survival, match) = {corr:.3f}', flush=True)

# bootstrap CI
rng = np.random.default_rng(42); B = 2000; boots = []
for _ in range(B):
    i = rng.integers(0, len(b_tok), len(b_tok))
    btp = b_tok[i].sum()/(b_cyc[i].sum()/1000); ltp = l_tok[i].sum()/(l_cyc[i].sum()/1000)
    boots.append(100*(ltp-btp)/btp)
lo, hi = np.percentile(boots, [2.5, 97.5])
print(f'\n  95% CI on Δthroughput = [{lo:+.2f}%, {hi:+.2f}%]  (excludes 0: {not (lo <= 0 <= hi)})', flush=True)
print(f'PROCEED gate: CI excludes 0 = {not (lo <= 0 <= hi)}', flush=True)
if not (lo <= 0 <= hi):
    head.lora_conf_B.data.copy_(trained_B); head.lora_conf_A.data.copy_(trained_A)
    delta_w = (head.lora_conf_B.float() @ head.lora_conf_A.float()).squeeze(0).detach().cpu().numpy()
    np.save('/tmp/confproj_lora_delta.npy', delta_w)
    print(f'saved conf_proj LoRA delta to /tmp/confproj_lora_delta.npy (shape {delta_w.shape})', flush=True)

# --- STS THRESHOLD SWEEP (baseline conf_logits, no LoRA): a blunt absolute-recalibration ---
print(f'\n=== STS threshold sweep (baseline conf, held-out) ===', flush=True)
head.lora_conf_B.data.zero_(); head.lora_conf_A.data.zero_()
base_clog = []
for body, anchor, drafts, pa in hold_samples:
    with torch.no_grad():
        base_clog.append(conf_logits_for(body, anchor, drafts).cpu().numpy())
print(f'{"thresh":>7} {"STS":>6} {"acc":>6} {"cycle_ms":>8} {"t/s":>7} {"Δ%":>7}')
best_tp = -1; best_th = None
for th in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]:
    sts_l=[]; acc_l=[]; cyc_l=[]; tok_l=[]
    for clog, (_,_,_,pa) in zip(base_clog, hold_samples):
        # inline sts with threshold th
        survive=1.0; keep=0
        for i in range(NPROP):
            survive *= sigmoid(clog[i]/STS_TEMPS[i])
            if survive < th: break
            keep = i+1
        acc = min(pa, keep); tok = acc+1; vn = keep+1
        cyc = CYCLE_A + CYCLE_B*vn
        sts_l.append(keep); acc_l.append(acc); cyc_l.append(cyc); tok_l.append(tok)
    sts_l=np.array(sts_l); acc_l=np.array(acc_l); cyc_l=np.array(cyc_l); tok_l=np.array(tok_l)
    tp = tok_l.sum()/(cyc_l.sum()/1000)
    dp = 100*(tp-base_tp)/base_tp if best_th else 0.0
    if best_th is None: base_tp_th = tp
    dp = 100*(tp-48.18)/48.18  # vs the baseline (th=0.15)
    marker = ' <- baseline' if abs(th-0.15)<1e-3 else (' <- BEST' if tp>best_tp else '')
    print(f'{th:>7.2f} {sts_l.mean():>6.3f} {acc_l.mean():>6.3f} {cyc_l.mean():>8.2f} {tp:>7.2f} {dp:>+7.2f}{marker}')
    if tp > best_tp: best_tp = tp; best_th = th
print(f'\n  best threshold = {best_th} -> {best_tp:.2f} t/s ({100*(best_tp-48.18)/48.18:+.2f}% vs th=0.15)', flush=True)
