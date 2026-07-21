#!/usr/bin/env python3
"""Task-7: hc_fn draft-quality LoRA via REINFORCE on the THROUGHPUT reward.
Uses the SAME-RUN body dump v2 (block_size=4) + the sel reconstructed from bench20_v2
(no cross-run alignment drift). The conf_proj is the calibrated rank1/3ep LoRA (loaded)
so the verify_n reflects the task-6 deployment. Reward = (accepted+1) / cycle, where
cycle = draft(6.5) + verify(verify_n), verify_n = sts(conf_logits)+1.

The hc_fn affects the drafts -> the accepted (longest_match) AND the conf_logits
(via prev_tok) -> the verify_n. REINFORCE captures the end-to-end throughput.
"""
import sys, struct, numpy as np, torch, json, os
sys.path.insert(0, 'issue468/dspark_train')
from drafter_head import build_head, HC, DIM, VOCAB, NORM_EPS

DSPARK = '/Users/lobanov/Projects/ds4/gguf/dspark.gguf'
TARGET = '/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf'
BODY_DUMP = '/tmp/live_body20_v2.bin'
BENCH = '/tmp/bench20_v2.jsonl'
CONFPROJ_DELTA = '/tmp/confproj_lora_delta.npy'  # the calibrated rank1/3ep delta
dev = 'mps'; HC_DIM = HC * DIM
RANK = int(os.environ.get('HRANK', '32')); LR = float(os.environ.get('HLR', '1e-3'))
EPOCHS = int(os.environ.get('HEP', '8')); BS = 16; SAMPLES_PER = 2
TEMP_SAMPLING = float(os.environ.get('HTMP', '0.7'))  # sampling temperature for exploration (eval is greedy)
NPROP = 4

STS_TEMPS = [1.057018, 0.757858, 1.037660, 1.369200, 1.295342]; STS_THRESH = 0.15
CYCLE_A = 32.57; CYCLE_B = 9.36; DRAFT_MS = 6.5

def sigmoid(x): return 1.0/(1.0+np.exp(-np.clip(x,-60,60)))
def sts_verify_t(clog_t, max_n):
    survive = torch.ones((), device=clog_t.device); keep = 0
    temps = torch.tensor(STS_TEMPS[:max_n], device=clog_t.device, dtype=clog_t.dtype)
    for i in range(max_n):
        survive = survive * torch.sigmoid(clog_t[i] / temps[i])
        if float(survive) < STS_THRESH: break
        keep = i + 1
    return keep

# --- parse body dump ---
body_recs = []
with open(BODY_DUMP, 'rb') as f: buf = f.read()
off = 0
while off < len(buf):
    if off + 12 > len(buf): break
    a, p, bs = struct.unpack_from('<iii', buf, off); off += 12
    body = np.frombuffer(buf, dtype='<f4', count=bs*HC_DIM, offset=off).reshape(bs, HC_DIM).copy(); off += bs*HC_DIM*4
    drafts = np.frombuffer(buf, dtype='<i4', count=bs, offset=off).copy(); off += bs*4
    body_recs.append((int(a), int(p), int(bs), body, [int(d) for d in drafts]))

# --- parse bench + reconstruct sel per prompt ---
bench_rows = [json.loads(l) for l in open(BENCH)]
body_keys = [tuple(rec[4][:NPROP]) for rec in body_recs]

def reconstruct_sel(cycles):
    sel = []
    for ci, c in enumerate(cycles):
        drafts = c.get('draft_ids', []); acc = c.get('accepted', 0)
        if acc > 0:
            for k in range(min(acc, len(drafts))):
                idx = len(sel)
                if idx < len(sel) + acc - ci: pass
                break
        # commit accepted drafts[0..acc-1] then the corrected (next anchor)
        for k in range(acc):
            if len(sel) < 128 * 4:  # safety
                # only extend if at the frontier
                pass
        break
    # simpler correct walk: running pos, fill sel
    sel_map = {}
    pos = 0
    for ci, c in enumerate(cycles):
        drafts = c.get('draft_ids', []); acc = c.get('accepted', 0)
        for k in range(min(acc, len(drafts))):
            sel_map[pos + k] = drafts[k]
        if ci + 1 < len(cycles):
            sel_map[pos + acc] = cycles[ci+1].get('draft_ids', [None])[0]
        pos += acc
    max_pos = max(sel_map.keys()) if sel_map else -1
    return [sel_map[i] for i in range(max_pos + 1)] if max_pos >= 0 else []

sels = [reconstruct_sel(r.get('dspark_cycles', [])) for r in bench_rows]

# --- build samples: (body[NPROP,HC,DIM], anchor, sel_window[NPROP+1]) ---
samples = []; sample_sel = []
body_ptr = 0
for pi, cyc_list in enumerate([r.get('dspark_cycles', []) for r in bench_rows]):
    sel = sels[pi]
    running = 0  # running pos in sel
    for c in cyc_list:
        bench_drafts = c.get('draft_ids', [])
        if len(bench_drafts) < NPROP + 1: continue
        key = tuple(int(d) for d in bench_drafts[1:1+NPROP])
        found = -1
        for j in range(body_ptr, min(body_ptr + 8, len(body_recs))):
            if body_keys[j] == key: found = j; break
        if found < 0:
            running += c.get('accepted', 0); continue
        body_ptr = found + 1
        anchor, bpos, bs, body, drafts = body_recs[found]
        if bs < NPROP: running += c.get('accepted', 0); continue
        # the sel window: drafts propose sel[running+1..running+NPROP]; anchor = sel[running]
        window = []
        for k in range(NPROP + 1):
            idx = running + k
            window.append(sel[idx] if 0 <= idx < len(sel) else -1)
        if window[0] != anchor or any(w < 0 for w in window):
            running += c.get('accepted', 0); continue
        samples.append((torch.as_tensor(body[:NPROP], dtype=torch.float32).reshape(NPROP, HC, DIM), anchor))
        sample_sel.append(window[1:])  # the NPROP sel targets
        running += c.get('accepted', 0)
print(f'samples={len(samples)}  (body_recs={len(body_recs)})', flush=True)
# sanity: the greedy drafts should match the sel window heavily
greedy_match = []
for (body, anchor), sel_w in zip(samples, sample_sel):
    # greedy draft via the head (computed below) -- skip for now, use the body rec drafts
    pass

np.random.seed(42); n_hold = 4; n_prompts = len(bench_rows)
hold_idx = set(np.random.choice(n_prompts, n_hold, replace=False))
# assign prompt by approximate quantile
train_samples=[]; train_sel=[]; hold_samples=[]; hold_sel=[]
for si, (s, sw) in enumerate(zip(samples, sample_sel)):
    pi = min(n_prompts-1, si * n_prompts // max(len(samples),1))
    (hold_samples if pi in hold_idx else train_samples).append(s)
    (hold_sel if pi in hold_idx else train_sel).append(sw)
print(f'train={len(train_samples)} held-out={len(hold_samples)}', flush=True)

head = build_head(DSPARK, TARGET, dev, lora_rank=RANK, dtype=torch.float32)
# load the calibrated conf_proj LoRA (task-6 deployment) + FREEZE it
if os.path.exists(CONFPROJ_DELTA):
    d = torch.as_tensor(np.load(CONFPROJ_DELTA), dtype=torch.float32)
    # conf_proj LoRA = lora_conf_B(1,r) @ lora_conf_A(r, 4352) = d(4352). Set A=I_row, B=d_row.
    head.lora_conf_A.data.copy_(torch.eye(RANK)[:, :d.shape[0]] if RANK >= d.shape[0] else torch.randn(RANK, d.shape[0])*0.02)
    head.lora_conf_B.data.zero_()
    # simpler: directly absorb into conf_proj buffer (freeze)
    head.conf_proj.add_(d.to(head.conf_proj.device).to(head.conf_proj.dtype))
    head.lora_conf_A.requires_grad_(False); head.lora_conf_B.requires_grad_(False)
    print(f'loaded calibrated conf_proj delta (‖d‖={float(d.norm()):.4f}) into the frozen conf_proj', flush=True)
# train ONLY hc_fn LoRA
for nm, p in head.named_parameters(): p.requires_grad_(False)
head.lora_hc_A.requires_grad_(True); head.lora_hc_B.requires_grad_(True)
opt = torch.optim.AdamW([head.lora_hc_A, head.lora_hc_B], lr=LR, weight_decay=0.01)
mw2_T = head._mw2_T(torch.float32).detach(); lm_head = head.lm_head
head.train()

def rollout(body, anchor, sample=True):
    """Autoregressive rollout. Returns drafts[NPROP], logpi, h(norm), prev_toks."""
    x = body.unsqueeze(0).to(dev)
    h = head.hc_head(x)[0].float()
    hf = h * (1.0/torch.sqrt((h*h).mean(-1,keepdim=True)+NORM_EPS)) * head._norm_w().to(dev,torch.float32)
    base = hf.to(torch.float32) @ lm_head.T  # [NPROP, VOCAB]
    prev = torch.tensor([anchor], device=dev)
    drafts = []; logpi = 0.0; prev_toks = [anchor]
    for k in range(NPROP):
        emb = head._mw1(prev).to(dev, torch.float32)
        logits = (base[k] + emb @ mw2_T) / (TEMP_SAMPLING if sample else 1.0)
        logp = torch.log_softmax(logits, -1)
        if sample: d = torch.multinomial(torch.exp(logp), 1).squeeze(0)
        else: d = logits.argmax(-1)
        logpi = logpi + logp[0, d]
        drafts.append(int(d.item())); prev = d; prev_toks.append(int(d.item()))
    return drafts, logpi, hf, prev_toks[:NPROP]

def reward_of(drafts, hf, prev_toks, sel_w):
    """accepted + verify_n + cycle, differentiable-friendly (returns scalars)."""
    accepted = 0
    for k in range(NPROP):
        if drafts[k] == sel_w[k]: accepted += 1
        else: break
    with torch.no_grad():
        prev = torch.tensor([prev_toks], device=dev)
        clog, _ = head.confidence_logits(hf.unsqueeze(0), prev)
        clog = clog.squeeze(0)
        sts = sts_verify_t(clog, NPROP)
    verify_n = sts + 1
    cycle = DRAFT_MS + CYCLE_A + CYCLE_B * verify_n
    tokens = accepted + 1
    return tokens, cycle, accepted, verify_n

# baseline (EMA)
baseline = None; BETA = 0.9
print(f'\n=== REINFORCE on throughput (rank={RANK}, {EPOCHS} epochs, temp={TEMP_SAMPLING}) ===', flush=True)
for ep in range(EPOCHS):
    perm = np.random.permutation(len(train_samples))
    ep_R = []; nb = 0
    for i in range(0, len(perm), BS):
        batch_idx = perm[i:i+BS]
        logpis = []; Rs = []
        for bi in batch_idx:
            body, anchor = train_samples[bi]; sel_w = train_sel[bi]
            for _ in range(SAMPLES_PER):
                drafts, logpi, hf, prev_toks = rollout(body, anchor, sample=True)
                tokens, cycle, acc, vn = reward_of(drafts, hf, prev_toks, sel_w)
                R = tokens / (cycle / 1000.0)  # t/s
                logpis.append(logpi); Rs.append(R)
        Rs_t = torch.tensor(Rs, dtype=torch.float32, device=dev)
        if baseline is None: baseline = Rs_t.mean().item()
        adv = Rs_t - baseline
        logpis_t = torch.stack(logpis)
        loss = -(adv * logpis_t).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        ep_R.extend(Rs); baseline = BETA*baseline + (1-BETA)*np.mean(Rs); nb += 1
    print(f'  ep{ep+1}: mean reward(train,sampled)={np.mean(ep_R):.3f} t/s  baseline={baseline:.3f}', flush=True)

# held-out eval: greedy throughput with vs without LoRA
trained_B = head.lora_hc_B.data.clone(); trained_A = head.lora_hc_A.data.clone()
def eval_greedy(samples, sel_w_list, with_lora):
    if with_lora:
        head.lora_hc_B.data.copy_(trained_B); head.lora_hc_A.data.copy_(trained_A)
    else:
        head.lora_hc_B.data.zero_(); head.lora_hc_A.data.zero_()
    acc_l=[]; tok_l=[]; cyc_l=[]; vn_l=[]
    for (body, anchor), sel_w in zip(samples, sel_w_list):
        with torch.no_grad():
            drafts, _, hf, prev_toks = rollout(body, anchor, sample=False)
            tokens, cycle, acc, vn = reward_of(drafts, hf, prev_toks, sel_w)
        acc_l.append(acc); tok_l.append(tokens); cyc_l.append(cycle); vn_l.append(vn)
    acc_l=np.array(acc_l); tok_l=np.array(tok_l); cyc_l=np.array(cyc_l); vn_l=np.array(vn_l)
    tp = tok_l.sum()/(cyc_l.sum()/1000.0)
    return tp, acc_l, vn_l, cyc_l
base_tp, b_acc, b_vn, b_cyc = eval_greedy(hold_samples, hold_sel, False)
lor_tp, l_acc, l_vn, l_cyc = eval_greedy(hold_samples, hold_sel, True)
print(f'\n=== held-out ({len(hold_samples)} samples) ===')
print(f'baseline: E[acc]={b_acc.mean():.3f} verify_n={b_vn.mean():.3f} cycle={b_cyc.mean():.2f}ms -> {base_tp:.2f} t/s')
print(f'LoRA:      E[acc]={l_acc.mean():.3f} verify_n={l_vn.mean():.3f} cycle={l_cyc.mean():.2f}ms -> {lor_tp:.2f} t/s')
d = 100*(lor_tp-base_tp)/base_tp
print(f'Δthroughput = {d:+.2f}%  (offline)')
# bootstrap CI
rng=np.random.default_rng(42); B=2000; boots=[]
bt=b_acc+1; bc=b_cyc; lt=l_acc+1; lc=l_cyc
for _ in range(B):
    i=rng.integers(0,len(bt),len(bt))
    btp=bt[i].sum()/(bc[i].sum()/1000); ltp=lt[i].sum()/(lc[i].sum()/1000)
    boots.append(100*(ltp-btp)/btp)
lo,hi=np.percentile(boots,[2.5,97.5])
print(f'95% CI = [{lo:+.2f}%, {hi:+.2f}%]  (excludes 0: {not (lo<=0<=hi)})')
print(f'PROCEED gate (bake+live): CI excludes 0 = {not (lo<=0<=hi)}')
if not (lo<=0<=hi):
    delta_w=(head.lora_hc_B.float()@head.lora_hc_A.float()).detach().cpu().numpy()
    np.save('/tmp/hcfn_lora_delta.npy', delta_w)
    print(f'saved hc_fn LoRA delta (shape {delta_w.shape})')
