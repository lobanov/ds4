#!/usr/bin/env python3
"""Lead 10 SC7: bake the trained head.hc_fn LoRA into the dspark GGUF.

Trains the head.hc_fn KL LoRA on all 60 lead3 prompts (the full set, for deployment),
computes the delta (lora_hc_B @ lora_hc_A), copies dspark.gguf -> dspark_lora.gguf, and
modifies the mtp.2.hc_head_fn tensor (F16, [4,16384], offset 11357395488, 131072 bytes)
+= the delta (cast to F16). The resulting dspark_lora.gguf is a drop-in replacement for
the live ds4 drafter.
"""
import sys, json, shutil, numpy as np, torch, struct
sys.path.insert(0, 'issue468/dspark_train'); sys.path.insert(0, 'issue468/dspark_oracle')
from drafter_head import build_head

DSPARK = '/Users/lobanov/Projects/ds4/gguf/dspark.gguf'
TARGET = '/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf'
DSPARK_LORA = '/Users/lobanov/Projects/ds4/gguf/dspark_lora.gguf'
FEAT = 'issue468/artifacts/lead10_phase1/lead3_features.npz'
RANK = 32; EPOCHS = 12; LR = 3e-4; WD = 0.01; BS = 256
HC_OFFSET = 11357395488  # the mtp.2.hc_head_fn.weight offset in the GGUF
HC_BYTES = 131072         # 4 * 16384 * 2 (F16)
dev = 'mps'

# --- train the head.hc_fn KL LoRA on ALL 60 ---
cz = np.load(FEAT, allow_pickle=True)
pids = list(cz['pids'])
feats = {pid: (cz[f'{pid}__x0'], cz[f'{pid}__a'], cz[f'{pid}__tids'], cz[f'{pid}__tlogp'], cz[f'{pid}__tgt']) for pid in pids}
xtr = torch.from_numpy(np.concatenate([feats[p][0] for p in pids], 0))
atr = torch.from_numpy(np.concatenate([feats[p][1] for p in pids], 0))
tids = torch.from_numpy(np.concatenate([feats[p][2] for p in pids], 0))
tlogp = torch.from_numpy(np.concatenate([feats[p][3] for p in pids], 0))
print(f'training head.hc_fn KL LoRA on all {len(pids)} prompts ({xtr.shape[0]} anchors)...', flush=True)
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
        dlp = torch.log_softmax(logits, -1)
        dlpt = torch.gather(dlp, 1, tids[idx].to(dev))
        tlp = torch.log_softmax(tlogp[idx].to(dev), -1)
        loss = (torch.exp(tlp) * (tlp - dlpt)).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item() * len(idx)
    print(f'  ep{ep+1:2d}: KL={tl/ntr:.4f}', flush=True)

# --- compute the delta ---
delta = (head.lora_hc_B.float() @ head.lora_hc_A.float()).detach().cpu().numpy()  # [4, 16384]
print(f'\nLoRA delta: shape {delta.shape} | ‖delta‖={np.linalg.norm(delta):.4f} | max|delta|={np.abs(delta).max():.4f}', flush=True)

from gguf_loader import index_gguf, read_tensor

# --- copy + bake ---
print(f'copying {DSPARK} -> {DSPARK_LORA}...', flush=True)
shutil.copy2(DSPARK, DSPARK_LORA)

# read the current hc_head_fn via the LOADER (the correct [4,16384] interpretation)
_, ginfos, gdoff = index_gguf(DSPARK)
hc_ne, hc_dt, hc_off = ginfos['mtp.2.hc_head_fn.weight']
hc_abs_off = gdoff + hc_off  # the ABSOLUTE file offset (read_tensor seeks to data_off + off)
print(f'hc_head_fn: ne={hc_ne} off={hc_off} data_off={gdoff} abs_off={hc_abs_off}', flush=True)
current = read_tensor(DSPARK, ginfos, gdoff, 'mtp.2.hc_head_fn.weight').astype(np.float32)  # [4,16384]
print(f'current hc_head_fn (via loader): shape {current.shape} | ‖current‖={np.linalg.norm(current):.4f}', flush=True)

# add the delta
modified = current + delta  # [4,16384] F32
print(f'modified: ‖modified‖={np.linalg.norm(modified):.4f} | max diff={np.abs(modified-current).max():.6f}', flush=True)

# write back: the C-order flat bytes (matches read_tensor's reshape(reversed(dims))) + F16
with open(DSPARK_LORA, 'r+b') as f:
    f.seek(hc_abs_off)
    f.write(modified.astype(np.float16).tobytes())
print(f'\nbaked the LoRA delta into {DSPARK_LORA} (correct offset + layout)', flush=True)

# verify: read back via the loader + check the delta is applied correctly
_, ginfos2, gdoff2 = index_gguf(DSPARK_LORA)
check = read_tensor(DSPARK_LORA, ginfos2, gdoff2, 'mtp.2.hc_head_fn.weight').astype(np.float32)
print(f'verify: ‖check - modified‖={np.linalg.norm(check - modified):.6f} (should be ~0)', flush=True)
print(f'verify: ‖check - current‖={np.linalg.norm(check - current):.4f} (should be ~‖delta‖={np.linalg.norm(delta):.4f})', flush=True)
print('\nNext: run ds4-spec-bench with --dspark dspark_lora.gguf on the lead3 spec config.')
