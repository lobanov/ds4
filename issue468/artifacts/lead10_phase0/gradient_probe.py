#!/usr/bin/env python3
"""Lead 10 Phase-0: gradient-magnitude probe — rank the dense LoRA candidates.

Loads the D_f32 drafter at F16 (experts fit), enables grad on ALL dense weights (experts
frozen), accumulates the CE loss vs the IQ2 greedy (sel, from lead3_h.bin's tok field) over
~40 prompts (max_step=8 each to fit the MoE backward activation retention), and ranks the
dense candidates by ‖∇W‖/‖W‖ (depth-normalized "want to move").

NOTE: uses hard-label CE (vs sel) as the probe loss. The Lead-11 soft labels (IQ2 top-128)
were dropped in the Exp-0a consolidation — so KL is unavailable for the probe. CE is a valid
proxy for the gradient RANKING (both push toward the IQ2 distribution); the training (Phase 1)
will use KL once the soft labels are re-captured.
"""
import sys, struct, json, numpy as np, torch
sys.path.insert(0, 'issue468/dspark_train'); sys.path.insert(0, 'issue468/dspark_oracle')
from drafter_body import build_body, BLOCK
from drafter_head import build_head

DSPARK = '/Users/lobanov/Projects/ds4/gguf/dspark.gguf'
TARGET = '/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf'
CAP_H = 'issue468/artifacts/lead11_unified_capture/lead3_h.bin'
MAX_STEP = 3
N_PROMPTS = 60

# target-type mapping (per the inventory)
LAYER_KEYS_ATTN = ['q_a','q_a_norm','q_b','kv','kv_a_norm','attn_sinks','output_a','output_b','attn_norm']
LAYER_KEYS_HC_ATTN = ['hc_attn_fn','hc_attn_scale','hc_attn_base']
LAYER_KEYS_HC_FFN = ['hc_ffn_fn','hc_ffn_scale','hc_ffn_base']
LAYER_KEYS_ROUTER = ['ffn_gate_inp','ffn_exp_probs_b']
LAYER_KEYS_SHARED = ['ffn_gate_shexp','ffn_up_shexp','ffn_down_shexp']
LAYER_KEYS_FFN_NORM = ['ffn_norm']
def layer_key_type(k):
    if k in LAYER_KEYS_ATTN: return 'attn'
    if k in LAYER_KEYS_HC_ATTN: return 'hc_attn'
    if k in LAYER_KEYS_HC_FFN: return 'hc_ffn'
    if k in LAYER_KEYS_ROUTER: return 'router'
    if k in LAYER_KEYS_SHARED: return 'shared_expert'
    if k in LAYER_KEYS_FFN_NORM: return 'ffn_norm'
    return f'other:{k}'

# --- parse h.bin -> per-prompt (H + sel), pos-sorted ---
prompts = {}
with open(CAP_H, 'rb') as f: data = f.read()
off = 0
while off < len(data):
    if off + 4 > len(data): break
    (idl,) = struct.unpack_from('<i', data, off); off += 4
    pid = data[off:off+idl].decode(); off += idl
    (pos, tok) = struct.unpack_from('<ii', data, off); off += 8
    h = np.frombuffer(data, dtype='<f4', count=12288, offset=off); off += 12288 * 4
    prompts.setdefault(pid, {'pos': [], 'tok': [], 'h': []})
    prompts[pid]['pos'].append(pos); prompts[pid]['tok'].append(tok); prompts[pid]['h'].append(h)
for pid in prompts:
    o = np.argsort(prompts[pid]['pos'])
    prompts[pid]['h'] = np.stack(prompts[pid]['h'])[o]
    prompts[pid]['tok'] = [prompts[pid]['tok'][i] for i in o]
pids = sorted(prompts.keys())[:N_PROMPTS]
print(f'{len(prompts)} prompts in capture; probing {len(pids)} (max_step={MAX_STEP})', flush=True)

dev = 'mps'; dt = torch.float32  # F32: F16 backward NaNs through the MoE; F32 is stable + fits at small max_step
print('loading drafter F16 (MPS)...', flush=True)
body = build_body(DSPARK, TARGET, dev, dtype=dt)
head = build_head(DSPARK, TARGET, dev, lora_rank=0, dtype=dt)
print('loaded.', flush=True)

# enable grad on ALL dense weights (experts exp_gate/up/down stay frozen; embed/cos/sin frozen)
dense = []  # (name, tensor, type, layer)
body.main_proj.requires_grad_(True); dense.append(('main_proj', body.main_proj, 'main_proj', '-'))
body.main_norm.requires_grad_(True); dense.append(('main_norm', body.main_norm, 'main_norm', '-'))
for s in range(3):
    for k, v in body.layers[s].items():
        v.requires_grad_(True); dense.append((f'L{s}_{k}', v, layer_key_type(k), str(s)))
for name, buf in head.named_buffers():
    buf.requires_grad_(True); dense.append((f'head.{name}', buf, f'head:{name}', '-'))

# accumulate gradients over prompts (CE vs sel, mean)
for i, pid in enumerate(pids):
    H = prompts[pid]['h']; tt = prompts[pid]['tok']
    N = H.shape[0]; ms = min(MAX_STEP, N - BLOCK - 1, N - 1)
    if ms < 1: continue
    mh_seq = H[:ms+1]; anchors = [int(tt[s]) for s in range(1, ms+1)]
    xs = body.forward_prompt(mh_seq, anchors)
    out, base = head(xs, torch.tensor(anchors, device=dev))
    bias = head._mw1(torch.tensor(anchors, device=dev)) @ head._mw2_T(dt)
    ftl = base[:, 0, :] + bias
    target = torch.tensor([int(tt[s+1]) for s in range(1, ms+1)], device=dev)
    loss = torch.nn.functional.cross_entropy(ftl.float(), target) / len(pids)
    loss.backward()
    del xs, out, base, bias, ftl, target, loss
    if (i+1) % 10 == 0: print(f'  {i+1}/{len(pids)} prompts done', flush=True)

# collect ‖∇W‖/‖W‖ per weight + aggregate by type/layer
per_weight = []
for name, W, typ, lyr in dense:
    if W.grad is None: continue
    gw = float(W.grad.float().norm().item()); ww = float(W.float().norm().item())
    per_weight.append({'name': name, 'type': typ, 'layer': lyr, 'grad_norm': gw, 'weight_norm': ww, 'ratio': gw / (ww + 1e-12)})
per_weight.sort(key=lambda r: -r['ratio'])

# aggregate by type (mean ratio over the type's weights × layers)
by_type = {}
for r in per_weight:
    by_type.setdefault(r['type'], []).append(r['ratio'])
type_rank = sorted(((t, float(np.mean(rs)), len(rs)) for t, rs in by_type.items()), key=lambda x: -x[1])

print('\n' + '=' * 64)
print('PHASE-0 GRADIENT PROBE — dense LoRA candidates ranked by ‖∇W‖/‖W‖ (mean over weights+layers)')
print('=' * 64)
print(f'{"target-type":<16} {"mean ‖∇W‖/‖W‖":>16} {"n weights":>10}')
for t, m, n in type_rank:
    print(f'{t:<16} {m:>16.5f} {n:>10d}')
print('\ntop 15 individual weights:')
for r in per_weight[:15]:
    print(f'  {r["name"]:<22} ratio={r["ratio"]:.5f} (‖∇W‖={r["grad_norm"]:.4f} ‖W‖={r["weight_norm"]:.2f})')

out = {'n_prompts': len(pids), 'max_step': MAX_STEP, 'loss': 'CE vs sel (hard-label proxy)',
       'type_rank': [{'type': t, 'mean_ratio': m, 'n': n} for t, m, n in type_rank],
       'per_weight': per_weight}
import os; os.makedirs('issue468/artifacts/lead10_phase0', exist_ok=True)
json.dump(out, open('issue468/artifacts/lead10_phase0/gradient_probe_result.json', 'w'), indent=2)
print(f'\nwrote issue468/artifacts/lead10_phase0/gradient_probe_result.json')
