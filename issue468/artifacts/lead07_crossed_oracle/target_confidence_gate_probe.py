#!/usr/bin/env python3
"""Lead-10 / scheduling pre-experiment (existing captures): does the target's per-position
confidence (margin / entropy / top-1 prob) predict the drafter's first-draft rejection?

If YES  -> a target-entropy draft-gate is viable (skip the draft at low-confidence anchors),
           AND the drafter's failures are at "hard" (target-uncertain) positions => distillation
           (Lead 10) won't help much (the positions are intrinsically hard).
If NO   -> the gate is dead, AND the failures are drafter bugs at *confident* positions
           => distillation (Lead 10) might help.

Runs on the EXISTING FP captures (phaseB) — the only captures with the target's top-128.
Two drafter-input variants per prompt:
  (a) H_fp   judged vs Y_fp    -- the FP-native case (p1 ~0.85)
  (b) H_iq2_tf judged vs Y_iq2_tf -- the IQ2 drafter on the FP trajectory (Lead-07 baseline)
The target confidence (margin/entropy/top1) is the FP target's in both (the IQ2 target's topk
was never captured; the FP target's is the available proxy for position-difficulty).
"""
import json, sys, numpy as np, torch
sys.path.insert(0, 'issue468/dspark_train'); sys.path.insert(0, 'issue468/dspark_oracle')
from drafter_body import build_body, BLOCK, HC, DIM
from drafter_head import build_head
sys.path.insert(0, 'issue468/run_lead04_modal')
from convert_vllm_to_oracle import surgery_to_main_hidden

DSPARK='/Users/lobanov/Projects/ds4/gguf/dspark.gguf'
TARGET='/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf'
FP_DIR='issue468/artifacts/lead07_crossed_oracle/fp_captures_raw'
IQ2TF_DIR='issue468/artifacts/lead07_crossed_oracle/bundles_iq2tf'

def load_iq2tf(pid):
    oi=np.load(f'{IQ2TF_DIR}/{pid}/oracle/oracle_inputs.npz')
    y=json.load(open(f'{IQ2TF_DIR}/{pid}/target_selected_tokens.json'))  # = Y_fp (forced)
    ytf=[int(x) for x in open(f'issue468/artifacts/lead07_crossed_oracle/tf_dump_full/{pid}.iq2argmax').read().split()]
    return oi['main_hidden'].astype(np.float32), [int(t) for t in y], ytf

def drafter_drafts(mh, anchors_seq, body, head, dev):
    n_cap=mh.shape[0]; max_step=min(len(anchors_seq)-1,n_cap-1)
    if max_step<2: return None,None
    anc=[int(anchors_seq[s]) for s in range(1,max_step+1)]
    with torch.no_grad():
        xs=body.forward_prompt(mh[:max_step+1],anc); at=torch.tensor(anc,device=dev,dtype=torch.long)
        d=[]
        for i in range(0,xs.shape[0],24):
            out,_=head(xs[i:i+24],at[i:i+24]); d.append(out[:,1:].cpu().numpy())
    return np.concatenate(d,axis=0)[:,0],max_step  # first-draft per anchor, max_step

def stats(topk_lp, s):  # margin, top1prob, entropy at index s (predicts token s)
    lp=topk_lp[s]; p=np.exp(lp-lp.max()); p=p/p.sum()
    return float(lp[0]-lp[1]), float(p[0]), float(-(p*np.log(p+1e-12)).sum())

dev='mps' if torch.backends.mps.is_available() else 'cpu'
print(f'loading D_f32 drafter on {dev}...',flush=True)
body=build_body(DSPARK,TARGET,dev,dtype=torch.float32); head=build_head(DSPARK,TARGET,dev,lora_rank=0,dtype=torch.float32)
print('drafter loaded.',flush=True)

rows=[]
for src in ['codealpaca','dolly','jsonex']:
  for i in range(80,100):
    pid=f'{src}_{i:04d}'; fp=f'{FP_DIR}/phaseB_{pid}.npz'
    try:
      mh_fp,pos,y_fp,plen,_=surgery_to_main_hidden(fp)
      cap=np.load(fp,allow_pickle=True); topk_lp=cap['topk_logprobs']; greedy=cap['greedy_tokens']
      assert len(greedy)==len(y_fp)
      # NOTE: tf_dump_full/ (the teacher-force recapture) is no longer on disk, so only the
      # H_fp/Y_fp (FP-native) variant runs on existing captures. The H_iq2_tf variant would
      # need re-running the ds4 teacher_force capture (a new run, not existing).
    except Exception as e: print('skip',pid,e); continue
    for label,(mh,lab) in [('H_fp/Y_fp',(mh_fp,y_fp))]:
      drafts,max_step=drafter_drafts(mh,y_fp,body,head,dev)
      if drafts is None: continue
      for ii in range(len(drafts)):
        s=ii+1  # anchor; predicts token s+1=ii+2
        tgt=ii+2
        if tgt>=len(greedy) or tgt>=len(topk_lp): continue
        ok=int(drafts[ii]==lab[tgt])
        m,t1,H=stats(topk_lp,tgt)
        rows.append({'variant':label,'margin':m,'top1':t1,'entropy':H,'ok':ok})
print(f'\n{len(rows)} anchor-observations collected.')

import numpy as np
def analyze(variant):
    r=[x for x in rows if x['variant']==variant]
    ok=[x for x in r if x['ok']]; bad=[x for x in r if not x['ok']]
    margin_ok=np.array([x['margin'] for x in ok]); margin_bad=np.array([x['margin'] for x in bad])
    ent_ok=np.array([x['entropy'] for x in ok]); ent_bad=np.array([x['entropy'] for x in bad])
    print(f'\n=== {variant} (n={len(r)}, fail={len(bad)} ({len(bad)/len(r)*100:.0f}%)) ===')
    print(f'  margin: ok median={np.median(margin_ok):.2f} | fail median={np.median(margin_bad):.2f}')
    print(f'  entropy: ok median={np.median(ent_ok):.3f} | fail median={np.median(ent_bad):.3f}')
    # point-biserial corr (margin vs success)
    m_all=np.array([x['margin'] for x in r]); ok_all=np.array([x['ok'] for x in r])
    corr=np.corrcoef(m_all,ok_all)[0,1]
    print(f'  corr(margin, success) = {corr:+.3f}  corr(entropy, success) = {np.corrcoef(np.array([x["entropy"] for x in r]),ok_all)[0,1]:+.3f}')
    # ROC for "skip if margin < theta": TPR=caught failures, FPR=wrongly-skipped successes
    best=None
    for theta in np.percentile(m_all,np.arange(5,51,2.5)):
        skip=m_all<theta
        tpr=(skip&(ok_all==0)).sum()/max(1,(ok_all==0).sum())  # caught failures
        fpr=(skip&(ok_all==1)).sum()/max(1,(ok_all==1).sum())  # wrongly-skipped successes
        # net: skipping saves draft on caught-failures (1 token plain-decoded) but loses K-1 on wrongly-skipped
        net=tpr*len(bad)-fpr*len(ok)  # rough: saved cycles - lost cycles (each lost = K-1 tokens forgone)
        if best is None or net>best[3]: best=(theta,tpr,fpr,net)
    print(f'  best skip-threshold: margin<{best[0]:.2f} -> TPR={best[1]:.2f} (caught failures), FPR={best[2]:.2f} (wrongly-skipped successes), net={best[3]:+.0f} cycles')
    return corr
print('\n'+'='*70)
print('TARGET-CONFIDENCE vs DRAFTER FIRST-DRAFT REJECTION')
print('='*70)
for v in ['H_fp/Y_fp']: analyze(v)
print('\nInterpretation: corr ~0 + fail-median ≈ ok-median => failures are at CONFIDENT positions')
print('  (drafter bugs, not hard positions) => gate DEAD; Lead-10 (distillation) MIGHT help.')
print('corr strongly negative + fail-median << ok-median => failures at UNCERTAIN positions')
print('  (hard positions) => gate VIABLE; Lead-10 (distillation) WON\'T help much.')
