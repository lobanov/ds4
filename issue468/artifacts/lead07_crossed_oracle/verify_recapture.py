#!/usr/bin/env python3
"""Lead 07 phase1-recapture verify + convert.

For all 60 matched prompts:
  1. verify H_iq2_tf (.bin): record count == gen_tokens, dump-tok == forced Y_fp, sane hidden.
  2. convert tf_dump_full/<id>.bin -> bundles_iq2tf/<id>/ (oracle_inputs.npz=H_iq2_tf,
     target_selected_tokens.json=Y_fp). Y_iq2_tf stays at tf_dump_full/<id>.iq2argmax.
  3. convert the FP captures -> bundles_fp/<id>/ (H_fp + Y_fp).
  4. report the aggregate Y_iq2_tf-vs-Y_fp agreement (the per-position label-drift signal
     on the common trajectory) + the IQ2-native baseline availability.
"""
import json, sys, numpy as np
sys.path.insert(0, 'issue468/run_lead04_modal')
from convert_vllm_to_oracle import surgery_to_main_hidden, convert as convert_fp
from pathlib import Path

ROOT = Path('issue468/artifacts/lead07_crossed_oracle')
TF_DUMP = ROOT / 'tf_dump_full'
FORCE = ROOT / 'force_tokens'
FP_RAW = ROOT / 'fp_captures_raw'
B_IQ2TF = ROOT / 'bundles_iq2tf'
B_FP = ROOT / 'bundles_fp'
B_IQ2TF.mkdir(parents=True, exist_ok=True)
B_FP.mkdir(parents=True, exist_ok=True)
Q2_REF = Path('issue468/artifacts/acceptance_powered/combined300/per_prompt')

REC = np.dtype([('pos', '<i4'), ('tok', '<i4'), ('h', '<f4', 12288)])
total_pos = 0
agree_pos = 0
agree_total = 0
n_q2_ref = 0
bad = 0
for src in ['codealpaca', 'dolly', 'jsonex']:
    for i in range(80, 100):
        pid = f'{src}_{i:04d}'
        binf = TF_DUMP / f'{pid}.bin'
        arf = TF_DUMP / f'{pid}.iq2argmax'
        fof = FORCE / f'{pid}.tokens'
        fp_cap = FP_RAW / f'phaseB_{pid}.npz'
        if not binf.exists():
            print(f'MISSING {pid}.bin'); bad += 1; continue
        raw = np.fromfile(binf, dtype=REC)
        y_fp = np.array([int(x) for x in fof.read_text().split()])
        y_iq2tf = np.array([int(x) for x in arf.read_text().split()])
        # verify: record count == len(Y_fp), dump-tok == Y_fp, hidden sane
        if len(raw) != len(y_fp):
            print(f'  {pid}: record count {len(raw)} != Y_fp len {len(y_fp)}'); bad += 1; continue
        if not np.array_equal(raw['tok'], y_fp):
            print(f'  {pid}: dump-tok != forced Y_fp'); bad += 1; continue
        h = raw['h']
        if np.isnan(h).any() or np.isinf(h).any():
            print(f'  {pid}: NaN/inf in hidden'); bad += 1; continue
        # convert H_iq2_tf -> oracle bundle (target_tokens = Y_fp, the forced/dumped tokens)
        bdir = B_IQ2TF / pid
        (bdir / 'oracle').mkdir(parents=True, exist_ok=True)
        np.savez(bdir / 'oracle' / 'oracle_inputs.npz', positions=raw['pos'].astype(np.int32), main_hidden=np.ascontiguousarray(h))
        (bdir / 'target_selected_tokens.json').write_text(json.dumps([int(t) for t in y_fp.tolist()]) + '\n')
        (bdir / 'bundle_manifest.json').write_text(json.dumps({'prompt_name': pid, 'prompt_tokens': 0, 'temperature': 0.0, 'seed': 0, 'ctx': 4096, 'block': 5, 'generated_tokens': int(len(y_fp)), 'measure_steps': int(len(y_fp)), 'reference_mode': 'greedy'}))
        # convert FP capture -> oracle bundle (H_fp + Y_fp)
        convert_fp(fp_cap, B_FP / pid, pid)
        # stats
        total_pos += len(y_fp)
        a = int((y_iq2tf == y_fp).sum()); agree_pos += a; agree_total += len(y_fp)
        if (Q2_REF / f'{pid}.json').exists(): n_q2_ref += 1

print(f'\n=== phase1-recapture verify ===')
print(f'prompts: 60 | bad: {bad} | total forced positions: {total_pos}')
print(f'Y_iq2_tf==Y_fp (per-position argmax agreement on common traj): {agree_pos}/{agree_total} = {agree_pos/agree_total*100:.1f}%')
print(f'  -> the per-step label-drift is {100-agree_pos/agree_total*100:.1f}% (the trajectory divergence is compounding, not per-step)')
print(f'IQ2-native Q2 reference available: {n_q2_ref}/60')
print(f'bundles_iq2tf/ (H_iq2_tf) + bundles_fp/ (H_fp) written; Y_iq2_tf at tf_dump_full/<id>.iq2argmax')
