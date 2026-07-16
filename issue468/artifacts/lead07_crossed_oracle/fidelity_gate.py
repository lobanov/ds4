#!/usr/bin/env python3
"""Lead 07 teacher-force FIDELITY GATE (phase0-ds4-teacherforce).

Validates that the teacher-force capture H_iq2_tf is faithful: the D_f32 drafter run
on H_iq2_tf (with the FP-trajectory anchors Y_fp) must reproduce a SANE p=1 in the
IQ2-native band (~0.70-0.80), not garbage (~0 or ~1). This guards against a capture bug
(the Lead 04 hidden-capture-fidelity lesson).

Also reports A(D_f32, H_fp, Y_fp) [the native ceiling, from the FP capture] for context,
so the smoke shows all four inputs are sane + measurable.
"""
import sys, json, numpy as np, torch
sys.path.insert(0, 'issue468/dspark_train')
sys.path.insert(0, 'issue468/dspark_oracle')
from drafter_body import build_body, BLOCK, HC, DIM  # noqa
from drafter_head import build_head  # noqa
from analyze_phaseB_gap import measure_fp_p1  # noqa
sys.path.insert(0, 'issue468/run_lead04_modal')
from convert_vllm_to_oracle import surgery_to_main_hidden  # noqa

DSPARK = '/Users/lobanov/Projects/ds4/gguf/dspark.gguf'
TARGET = '/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AprojQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf'
DUMP = 'issue468/artifacts/lead07_crossed_oracle/tf_dump/dolly_0090.bin'
ARGMAX = 'issue468/artifacts/lead07_crossed_oracle/tf_dump/dolly_0090.iq2argmax'
FORCE = 'issue468/artifacts/lead07_crossed_oracle/force_tokens/dolly_0090.tokens'
FP_CAP = 'issue468/artifacts/lead07_crossed_oracle/fp_captures_raw/phaseB_dolly_0090.npz'
Q2_REF = 'issue468/artifacts/acceptance_powered/combined300/per_prompt/dolly_0090.json'

dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'loading D_f32 drafter on {dev}...', flush=True)
body = build_body(DSPARK, TARGET, dev, dtype=torch.float32)
head = build_head(DSPARK, TARGET, dev, lora_rank=0, dtype=torch.float32)
print('drafter loaded.', flush=True)

# --- teacher-force capture: H_iq2_tf + Y_fp (the forced anchors/labels) ---
REC = np.dtype([('pos', '<i4'), ('tok', '<i4'), ('h', '<f4', 12288)])
raw = np.fromfile(DUMP, dtype=REC)
mh_iq2tf = raw['h'].astype(np.float32)
pos_iq2tf = raw['pos']
y_fp = np.array([int(x) for x in open(FORCE).read().split()])
y_iq2tf = np.array([int(x) for x in open(ARGMAX).read().split()])

# --- FP capture: H_fp + Y_fp (the native ceiling) ---
mh_fp, pos_fp, y_fp2, plen, _ = surgery_to_main_hidden(FP_CAP)
assert np.array_equal(y_fp, y_fp2), 'FP forced tokens != FP capture greedy_tokens'

# A(D_f32, H_iq2_tf, Y_fp): drafter on IQ2 hidden (FP context), judge vs FP labels.
# Anchors + labels both = Y_fp (the context tokens). THE FIDELITY CHECK.
p1_iq2tf, n = measure_fp_p1(mh_iq2tf, pos_iq2tf, y_fp.tolist(), int(plen), body, head, dev)
# A(D_f32, H_fp, Y_fp): the native ceiling (self-consistent FP).
p1_fp, _ = measure_fp_p1(mh_fp, pos_fp, y_fp.tolist(), int(plen), body, head, dev)

q2_p1 = json.load(open(Q2_REF)).get('p1')

print('\n' + '=' * 60)
print('FIDELITY GATE (dolly_0090 teacher-force smoke):')
print(f'  A(D_f32, H_iq2_tf, Y_fp) p1 = {p1_iq2tf:.3f}  (n={n})  <-- fidelity check')
print(f'  A(D_f32, H_fp,  Y_fp)     p1 = {p1_fp:.3f}          <-- native ceiling')
print(f'  IQ2-native Q2 reference   p1 = {q2_p1}')
print(f'  Y_iq2_tf==Y_fp agreement: {int((y_iq2tf==y_fp).sum())}/{len(y_fp)} ({(y_iq2tf==y_fp).mean()*100:.0f}%)')
sane = 0.55 <= p1_iq2tf <= 0.95
print(f'\nFIDELITY: {"PASS" if sane else "FAIL"} (H_iq2_tf p1={p1_iq2tf:.3f} {"in" if sane else "OUTSIDE"} the sane [0.55,0.95] band)')
print('=' * 60)
