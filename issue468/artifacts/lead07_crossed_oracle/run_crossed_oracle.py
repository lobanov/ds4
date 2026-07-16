#!/usr/bin/env python3
"""Lead 07 phase2-oracle: the crossed FP/IQ2 oracle on the common (teacher-forced) trajectory.

4 cells (D_f32 torch drafter), anchors always = Y_fp (the FP-trajectory context tokens):
  baseline    A(D_f32, H_iq2_tf, Y_iq2_tf)  -- IQ2 quality at the FP context
  hidden-side A(D_f32, H_fp,     Y_iq2_tf)  -- FP hidden, IQ2 labels
  label-side  A(D_f32, H_iq2_tf, Y_fp )     -- IQ2 hidden, FP labels
  ceiling     A(D_f32, H_fp,     Y_fp )     -- native FP ceiling

Attribution (2x2 on the common FP trajectory):
  lift            = ceiling - baseline = A(H_fp,Y_fp) - A(H_iq2_tf,Y_iq2_tf)
  hidden_main     = mean(ceiling, hidden-side) - mean(label-side, baseline)
  label_main      = mean(ceiling, label-side)  - mean(hidden-side, baseline)
  (hidden_main + label_main == lift, by construction; the larger drives the decision)
  interaction     = ceiling - label-side - hidden-side + baseline

Decision (Lead 07): lift small (~neither) -> PIVOT (not actionable on the common traj,
                    the native-vs-IQ2 gap was trajectory/context). Else if hidden_main >
                    label_main -> PROCEED (hidden-side recoverable); else PIVOT (label-drift).
"""
import json, sys, numpy as np, torch
sys.path.insert(0, 'issue468/dspark_train')
sys.path.insert(0, 'issue468/dspark_oracle')
from drafter_body import build_body, BLOCK, HC, DIM  # noqa
from drafter_head import build_head  # noqa

DSPARK = '/Users/lobanov/Projects/ds4/gguf/dspark.gguf'
TARGET = '/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf'
ROOT = 'issue468/artifacts/lead07_crossed_oracle'
K = 4  # E[a|4]


def load_bundle_h(bdir):
    oi = np.load(f'{bdir}/oracle/oracle_inputs.npz')
    return oi['main_hidden'].astype(np.float32), oi['positions']


def measure_crossed(mh, anchors_seq, labels_seq, body, head, dev):
    """Return (p1, ea4, n_anchors). anchors_seq = context tokens (Y_fp); labels_seq = judgment source."""
    n_cap = mh.shape[0]
    max_step = min(len(labels_seq) - BLOCK - 1, len(anchors_seq) - 1, n_cap - 1)
    if max_step < 1:
        return None, 0.0, 0
    anc = [int(anchors_seq[s]) for s in range(1, max_step + 1)]
    mh_seq = mh[:max_step + 1]
    with torch.no_grad():
        xs = body.forward_prompt(mh_seq, anc)
        anc_t = torch.tensor(anc, device=dev, dtype=torch.long)
        drafts = []
        for i in range(0, xs.shape[0], 24):
            out, _ = head(xs[i:i + 24], anc_t[i:i + 24])
            drafts.append(out[:, 1:].cpu().numpy())
    drafts = np.concatenate(drafts, axis=0)  # [max_step, BLOCK]
    n = max_step
    p1_hits = 0
    acc_sum = 0.0
    for i in range(n):
        s = i + 1
        if drafts[i, 0] == labels_seq[s + 1]:
            p1_hits += 1
        acc = 0
        for b in range(min(K, BLOCK)):
            if s + 1 + b < len(labels_seq) and drafts[i, b] == labels_seq[s + 1 + b]:
                acc += 1
            else:
                break
        acc_sum += acc
    return p1_hits / n, acc_sum / n, n


def boot_ci(vals, n_boot=10000, seed=0):
    vals = np.array(vals, dtype=float)
    if len(vals) == 0:
        return float('nan'), float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    n = len(vals)
    means = [vals[rng.integers(0, n, size=n)].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(vals.mean()), float(lo), float(hi)


def main():
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'loading D_f32 drafter on {dev}...', flush=True)
    body = build_body(DSPARK, TARGET, dev, dtype=torch.float32)
    head = build_head(DSPARK, TARGET, dev, lora_rank=0, dtype=torch.float32)
    print('drafter loaded.', flush=True)

    rows = []
    for src in ['codealpaca', 'dolly', 'jsonex']:
        for i in range(80, 100):
            pid = f'{src}_{i:04d}'
            try:
                h_iq2tf, _ = load_bundle_h(f'{ROOT}/bundles_iq2tf/{pid}')
                h_fp, _ = load_bundle_h(f'{ROOT}/bundles_fp/{pid}')
                y_fp = json.loads(open(f'{ROOT}/bundles_iq2tf/{pid}/target_selected_tokens.json').read())
                y_fp = [int(t) for t in y_fp]
                y_iq2tf = [int(x) for x in open(f'{ROOT}/tf_dump_full/{pid}.iq2argmax').read().split()]
            except Exception as e:
                print(f'  SKIP {pid}: {e}'); continue
            # 4 cells: (mh, anchors, labels)
            cells = {}
            for name, mh, lab in [('baseline', h_iq2tf, y_iq2tf),
                                  ('hidden', h_fp, y_iq2tf),
                                  ('label', h_iq2tf, y_fp),
                                  ('ceiling', h_fp, y_fp)]:
                p1, ea4, n = measure_crossed(mh, y_fp, lab, body, head, dev)
                cells[name] = (p1, ea4, n)
            if any(cells[k][0] is None for k in cells):
                print(f'  SKIP {pid}: a cell had too few anchors'); continue
            # IQ2-native reference (the deployable IQ2 trajectory, for context)
            try:
                q2_p1 = json.load(open(f'issue468/artifacts/acceptance_powered/combined300/per_prompt/{pid}.json')).get('p1')
            except Exception:
                q2_p1 = None
            a, b = cells['ceiling'][0], cells['baseline'][0]
            c, d = cells['hidden'][0], cells['label'][0]
            row = {
                'pid': pid, 'source': src, 'q2_native_p1': q2_p1,
                'ceiling_p1': a, 'baseline_p1': b, 'hidden_p1': c, 'label_p1': d,
                'ceiling_ea4': cells['ceiling'][1], 'baseline_ea4': cells['baseline'][1],
                'hidden_ea4': cells['hidden'][1], 'label_ea4': cells['label'][1],
                'lift': a - b,
                'hidden_main': (a + c) / 2 - (d + b) / 2,
                'label_main': (a + d) / 2 - (c + b) / 2,
                'interaction': a - d - c + b,
                'n_anchors': cells['ceiling'][2],
            }
            rows.append(row)
            print(f"  {pid}: ceil={a:.3f} base={b:.3f} hid={c:.3f} lab={d:.3f} | lift={a-b:+.3f} hid_main={row['hidden_main']:+.3f} lab_main={row['label_main']:+.3f} | q2nat={q2_p1}", flush=True)

    # aggregate
    def agg(field):
        return boot_ci([r[field] for r in rows if r[field] is not None])
    res = {
        'n_prompts': len(rows),
        'total_anchors': int(sum(r['n_anchors'] for r in rows)),
        'cells_p1': {k: agg(k + '_p1') for k in ['ceiling', 'baseline', 'hidden', 'label']},
        'cells_ea4': {k: agg(k + '_ea4') for k in ['ceiling', 'baseline', 'hidden', 'label']},
        'lift': agg('lift'),
        'hidden_main': agg('hidden_main'),
        'label_main': agg('label_main'),
        'interaction': agg('interaction'),
        'q2_native_p1': agg('q2_native_p1'),
        'per_source': {},
    }
    for s in ['codealpaca', 'dolly', 'jsonex']:
        sr = [r for r in rows if r['source'] == s]
        res['per_source'][s] = {
            'n': len(sr),
            'lift': boot_ci([r['lift'] for r in sr]),
            'hidden_main': boot_ci([r['hidden_main'] for r in sr]),
            'label_main': boot_ci([r['label_main'] for r in sr]),
            'ceiling_p1': boot_ci([r['ceiling_p1'] for r in sr]),
            'baseline_p1': boot_ci([r['baseline_p1'] for r in sr]),
        }
    # decision
    lift_m, lift_lo, lift_hi = res['lift']
    hid_m, _, _ = res['hidden_main']
    lab_m, _, _ = res['label_main']
    if lift_hi < 0.03:
        decision = 'PIVOT (neither): lift on the common trajectory is small — the native-vs-IQ2 gap was trajectory/context, not hidden-precision; not actionable locally'
    elif hid_m > lab_m:
        decision = 'PROCEED: hidden-side main effect > label-side — the (small) common-traj lift is hidden-side recoverable'
    else:
        decision = 'PIVOT (label-side): label-side main effect >= hidden-side — drift dominates'
    res['decision'] = decision
    res['rows'] = rows

    print('\n' + '=' * 70)
    print(f'CROSSED ORACLE (n={res["n_prompts"]} prompts, {res["total_anchors"]} anchors, common FP traj):')
    for k in ['ceiling', 'baseline', 'hidden', 'label']:
        m, lo, hi = res['cells_p1'][k]
        em, _, _ = res['cells_ea4'][k]
        print(f'  A(D_f32, {k:9s}) p1 = {m:.4f}  CI[{lo:+.4f},{hi:+.4f}]   E[a|4] = {em:.3f}')
    m, lo, hi = res['lift']
    print(f'\n  lift (ceiling-baseline)     = {m:+.4f}  CI[{lo:+.4f},{hi:+.4f}]')
    m, lo, hi = res['hidden_main']
    print(f'  hidden-side main effect     = {m:+.4f}  CI[{lo:+.4f},{hi:+.4f}]')
    m, lo, hi = res['label_main']
    print(f'  label-side main effect      = {m:+.4f}  CI[{lo:+.4f},{hi:+.4f}]')
    m, lo, hi = res['interaction']
    print(f'  interaction                 = {m:+.4f}  CI[{lo:+.4f},{hi:+.4f}]')
    m, lo, hi = res['q2_native_p1']
    print(f'  IQ2-native reference p1     = {m:.4f}  CI[{lo:+.4f},{hi:+.4f}]  (deployable IQ2 traj, context only)')
    print('\n  per-source lift / hidden_main / label_main:')
    for s, v in res['per_source'].items():
        print(f'    {s:11s} (n={v["n"]}): lift={v["lift"][0]:+.4f}  hid_main={v["hidden_main"][0]:+.4f}  lab_main={v["label_main"][0]:+.4f}')
    print(f'\nDECISION: {decision}')
    print('=' * 70)

    out = f'{ROOT}/crossed_oracle_result.json'
    json.dump(res, open(out, 'w'), indent=2)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
