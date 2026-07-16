#!/usr/bin/env python3
"""Lead 07 phase1-recapture prep: generate forced-tokens (Y_fp) for the 60 matched
prompts + the teacher_force bench config.

For each prompt (lead3_corpus, codealpaca/dolly/jsonex x 0080-0099):
  - reads the FP capture (fp_captures_raw/phaseB_<id>.npz) -> greedy_tokens (Y_fp)
  - writes force_tokens/<id>.tokens (space-separated, consumed by ds4-spec-bench)
  - emits a teacher_force bench-config line (frontier_tokens = FP prompt_tokens as a
    PLACEHOLDER; --rewrite-frontier corrects it to the exact ds4 prompt length).
"""
import json, sys, numpy as np
sys.path.insert(0, 'issue468/run_lead04_modal')
from convert_vllm_to_oracle import surgery_to_main_hidden
from pathlib import Path

ROOT = Path('issue468/artifacts/lead07_crossed_oracle')
FP_DIR = ROOT / 'fp_captures_raw'
FORCE_DIR = ROOT / 'force_tokens'
FORCE_DIR.mkdir(parents=True, exist_ok=True)
PROMPT_DIR = Path('issue468/prompts/lead3_corpus').resolve()

cfg_lines = []
n_ok = 0
for src in ['codealpaca', 'dolly', 'jsonex']:
    for i in range(80, 100):
        pid = f'{src}_{i:04d}'
        fp_cap = FP_DIR / f'phaseB_{pid}.npz'
        prompt_file = PROMPT_DIR / f'{pid}.txt'
        if not fp_cap.exists() or not prompt_file.exists():
            print(f'SKIP {pid}: missing fp_cap={fp_cap.exists()} prompt={prompt_file.exists()}')
            continue
        try:
            _, _, y_fp, plen, _ = surgery_to_main_hidden(str(fp_cap))
        except Exception as e:
            print(f'SKIP {pid}: convert failed: {e}')
            continue
        (FORCE_DIR / f'{pid}.tokens').write_text(' '.join(str(int(t)) for t in y_fp))
        cfg = {
            'id': pid, 'mode': 'teacher_force',
            'chat_prompt_file': str(prompt_file),
            'system': 'You are a helpful assistant',
            'frontier_tokens': int(plen),   # placeholder; --rewrite-frontier corrects to ds4 length
            'gen_tokens': int(len(y_fp)),
            'exclude_eos': False, 'seed': 1, 'temperature': 0,
        }
        cfg_lines.append(json.dumps(cfg))
        n_ok += 1

(ROOT / 'recapture_config.jsonl').write_text('\n'.join(cfg_lines) + '\n')
print(f'\nwrote {n_ok} forced-token files + recapture_config.jsonl ({len(cfg_lines)} lines)')
