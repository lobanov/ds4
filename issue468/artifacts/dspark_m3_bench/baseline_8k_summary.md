# M3 8k baseline (refreshed 2026-07-14, warm, DS4_DSPARK_TIMING=1, frontier=3072, gen=128)

8k corpus = code_8k/synthesis_8k/grounded_8k (~3.9-4.0k tokens each; "8k" is a label).
3 prompts, --warm-weights, ctx-alloc 4096, t=8.

| path | t/s | decode_ms | draft_ms | verify_ms | total_ms | verified | verify_n |
|---|---:|---:|---:|---:|---:|---:|---:|
| plain (argmax) | 35.50 | — | — | — | — | — | — |
| seq (exact sequential verify) | 19.96 | 29.3 | 85.5 | 93.0 | 207.9 | 3.10 | 4.42 |
| batched (DS4_DSPARK_VERIFY_BATCHED=1) | 23.50 | 28.2 | 48.0 | 104.8 | 181.1 | 3.25 | 4.50 |

Notes:
- batched +18% over seq (23.5 vs 20.0); both ~0.56-0.66x plain (35.5).
- draft_ms differs seq vs batched (same rows_computed ~4.7) — a CPU/GPU timing-overlap artifact at this context; total_ms is authoritative.
- Longer context raises draft+verify (DSpark window + KV grow): vs the exactness corpus (draft ~25, verify ~52), the 8k context has draft ~48-86, verify ~93-105.
- batched verify (104.8ms for 4.5 tokens) > seq short-circuit (93.0ms for ~3.1 verified): the batched does ALL verify_n tokens (no short-circuit), so at low acceptance (verified 3.25 < verify_n 4.5) the batched verify is slower than the seq short-circuit on the verify axis — the batched's total win comes from the draft side (48 vs 85.5).

## Recorded first-20 no-regression reference (from the 92Q, m3_92q_score_comparison.txt)
- plain first-20: 17/20 (Q6, Q9, Q15 FAIL; rest PASS)
- DSpark-batched first-20: 17/20 (identical pattern — Q6, Q9, Q15 FAIL; rest PASS)
- => the 20-question no-regression gate: pass/fail matches this (Q6/9/15 fail, others pass), for both plain + DSpark (engaged).
