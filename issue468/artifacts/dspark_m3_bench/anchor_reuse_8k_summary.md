# M3 anchor reuse (lever 2) — 8k bench (2026-07-14, warm, batched+reuse)
DS4_DSPARK_VERIFY_BATCHED=1 DS4_DSPARK_ANCHOR_REUSE=1, frontier=3072, gen=128, 3 prompts, ctx-alloc 4096, t=8.

| path | t/s | decode_ms | draft_ms | verify_ms | total_ms | verified | verify_n | rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| batched (no reuse) | 23.50 | 28.2 | 48.0 | 104.8 | 181.1 | 3.25 | 4.50 | 4.73 |
| batched + anchor reuse | 25.97 | 0.1 | 45.2 | 116.9 | 162.2 | 4.18 | 4.86 | 3.91 |

- reuse vs batched: **+10.5%** (25.97 vs 23.50 t/s). reuse vs plain (35.5): -26.9% (0.73x; verify still dominates).
- decode 28.2 -> 0.1 ms (anchor folded into the batched verify; fires ~100% of cycles).
- verify 104.8 -> 116.9 ms (sublinear +12 ms for the folded anchor; verify_n 4.50 -> 4.86).
- continuation acceptance HELD: verified-minus-anchor ~3.18 vs batched 3.25 (on 8k the one-position window staleness is amortized; the short-prompt smoke showed a drop, but 8k does not).
- total 181.1 -> 162.2 ms (-19 ms/cycle).
- The reuse engages only when batched is on + first_token is the greedy argmax (precondition); falls back to standalone decode otherwise (smoke 3 confirmed: reuse without batched -> decode 26.7, not engaged).
