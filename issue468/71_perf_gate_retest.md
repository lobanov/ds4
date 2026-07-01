# Perf-Gate Re-test (post Q8_0 + pi-agent leads)

Date: 2026-07-01. Records the final perf-gate measurement after the Q8_0 re-quant
(issue468/69, reverted) and the pi-agent-review lead (issue468/70, the
`DS4_DSPARK_NREAL_CAP` window cap). Doc-only.

## Paired measurement (fair: same model, prompt, ctx, n, temp, seed, power)

Command: `./ds4 -m <Flash-imatrix> [-c 8192 -n 384 --temp 0 --seed 1 --power 100 -p <Fibonacci+complexity prompt>]`
DSpark runs add `--dspark ../ds4/gguf/dspark.gguf`.

| config | gen t/s | ratio vs baseline |
|---|---|---|
| **baseline (plain)** | **38.64** | 1.00× |
| dspark default (cap=127, growing window) | 24.22 | 0.63× |
| **dspark cap=5 (best config from issue468/70)** | **29.37** | **0.76×** |

## Verdict: gate NOT met

Success criterion: "DSpark end-to-end gen t/s is STRICTLY GREATER than baseline."
**DSpark (best, cap=5) = 0.76× baseline.** Gate fails by ~9 t/s (24 %).

Notes:
- Run-to-run variance is ±3–5 t/s (M5 Max thermal under sustained load); even
  the most favorable single measurement this session (Fibonacci, n=96, early
  generation) was 36.7 t/s — still below 38.6. No prompt/config crossed.
- The 0.76× best ratio matches the task brief's cited "0.76× baseline."
- Prose prompts are worse (≈19–25 t/s); the Fibonacci/code prompt is favorable.

## All 10 blocker preconditions now satisfied

1. baseline-regression clean ✓  2. P1–P4 ✓  3. block-size sweep ✓
4. Bug #1 ✓  5. codex-review converged ✓  6. F32-input MMA (disproven) ✓
7. op-level bisection ✓  8. exact-Q4 (disproven) ✓  9. Q8_0 re-quant (worse) ✓
10. **independent pi-agent review + viable lead implemented + measured ✓ (issue468/70)**

Per the goal's blocker rule, with all ten satisfied and DSpark still < baseline,
this is a structural perf-blocker to surface to the user.
