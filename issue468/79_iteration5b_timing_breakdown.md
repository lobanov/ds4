# Iteration 5b: Timing Breakdown — Verify is 54% of Cycle; Need +0.74 n_accept

Date: 2026-07-01. Iteration 5b. Doc-only research record.

## 0. Actual per-phase timing (target-pos0, temp=1)

| phase | ms | % cycle |
|---|---|---|
| anchor decode | 26.2 | 20% |
| drafter forward + Markov recompute | 11.0 | 8% |
| **batch verify (5 positions)** | **69.9** | **54%** |
| B2 accept (CPU) | 1.4 | 1% |
| correction decode | 21.4 | 16% |
| **total** | **130.0** | |

n_accept = 4.34, drafts = 2.53, ms/tok = 30.0, gen t/s = 33.4.

## 1. Break-even analysis

To cross baseline (39 t/s, 25.6 ms/tok):
- Need n_accept ≥ 130 / 25.6 = **5.08 tokens/cycle**
- Current: 4.34
- Gap: **0.74 tokens/cycle** (17% improvement needed)

## 2. Where can the gap come from?

The 0.74 token gap requires either:
- **Higher acceptance**: pos 1-4 need to improve from current rates (75.4/55.8/65.2/73.3%) to closer to oracle rates (98.1/91.8/87.9/78.5%). This requires the drafter to produce better drafts on stochastic-path inputs — not code-fixable (architectural).
- **Lower cycle cost**: save ~19ms/cycle. The verify (70ms) is compute-bound and unlikely to drop by 19ms without a different kernel. The correction (21ms) is structurally needed. The anchor (26ms) is irreducible.
- **Eliminate a forward**: if the correction decode could be avoided (Opp-a), the cycle drops to ~109ms, needing n_accept ≥ 4.26 (currently 4.34 — would cross!). But Opp-a was ruled infeasible (issue468/38).

## 3. The correction decode is the marginal cost

The correction decode (21.4ms avg) runs on ~82% of cycles. If we could eliminate it (via Opp-a-style KV commit), the cycle drops to ~109ms:
- ms/tok = 109 / 4.34 = 25.1 → **39.8 t/s** (CROSSES!)

But Opp-a was ruled structurally infeasible (issue468/38): the correction token C needs its own forward because the verify batch didn't process it.

## 4. Structural conclusion (confirmed)

The M5 Max baseline is too fast (25.6 ms/tok) and flat for 3-forward speculation to win at the achieved acceptance rates. The cycle cost is dominated by the verify (54%) and correction (16%) forwards. Eliminating the correction would cross the gate, but it's structurally infeasible in B2.

The target-pos0 improvement (+3.7 t/s) narrowed the gap from 9.5 to 5.8, but the remaining gap requires either:
- Opp-a (correction-decode elimination) — infeasible (issue468/38)
- Higher acceptance at temp=1 — architectural (hidden-state conditioning)
- Slower baseline hardware — not controllable

## 5. Iteration report

| metric | baseline (no target-pos0) | target-pos0 v3 | break-even needed |
|---|---|---|---|
| accepted drafts/cycle | 1.97 | **2.53** | — |
| n_accept | 3.86 | **4.34** | 5.08 |
| committed/cycle | 2.97 | **3.54** | 5.08 |
| gen t/s | 29.5 | **33.4** | 39+ |
| ms/token | 33.9 | **30.0** | 25.6 |
| gap | 9.5 | **5.6** | 0 |
