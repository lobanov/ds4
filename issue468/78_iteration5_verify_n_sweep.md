# Iteration 5: Verify-N Sweep — d=5 Optimal; Target-Pos0 v3 = 33.2 t/s

Date: 2026-07-01. Iteration 5. Doc-only research record.

## 0. What was tested

Codex xhigh lead #2: runtime verify-length truncation. Added
DS4_DSPARK_VERIFY_N to verify only N draft positions (reducing verify compute).

## 1. Verify_n sweep (target-pos0, temp=1, n=256, seed=1)

| verify_n | gen t/s | delta vs d=5 |
|---|---|---|
| 5 | **33.18** | — |
| 4 | 32.78 | −0.4 |
| 3 | 30.85 | −2.3 |
| 2 | 28.06 | −5.1 |

verify_n=5 is optimal. Reducing depth is WORSE — the marginal committed tokens
at pos 3-4 are worth more than the verify compute saved.

## 2. Current best configuration

**DS4_DSPARK_TARGET_POS0=1, verify_n=5, temp=1: 33.2 t/s**
- Baseline (no target-pos0): 29.5 t/s
- Improvement: +3.7 t/s (+12.5%)
- Gap to baseline (39): 5.8 t/s (was 9.5)

## 3. Iteration report

| metric | baseline | target-pos0 v3 | oracle MC (greedy) |
|---|---|---|---|
| accepted drafts/cycle | 1.97 | **2.54** | 4.10 |
| committed tokens/cycle | 2.97 | **3.54** | 5.10 |
| gen t/s | 29.5 | **33.2** | — |
| ms/token | 33.9 | **30.1** | — |
| baseline t/s | ~39 | ~39 | — |
| gap | 9.5 | **5.8** | — |

## 4. Remaining codex leads

- Lead #3 (second drafter forward with target-pos0): diagnostic only, likely
  speed-negative (adds a full drafter pass). Skip.
- Lead #4 (draft trees): algorithm redesign, out of current scope.
- Lead #5 (GPU/fused B2 cleanup): <1 t/s ceiling.

## 5. Convergence assessment

Review #1: architectural limitation (issue468/73) → refined by codex →
target-pos0 lead found → +3.7 t/s.

Review #2: this iteration. The target-pos0 + Markov recompute is the best
code-only improvement found. The remaining 5.8 t/s gap is from:
- Pos 2 dropping to 55.8% (drafter base_logits computed with wrong prev chain)
- Positions 1-4 still below oracle rates (OOD main_hidden on stochastic paths)
- No further code-only leads identified by codex or self-analysis

This is review #2. The gap is not fully closed but no further viable code-only
leads have been identified. The investigation may converge here.
