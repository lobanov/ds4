# Iteration 4: Target-Sampled Pos 0 — +30pp at pos 0, −36pp at pos 1 (Markov mismatch)

Date: 2026-07-01. Iteration 4. Doc-only research record.

## 0. Codex xhigh critique (issue468/76)

Codex gpt-5.5 xhigh reviewed the full investigation. Key lead: target-sampled
position 0. Sample drafts[0] from softmax(s->logits) instead of the drafter's
argmax. This attacks the pos-0 failure (68.2% live vs 95.5% oracle) directly.

## 1. Implementation + result

Implemented DS4_DSPARK_TARGET_POS0: overrides drafts[0] with a sample from
softmax(s->logits) AFTER the drafter forward loop. The drafter's q_dist[0] is
still used for the accept probability. The verify path verifies all 5 drafts.

Result (temp=1, n=256, seed=1):

| pos | baseline (argmax) | target-pos0 | delta |
|---|---|---|---|
| 0 | 68.2% | **98.4%** | **+30.2pp** |
| 1 | 82.2% | 46.0% | −36.2pp |
| 2 | 67.6% | 72.4% | +4.8pp |
| 3 | 64.0% | 66.7% | +2.7pp |
| 4 | 43.8% | 50.0% | +6.2pp |
| drafts/cycle | 1.97 | 2.09 | +0.12 |
| gen t/s | 29.5 | **30.4** | **+0.9** |

## 2. The pos-1 crash: Markov chain mismatch

The drafter's Markov head computes drafts sequentially: drafts[i] depends on
drafts[i-1] (via `prev = drafts[i-1]`). The drafter forward loop used the
argmax at each position for `prev`. But after overriding drafts[0] with the
target token, the `prev` for position 1 is wrong — it was the drafter's argmax,
not the target token.

This causes pos-1 to crash (82.2% → 46.0%): the drafter computed its pos-1
draft based on a different pos-0 token than what was actually proposed.

## 3. Fix: recompute drafts[1..4] with prev = target token

After overriding drafts[0], re-run the Markov head for positions 1-4 using
`prev = drafts[0]` (the target token). The Markov head is cheap (5 GPU matvecs
+ CPU argmax). This should recover pos-1 and amplify the gain.

## 4. Iteration report

| metric | baseline | target-pos0 (v1) | oracle MC (greedy) |
|---|---|---|---|
| accepted drafts/cycle | 1.97 | 2.09 | 4.10 |
| committed tokens/cycle | 2.97 | 3.09 | 5.10 |
| gen t/s | 29.5 | 30.4 | — |
| ms/token | 33.9 | 32.9 | — |

## 5. Next: recompute the Markov chain after the pos-0 override
