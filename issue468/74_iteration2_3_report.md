# Iteration 2 Report + Iteration 3 Lead: OOD or Capture Bug?

Date: 2026-07-01. Iterations 2-3 of the live-vs-oracle investigation. Doc-only.

## 0. Iteration 2 report

| metric | live (temp=1) | oracle MC (greedy) | gap |
|---|---|---|---|
| accepted drafts/cycle | 1.97 | 4.10 | 2.13 |
| committed tokens/cycle | 2.97 | 5.10 | 2.13 |
| gen t/s | 29.5 | — | — |
| ms/token | 33.9 | — | — |
| pos 0 accept rate | 68.2% | 95.5% | −27.3pp |

Pos-0 rejects show the drafter CONFIDENTLY proposing tokens (q>0.8) that the
target assigns ~0 probability (p≈0). Examples: q=0.999/p=0.000, q=0.893/p=0.000.
The drafter is confidently wrong on stochastic-path inputs.

## 1. Code flow verification (no stale-logits bug)

Verified the B2 code flow:
1. Main loop samples anchor from s->logits (pos N)
2. eval_dspark_b2 decodes anchor via ds4_session_eval → OVERWRITES s->logits
3. s->logits now = target's prediction for position N+1
4. Drafter produces drafts[0..4] conditioned on main_hidden from the same anchor decode
5. Pos-0 compares draft[0] against the NEW s->logits

Both draft[0] and s->logits predict position N+1, from the same anchor decode.
No stale logits. The comparison is correct.

## 2. The p=0 values are real (not printf truncation)

The debug prints p=%.5f. Values showing p=0.00000 mean p < 5e-6. These are
genuinely near-zero target probabilities. The drafter is proposing tokens the
target model considers essentially impossible.

## 3. Iteration 3 lead: is the drafter input actually OOD, or is there a capture quality issue?

The drafter is confidently wrong on ~32% of pos-0 proposals. This is too high
for simple OOD drift — it suggests the drafter's main_hidden input is materially
degraded on stochastic paths. Two hypotheses:

(A) The drafter genuinely doesn't generalize (architectural, not fixable).
(B) The main_hidden capture has a quality issue at temp=1 that degrades the
    drafter's input beyond what it should be.

Test for (B): compare the live main_hidden to a reference main_hidden at the
same position. If they differ significantly, the capture may be wrong. If they
match (same hidden state, different token), the issue is architectural.

This requires dumping the live main_hidden during B2 and comparing to what the
drafter would receive on the greedy path at the same position. The existing
DS4_DSPARK_B2_DEBUG already prints the first 6 values of main_hidden — I can
use that to compare live vs probe values.

## 4. Why temperature scaling won't help

On pos-0 rejects: q>0.8 but p≈0. The issue is p=0 (target says NO), not q being
too peaked. Temperature scaling on q would spread probability mass but wouldn't
make p>0. The rejection probability = 1 - p/q ≈ 1 when p≈0 regardless of q's
temperature. So temperature scaling doesn't help.

## 5. Convergence assessment

Review #1 (iteration 2): architectural root cause identified. Not fixable code-only.
Review #2 (iteration 3): need to rule out capture quality before concluding.
The main_hidden comparison is the decisive test — if live main_hidden matches
the reference, the convergence criterion (two consecutive reviews with no viable
leads) is satisfied.
