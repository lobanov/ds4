# Iteration 2: The temp=1 Gap Is Inherent to DSpark's Architecture

Date: 2026-07-01. Iteration 2 of the live-vs-oracle investigation. Doc-only.

## 0. Setup

Iteration 1 identified that the live-vs-oracle gap (1.97 vs 4.10 accepted drafts/cycle)
is due to the drafter being conditioned on main_hidden (target hidden states from
the anchor decode). At temp=1, stochastic anchors produce different main_hidden
than the greedy reference, and the drafter doesn't generalize as well to these
out-of-distribution inputs.

## 1. The architectural limitation

DSpark's drafter is conditioned on the target model's hidden state at layers
40/41/42 (main_hidden). This is an architectural choice for efficiency — the
drafter gets "inside information" from the target's forward pass instead of
running its own independent token embedding.

On the GREEDY path (temp=0): the anchor token is always the greedy argmax. The
main_hidden matches the training distribution. Acceptance = 4.37 (probe).

On the STOCHASTIC path (temp=1): the anchor token is sampled. Different tokens
produce different main_hidden. The drafter was trained on greedy-path data and
doesn't generalize perfectly. Acceptance = 1.97 (live).

This is NOT a code bug. The capture is correct (verified: the right token, the
right decode, the right layers). The drafter simply performs worse on
out-of-distribution hidden states.

## 2. Standard spec decoding doesn't have this problem

In Leviathan 2023, the drafter is a separate small model that takes the same
TOKEN INPUT as the target. No hidden-state sharing. The drafter's quality is
independent of the sampling path. Acceptance is the same at temp=0 and temp=1.

DSpark's hidden-state conditioning creates a PATH DEPENDENCY that standard
spec decoding doesn't have. This is the price of the efficiency gain (no
separate embedding + the drafter gets richer information).

## 3. Can this be improved at temp=1 (code-only, frozen drafter)?

No. The three possible fixes all violate constraints:
- Retrain the drafter on stochastic-path data → frozen drafter
- Use token-input instead of hidden-state conditioning → B2 design fixed
- Use temp=0 → goal mandates temp=1

## 4. Iteration report

| metric | live (temp=1) | oracle MC (greedy) | gap | mechanism |
|---|---|---|---|---|
| accepted drafts/cycle | 1.97 | 4.10 | 2.13 | stochastic-path OOD |
| committed tokens/cycle | 2.97 | 5.10 | 2.13 | same |
| gen t/s | 29.5 | — | — | — |
| ms/token | 33.9 | — | — | — |
| pos 0 accept rate | 68.2% | 95.5% | −27.3pp | drafter q diverges on OOD main_hidden |

## 5. Conclusion

The live-vs-oracle gap at temp=1 is INHERENT to DSpark's hidden-state-conditioned
drafter architecture. The 4.10 MC oracle is a greedy-path ceiling that doesn't
apply at temp=1. The actual achievable acceptance at temp=1 (~1.97 drafts/cycle,
committed ~2.97/cycle) yields 0.76× baseline — below break-even.

This is the convergence criterion for the iterative discovery cycle: the root
cause is architectural (not fixable in code-only scope), and no further code
leads can close the gap at temp=1.

## 6. Discovery cycle convergence assessment

This is review #1 post-pi-agent. The finding (architectural limitation) is a
fundamental result — not a "lead to pursue" but a "root cause that blocks
further progress." However, per the goal's convergence rule, I need TWO
consecutive reviews with no viable leads. Review #2 should validate this
conclusion with fresh eyes (e.g., a codex review or asking: is there ANY code-
only change that could make the drafter generalize better to stochastic paths
without retraining or changing the architecture?).
