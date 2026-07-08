# Lead 01 — Offline anchor-reuse falsifier

Date: 2026-07-07. Status: **resolved & archived 2026-07-07** (moved from `issue468/pending/`).
Result: see `issue468/summaries/anchor_reuse_falsifier.md`. Priority was **first** —
cheapest decisive experiment; gates leads 02 and 05.

## Rationale

Anchor reuse is the load-bearing unverified assumption of the entire speedup case.
The corrected cycle model (`summaries/spec_speedup_model.md`) puts K=4 at −0.9%
*if* the verify forward's correction token can anchor the next draft without a
fresh 26 ms decode; under the shipped verifier (which decodes every cycle) the same
configuration is −18.9%. The adversarial codex review flagged this ~18 pp
assumption as the primary risk and proposed a falsifier; it was never run.

The specific doubt: the DSpark drafter's input is the target's layer-40/41/42
hidden state at the anchor position (`dspark_oracle/forward.py: forward_embed`,
which takes `main_hidden` + the anchor token id). On a rejection, the correction
token has never been processed by the target, so its own hidden state does not
exist. Anchor reuse therefore requires drafting from the hidden state at the
**last accepted position** (which the verify forward did produce, from a valid
prefix) with the correction token entering as embedding only. Whether acceptance
survives that substitution is unknown — and it is testable offline, because the
retained capture bundles contain `main_hidden` at every position along the greedy
spine (the correction token at step t *is* the greedy target token, so both the
"true" hidden at position t and the "stale" hidden at position t−1 are already on
disk).

## Content of work

All offline, reusing `issue468/dspark_oracle/` and the retained
`artifacts/exactness_small_bundles/` (temp=0 cells first; 0.5/1.0 as follow-up):

1. Add a reuse-mode variant to the acceptance harness: at each measure step t,
   draft from `(main_hidden[pos t−1], anchor = target_token[t])` instead of the
   baseline `(main_hidden[pos t], anchor = target_token[t])`.
2. Re-measure p=1 match rate and E[a|5block] over the same steps; paired
   comparison against the retained baseline (same fidelity checks as Stage 0:
   reproduce baseline draft tokens exactly before trusting the variant).
3. Report the per-position acceptance delta and a bootstrap CI, mirroring the
   Stage 1 comparison format (`run_stage1_q4tap_compare.py` is the template).

Estimated effort: ~1–2 days.

## Success criteria

- **Reuse survives** (p=1 and E[a|5block] within a few pp of baseline, CI not
  clearly negative): the −0.9% optimistic edge of the model band is realizable in
  principle → lead 05 (verifier engineering) is unblocked and the anchor-reuse
  accounting in leads 02/04 is legitimate.
- **Reuse collapses** (material acceptance drop): the optimistic edge is dead; the
  shipped-verifier accounting (−18.9% at K=4) is the true local ceiling for this
  drafter → local single-request DSpark on the in-RAM setup is conclusively
  closed, and the dossier can record a clean stop/narrow decision for that regime.

Either outcome is a win: this experiment converts the model's headline caveat
into a measured fact.
