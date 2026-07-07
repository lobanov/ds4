# Lead 03 — Acceptance statistical power + realistic-trajectory measurement

Date: 2026-07-07. Status: pending. De-risks every other lead; can run in
parallel with leads 01/02.

## Rationale

Every decision number in this dossier rests on 10 prompts at temp 0 (80 measure
steps). The adversarial review of the speedup model quantified the consequence:
E[a|4] has prompt-level sd ≈ 0.43 (95% half-width ±0.3) against a **0.028**
drafts/cycle break-even gap; per-prompt modeled K=4 spans −18.5% to +10.6%. The
headline "~2 pp short of beating baseline" is therefore inside the noise — and so
would be "1 pp over." Stage 1 was likewise declared underpowered (bootstrap CI
[−0.16, +0.41] straddling zero). No stop/go decision on this branch is defensible
at n=10.

A second, independent bias is unmeasured: all acceptance was taken along the
drafter's own greedy spine. Real speculative trajectories restart each cycle from
a *correction* token the drafter just mispredicted; acceptance on post-rejection
cycles is plausibly lower (the model's own caveat). The direction and size of
this bias directly moves the break-even arithmetic and is measurable with the
existing protocol.

## Content of work

1. **Widen the corpus.** Extend the prompt corpus to 100+ prompts spanning the
   existing families (code / synthesis / grounded / creative) and length classes;
   re-run the capture + acceptance pipeline (`run_exactness_small_bundles.py` +
   `dspark_oracle/measure_acceptance_bundle.py`) at temp 0, with a 0.5/1.0 subset.
   The tooling exists; this is compute time, not new code.
2. **Realistic-trajectory acceptance.** From the captures, measure acceptance
   conditioned on cycle type: cycles anchored on a token the drafter had
   predicted vs cycles anchored on a correction (i.e., positions immediately
   following a p=1 miss). Report E[a|K] and S(K) for both populations and the
   trajectory-weighted mixture.
3. Re-emit the speedup-model inputs (`artifacts/spec_speedup_model/model_inputs.json`)
   from the widened corpus with CIs tight enough that the K=4 break-even verdict
   has a sign.

Estimated effort: mostly unattended compute; ~1–2 days of attended work.

## Success criteria

- 95% CI half-width on E[a|4] ≤ ~0.05 drafts/cycle (vs ±0.3 today), so the
  fixed-K break-even question and the lead 02 policy simulation both get a
  signed answer instead of a band.
- Post-rejection-cycle acceptance quantified; the speedup model updated with the
  trajectory-weighted acceptance (if it drops E[a|4] materially, that is itself
  a decision-grade negative finding and should be recorded as such).
- Updated per-position histogram feeding leads 01/02 re-runs at scale.
