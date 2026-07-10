## Verdict per Claim

- **C1 questionable**: code uses post-load `apply_model`, hooks layers 40/41/42, calls `mhc_post` on clones. Plausible. But no retained algebraic vLLM-vs-ds4 `after_ffn_hc` equality artifact. Prior gate’s “sane p1 is not semantic proof” is still true. See [dspark_hc_patch.py](/Users/lobanov/Projects/ds4-dspark-research/issue468/run_lead04_modal/dspark_hc_patch.py:82).
- **C2 questionable/wrong**: reported p1 arithmetic is right, but repo `validate_fidelity.py` still computes `total_match/total_positions`, the known wrong metric. See [validate_fidelity.py](/Users/lobanov/Projects/ds4-dspark-research/issue468/run_lead04_modal/validate_fidelity.py:110). GREEN is marginal and underpowered.
- **C3 wrong as stated**: `Δ=-0.061` is apples-to-oranges: FP uses 7 anchors/prompt, Q2 values use 8 anchors/prompt. Restricting Q2 to first 7 rows gives Q2 mean `0.7429`, Δ `-0.0286`, not `-0.061`. Either way, n=35 is too small for causal attribution.
- **C4 confirmed**: HOLD full Phase B is justified, but because evidence is incomplete/underpowered, not because the negative pilot proves “no FP headroom.”
- **C5 questionable**: meaningful-looking smoke, but no local FP pilot bundles/summaries retained. Shapes/replication/non-repeating are prose claims only in the checked-in worklog.
- **C6 wrong**: no retained Lead04 review outputs found in worktree or `/tmp`; only prose references to `/tmp/lead04_*.final.md`. This review is also not retained by definition.
- **C7 confirmed locally**: `rg` found secret names/env names, not live token-shaped strings. I could not inspect remote Modal secrets.

## p1 Measurement

Re-derived Q2 from retained local JSON:

- Q2 all anchors: `6/8, 6/8, 6/8, 6/8, 7/8` -> mean `31/40 = 0.775`.
- Reported FP implies: `6/7, 6/7, 4/7, 4/7, 5/7` -> mean `25/35 = 0.7143`.

So the reported `-0.061` is arithmetically `25/35 - 31/40`, but not a paired same-anchor comparison. With same first 7 Q2 rows: `26/35 = 0.7429`, Δ `-0.0286`.

GREEN is not honest as a validation conclusion: `25/35` has huge uncertainty, two prompts are `4/7 = 0.571`, and the original stricter rule `|p1 - 0.8125| <= 0.08` fails for the aggregate.

## FP Ceiling ≈ Q2?

Only as a weak pilot observation: no robust positive FP lift was shown. It is not a real “ceiling” for a retrained/calibrated drafter. If the drafter is calibrated to Q2 hiddens, native hiddens can plausibly hurt; that would make this a distribution-swap test, not an upper bound.

## Phase A Complete?

No. Blockers: no retained FP pilot artifacts, stale [STATUS.md](/Users/lobanov/Projects/ds4-dspark-research/issue468/STATUS.md:82), unresolved template/token offset and D1, wrong p1 code still present, no retained codex review outputs, no gate-2/go-no-go propagation.

## One Fix

Make the pilot reproducible: retain the converted FP bundles plus acceptance summaries, fix p1 to `1 - prefix_hist[0] / sum(prefix_hist)`, and recompute a same-anchor paired comparison.