You are a skeptical senior engineer doing an ADVERSARIAL independent review. A
previous investigator (another agent) reached a conclusion you must challenge.
Re-derive every claim from the data; run code to verify where you can. Be terse.

WORKTREE: /Users/lobanov/Projects/ds4-dspark-research. Python env:
`issue468/dspark_oracle/.venv` (numpy 2.5.1). Relevant paths:
- canonical report: `issue468/summaries/stage1_tap_precision.md`
- comparison artifact: `issue468/artifacts/exactness_small_bundles_q4tap/comparison_vs_baseline/comparison.json` (+ `.csv`)
- Stage 1 bundles (Q4-tap): `issue468/artifacts/exactness_small_bundles_q4tap/*__t0p0/`
- baseline bundles (IQ2XXS): `issue468/artifacts/exactness_small_bundles/*__t0p0/`
- capture harness: `issue468/run_exactness_small_bundles.py`; comparison tool: `issue468/run_stage1_q4tap_compare.py`
- Stage 0 report (context, already codex-reviewed): `issue468/summaries/quant_mismatch_diagnostic.md`
- speedup model (anchors): `issue468/summaries/spec_speedup_model.md`

RECAP: Testing whether raising the drafter's tap-layer precision improves draft
acceptance. Re-captured layer-40/41/42 hidden states + greedy tokens from a Q4-tap
target variant (`Layers37-42Q4KExperts`; layers 37–42 routed experts at Q4_K,
everything else IQ2XXS, same AProjQ8/SExpQ8/OutQ8) on the same 10 temp=0 prompts,
seed=2, and re-measured the unchanged DSpark drafter's acceptance. Claimed
conclusion: raising tap-layer precision does NOT materially improve acceptance
(p=1 match 0.8125→0.7875, within noise; E[a|5block] +0.10 but per-prompt 4 up/3
down/3 flat = noise) → "tap-localized quant mismatch is falsified"; the deficit is
reframed as shallow drafter-vs-target calibration error, with fine-tuning as the
only viable lever (ceiling ~+10pp at p=1 from Stage 0 top-2 coverage).

THE INVESTIGATOR'S CLAIMS (verify, do not trust):
- D1 FRESH CAPTURE: the Q4-tap main_hidden differs from baseline (max abs diff
  59.5, mean 1.09, not identical) and 33/140 target tokens differ → genuinely a
  Q4-tap run, not reused baseline numbers.
- D2 CLEAN A/B: same 10 prompts, temp=0, seed=2, ctx=4096, 14 tokens; only the
  target (hidden states + ground-truth tokens) changed; drafter/embed/lm_head
  unchanged. prompt_tokens identical per prompt.
- D3 p=1 FLAT: p=1 match 0.8125→0.7875 (65/80→63/80), within binomial noise
  (σ≈0.045) → no material p=1 effect.
- D4 E[a|5block] +0.10 IS NOISE: per-prompt deltas 4 up / 3 down / 3 flat; the
  +0.10 mean is driven by deeper-position movement in a few prompts and is not a
  consistent effect.
- D5 FALSIFICATION: "tap-localized quant mismatch is falsified" — raising the
  drafter's input layers (40/41/42) to Q4 did not convert rank-2 misses into
  rank-1 matches.
- D6 REFRAME: the deficit is shallow drafter-vs-target calibration error; the only
  viable lever is a fine-tune on the served target's distribution, ceiling ~+10pp
  at p=1 (Stage 0 top-2 coverage 0.91).

PREMISES / SCOPE (exogenous — do NOT flag as bugs; assess (a) realizability and
(b) how conclusions swing if a premise fails):
- P1 The Q4-tap variant raises ONLY layers 37–42 routed experts to Q4_K; layers
  1–36 stay IQ2XXS, and layers 40–42 still receive Q2-corrupted activations from
  below. So the variant CANNOT fully isolate tap-precision from lower-layer Q2.
- P2 Stage 0's shallow-error finding (median p=1 target-rank 1.0, top-2 coverage
  0.91) is taken as valid (independently codex-reviewed).
- P3 The speedup-model acceptance anchors (baseline E[a|5]=2.338) are valid.

YOUR MANDATE — challenge assumptions, find new avenues. Decisive claims will be
INDEPENDENTLY re-verified by the dispatcher before acceptance, so prioritize
runnable checks. Not exhaustive:
1. END-TO-END VALIDITY (D1/D2): independently confirm the Q4-tap captures are
   fresh (load two main_hidden arrays, diff them; count differing target tokens).
   Confirm prompt_tokens match per prompt. Is the A/B actually clean — did the
   drafter GGUF, embed/lm_head, seed, ctx, temps all stay fixed? Check the
   acceptance_summary rows: are draft tokens produced by the SAME drafter on
   DIFFERENT inputs (base vs q4tap)? Spot-check one prompt's drafts differ where
   inputs differ.
2. STATISTICAL POWER (D3/D4): with n=10 prompts × 8 steps = 80 p=1 positions, is
   "no material effect" defensible, or is the study underpowered to detect a real
   small effect? Compute a proper paired test on per-prompt E[a|5block] (base vs
   q4tap): paired t / Wilcoxon / sign test, and a bootstrap CI on the +0.10 mean.
   Is +0.10 really indistinguishable from zero, or is the investigator dismissing
   a real (if small) signal? Also: is p=1 n=80 enough to distinguish 0.8125 vs
   0.7875? (Compute the CI.)
3. THE FALSIFICATION (D5): "tap-localized quant mismatch falsified" — given P1
   (lower layers still Q2), is this OVERCLAIMED? A flat result is ambiguous between
   "not quant" and "quant but lower-layer." Does the investigator acknowledge this
   adequately? Is there a cheaper way to push on localization than a full Q8 target?
4. THE REFRAME (D6): "shallow calibration error → fine-tuning is the only lever,
   ceiling +10pp" — is the +10pp ceiling (Stage 0 top-2 coverage) a valid bound on
   what a fine-tune could achieve at p=1, or does it conflate "target in drafter
   top-2" with "fine-tune can make target the drafter top-1"? Could a fine-tune do
   better OR worse than the top-2 coverage suggests? Is "calibration error" itself
   an inference beyond the data (the data shows shallow disagreement, not its cause)?
5. THE +0.10 / DEEPER-POSITION STORY: the investigator says the E[a|5block] gain
   is "deeper-position noise in a few prompts." Re-derive the per-POSITION delta
   (p=1..5) aggregated across prompts from the acceptance_summary rows. Is there
   any position where Q4-tap consistently helps, that the "noise" framing hides?
6. SAMPLING/REPRESENTATIVENESS: temp=0 only, 10 short prompts (67–165 prompt
   tokens). Does the conclusion generalize? Note the speedup model's caveat that
   temp 0.5/1.0 lowers E[a|4].

CONSTRAINTS: do not modify tracked files. Run read-only commands and inline
scripts (python -c, rg, jq). The venv numpy is available. Do NOT re-run the ds4
capture (it loads a 98GB model); verify from persisted bundles and the comparison
artifact. The per-step acceptance rows are in each bundle's
`oracle/acceptance_summary.json` (rows[].draft, rows[].target, rows[].prefix).

OUTPUT (terse, evidence-based):
## Verdict per claim (D1..D6: sound / questionable / likely-wrong + 1-line why)
## Premise sensitivity (P1..P3: does the headline survive if it fails?)
## New experiments to try (ranked; each: what to run, expected signal, effort)
## Statistical re-analysis (paired test + CI on the +0.10 E[a|5block] and on p=1)
## Leading hypothesis after re-examination + the one decisive test for it
## Bottom line: does "no material tap-precision effect → fine-tuning is the lever"
   survive your review, and is the falsification claim (D5) overstated?
