# FP8 Torch Bundle Scoring Blocker

Date: 2026-07-01

## Purpose

Record the first bundle-level FP8 source-ceiling scores from the torch fallback
reference path, and explain why they should **not** yet be treated as the true
source ceiling.

This note follows `80_fp8_torch_fallback_harness_revival.md`.

## New tooling

Added:

- `issue468/ref/score_ref_bundle_torch.py`

Purpose:

- load the torch-fallback reference model once
- walk a full capture bundle sequentially with increasing `start_pos`
- preserve DSpark KV state across captured decode steps
- score the resulting base logits with:
  - greedy accepted-prefix statistics
  - analytical `1 - TV` upper-bound statistics
  - Monte Carlo B2 committed / accepted statistics

The script now supports both bundle layouts:

- sweep-root style:
  - `target_topk.json`
  - `target_greedy.json`
- older low-fill bundle style:
  - `target_topk200.json`
  - `greedy25_tokens.json`

## First sweep-root result

Bundle:

- `/tmp/dspark_sweep8/ctx_08192`

Reference-path output:

- `average_committed = 2.802220394736842`
- `average_accepted = 1.8388157894736843`
- `greedy_avg_prefix = 2.1578947368421053`
- `analytical_committed_upper_bound = 2.7601127974777784`

Saved JSON:

- `/tmp/dspark_sweep8/ctx_08192/ref-fp8-torch.b2.json`

Existing baseline on the same bundle:

- `/tmp/dspark_sweep8/ctx_08192/baseline.b2.json`
- `average_committed = 4.473684210526316`
- `average_accepted = 4.072368421052632`

Delta vs baseline:

- committed: about `-37.36%`

That is implausible for a supposed source-side ceiling.

## Low-fill diagnostic result

Bundle:

- `~/ds4/issue468/baseline/dspark_capture`

Reference-path output:

- `average_committed = 2.538651315789474`
- `average_accepted = 1.5785361842105263`
- `greedy_avg_prefix = 1.5789473684210527`
- `analytical_committed_upper_bound = 2.70082411007089`

Saved JSON:

- `/tmp/dspark_ref_lowfill_fp8.b2.json`

Why this matters:

- the older low-fill bundle removes the sweep-root-specific objection that the
  reference path lacks the 128 real prompt-window anchors
- yet the torch fallback still lands below the older documented oracle/F32
  level of about `2.80` committed from `18_verdict_negative_result.md` and
  `25_assignment1_measurement_integrity.md`

So the weak sweep-root result cannot be explained **only** by missing prompt
window state.

## Most likely causes

Two independent issues now look real.

### 1. Sweep-root mismatch: missing real prompt-window cache state

The sweep-root bundles contain:

- target hidden captures for generated positions only

They do **not** contain:

- the preceding prompt-window `main_hidden` states needed to reconstruct the
  DSpark window cache exactly at the first scored decode step

The live baseline on `ctx_08192` already has those anchors resident, while the
reference scorer starts from:

- prefill of only the first generated anchor

That will depress the sweep-root reference score.

### 2. The torch fallback itself is not yet faithful enough

The low-fill diagnostic shows the problem survives even when the prompt-window
objection is largely removed.

That points to remaining fidelity risk in at least one of:

- activation quantization emulation
- FP4/FP8 dequant / arithmetic semantics
- sparse attention / HC path details
- source-checkpoint interpretation relative to the existing numpy oracle

## Interpretation

The torch-fallback branch achieved a real unblock:

- source reference code can run on DGX without tilelang

But as a **measurement** branch it is now exhausted.

Reason:

- its first bundle-level numbers are too implausible to use as the FP8 source
  ceiling
- the low-fill diagnostic weakens the main alternative explanation
  ("only missing prompt cache")

So the current torch fallback is a useful execution scaffold, but not yet a
trustworthy oracle-level score path.

## Recommended next move

Switch the FP8 ceiling branch away from torch-fallback scoring and toward a
pure numpy / oracle-style source path.

Why this is the best next avenue:

1. the user explicitly authorized falling back to numpy if torch failed as a
   trustworthy path
2. the existing validated numpy oracle already encodes the DSpark structure and
   acceptance methodology
3. a source-weight import / dequant path into the oracle is more aligned with
   the goal than spending more time tuning the fallback-kernel approximation

Concretely, the next branch should target:

- importing the source FP8 / FP4 checkpoint semantics into an oracle-style
  forward path
- first matching the known low-fill legacy bundle
- only then revisiting the sweep-root ceiling question
