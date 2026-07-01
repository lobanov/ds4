# Stronger Recoverable-Gap Overlay Modes

Date: 2026-07-01

## Purpose

Record the first Step-1 follow-up landed after the independent review in `63`:
the recoverable-gap overlay path no longer supports only mild soft weighting.

It now also supports stronger collector-compatible emphasis on recoverable
steps, which keeps the current imatrix collector and quantizer flow unchanged
while testing a meaningfully different routed-`Q4_K` search hypothesis.

This note follows:

- `issue468/60_recoverable_gap_dataset_definition.md`
- `issue468/61_first_recoverable_gap_overlay_result.md`
- `issue468/63_independent_review_after_oracle_pivot.md`

## Problem addressed

`63` highlighted that the first overlay from `61` was probably too mild:

- only `17 / 76` steps were marked recoverable
- most normalized weights stayed close to `1.0`

That creates a real risk that the first routed-`Q4_K` pass is too close to the
earlier weighted-imatrix family to count as a distinct branch result.

## What landed

`issue468/build_recoverable_gap_overlay.py` and
`issue468/run_dspark_weighted_from_sweep_root.py` now support:

- `--mode soft|binary|boosted`
- `--recoverable-boost`
- `--nonrecoverable-weight`

Meaning:

### `soft`

Original behavior:

- recoverable steps scale with normalized oracle-baseline gap
- non-recoverable steps keep raw weight `1.0`

### `binary`

Harder recoverable-step concentration:

- recoverable steps use raw weight `1.0`
- non-recoverable steps use a small positive raw weight

### `boosted`

Strongest collector-compatible version:

- recoverable steps use a large raw weight such as `8.0`
- non-recoverable steps use a small positive raw weight such as `0.05`

## Why non-recoverable steps are not zeroed

The current collector rejects non-positive anchor weights.

So the closest no-code-change approximation to:

- "collect only recoverable steps"

is:

- tiny positive non-recoverable weight
- large recoverable/non-recoverable ratio

This keeps the execution path frozen while still testing the review's stronger
collector-shaping idea.

## Why this matters for `59`

This still belongs squarely in Step 1 and Step 2 of `59`:

1. refine the recoverable-gap dataset
2. feed that dataset into routed `Q4_K` expert re-quantization first

But it is now a genuinely new branch shape rather than just another mild
hardness-style weighting variant.

## Practical next step

Sync the branch to DGX and run the first boosted recoverable-gap routed-`Q4_K`
pass against `/tmp/dspark_sweep8`, comparing:

- the original soft recoverable-gap overlay
- a boosted recoverable-gap overlay

If the boosted pass is still flat, that is a stronger negative result for the
current routed-`Q4_K` collector family than the earlier soft overlay alone.
