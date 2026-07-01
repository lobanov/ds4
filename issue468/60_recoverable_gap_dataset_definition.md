# Recoverable-Gap Dataset Definition

Date: 2026-07-01

## Purpose

Turn Step 1 from `59_oracle_only_quantization_search_goal.md` into a concrete
artifact definition that can drive the existing imatrix collector workflow.

The key idea is to stop weighting anchor steps by generic hardness alone and
instead weight them by **recoverable accepted-token loss**:

- the live baseline underperforms on the step
- the oracle does better on the same checkpointed step

This note defines that dataset in repo-local terms.

## Dataset unit

One dataset item is one DSpark anchor step inside one captured context bundle:

- bundle: `ctx_#####`
- anchor step: `1..N`

For each such step we need two existing measurements:

1. baseline B2 accepted length
2. oracle-side B2 accepted length

Both are already represented in bundle-local `*.b2.json` files via:

- `per_step_accepted`

## Recoverable step rule

A step is treated as recoverable if both are true:

1. the baseline is not already effectively solved
2. the oracle beats the baseline by a non-trivial margin

Concrete default rule in the first implementation:

- baseline hard if `baseline_accepted < 4.999`
- recoverable if `oracle_accepted - baseline_accepted >= 0.05`

This is intentionally conservative:

- trivial numerical noise does not create signal
- already-maxed steps do not absorb collector budget

## Weighting rule

For each recoverable step:

- `gap = oracle_accepted - baseline_accepted`
- normalize to accepted-length scale: `gap / 5.0`
- raw weight: `1 + alpha * normalized_gap`

For non-recoverable steps:

- raw weight: `1.0`

Then normalize the bundle's raw weights back to unit mean and clamp:

- floor: `0.5`
- ceil: `2.0`

This preserves compatibility with the existing weighted-imatrix overlay shape:

- one `imatrix_anchor_weights.txt` per bundle

## Stronger collector modes

The initial implementation used only the soft normalized-gap weighting above.

After the independent review in `63`, the helper now supports stronger
collector-compatible variants:

- `soft`
  - the original normalized-gap weighting
- `binary`
  - recoverable steps get raw weight `1.0`
  - non-recoverable steps get a small positive raw weight
- `boosted`
  - recoverable steps get a large raw weight such as `8.0`
  - non-recoverable steps get a small positive raw weight such as `0.05`

Why not use literal zero for non-recoverable steps:

- the DSpark collector currently rejects non-positive anchor weights

So "collect only recoverable steps" must be approximated as:

- very small positive non-recoverable weight
- large recoverable/non-recoverable ratio

This is still aligned with the review's intent:

- heavily concentrate collector budget on recoverable states
- without changing collector code

## Why this is different from the old hardness weighting

The earlier hardness overlay used:

- baseline accepted length only

That mixes together:

- fundamentally hard states
- quantization-sensitive but potentially recoverable states
- states where the oracle offers no real improvement

The recoverable-gap dataset filters that down to:

- states where a model-side quantization improvement has already been shown to
  matter in the oracle

So the collector signal becomes:

- not "this step is hard"
- but "this step is hard **and** the current quantization appears to be leaving
  acceptance on the table"

## First implementation

New helper:

- `issue468/build_recoverable_gap_overlay.py`

Inputs:

- sweep root with `ctx_#####` bundles
- one baseline `*.b2.json` label
- and either:
  - one oracle `*.b2.json` label
  - or one top-level oracle-envelope `*.details.json`

Outputs:

1. overlay root:
   - `.<out-label>.overlay/ctx_#####/imatrix_anchor_weights.txt`
2. manifest:
   - `<out-label>.anchor_weights.json`

The overlay mirrors the existing weighted-collector format, so it can feed the
same collector path used by `run_dspark_weighted_from_sweep_root.py`.

## Example usage

Bundle-local oracle label example:

```sh
python issue468/build_recoverable_gap_overlay.py \
  --sweep-root /tmp/dspark_sweep8 \
  --baseline-label baseline-weighted4ctx_19t_default_256tr \
  --oracle-label candidate-weighted4ctx_19t_default_256tr \
  --out-label recoverable-gap-default
```

This does **not** assume that the oracle label is deployable. It only assumes:

- the oracle label is a better model-side reference on the same anchor steps

In other words, `--oracle-label` must name a **bundle-local** `*.b2.json`
series, not a top-level sweep summary file.

Oracle-envelope details example:

```sh
python issue468/build_recoverable_gap_overlay.py \
  --sweep-root /tmp/dspark_sweep8 \
  --baseline-label baseline-weighted4ctx_19t_default_256tr \
  --oracle-details-json /tmp/dspark_sweep8/oracle-envelope-existing256.details.json \
  --out-label recoverable-gap-envelope
```

Hard recoverable-step emphasis example:

```sh
python issue468/build_recoverable_gap_overlay.py \
  --sweep-root /tmp/dspark_sweep8 \
  --baseline-label baseline-weighted4ctx_19t_default_256tr \
  --oracle-details-json /tmp/dspark_sweep8/oracle-envelope-existing256.details.json \
  --mode boosted \
  --recoverable-boost 8.0 \
  --nonrecoverable-weight 0.05 \
  --out-label recoverable-gap-envelope-boosted
```

This second form is the more faithful oracle-only path when the best available
model-side reference is a top-level per-step envelope rather than a single
bundle-local candidate label.

## What this enables next

This dataset definition is the bridge from `59` Step 1 to Step 2:

1. build recoverable-gap overlays from current baseline/oracle evidence
2. feed those overlays into routed `Q4_K` expert re-quantization first
3. compare against the earlier hardness-only collector family

That is the first concrete oracle-only search path that remains aligned with the
current plan.
