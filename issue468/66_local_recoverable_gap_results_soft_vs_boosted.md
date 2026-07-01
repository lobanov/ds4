# Local Recoverable-Gap Results: Soft 4ctx and Boosted 2ctx

Date: 2026-07-01

## Purpose

Record the first local Metal results after moving the imatrix collection stage
onto this machine for the oracle-only `59` branch.

This note captures three concrete outcomes:

1. the missing recoverable-gap imatrix artifacts were produced locally
2. the first soft routed-`Q4_K` candidate was measured on the first two
   contexts and was negative
3. the stronger boosted recoverable-gap branch improved that result on the same
   two-context slice, but still remained slightly negative

This note follows:

- `issue468/59_oracle_only_quantization_search_goal.md`
- `issue468/64_stronger_recoverable_gap_overlay_modes.md`
- `issue468/65_cuda_imatrix_collection_probe_blocker.md`

## Local artifact production that now succeeded

Using local Metal `ds4` with the existing baseline drafter
`../ds4/gguf/dspark.gguf`, the following imatrix files were successfully
collected:

- `/private/tmp/dspark_sweep8/recoverablegap_soft4ctx.imatrix.dat`
- `/private/tmp/dspark_sweep8/recoverablegap_boosted4ctx.imatrix.dat`
- `/private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx.imatrix.dat`

This unblocks the split-execution logic from `65`:

- local Metal for imatrix collection
- candidate build / reprobe on the real runtime path afterward

## Soft recoverable-gap routed-`Q4_K` result

Candidate:

- `/private/tmp/dspark_sweep8/recoverablegap_soft4ctx_local.gguf`

Measured contexts that completed:

### `ctx_08192`

- baseline accepted: `4.1865`
- candidate accepted: `4.1361`
- accepted delta: `-1.20%`
- baseline committed: `4.5726`
- candidate committed: `4.5259`
- committed delta: `-1.02%`

### `ctx_16384`

- baseline accepted: `4.2364`
- candidate accepted: `4.2292`
- accepted delta: `-0.17%`
- baseline committed: `4.5154`
- candidate committed: `4.5064`
- committed delta: `-0.20%`

Two-context mean:

- accepted delta: `-0.68%`
- committed delta: `-0.61%`

Interpretation:

- the first soft recoverable-gap branch is not improving quality on the clean
  early contexts
- but it is only mildly negative after the first context

## Soft branch longer-context failure

The same soft candidate did not complete cleanly at `ctx_24576`.

Observed failure during the candidate probe path:

- `Metal command batch failed: Insufficient Memory`
- and, on isolated retry with a slightly larger `--ctx`, later:
  - `Metal graph compressed KV cache capacity exceeded at layer 2`

Important distinction:

- baseline at `ctx_24576` completed and produced B2 output
- the candidate path failed before producing a usable q-dump

So this is candidate-side runtime instability on the long-context path, not a
generic local reprobe failure.

## Boosted recoverable-gap routed-`Q4_K` result on the clean 2ctx slice

To keep moving on the stronger branch without waiting on the unstable
`ctx_24576+` path, a two-context root was evaluated:

- `ctx_08192`
- `ctx_16384`

Candidate:

- `/private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_local.gguf`

### `ctx_08192`

- baseline accepted: `4.1865`
- candidate accepted: `4.1737`
- accepted delta: `-0.30%`
- baseline committed: `4.5726`
- candidate committed: `4.5676`
- committed delta: `-0.11%`

### `ctx_16384`

- baseline accepted: `4.2364`
- candidate accepted: `4.2342`
- accepted delta: `-0.05%`
- baseline committed: `4.5154`
- candidate committed: `4.5113`
- committed delta: `-0.09%`

Two-context mean:

- accepted delta: `-0.18%`
- committed delta: `-0.10%`

## Comparison: soft vs boosted on the same two contexts

Soft 2ctx mean:

- accepted delta: `-0.68%`
- committed delta: `-0.61%`

Boosted 2ctx mean:

- accepted delta: `-0.18%`
- committed delta: `-0.10%`

Interpretation:

- the boosted recoverable-gap shaping is directionally better than the soft
  version on the clean 2-context slice
- but it still does **not** cross above baseline

So the stronger collector shape helps, but not enough yet to make the first
Lane A routed-`Q4_K` branch positive.

## Updated branch read

Current status of the first routed-`Q4_K` recoverable-gap family:

- soft 4ctx branch:
  - negative on the first two measured contexts
  - candidate instability at `ctx_24576`
- boosted 2ctx branch:
  - meaningfully less negative than soft
  - still below baseline

This does **not** exhaust `59`, but it does narrow the next useful moves:

1. do not keep spending cycles on the same soft weighting family
2. treat stronger recoverable-step shaping as mildly helpful but insufficient so
   far
3. move next to a more distinct artifact lever from `63`, such as:
   - acceptance-aware local routed-`Q4_K` block search
   - or dense legal `Q8_0` requantization
