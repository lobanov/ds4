# Independent Review After Oracle-Only Pivot

Date: 2026-07-01

## Purpose

Record the mandatory independent `gpt-5.5 xhigh` review required by
`issue468/53_frozen_runtime_quantization_goal.md` after the currently known
plain-tensor avenue was exhausted and the oracle-only search branch became the
lead program.

This note follows:

- `issue468/57_cuda_probe_unblock_and_plain_layout_compatibility.md`
- `issue468/58_oracle_only_pivot_from_euler_review.md`
- `issue468/59_oracle_only_quantization_search_goal.md`
- `issue468/60_recoverable_gap_dataset_definition.md`
- `issue468/61_first_recoverable_gap_overlay_result.md`
- `issue468/62_split_execution_plan_after_metal_only_imatrix_blocker.md`

## Review verdict

The independent review concluded that the current avenue set is **not
exhausted**.

What is exhausted:

- generic weighted `Q4_K + imatrix` variants from the earlier branch
- broad plain-layout `F32` promotion under the frozen runtime
- `hc_head_fn`-only `F32` as a lead branch

What is **not** exhausted:

- oracle-first artifact-side quantization search after the pivot in `58`

So the branch should not be treated as ready for a final negative conclusion.

## Main flaws in the current interpretation

### 1. The current recoverable-gap proxy is not yet the right oracle

The first recoverable-gap overlay in `61` is built from the
`oracle-envelope-existing256.details.json` q-dump envelope, not from a true
same-checkpoint `F32` oracle.

That makes it useful as a first concentration signal, but risky as the main
calibration truth because:

- it inherits limitations of the already-tested candidate family
- it can overfit to the envelope's best-observed candidates
- it is weaker than the intended "baseline loses, same-checkpoint oracle wins"
  definition from `59`

### 2. The first overlay is probably too mild

The current overlay marks only `17 / 76` anchor steps as recoverable, and most
normalized weights remain close to `1.0`.

Interpretation:

- the dataset signal is real
- but the weighting may be too soft to force a meaningfully different routed
  `Q4_K` candidate

### 3. Runtime-vs-validator compatibility remains easy to conflate

`57` already showed that validator-compatible plain tensors are not necessarily
execution-compatible on the frozen runtime path.

This remains an explicit interpretation hazard for any follow-up that promotes
plain tensors without checking the actual DSpark kernel call sites.

## Code/doc corrections surfaced by the review

### 1. `62` is stale after `ecb8773`

`62` correctly documented the first blocker at the time it was written, but it
is no longer the current code state.

Current code now shows:

- [ds4_cli.c](/Users/lobanov/Projects/ds4-dspark-drafter/ds4_cli.c:1588) no
  longer forces `--imatrix-out` to Metal
- [ds4.c](/Users/lobanov/Projects/ds4-dspark-drafter/ds4.c:25926) now accepts
  any initialized graph backend for imatrix collection

So split execution is no longer mandatory for that specific reason. It may
still be convenient operationally, but it is no longer a code-level backend
requirement.

### 2. Weighted runner reprobe bug

The review found a real bug in
[issue468/run_dspark_weighted_from_sweep_root.py](/Users/lobanov/Projects/ds4-dspark-drafter/issue468/run_dspark_weighted_from_sweep_root.py:229):

- `reprobe_bundle(...)` referenced `args.ds4_bin`
- but `args` is local to `main()`

That bug has been fixed in the current branch by passing `ds4_bin` explicitly
into `reprobe_bundle(...)`.

## New in-scope avenues the review generated

### 1. Build the true `F32`-oracle recoverable-gap dataset

Replace the q-dump envelope proxy with the intended same-checkpoint `F32`
oracle where possible.

Minimum rule:

- construct recoverable states from "baseline loses, same-checkpoint `F32`
  oracle wins"

And keep train/holdout separation so the overlay is not tuned on its own
evaluation states.

### 2. Try hard recoverable-step collection, not only soft weighting

New collector shapes to test:

- collect only recoverable steps
- or oversample recoverable steps by about `5x` to `10x`

Reason:

- this is materially different from the current mild overlay
- it is still fully budget-feasible

### 3. Acceptance-aware local routed-`Q4_K` block search

Do local block search only on routed expert blocks exercised during
recoverable states.

Reason:

- footprint-neutral
- still frozen-runtime-plausible
- different from generic activation-preservation imatrix

### 4. Source-FP4-aware routed-`Q4_K` initialization

Instead of dequantizing to generic `F32` and then re-quantizing with generic
GGML heuristics, seed the routed-expert `Q4_K` blocks from the source FP4 scale
structure.

Reason:

- the source model is already mixed-precision by tensor class
- this may preserve expert-local dynamic range better than generic requant

### 5. Dense legal `Q8_0` activation-aware requantization

Still in scope:

- `main_proj`
- attention projections
- shared expert projections

Reason:

- these are legal artifact-side levers
- they remain largely untouched by the current routed-expert-first branch

### 6. Calibrated `F16` rounding on runtime-F16 plain tensors

Lead tensor remains:

- `ffn_gate_inp`

Reason:

- it changes router-sensitive bits without changing runtime type or footprint

## Updated operative conclusion

The completed independent review does **not** support declaring the
quantization-only branch exhausted.

The right next interpretation is narrower:

- earlier generic weighted-imatrix and broad plain-`F32` avenues are exhausted
- oracle-first artifact-side search is still underexplored
- any final negative verdict remains premature until the new avenues above are
  checked or ruled out
