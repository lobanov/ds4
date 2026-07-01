# First Recoverable-Gap Overlay Result

Date: 2026-07-01

## Purpose

Record the first real execution of the Step-1 recoverable-gap dataset path from
`59_oracle_only_quantization_search_goal.md`.

This note follows:

- `issue468/59_oracle_only_quantization_search_goal.md`
- `issue468/60_recoverable_gap_dataset_definition.md`

## Input data used

Sweep root:

- `/tmp/dspark_sweep8`

Baseline stepwise B2 label:

- `baseline-weighted4ctx_19t_default_256tr`

Oracle-side reference:

- `/tmp/dspark_sweep8/oracle-envelope-existing256.details.json`

Reason for using the envelope details instead of a single bundle-local
candidate label:

- the envelope is the better local approximation to the "oracle-only" notion of
  recoverable headroom
- it is built from the best per-step accepted values already observed across the
  existing candidate family

## Command used

```sh
python issue468/build_recoverable_gap_overlay.py \
  --sweep-root /tmp/dspark_sweep8 \
  --baseline-label baseline-weighted4ctx_19t_default_256tr \
  --oracle-details-json /tmp/dspark_sweep8/oracle-envelope-existing256.details.json \
  --out-label recoverable-gap-envelope-test
```

## Observed result

Summary emitted by the helper:

- contexts: `4`
- total steps: `76`
- recoverable steps: `17`
- recoverable fraction: `0.2237`
- mean positive gap: `0.1645`

Per-context recoverable-step counts:

- `ctx_08192`: `4`
- `ctx_16384`: `5`
- `ctx_24576`: `5`
- `ctx_32768`: `3`

Observed maximum normalized bundle weights:

- `ctx_08192`: `1.3282`
- `ctx_16384`: `1.0563`
- `ctx_24576`: `1.0496`
- `ctx_32768`: `1.0546`

## Interpretation

This is enough signal to justify the recoverable-gap branch as a real dataset,
not just a planning abstraction.

Two concrete observations matter:

1. recoverable signal is sparse
   - only about `22%` of anchor steps were flagged as recoverable under the
     first conservative rule
   - this is exactly the kind of concentration the oracle-only pivot expected

2. the strongest concentration is early
   - `ctx_08192` carries the largest max weight
   - later contexts still contain recoverable steps, but the first extracted
     envelope does not spread weight broadly

So the first routed-`Q4_K` search should prefer:

- sparse acceptance-sensitive weighting
- comparison against the older baseline-hardness family

not:

- broader uniform reweighting

## Practical next step

Use the integrated `recoverable-gap` source in
`run_dspark_weighted_from_sweep_root.py` to drive the first routed-`Q4_K`
re-quantization pass, with the oracle-envelope details file as the weighting
reference.
