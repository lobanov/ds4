# Split Execution Plan After the Metal-Only Imatrix Blocker

Date: 2026-07-01

## Status

This note is now partially superseded by commit `ecb8773`
(`issue468: allow cuda dspark imatrix collection`).

It remains accurate as the first blocker encountered, but the specific
"imatrix collection requires Metal" code-path conclusion is no longer current.
See:

- `issue468/63_independent_review_after_oracle_pivot.md`
- `issue468/65_cuda_imatrix_collection_probe_blocker.md`

## Purpose

Record the first execution blocker encountered when trying to push the
recoverable-gap `Q4_K` branch directly on DGX, and define the revised workflow
that still aligns with `59_oracle_only_quantization_search_goal.md`.

This note follows:

- `issue468/59_oracle_only_quantization_search_goal.md`
- `issue468/61_first_recoverable_gap_overlay_result.md`

## What was attempted

Tried to run the first recoverable-gap weighted `Q4_K` search pass remotely on
DGX using:

- the staged `ctx_08192` bundle root at `/tmp/dspark_sweep8`
- recoverable-gap weighting from
  `/tmp/dspark_sweep8/oracle-envelope-existing256.details.json`
- CUDA reprobe backend

The weighted runner reached the imatrix collection phase and failed before
quantization.

## Actual blocker

The failure is not in the new recoverable-gap logic. It is in the existing
collector backend constraint.

Observed code path:

- `ds4_cli.c`
  - `--imatrix-out` forcibly sets `c.engine.backend = DS4_BACKEND_METAL`
- `ds4.c`
  - `ds4_engine_collect_imatrix(...)` rejects any non-Metal backend:
    - `ds4: imatrix collection currently requires --metal`

Observed DGX runtime result:

- CUDA build receives the imatrix command line
- but the CLI forces the backend back to Metal
- startup then fails on Linux with:
  - `ds4: Metal backend requested but this build is linked with CUDA, not Metal`

## Interpretation

This means the current routed-`Q4_K` oracle workflow is not a single-machine
DGX-only path today.

The phases are split by backend support:

1. overlay / weighting construction
   - can run anywhere
2. imatrix collection
   - currently requires Metal
3. GGUF quantization
   - can run on DGX
4. CUDA reprobe / B2 measurement
   - can run on DGX

So the correct near-term execution model is:

- **split execution**, not "do everything on DGX"

## Workflow update landed

To support that split, `run_dspark_weighted_from_sweep_root.py` now supports:

- `--imatrix-in`
  - skip collection and use an existing imatrix file
- `--candidate-gguf`
  - skip quantization and use an existing candidate file
- `--skip-reprobe`
  - stop after collection / quantization and print produced paths
- `--quantizer-bin`
  - point at an explicit quantizer binary such as `/tmp/deepseek4-quantize`

This lets the branch run as:

### Phase A. Metal host

1. build recoverable-gap overlay
2. collect imatrix with the existing Metal-only path
3. optionally stop there with `--skip-reprobe`

### Phase B. DGX

1. quantize from `--imatrix-in` if needed
2. reprobe from `--candidate-gguf` or the newly quantized GGUF

## Why this is still aligned with `59`

`59` requires:

- recoverable-gap dataset first
- routed `Q4_K` branch first

It does **not** require that every phase execute on the same machine.

The split workflow preserves the actual research logic:

- oracle-derived weighting still drives the first Lane A branch
- DGX still performs the heavy reprobe/measurement step
- the only Metal-only portion is the existing collector backend

## Practical next step at time of writing

Use the new split runner shape to:

1. collect the first recoverable-gap imatrix on a Metal-capable host
2. hand the resulting `imatrix.dat` to DGX via `--imatrix-in`
3. run the first CUDA reprobe comparison against baseline
