# Same-Checkpoint Ref-Oracle Recoverable-Gap Dataset — First 2ctx Read

Date: 2026-07-02

## Purpose

Use the new same-checkpoint reference-oracle path to build a real
recoverable-gap details file for the contexts already measured in `83`, instead
of relying only on the older oracle-envelope proxy.

This is a direct follow-up to the critique in `63` that the existing
recoverable-gap overlays were still calibrated against an envelope assembled
from prior candidate runs rather than a same-checkpoint oracle.

## New tooling

Added:

- `issue468/ref/build_oracle_details_json.py`

Purpose:

- convert one or more oracle-side `*.b2.json` files into the compact
  `[{context, per_step_sources, per_step_oracle_envelope}, ...]` format already
  consumed by `issue468/build_recoverable_gap_overlay.py`

So the ref-checkpoint oracle path can now feed the existing recoverable-gap
overlay machinery directly, without another schema fork.

## Inputs used

Existing same-checkpoint ref-oracle B2 outputs:

- `/tmp/ref-oracle-fp8-ctx08192-19.b2.json`
- `/tmp/ref-oracle-fp8-ctx16384-19.b2.json`

Existing 2-context sweep root:

- `/private/tmp/dspark_sweep2ctx`

Existing envelope-proxy details for comparison:

- `/private/tmp/dspark_sweep2ctx/oracle-envelope-existing256.details.json`

## Commands

Build the ref-oracle details file:

```sh
python3 issue468/ref/build_oracle_details_json.py \
  --context-b2 8192=/tmp/ref-oracle-fp8-ctx08192-19.b2.json \
  --context-b2 16384=/tmp/ref-oracle-fp8-ctx16384-19.b2.json \
  --default-source ref-oracle-fp8 \
  --out-json /private/tmp/dspark_sweep8/ref-oracle-fp8-2ctx.details.json
```

Build the first 2-context boosted recoverable-gap overlay from the real
same-checkpoint oracle:

```sh
python3 issue468/build_recoverable_gap_overlay.py \
  --sweep-root /private/tmp/dspark_sweep2ctx \
  --baseline-label baseline \
  --oracle-details-json /private/tmp/dspark_sweep8/ref-oracle-fp8-2ctx.details.json \
  --out-label recoverablegap_reforacle2ctx_test \
  --steps-cap 19 \
  --mode boosted \
  --recoverable-boost 8 \
  --nonrecoverable-weight 0.05
```

Control overlay from the older envelope-proxy details:

```sh
python3 issue468/build_recoverable_gap_overlay.py \
  --sweep-root /private/tmp/dspark_sweep2ctx \
  --baseline-label baseline \
  --oracle-details-json /private/tmp/dspark_sweep2ctx/oracle-envelope-existing256.details.json \
  --out-label recoverablegap_envelope2ctx_test \
  --steps-cap 19 \
  --mode boosted \
  --recoverable-boost 8 \
  --nonrecoverable-weight 0.05
```

## Results

### 1. The ref-oracle details file is live

Output created:

- `/private/tmp/dspark_sweep8/ref-oracle-fp8-2ctx.details.json`

Contexts covered:

- `ctx_08192`
- `ctx_16384`

### 2. The same-checkpoint oracle is less optimistic than the envelope proxy

Per-context average accepted tokens:

| context | envelope proxy | ref oracle |
|---|---:|---:|
| `ctx_08192` | `4.166324013157895` | `3.973684210526316` |
| `ctx_16384` | `4.2485608552631575` | `4.217105263157895` |

### 3. Recoverable-step count drops under the real oracle

Boosted 2ctx overlay summary:

| oracle source | recoverable steps | recoverable fraction | mean positive gap |
|---|---:|---:|---:|
| envelope proxy | `11 / 38` | `0.2894736842105263` | `0.37109375` |
| ref oracle | `8 / 38` | `0.21052631578947367` | `0.53515625` |

Interpretation:

- the real same-checkpoint oracle marks **fewer** steps as recoverable
- but the remaining recoverable steps have a **larger mean positive gap**

So the older envelope proxy appears broader but noisier, while the ref-oracle
dataset is narrower and harder.

### 4. The step set changes materially

At `ctx_08192`:

- ref-oracle recoverable steps:
  - `2, 4, 5, 6, 11`
- envelope-only steps:
  - `9, 16, 19`
- ref-only step:
  - `5`

At `ctx_16384`:

- ref-oracle recoverable steps:
  - `1, 6, 7`
- envelope-only step:
  - `3`

So the difference is not only a rescaling of the old weights. The actual
recoverable step identity changes.

## Read

This is the first concrete evidence that `63` was directionally right:

- the existing recoverable-gap proxy is usable as a first concentration signal
- but it is not equivalent to the same-checkpoint oracle

The real oracle dataset is:

- smaller
- harsher
- and likely a better calibration target for acceptance-aware local search

## Decision impact

What this does support:

- move future routed-local acceptance-aware work toward the same-checkpoint
  oracle details format
- treat prior envelope-calibrated results as weaker evidence than previously
  assumed

What this does **not** yet support:

- a full 4-context replacement of the old overlay

Current limit:

- only `ctx_08192` and `ctx_16384` currently have local ref-oracle B2 outputs
  on disk

So the next clean extension is:

1. complete the missing ref-oracle per-context runs for `ctx_24576` and
   `ctx_32768`
2. build the 4ctx same-checkpoint details file
3. rerun the recoverable-gap overlay and any routed-local follow-up against
   that harder calibration set
