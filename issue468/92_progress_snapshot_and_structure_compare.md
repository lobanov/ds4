# Progress Snapshot and Baseline-vs-Best Structure Compare

Date: 2026-07-02

## Purpose

Record the current branch state after the first oracle-side and runtime-side
follow-up passes, and pin down the exact structural relationship between:

- the shipped baseline drafter `dspark.gguf`
- the current best single measured quantized artifact

This note is meant to answer two practical questions:

1. how much acceptance has actually been recovered so far
2. whether the current best artifact changes the drafter footprint or tensor
   layout

## Current measured acceptance reference points

On the currently completed same-checkpoint `2ctx` slice (`ctx_08192`,
`ctx_16384`):

| artifact | `ctx_08192` accepted | `ctx_16384` accepted | 2ctx mean accepted |
|---|---:|---:|---:|
| source-checkpoint numpy oracle | `3.973684210526316` | `4.217105263157895` | `4.0953947368421055` |
| baseline `dspark.gguf` | `4.072368421052632` | `4.2368421052631575` | `4.154605263157895` |
| best single measured quantized artifact | `4.19921875` | `4.2386924342105265` | `4.218955592105264` |

Read:

- the current best measured deployable artifact is above the live baseline on
  this `2ctx` slice
- the gain is still small:
  - `+0.06435032894736892` accepted tokens per 5-token block vs baseline
  - about `+1.549%` relative vs baseline on the 2-context mean
- the current same-checkpoint source-import oracle is **not** above the live
  baseline on these two contexts, so it should be treated as a different
  branch discriminator rather than a simple upper envelope

## Current best single measured artifact

Current leader:

- `/private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_layer2gateup_mainproj_q8imat.gguf`

How it was formed:

1. start from the positive selective routed-expert splice
   `recoverablegap_boosted2ctx_layer2gateup.gguf`
2. compose in the `mtp.0.main_proj.weight` payload from
   `recoverablegap_boosted2ctx_mainproj_q8imat.gguf`

Measured `2ctx` result:

- `ctx_08192`: `4.19921875`
- `ctx_16384`: `4.2386924342105265`
- mean: `4.218955592105264`

So this is currently the best single artifact that is actually constructible,
measured, and footprint-feasible.

## Structure comparison: baseline vs best artifact

Compared files:

- baseline: `../ds4/gguf/dspark.gguf`
- best artifact:
  `/private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_layer2gateup_mainproj_q8imat.gguf`

Result:

- tensor metadata structure is identical
- same tensor count: `81`
- same tensor names
- same tensor types
- same per-tensor byte sizes
- same per-layer aggregate sizes
- same total payload size

So this artifact is footprint-neutral relative to baseline in the strict
GGUF-structure sense.

## Payloads that actually changed

Only `3` tensor payloads differ:

| tensor | layer | type | size |
|---|---|---|---:|
| `mtp.0.main_proj.weight` | `mtp.0` | `Q8_0` | `51.000 MiB` |
| `mtp.2.ffn_gate_exps.weight` | `mtp.2` | `Q4_K` | `1152.000 MiB` |
| `mtp.2.ffn_up_exps.weight` | `mtp.2` | `Q4_K` | `1152.000 MiB` |

Interpretation:

- the `main_proj_q8imat` branch changes one dense `Q8_0` tensor payload
- the `layer2gateup` branch changes two routed `Q4_K` expert tensors
- no tensor class promotion was required
- no additional tensors were introduced
- no tensor grew in size

## Per-layer aggregate structure

| layer | tensor count | baseline size | candidate size | delta |
|---|---:|---:|---:|---:|
| `mtp.0` | `26` | `3.559013 GiB` | `3.559013 GiB` | `0` |
| `mtp.1` | `24` | `3.509193 GiB` | `3.509193 GiB` | `0` |
| `mtp.2` | `31` | `3.632629 GiB` | `3.632629 GiB` | `0` |
| `total` | `81` | `10.700835 GiB` | `10.700835 GiB` | `0` |

Type mix by total payload remains:

- `Q4_K`: `10.125 GiB`
- `Q8_0`: `0.442017 GiB`
- `BF16`: `0.123299 GiB`
- `F16`: `0.010376 GiB`
- `F32`: `0.000143 GiB`

So the current best artifact is still fundamentally the same DSpark recipe:

- routed experts dominated by `Q4_K`
- a small dense `Q8_0` side budget
- tiny plain-tensor overhead

## Progress read

What is now established:

1. there is a real positive branch above baseline, but it is small
2. that branch does not require a larger drafter footprint
3. the best measured artifact is still structurally the same DSpark GGUF recipe,
   with only three payload substitutions

What is not established:

1. a path to the assignment-scale `+10%` quality target
2. a source-import oracle ceiling that cleanly dominates the live baseline
3. broad evidence that nearby dense or source-copy perturbations will close the
   gap

So the branch status remains:

- positive enough to keep acceptance-aware local search alive
- not yet strong enough to claim a major quality improvement path

## Implication for next work

The current comparison strengthens the case for continuing to search inside the
same footprint-neutral recipe family:

- acceptance-aware routed `Q4_K` local search around the positive
  `layer2gateup` seed
- small compositional additions like `main_proj_q8imat`

because the best observed gain so far came from exactly that kind of
same-structure payload substitution, rather than from any larger mixed-precision
or footprint-expanding move.
