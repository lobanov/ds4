# Selective Routed-Expert Splice Results After Recoverable-Gap Hotspot Ranking

Date: 2026-07-01

## Purpose

Record the first selective-layer follow-up after the boosted recoverable-gap
donor from `66`.

This note answers a narrower question than `66`:

- if the full boosted donor is still slightly negative, can the routed-expert
  gain be recovered by taking only the highest-signal expert layers while
  keeping the same frozen runtime path and the same artifact footprint?

## Setup

Starting point:

- boosted donor:
  - `/private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_local.gguf`
- base drafter:
  - `../ds4/gguf/dspark.gguf`
- evaluation root:
  - `/private/tmp/dspark_sweep2ctx`
  - containing `ctx_08192` and `ctx_16384`

Hotspot ranking came from the new imatrix summary on:

- `/private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx.hotspots.json`

Top routed-expert layers by summed recoverable-gap overlay mass were:

1. layer `2`
2. layer `0`
3. layer `1`

Using `gguf-tools/mixed/splice_mixed_expert_layers_gguf.py`, four selective
donors were built by copying only the routed expert tensors for selected
layers from the boosted donor back into the baseline drafter:

- `layer2`
- `layer02`
- `layer0`
- `layer1`

All kept the same tensor payload size as the baseline drafter:

- about `10.70 GiB` tensor payload
- within the existing frozen-artifact budget

## Measured results

### Full boosted donor reference from `66`

Two-context mean:

- accepted delta: `-0.1782%`
- committed delta: `-0.0995%`

This is the comparison point for all selective splices below.

### Layer `2` only

Candidate:

- `/private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_layer2.gguf`

`ctx_08192`:

- accepted delta: `+0.0638%`
- committed delta: `+0.1529%`

`ctx_16384`:

- accepted delta: `-0.1844%`
- committed delta: `-0.1457%`

Two-context mean:

- accepted delta: `-0.0603%`
- committed delta: `+0.0036%`

Interpretation:

- this is materially better than the full boosted donor
- it is the first selective branch to reach effectively flat-to-slightly
  positive committed quality on the clean `2ctx` slice
- but accepted-token quality is still slightly negative overall

### Layers `0` and `2`

Candidate:

- `/private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_layer02.gguf`

`ctx_08192`:

- accepted delta: `-0.4076%`
- committed delta: `-0.1079%`

`ctx_16384`:

- accepted delta: `+0.1601%`
- committed delta: `+0.0228%`

Two-context mean:

- accepted delta: `-0.1237%`
- committed delta: `-0.0426%`

Interpretation:

- adding layer `0` to the best single-layer candidate does not help on
  average
- the hotspot signal is therefore not simply additive across top-ranked layers

### Layer `0` only

Candidate:

- `/private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_layer0.gguf`

Two-context mean:

- accepted delta: `-0.2010%`
- committed delta: `-0.1446%`

Interpretation:

- worse than the full boosted donor
- clearly not the lead slice despite its large hotspot mass

### Layer `1` only

Candidate:

- `/private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_layer1.gguf`

Two-context mean:

- accepted delta: `-0.1173%`
- committed delta: `-0.0721%`

Interpretation:

- better than `layer0`
- still worse than `layer2`
- still below baseline

## Ranking of the selective splice variants

By two-context committed delta:

1. `layer2`: `+0.0036%`
2. `layer02`: `-0.0426%`
3. `layer1`: `-0.0721%`
4. full boosted donor: `-0.0995%`
5. `layer0`: `-0.1446%`

The important point is not the exact ordering of the negative variants. The
important result is:

- `layer2` is the only selective slice that meaningfully improved on the full
  donor
- none of the tested selective slices produced a clearly positive accepted
  gain across both contexts

## Read on the avenue

The current hotspot-ranked selective-layer branch is now tested enough to make
two claims.

### 1. The boosted donor signal is concentrated, not broad

The fact that `layer2` beats:

- the full donor
- `layer0`
- `layer1`
- and `layer02`

suggests that recoverable-gap weighting is not producing a uniformly useful
expert-wide shift. Most of the useful signal currently visible on the clean
`2ctx` root seems concentrated in a narrow routed-expert slice.

### 2. Layer selection alone is not enough to hit a strong win

Even the best slice:

- only reaches about flat committed quality
- and stays slightly negative on accepted tokens

So pure donor-splicing by hotspot-ranked routed-expert layer is not yet the
`+10%`-quality answer sought by `59`.

## Immediate implication for `59`

This does not falsify routed-expert work entirely, but it does narrow the next
useful variants:

1. if routed-expert work continues, it should move from whole-layer splicing to
   finer local selection inside the winning `layer2` slice
2. otherwise the next search budget should move to a more distinct model-side
   lever, such as:
   - dense legal `Q8_0` families
   - or another oracle-only route that can explain the residual gap more
     sharply than whole-layer routed-expert swaps
