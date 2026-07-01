# Layer 2 Tensor-Part Splice Results After the Selective-Layer Follow-Up

Date: 2026-07-01

## Purpose

Record the next refinement after `67` showed that:

- `layer2` was the best whole-layer selective splice
- but the whole layer still only reached slightly negative accepted quality on
  the clean `2ctx` root

This note asks the next narrower question:

- inside the winning `layer2` routed-expert slice, which tensor part is
  actually carrying the recoverable-gap signal?

## Tooling change used for this branch

`gguf-tools/mixed/splice_mixed_expert_layers_gguf.py` now supports routed
selection at tensor-part granularity via:

- `--q4-select`

Examples:

- `2:down`
- `2:gate,2:up`

This keeps the splice branch footprint-neutral while avoiding a premature move
to full block surgery.

## Candidates evaluated

Using the boosted donor:

- `/private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_local.gguf`

and the same clean two-context root:

- `/private/tmp/dspark_sweep2ctx`

the following candidates were built and measured:

1. `layer2down`
2. `layer2gateup`
3. `layer2gate`
4. `layer2up`

All preserved the baseline drafter footprint.

## Reference points

From `67`:

- full boosted donor 2ctx mean:
  - accepted `-0.1782%`
  - committed `-0.0995%`
- whole `layer2` 2ctx mean:
  - accepted `-0.0603%`
  - committed `+0.0036%`

## Results

### `layer2down`

Candidate:

- `/private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_layer2down.gguf`

`ctx_08192`:

- accepted delta: `-0.2357%`
- committed delta: `-0.1888%`

`ctx_16384`:

- accepted delta: `+0.1165%`
- committed delta: `+0.0455%`

Two-context mean:

- accepted delta: `-0.0596%`
- committed delta: `-0.0717%`

Read:

- `down` alone captures some context-dependent signal
- but it is not the source of the best committed result

### `layer2gateup`

Candidate:

- `/private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_layer2gateup.gguf`

`ctx_08192`:

- accepted delta: `+0.2652%`
- committed delta: `+0.2473%`

`ctx_16384`:

- accepted delta: `-0.0291%`
- committed delta: `+0.0046%`

Two-context mean:

- accepted delta: `+0.1180%`
- committed delta: `+0.1259%`

Read:

- this is the first selective splice branch that is clearly positive on both
  mean accepted and mean committed quality across the clean `2ctx` slice
- it materially beats the earlier whole-layer `layer2` result

### `layer2gate`

Candidate:

- `/private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_layer2gate.gguf`

`ctx_08192`:

- accepted delta: `-0.2455%`
- committed delta: `-0.1349%`

`ctx_16384`:

- accepted delta: `-0.1262%`
- committed delta: `-0.0683%`

Two-context mean:

- accepted delta: `-0.1859%`
- committed delta: `-0.1016%`

Read:

- `gate` alone is clearly harmful on both clean contexts
- so the positive `gateup` result is not coming from gate-only recovery

### `layer2up`

Candidate:

- `/private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_layer2up.gguf`

`ctx_08192`:

- accepted delta: `+0.3045%`
- committed delta: `+0.2473%`

`ctx_16384`:

- accepted delta: `-0.1941%`
- committed delta: `-0.0911%`

Two-context mean:

- accepted delta: `+0.0552%`
- committed delta: `+0.0781%`

Read:

- `up` alone is the cleanest single-tensor positive result found so far
- it beats whole `layer2`
- it stays below `layer2gateup` on the two-context mean
- so the current best tensor-part candidate remains the pair, not `up` alone

## Ranking inside layer 2

By two-context accepted delta:

1. `layer2gateup`: `+0.1180%`
2. `layer2up`: `+0.0552%`
3. `layer2down`: `-0.0596%`
4. whole `layer2`: `-0.0603%`
5. `layer2gate`: `-0.1859%`

By two-context committed delta:

1. `layer2gateup`: `+0.1259%`
2. `layer2up`: `+0.0781%`
3. whole `layer2`: `+0.0036%`
4. `layer2down`: `-0.0717%`
5. `layer2gate`: `-0.1016%`

## Main conclusion

The recoverable-gap signal inside the previously winning whole-layer `layer2`
splice is not distributed the way the whole-layer hotspot summary suggested.

What now appears true is:

- the useful signal is concentrated in the `up` family
- adding `gate` to `up` improves the current two-context mean despite `gate`
  being harmful on its own
- `down` is mixed and does not explain the best result

So the right next read is not:

- "whole layer 2 is mildly useful"

It is:

- "`layer2gateup` is the current best selective oracle candidate"
- and `layer2up` is the best single-tensor lead if one wants the narrowest
  current slice

## Immediate implication for `59`

This is the first clearly positive selective routed-expert result on the clean
`2ctx` root under the frozen footprint.

That does not yet prove a final winning route, but it changes the next useful
continuations:

1. if staying in the routed-expert lane, narrow further from:
   - whole layer search
   to:
   - `layer2gateup` local block search
   - or `layer2up` local block search
2. do not spend more on `layer2gate` alone
3. treat `down` as secondary unless a later block-level search shows a sharper
   local pocket than the current tensor-level evidence suggests
