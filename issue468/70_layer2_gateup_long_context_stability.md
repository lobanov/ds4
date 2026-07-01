# Layer 2 Gate+Up Long-Context Stability Check

Date: 2026-07-01

## Purpose

Record the next verification step after `69` identified:

- `layer2gateup` as the best current selective routed-expert candidate on the
  clean `2ctx` root
- `layer2up` as the best single-tensor slice
- `layer2updown` as weaker than both, which implies `gate` contributes a
  positive interaction with `up` even though `gate` is harmful on its own

The open question after `69` was not "which tensor part is best" anymore.
It was:

- does the current best candidate remain stable beyond the first two contexts?

## Additional tensor-part interaction result

Before the long-context checks, one extra pair was measured:

- `layer2updown`

Candidate:

- `/private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_layer2updown.gguf`

Two-context results:

- `ctx_08192`:
  - accepted delta: `+0.0491%`
  - committed delta: `+0.1304%`
- `ctx_16384`:
  - accepted delta: `-0.0291%`
  - committed delta: `-0.0455%`

Two-context mean:

- accepted delta: `+0.0100%`
- committed delta: `+0.0424%`

Interpretation:

- removing `gate` from the best pair made the result worse
- so the best current read from `69` holds:
  - `gate` is harmful alone
  - but beneficial in interaction with `up`

## Long-context check setup

Two isolated one-context roots were built from the existing `4ctx` sweep:

- `/private/tmp/dspark_sweep24576`
- `/private/tmp/dspark_sweep32768`

Each re-used:

- the existing prompt bundle
- the existing oracle details file
- the same measurement path already used for the earlier `2ctx` checks

Candidate:

- `/private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_layer2gateup.gguf`

## `ctx_24576`

Baseline completed cleanly:

- accepted: `4.2375`
- committed: `4.6127`

Candidate also completed cleanly:

- accepted: `4.2407`
- committed: `4.6199`

Delta:

- accepted: `+0.0776%`
- committed: `+0.1560%`

Important branch result:

- unlike the earlier soft full-donor branch from `66`, this candidate did **not**
  fail on the candidate side at `24576`

## `ctx_32768`

Baseline completed cleanly:

- accepted: `4.2693`
- committed: `4.5438`

Candidate also completed cleanly:

- accepted: `4.2658`
- committed: `4.5417`

Delta:

- accepted: `-0.0819%`
- committed: `-0.0452%`

Interpretation:

- this longer context is slightly negative
- but it is near-flat, and still far from the earlier candidate-side runtime
  failure mode

## Updated `layer2gateup` context table

All four measured contexts now read:

1. `ctx_08192`
   - accepted delta: `+0.2652%`
   - committed delta: `+0.2473%`
2. `ctx_16384`
   - accepted delta: `-0.0291%`
   - committed delta: `+0.0046%`
3. `ctx_24576`
   - accepted delta: `+0.0776%`
   - committed delta: `+0.1560%`
4. `ctx_32768`
   - accepted delta: `-0.0819%`
   - committed delta: `-0.0452%`

Four-context mean:

- accepted delta: `+0.0580%`
- committed delta: `+0.0906%`

## Conclusion

This changes the branch read in two concrete ways.

### 1. `layer2gateup` is no longer just a short-context win

It remains positive on the `4ctx` mean:

- accepted `+0.0580%`
- committed `+0.0906%`

The margin is modest, but it is now supported by:

- two short/medium contexts
- one clearly positive long context
- one near-flat long context

### 2. The earlier long-context instability is not intrinsic to this lane

The candidate-side `24576` failure from the older soft full-donor branch in
`66` does **not** generalize to the current best selective candidate.

So the current lead branch is now:

- quality-positive on the measured `4ctx` mean
- footprint-neutral
- frozen-runtime-compatible
- and no longer blocked by the earlier candidate-side long-context failure

## Immediate implication for `59`

The strongest current routed-expert candidate is now:

- `layer2gateup`

and it has crossed the minimum threshold for a more expensive next step.

The next useful routed-expert move should therefore be one of:

1. local block search inside `layer2gateup`
2. local block search inside `layer2up` as the narrowest single-tensor lead

But the branch no longer needs more whole-tensor-part combinations before that.
