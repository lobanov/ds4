# Sparse `layer2up` Block-Delta Probe

Date: 2026-07-01

## Purpose

Record the first true local-block follow-up after `70`.

The branch question was:

- can the positive `layer2up` tensor-level signal be recovered by copying only a
  sparse subset of donor `Q4_K` blocks back into the baseline drafter?

## Why this heuristic was used

The existing recoverable-gap imatrix was enough to rank:

- routed-expert layers
- tensor parts

But it was **not** sufficient to give a meaningful non-uniform block ranking
inside `mtp.2.ffn_up_exps.weight`.

Direct inspection showed:

- the imatrix view for `layer2up` was effectively flat at the simple
  row/block-sum level

So the first local-block heuristic used instead was:

- rank `Q4_K` payload blocks inside `mtp.2.ffn_up_exps.weight` by raw donor vs
  base byte-`L1` delta
- copy only the top-N donor blocks into the baseline tensor

This is not the final intended search criterion. It is the cheapest honest
first local-block probe with the current evidence.

## Tooling change

`gguf-tools/mixed/splice_mixed_expert_layers_gguf.py` now supports sparse block
override mode:

- `--block-delta-top <tensor_name>:<count>`

Current scope:

- `Q4_K` tensors only
- donor blocks selected by highest byte-`L1` delta against the baseline tensor

This keeps the candidate footprint unchanged while allowing local block copies
instead of whole-tensor-part swaps.

## Candidate 1: top 128 donor-delta blocks in `layer2up`

Candidate:

- `/private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_layer2up_top128blocks.gguf`

Tensor patched:

- `mtp.2.ffn_up_exps.weight`

Donor blocks copied:

- top `128` `Q4_K` payload blocks by raw donor-vs-base byte delta

Results:

- `ctx_08192`:
  - accepted delta: `0.0%`
  - committed delta: `0.0%`
- `ctx_16384`:
  - accepted delta: `0.0%`
  - committed delta: `0.0%`

Read:

- this sparse patch is functionally indistinguishable from baseline on the clean
  `2ctx` root

## Candidate 2: top 2048 donor-delta blocks in `layer2up`

Candidate:

- `/private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_layer2up_top2048blocks.gguf`

Tensor patched:

- `mtp.2.ffn_up_exps.weight`

Donor blocks copied:

- top `2048` `Q4_K` payload blocks by the same byte-delta ranking

Results:

- `ctx_08192`:
  - accepted delta: `-0.0049%`
  - committed delta: `0.0%`
- `ctx_16384`:
  - accepted delta: `0.0%`
  - committed delta: `0.0%`

Two-context mean:

- accepted delta: `-0.0025%`
- committed delta: `0.0%`

Read:

- scaling the same sparse-byte-delta heuristic up by `16x` still leaves the
  candidate effectively at baseline

## Conclusion

This probe gives a useful negative result:

- raw donor-vs-base payload delta is **not** a good first-order proxy for the
  useful local routed-expert blocks inside `layer2up`

What this rules out is narrow but important:

- "take the most-changed `Q4_K` blocks first" is not enough

What it does **not** rule out:

- a better local-block search seeded by:
  - expert-routing activity
  - recoverable-step usage
  - or a tensor-aware block ranking richer than raw byte delta

## Immediate implication for `59`

The best current routed-expert lead remains:

- `layer2gateup` at tensor-part granularity

The new local-block tooling is still useful, but the next local search should
not use raw byte-delta ranking as its selection rule.
