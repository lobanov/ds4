# Recoverable-Step Expert-Block Cohort Probe on `layer2gateup`

Date: 2026-07-02

## Purpose

Record the first acceptance-local sparse block follow-up after `88` pushed the
branch back toward routed-local `Q4_K` search and after the earlier sparse
heuristics had already failed:

- `71`: raw donor-vs-base byte-delta top-blocks in `layer2up`
- `72`: top recoverable-usage expert slices in `layer2gateup`

The open question here was narrower:

- can the positive full `layer2gateup` splice be recovered more efficiently by
  copying only the most recoverably active **expert-column block cohorts**
  rather than full expert slices or byte-delta-ranked blocks?

## New tooling

Added:

- `issue468/summarize_recoverable_expert_blocks.py`

Updated:

- `gguf-tools/mixed/splice_mixed_expert_layers_gguf.py`

New capability:

- `--expert-block-select <tensor_name>:expert:block,expert:block,...`

Meaning:

- for a routed `Q4_K` tensor, copy all output rows for selected
  `(expert, column-block)` pairs from donor to baseline
- where one column block spans `256` input columns, matching one `Q4_K` block

So this is more local than:

- full `--expert-select`

and more mechanism-aware than:

- raw `--block-delta-top`

## Ranking signal used

The new block summarizer replays the numpy/oracle path on the recoverable-step
same-checkpoint ref-oracle dataset from `87`:

- sweep root:
  - `/private/tmp/dspark_sweep2ctx`
- oracle details:
  - `/private/tmp/dspark_sweep8/ref-oracle-fp8-2ctx.details.json`

For each recoverable step it:

1. reconstructs the selected experts at `mtp.2`
2. captures the FFN input state before routed experts
3. scores each `256`-column block by squared activation mass
4. weights by recoverable-step weight and gate probability
5. aggregates mass over `(expert, block_index)` cohorts

Top cohorts were strongly concentrated in:

- expert `183`
- then experts `131` and `218`

The first `16` top cohorts were all:

- expert `183`
- all `16` of its `256`-column blocks

So the first sparse control is effectively:

- one full expert, but only inside the `layer2gateup` donor pair

## Candidates

Donor:

- `/private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_layer2gateup.gguf`

### Candidate 1: `top16cohorts`

Candidate:

- `/private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_layer2gateup_top16cohorts.gguf`

Copied cohorts:

- expert `183`
- all `16` column blocks

applied to both:

- `mtp.2.ffn_gate_exps.weight`
- `mtp.2.ffn_up_exps.weight`

### Candidate 2: `top32cohorts`

Candidate:

- `/private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_layer2gateup_top32cohorts.gguf`

Copied cohorts:

- all `16` blocks of expert `183`
- plus the next strongest partial cohorts from experts `131` and `218`

again applied to both:

- `mtp.2.ffn_gate_exps.weight`
- `mtp.2.ffn_up_exps.weight`

## Measurement

Both candidates were measured on the clean `2ctx` root with the same harness
used in `72`:

- sweep root:
  - `/private/tmp/dspark_sweep2ctx`
- trials:
  - `256`
- steps:
  - `19`

## Results

### `top16cohorts`

`ctx_08192`:

- accepted delta: `+0.009821735500659123%`
- committed delta: `0.0%`

`ctx_16384`:

- accepted delta: `-0.13102979714646468%`
- committed delta: `-0.07740290488550405%`

Two-context mean:

- accepted delta: `-0.06060403082290278%`
- committed delta: `-0.03870145244275203%`

### `top32cohorts`

`ctx_08192`:

- accepted delta: `+0.19152384226293062%`
- committed delta: `+0.11240501775999778%`

`ctx_16384`:

- accepted delta: `-0.25235368339318054%`
- committed delta: `-0.13659336156263135%`

Two-context mean:

- accepted delta: `-0.03041492056512496%`
- committed delta: `-0.012094171901316786%`

## Comparison to earlier routed-local controls

Two-context mean accepted delta:

1. full `layer2gateup`:
   - `+0.118034562909336%`
2. `top32cohorts`:
   - `-0.03041492056512496%`
3. `top16cohorts`:
   - `-0.06060403082290278%`
4. `top8experts`:
   - `-0.0855347995275646%`

Two-context mean committed delta:

1. full `layer2gateup`:
   - `+0.12592207556202384%`
2. `top32cohorts`:
   - `-0.012094171901316786%`
3. `top16cohorts`:
   - `-0.03870145244275203%`
4. `top8experts`:
   - `-0.06075562093208475%`

So the new cohort ranking is better than the old top-expert slice heuristic,
but it still does not recover the positive full `layer2gateup` result.

## Read

This is a useful negative result with a narrower meaning than `72`.

What it shows:

- acceptance-local expert-column block ranking is more informative than
  raw byte-delta ranking
- and more informative than top-expert-only ranking

But it still does **not** recover the positive signal of the full
`layer2gateup` splice.

The pattern now looks like:

- the useful `layer2gateup` signal is not confined to a tiny sparse set of
  high-mass routed cohorts
- the effect likely depends on a broader interaction across more of the donor
  tensor pair

## Decision impact

This changes the routed-local read in two ways.

### 1. Very sparse local block selection is no longer a strong lead

Both sparse routed-local heuristics now failed:

- raw block delta in `71`
- recoverable-step expert-block cohorts here

So the branch should not keep spending immediate budget on ever-finer sparse
subset selection ahead of broader mechanism changes.

### 2. `layer2gateup` remains the positive reference point, not a decomposed recipe

The current best read is still:

- full tensor-part pair `layer2gateup` is mildly positive

while:

- smaller local routed subsets keep losing that gain

So future work in this lane should treat `layer2gateup` as evidence of
distributed sensitivity, not as something already reduced to a sparse patch.
