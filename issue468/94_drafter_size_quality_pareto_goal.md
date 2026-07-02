# DSpark Drafter Size/Quality Pareto Frontier Goal

Date: 2026-07-02

## Purpose

Define the next research goal after the direct raw-HF ceiling measurement in
`93`.

The new goal is to map the practical Pareto frontier between:

- drafter artifact footprint
- drafter prediction quality

under the same fixed constraints used throughout the current branch:

- inference code frozen
- base model weights frozen
- no drafter finetuning

This replaces the earlier single-point framing of:

- "find one recipe that clearly beats the current baseline"

with the more realistic question:

- "for each achievable drafter size, what is the best acceptance we can retain?"

## Why the goal changes

Three findings justify the pivot.

### 1. The current `Q4_K` baseline is already close to the direct DSpark tensor path

`93` measured the original DSpark HF MTP tensors directly through the oracle and
found that they do **not** provide a broad positive ceiling above the shipped
`dspark.gguf` baseline.

So the branch should stop assuming that:

- there is a large hidden quality reserve above the current drafter

and instead assume:

- the current drafter is already near a practical quality knee

### 2. Material size reduction must come from routed experts

The drafter remains dominated by routed expert payload:

- about `10.125 GiB` of the `10.70 GiB` baseline artifact is `Q4_K` experts

So any serious size reduction has to change routed-expert precision or routed
expert allocation, not only small dense tensors.

### 3. A single fixed target hides the useful engineering tradeoff

The right product question is no longer only:

- can one candidate beat baseline quality?

It is:

- what quality is available at `Q4_K` size?
- what quality is available at mixed `Q2/Q4` sizes?
- where is the best size/quality knee?

That is a frontier question, not a single-threshold question.

## Goal definition

Build and measure a compact set of drafter quantization recipes that span a
range of artifact sizes, then identify the non-dominated frontier in:

- total drafter size
- mean accepted tokens

Primary output:

- a measured Pareto frontier for the DSpark drafter under frozen runtime

## Fixed constraints

The following remain fixed:

- `ds4` runtime and kernels
- target/base model artifact
- DSpark architecture and tensor layout
- decode protocol and B2-style quality measurement

Allowed levers:

- routed-expert quantization type or allocation
- dense-tensor quantization choices already compatible with the runtime
- imatrix / calibration strategy
- selective mixed recipes across layers or tensor families

## Primary metrics

Every candidate should be scored with:

- total drafter bytes
- mean `average_accepted`
- mean `average_committed`

on the current 4-context sweep:

- `8192`
- `16384`
- `24576`
- `32768`

The primary frontier axis is:

- size vs mean `average_accepted`

Committed tokens remain important as a secondary throughput-facing read, but
accepted tokens are the cleaner predictor-quality metric.

## Reference points

Current anchor points:

1. baseline `dspark.gguf`
2. raw-HF DSpark oracle reference from `93`
3. current best same-size quantized artifact from `92`

These establish:

- the top-right practical baseline
- the direct source-tensor reference
- the best currently known same-footprint local improvement

## Search space

The main search axis should be routed experts, because they dominate size.

### Tier 1. Coarse routed recipes

Measure:

1. all routed experts at current `Q4_K` baseline
2. all routed experts at `Q2` family baseline
3. mixed `Q2/Q4` by drafter layer

Initial layer-mix grid:

1. `mtp.2 = Q4`, `mtp.0/1 = Q2`
2. `mtp.0 = Q4`, `mtp.1/2 = Q2`
3. `mtp.0/2 = Q4`, `mtp.1 = Q2`

Purpose:

- locate the first coarse knees in size vs quality

### Tier 2. Structured routed mixes

If the coarse layer mixes show usable tradeoffs, refine with:

- `gate/up` higher precision than `down`
- selective `Q4` only on the most sensitive routed layers
- selective `Q4` only on the most sensitive routed tensor parts

Purpose:

- see whether the frontier can be bent upward by spending limited `Q4` budget
  only where it matters most

### Tier 3. Small dense add-backs

On promising reduced-size routed recipes, test cheap dense controls such as:

- `main_proj_q8imat`

Purpose:

- check whether a small dense cost can recover disproportionate quality on top
  of a reduced expert recipe

## What success looks like

This goal does **not** require one recipe to dominate the baseline on both size
and quality.

It succeeds if it produces a measured frontier that clearly identifies:

1. the quality loss for major size reductions
2. whether mixed `Q2/Q4` recipes dominate pure `Q2`
3. whether there is a useful knee materially below the current `Q4_K` size
4. which recipe should be preferred at each practical memory budget

## Expected outcomes

Most likely:

- baseline `Q4_K` remains near the high-quality edge
- full routed `Q2` is materially smaller but clearly worse
- the interesting frontier points are mixed `Q2/Q4` recipes, especially ones
  that preserve more precision in `mtp.2`

Less likely but important to test:

- a mixed recipe that is materially smaller than baseline while staying very
  close in acceptance

## Immediate execution plan

1. define a small first-pass candidate set spanning current size down to full
   routed `Q2`
2. build each candidate GGUF
3. record exact artifact bytes
4. score each candidate on the 4-context acceptance sweep
5. tabulate the non-dominated frontier
6. only then decide whether a finer local search is justified

## Decision rule after first pass

After the first coarse frontier is measured:

- if every materially smaller point is sharply worse, stop spending on
  compression and treat current `Q4_K` as the practical knee
- if one or more mixed points are only slightly worse for a large size win,
  refine around those points
- if a mixed point is nearly flat in quality, prioritize it as the new main
  deployment candidate

## Final read

This goal matches the actual state of the branch better than the old
"single better recipe" framing.

The current evidence suggests that:

- upward quality headroom is small
- downward size headroom is real
- the remaining useful question is how much of the current quality can be
  preserved as the routed-expert budget is reduced

That is exactly a Pareto-frontier problem.
