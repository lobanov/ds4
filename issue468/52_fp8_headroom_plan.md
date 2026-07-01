# Plan: establish FP8 ceiling before new quantization routes

Date: 2026-07-01

## Why this should be next

`51_numpy_oracle_q4k_ceiling.md` established that the current Metal `Q4_K`
baseline is still materially below the old F32-oracle ceiling of the **same
Q4_K checkpoint**.

That means two questions are still conflated:

1. how much headroom is left inside the current `Q4_K` route
2. how much additional headroom exists above `Q4_K` if the quantization route
   changes

The right next measurement is therefore the **true source-side ceiling**:

- DSpark source precision path (`FP8` source weights / official reference path)
- scored on the same acceptance bundles

Only after that should more quantization routes be compared.

## Goal

Measure an acceptance ceiling that is strictly above the current `Q4_K`
checkpoint and strictly closer to the true source model:

- `Q4_K` Metal baseline
- `Q4_K` F32-oracle ceiling
- source-precision FP8 reference ceiling

Then use that gap to decide whether:

- more `Q4_K` work is justified
- a different quantization route is necessary

## Proposed sequence

### Step 1. Re-establish the reference harness as the source-ceiling path

Use the older reference-model path in:

- `issue468/ref/dspark_ref_harness.py`
- `issue468/ref/inference/model.py`

Target deliverable:

- a reproducible command that runs DSpark source-precision forward on one
  captured acceptance bundle and emits draft logits / tokens for all 5 draft
  positions

Success condition:

- the harness runs on current captures without depending on the old synthetic
  smoke path
- output is saved in a format that can be scored by the existing acceptance
  scripts

### Step 2. Adapt the scorer to consume reference-path logits

Reuse the current scoring logic rather than inventing a new metric.

Candidates:

- extend `issue468/baseline/dspark_capture/measure_b2_acceptance.py`
- or write a small adapter that emits the same kind of per-step base-logit cache
  the scorer already expects

Target deliverable:

- score source-FP8 outputs on the current sweep-root bundles
- report:
  - greedy accepted prefix
  - Monte Carlo B2 committed tokens / cycle
  - analytical `1 - TV` upper bound

Success condition:

- the scoring path for source FP8 is numerically aligned with the already-used
  acceptance methodology

### Step 3. Run the FP8 ceiling on the existing 4-context root first

Use:

- `/tmp/dspark_sweep8/ctx_08192`
- `/tmp/dspark_sweep8/ctx_16384`
- `/tmp/dspark_sweep8/ctx_24576`
- `/tmp/dspark_sweep8/ctx_32768`

Reason:

- these are the same contexts already used for the current-family falsification
  and the `Q4_K` F32-oracle ceiling
- they are enough to determine whether the gap above `Q4_K` is large or small

Target outputs:

- per-context FP8 committed score
- mean delta vs:
  - live Metal `Q4_K` baseline
  - F32-oracle `Q4_K` ceiling

Decision threshold:

- if FP8 is only marginally above F32-oracle `Q4_K`, then changing quantization
  route is probably low value
- if FP8 is materially above it, then `Q4_K` is leaving real acceptance on the
  table and route changes deserve priority

### Step 4. Only then compare alternative quantization routes

Once the FP8 ceiling is known, test new routes in descending cost order:

1. `Q6_K`
2. `Q8_0`
3. mixed-precision slices if one tensor class dominates

All should be evaluated on the same 4-context root first.

Key question for each route:

- how much of the gap
  `source FP8 ceiling - live Q4_K baseline`
  does this route recover?

That is the relevant ranking criterion, not generic tensor correlation.

### Step 5. Add tensor / layer sensitivity only if the route gap is real

If FP8 meaningfully beats F32-oracle `Q4_K`, then the next diagnostic is:

- which `mtp.*` tensors or layers cause most of the remaining loss?

That points to:

- full-route replacement
- or mixed-precision / selective-precision strategies

If FP8 does **not** beat F32-oracle `Q4_K` by much, this diagnostic can wait.

## Minimal decision tree

### Case A. FP8 only slightly beats F32-oracle Q4_K

Interpretation:

- most of the gap is not due to the quantization route itself
- the main remaining opportunity is inside the existing `Q4_K` route

Recommendation:

- investigate runtime-path / scoring-path loss first
- de-prioritize expensive new quantization routes

### Case B. FP8 materially beats F32-oracle Q4_K

Interpretation:

- `Q4_K` is the real limiter
- current imatrix tuning failure is not the main story

Recommendation:

- stop spending on more `Q4_K`-specific imatrix tuning
- prioritize `Q6_K` / `Q8_0` / mixed-precision route evaluation

## Concrete next execution order

1. revive the reference harness on one current sweep-root bundle
2. make it emit scoreable logits / tokens
3. score FP8 on `8192` first
4. if sane, expand to `16384/24576/32768`
5. write the FP8 ceiling note
6. only then choose whether to build `Q6_K`, `Q8_0`, or a mixed route

## Stop conditions

Stop and reassess if either happens:

- the reference FP8 harness cannot be made compatible with the current capture
  bundles without substantial new infrastructure
- or FP8 shows too little gain above F32-oracle `Q4_K` to justify route work

Those are the two main ways this branch can fail early without wasting another
large block of effort.
