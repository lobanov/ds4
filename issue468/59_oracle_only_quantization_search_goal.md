# DSpark Research Goal — Oracle-Only Quantization Search After the Frozen-Runtime Pivot

Date: 2026-07-01

## Purpose

Define the operative research plan after the runtime findings in `57` and the
oracle-only pivot in `58`.

The deployment constraints from `53` still matter:

- inference code frozen
- base model weights frozen
- drafter artifact footprint within `+10%` of the current footprint

But they no longer define the **immediate execution plan**.

The immediate task is now:

- use the oracle to rank model-side quantization headroom first
- preserve frozen-runtime compatibility as a prioritization rule, not as a
  requirement for every early experiment

This is an oracle-first search program, not a runtime-first candidate program.

## Why the plan changed

Three findings changed the right order of work.

### 1. The broad plain-layout precision branch is exhausted under the frozen runtime

`57` showed that the live CUDA DSpark probe now runs, and that:

- broad plain-layout `F32` promotion collapses quality completely
- several apparently legal plain tensors still execute through F16-only kernels
- the actually runtime-compatible plain subset is much smaller than validator
  acceptance suggested

So "search only what definitely runs today" is too narrow to be an efficient
research plan.

### 2. The best currently proven runtime-compatible plain-tensor follow-up was low signal

`57` also showed that:

- `mtp.2.hc_head_fn.weight=f32` preserves baseline quality
- but showed no visible gain on the first live context

So the remaining obviously runtime-compatible plain-tensor avenue is not a good
lead branch.

### 3. The source model already uses mixed precision by tensor class

`58` records that the source safetensors are already heterogeneous:

- some classes are `F32`
- some are `BF16`
- some are `F8`
- routed experts are integer-plus-scale packed

So the long-run search space should not be framed as "recover `Q4_K` only."

## Operative question

The current research question is:

- which quantization changes produce the largest acceptance gains on
  recoverable failure states in the oracle, while staying within the global
  drafter size budget?

The frozen-runtime constraint now affects **how we prioritize** those changes:

- runtime-plausible classes first
- broader oracle-only classes second

## What stays fixed

The following remain fixed throughout the research program:

- no target/base model edits
- no DSpark architecture edits
- no runtime-kernel changes for candidate evaluation
- no source-model finetuning
- global drafter footprint budget remains within `+10%`

The current artifact budget anchor from `53` remains:

- baseline drafter footprint about `10.71 GiB`
- max budget about `11.78 GiB`

## Execution lanes

Adopt the two-lane structure from `58`.

### Lane A. Priority lane

Classes that are deployment-plausible under the frozen runtime, or close enough
 that a strong oracle signal would immediately justify more work.

Lane A gets the first search budget.

### Lane B. Secondary lane

Classes that are useful as oracle headroom detectors but are not currently
proven on the frozen runtime path.

Lane B is still in scope. A positive result there means:

- real model-side headroom exists

It does **not** mean:

- the candidate is ready for the current runtime

## Primary execution order

### Step 1. Build a recoverable-gap dataset

This replaces the old generic imatrix collection logic as the main search
dataset.

Definition:

1. identify states where the live baseline loses accepted tokens
2. intersect them with states where the same-checkpoint oracle performs better
3. use those states as the scoring/calibration set for quantization search

This changes the optimization target from:

- generic activation preservation

to:

- acceptance improvement on recoverable rejection-boundary states

### Step 2. Search routed `Q4_K` expert re-quantization first

This is the first Lane A branch.

Why first:

- largest clear artifact lever already aligned with the frozen runtime
- no runtime-type ambiguity
- best immediate path from oracle signal to deployment plausibility

Target search family:

- within-`Q4_K` block encoding improvements
- activation-aware or acceptance-aware scale/min optimization
- expert-selection-weighted block error

### Step 3. Search legal dense `Q8_0` requantization next

After the first routed-expert pass, test legal dense quantized classes one
tensor family at a time in the oracle.

Examples from `58`:

- `main_proj`
- attention projections
- shared expert projections

Goal:

- determine whether acceptance-sensitive loss is concentrated in dense legal
  quantized tensors rather than only in routed `Q4_K`

### Step 4. Test calibrated F16 rounding one tensor class at a time

This remains valid as a targeted oracle branch.

Priority rule from `58`:

- type-generic plain paths belong in Lane A
- plain tensors that currently hit F16-only runtime kernels belong in Lane B

Lead tensor from `58`:

- `ffn_gate_inp`

Reason:

- router perturbations can change expert selection and therefore acceptance

### Step 5. Open the dynamic-quantization oracle branch only after static gains flatten

Dynamic quantization is not an immediate deployment route.

It is a headroom detector for the question:

- is the remaining loss fundamentally static-route-limited?

This branch belongs in Lane B unless a clearly runtime-plausible form emerges.

## Lane assignment for current known avenues

### Lane A

- recoverable-gap dataset construction
- routed `Q4_K` expert re-quantization
- legal dense `Q8_0` requantization
- calibrated F16 rounding on type-generic paths
- Markov-head precision sensitivity, if source-vs-BF16 gap is real

### Lane B

- broader plain-tensor precision experiments on `hc_attn_fn`, `hc_ffn_fn`, and
  `ffn_gate_inp`
- dynamic-quantization oracle experiments
- mixed-precision reallocations whose only current blocker is backend support

## Candidate ranking rule

Rank oracle candidates by:

1. mean acceptance improvement on the recoverable-gap scoring set
2. quality gain per added GiB
3. stability across contexts / prompts
4. deployment plausibility under the frozen runtime

Do not rank mainly by:

- raw tensor correlation
- generic reconstruction loss
- one-context wins without recoverable-gap lift

## Interpretation rule

Oracle results must now be read in two layers:

### Model-side signal

- does this tensor class contain real quality headroom?

### Deployment signal

- is that headroom likely reachable under the frozen runtime?

Lane A positive results count strongly on both dimensions.

Lane B positive results count strongly only on the first dimension.

## Stop conditions for a branch

Close a branch when one of these becomes true:

1. oracle gains flatten relative to the branch's search cost
2. added footprint per unit gain becomes clearly unattractive
3. the branch loses head-to-head against a simpler Lane A route
4. the branch turns out to depend on a source precision fact that is false

Examples:

- no real source-vs-BF16 gap for Markov head
- no measurable gain from `hc_head_fn`-only precision changes
- repeated `Q4_K` local search variants with no recoverable-gap lift

## What is no longer the lead strategy

Per `58`, the following should not drive the next cycle:

- broad plain-`F32` promotion
- `hc_head_fn`-only promotion as the primary avenue
- more local variants of the already-falsified generic imatrix family
- runtime FP8 or kernel work
- treating the whole program as permanently `Q4_K`-only

## Immediate next actions

The next concrete work should be:

1. turn the current baseline/oracle mismatch data into a recoverable-gap
   dataset definition
2. wire the first oracle scoring path around routed `Q4_K` expert block search
3. keep Lane B experiments as parallel headroom screens, not as the mainline

## Relationship to earlier notes

- `53` remains the historical deployment-goal statement
- `57` narrows what is truly runtime-compatible today
- `58` provides the oracle-only prioritization logic

This note is the operative synthesis of those three.
