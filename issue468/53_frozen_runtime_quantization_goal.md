# DSpark Research Goal — Frozen-Runtime Quantization Path to +10% Drafter Quality

Date: 2026-07-01

## Purpose

Define the next research goal after the `Q4_K + imatrix` collector family was
falsified as a path to the old `+5%` gate.

This goal targets a larger improvement:

- **`>= +10%` drafter-quality improvement**

while holding the following fixed:

- **inference code frozen**
- **base model weights frozen**
- **drafter artifact footprint within `+10%` of the current footprint**

In practical terms, this means the only allowed product lever is:

- a new **drafter quantization recipe**

not:

- runtime numerical fixes
- new kernels
- target-model re-export
- source-model finetuning

---

## Fixed constraints

### 1. What is frozen

The following must not change for this assignment:

- `ds4` inference code and kernel behavior
- target/base model GGUF weights
- DSpark architecture and tensor layout

Allowed changes:

- drafter GGUF tensor quantization choices
- drafter-specific calibration data
- drafter imatrix / selective-precision allocation

### 2. Footprint budget

Current drafter footprint, from the existing artifact history, is about:

- **`10.71 GiB`**

Maximum allowed footprint for this goal:

- **`11.78 GiB`** (`10.71 * 1.10`)

So the total extra budget is only about:

- **`1.07 GiB`**

This immediately rules out broad whole-model upgrades such as:

- full-route `Q6_K`
- full-route `Q8_0`

unless a measurement later proves the actual shipped footprint baseline is
different from `10.71 GiB`.

---

## Starting point from the later artifacts

### 1. The old `Q4_K + imatrix` family is exhausted

The strongest measured acceptance-bundle variants remained far below the old
assignment gate:

- best real 4-context result: **`+0.3620%`** mean accepted
- impossible q-dump oracle over the whole current family: **`+1.0051%`**

So further local tuning inside the same collector family is not a credible path
to `+10%`.

### 2. There is real headroom above the current live baseline

`51_numpy_oracle_q4k_ceiling.md` showed that the same current `Q4_K` drafter
weights, dequantized to F32 and run through the oracle path, score:

- about **`+8.27%`** mean MC-B2 accepted vs the live Metal baseline

So the live baseline is not a hard ceiling for the current weights.

### 3. Some of that headroom is runtime-bound, not quantization-route-bound

`52_metal_vs_oracle_fp8_bisection.md` then showed that:

- drafter KV FP8 quantization alone costs about **`+2.6%`** deterministic draft quality

That lever is explicitly **out of scope here**, because inference code is frozen.

### 4. Therefore the new research problem must be stated carefully

The next goal cannot assume that all headroom above the live baseline is
recoverable by a new quantization recipe alone.

The right question is:

- **can a new frozen-runtime drafter quantization recipe recover `>= +10%` drafter quality vs the current live baseline while staying within the footprint budget?**

And the right evaluation must separate:

- improvement vs the current live baseline
- improvement vs the current `Q4_K` F32-oracle ceiling

Without that separation, runtime loss and quantization-route loss stay
conflated.

---

## Goal definition

### Primary goal

Produce a new drafter GGUF, compatible with the unchanged runtime, that improves
mean drafter quality by:

- **`>= +10%` vs the current shipped drafter baseline**

while staying within:

- **`<= 11.78 GiB`**

### Primary metric

Use the existing acceptance methodology on the same context sweep family, with
the live runtime frozen:

- **mean MC-B2 accepted tokens / cycle**

Preferred sweep:

- `8k, 16k, 24k, 32k, 40k, 48k, 56k, 64k`

Minimum first-pass sweep:

- `8192, 16384, 24576, 32768`

### Secondary metrics

Track, but do not replace the primary metric with:

- mean committed tokens / cycle
- greedy accepted prefix
- deterministic probe average prefix on captured bundles
- footprint delta vs current drafter artifact

### Success condition

This goal succeeds only if all are true:

1. the runtime and base model remain unchanged
2. the new drafter artifact fits within `+10%` footprint
3. the artifact runs on the existing production path
4. measured mean MC-B2 accepted improves by `>= +10%` vs the current live baseline

---

## What counts as an in-scope quantization change

The new route must be expressible as a drafter-only artifact change compatible
with the current runtime.

In-scope candidate families:

1. **better `Q4_K` calibration**
   - new drafter-specific imatrix objective
   - new calibration corpus emphasizing acceptance-sensitive states
   - tensor-class or layer-aware weighting

2. **selective mixed precision within the existing runtime-supported types**
   - higher precision only on a small subset of `mtp.*` tensors or layers
   - leave the rest at the current baseline precision
   - stay inside the `~1.07 GiB` extra budget

3. **hybrid route allocation**
   - preserve the current default route for most tensors
   - spend the budget only where sensitivity mapping says acceptance is concentrated

Out of scope:

- full-model `Q6_K`
- full-model `Q8_0`
- any route needing new inference support
- code changes such as FP8-off KV, new drift patches, or new kernels

---

## Main hypothesis

The current family failed because it tried to improve all drafter expert weights
roughly uniformly, while the real acceptance loss is likely concentrated in a
small subset of tensors, layers, or expert classes.

Under a strict `+10%` footprint budget, the only plausible path is:

- **find the highest-sensitivity drafter slices**
- **spend the limited precision budget there**
- **leave the rest of the drafter near the current footprint**

This is a broader quantization-allocation hypothesis, not another same-family
collector-weight tweak.

---

## Execution plan

### Step 1. Establish the two ceilings that matter

Before ranking routes, keep these baselines side by side on the same bundles:

1. current live Metal baseline
2. current `Q4_K` F32-oracle ceiling
3. source-side FP8 ceiling, if the reference harness can be revived cheaply

Reason:

- the live-to-oracle gap contains frozen-runtime loss
- route work should be judged on how much it closes the live gap
- but also on how much of the quantization-route gap it closes

Decision use:

- if the source-FP8 ceiling is only slightly above the `Q4_K` F32-oracle
  ceiling, broad route changes are low value
- if it is materially above, selective precision work remains justified

### Step 2. Build tensor / layer sensitivity maps under the frozen runtime

This is the first load-bearing experiment.

Question:

- which `mtp.*` tensors or layers produce the largest acceptance loss when kept
  at the current route?

Target output:

- a ranked list of sensitive tensor classes / layers
- an estimated quality gain per unit of added footprint

This mapping is what decides whether the `+10%` goal is realistic under a
`+10%` footprint budget.

### Step 3. Define budget-feasible candidate recipes

Only evaluate candidates that fit the budget on paper before build time.

Candidate template examples:

1. baseline route everywhere + higher precision on one drafter layer
2. baseline route everywhere + higher precision on one tensor class across all 3 layers
3. baseline route everywhere + higher precision on the top-N most sensitive slices
4. improved `Q4_K` calibration only, if sensitivity does not justify mixed precision

Each recipe should record:

- predicted footprint
- changed tensors
- rationale from sensitivity data

### Step 4. Score candidates first on the 4-context root

Use the same first-pass root already used in the recent falsification notes:

- `8192`
- `16384`
- `24576`
- `32768`

Keep only candidates that show credible lift there.

Promotion rule:

- no candidate should be scaled to the full 8-context sweep unless it clears a
  meaningful first-pass threshold, for example `>= +5%` mean accepted on the
  4-context root

### Step 5. Run the full 8-context acceptance sweep on finalists

Final evaluation sweep:

- `8k, 16k, 24k, 32k, 40k, 48k, 56k, 64k`

Ship/no-ship decision is based on that final mean, not on a single-context win.

### Step 6. After each exhausted avenue, run an independent review pass

Each time one currently identified avenue is exhausted, require a fresh
independent review before declaring the branch closed.

Required review shape:

- use an independent **`gpt-5.5 xhigh` subagent** review pass
- review both the current code and the accumulated research artifacts
- ask specifically for:
  - flaws in the current interpretation
  - untested hypotheses that remain in scope
  - ways runtime loss and quantization loss may still be conflated
  - new budget-feasible quantization ideas

Minimum trigger points:

1. after the ceiling-measurement branch is judged complete
2. after tensor / layer sensitivity mapping stops producing actionable slices
3. after the current selective-precision candidate family is exhausted
4. before writing any final negative verdict

Purpose:

- prevent the loop from stopping at the first locally exhausted hypothesis set
- force a broader re-read of the code and evidence
- generate new ideas that are not just minor variants of the failed avenue

This review step is mandatory unless a new avenue is already in hand from the
current branch.

---

## Candidate ranking rule

Rank candidate quantization recipes by:

1. primary metric improvement vs live baseline
2. footprint efficiency: quality gain per added GiB
3. closeness to the `Q4_K` F32-oracle ceiling
4. stability across contexts

Do **not** rank primarily by:

- tensor correlation
- isolated greedy wins at one context
- generic calibration-loss proxies

The measured acceptance metric remains the decision metric.

---

## Stop conditions

Stop the branch and write a negative conclusion if any of these become true:

1. sensitivity mapping shows that the tensors needing higher precision do not
   fit inside the `+10%` footprint budget
2. budget-feasible mixed recipes fail to beat the best `Q4_K` baseline by a
   meaningful margin on the 4-context root
3. the source-FP8 ceiling turns out to be too close to the current `Q4_K`
   ceiling to support a `+10%` target
4. the current runtime supports no useful selective-precision route without
   code changes

Before treating any of those as terminal, run the independent `gpt-5.5 xhigh`
subagent review from Step 6 and record any newly generated in-scope avenues.

If any of those happen, the honest conclusion is:

- the `+10%` frozen-runtime / frozen-base / `+10%`-footprint goal is not
  reachable through quantization-only changes

---

## Recommendation

The next research cycle should no longer be framed as:

- "find a better `Q4_K` imatrix"

It should be framed as:

- "find the best budget-feasible drafter quantization allocation under a frozen runtime"

The practical default path is:

1. keep the live baseline, `Q4_K` oracle ceiling, and source-FP8 ceiling distinct
2. do tensor / layer sensitivity mapping first
3. evaluate only selective-precision recipes that fit the `+10%` footprint cap
4. treat uniform same-family imatrix retuning as low-priority unless sensitivity
   data says the loss is still broad and calibration-dominated

That is the narrowest goal statement that still has a defensible chance of
producing a real `>= +10%` drafter-quality gain under the new constraints.
