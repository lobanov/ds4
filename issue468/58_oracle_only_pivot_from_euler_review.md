# Oracle-Only Pivot After Euler Review

Date: 2026-07-01

## Purpose

Record what remains useful from the independent `gpt-5.5 xhigh` review once we
explicitly pivot away from runtime work and toward oracle-only DSpark quality
search.

This note follows:

- `issue468/53_frozen_runtime_quantization_goal.md`
- `issue468/57_cuda_probe_unblock_and_plain_layout_compatibility.md`

## Source precision map

The original DSpark safetensors are already mixed-precision, not uniformly
BF16 or FP8.

Representative source tensor classes observed from the `mtp.*` safetensor
shards:

- `F32`
  - `hc_attn_fn`, `hc_attn_base`, `hc_attn_scale`
  - `hc_ffn_fn`, `hc_ffn_base`, `hc_ffn_scale`
  - `hc_head_fn`, `hc_head_base`, `hc_head_scale`
  - `attn.attn_sink`
  - `ffn.gate.bias`

- `BF16`
  - `attn_norm.weight`, `ffn_norm.weight`, `main_norm.weight`, `norm.weight`
  - `attn.q_norm.weight`, `attn.kv_norm.weight`
  - `ffn.gate.weight`
  - `markov_head.markov_w1.weight`, `markov_head.markov_w2.weight`
  - `confidence_head.proj.weight`

- `F8_E4M3`
  - `attn.wq_a.weight`, `attn.wq_b.weight`, `attn.wkv.weight`
  - `attn.wo_a.weight`, `attn.wo_b.weight`
  - `ffn.shared_experts.w1/w2/w3.weight`
  - `main_proj.weight`

- `I8` with `F8_E8M0` scales
  - `ffn.experts.*.w1.weight`, `w2.weight`, `w3.weight`
  - with companion scale tensors such as `ffn.experts.*.w1.scale`

This matters for the oracle-only pivot because it confirms the long-run search
space should not be framed as "recover `Q4_K` only." The source model itself
already mixes precision aggressively by tensor class.

## What Euler contributed that still matters

Euler's most useful conclusion is structural:

- the main limiter is no longer raw footprint budget
- it is the mismatch between validator acceptance and true runtime type support

That conclusion is already confirmed by the live CUDA probe result:

- broad plain-layout F32 looked legal to the loader
- but collapsed at runtime because several DSpark paths still call F16-only
  kernels

For the oracle-only pivot, the value of the review is not runtime advice.
It is the ranking of **artifact-side research avenues** that can still show
real model-side headroom before we choose whether runtime validation is worth it.

## Useful avenues for oracle-only work

## Prioritization rule

For the oracle-only pivot, candidate tensor classes should be split into two
lanes.

### Lane A. Priority lane

Tensor classes that are plausible under the frozen runtime and therefore have a
shorter path from oracle signal to eventual deployment.

These should get the first search budget.

### Lane B. Secondary lane

Tensor classes that are not currently proven on the frozen runtime path, but
still look plausibly supportable on Metal later.

These should not be excluded from oracle experiments outright. They should just
be treated as:

- lower-priority for immediate search
- higher-risk when interpreting positive oracle results

This keeps the oracle useful as a model-side ranking tool without confusing:

- "shows real quality headroom in the model"

with:

- "can ship immediately under the current frozen runtime"

Priority is about execution order, not about permanently excluding tensor
classes from the oracle search.

### 1. Recoverable-gap-driven quantization objective

This is the strongest avenue from the review.

Idea:

- identify states where the current live baseline loses
- but the same-checkpoint F32 oracle wins

Then optimize quantization specifically against those states.

Why it remains useful in oracle-only mode:

- it does not require stable CUDA/Metal measurement to generate candidate GGUFs
- it uses the oracle exactly where the oracle is strongest:
  - ranking model-side error on rejection-boundary states

Why it is different from the old imatrix family:

- the old family optimized generic activation preservation
- this one would optimize **recoverable acceptance failures**

Meaning:

- not "which activations are important in general?"
- but "which errors are actually correlated with lost accepted tokens?"

This branch belongs in **Lane A**.

### 2. Routed `Q4_K` expert re-quantization

This is the best **starting** fully legal tensor family for the pivot.

Reason:

- routed experts are already constrained to `Q4_K` under the frozen runtime
- but the exact quantization of those `Q4_K` blocks is still an artifact lever

Oracle-only interpretation:

- start by searching for a better **within-`Q4_K` block encoding**
- do not mistake that starting point for a permanent `Q4_K`-only restriction on
  the oracle program

Candidate objective classes:

- weighted least squares over captured expert activations
- local block search around current scales/mins/nibbles
- expert-selection-weighted error on states near acceptance boundaries

Why this is promising:

- it stays inside the runtime-supported route
- it targets the dominant routed part of the drafter

This branch belongs in **Lane A** and should get the first search budget, but it
is not the only acceptable class once returns begin to flatten.

### 3. Activation-aware requantization of legal `Q8_0` tensors

Euler was right that routed experts are not the only remaining artifact lever.

Still-legal quantized dense tensors include:

- `main_proj`
- attention projections
- shared expert projections

Oracle-only interpretation:

- rebuild these `Q8_0` tensors with activation-aware calibration
- test one tensor class at a time in the oracle

Why useful:

- these tensors already run legally
- improvements should be visible in numpy without any runtime stabilization work

This branch belongs in **Lane A**.

### 3b. Dynamic quantization as the post-`Q4_K` fallback

If static `Q4_K` improvements start showing diminishing returns, the next
oracle-only branch should be **dynamic quantization** rather than forcing the
entire search to remain inside one static route family.

Meaning here:

- allow per-state, per-token, per-expert, or per-tensor-class quantization
  choices inside the oracle
- use this as a headroom detector, not as an immediate runtime commitment

Why useful:

- it answers whether the remaining loss looks fundamentally tied to static
  one-size-fits-all quantization
- it helps distinguish:
  - "better static calibration is enough"
  - from
  - "the remaining headroom likely needs state-conditional quantization"

Interpretation:

- a positive dynamic-quantization result does **not** mean it is deployable
  today
- it does mean the model has quality headroom that static `Q4_K` search may no
  longer be able to recover efficiently

This branch belongs in **Lane B** unless a clearly runtime-plausible dynamic
form later emerges for Metal.

## Global search-space rule

The eventual oracle search space should be allowed to cover **different
quantizations of all tensor classes**, provided the resulting drafter artifact
stays inside the size budget.

That means:

- do not artificially stop at routed `Q4_K`
- do not artificially stop at today's runtime-supported subset
- do not restrict the oracle program to only the currently easiest-to-ship
  tensor families

Instead:

- search the most deployment-plausible classes first
- then widen to the rest of the tensor classes as long as:
  - the candidate remains within the artifact size budget
  - the oracle result is interpreted correctly as model-side headroom rather
    than immediate runtime readiness

So the search policy is:

1. prioritize frozen-runtime-plausible classes first
2. widen to broader tensor classes once local returns flatten
3. keep the budget constraint global across all such experiments

### 4. Calibrated F16 rounding for plain tensors

This remains useful, but only in the oracle as a model-side ranking tool.

Key correction after `57`:

- broad `F32` promotion is not a frozen-runtime candidate for several plain
  tensors
- but **better F16 rounding** still is

Priority tensor:

- `ffn_gate_inp`

Reason:

- small router perturbations can change selected experts
- that can have outsized acceptance effects

Oracle-only use:

- generate alternative F16 encodings
- rank them by oracle quality first
- postpone runtime compatibility questions until a strong oracle signal exists

Priority split inside this branch:

- plain tensors already known to execute through type-generic paths stay in
  **Lane A**
- plain tensors that currently hit F16-only kernels but still look like obvious
  future-Metal candidates belong in **Lane B**

### 5. Markov-head precision sensitivity

This avenue is conditional.

If the source Markov head is effectively higher precision than the shipped BF16
representation, then:

- row-targeted BF16 rounding for `markov_w1` / `markov_w2`
- especially on high-frequency previous-token rows

could matter.

If source is already BF16-equivalent, this avenue is a no-op and should be
discarded quickly.

This branch belongs in **Lane A** if a real source-vs-BF16 gap exists.

## Secondary oracle lane: plausible future-Metal classes

The current runtime findings should narrow immediate priorities, but they should
not be over-read as a universal statement about what the Metal runtime could
support later.

So the following remain fair oracle-only research targets in **Lane B**:

- broader plain-tensor precision experiments on `hc_attn_fn`, `hc_ffn_fn`, and
  `ffn_gate_inp`
- mixed precision on plain DSpark tensors that are structurally simple and
  likely to be supportable once Metal paths are made type-generic
- other tensor-class reallocations whose only current blocker is backend support,
  not obvious model invalidity

Interpretation rule for Lane B:

- a positive oracle result means "real model-side headroom exists here"
- not "this is ready under the current frozen runtime"

That still makes Lane B valuable, because it helps decide whether later Metal
support work would have meaningful upside.

## What is *not* useful for the oracle-only pivot

Euler's review also helps by excluding work:

- no more broad plain-F32 promotion
- no more `hc_head_fn`-only work as a lead branch
- no assumption that the oracle program must remain forever `Q4_K`-only once
  static `Q4_K` gains flatten
- no runtime FP8 or kernel work
- no more local variants of the already-falsified generic imatrix family

## Recommended oracle-only execution order

### First branch

Build a recoverable-gap dataset:

1. collect states where live baseline loses accepted tokens
2. intersect with states where the same-checkpoint oracle is better
3. use those states as the scoring/calibration set for quantization search

This becomes the replacement for the old generic imatrix dataset.

### Second branch

Apply that dataset to routed `Q4_K` expert block optimization first.

Reason:

- largest legal artifact lever
- no runtime-type ambiguity
- best alignment with the frozen-runtime constraint

### Third branch

Only after that, test dense `Q8_0` requantization and calibrated F16 rounding
one tensor class at a time.

### Fourth branch

If the best static `Q4_K` candidates stop yielding meaningful gains, open the
dynamic-quantization branch in the oracle.

Purpose:

- determine whether the remaining recoverable gap is static-route-limited
- avoid over-investing in ever-finer static `Q4_K` tuning once signal weakens

Lane B experiments can run in parallel as lower-priority oracle screens, but
they should not displace the first-pass Lane A search.

## Practical implication

The oracle-only pivot is not merely "use numpy instead of CUDA."

It changes the research question from:

- "which tensors preserve activations best?"

to:

- "which legal artifact changes most improve acceptance on recoverable
  rejection-boundary states?"

That is the useful core of Euler's review for the current pivot.
