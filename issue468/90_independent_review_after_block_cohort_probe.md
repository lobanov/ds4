# Independent Review After Sparse `layer2gateup` Block-Cohort Probe

Date: 2026-07-02

## Purpose

Record the requested independent `gpt-5.5 xhigh` review after exhausting the
sparse routed-local block-cohort branch in `89`.

Reviewed inputs:

- `issue468/59_oracle_only_quantization_search_goal.md`
- `issue468/86_independent_review_after_nearby_source_splice.md`
- `issue468/88_candidate_complementarity_2ctx.md`
- `issue468/89_layer2gateup_block_cohort_probe.md`

## Review verdict

The review concluded that the sparse routed-local branch is exhausted as a lead.

What is now considered exhausted:

- raw byte-delta top-block selection
- top recoverable-usage expert slices
- recoverable-step expert-column block cohorts

Reason:

- all three sparse heuristics failed to recover the mild positive full
  `layer2gateup` splice
- `top16cohorts` and `top32cohorts` stayed negative on the clean `2ctx` mean

So the right interpretation is:

- stop prioritizing "find a tiny sparse routed patch"
- do not yet close the broader routed `Q4_K` lane

## Recommended next experiments

Priority order from the review:

### 1. Full-pair local `Q4_K` requantization for `layer2gateup`

Target tensors:

- `mtp.2.ffn_gate_exps.weight`
- `mtp.2.ffn_up_exps.weight`

Mechanism:

- keep the footprint and `Q4_K` route fixed
- optimize scale/min/code choices across the full pair against recoverable-step
  acceptance or ref-oracle margin

Why this is different from exhausted work:

- prior routed work copied donor/source subsets
- it did **not** acceptance-optimize the distributed encoding of the full pair

Decision signal:

- a stable lift above the small full-splice baseline would keep routed `Q4_K`
  alive as Lane A
- another flat / sign-flipping result would sharply lower its priority

### 2. Real composed artifact test: `layer2gateup × mainproj_q8imat`

Mechanism:

- evaluate a simultaneous candidate containing both:
  - the best current routed-local splice
  - the strongest complementary dense runtime-side control from `88`

Why this is different:

- `88` only measured an impossible per-step chooser
- it did not test whether the complementarity survives in one static artifact

Decision signal:

- if the composite beats both individual artifacts, the complementarity is
  actionable
- if not, the best-of ceiling in `88` is mostly cancellation, not composition

### 3. Controlled local `attn_kv` perturbation

Mechanism:

- test small signed or calibrated local perturbations around the baseline for
  `attn_kv`
- not another source-endpoint splice

Why this is different:

- the source splice only tested one endpoint
- the perturbation screen tests local derivative / sensitivity around the
  baseline

Decision signal:

- asymmetric movement on recoverable steps would keep nearby attention alive
- flat movement would close it more confidently

## Decision impact

This independent review updates the branch priority as follows:

1. sparse routed-local subset search is closed as a lead family
2. the next routed-local move, if any, should be broader than sparse patching
3. the best immediate non-routed discriminator is a real composite artifact
   test rather than another impossible chooser
