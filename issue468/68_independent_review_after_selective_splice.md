# Independent Review After Selective Routed-Expert Splice Search

Date: 2026-07-01

## Purpose

Record the requested independent `gpt-5.5 xhigh` review after exhausting the
currently identified hotspot-ranked selective-splice avenue from `67`.

This note is intentionally short. Its job is to preserve the external review
read and the resulting branch recommendation.

## Inputs reviewed

The review was asked to inspect:

- `issue468/52_fp8_headroom_plan.md`
- `issue468/59_oracle_only_quantization_search_goal.md`
- `issue468/63_independent_review_after_oracle_pivot.md`
- `issue468/66_local_recoverable_gap_results_soft_vs_boosted.md`
- `issue468/67_selective_routed_expert_splice_results.md`
- `issue468/summarize_dspark_imatrix_hotspots.py`
- `gguf-tools/mixed/splice_mixed_expert_layers_gguf.py`

## Independent read

The review's strongest conclusion was:

- the old broad routed-`Q4_K` plus imatrix family looks weak
- but the evidence still does **not** show that model-side quantization
  headroom is absent

Two earlier points were highlighted as the backdrop:

- the q-dump envelope only reached about `+1.01%` mean accepted in `49`
- the same `Q4_K` checkpoint through the F32 oracle showed about `+8.27%`
  accepted headroom in `51`

So the review agreed that the current problem is:

- the existing search family is weak

not:

- that there is no remaining quality to recover

## Read on the just-finished avenue

The review agreed with the local measurement read from `67`:

- whole-layer hotspot selective splicing is exhausted

Reason:

- full boosted donor stayed negative
- `layer2` improved strongly relative to the donor but remained only
  near-flat overall
- `layer02` got worse
- `layer0` and `layer1` stayed negative

The review explicitly recommended **not** spending more runs on:

- `layer12`
- `layer01`

because the observed marginal effect of adding layers `0` or `1` is already
negative.

## Recommended next ideas

The review proposed three next branches, in this order:

1. move from whole-layer splicing to fine-grained routed-`Q4_K` local search
   inside `layer2`
2. measure the source/reference FP8 ceiling from `52`
3. start legal dense `Q8_0` requantization one family at a time

The most important nearby variant was:

- a `layer2` tensor-part or block-level search
- especially `down` or top-block local selection

That is the only routed-expert continuation the review considered justified
without first changing search family.

## Tooling cautions called out by the review

The review also flagged three measurement/tooling risks:

1. the hotspot summarizer currently ranks by raw summed mass and does not
   normalize by tensor size or `ncall`
2. the splicer checks tensor compatibility but does not reason about qtype
   policy beyond shape/type matching
3. the B2 measurement remains a `256`-trial estimate without reported
   confidence bounds

## Decision impact

This independent review reinforces the local branch read:

- do not continue whole-layer hotspot combinations
- if routed-expert search continues immediately, narrow it to a finer local
  `layer2` search
- otherwise shift the next search budget to the FP8 ceiling or dense legal
  `Q8_0` lane from `59`
