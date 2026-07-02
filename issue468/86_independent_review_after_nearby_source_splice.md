# Independent Review After Nearby Source-Splice Screen

Date: 2026-07-02

## Purpose

Record the requested independent `gpt-5.5 xhigh` review after exhausting the
nearby direct source-splice family closed out in `85`.

Reviewed inputs:

- `issue468/59_oracle_only_quantization_search_goal.md`
- `issue468/79_independent_review_after_dense_q8_lane.md`
- `issue468/83_fp8_source_ceiling_4ctx.md`
- `issue468/84_ffn_gate_inp_source_splice_probe.md`
- `issue468/85_nearby_source_splice_screen_closeout.md`

## Review verdict

The review said the new negative source-splice results should change branch
priority, but only narrowly.

What the review considers exhausted as a lead branch:

- nearby direct source-copy splicing of
  - `ffn_gate_inp`
  - `attn_kv`
  - `attn_q_a`

What the review does **not** consider revived by those negatives:

- nearby dense legal `Q8_0` as a new lead
- broad FP8/source-copy work as a new lead

Reason:

- `79` already demoted nearby dense legal `Q8_0`
- `83` already showed the broad FP8/source ceiling is near parity, not a
  strong positive discriminator

So the right interpretation is:

- stop pushing more source-direction copies
- move next to acceptance-local mechanisms

## Recommended next experiments

Priority order from the review:

### 1. Acceptance-aware routed `Q4_K` local search in `mtp.2`

Lead tensors:

- `layer2gateup`
- `layer2up`

Mechanism:

- keep qtype and footprint fixed
- optimize local block encodings or scale/min choices against
  recoverable-gap acceptance

Why this is different from exhausted work:

- prior routed negatives were donor/source block copies
- or byte-delta / activity-ranked slice transfers
- not acceptance-objective local requantization

Decision signal:

- a stable 4-context lift would keep the routed-static Lane A branch alive
- another flat / negative result would come close to closing it

### 2. Per-state oracle complementarity screen over existing artifacts

Candidate families to compare:

- baseline
- `layer2gateup`
- `layer2up`
- dense closeout candidates
- FP8/source
- nearby source splices

Mechanism:

- compare candidate winners at recoverable-step granularity
- compute a "best existing candidate per state" ceiling

Why this is different:

- earlier notes mostly compare context means
- this tests whether current branch losses are cancellation across states

Decision signal:

- if best-of is materially positive, the branch still has state-conditional
  headroom
- if best-of is also flat, several nearby families can be retired more
  confidently

### 3. Controlled local perturbation for nearby attention tensors

Priority order:

- `attn_kv`
- then `attn_q_a` only if needed

Mechanism:

- test small local perturbation or calibrated-rounding directions around the
  baseline
- do **not** just copy the source tensor

Why this is different:

- the source-splice screen only tested one externally defined direction
- it did not test whether a beneficial local quantization direction exists

Decision signal:

- asymmetric `+/-` movement on recoverable states would show local sensitivity
- no movement would close nearby attention as a mechanism

## Decision impact

This independent review updates the branch priority as follows:

- nearby direct source-copy splicing is closed as a lead family
- acceptance-local routed `Q4_K` search moves back ahead of more source-copy
  variants
- a per-state complementarity screen is now a high-value discriminator
- nearby attention work, if continued at all, should switch from source-copy to
  local perturbation logic
