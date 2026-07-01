# Independent Review After Activity-Ranked Expert-Slice Probe

Date: 2026-07-01

## Purpose

Record the requested independent `gpt-5.5 xhigh` review after exhausting the
first recoverable-usage-ranked routed-local expert-slice avenue from `72`.

## Inputs reviewed

The review was asked to inspect:

- `issue468/52_fp8_headroom_plan.md`
- `issue468/54_fp8_reference_harness_revival_status.md`
- `issue468/59_oracle_only_quantization_search_goal.md`
- `issue468/67_selective_routed_expert_splice_results.md`
- `issue468/68_independent_review_after_selective_splice.md`
- `issue468/69_layer2_tensor_part_splice_results.md`
- `issue468/70_layer2_gateup_long_context_stability.md`
- `issue468/71_sparse_up_block_delta_probe.md`
- `issue468/72_activity_ranked_expert_slice_probe.md`
- `issue468/summarize_recoverable_expert_usage.py`
- `gguf-tools/mixed/splice_mixed_expert_layers_gguf.py`

## Independent read

The review agreed with the current branch evidence:

- routed-local activity-ranked expert slicing is exhausted enough to stop as
  the main lane

Reason:

- the best routed signal remains full `layer2gateup` at tensor-part granularity
- both finer local continuations tried so far are negative or flat:
  - sparse `layer2up` byte-delta blocks from `71`
  - top-8 recoverable-usage `layer2gateup` expert slices from `72`

The review's read was that this is enough to stop treating routed-local search
as the best next use of search budget.

## Recommended next experiments

Priority order from the review:

1. dense legal `Q8_0` family screen
2. FP8 ceiling revival as a parallel ceiling task, not a blocker
3. at most one narrow routed closeout control if desired

### 1. Dense legal `Q8_0` family screen

The review recommended this as the next primary lane.

Why:

- it is already in Lane A in `59`
- it is deployment-plausible under the frozen runtime
- it tests a genuinely different quantization bottleneck than routed `Q4_K`
  local tuning

Recommended initial family order:

1. `mtp.0.main_proj.weight`
2. `mtp.2.ffn_gate_shexp.weight`
3. `mtp.2.ffn_up_shexp.weight`
4. `mtp.2.ffn_down_shexp.weight`
5. `mtp.*.attn_output_a.weight`
6. `mtp.*.attn_output_b.weight`

The core point is:

- move to dense legal `Q8_0` tensors one family at a time rather than spending
  more immediately on routed-local variants

### 2. FP8 ceiling revival

The review still considers the FP8 ceiling important, but no longer worth
blocking the main search lane.

Read:

- the ceiling remains useful as a headroom detector
- but `54` shows the current official path is blocked by tilelang / TVM Python
  wrapper failures

Recommended posture:

- keep FP8 revival alive as a secondary branch
- do not pause dense legal `Q8_0` search waiting for it

### 3. Optional routed closeout control

The only additional routed-local control the review considered justified was:

- top-8 experts on `mtp.2.ffn_up_exps.weight` alone

Reason:

- `69` showed `gate` alone is harmful
- the top-8 probe in `72` preserved `gate+up` coupling by assumption
- a very narrow `up`-only control could resolve whether the negative result came
  from reintroducing harmful gate-local structure

This was explicitly framed as:

- one closeout control only
- not justification for an open-ended top-16 / top-32 / top-64 routed-local
  sweep

## Tooling and measurement cautions

The review flagged three cautions worth preserving:

1. `--expert-select` was new and needed an integrity check
2. the recoverable-usage ranking in `72` was built from a small number of
   recoverable rows
3. B2 deltas of about `0.1%` should guide branch ordering, not be overread as
   high-confidence final quality claims

The first caution has now been addressed locally:

- the generated expert-slice candidate was checked bytewise
- selected experts matched donor payload
- non-selected experts matched baseline payload

So the main decision impact is unchanged.

## Decision impact

This independent review updates the branch priority as follows:

- stop treating routed-local expert slicing as the lead lane
- move the next main search budget to dense legal `Q8_0` family search
- keep FP8 ceiling revival alive, but as a secondary ceiling branch
- only spend one more routed-local run if an `up`-only top-8 control is needed
  for closure
