# Independent Review After Dense Legal `Q8_0` Lane

Date: 2026-07-01

## Purpose

Record the requested independent `gpt-5.5 xhigh` review after exhausting the
currently identified nearby dense legal `Q8_0` lane, including the coupled
closeout control from `78`.

## Inputs reviewed

The review was asked to inspect:

- `issue468/59_oracle_only_quantization_search_goal.md`
- `issue468/52_fp8_headroom_plan.md`
- `issue468/54_fp8_reference_harness_revival_status.md`
- `issue468/72_activity_ranked_expert_slice_probe.md`
- `issue468/73_independent_review_after_activity_ranked_expert_probe.md`
- `issue468/74_main_proj_q8_imatrix_probe.md`
- `issue468/75_layer2_gate_shexp_q8_imatrix_probe.md`
- `issue468/76_layer2_up_shexp_q8_imatrix_probe.md`
- `issue468/77_layer2_down_shexp_q8_imatrix_probe.md`
- `issue468/78_dense_combo_q8_imatrix_closeout.md`
- `issue468/build_dense_recoverable_imatrix.py`
- `gguf-tools/quants.c`

## Independent read

The review agreed that the nearby dense legal `Q8_0` lane is exhausted enough
to stop as the lead branch.

Reason:

- the intended tensors were actually changed by the dense-imatrix path
- but the observed results stayed non-positive across the current neighborhood:
  - `main_proj`
  - `mtp.2.ffn_gate_shexp.weight`
  - `mtp.2.ffn_up_shexp.weight`
  - `mtp.2.ffn_down_shexp.weight`
- and the coupled dense closeout from `78` also remained negative

So this does not rule out all future dense work, but it does rule out:

- nearby single-tensor dense-imatrix requantization
- plus the one immediate coupled dense closeout

as the best next use of search budget.

## Recommended next experiments

Priority order from the review:

1. revive the FP8 source ceiling as the lead branch
2. treat the dense combo closeout as the final dense-nearby control
3. if FP8 shows real headroom, move to a different mechanism such as router /
   selection sensitivity

### 1. FP8 ceiling revival

The review's strongest recommendation was:

- move next to the FP8 source ceiling branch

Suggested execution order:

1. recover one working source-side forward path
2. score `ctx_08192` and `ctx_16384`
3. if sane, expand to the 4-context root from `52`

Reason:

- after routed-local and dense-local search both flattened, the highest-value
  remaining discriminator is whether source-precision FP8 still shows material
  headroom above the current `Q4_K` family

### 2. Dense closeout interpretation

The review explicitly recommended exactly one coupled dense control.

That control has now been run in `78`, and it remained negative.

Decision impact:

- do not keep spending more immediate search budget on nearby dense `Q8_0`
  families ahead of the FP8 ceiling read

### 3. Different mechanism if FP8 is real

If FP8 still shows meaningful headroom, the review recommended shifting from
projection-local error correction to a different mechanism:

- router / selection sensitivity

Lead tensor suggested:

- `ffn_gate_inp`

This is aligned with Step 4 in `59`.

## Tooling and measurement cautions

The review preserved three relevant cautions:

1. dense-imatrix weighting may still be the wrong proxy for acceptance
2. the current `2ctx` recoverable sample is still small
3. the `Q8_0` weighted local scale search in `quants.c` is plausible, but not
   a direct acceptance-aware objective

These cautions do not overturn the branch decision, but they do limit the
strength of any claim stronger than:

- "the nearby dense `Q8_0` lane is not the best next lead"

## Decision impact

This independent review updates the branch priority as follows:

- stop treating nearby dense legal `Q8_0` as the lead branch
- move next priority to FP8 source ceiling revival
- keep any further dense work behind that ceiling read
- if FP8 shows real route-level headroom, consider router-sensitive oracle work
  such as `ffn_gate_inp`
