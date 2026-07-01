# Plain-Layout F32 Diagnostic Under Frozen Runtime

Date: 2026-07-01

## Why this is the next avenue

After `issue468/54_fp8_reference_harness_revival_status.md`, the official
source-FP8 harness path on the DGX is no longer the clean critical path.

The independent `gpt-5.5 xhigh` review required by
`issue468/53_frozen_runtime_quantization_goal.md` Step 6 reached three useful
conclusions:

1. there is no clean exact source-FP8 ceiling path available right now without
   leaning on the brittle tilelang/TVM stack
2. under the frozen-runtime constraint, many apparent quantization routes are
   not actually executable by the unchanged runtime
3. the highest-value next diagnostic is a runtime-compatible precision upgrade
   on the DSpark tensors that already support plain `F16/F32` layout

So the branch should now answer a narrower but very important question:

- if all currently runtime-legal plain-layout DSpark tensors are upgraded from
  `F16` to `F32`, does acceptance move meaningfully?

If the answer is no, then the remaining in-scope path shrinks sharply toward:

- `Q4_K` calibration only

and the evidence for that family is already weak.

## Constraint check

The new goal in `53` freezes:

- inference code
- base model weights
- runtime behavior

So this avenue must use only tensor-type choices the current runtime already
supports.

The independent review highlighted the main distinction correctly:

- converter support is not enough
- the unchanged runtime must also be able to execute the resulting tensor types

## Runtime-compatible tensor class identified

The DSpark runtime already accepts certain drafter tensors through the
plain-layout path:

- `mtp.{0,1,2}.hc_attn_fn.weight`
- `mtp.{0,1,2}.hc_ffn_fn.weight`
- `mtp.{0,1,2}.ffn_gate_inp.weight`
- `mtp.2.hc_head_fn.weight`

Relevant checks in `ds4.c`:

- `tensor_expect_plain_layout(...)`
- `tensor_expect_plain_layout(l->hc_attn_fn, ...)`
- `tensor_expect_plain_layout(l->hc_ffn_fn, ...)`
- `tensor_expect_plain_layout(l->ffn_gate_inp, ...)`
- `tensor_expect_plain_layout(w->hc_head_fn, ...)`

The same path explicitly allows:

- `F16`
- `F32`

That makes these tensors the cleanest ship-legal precision lever under the
current goal.

## Candidate definition

### Diagnostic candidate: all plain-layout F32

Keep everything else unchanged from the DSpark template, but upgrade:

- `mtp.0.hc_attn_fn.weight -> f32`
- `mtp.1.hc_attn_fn.weight -> f32`
- `mtp.2.hc_attn_fn.weight -> f32`
- `mtp.0.hc_ffn_fn.weight -> f32`
- `mtp.1.hc_ffn_fn.weight -> f32`
- `mtp.2.hc_ffn_fn.weight -> f32`
- `mtp.0.ffn_gate_inp.weight -> f32`
- `mtp.1.ffn_gate_inp.weight -> f32`
- `mtp.2.ffn_gate_inp.weight -> f32`
- `mtp.2.hc_head_fn.weight -> f32`

Everything else remains at template type.

## Footprint result from quantizer dry-run

Using `issue468/baseline/dspark_template.gguf` as the planning template on the
DGX:

Baseline dry-run:

- `tensor_bytes_unpadded: 11489934236`
- `approx_file_bytes:    11489939840`

All-plain-F32 dry-run:

- `tensor_bytes_unpadded: 11501075356`
- `approx_file_bytes:    11501080960`

Delta:

- `11,141,120` bytes
- about `+0.097%`

Interpretation:

- this candidate is comfortably inside the `+10%` footprint budget from `53`
- if it fails to move acceptance, that failure is **not** because the budget was
  too small for this class of upgrade

## Planned measurement

Use the existing DSpark acceptance path on the 4-context root:

- `ctx_08192`
- `ctx_16384`
- `ctx_24576`
- `ctx_32768`

Method:

1. build the diagnostic candidate GGUF on DGX
2. stage the existing 4-context sweep root onto DGX if needed
3. run the existing probe/B2 measurement path with the unchanged runtime
4. compare mean accepted delta vs the current live baseline

## Decision rule

### If the all-plain-F32 candidate moves acceptance materially

Then:

- there is still real frozen-runtime headroom in plain-layout precision
- follow-up work should do sensitivity mapping within that tensor class rather
  than returning immediately to imatrix-only work

### If the all-plain-F32 candidate is flat or negative

Then:

- the frozen-runtime plain-layout precision avenue is largely exhausted
- the remaining in-scope path is mostly `Q4_K` calibration / allocation
- and current evidence says that family is unlikely to deliver the full goal by
  itself

## Important interpretation guardrails

This candidate is a **diagnostic** first, not a presumed ship answer.

It does not prove:

- the exact source-FP8 ceiling
- the best possible selective-precision recipe
- that all useful frozen-runtime headroom is gone if it fails

What it does test cleanly is:

- whether the simplest runtime-compatible precision upgrade class has real
  acceptance leverage

That is the right next question after the DGX source-reference harness branch
failed on environment/runtime-stack compatibility rather than model semantics.
