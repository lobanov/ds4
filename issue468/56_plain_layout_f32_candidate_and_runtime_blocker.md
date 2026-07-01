# Plain-Layout F32 Candidate Built; Runtime Measurement Blocked on DGX

Date: 2026-07-01

## Purpose

Record the first concrete result from the post-review frozen-runtime precision
branch:

- the all-plain-layout-F32 diagnostic candidate is now a real GGUF artifact on
  the DGX
- but end-to-end acceptance measurement on DGX is currently blocked by the lack
  of a usable DSpark-capable runtime binary there

This note follows:

- `issue468/55_plain_layout_f32_diagnostic_plan.md`

## Candidate branch selected

The independent `gpt-5.5 xhigh` review recommended the simplest runtime-legal
diagnostic first:

- upgrade all DSpark plain-layout `F16` tensors that already support `F16/F32`
  in the unchanged runtime

Tensor set:

- `mtp.0.hc_attn_fn.weight`
- `mtp.1.hc_attn_fn.weight`
- `mtp.2.hc_attn_fn.weight`
- `mtp.0.hc_ffn_fn.weight`
- `mtp.1.hc_ffn_fn.weight`
- `mtp.2.hc_ffn_fn.weight`
- `mtp.0.ffn_gate_inp.weight`
- `mtp.1.ffn_gate_inp.weight`
- `mtp.2.ffn_gate_inp.weight`
- `mtp.2.hc_head_fn.weight`

All overridden to:

- `f32`

Everything else stays at template type.

## Dry-run result

Using:

- template: `issue468/baseline/dspark_template.gguf`
- HF source: `~/ds4/hf-dspark`

Baseline plan:

- `approx_file_bytes: 11489939840`

All-plain-F32 plan:

- `approx_file_bytes: 11501080960`

Delta:

- `11,141,120` bytes
- about `+0.097%`

Interpretation:

- comfortably inside the `+10%` footprint budget from `53`

## Real artifact build on DGX

Built successfully on DGX with:

```sh
/tmp/deepseek4-quantize \
  --hf ~/ds4/hf-dspark \
  --template issue468/baseline/dspark_template.gguf \
  --out /tmp/dspark_plainf32_diag.gguf \
  --overwrite \
  --tensor-type mtp.2.hc_head_fn.weight=f32 \
  --tensor-type mtp.0.hc_attn_fn.weight=f32 \
  --tensor-type mtp.1.hc_attn_fn.weight=f32 \
  --tensor-type mtp.2.hc_attn_fn.weight=f32 \
  --tensor-type mtp.0.hc_ffn_fn.weight=f32 \
  --tensor-type mtp.1.hc_ffn_fn.weight=f32 \
  --tensor-type mtp.2.hc_ffn_fn.weight=f32 \
  --tensor-type mtp.0.ffn_gate_inp.weight=f32 \
  --tensor-type mtp.1.ffn_gate_inp.weight=f32 \
  --tensor-type mtp.2.ffn_gate_inp.weight=f32
```

Observed result:

- quantizer completed successfully
- wrote:
  - `/tmp/dspark_plainf32_diag.gguf`

This proves the candidate is not merely hypothetical.

## Additional workflow improvement landed

Updated helper scripts so they no longer hardcode `./ds4`:

- `issue468/run_dspark_context_acceptance_sweep.py`
- `issue468/run_dspark_weighted_from_sweep_root.py`

New argument:

- `--ds4-bin`

Reason:

- DGX measurement currently depends on whichever runtime binary is available
- hardcoding `./ds4` makes the remote workflow unnecessarily brittle

This does **not** change inference logic. It only lets the measurement harness
point at an explicit runtime binary.

## Measurement blocker on DGX

### 1. Current branch does not build a usable DGX runtime binary

Attempted CUDA build in `~/ds4`:

- link failure on:
  - `ds4_gpu_attention_decode_raw_batch_heads_noncausal_tensor`

Observed reason:

- symbol exists in the Metal path (`ds4_metal.m`) but not in the CUDA objects
- current branch `make ds4` on DGX therefore fails at final link

Attempted CPU-only build:

- compile failure in `ds4.c`

Observed reason:

- DSpark probe/input-stage code references GPU graph fields even under the CPU
  build path
- current branch `make cpu` therefore also fails

### 2. Available prebuilt DGX binaries are too old

Tested existing binaries at:

- `~/ds4-profile-current/ds4`
- `~/ds4-codex-test/ds4`

Observed result:

- `~/ds4-profile-current/ds4` does not support `--verifier-curve-test`
- `~/ds4-codex-test/ds4` does not support `--dspark`

So neither binary can run the current acceptance measurement path.

### 3. Bundle staging itself is no longer the blocker

At least one context bundle was staged successfully onto DGX:

- `/tmp/dspark_sweep8/ctx_08192`

That means the remaining blocker is no longer capture-data availability.

It is:

- lack of a usable DSpark-capable runtime binary on DGX for the current branch

## Current state

What is complete:

1. independent review step executed
2. runtime-compatible plain-layout tensor class identified
3. candidate plan costed and confirmed in-budget
4. real candidate GGUF built on DGX
5. measurement helpers generalized to `--ds4-bin`

What is still incomplete:

1. acceptance measurement of the candidate on the 4-context root
2. mean delta vs live baseline
3. conclusion on whether plain-layout precision has real frozen-runtime headroom

## Practical next step

The next enabling move is now clear:

- obtain or build one DGX runtime binary that supports both:
  - `--dspark`
  - `--verifier-curve-test`

Once that exists, the candidate artifact is already ready:

- `/tmp/dspark_plainf32_diag.gguf`

and the first staged context bundle is already in place:

- `/tmp/dspark_sweep8/ctx_08192`

So measurement can proceed immediately after the runtime binary issue is solved.
