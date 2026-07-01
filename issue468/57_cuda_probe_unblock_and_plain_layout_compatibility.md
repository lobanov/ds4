# CUDA Probe Unblocked; Broad Plain-F32 Candidate Invalid Under Frozen Runtime

Date: 2026-07-01

## Purpose

Record the first live DGX result after unblocking the CUDA DSpark probe path,
and tighten the frozen-runtime search space based on what actually runs.

This note follows:

- `issue468/53_frozen_runtime_quantization_goal.md`
- `issue468/56_plain_layout_f32_candidate_and_runtime_blocker.md`

## What was unblocked

Two independent runtime/harness blockers were removed:

1. CUDA build blocker in the current branch
   - `ds4_gpu_attention_decode_raw_batch_heads_noncausal_tensor` existed only
     in the Metal backend
   - a CUDA implementation was added so `make ds4` now links on DGX

2. DGX probe harness backend selection
   - `--verifier-curve-test` in `ds4_cli.c` was forcibly resetting the backend
     to Metal on non-ROCm builds
   - acceptance scripts also hardcoded `--metal`
   - result: even explicit CUDA requests on DGX were being redirected into a
     Metal-only path

After fixing both, the current branch runs DSpark probe sessions on DGX with:

- backend: `cuda`
- target model: `~/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`

## Artifact correction

The previous note used `issue468/baseline/dspark_template.gguf` as if it were a
runnable drafter artifact. It is not.

Observed DGX failure:

- `ds4: tensor points outside GGUF file`

Interpretation:

- the template is valid for quantizer layout planning
- but probe execution requires a fully materialized DSpark GGUF

So a real baseline artifact was built on DGX:

- `/tmp/dspark_baseline_full.gguf`

## Live baseline reprobe on DGX

Context bundle:

- `/tmp/dspark_sweep8/ctx_08192`

Command path:

- rebuilt `~/ds4/ds4`
- `--backend cuda`
- baseline drafter: `/tmp/dspark_baseline_full.gguf`

Observed result:

- greedy match: `87/95`
- avg prefix: `4.37/5`
- prefix hist: `[1 0 0 2 3 13]`
- draft median cost: `31.55 ms`

This is the first confirmed live CUDA DSpark probe result on the current
branch.

## Broad plain-layout F32 candidate result

Candidate:

- `/tmp/dspark_plainf32_diag.gguf`

Overrides:

- all `hc_attn_fn`
- all `hc_ffn_fn`
- all `ffn_gate_inp`
- `mtp.2.hc_head_fn`

Observed result on the same `ctx_08192` bundle:

- greedy match: `0/95`
- avg prefix: `0.00/5`
- every draft token emitted as `-1`
- draft median cost: `24.45 ms`

Interpretation:

- the broad "plain-layout F16->F32" candidate is **not** a valid
  frozen-runtime candidate, even though the loader/validator admits those
  tensors as plain-layout

## Why it failed

The relevant DSpark runtime call sites are not uniformly type-generic.

Actually type-generic:

- `metal_graph_matmul_plain_tensor(...)`
  - dispatches `F16`, `F32`, and `Q8_0`

Still hardwired to F16 kernels in the DSpark graph path:

- `ds4_gpu_matmul_f16_tensor(... layer->hc_attn_fn ...)`
- `ds4_gpu_matmul_f16_tensor(... layer->hc_ffn_fn ...)`
- `ds4_gpu_matmul_f16_tensor(... layer->ffn_gate_inp ...)`

So under the frozen-runtime constraint:

- `hc_attn_fn`, `hc_ffn_fn`, and `ffn_gate_inp` are **validator-compatible**
  but not **execution-compatible**

This narrows the true search space materially.

## Narrow runtime-compatible follow-up: `hc_head_fn` only

Because `mtp.2.hc_head_fn` is exercised through `metal_graph_matmul_plain_tensor`,
it is still a valid frozen-runtime precision candidate.

Built on DGX:

- `/tmp/dspark_hcheadf32_diag.gguf`

Override:

- `mtp.2.hc_head_fn.weight=f32`

Observed result on the same `ctx_08192` bundle:

- greedy match: `87/95`
- avg prefix: `4.37/5`
- prefix hist: `[1 0 0 2 3 13]`
- draft median cost: `27.89 ms`

Interpretation:

- no visible quality gain on the first live context
- the one clearly runtime-compatible plain-layout precision upgrade does not
  currently justify the avenue

## Updated conclusion

The original "all plain-layout tensors to F32" avenue is now split:

1. broad plain-layout upgrade
   - invalid under the frozen runtime because several accepted tensors still run
     through F16-only kernels

2. actually runtime-compatible subset
   - currently reduced to `hc_head_fn` on the measured path
   - first live result shows no quality gain at `ctx_08192`

## Practical next step

Treat the broad plain-layout branch as exhausted under the frozen-runtime
constraint and trigger the independent `gpt-5.5 xhigh` review step again with
these new facts:

- CUDA probe path is now live
- broad plain-layout F32 candidate collapses to zero quality
- true runtime-compatible subset is much smaller than validator acceptance
- `hc_head_fn`-only F32 shows no first-context gain
