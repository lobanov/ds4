# FP8 Torch-Fallback Harness Revival

Date: 2026-07-01

## Purpose

Record the first successful revival of the DSpark source-precision reference
path after the tilelang blocker in `54_fp8_reference_harness_revival_status.md`.

The key change is that the source-ceiling path no longer depends on tilelang
executing successfully on the DGX Python stack.

## What was added

New files:

- `issue468/ref/kernel_torch_fallback.py`
- `issue468/ref/dspark_ref_harness_torch.py`
- `issue468/ref/score_ref_logits_case.py`

Design:

- inject a pure-torch module as `kernel` before importing
  `issue468/ref/inference/model.py`
- preserve the official reference model structure and checkpoint loading
- emulate the required kernel surface in torch:
  - `act_quant`
  - `fp4_act_quant`
  - `fp8_gemm`
  - `fp4_gemm`
  - `sparse_attn`
  - `hc_split_sinkhorn`
- save reference-path logits/tokens to a scoreable `.ref_torch.npz`

This keeps the experiment source-oriented:

- official reference model code still drives the forward
- official converted source checkpoint still provides weights
- only the kernel backend is replaced

## DGX validation

Host:

- `dgx-direct`
- repo: `~/ds4`
- venv: `~/dref-venv`

Real captured case:

- input bundle: `/tmp/dspark_ref_case_152_153.npz`

The torch fallback harness ran successfully on that case:

- checkpoint coverage:
  - `ckpt tensors=4708`
  - `model params=4714`
  - `populated=4705`
  - `missing=9`
- the missing names were the expected shared aliases / hash-routing tables:
  - `mtp.{0,1,2}.embed.weight`
  - `mtp.{0,1,2}.head.weight`
  - `mtp.{0,1,2}.ffn.gate.tid2eid`

That means the old FP8 branch is now unblocked at the forward-pass level.

## First scored outputs

### Sampled path (`temperature=1`)

Harness output:

- draft tokens: `[1309, 304, 5085, 15255, 6177, 28]`

Single-case scoring via `score_ref_logits_case.py` with `decode_step=1`:

- sampled accepted prefix vs truth: `1/5`
- greedy accepted prefix vs truth from the same logits: `2/5`
- sampled-chain accept probabilities:
  - `[1.0000, 0.7236, 0.0001, 0.0029, 0.0000]`
- sampled-chain per-position `1 - TV` values:
  - `[0.9877, 0.6354, 0.2395, 0.0015, 0.0000]`
- expected committed tokens on that sampled chain:
  - `2.7237`

### Greedy path (`temperature=0`)

Harness output:

- draft tokens: `[1309, 304, 5553, 15255, 4181, 28]`

Single-case scoring:

- sampled/greedy accepted prefix vs truth: `2/5`
- sampled-chain accept probabilities:
  - `[1.0000, 1.0000, 0.0000, 0.0000, 0.0000]`
- sampled-chain per-position `1 - TV` values:
  - `[0.9877, 0.6354, 0.0121, 0.0015, 0.0042]`
- expected committed tokens on that greedy chain:
  - `3.0000`

## Interpretation

Three conclusions are now justified.

1. The FP8 source-ceiling lane is no longer blocked by tilelang.
2. The official source checkpoint can now be driven through the reference model
   on the DGX with a real captured case.
3. The scoring path is partially revived:
   - enough for single-case greedy/sample diagnostics
   - not yet enough for a proper multi-context ceiling note

## What remains before closing the FP8 lane

Still needed:

- a sweep-root adapter instead of the current single-case scorer
- repeated sampled runs or a more direct reference-path B2 estimator if Monte
  Carlo committed-score comparison is required
- the actual FP8-vs-`Q4_K` ceiling note on the current `ctx_08192..ctx_32768`
  root

So this is a real unblock, not the finished FP8 result.
