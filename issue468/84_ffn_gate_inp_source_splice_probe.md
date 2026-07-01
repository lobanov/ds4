# `ffn_gate_inp` Source-Splice Probe

Date: 2026-07-01

## Purpose

Record the first router-sensitive source-splice probe after the flat 4-context
 source ceiling result in `83_fp8_source_ceiling_4ctx.md`.

Lead hypothesis from `59` and `79`:

- `ffn_gate_inp` might matter disproportionately because small routing changes
  can alter selected experts and accepted tokens

This note tests that hypothesis directly by replacing `ffn_gate_inp.weight`
from the source reference checkpoint inside the baseline oracle path.

## Tooling

Added:

- `issue468/ref/measure_spliced_ref_oracle_b2.py`

Purpose:

- load the baseline oracle weights from `dspark.gguf`
- splice selected dense tensors from the source reference checkpoint
- score the mixed artifact on the same B2 oracle methodology

Because the local environment lacks `safetensors`, the actual source tensor
used in this probe was exported from DGX to local `.npy` files and loaded via
the scorer's `--replace-npy` fallback.

## Probe 1: stage-2 only

Replaced:

- `mtp.2.ffn_gate_inp.weight`

Contexts:

- `ctx_08192`
- `ctx_16384`

### `ctx_08192`

- splice committed:
  - `4.4300986842105265`
- source-reference committed:
  - `4.425986842105263`
- baseline committed:
  - `4.473684210526316`
- splice vs baseline:
  - `-0.974264705882355%`

### `ctx_16384`

- splice committed:
  - `4.493421052631579`
- source-reference committed:
  - `4.503700657894737`
- baseline committed:
  - `4.519736842105263`
- splice vs baseline:
  - `-0.5822416302765587%`

## Probe 2: all three drafter stages

Replaced:

- `mtp.0.ffn_gate_inp.weight`
- `mtp.1.ffn_gate_inp.weight`
- `mtp.2.ffn_gate_inp.weight`

Observed result:

- bit-identical committed scores to the stage-2-only probe on both tested
  contexts

That strongly suggests the `ffn_gate_inp` family is not carrying actionable
headroom in the current baseline artifact.

## Why the result is flat

Direct tensor comparison between baseline GGUF and source-exported
`ffn_gate_inp` arrays:

- stage 0:
  - max abs diff `6.079673767089844e-05`
  - mean abs diff `2.369823270953475e-08`
- stage 1:
  - max abs diff `6.079673767089844e-05`
  - mean abs diff `2.3924570768940612e-08`
- stage 2:
  - max abs diff `6.079673767089844e-05`
  - mean abs diff `2.5777215029165745e-08`

Interpretation:

- the baseline artifact already preserves `ffn_gate_inp.weight` almost exactly
  relative to the source checkpoint
- so replacing it with the source copy cannot reasonably be expected to move
  routing much

## Conclusion

This specific router-sensitive source-splice avenue is exhausted.

Meaning:

- `ffn_gate_inp` was the right first router-sensitive tensor to check
- but **source-vs-baseline difference in that tensor is too small to matter**

So the next router-sensitive lead should not be:

- more `ffn_gate_inp` source-splice variants

Instead it should come from a different mechanism, for example:

- another tensor class whose source/baseline difference is materially larger
- or an acceptance-local perturbation/search objective rather than direct
  source-copy splicing
