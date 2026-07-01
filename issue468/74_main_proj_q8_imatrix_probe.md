# `main_proj` Recoverable-Gap `Q8_0` Probe

Date: 2026-07-01

## Purpose

Record the first actual dense legal `Q8_0` probe after the lane shift in `73`.

The target family was:

- `mtp.0.main_proj.weight`

This was chosen because:

- it is explicitly in scope in `58` and `59`
- it is deployment-plausible under the frozen runtime
- it is small enough to isolate cleanly as one exact tensor

## Immediate blocker found

The dense lane was not directly runnable with the existing recoverable-gap
artifacts.

Reason:

- the current recoverable-gap imatrix file
  `/private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx.imatrix.dat`
  contains only the `9` routed expert tensors:
  - `mtp.{0,1,2}.ffn_gate_exps.weight`
  - `mtp.{0,1,2}.ffn_up_exps.weight`
  - `mtp.{0,1,2}.ffn_down_exps.weight`

So even after the branch pivot to dense legal `Q8_0`, the existing DSpark
collector artifacts still did **not** provide any dense calibration vector for:

- `main_proj`
- attention projections
- shared experts

This explained an earlier false start:

- rebuilding `mtp.0.main_proj.weight` as `Q8_0` with the routed-only imatrix was
  byte-identical to baseline

## Tooling change 1: imatrix-aware `Q8_0`

Updated:

- `gguf-tools/quants.c`

Change:

- `Q8_0` quantization now consumes an imatrix vector when one is present
- the no-imatrix path is intentionally unchanged

Current approach:

- per 32-value `Q8_0` block, scan a small neighborhood around the baseline
  max-abs scale
- compute block codes by clamped nearest rounding
- refine the block scale with weighted least squares
- pick the lowest weighted squared-error candidate

This keeps the dense lane compatible with the current quantizer surface while
making recoverable-gap calibration actually matter for `Q8_0`.

## Tooling change 2: dense recoverable-step imatrix builder

Added:

- `issue468/build_dense_recoverable_imatrix.py`

Purpose:

- synthesize a one-entry imatrix file for a dense tensor directly from
  recoverable-step capture bundles
- avoid changing the frozen DSpark runtime / collector path

Current scope:

- first target is `mtp.0.main_proj.weight`
- accumulates weighted squared decode-time `main_hidden` activity over
  recoverable steps

## Dense `main_proj` imatrix

Command:

```sh
PYTHONPATH=. issue468/.venv/bin/python issue468/build_dense_recoverable_imatrix.py \
  --sweep-root /private/tmp/dspark_sweep2ctx \
  --baseline-label baseline \
  --oracle-details-json /private/tmp/dspark_sweep2ctx/oracle-envelope-existing256.details.json \
  --out /private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_main_proj.imatrix.dat
```

Output:

- `/private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_main_proj.imatrix.dat`

Properties:

- entries: `1`
- tensor: `mtp.0.main_proj.weight`
- recoverable calls: `11`
- vector length: `12288`

## Sanity checks

### 1. No-imatrix path still matches baseline

Command:

```sh
gguf-tools/deepseek4-quantize \
  --hf ../ds4/hf-dspark \
  --template ../ds4/gguf/dspark.gguf \
  --compare-tensor mtp.0.main_proj.weight
```

Result:

- byte-compare `OK`

### 2. Routed-only imatrix still matches baseline

Command:

```sh
gguf-tools/deepseek4-quantize \
  --hf ../ds4/hf-dspark \
  --template ../ds4/gguf/dspark.gguf \
  --compare-tensor mtp.0.main_proj.weight \
  --imatrix /private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx.imatrix.dat
```

Result:

- byte-compare `OK`

Read:

- confirms the old recoverable-gap imatrix does not calibrate `main_proj`

### 3. Dense `main_proj` imatrix changes the tensor

Command:

```sh
gguf-tools/deepseek4-quantize \
  --hf ../ds4/hf-dspark \
  --template ../ds4/gguf/dspark.gguf \
  --compare-tensor mtp.0.main_proj.weight \
  --imatrix /private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_main_proj.imatrix.dat
```

Result:

- byte-compare `FAIL`
- mismatch count: `22020326`

So the dense calibration path is now live.

## Candidate

Candidate:

- `/private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_mainproj_q8imat.gguf`

Build command:

```sh
gguf-tools/deepseek4-quantize \
  --hf ../ds4/hf-dspark \
  --template ../ds4/gguf/dspark.gguf \
  --out /private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_mainproj_q8imat.gguf \
  --overwrite \
  --imatrix /private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_main_proj.imatrix.dat
```

Integrity check:

- payload diff count vs baseline: `1`
- the only changed tensor payload is:
  - `mtp.0.main_proj.weight`

## Measurement

Command:

```sh
python3 issue468/run_dspark_weighted_from_sweep_root.py \
  --ds4-bin ./ds4 \
  --backend metal \
  --sweep-root /private/tmp/dspark_sweep2ctx \
  --model ../ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --baseline-dspark ../ds4/gguf/dspark.gguf \
  --candidate-gguf /private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_mainproj_q8imat.gguf \
  --measure-script issue468/baseline/dspark_capture/measure_metal_b2.py \
  --measure-python issue468/.venv/bin/python \
  --imatrix-in /private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx.imatrix.dat \
  --run-label recoverablegap_boosted2ctx_mainproj_q8imat \
  --steps 19 --trials 256 --ctx-size 4096 --power 100
```

Results:

- `ctx_08192`
  - accepted delta: `+0.02455433875166077%`
  - committed delta: `0.0%`
- `ctx_16384`
  - accepted delta: `-0.07279433174804448%`
  - committed delta: `-0.04553112052088223%`

Two-context mean:

- accepted delta: `-0.024119996498191854%`
- committed delta: `-0.022765560260441114%`

## Read

This first dense legal `Q8_0` probe is useful but not positive.

What it shows:

- `main_proj` is now a real measurable dense `Q8_0` branch
- but the first recoverable-gap-weighted `main_proj` requantization is slightly
  negative overall on the clean `2ctx` root

What it does not show:

- that the dense legal `Q8_0` lane is exhausted

The main branch value here is structural:

- dense `Q8_0` calibration required new tensor-specific imatrix generation
- that path now exists without changing frozen inference/runtime code

## Immediate implication

The next dense-family candidates should move to tensors where the recoverable
signal may be more local to acceptance decisions, such as:

- `mtp.2.ffn_gate_shexp.weight`
- `mtp.2.ffn_up_shexp.weight`
- `mtp.2.ffn_down_shexp.weight`

before treating the dense legal `Q8_0` lane as weak overall.
