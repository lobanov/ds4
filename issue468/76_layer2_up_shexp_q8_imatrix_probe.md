# `layer2` `ffn_up_shexp` Recoverable-Gap `Q8_0` Probe

Date: 2026-07-01

## Purpose

Record the next dense legal `Q8_0` follow-up after `75`.

The target tensor was:

- `mtp.2.ffn_up_shexp.weight`

This stayed on the same newly opened dense-oracle path but shifted from the
negative shared-gate tensor to the shared-up tensor, which is closer to the
value path inside the layer-2 shared expert.

## Dense imatrix

Command:

```sh
PYTHONPATH=. issue468/.venv/bin/python issue468/build_dense_recoverable_imatrix.py \
  --sweep-root /private/tmp/dspark_sweep2ctx \
  --baseline-label baseline \
  --oracle-details-json /private/tmp/dspark_sweep2ctx/oracle-envelope-existing256.details.json \
  --tensor-name mtp.2.ffn_up_shexp.weight \
  --out /private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_layer2_up_shexp.imatrix.dat
```

Output:

- `/private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_layer2_up_shexp.imatrix.dat`

Properties:

- entries: `1`
- tensor: `mtp.2.ffn_up_shexp.weight`
- calls: `55`
- vector length: `4096`

## Sanity check

Command:

```sh
gguf-tools/deepseek4-quantize \
  --hf ../ds4/hf-dspark \
  --template ../ds4/gguf/dspark.gguf \
  --compare-tensor mtp.2.ffn_up_shexp.weight \
  --imatrix /private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_layer2_up_shexp.imatrix.dat
```

Result:

- byte-compare `FAIL`
- mismatch count: `3507762`

So the dense imatrix does change the intended tensor.

## Candidate

Candidate:

- `/private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_layer2_up_shexp_q8imat.gguf`

Build command:

```sh
gguf-tools/deepseek4-quantize \
  --hf ../ds4/hf-dspark \
  --template ../ds4/gguf/dspark.gguf \
  --out /private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_layer2_up_shexp_q8imat.gguf \
  --overwrite \
  --imatrix /private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_layer2_up_shexp.imatrix.dat
```

Integrity check:

- payload diff count vs baseline: `1`
- the only changed tensor payload is:
  - `mtp.2.ffn_up_shexp.weight`

## Measurement

Command:

```sh
python3 issue468/run_dspark_weighted_from_sweep_root.py \
  --ds4-bin ./ds4 \
  --backend metal \
  --sweep-root /private/tmp/dspark_sweep2ctx \
  --model ../ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --baseline-dspark ../ds4/gguf/dspark.gguf \
  --candidate-gguf /private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_layer2_up_shexp_q8imat.gguf \
  --measure-script issue468/baseline/dspark_capture/measure_metal_b2.py \
  --measure-python issue468/.venv/bin/python \
  --imatrix-in /private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx.imatrix.dat \
  --run-label recoverablegap_boosted2ctx_layer2_up_shexp_q8imat \
  --steps 19 --trials 256 --ctx-size 4096 --power 100
```

Results:

- `ctx_08192`
  - accepted delta: `0.0%`
  - committed delta: `0.0%`
- `ctx_16384`
  - accepted delta: `-0.038823643598952864%`
  - committed delta: `-0.022765560260441114%`

Two-context mean:

- accepted delta: `-0.019411821799476432%`
- committed delta: `-0.011382780130220557%`

## Read

This is the strongest dense legal `Q8_0` result so far in the current lane, but
it is still not positive.

Compared with prior dense probes:

- `74` `main_proj`: mixed-sign, near-flat, mean accepted `-0.0241%`
- `75` `ffn_gate_shexp`: negative on both contexts, mean accepted `-0.0440%`
- `76` `ffn_up_shexp`: flat on `8192`, negative on `16384`, mean accepted
  `-0.0194%`

So the current ordering is:

1. `ffn_up_shexp` best
2. `main_proj`
3. `ffn_gate_shexp` worst

But none of the tested dense legal `Q8_0` tensors are positive yet on the clean
`2ctx` root.

## Immediate implication

If the dense lane continues immediately, the remaining nearby follow-up is:

- `mtp.2.ffn_down_shexp.weight`

If that is also negative, the dense legal `Q8_0` family will have a much weaker
case as the lead lane and should likely trigger another independent review
before spending more local search budget here.
