# `layer2` `ffn_gate_shexp` Recoverable-Gap `Q8_0` Probe

Date: 2026-07-01

## Purpose

Record the next dense legal `Q8_0` probe after `74`.

The target tensor was:

- `mtp.2.ffn_gate_shexp.weight`

This was chosen because `74` showed:

- `main_proj` is measurable but slightly negative
- the next dense-family follow-up should move closer to layer-2 acceptance
  decisions

## Tooling extension

`issue468/build_dense_recoverable_imatrix.py` now supports layer-2 shared
expert tensors directly from the numpy oracle path.

Current supported dense entries are:

- `mtp.0.main_proj.weight`
- `mtp.2.ffn_gate_shexp.weight`
- `mtp.2.ffn_up_shexp.weight`
- `mtp.2.ffn_down_shexp.weight`

For `ffn_gate_shexp` and `ffn_up_shexp`, the builder now:

- replays recoverable steps through layers `0` and `1`
- runs the attention half of layer `2`
- captures the layer-2 FFN input after `hc_pre + rmsnorm`
- accumulates weighted squared activation mass over the `5` drafted positions

For `ffn_down_shexp`, the same path also derives the shared expert intermediate
`silu(gate) * up`.

This keeps the dense lane inside oracle-only artifact work without touching the
frozen DS4 runtime.

## Dense imatrix

Command:

```sh
PYTHONPATH=. issue468/.venv/bin/python issue468/build_dense_recoverable_imatrix.py \
  --sweep-root /private/tmp/dspark_sweep2ctx \
  --baseline-label baseline \
  --oracle-details-json /private/tmp/dspark_sweep2ctx/oracle-envelope-existing256.details.json \
  --tensor-name mtp.2.ffn_gate_shexp.weight \
  --out /private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_layer2_gate_shexp.imatrix.dat
```

Output:

- `/private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_layer2_gate_shexp.imatrix.dat`

Properties:

- entries: `1`
- tensor: `mtp.2.ffn_gate_shexp.weight`
- calls: `55`
- vector length: `4096`

## Sanity check

Command:

```sh
gguf-tools/deepseek4-quantize \
  --hf ../ds4/hf-dspark \
  --template ../ds4/gguf/dspark.gguf \
  --compare-tensor mtp.2.ffn_gate_shexp.weight \
  --imatrix /private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_layer2_gate_shexp.imatrix.dat
```

Result:

- byte-compare `FAIL`
- mismatch count: `3591968`

So the new dense imatrix does change the intended tensor.

## Candidate

Candidate:

- `/private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_layer2_gate_shexp_q8imat.gguf`

Build command:

```sh
gguf-tools/deepseek4-quantize \
  --hf ../ds4/hf-dspark \
  --template ../ds4/gguf/dspark.gguf \
  --out /private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_layer2_gate_shexp_q8imat.gguf \
  --overwrite \
  --imatrix /private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_layer2_gate_shexp.imatrix.dat
```

Integrity check:

- payload diff count vs baseline: `1`
- the only changed tensor payload is:
  - `mtp.2.ffn_gate_shexp.weight`

## Measurement

Command:

```sh
python3 issue468/run_dspark_weighted_from_sweep_root.py \
  --ds4-bin ./ds4 \
  --backend metal \
  --sweep-root /private/tmp/dspark_sweep2ctx \
  --model ../ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --baseline-dspark ../ds4/gguf/dspark.gguf \
  --candidate-gguf /private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_layer2_gate_shexp_q8imat.gguf \
  --measure-script issue468/baseline/dspark_capture/measure_metal_b2.py \
  --measure-python issue468/.venv/bin/python \
  --imatrix-in /private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx.imatrix.dat \
  --run-label recoverablegap_boosted2ctx_layer2_gate_shexp_q8imat \
  --steps 19 --trials 256 --ctx-size 4096 --power 100
```

Results:

- `ctx_08192`
  - accepted delta: `-0.04910867750331782%`
  - committed delta: `-0.04496200710399911%`
- `ctx_16384`
  - accepted delta: `-0.03882364359895085%`
  - committed delta: `-0.02276556026044041%`

Two-context mean:

- accepted delta: `-0.043966160551134334%`
- committed delta: `-0.03386378368221976%`

## Read

This probe is a clearer negative result than `74`.

What it shows:

- dense recoverable-gap calibration on `mtp.2.ffn_gate_shexp.weight` is live
- but the first isolated `layer2` shared-gate `Q8_0` probe is negative on both
  clean `2ctx` contexts

Compared with `74`:

- `main_proj` was mixed-sign and near-flat overall
- `ffn_gate_shexp` is consistently negative

So `ffn_gate_shexp` does not currently look like the best dense legal `Q8_0`
lead.

## Immediate implication

If the dense lane continues immediately, the next stronger follow-ups are:

- `mtp.2.ffn_up_shexp.weight`
- `mtp.2.ffn_down_shexp.weight`

Those keep the search on the same newly opened dense-oracle path without
reopening the routed-local lane.
