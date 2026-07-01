# `layer2` `ffn_down_shexp` Recoverable-Gap `Q8_0` Probe

Date: 2026-07-01

## Purpose

Record the last nearby dense legal `Q8_0` follow-up after `76`.

The target tensor was:

- `mtp.2.ffn_down_shexp.weight`

This was the remaining obvious local dense-family probe after:

- `main_proj`
- `mtp.2.ffn_gate_shexp.weight`
- `mtp.2.ffn_up_shexp.weight`

## Dense imatrix

Command:

```sh
PYTHONPATH=. issue468/.venv/bin/python issue468/build_dense_recoverable_imatrix.py \
  --sweep-root /private/tmp/dspark_sweep2ctx \
  --baseline-label baseline \
  --oracle-details-json /private/tmp/dspark_sweep2ctx/oracle-envelope-existing256.details.json \
  --tensor-name mtp.2.ffn_down_shexp.weight \
  --out /private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_layer2_down_shexp.imatrix.dat
```

Output:

- `/private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_layer2_down_shexp.imatrix.dat`

Properties:

- entries: `1`
- tensor: `mtp.2.ffn_down_shexp.weight`
- calls: `55`
- vector length: `2048`

This uses the new dense builder path from `75` and accumulates the shared expert
intermediate `silu(gate) * up`, which is the correct column-side signal for the
shared down projection.

## Sanity check

Command:

```sh
gguf-tools/deepseek4-quantize \
  --hf ../ds4/hf-dspark \
  --template ../ds4/gguf/dspark.gguf \
  --compare-tensor mtp.2.ffn_down_shexp.weight \
  --imatrix /private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_layer2_down_shexp.imatrix.dat
```

Result:

- byte-compare `FAIL`
- mismatch count: `3626846`

So the dense imatrix does change the intended tensor.

## Candidate

Candidate:

- `/private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_layer2_down_shexp_q8imat.gguf`

Build command:

```sh
gguf-tools/deepseek4-quantize \
  --hf ../ds4/hf-dspark \
  --template ../ds4/gguf/dspark.gguf \
  --out /private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_layer2_down_shexp_q8imat.gguf \
  --overwrite \
  --imatrix /private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_layer2_down_shexp.imatrix.dat
```

Integrity check:

- payload diff count vs baseline: `1`
- the only changed tensor payload is:
  - `mtp.2.ffn_down_shexp.weight`

## Measurement

Command:

```sh
python3 issue468/run_dspark_weighted_from_sweep_root.py \
  --ds4-bin ./ds4 \
  --backend metal \
  --sweep-root /private/tmp/dspark_sweep2ctx \
  --model ../ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --baseline-dspark ../ds4/gguf/dspark.gguf \
  --candidate-gguf /private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_layer2_down_shexp_q8imat.gguf \
  --measure-script issue468/baseline/dspark_capture/measure_metal_b2.py \
  --measure-python issue468/.venv/bin/python \
  --imatrix-in /private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx.imatrix.dat \
  --run-label recoverablegap_boosted2ctx_layer2_down_shexp_q8imat \
  --steps 19 --trials 256 --ctx-size 4096 --power 100
```

Results:

- `ctx_08192`
  - accepted delta: `0.0%`
  - committed delta: `0.0%`
- `ctx_16384`
  - accepted delta: `0.0%`
  - committed delta: `0.0%`

Two-context mean:

- accepted delta: `0.0%`
- committed delta: `0.0%`

## Read

This probe is exactly baseline on both clean `2ctx` contexts.

Compared with prior dense probes:

- `74` `main_proj`: mean accepted `-0.0241%`
- `75` `ffn_gate_shexp`: mean accepted `-0.0440%`
- `76` `ffn_up_shexp`: mean accepted `-0.0194%`
- `77` `ffn_down_shexp`: mean accepted `0.0%`

So the current dense-nearby ordering is:

1. `ffn_down_shexp` best
2. `ffn_up_shexp`
3. `main_proj`
4. `ffn_gate_shexp`

But "best" here still means:

- no measurable gain on the current `2ctx` root

## Immediate implication

This result exhausts the obvious nearby dense legal `Q8_0` tensor-family
follow-ups from the current branch plan.

The next branch choice should now be made with another independent review,
rather than by continuing to spend local search budget on more of the same
dense-family neighborhood.
