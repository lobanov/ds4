# Dense `Q8_0` Combo Closeout

Date: 2026-07-01

## Purpose

Record the one coupled dense legal `Q8_0` closeout run recommended by the
independent review after `77`.

The branch question was:

- do the nearby dense legal `Q8_0` tensors compensate each other when calibrated
  together, even though the isolated probes were flat or negative?

This was the one remaining honest control before leaving the dense `Q8_0` lane
as the lead branch.

## Combined dense imatrix

Command:

```sh
PYTHONPATH=. issue468/.venv/bin/python issue468/build_dense_recoverable_imatrix.py \
  --sweep-root /private/tmp/dspark_sweep2ctx \
  --baseline-label baseline \
  --oracle-details-json /private/tmp/dspark_sweep2ctx/oracle-envelope-existing256.details.json \
  --tensor-name mtp.0.main_proj.weight \
  --tensor-name mtp.2.ffn_gate_shexp.weight \
  --tensor-name mtp.2.ffn_up_shexp.weight \
  --tensor-name mtp.2.ffn_down_shexp.weight \
  --out /private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_dense_combo.imatrix.dat
```

Output:

- `/private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_dense_combo.imatrix.dat`

Entries:

- `mtp.0.main_proj.weight`
- `mtp.2.ffn_gate_shexp.weight`
- `mtp.2.ffn_up_shexp.weight`
- `mtp.2.ffn_down_shexp.weight`

## Sanity check

Each intended dense tensor was regenerated with:

```sh
gguf-tools/deepseek4-quantize \
  --hf ../ds4/hf-dspark \
  --template ../ds4/gguf/dspark.gguf \
  --compare-tensor <tensor> \
  --imatrix /private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_dense_combo.imatrix.dat
```

Result:

- all four intended dense tensors changed under the combined imatrix

## Candidate

Candidate:

- `/private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_dense_combo_q8imat.gguf`

Build command:

```sh
gguf-tools/deepseek4-quantize \
  --hf ../ds4/hf-dspark \
  --template ../ds4/gguf/dspark.gguf \
  --out /private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_dense_combo_q8imat.gguf \
  --overwrite \
  --imatrix /private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_dense_combo.imatrix.dat
```

Integrity check:

- payload diff count vs baseline: `4`
- changed tensors:
  - `mtp.0.main_proj.weight`
  - `mtp.2.ffn_gate_shexp.weight`
  - `mtp.2.ffn_up_shexp.weight`
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
  --candidate-gguf /private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_dense_combo_q8imat.gguf \
  --measure-script issue468/baseline/dspark_capture/measure_metal_b2.py \
  --measure-python issue468/.venv/bin/python \
  --imatrix-in /private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx.imatrix.dat \
  --run-label recoverablegap_boosted2ctx_dense_combo_q8imat \
  --steps 19 --trials 256 --ctx-size 4096 --power 100
```

Results:

- `ctx_08192`
  - accepted delta: `-0.004910867750323668%`
  - committed delta: `-0.022481003552001628%`
- `ctx_16384`
  - accepted delta: `-0.038823643598952864%`
  - committed delta: `-0.022765560260441114%`

Two-context mean:

- accepted delta: `-0.021867255674638267%`
- committed delta: `-0.02262328190622137%`

## Read

This closeout control does **not** rescue the dense lane.

What it shows:

- allowing the nearby dense tensors to move together does not recover a hidden
  positive interaction
- the coupled dense candidate remains slightly negative overall

Comparison:

- the combo result is slightly worse than the best isolated result
  (`ffn_up_shexp`)
- and clearly better than the worst isolated result (`ffn_gate_shexp`)
- but still not positive

## Conclusion

This closes the nearby dense legal `Q8_0` branch as the lead lane.

The branch evidence is now:

- `main_proj`: negative
- `ffn_gate_shexp`: negative
- `ffn_up_shexp`: near-flat but negative overall
- `ffn_down_shexp`: exactly flat
- dense combo closeout: negative

So the current dense-imatrix neighborhood does not justify more immediate
search budget ahead of the next branch.
