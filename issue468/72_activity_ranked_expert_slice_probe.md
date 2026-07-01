# Activity-Ranked `layer2gateup` Expert-Slice Probe

Date: 2026-07-01

## Purpose

Record the first routed-local follow-up that ranks donor expert slices by
recoverable-step usage rather than by raw imatrix hotspot mass or donor/base
byte delta.

This is the direct continuation after:

- `issue468/69_layer2_tensor_part_splice_results.md`
- `issue468/70_layer2_gateup_long_context_stability.md`
- `issue468/71_sparse_up_block_delta_probe.md`

The branch question was:

- can the weak but real `layer2gateup` tensor-part signal be recovered more
  efficiently by copying only the most recoverably active routed experts?

## New tooling used

### 1. Recoverable-step expert usage summarizer

Added:

- `issue468/summarize_recoverable_expert_usage.py`

Purpose:

- replay the numpy oracle path on recoverable-gap bundle steps
- extract which routed experts `mtp.2` actually selects
- rank experts by recoverable-step gate-weight mass

Current scope:

- target layer `2`
- uses the existing recoverable-gap envelope details JSON
- records both per-context selections and aggregate expert mass

### 2. Expert-slice splicing mode

Updated:

- `gguf-tools/mixed/splice_mixed_expert_layers_gguf.py`

New capability:

- `--expert-select <tensor_name>:id,id,...`

This copies only selected routed-expert slices from the donor GGUF into the
baseline tensor while keeping the overall candidate footprint unchanged.

The implementation assumes the existing DSpark routed tensor layout already used
by the oracle loader:

- dims `[in_dim, out_dim, n_exp]`
- one expert slice is contiguous

## Recoverable usage ranking

Command:

```sh
PYTHONPATH=. issue468/.venv/bin/python issue468/summarize_recoverable_expert_usage.py \
  --sweep-root /private/tmp/dspark_sweep2ctx \
  --baseline-label baseline \
  --oracle-details-json /private/tmp/dspark_sweep2ctx/oracle-envelope-existing256.details.json \
  --json-out /private/tmp/dspark_sweep2ctx/recoverable_layer2_expert_usage.json
```

Output:

- `/private/tmp/dspark_sweep2ctx/recoverable_layer2_expert_usage.json`

Top recoverable-mass experts for `layer=2`:

1. `183`
2. `131`
3. `222`
4. `218`
5. `140`
6. `61`
7. `38`
8. `186`

The first probe used exactly those top `8` experts.

## Candidate

Candidate:

- `/private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_layer2gateup_top8experts.gguf`

Construction command:

```sh
python3 gguf-tools/mixed/splice_mixed_expert_layers_gguf.py \
  --base ../ds4/gguf/dspark.gguf \
  --donor /private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_local.gguf \
  --out /private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_layer2gateup_top8experts.gguf \
  --expert-select mtp.2.ffn_gate_exps.weight:183,131,222,218,140,61,38,186 \
  --expert-select mtp.2.ffn_up_exps.weight:183,131,222,218,140,61,38,186 \
  --force
```

Interpretation:

- preserve the earlier positive `layer2gateup` coupling
- but localize it to the most recoverably active expert slices only

## Measurement

Command:

```sh
python3 issue468/run_dspark_weighted_from_sweep_root.py \
  --ds4-bin ./ds4 \
  --backend metal \
  --sweep-root /private/tmp/dspark_sweep2ctx \
  --model ../ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --baseline-dspark ../ds4/gguf/dspark.gguf \
  --candidate-gguf /private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx_layer2gateup_top8experts.gguf \
  --measure-script issue468/baseline/dspark_capture/measure_metal_b2.py \
  --measure-python issue468/.venv/bin/python \
  --imatrix-in /private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx.imatrix.dat \
  --run-label recoverablegap_boosted2ctx_layer2gateup_top8experts \
  --steps 19 --trials 256 --ctx-size 4096 --power 100
```

Results:

- `ctx_08192`
  - accepted delta: `-0.10312822275696674%`
  - committed delta: `-0.11240501775998872%`
- `ctx_16384`
  - accepted delta: `-0.06794137629816227%`
  - committed delta: `-0.00910622410418038%`

Two-context mean:

- accepted delta: `-0.08553479952756451%`
- committed delta: `-0.06075562093208455%`

## Integrity check

The main tooling risk in this probe was:

- did `--expert-select` copy the intended expert slices correctly?

That was checked directly against the generated GGUF payload:

- selected experts in `mtp.2.ffn_gate_exps.weight` matched donor bytes
- selected experts in `mtp.2.ffn_up_exps.weight` matched donor bytes
- non-selected experts in both tensors matched baseline bytes
- no mismatched expert slices were found across all `256` experts

So this negative result is not explained by an obvious slice-addressing bug in
the new splicing mode.

## Read

This probe gives a stronger negative result than the earlier sparse-block test.

What it shows:

- recoverable-step activity ranking is not enough, by itself, to recover the
  `layer2gateup` tensor-part gain through a small top-expert splice

What it does not yet prove:

- that no routed-local expert selection can work at all

But it is enough to change branch priority:

- routed-local expert-slice search should stop being the lead lane
- the next search budget should move to a different family unless a very narrow
  control run is needed for closure
