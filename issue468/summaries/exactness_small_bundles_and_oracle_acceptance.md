# Exactness small bundles and oracle acceptance

Date: 2026-07-05.

## Summary

A fresh bundle set was captured from the retained small exactness corpus for three target regimes:

- `temp=0.0`
- `temp=0.5`
- `temp=1.0`

For each prompt/temperature cell, the retained bundle includes:

- prompt copy
- target top-k/logprob dump
- retained selected target token stream
- hidden-state capture bundle for target layers 40/41/42
- compact packed captures: `captures/captures.npz`
- derived oracle inputs `main_hidden_pos*.npy`
- compact packed oracle inputs: `oracle/oracle_inputs.npz`
- oracle acceptance summary
- for `temp=0`, `oracle_ref.npz` validating the drafter forward on the first retained pair

Bundle root:
- `issue468/artifacts/exactness_small_bundles/`

Aggregate summaries:
- `issue468/artifacts/exactness_small_bundles/summary.json`
- `issue468/artifacts/exactness_small_bundles/summary.csv`

## Regime handling

### `temp=0`
Reference mode:
- greedy

Retained files include:
- `target_greedy.json`
- `target_selected_tokens.json`
- `oracle/oracle_ref.npz`

This is suitable for deterministic exactness-style block validation.

### `temp=0.5` and `temp=1.0`
Reference mode:
- sampled-stream

Retained files include:
- `target_selected_tokens.json`
- `target_topk.json`
- derived `main_hidden_pos*.npy`
- oracle acceptance summaries

For these stochastic regimes, the retained comparison target is the sampled target continuation from the same seeded target run, rather than a greedy-only reference.

## Oracle acceptance measurement

The retained oracle was run against `dspark.gguf` on every prompt bundle.

Metric recorded:
- average accepted prefix length over the 5-token drafted block
- frequency histogram for accepted prefix lengths `0..5`

## Aggregate results by temperature

### `temp=0.0`
- average prefix across 10 prompts: **2.337**
- aggregate prefix histogram:
  - `0`: 15
  - `1`: 13
  - `2`: 18
  - `3`: 11
  - `4`: 10
  - `5`: 13

### `temp=0.5`
- average prefix across 10 prompts: **2.087**
- aggregate prefix histogram:
  - `0`: 20
  - `1`: 15
  - `2`: 17
  - `3`: 7
  - `4`: 8
  - `5`: 13

### `temp=1.0`
- average prefix across 10 prompts: **2.087**
- aggregate prefix histogram:
  - `0`: 18
  - `1`: 20
  - `2`: 14
  - `3`: 7
  - `4`: 7
  - `5`: 14

## Interpretation

- The retained oracle shows higher average draft-prefix agreement in the deterministic regime than in the stochastic sampled-stream regimes.
- `temp=0.5` and `temp=1.0` both reduce average accepted prefix versus greedy on this corpus.
- The stochastic bundle format is therefore useful for validating behavior under sampled target trajectories, while `temp=0` remains the strongest mode for strict exactness-style comparison.

## Reproducibility

Recreate the local oracle environment:

```sh
cd issue468/dspark_oracle
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
cd ../..
```

Rebuild `ds4` if needed:

```sh
make ds4
```

Capture the full retained bundle set and recompute oracle acceptance summaries:

```sh
. issue468/dspark_oracle/.venv/bin/activate
python issue468/run_exactness_small_bundles.py \
  --ctx 4096 \
  --tokens 14 \
  --temps 0.0 0.5 1.0 \
  --seed 2
```

The script reuses existing compact bundle outputs when they are already present.
Add `--force-recapture` to rebuild target-side captures.

Outputs:

- `issue468/artifacts/exactness_small_bundles/summary.json`
- `issue468/artifacts/exactness_small_bundles/summary.csv`
- one retained bundle per prompt/temperature cell under
  `issue468/artifacts/exactness_small_bundles/`

Re-run the main bundle script against a different DSpark GGUF, reusing the
already-captured target bundles:

```sh
. issue468/dspark_oracle/.venv/bin/activate
python issue468/run_exactness_small_bundles.py \
  --ctx 4096 \
  --tokens 14 \
  --temps 0.0 0.5 1.0 \
  --seed 2 \
  --dspark /path/to/candidate-dspark.gguf
```

This reuses captures by default unless `--force-recapture` is added, but it
updates the bundle-local oracle outputs in place.

Re-run oracle acceptance only, against a different DSpark GGUF but the same
already-captured bundles, while preserving the canonical bundle-local outputs:

```sh
. issue468/dspark_oracle/.venv/bin/activate
python issue468/run_exactness_small_acceptance_from_bundles.py \
  --bundles-dir issue468/artifacts/exactness_small_bundles \
  --model /Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark /path/to/candidate-dspark.gguf \
  --label candidate_name
```

Outputs for the candidate rerun:

- `issue468/artifacts/exactness_small_acceptance/candidate_name/summary.json`
- `issue468/artifacts/exactness_small_acceptance/candidate_name/summary.csv`
- one JSON result per retained bundle under the same directory

## Which rerun path to use

- Use `run_exactness_small_bundles.py` when you want to refresh the canonical
  bundle-local oracle results, optionally with `--force-recapture` for fresh
  target-side capture.
- Use `run_exactness_small_acceptance_from_bundles.py` when you want to compare
  another DSpark GGUF against the same retained bundles **without overwriting**
  the canonical bundle-local oracle outputs.

## Notes

The target capture path in this branch currently uses retained layer-wise `hc_ffn_post` dumps for layers 40/41/42, from which `main_hidden_pos*.npy` is derived offline. This preserves the oracle input reconstruction path needed by the retained numpy oracle.
