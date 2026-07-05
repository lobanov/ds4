# Exactness Small Bundle Inventory

## Purpose

This inventory tracks the retained exactness/debug bundles built from:

- `issue468/prompts/exactness_small_corpus/`

The bundles are intended to support:

- deterministic `temp=0` exactness validation
- sampled-stream `temp=0.5` and `temp=1.0` comparisons
- retained oracle-input reconstruction
- oracle acceptance-prefix measurement

## Location

- bundle root: `issue468/artifacts/exactness_small_bundles/`
- summaries: `summary.json`, `summary.csv`

## Bundle contents per prompt/regime

Each prompt/temperature bundle may include:

- `prompt.txt`
- `target_topk.json`
- `target_selected_tokens.json`
- `target_greedy.json` for `temp=0`
- `captures/` hidden-state dumps for layers 40/41/42
- `captures/captures.npz` — compact packed layer captures
- `oracle/main_hidden_pos*.npy`
- `oracle/oracle_inputs.npz` — compact packed oracle inputs
- `oracle/main_hidden_manifest.json`
- `oracle/acceptance_summary.json`
- `oracle/oracle_ref.npz` for `temp=0`

## Reference semantics

- `temp=0`: greedy reference
- `temp>0`: sampled-stream reference from the same seeded target run

## Reproduction

Minimal reproduction flow:

```sh
cd issue468/dspark_oracle
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
cd ../..
make ds4
. issue468/dspark_oracle/.venv/bin/activate
python issue468/run_exactness_small_bundles.py \
  --ctx 4096 \
  --tokens 14 \
  --temps 0.0 0.5 1.0 \
  --seed 2
```

The script will reuse existing compact bundle outputs when present. Add
`--force-recapture` to rebuild target-side captures.

To refresh the canonical bundle-local oracle results while reusing existing captures:

```sh
. issue468/dspark_oracle/.venv/bin/activate
python issue468/run_exactness_small_bundles.py \
  --ctx 4096 \
  --tokens 14 \
  --temps 0.0 0.5 1.0 \
  --seed 2 \
  --dspark /path/to/candidate-dspark.gguf
```

This reuses existing compact outputs by default. Add `--force-recapture` only if
you want to rebuild the target-side captures.

To re-run oracle acceptance against a different DSpark GGUF without re-capturing
and without overwriting the canonical bundle-local oracle outputs:

```sh
. issue468/dspark_oracle/.venv/bin/activate
python issue468/run_exactness_small_acceptance_from_bundles.py \
  --bundles-dir issue468/artifacts/exactness_small_bundles \
  --model /Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark /path/to/candidate-dspark.gguf \
  --label candidate_name
```

Use the second path when you want clean side-by-side candidate results retained
separately from the canonical bundle-local outputs.

## Retention notes

These bundles are intentionally compact compared with the historical issue468 captures.
They retain the minimum inputs and summaries needed to re-run oracle-side acceptance measurement across deterministic and stochastic regimes.
