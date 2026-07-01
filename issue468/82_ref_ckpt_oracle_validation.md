# Reference-Checkpoint Oracle Validation

Date: 2026-07-01

## Purpose

Record the first successful source-checkpoint import into the existing numpy
oracle path, after the torch-fallback ceiling route was closed out in `81`.

This is the first branch result that looks suitable to carry the FP8/source
ceiling work forward.

## New implementation

Added:

- `issue468/dspark_oracle/ref_ckpt_loader.py`
- `issue468/dspark_oracle/ref_expert_store.py`
- `issue468/ref/measure_ref_oracle_b2.py`

What changed:

1. dense DSpark weights now load directly from
   `~/ds4/ref-ckpt/model0-mp1.safetensors`
2. dense FP8 weights are dequantized by expanding their `128 x 128` block
   scales
3. routed expert FP4 weights are dequantized lazily per expert using their
   `1 x 32` scales
4. converted reference-model tensor names are mapped onto the older oracle
   names expected by `issue468/dspark_oracle/forward.py`
5. the old B2-style low-fill scoring path is reused rather than inventing a
   new metric

This keeps the working acceptance/oracle methodology while changing the weight
source from:

- `dspark.gguf`

to:

- `ref-ckpt/model0-mp1.safetensors`

## Validation 1: low-fill legacy bundle

Command run on DGX:

```sh
source ~/dref-venv/bin/activate
cd ~/ds4
python issue468/ref/measure_ref_oracle_b2.py \
  --out-json /tmp/ref-oracle-fp8-lowfill.b2.json
```

Result:

- `average_committed = 3.2125822368421053`
- `average_accepted = 2.314967105263158`
- `greedy_avg_prefix = 2.8421052631578947`
- `analytical_committed_upper_bound = 3.4705801700131698`

Interpretation:

- this is materially above the torch-fallback low-fill result from `81`
  (`2.5387`)
- it is also above the older documented low-fill oracle/F32 reference of about
  `2.80`

So the source-import oracle path passes the first credibility check that the
torch-fallback path failed.

## Validation 2: first matched sweep-root context

Command run on DGX:

```sh
source ~/dref-venv/bin/activate
cd ~/ds4
python issue468/ref/measure_ref_oracle_b2.py \
  --capture-dir /tmp/dspark_sweep8/ctx_08192 \
  --target-json /tmp/dspark_sweep8/ctx_08192/target_topk.json \
  --greedy-json /tmp/dspark_sweep8/ctx_08192/target_greedy.json \
  --pos0 8192 \
  --steps-cap 19 \
  --out-json /tmp/ref-oracle-fp8-ctx08192-19.b2.json
```

Result:

- `average_committed = 4.425986842105263`
- `average_accepted = 3.973684210526316`
- `greedy_avg_prefix = 4.105263157894737`
- `analytical_committed_upper_bound = 4.974497792489657`

Existing matched baseline on the same bundle:

- `average_committed = 4.473684210526316`
- `average_accepted = 4.072368421052632`

Delta vs baseline:

- committed: about `-1.066%`

Interpretation:

- unlike the torch-fallback branch, this result is in the same ballpark as the
  current sweep-root baseline
- it does **not** yet establish a strong positive FP8 ceiling at `ctx_08192`
- but it does establish that the new source-import oracle path is plausible
  enough to continue

## Meaning for the branch

This new path changes the status of the FP8/source ceiling branch.

Before:

- torch-fallback execution existed
- but scoring was not trustworthy

Now:

- a source-checkpoint import exists inside the validated numpy/oracle family
- it passes low-fill validation
- it produces a believable matched-horizon sweep-root result

So the FP8/source ceiling branch is active again, now on the oracle path rather
than the torch-fallback path.

## Immediate next step

Use this new oracle/source path to score the remaining current sweep-root
contexts:

- `ctx_16384`
- `ctx_24576`
- `ctx_32768`

Then compare:

1. mean source-reference committed score
2. mean current baseline committed score
3. gap recovered vs the old `Q4_K`-specific oracle ceiling branch

That is now the correct next measurement.
