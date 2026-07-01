# Weighted sweep-root helper and 3-context reprobe

Date: 2026-07-01

## What changed

Added `issue468/run_dspark_weighted_from_sweep_root.py` to automate the
post-sweep evaluation loop for an existing `ctx_*` sweep root:

1. collect one weighted imatrix from the sweep root
2. quantize one DSpark candidate from that imatrix
3. direct-reprobe each bundle against baseline and candidate
4. write JSON/TSV summary outputs beside the sweep root

The first run exposed a real bug in the helper: it resolved
`--measure-python` through `Path(...).resolve()`, which collapsed the venv
interpreter symlink to the system Python binary. That caused the reprobe
measurement phase to execute outside the intended virtualenv. The helper now
passes `--measure-python` through unchanged.

## 3-context sweep-root reprobe

Used the existing partial sweep root at `/tmp/dspark_sweep8`, which currently
contains completed bundles for:

- `ctx_08192`
- `ctx_16384`
- `ctx_24576`

Command:

```sh
python3 issue468/run_dspark_weighted_from_sweep_root.py \
  --sweep-root /tmp/dspark_sweep8 \
  --collector-max-tokens 3 \
  --trials 32 \
  --run-label weighted3ctxb
```

Artifacts:

- imatrix: `/tmp/dspark_sweep8/weighted3ctxb.imatrix.dat`
- candidate: `/tmp/dspark_sweep8/weighted3ctxb.gguf`
- summary JSON: `/tmp/dspark_sweep8/weighted3ctxb.summary.json`
- summary TSV: `/tmp/dspark_sweep8/weighted3ctxb.summary.tsv`

Results:

| context | baseline accepted | candidate accepted | accepted delta | baseline committed | candidate committed | committed delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 4.1382 | 4.1447 | +0.159% | 4.5263 | 4.5313 | +0.109% |
| 16384 | 4.2188 | 4.2122 | -0.156% | 4.5000 | 4.4934 | -0.146% |
| 24576 | 4.2484 | 4.2368 | -0.271% | 4.6201 | 4.6036 | -0.356% |

Mean delta across the three completed contexts:

- accepted: `-0.0893%`
- committed: `-0.1311%`

## Interpretation

This is the first direct reprobe of the new sweep-root collector on more than
two contexts. Mechanically the path is now working end to end, but the result
is not a win:

- the earlier 2-context experiment (`ctx_08192` + `ctx_16384`) showed a
  positive acceptance signal
- once `ctx_24576` is included, the aggregate signal becomes effectively flat
  to slightly negative

So the current weighting recipe and current sample size do not yet establish a
robust improvement. The next useful step is not more collector plumbing; it is
to extend the sweep root to more contexts and rerun the same helper so the
result is driven by broader evidence rather than a small-context subset.
