# Numpy-oracle ceiling on the 4-context sweep root

Date: 2026-07-01

## Purpose

After `49_qdump_oracle_envelope_falsification.md`, the remaining question was:

- how much headroom exists above the current Metal `Q4_K` baseline if the
  runtime matched the old numpy oracle numerically?

This does **not** measure the full source-precision / non-quantized DSpark
ceiling. It measures a narrower but still useful ceiling:

- the current `dspark.gguf` routed-expert weights
- dequantized to `float32`
- run through the validated numpy oracle forward
- scored with the old B2 estimator on the current 4-context sweep-root bundles

So this isolates the headroom above the live Metal baseline that is attributable
to the runtime/forward path vs the current `Q4_K` checkpoint, while explicitly
**not** claiming to measure the HF source / FP8 ceiling.

## Reused tool

Reused:

- `issue468/baseline/dspark_capture/measure_b2_acceptance.py`

against the current sweep-root bundles:

- `/tmp/dspark_sweep8/ctx_08192`
- `/tmp/dspark_sweep8/ctx_16384`
- `/tmp/dspark_sweep8/ctx_24576`
- `/tmp/dspark_sweep8/ctx_32768`

Each run consumes:

- `hc_dspark_main_hc-{40,41,42}_pos*.bin`
- `target_topk.json`
- `target_greedy.json`

and reports two useful ceilings:

1. **Monte Carlo B2** on the F32 oracle
2. **Greedy-conditioned `1 - TV` analytical upper bound**

## Commands

```sh
issue468/.venv/bin/python issue468/baseline/dspark_capture/measure_b2_acceptance.py \
  --cap-dir /tmp/dspark_sweep8/ctx_08192 \
  --target-json /tmp/dspark_sweep8/ctx_08192/target_topk.json \
  --greedy-json /tmp/dspark_sweep8/ctx_08192/target_greedy.json \
  --pos0 8192 \
  --label ctx_08192

issue468/.venv/bin/python issue468/baseline/dspark_capture/measure_b2_acceptance.py \
  --cap-dir /tmp/dspark_sweep8/ctx_16384 \
  --target-json /tmp/dspark_sweep8/ctx_16384/target_topk.json \
  --greedy-json /tmp/dspark_sweep8/ctx_16384/target_greedy.json \
  --pos0 16384 \
  --label ctx_16384

issue468/.venv/bin/python issue468/baseline/dspark_capture/measure_b2_acceptance.py \
  --cap-dir /tmp/dspark_sweep8/ctx_24576 \
  --target-json /tmp/dspark_sweep8/ctx_24576/target_topk.json \
  --greedy-json /tmp/dspark_sweep8/ctx_24576/target_greedy.json \
  --pos0 24576 \
  --label ctx_24576

issue468/.venv/bin/python issue468/baseline/dspark_capture/measure_b2_acceptance.py \
  --cap-dir /tmp/dspark_sweep8/ctx_32768 \
  --target-json /tmp/dspark_sweep8/ctx_32768/target_topk.json \
  --greedy-json /tmp/dspark_sweep8/ctx_32768/target_greedy.json \
  --pos0 32768 \
  --label ctx_32768
```

## Results

Metal baseline here means the existing `baseline-weighted4ctx_19t_default_256tr`
accepted score already measured on the same four contexts.

### A. F32 oracle Monte Carlo B2 ceiling

| context | Metal baseline accepted | F32 oracle MC B2 | delta vs baseline |
| --- | ---: | ---: | ---: |
| 8192  | 4.0997 | 4.491 | +9.54% |
| 16384 | 4.2136 | 4.543 | +7.82% |
| 24576 | 4.2315 | 4.529 | +7.03% |
| 32768 | 4.2572 | 4.627 | +8.69% |

Mean delta across the four contexts:

- `+8.27%`

### B. F32 oracle analytical upper bound (`1 - TV`, greedy-conditioned)

| context | Metal baseline accepted | F32 oracle analytical upper bound | delta vs baseline |
| --- | ---: | ---: | ---: |
| 8192  | 4.0997 | 5.060 | +23.42% |
| 16384 | 4.2136 | 5.143 | +22.06% |
| 24576 | 4.2315 | 5.147 | +21.64% |
| 32768 | 4.2572 | 5.264 | +23.65% |

Mean delta across the four contexts:

- `+22.69%`

## Interpretation

This sharply changes the meaning of the earlier falsification result.

`49_qdump_oracle_envelope_falsification.md` showed that the **current imatrix
candidate family** only contains:

- `+1.0051%` mean accepted headroom

But the reused numpy oracle now shows that the **same current `Q4_K` checkpoint,
run through the old F32 oracle path**, has much more headroom above the live
Metal baseline:

- about `+8.27%` on Monte Carlo B2
- about `+22.69%` on the analytical greedy-conditioned upper bound

So the negative result from note 49 should be read narrowly:

- it falsifies the **current imatrix candidate family**
- it does **not** falsify the broader hypothesis that substantially more
  acceptance exists above the current Metal baseline

In other words:

- current local imatrix tuning looks exhausted
- but the broader quantization / runtime ceiling is not low

## Important caveat

This note does **not** prove the full non-quantized or HF-source ceiling.

The reused numpy oracle path here still uses:

- the existing `dspark.gguf` `Q4_K` routed-expert weights
- dequantized to F32

So this note isolates:

- **current Q4_K checkpoint + idealized F32 oracle forward**

not:

- source FP8 / Q8 / BF16 / unquantized DSpark

That stronger ceiling remains unmeasured in the current tree and would require a
different harness, likely based on the older reference-model / HF-source path.
