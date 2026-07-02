# Raw-HF DSpark Oracle Ceiling Reference

Date: 2026-07-02

## Purpose

Measure the DSpark acceptance ceiling directly from the original Hugging Face
MTP shard files, without passing through the local `ref-ckpt` conversion path,
and record the result as the reference ceiling for future branch comparisons.

This closes the ambiguity left in `82` and `92`:

- whether the oracle was really scoring the original DSpark tensors
- whether conversion into `ref-ckpt/model0-mp1.safetensors` could be depressing
  the measured ceiling

## New loader path

Added:

- `issue468/dspark_oracle/raw_hf_ckpt_loader.py`
- `issue468/dspark_oracle/raw_hf_expert_store.py`

Updated:

- `issue468/ref/measure_ref_oracle_b2.py`

New scorer option:

- `--source raw_hf`

What this path does:

1. read tensor bytes directly from `~/ds4/hf-dspark/*.safetensors`
2. parse safetensors headers without external `safetensors` Python bindings
3. dequantize source `F8_E4M3 + F8_E8M0` dense tensors using the same local
   FP8 math as the C quantizer
4. dequantize packed expert `I8 + F8_E8M0` tensors using the same local FP4
   table as the C quantizer
5. keep the scorer, capture bundles, and B2 procedure unchanged

Shared `embed` and `lm_head` are still read from the target GGUF, so this
reference isolates the DSpark MTP tensor path rather than changing unrelated
shared model pieces.

## Commands

Measured on the existing 4-context sweep-root captures:

```sh
issue468/.venv/bin/python issue468/ref/measure_ref_oracle_b2.py \
  --source raw_hf \
  --hf-dspark ../ds4/hf-dspark \
  --capture-dir /tmp/dspark_sweep8/ctx_08192 \
  --target-json /tmp/dspark_sweep8/ctx_08192/target_topk.json \
  --greedy-json /tmp/dspark_sweep8/ctx_08192/target_greedy.json \
  --pos0 8192 \
  --steps-cap 19 \
  --trials 128 \
  --label raw-hf-oracle \
  --out-json /tmp/raw-hf-oracle-ctx08192-19.b2.json
```

Repeated for:

- `ctx_16384`
- `ctx_24576`
- `ctx_32768`

## Results

### Raw-HF oracle accepted tokens

| context | raw-HF oracle accepted | baseline `dspark.gguf` accepted | delta vs baseline |
|---|---:|---:|---:|
| `8192` | `3.9745065789473686` | `4.072368421052632` | `-2.4030694668820685%` |
| `16384` | `4.2162828947368425` | `4.2368421052631575` | `-0.48524844720495564%` |
| `24576` | `4.087171052631579` | `4.2105263157894735` | `-2.9296875%` |
| `32768` | `4.292763157894737` | `4.2631578947368425` | `+0.694444444444442%` |

4-context mean:

- raw-HF oracle accepted: `4.142680921052632`

### Raw-HF oracle committed tokens

| context | raw-HF oracle committed |
|---|---:|
| `8192` | `4.425986842105263` |
| `16384` | `4.503700657894737` |
| `24576` | `4.503700657894737` |
| `32768` | `4.569078947368421` |

## Comparison against the older converted-path oracle

The overlap contexts from `82` / `87` were:

| context | converted ref-ckpt oracle | direct raw-HF oracle | delta |
|---|---:|---:|---:|
| `8192` | `3.973684210526316` | `3.9745065789473686` | `+0.0008223684210526558` |
| `16384` | `4.217105263157895` | `4.2162828947368425` | `-0.0008223684210526558` |

Read:

- the difference is tiny
- it is far smaller than the baseline-vs-oracle gap
- it is consistent with ordinary Monte Carlo noise at this trial count

So the direct raw-HF measurement does **not** support the hypothesis that
`ref-ckpt` conversion loss was the main reason the prior source-import oracle
sat below baseline.

## Reference point to use going forward

For future branch comparisons, the correct DSpark tensor ceiling reference is:

- `raw-hf-oracle`

Meaning:

- original DSpark HF MTP tensors
- direct local header parsing
- local FP8/FP4 dequantization on load
- unchanged numpy oracle and B2 scorer

This is now the cleanest available acceptance reference for the original DSpark
tensor path.

## Decision impact

This result changes the interpretation of the ceiling branch in an important
way.

What is now ruled out:

- "the oracle is low mainly because `ref-ckpt` conversion lost too much
  precision"

What remains true:

- the original DSpark tensor path, scored directly, is still roughly at or
  below the shipped baseline on most measured contexts
- any large future positive claim should therefore come from a different
  quantization/search mechanism, not from relabeling the prior source-import
  path as a bad conversion artifact

So the ceiling reference is now stronger, but the conclusion is harsher:

- direct raw-HF DSpark tensors do **not** currently provide a broad acceptance
  ceiling above the shipped `dspark.gguf` baseline.
