# First Frontier Artifact Builds

Date: 2026-07-02

## Purpose

Record the first successful materialization of the routed-expert `Q2/Q4`
frontier endpoints defined in `96`.

This note now covers:

- buildability
- exact artifact size
- first 4-context acceptance measurements
- disk-space state

## Result

Two endpoint candidates were built successfully into:

- `/private/tmp/dspark_pareto_q2q4`

Artifacts:

| label | bytes | GiB |
|---|---:|---:|
| `all_q2` | `6054122176` | `5.638335` |
| `only_mtp2_q4` | `7866061504` | `7.325835` |

These exactly match the size estimates from `95` and `96` to the byte-level
`approx_file_bytes` reported by the quantizer dry-run / build path.

## Build-path correction

The first build attempt exposed a small helper bug:

- the planner emitted overrides as `--tensor-type=NAME=TYPE`

but `deepseek4-quantize` expects:

- `--tensor-type NAME=TYPE`

The helper surface is now split correctly:

- `plan_q2_q4_frontier_candidates.py` emits bare `NAME=TYPE`
- `build_q2_q4_frontier_candidates.py` expands each one into a separate
  `--tensor-type` argv pair

After that correction, both endpoint artifacts built cleanly without template
or metadata issues.

## Why these two points first

These are the most informative first endpoints:

1. `all_q2`
2. `only_mtp2_q4`

They answer the first practical question of the new branch:

- what is the full routed-`Q2` size floor?
- what is the cost of keeping only the final drafter layer at baseline routed
  precision?

The memory frontier therefore has two concrete reduced-size anchor points below
the `Q4_K` baseline.

## 4-context measurement result

Both artifacts were scored on the existing 4-context sweep root:

- `/private/tmp/dspark_sweep8`

using:

- `issue468/run_dspark_weighted_from_sweep_root.py`
- existing bundle contexts `8192`, `16384`, `24576`, `32768`
- `19` draft steps
- `128` B2 trials

Mean results:

| label | size GiB | mean accepted | accepted delta vs baseline | mean committed | committed delta vs baseline |
|---|---:|---:|---:|---:|---:|
| baseline `Q4_K` | `10.700835` | `4.229749` | `0.000%` | `4.557052` | `0.000%` |
| `only_mtp2_q4` | `7.325835` | `4.155633` | `-1.752%` | `4.495785` | `-1.344%` |
| `all_q2` | `5.638335` | `4.123869` | `-2.503%` | `4.476151` | `-1.775%` |

Per-context accepted-token deltas vs baseline:

| context | `only_mtp2_q4` | `all_q2` |
|---|---:|---:|
| `8192` | `-3.175%` | `-3.234%` |
| `16384` | `-1.031%` | `-2.033%` |
| `24576` | `-1.713%` | `-2.453%` |
| `32768` | `-1.113%` | `-2.304%` |

## Interpretation

The first reduced-size frontier is now visible.

`all_q2` establishes the current size floor:

- about `47.3%` smaller than baseline
- with about `2.50%` lower mean accepted tokens

`only_mtp2_q4` is the first plausible knee:

- about `31.5%` smaller than baseline
- with only about `1.75%` lower mean accepted tokens

That point does **not** dominate baseline, but it does dominate `all_q2` in
quality at a still-material size reduction. The gain over `all_q2` is modest:

- `+0.031764` mean accepted
- for `+1.687500 GiB`

So the current evidence supports the branch hypothesis that:

- the interesting frontier lies in mixed routed `Q2/Q4` recipes, not pure
  routed `Q2`

and specifically suggests that:

- keeping `mtp.2` at `Q4_K` is directionally useful
- but the quality recovery per added GiB is not yet strong enough to assume the
  rest of the mixed grid will be compelling

## Disk-space state

Current local state after build:

- `/private/tmp/dspark_pareto_q2q4`: about `13 GiB`
- `/private/tmp/dspark_sweep8`: about `6.6 GiB`
- free disk: about `356 GiB`

Transient reprobe artifacts for the two new 4-context runs were pruned after
summary extraction:

- deleted generated `metal_base_logits_19steps.bin`
- deleted generated `*.probe.stdout` / `*.probe.stderr`
- retained per-context `*.b2.json` plus top-level summary JSON/TSV

No further cleanup was applied here because:

- the two new frontier artifacts are active research outputs
- the remaining large files in `/private/tmp/dspark_sweep2ctx` are the three
  intentionally retained reference GGUFs plus the imatrix path used by this
  sweep

So disk pressure is currently acceptable, but future frontier expansion should
continue pruning superseded GGUFs as soon as they are no longer needed.

## Immediate next step

The next branch action should be more selective than the original full-grid
plan:

- use these two measured points to decide whether the remaining mixed layer
  recipes are likely worth their build cost
- if more builds are justified, prioritize the other one-`Q4` and two-`Q4`
  mixes before any finer tensor-family search
