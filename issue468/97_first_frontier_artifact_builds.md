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

The first frontier candidates were built successfully into:

- `/private/tmp/dspark_pareto_q2q4`

Artifacts built so far:

| label | bytes | GiB |
|---|---:|---:|
| `all_q2` | `6054122176` | `5.638335` |
| `only_mtp0_q4` | `7866061504` | `7.325835` |
| `only_mtp1_q4` | `7866061504` | `7.325835` |
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

Measured artifacts were scored on the existing 4-context sweep root:

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
| `only_mtp0_q4` | `7.325835` | `4.185958` | `-1.035%` | `4.522718` | `-0.753%` |
| `only_mtp2_q4` | `7.325835` | `4.155633` | `-1.752%` | `4.495785` | `-1.344%` |
| `all_q2` | `5.638335` | `4.123869` | `-2.503%` | `4.476151` | `-1.775%` |
| `only_mtp1_q4` | `7.325835` | `4.105674` | `-2.933%` | `4.468750` | `-1.938%` |

Per-context accepted-token deltas vs baseline:

| context | `only_mtp0_q4` | `only_mtp2_q4` | `only_mtp1_q4` | `all_q2` |
|---|---:|---:|---:|---:|
| `8192` | `-0.167%` | `-3.175%` | `-3.785%` | `-3.234%` |
| `16384` | `-1.138%` | `-1.031%` | `-2.091%` | `-2.033%` |
| `24576` | `-1.148%` | `-1.713%` | `-3.474%` | `-2.453%` |
| `32768` | `-1.670%` | `-1.113%` | `-2.400%` | `-2.304%` |

## Interpretation

The first reduced-size frontier is now visible.

The full single-layer-`Q4` tier at `7.325835 GiB` is now resolved:

1. `only_mtp0_q4`
2. `only_mtp2_q4`
3. `only_mtp1_q4`

`all_q2` establishes the current size floor:

- about `47.3%` smaller than baseline
- with about `2.50%` lower mean accepted tokens

`only_mtp0_q4` is currently the best measured reduced-size knee:

- about `31.5%` smaller than baseline
- with only about `1.04%` lower mean accepted tokens

That point does **not** dominate baseline, but it clearly dominates `all_q2`
in quality at a still-material size reduction. The gain over `all_q2` is:

- `+0.062089` mean accepted
- for `+1.687500 GiB`

`only_mtp2_q4` remains better than `all_q2`, but it is now clearly inferior to
`only_mtp0_q4` at the same byte cost:

- `-0.030325` mean accepted vs `only_mtp0_q4`

`only_mtp1_q4` is the weakest same-size variant and does not justify further
attention as a deployment point:

- `-0.080284` mean accepted vs `only_mtp0_q4`
- `-0.049959` mean accepted vs `only_mtp2_q4`
- even `-0.018195` mean accepted vs `all_q2` despite costing `+1.687500 GiB`

So the current evidence still supports the branch hypothesis that:

- the interesting frontier lies in mixed routed `Q2/Q4` recipes, not pure
  routed `Q2`

but it no longer supports the earlier assumption that `mtp.2` is the most
valuable single routed layer to keep at `Q4_K`.

Instead, the first same-size contrast suggests:

- layer identity matters materially even when total bytes are identical
- `mtp.0` is the strongest measured single-layer `Q4` keep on this metric
- `mtp.1` is a poor use of the `Q4` budget at this size
- `mtp.2` is helpful, but not the best single-layer keep
- the next unresolved frontier question is whether any `9.01 GiB` two-layer
  `Q4` recipe closes enough of the remaining `only_mtp0_q4 -> baseline` gap to
  justify its extra `1.687500 GiB`

## Disk-space state

Current local state after build:

- `/private/tmp/dspark_pareto_q2q4`: about `28 GiB`
- `/private/tmp/dspark_sweep8`: about `6.6 GiB`
- free disk: about `342 GiB`

Transient reprobe artifacts for the measured 4-context runs were pruned after
summary extraction:

- deleted generated `metal_base_logits_19steps.bin`
- deleted generated `*.probe.stdout` / `*.probe.stderr`
- retained per-context `*.b2.json` plus top-level summary JSON/TSV

No further cleanup was applied here because:

- the four frontier artifacts are active research outputs
- the remaining large files in `/private/tmp/dspark_sweep2ctx` are the three
  intentionally retained reference GGUFs plus the imatrix path used by this
  sweep

So disk pressure is currently acceptable, but future frontier expansion should
continue pruning superseded GGUFs as soon as they are no longer needed.

## Immediate next step

The next branch action should be more selective than the original full-grid
plan:

- decide whether the measured gain from `only_mtp0_q4` over the rest of the
  `7.33 GiB` tier is strong enough to justify spending on the `9.01 GiB`
  two-layer-`Q4` tier
- if more builds are justified, prioritize `mtp01_q4` before the other
  `9.01 GiB` variants because `mtp.0` is clearly worth keeping and `mtp.1`
  looks weak in isolation
