# First Frontier Artifact Builds

Date: 2026-07-02

## Purpose

Record the first successful materialization of the routed-expert `Q2/Q4`
frontier endpoints defined in `96`.

This note is intentionally about:

- buildability
- exact artifact size
- disk-space state

It does **not** yet claim anything about acceptance quality. That requires the
next measurement pass.

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

So even before quality measurement, the memory frontier now has two concrete
anchor points below the `Q4_K` baseline.

## Disk-space state

Current local state after build:

- `/private/tmp/dspark_pareto_q2q4`: about `13 GiB`
- free disk: about `368 GiB`

No additional cleanup was applied here because:

- the two new frontier artifacts are active research outputs
- the remaining large files in `/private/tmp/dspark_sweep2ctx` are the three
  intentionally retained reference GGUFs plus the imatrix path used by this
  sweep

So disk pressure is currently acceptable, but future frontier expansion should
continue pruning superseded GGUFs as soon as they are no longer needed.

## Immediate next step

The next branch action is straightforward:

- run the standard 4-context acceptance sweep on these two artifacts
- compare them against baseline `Q4_K`
- decide whether the rest of the first-pass grid is worth building before a
  finer local search
