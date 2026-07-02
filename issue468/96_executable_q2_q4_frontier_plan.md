# Executable First-Pass `Q2/Q4` Frontier Plan

Date: 2026-07-02

## Purpose

Convert the size/quality Pareto framing from `94` and the candidate grid from
`95` into a directly executable build plan for the first mixed routed
`Q2/Q4` sweep.

The main question answered here is:

- can the first-pass frontier candidates be built directly from the existing
  `dspark.gguf` template, or do they require a new template path?

## Result

They can be built directly from the existing shipped drafter template:

- `../ds4/gguf/dspark.gguf`

using `gguf-tools/deepseek4-quantize` with routed expert `--tensor-type`
overrides.

So the first frontier sweep does **not** need:

- a new DSpark template format
- a second GGUF metadata path
- an expert-splicing donor GGUF

It only needs:

1. the existing `hf-dspark` source
2. the existing DSpark imatrix file
3. explicit routed expert type overrides per candidate

## New helper

Added:

- `issue468/plan_q2_q4_frontier_candidates.py`

Purpose:

- emit the exact routed `--tensor-type` overrides for each first-pass frontier
  candidate
- print the estimated total GiB for each candidate

Example:

```sh
python3 issue468/plan_q2_q4_frontier_candidates.py \
  --baseline-gguf ../ds4/gguf/dspark.gguf
```

This prints the eight planned first-pass candidates:

1. `all_q4`
2. `all_q2`
3. `only_mtp2_q4`
4. `only_mtp1_q4`
5. `only_mtp0_q4`
6. `mtp01_q4`
7. `mtp02_q4`
8. `mtp12_q4`

## Validation

Validated the all-routed-`Q2` endpoint with a dry run:

```sh
gguf-tools/deepseek4-quantize \
  --hf ../ds4/hf-dspark \
  --template ../ds4/gguf/dspark.gguf \
  --out /tmp/dspark-all-q2-frontier.gguf \
  --overwrite \
  --imatrix /private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx.imatrix.dat \
  --tensor-type mtp.0.ffn_gate_exps.weight=iq2_xxs \
  --tensor-type mtp.0.ffn_up_exps.weight=iq2_xxs \
  --tensor-type mtp.0.ffn_down_exps.weight=q2_k \
  --tensor-type mtp.1.ffn_gate_exps.weight=iq2_xxs \
  --tensor-type mtp.1.ffn_up_exps.weight=iq2_xxs \
  --tensor-type mtp.1.ffn_down_exps.weight=q2_k \
  --tensor-type mtp.2.ffn_gate_exps.weight=iq2_xxs \
  --tensor-type mtp.2.ffn_up_exps.weight=iq2_xxs \
  --tensor-type mtp.2.ffn_down_exps.weight=q2_k \
  --dry-run
```

Dry-run output confirmed:

- all `9` routed expert tensors changed type as intended
- no metadata/template mismatch
- `type_changes: 9`
- `approx_file_bytes: 6054122176`

That is about:

- `5.6383 GiB`

which matches the estimate from `95`.

## Candidate generation rule

For the first pass:

- routed layers kept at `Q4_K` stay unchanged
- routed layers moved to `Q2` use:
  - `ffn_gate_exps.weight -> iq2_xxs`
  - `ffn_up_exps.weight -> iq2_xxs`
  - `ffn_down_exps.weight -> q2_k`

All non-routed tensors stay at baseline types.

This means the first sweep is a clean routed-expert precision frontier, not a
mixed dense/routed search yet.

## Planned build inputs

Base inputs:

- HF source: `../ds4/hf-dspark`
- template: `../ds4/gguf/dspark.gguf`
- imatrix: `/private/tmp/dspark_sweep2ctx/recoverablegap_boosted2ctx.imatrix.dat`

This is intentionally conservative:

- it reuses the known DSpark imatrix path
- it avoids introducing a new calibration branch before the first frontier
  exists

## Planned first-pass artifacts

Recommended artifact directory:

- `/private/tmp/dspark_pareto_q2q4`

Recommended naming:

- `dspark_frontier_all_q4.gguf`
- `dspark_frontier_all_q2.gguf`
- `dspark_frontier_only_mtp2_q4.gguf`
- `dspark_frontier_only_mtp1_q4.gguf`
- `dspark_frontier_only_mtp0_q4.gguf`
- `dspark_frontier_mtp01_q4.gguf`
- `dspark_frontier_mtp02_q4.gguf`
- `dspark_frontier_mtp12_q4.gguf`

## Measurement rule

After build, every candidate should be scored on the existing 4-context sweep:

- `8192`
- `16384`
- `24576`
- `32768`

using the same runtime-side `measure_metal_b2.py` path already used for the
current branch.

The first table to produce should be:

| label | total GiB | mean accepted | mean committed |
|---|---:|---:|---:|

with the non-dominated subset then promoted into the actual frontier note.

## Decision impact

This note removes one major uncertainty from the Pareto branch:

- the first frontier sweep is straightforward to generate from the current
  quantizer surface

So the next real branch action is no longer design work. It is:

- build the first-pass `Q2/Q4` routed candidates
- measure them
- record the frontier
