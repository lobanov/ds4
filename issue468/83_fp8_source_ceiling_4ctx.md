# FP8 Source Ceiling — 4-Context Oracle Sweep

Date: 2026-07-01

## Purpose

Record the first 4-context source-reference ceiling measurement using the new
converted-checkpoint import path from `82_ref_ckpt_oracle_validation.md`.

This is the first multi-context answer to the question from
`52_fp8_headroom_plan.md`:

- how much headroom is actually visible above the current `Q4_K` baseline on
  the current sweep family?

## Method

Weight source:

- `~/ds4/ref-ckpt/model0-mp1.safetensors`

Scoring path:

- `issue468/ref/measure_ref_oracle_b2.py`

Bundle family:

- `/tmp/dspark_sweep8/ctx_{08192,16384,24576,32768}`

Comparison baseline:

- bundle-local `baseline.b2.json`

Matched horizon:

- `19` decode steps for every context

This keeps the source-reference run directly comparable to the existing sweep
baseline summaries.

## Per-context results

### `ctx_08192`

- source-reference committed:
  - `4.425986842105263`
- baseline committed:
  - `4.473684210526316`
- delta:
  - `-1.0661764705882426%`

### `ctx_16384`

- source-reference committed:
  - `4.503700657894737`
- baseline committed:
  - `4.519736842105263`
- delta:
  - `-0.35480349344976236%`

### `ctx_24576`

- source-reference committed:
  - `4.505756578947368`
- baseline committed:
  - `4.578947368421052`
- delta:
  - `-1.5984195402298784%`

### `ctx_32768`

- source-reference committed:
  - `4.569078947368421`
- baseline committed:
  - `4.5394736842105265`
- delta:
  - `+0.6521739130434856%`

## 4-context mean

- mean source-reference committed:
  - `4.501130756578947`
- mean baseline committed:
  - `4.527960526315789`
- mean delta:
  - `-0.5925354159099228%`

## Interpretation

This is the key result:

- the source-reference FP8 path is **roughly at parity** with the current
  baseline on the current 4-context sweep family

It is **not** showing a large positive ceiling above the present artifact.

That does **not** mean the source path is useless. It means:

1. the current sweep family is not exposing a large source-vs-baseline gap
2. the main remaining opportunity is unlikely to come from broad route changes
   justified only by a supposed large FP8 ceiling
3. the earlier nearby dense-Q8 results being near-flat are more believable in
   light of this ceiling result

## Relation to earlier branches

This result resolves the immediate question left open by `52` and the failed
torch-fallback branch:

- the torch fallback was too weak to trust
- the oracle/source import path is trustworthy enough
- but the measured FP8/source ceiling on this sweep family is still close to
  the current baseline

So the branch answer is:

- there is **no strong evidence here for large acceptance headroom above the
  current route** on these four contexts

## Consequence for search priority

This lowers the priority of expensive new broad quantization-route work as the
lead branch.

The search should return to:

- concentrated tensor / mechanism sensitivity
- acceptance-boundary-local levers
- especially places where small routing perturbations can matter more than
  broad dtype replacement

That is more aligned with:

- `ffn_gate_inp`
- other router-sensitive tensor classes
- acceptance-local static reallocations

than with:

- another large source-ceiling revival effort
- or a broad route swap justified by a large missing FP8 ceiling
