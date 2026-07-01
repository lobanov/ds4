# Q-dump oracle envelope falsification on the 4-context sweep root

Date: 2026-07-01

## Why this was the right next move

After `48_teacher_forced_block_input_ablation.md`, the acceptance-bundle
collector family had exhausted its most plausible local hypotheses:

- more collector budget
- stronger draft-position weights
- anchor-hardness weights
- teacher-forced block inputs

The best real 4-context result was still only:

- `weighted4ctx_19t_hard_256tr`: `+0.3620%` mean accepted

against an assignment gate of:

- `>= +5%` average acceptance

At that point the next useful step was no longer “build another candidate.”
It was an **upper-bound / falsification check**: ask how much headroom is left
even if an oracle could cherry-pick the best already-observed q-dump at every
anchor step.

## New helper

Added:

- `issue468/run_dspark_qdump_envelope.py`

Purpose:

1. scan an existing sweep root with saved q-dumps
2. rescore q-dumps with `measure_metal_b2.py` or reuse existing B2 JSONs
3. for each context and anchor step, take the best `per_step_accepted` value
   across several sources
4. compare that impossible oracle chooser against a fixed baseline

This is explicitly **not** a real candidate-evaluation harness. It is a
falsification tool.

## Envelope definition

Sweep root:

- `/tmp/dspark_sweep8`

Contexts included:

- `8192`
- `16384`
- `24576`
- `32768`

Baseline source:

- `baseline-weighted4ctx_19t_default_256tr`

Candidate sources included in the chooser:

- `weighted4ctx_19t_default_256tr`
- `weighted4ctx_19t_front_256tr`
- `weighted4ctx_19t_hard_256tr`
- `weighted4ctx_19t_tf_default_256tr`

Oracle envelope rule:

1. for each context and each anchor step
2. take `max(per_step_accepted)` across:
   - baseline
   - default candidate
   - front-loaded candidate
   - hardness-weighted candidate
   - teacher-forced candidate
3. average those per-step maxima over the 19 measured steps
4. compare that mean against the fixed baseline mean

This chooser is impossible in practice because it selects after the fact on a
per-anchor basis.

## Quick reproduced result from existing 256-trial JSONs

Command:

```sh
python3 issue468/run_dspark_qdump_envelope.py \
  --trials 256 \
  --reuse-original-b2 \
  --out-label oracle-envelope-existing256
```

Artifacts:

- summary JSON: `/tmp/dspark_sweep8/oracle-envelope-existing256.summary.json`
- summary TSV: `/tmp/dspark_sweep8/oracle-envelope-existing256.summary.tsv`
- detail JSON: `/tmp/dspark_sweep8/oracle-envelope-existing256.details.json`

Per-context results:

| context | baseline accepted | oracle envelope | envelope delta |
| --- | ---: | ---: | ---: |
| 8192 | 4.0997 | 4.1663 | +1.625% |
| 16384 | 4.2136 | 4.2486 | +0.829% |
| 24576 | 4.2315 | 4.2751 | +1.030% |
| 32768 | 4.2572 | 4.2800 | +0.536% |

Mean envelope delta across the four contexts:

- accepted: `+1.0051%`

Per-step winner counts across all 76 anchor steps:

- baseline: `41`
- `weighted4ctx_19t_default_256tr`: `12`
- `weighted4ctx_19t_tf_default_256tr`: `10`
- `weighted4ctx_19t_front_256tr`: `9`
- `weighted4ctx_19t_hard_256tr`: `4`

## Interpretation

This is a very strong negative result.

Even an impossible oracle that is allowed to:

- inspect all currently saved q-dumps
- choose per anchor step after the fact
- mix baseline and multiple candidate families freely

still only gets:

- `+1.0051%` mean accepted

That is far below the assignment gate of:

- `>= +5%`

So the failure is no longer just “the current collector merge weights are
wrong.” The existing q-dump family itself does not contain enough acceptance
headroom to justify more Q4_K+imatrix iteration on this path.

## Practical consequence

This result materially lowers the value of:

- more sweep-root scaling with the same candidate family
- more local acceptance-bundle reweighting
- more post-hoc selection over the same q-dump sources

The next meaningful move, if any, should be a **larger hypothesis change** than
the current Q4_K+imatrix collector family. Under the present engine path, the
assignment now looks very unlikely to reach its `+5%` average-acceptance gate.
