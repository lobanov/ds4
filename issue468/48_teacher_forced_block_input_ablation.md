# Teacher-forced block-input ablation on the 4-context sweep root

Date: 2026-07-01

## Why this pivot was necessary

`47_anchor_weighted_hardness_ablation.md` showed that the current
acceptance-bundle family was still far below the assignment gate even after:

- multi-context aggregation
- larger collector budgets
- draft-position weighting
- anchor-hardness weighting

At that point the main unresolved gap in the assignment brief was no longer a
weighting issue. It was a **state-distribution issue**:

- the assignment explicitly called for collection on the **correct target
  trajectory**
- but the current acceptance-bundle collector still seeded the DSpark block as
  `[anchor, NOISE, NOISE, NOISE, NOISE]`

So the prior runs changed sample weights without changing the intra-block
states being collected.

## Code change made

Updated the acceptance-bundle collector path in `ds4.c` so that bundle
collection now uses **teacher-forced block inputs** from `target_greedy.json`.

For each anchor step `step`, the collector now seeds the DSpark block as:

```text
[greedy[step], greedy[step+1], greedy[step+2], greedy[step+3], greedy[step+4]]
```

instead of:

```text
[greedy[step], NOISE, NOISE, NOISE, NOISE]
```

Important scope:

- this is a **collector-only** change for acceptance bundles
- it does **not** change the production runtime path
- the generic corpus collector still uses the old noise-seeded block behavior

This matches the unresolved requirement in
`34_dspark_imatrix_q4k_assignment.md` much more closely than the earlier
reweighting-only runs.

## Fresh review before implementation

A fresh `gpt-5.5 xhigh` review agreed this was the highest-value next pivot and
called out the main risk clearly:

- it matches the assignment gap
- but it is an imatrix-only A/B, not a runtime correctness fix
- it may introduce distribution shift because real DSpark runtime still uses
  noise placeholders in the block

That makes this a legitimate test of the assignment hypothesis, but not
something to silently reinterpret as production behavior.

## Teacher-forced run

Ran the same strongest baseline comparison as before:

```sh
python3 issue468/run_dspark_weighted_from_sweep_root.py \
  --sweep-root /tmp/dspark_sweep8 \
  --collector-max-tokens 19 \
  --trials 256 \
  --run-label weighted4ctx_19t_tf_default_256tr
```

Artifacts:

- imatrix: `/tmp/dspark_sweep8/weighted4ctx_19t_tf_default_256tr.imatrix.dat`
- candidate: `/tmp/dspark_sweep8/weighted4ctx_19t_tf_default_256tr.gguf`
- summary JSON: `/tmp/dspark_sweep8/weighted4ctx_19t_tf_default_256tr.summary.json`
- summary TSV: `/tmp/dspark_sweep8/weighted4ctx_19t_tf_default_256tr.summary.tsv`

Results:

| context | baseline accepted | candidate accepted | accepted delta | baseline committed | candidate committed | committed delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 4.0997 | 4.1071 | +0.181% | 4.5068 | 4.5175 | +0.237% |
| 16384 | 4.2136 | 4.2169 | +0.078% | 4.5006 | 4.5002 | -0.009% |
| 24576 | 4.2315 | 4.1848 | -1.103% | 4.6114 | 4.5664 | -0.976% |
| 32768 | 4.2572 | 4.2642 | +0.164% | 4.5356 | 4.5397 | +0.091% |

Mean delta across the four contexts:

- accepted: `-0.1700%`
- committed: `-0.1644%`

## Comparison against prior 4-context recipes

| run | mean accepted delta |
| --- | ---: |
| `weighted4ctx_19t_front_256tr` | `-0.0297%` |
| `weighted4ctx_19t_tf_default_256tr` | `-0.1700%` |
| `weighted4ctx_19t_default_256tr` | `+0.2854%` |
| `weighted4ctx_19t_hard_256tr` | `+0.3620%` |

Interpretation:

- the teacher-forced block-input collector is **not** an improvement on this
  4-context root
- it performs materially worse than the prior non-teacher-forced default
- the main regression is at `24576`

## What this means

This was the most important remaining collector-logic hypothesis from the
assignment brief. The result is negative:

- matching the target continuation inside the DSpark block does **not** improve
  the measured candidate here
- on this root, it makes the result worse

That substantially lowers the probability that the current
acceptance-bundle-imatrix family can reach the assignment gate through further
incremental collector refinements alone.

Practical conclusion:

- do **not** spend the next turn scaling this teacher-forced variant to the
  full 8-context sweep
- if work continues on this assignment, it should likely shift to a larger
  alternative hypothesis rather than more local collector tuning
