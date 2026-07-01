# 4-context 19-token ablation: default vs front-loaded position weights

Date: 2026-07-01

## Why this run mattered

The earlier sweep-root evidence was too weak to justify spending on the full
`8k..64k` root with the current recipe:

- `44_sweeproot_collector_and_2ctx_weighted_signal.md` showed a positive
  2-context signal, but from a tiny `3`-token collector budget
- `45_weighted_sweeproot_helper_and_3ctx_reprobe.md` showed that once a third
  context was added, the measured signal became effectively flat / slightly
  negative

A fresh `gpt-5.5 xhigh` review recommended not scaling the current
`--collector-max-tokens 3` recipe further until a higher-budget controlled
ablation was run first.

## Setup

The aborted tail-capture attempt left a complete `ctx_32768` target bundle in
`/tmp/dspark_sweep8`, so the live root now contains four usable contexts:

- `ctx_08192`
- `ctx_16384`
- `ctx_24576`
- `ctx_32768`

All four have:

- `prompt_rendered.txt`
- `target_topk.json`
- `target_greedy.json`
- target hidden-state dumps

This made it possible to run a stronger same-root ablation without generating
new capture data.

## Run A: default position weights, larger collector budget

Command:

```sh
python3 issue468/run_dspark_weighted_from_sweep_root.py \
  --sweep-root /tmp/dspark_sweep8 \
  --collector-max-tokens 19 \
  --trials 256 \
  --run-label weighted4ctx_19t_default_256tr
```

Artifacts:

- imatrix: `/tmp/dspark_sweep8/weighted4ctx_19t_default_256tr.imatrix.dat`
- candidate: `/tmp/dspark_sweep8/weighted4ctx_19t_default_256tr.gguf`
- summary JSON: `/tmp/dspark_sweep8/weighted4ctx_19t_default_256tr.summary.json`
- summary TSV: `/tmp/dspark_sweep8/weighted4ctx_19t_default_256tr.summary.tsv`

Results:

| context | baseline accepted | candidate accepted | accepted delta | baseline committed | candidate committed | committed delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 4.0997 | 4.1371 | +0.913% | 4.5068 | 4.5428 | +0.798% |
| 16384 | 4.2136 | 4.2294 | +0.376% | 4.5006 | 4.5123 | +0.260% |
| 24576 | 4.2315 | 4.2177 | -0.326% | 4.6114 | 4.5975 | -0.303% |
| 32768 | 4.2572 | 4.2648 | +0.179% | 4.5356 | 4.5382 | +0.059% |

Mean delta across the four contexts:

- accepted: `+0.2854%`
- committed: `+0.2036%`

Interpretation:

- increasing the collector budget from `3` to `19` tokens stabilizes the sign
  compared with the noisy 3-context reprobe
- but the effect is still very small
- it is nowhere close to the assignment gate of `>= +5%` average acceptance

## Run B: front-loaded position weights

Command:

```sh
python3 issue468/run_dspark_weighted_from_sweep_root.py \
  --sweep-root /tmp/dspark_sweep8 \
  --collector-max-tokens 19 \
  --trials 256 \
  --draft-pos-weights 1,0.25,0.1,0.05,0.02 \
  --run-label weighted4ctx_19t_front_256tr
```

Artifacts:

- imatrix: `/tmp/dspark_sweep8/weighted4ctx_19t_front_256tr.imatrix.dat`
- candidate: `/tmp/dspark_sweep8/weighted4ctx_19t_front_256tr.gguf`
- summary JSON: `/tmp/dspark_sweep8/weighted4ctx_19t_front_256tr.summary.json`
- summary TSV: `/tmp/dspark_sweep8/weighted4ctx_19t_front_256tr.summary.tsv`

Results:

| context | baseline accepted | candidate accepted | accepted delta | baseline committed | candidate committed | committed delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 4.0997 | 4.0822 | -0.426% | 4.5068 | 4.4959 | -0.242% |
| 16384 | 4.2136 | 4.2331 | +0.464% | 4.5006 | 4.5156 | +0.333% |
| 24576 | 4.2315 | 4.2202 | -0.267% | 4.6114 | 4.6001 | -0.245% |
| 32768 | 4.2572 | 4.2619 | +0.111% | 4.5356 | 4.5387 | +0.068% |

Mean delta across the four contexts:

- accepted: `-0.0297%`
- committed: `-0.0214%`

Interpretation:

- a stronger front-loaded weighting schedule does **not** improve the result
- on this 4-context root it slightly underperforms the default weighting
- the problem is therefore unlikely to be solved by simply pushing more weight
  toward very early draft positions

## What this means

This branch now has a much stronger same-root ablation than the earlier
3-token experiments:

- `4` real contexts
- `19` collector tokens per bundle
- `256` B2 trials per context

That evidence is still far below the assignment threshold:

- best mean accepted delta seen here: `+0.2854%`
- required gate: `>= +5%`

So the current acceptance-bundle recipe does not yet justify spending on a
full `8`-context evaluation with the same weighting approach. The next useful
step should be a materially different collection or weighting objective, for
example:

- hard-case / low-margin weighting
- acceptance-length-conditioned weighting
- verifier-margin-aware weighting
- a broader change in what states are collected, rather than only how current
  per-position samples are merged
