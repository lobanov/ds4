# Anchor-weighted hardness ablation on the 4-context sweep root

Date: 2026-07-01

## Why this pivot was tried

`46_4ctx_19token_ablation_default_vs_front.md` established that:

- raising the collector budget from `3` to `19` tokens per bundle stabilizes
  the signal
- but the default position-weighted recipe still only reaches
  `+0.2854%` mean accepted on the live `4`-context root
- a more front-loaded position schedule makes the result slightly worse

That made pure draft-position weighting look largely exhausted. The next
plausible lever was to weight **anchor steps**, not just draft positions.

A fresh `gpt-5.5 xhigh` review recommended trying **empirical baseline
hardness** before target uncertainty:

- target uncertainty is cleaner, but only a proxy
- baseline per-step accepted length is closer to the actual objective and can
  capture DSpark-specific failure modes that low-margin target steps miss

## Code change made

Added optional **per-anchor scalar weighting** support to the DSpark
acceptance-bundle collector in `ds4.c`.

New behavior:

- if a bundle directory contains `imatrix_anchor_weights.txt`
- the collector reads one positive float weight per anchor step
- that scalar multiplies the sample contribution for:
  - gate/up imatrix sums
  - down imatrix sums
  - the corresponding normalization weights

This leaves the existing draft-position bucket merge intact, so the final
sample weight is effectively:

```text
anchor_weight(step) * draft_position_weight(pos)
```

Also extended `issue468/run_dspark_weighted_from_sweep_root.py` so it can:

1. build a temporary overlay sweep root
2. synthesize `imatrix_anchor_weights.txt` for each `ctx_*` bundle
3. collect the imatrix from that weighted overlay
4. quantize and reprobe as before

For this run the helper used:

- source: baseline hardness
- source file per bundle:
  `baseline-weighted4ctx_19t_default_256tr.b2.json`

Weight rule:

```text
h = clamp((5.0 - per_step_accepted) / 5.0, 0, 1)
raw = 1.0 + 1.5 * h
weight = clamp(raw / mean(raw), 0.5, 2.0)
```

This mostly upweighted the empirically bad anchor steps already visible in the
baseline curves, especially steps `1`, `6`, and `7`.

## Hardness-weighted run

Command:

```sh
python3 issue468/run_dspark_weighted_from_sweep_root.py \
  --sweep-root /tmp/dspark_sweep8 \
  --collector-max-tokens 19 \
  --trials 256 \
  --anchor-weight-source baseline-hardness \
  --anchor-weight-b2-label baseline-weighted4ctx_19t_default_256tr \
  --run-label weighted4ctx_19t_hard_256tr
```

Artifacts:

- imatrix: `/tmp/dspark_sweep8/weighted4ctx_19t_hard_256tr.imatrix.dat`
- candidate: `/tmp/dspark_sweep8/weighted4ctx_19t_hard_256tr.gguf`
- anchor-weight manifest:
  `/tmp/dspark_sweep8/weighted4ctx_19t_hard_256tr.anchor_weights.json`
- summary JSON: `/tmp/dspark_sweep8/weighted4ctx_19t_hard_256tr.summary.json`
- summary TSV: `/tmp/dspark_sweep8/weighted4ctx_19t_hard_256tr.summary.tsv`

Results:

| context | baseline accepted | candidate accepted | accepted delta | baseline committed | candidate committed | committed delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 4.0997 | 4.1497 | +1.219% | 4.5068 | 4.5448 | +0.844% |
| 16384 | 4.2136 | 4.2319 | +0.434% | 4.5006 | 4.5134 | +0.283% |
| 24576 | 4.2315 | 4.2179 | -0.321% | 4.6114 | 4.5954 | -0.348% |
| 32768 | 4.2572 | 4.2621 | +0.116% | 4.5356 | 4.5376 | +0.045% |

Mean delta across the four contexts:

- accepted: `+0.3620%`
- committed: `+0.2062%`

## Comparison against prior 4-context recipes

Same root, same collector budget, same `256`-trial B2 evaluation:

| run | mean accepted delta |
| --- | ---: |
| `weighted4ctx_19t_front_256tr` | `-0.0297%` |
| `weighted4ctx_19t_default_256tr` | `+0.2854%` |
| `weighted4ctx_19t_hard_256tr` | `+0.3620%` |

Interpretation:

- empirical hardness weighting is the best of the three tested 4-context
  recipes
- but the gain over the default recipe is small:
  `+0.3620% - +0.2854% = +0.0766%`
- that is still nowhere near the assignment gate of `>= +5%`

## What this means

This branch now has direct evidence that:

1. more collector budget helps only marginally
2. stronger front-loaded draft-position weighting does not help
3. anchor-hardness weighting is slightly better than the default recipe, but
   still far too small to matter

So the current acceptance-bundle family remains substantially short of the
target even after adding:

- multi-context aggregation
- larger token budgets
- sharper position weights
- anchor-step hardness weights

The next useful move should likely be a **larger objective change**, not
another small reweighting tweak. Plausible directions:

- held-out validation on new contexts only after a stronger candidate exists
- weighting by verifier-margin or rejection event using a more faithful target
  signal than truncated top-k entropy
- changing what hidden states are collected, not just how current frontier
  states are weighted
- separating recoverable quantization-sensitive failures from fundamentally
  hard drafter failures
