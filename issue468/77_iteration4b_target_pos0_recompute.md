# Iteration 4b: Target-Pos0 + Markov Recompute — +3.5 t/s (29.5→33.0)

Date: 2026-07-01. Iteration 4b. Doc-only research record.

## 0. Fix: Markov chain recompute after target-pos0 override

After overriding drafts[0] with a target-sampled token, recompute drafts[1..4]
by re-running the Markov head with prev = drafts[0] (the target token). The
drafter's base_logits are the same; only the Markov bias changes.

## 1. Result (temp=1, n=256, seed=1)

| pos | baseline | target-pos0 v1 (no recompute) | target-pos0 v2 (with recompute) | oracle |
|---|---|---|---|---|
| 0 | 68.2% | 98.4% | **96.6%** | 95.5% |
| 1 | 82.2% | 46.0% | **75.4%** | 98.1% |
| 2 | 67.6% | 72.4% | 55.8% | 91.8% |
| 3 | 64.0% | 66.7% | 65.2% | 87.9% |
| 4 | 43.8% | 50.0% | **73.3%** | 78.5% |

| metric | baseline | target-pos0 v2 | delta |
|---|---|---|---|
| drafts/cycle | 1.97 | **2.54** | **+0.57** |
| n_accept | 3.86 | **4.34** | **+0.48** |
| committed/cycle | 2.97 | **3.54** | **+0.57** |
| gen t/s | 29.5 | **33.0** | **+3.5** |
| ms/token | 33.9 | **30.3** | **−3.6** |
| baseline t/s | ~39 | ~39 | — |
| gap to baseline | 9.5 | **6.0** | −3.5 |

## 2. Analysis

Pos-0 is now near-oracle (96.6% vs 95.5%). Pos-1 recovered to 75.4% (from 46%
without recompute). Pos-4 improved dramatically (43.8% → 73.3%). Pos-2 dropped
(67.6% → 55.8%) — the sequential Markov chain still accumulates some divergence
because the drafter's base_logits[2] was computed with the original argmax chain.

The 6 t/s gap remains because committed/cycle is 3.54 vs the baseline's needed
~5.1 to break even. But this is a significant improvement from the code-only
target-pos0 change.

## 3. Iteration report

| metric | baseline | target-pos0 v2 | oracle MC (greedy) |
|---|---|---|---|
| accepted drafts/cycle | 1.97 | **2.54** | 4.10 |
| committed tokens/cycle | 2.97 | **3.54** | 5.10 |
| gen t/s | 29.5 | **33.0** | — |
| ms/token | 33.9 | **30.3** | — |
