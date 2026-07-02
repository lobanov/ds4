# Iteration 3: Argmax→Sampling Fix — WORSE (Reverted)

Date: 2026-07-01. Iteration 3 of the live-vs-oracle investigation. Doc-only.

## 0. What was tested

Replaced the drafter's argmax draft selection with sampling from softmax(q_row),
matching the MC oracle's `RNG.choice(len(q), p=q)`. This is the standard
rejection-sampling protocol: the drafter should SAMPLE from q, not take its mode.

## 1. Result: WORSE than argmax

| metric | argmax (baseline) | sampling (tested) | delta |
|---|---|---|---|
| accepted drafts/cycle | 1.97 | **1.54** | −0.43 |
| n_accept | 3.86 | 3.45 | −0.41 |
| pos 0 accept rate | 68.2% | **56.8%** | −11.4pp |
| gen t/s | 29.5 | 26.1 | −3.4 |

Sampling made things WORSE. Reverted.

## 2. Why sampling is worse

The drafter's q distribution on stochastic paths is genuinely wrong — it assigns
high probability to tokens the target assigns ~0 probability. This is the
hidden-state-conditioning OOD issue (issue468/72-73).

With argmax: the drafter at least proposes its best guess, which has some chance
of matching the target's argmax (~68% at pos 0).

With sampling: the drafter proposes random tokens from q, many of which are
WORSE than the argmax (because q is peaked on wrong tokens). This lowers
acceptance further (56.8% at pos 0).

## 3. Why the MC oracle achieves 95.5% at pos 0

The MC oracle's 95.5% acceptance = 1 − TV(p,q) on the GREEDY PATH. On the
greedy path, the drafter's q matches the target's p well (TV ≈ 0.045), so
sampling from q gives high acceptance.

On stochastic paths, TV(p,q) is much larger (~0.32 at pos 0), so both argmax
AND sampling give lower acceptance. The gap between the oracle's 95.5% and the
live 68% is the OOD generalization gap — sampling vs argmax doesn't change it.

## 4. This confirms: the gap is architectural, not a code bug

The argmax→sampling experiment confirms that the 2.13 accepted-drafts/cycle gap
is from the drafter's hidden-state conditioning on stochastic paths. Neither
argmax nor sampling can recover the gap — it's inherent to the drafter's
distributional quality on OOD inputs.

## 5. Iteration report

| metric | argmax (baseline) | sampling (tested) | oracle MC (greedy) |
|---|---|---|---|
| accepted drafts/cycle | 1.97 | 1.54 | 4.10 |
| committed tokens/cycle | 2.97 | 2.54 | 5.10 |
| gen t/s | 29.5 | 26.1 | — |
| ms/token | 33.9 | 38.3 | — |
| pos 0 accept | 68.2% | 56.8% | 95.5% |

## 6. Conclusion

The argmax→sampling hypothesis is DISPROVEN. The gap is architectural, not a
draft-selection-mode issue. The drafter's q distribution is genuinely wrong on
stochastic paths (OOD main_hidden from non-greedy anchors).
