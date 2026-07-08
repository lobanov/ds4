# Activity 4 — crossed hidden/label oracle diagnostic

Date: 2026-07-06. Status: **complete.** Harness: `issue468/run_stage2_crossed_oracle.py`
(reuses the Stage-1 IQ2XXS + Q4-tap exactness bundles; no new capture). Artifacts:
`issue468/artifacts/stage2_crossed_oracle/summary.json`.

## Question

Separate **input-shift** (the drafter's hidden-state representation differs from
what it was trained on) from **label-shift** (the served target's argmax drifts) —
to decide whether the LoRA should target the **input side** (main_proj + dense, to
remap the representation) or the **output side** (hc_head + markov, to fix the
argmax ordering). This localizes the deficit *before* any training.

## Method — 2×2 crossed acceptance at p=1

For each (prompt, measure-step) we have the drafter's p=1 prediction (`draft[0]`)
and the target's p=1 token (`target[0]`) for BOTH the IQ2XXS and Q4-tap targets
(from the Stage-1 acceptance summaries). Build:

`acc(input X, label Y) = P( drafter_draft_X[0] == target_Y[0] )`, X,Y ∈ {IQ2,Q4}.

- **input main effect** ≈ `[acc(IQ2,·) − acc(Q4,·)]` → representation/hidden-state side
- **label main effect** ≈ `[acc(·,IQ2) − acc(·,Q4)]` → argmax-drift side

Computed on **prefix-aligned cells** (IQ2 and Q4 generated the same token prefix
through the anchor → same context → hidden states differ *only* by precision → clean
isolation), 54/80 cells, plus the full set for reference.

## Result

| acc(input,label) | IQ2XXS hidden | Q4-tap hidden | (rows=label, cols=input) |
|---|---|---|---|
| **IQ2XXS token** | 0.815 | 0.815 | acc(·,IQ2)=0.815 both inputs |
| **Q4-tap token** | 0.796 | 0.796 | acc(·,Q4)=0.796 both inputs |

- **input main effect = 0.0**; **input_shift_rate = 1.85%** (drafter's p=1 prediction
  flips IQ2↔Q4 on only 1.85% of aligned cells).
- **label main effect = 1.85 pp**; **label_shift_rate = 9.26%** (target argmax flips
  ~5× more often than the drafter's prediction).

**For a fixed label, swapping the input does literally nothing** (`acc(IQ2,·) ==
acc(Q4,·)` at both labels). The drafter is **input-invariant** at p=1.

## Verdict — label/ordering-side dominates; input-side is negligible

The acceptance deficit is **not** a representation/input problem: the drafter's p=1
prediction is essentially identical whether it sees IQ2XXS or Q4-tap hidden states
(98.15% agreement). The only thing that moves acceptance is the **target argmax**
(label drift). 

*Observation:* the drafter's p=1 prediction being input-invariant implies base_logits
(the hidden-state-dependent term) is dominated at p=1 by the markov bias (which
depends on the anchor, shared on aligned cells) — i.e. the frozen drafter under-uses
the hidden state at the first position. Regardless of cause, the practical signal is
clear.

## Implication for the LoRA focus

- **Output side (hc_head + markov) is the lever** — train the ordering to match the
  served target's argmax. **Strongly supports Activity 6/7 (head-only) as the primary
  intervention.**
- **Input side (main_proj + dense) is unlikely to help** — the drafter already ignores
  the hidden-state perturbation, so remapping the representation would add little.
  This downweights the expected value of Activity 8's input-side ablations (still run
  them if 7 < 6, but expect a small incremental gain).

Net: the crossed oracle predicts the head-only LoRA (Activity 7) is where any
acceptance gain will come from, and that the from-scratch head ceiling (Activity 6)
is the gate to watch. Consistent with Stage 0 (shallow, output-ordering errors) and
Stage 1 (raising input precision didn't help).
