# Independent Pi Agent Review — Leads Reframe (issue468/70)

Date: 2026-07-01. Thirtieth productionization handoff note. Records the pi
agent's independent review of all issue468/32-69 notes. Doc-only research record.

## 0. The review

Launched `pi -p` (non-interactive) with a comprehensive prompt covering all
prior research. The agent read notes 32-69 (and 70, which was created during
the session by the parallel process) and produced a ranked lead assessment.

## 1. CRITICAL REFRAME: the 0.71 live-vs-probe gap, not the 0.12 probe-vs-oracle gap

The agent identified that notes 53-69 (the ENTIRE MoE/Q4_K/Q8_0/exact-Q4/F16-MMA
bisection) chased the SMALL gap:

| metric | avg prefix | gap from oracle |
|---|---|---|
| Oracle probe | 4.49 | — |
| Metal probe | 4.37 | 0.12 (notes 53-69) |
| Live B2 (cap=5) | ~4.1 | 0.27 (residual) |
| Live B2 (default) | 3.66 | **0.71 (the big one)** |

The 0.71 live-vs-probe gap is where the t/s lives. At cap=5, closing only the
0.27 residual gap would cross the gate (4.37/0.110s ≈ 39.7 t/s). You don't need
the +8.27%.

## 2. Ranked leads

### TIER 1 (in-scope, B2-safe, measured upside)

**1. Adaptive n_real cap** (extends a note-70 shipped win): note 70 implemented
a static env cap (cap=5 → 31.9 t/s vs default 25.7 Fibonacci; +5 t/s on
degenerate prompts, −1 t/s on good prompts). Making the cap ADAPTIVE (start
large, shrink when zero-accept detected) captures the wins without the losses.
Plausible net +1-3 t/s.

**2. Reconcile batch-verify ↔ decode logits divergence**: the probe's 4.37 uses
sequential decode; live B2 uses batch-verify logits. These diverge (batch argmax
≠ sequential argmax, note 41). Nobody tested whether this divergence costs
stochastic-B2 acceptance. Risk: target-path change (baseline regression surface).
Gate-crossing trajectory at cap=5 (4.1→4.37 = ~+8 t/s).

### TIER 2 (diagnostic/marginal/high-risk)

3. Root-cause long-window degradation (why n_real→128 collapses drafts)
4. Correction-decode elimination (note 38 ruled infeasible; note 70 optimistic bound ~38-40 t/s)
5. Drafter sync-fusion (+0.1-0.4 t/s, ceiling ≤2)

### TIER 3 (out of scope, but high theoretical value)

6. **Drafter steering** (Zhou et al., note 42): +35% accepted, +22% throughput.
   FROZEN drafter, no retraining. Completely unexplored. Brushes "B2 design fixed"
   boundary.
7. **Draft trees** (EAGLE-3, note 42): training-free, attacks accepted-tokens-per-
   verify. Substantial implementation, touches "B2 design fixed" boundary.
8. **FP8 source-precision ceiling** (note 52 plan, never executed): long shot given
   Q8_0 went worse.

## 3. What is genuinely exhausted (don't re-attempt)

- The entire per-op precision chase (notes 53-69): every op cos >0.9999 on
  identical input; the 0.12 probe-vs-oracle gap is cumulative drift, not single-
  op fixable.
- Q8_0 re-quant: worse (note 69).
- imatrix: falsified (note 49).
- Opp1/Opp-a: structurally infeasible.
- Block size: wash.

## 4. The agent's bottom line

The highest-viability in-scope lever is the **adaptive n_real cap** (Tier 1 #1).
The one lever with a gate-crossing acceptance trajectory is **batch-verify ↔
decode reconciliation** (Tier 1 #2, but target-path risk). Everything that
crosses the gate with margin — drafter steering, draft trees, unfreezing — is
out of the current scope and needs a /goal-tweak.

## 5. My assessment

The pi agent's reframe is correct and important. I was chasing the wrong gap.
The 0.12 probe-vs-oracle gap is ~17% of the total problem; the 0.71 live-vs-
probe gap is ~83% — and it's where the perf lives. The adaptive cap is a
straightforward extension of note 70's shipped win. The batch-verify
reconciliation is the highest-upside lever but needs careful evaluation.
