# Lead 01 — anchor-reuse falsifier: result

Date: 2026-07-07. Status: **resolved** (both codex gates passed; verdict recorded with
the GATE-2-required softening). Harness: `issue468/run_anchor_reuse_falsifier.py`
(reuse modes added to `dspark_oracle/measure_acceptance_bundle.py`). Artifacts:
`issue468/artifacts/anchor_reuse_falsifier/`. Reviews: `codex_setup_review.md`
(GATE 1), `codex_verdict_review.md` (GATE 2).

## Question

Does the DSpark drafter's acceptance survive drafting from the **last-accepted-position
(stale) target hidden** with the correction token entering only as embedding? This is
the realizability test for **anchor reuse** — the load-bearing unverified assumption of
`spec_speedup_model`'s optimistic edge (K=4 at **−0.9%** under reuse vs **−18.9%** under
the shipped verifier that decodes every cycle). The original adversarial review
(`spec_speedup_model.md` § Adversarial review) flagged anchor reuse as the primary risk
and proposed this falsifier; it had never been run.

## Method (offline, reusing retained greedy-spine captures)

Along the greedy spine the correction token at step t **is** the greedy target token, so
both the "true" hidden at `pos` and the "stale" hidden at `pos−1` are on disk
(`main_hidden[pos]` is the POST-token hidden, position pos; `target_tokens[k]` sits at
`pos0+k`). The harness swaps the drafter's per-step hidden source `main_hidden[pos]` →
`main_hidden[pos−1]`; `main_hidden` enters the draft only via the per-step KV-window
entry, so this is the entire change. Two KV-window models tested:

- **lag** (the lead's specified method): stale every step → KV window is a consistent
  one-position lag of baseline.
- **backfill** (added per GATE 1): current slot stale, prior slots backfilled true
  (models a folded verifier that commits accepted positions with true hiddens).

10 prompts × ~8 steps = 80 paired (prompt,step) units per cell; 3 temps (t0p0 greedy,
t0p5/t1p0 sampled-stream) × 2 modes = **6 cells**. Paired bootstrap CI (per-step AND
per-prompt-clustered) on E[a|5block] delta; exact McNemar on p=1. Two-tier rule
(user-specified): SURVIVES if ΔE[a|5block]≥0; COLLAPSES if Δ<0 and per-step CI excludes
0; else MARGINAL. **Fidelity gate passed at every temp** (fresh baseline reproduces the
retained `acceptance_summary.json` draft tokens bit-for-bit).

## Result — all 6 cells

| temp | mode | base E[a\|5] | reuse E[a\|5] | Δ | per-step CI95 | clustered CI95 | Δp1 | McNemar p | verdict |
|---|---|---:|---:|---:|---|---|---:|---:|---|
| t0p0 | lag | 2.3375 | 2.3625 | **+0.025** | [−0.33,+0.38] | [−0.15,+0.20] | +0.0875 | 0.039 | **SURVIVES** |
| t0p0 | backfill | 2.3375 | 2.2625 | **−0.075** | [−0.40,+0.25] | [−0.21,+0.10] | +0.0750 | 0.109 | **MARGINAL** |
| t0p5 | lag | 2.0875 | 2.2250 | +0.138 | [−0.20,+0.49] | [−0.05,+0.30] | +0.1125 | 0.012 | SURVIVES |
| t0p5 | backfill | 2.0875 | 2.2000 | +0.113 | [−0.23,+0.45] | [−0.09,+0.34] | +0.1125 | 0.023 | SURVIVES |
| t1p0 | lag | 2.0875 | 2.2000 | +0.113 | [−0.24,+0.46] | [−0.13,+0.33] | +0.0625 | 0.227 | SURVIVES |
| t1p0 | backfill | 2.0875 | 2.2000 | +0.113 | [−0.20,+0.43] | [−0.04,+0.28] | +0.0750 | 0.146 | SURVIVES |

### Column glossary

- **temp** — bundle temperature suffix. `t0p0` = greedy (temp=0; the model's currency
  and the primary axis). `t0p5`/`t1p0` = sampled-stream (temp=0.5/1.0), where the
  "correction token" is the *sampled* token, not greedy argmax — a robustness
  dimension, NOT a clean replication of the greedy cells.
- **mode** — KV-window reuse model. `lag` = stale hidden every step (the KV window
  becomes a consistent one-position lag of baseline; the lead's specified method).
  `backfill` = current step's slot is stale, prior slots are backfilled with the
  TRUE hidden after drafting (models a folded verifier that commits accepted
  positions with true hiddens; added per GATE 1). Both feed the correction token
  (`target_tokens[step]`) unchanged as the anchor embedding.
- **base E[a|5]** — baseline mean accepted-prefix length per 5-token draft block,
  drafting from the true hidden `main_hidden[pos]` (`reuse_mode="none"`). This is
  the retained acceptance metric `spec_speedup_model` uses.
- **reuse E[a|5]** — same metric under the reuse substitution (`main_hidden[pos−1]`).
- **Δ** — `reuse − base`, the mean per-(prompt,step) delta of accepted-prefix length.
  Positive = reuse drafts longer matching prefixes than baseline. **This is the
  decision-rule currency.**
- **per-step CI95** — 95% paired bootstrap CI on the mean Δ, resampling the 80
  individual (prompt,step) units with replacement (10k reps). Treats steps as
  independent (optimistic).
- **clustered CI95** — 95% paired bootstrap CI resampling the 10 *prompts* with
  replacement, then averaging each prompt's per-step Δ. Accounts for within-prompt
  correlation; **this is the CI that should govern inference.**
- **Δp1** — `reuse − base` first-token (position-1) match rate, where p=1 =
  `draft[0]==target[0]`. A secondary signal; see Verdict for why it must not be
  over-read (it masks a per-position reshaping).
- **McNemar p** — exact two-sided binomial McNemar p-value on the paired p=1
  match/mismatch table (b = baseline-correct & reuse-wrong; c = baseline-wrong &
  reuse-correct). p<0.05 ⇒ significant discordance. Sign of the effect is read
  from c vs b (c≫b ⇒ reuse is favored on p=1).
- **verdict** — two-tier rule (user-specified): `SURVIVES` if Δ≥0; `COLLAPSES` if
  Δ<0 AND per-step CI excludes 0; else `MARGINAL`. No cell collapsed.

## Verdict

**No LARGE collapse observed in any cell.** 5/6 cells SURVIVE; the lone MARGINAL
(t0p0/backfill, −0.075) has a CI that straddles 0. Per the user's two-tier rule the
primary (lag) model is **SURVIVES** at temp=0.

**But this is not a non-inferiority result** (GATE 2, verified): every CI straddles 0, so
"straddling 0" means **no significant harm detected**, not "acceptance preserved." The
corpus is ~10× too underpowered to confirm the model's fragile −0.9% edge: clustered 95%
CI half-widths are ~0.15–0.18 (80% MDE ~0.2–0.4 accepted tokens), vs the ~0.03 E[a|4]
budget that separates the −0.9% edge from baseline. A modest real cost (−0.05 to −0.15)
could be hiding in the noise.

**Per-position reshaping (verified, t0p0):** reuse is not free — it trades early-block
accuracy for late-block accuracy. Match-rate delta by block position:

- lag: `[+0.0875, 0, 0, −0.05, −0.025]`
- backfill: `[+0.075, +0.0375, −0.0125, −0.10, −0.0625]`

Position 1 (and 2 under backfill) improves under reuse; positions 4–5 degrade. Net
E[a|5block] is roughly flat because the early gains offset the late losses — so "p=1
robustly non-negative" should not be read as "reuse helps"; it masks a real per-position
cost. (Physical read: the stale hidden is a smoother context that biases the drafter
toward higher-probability early tokens but loses the specificity for later,
more-context-dependent tokens.)

**Do not over-read the t0p5/t1p0 SURVIVE cells as replications of t0p0/backfill's
MARGINAL:** those are sampled-stream cells (the "correction token" is a sampled token,
not greedy argmax), so they are not clean replications; they show the reuse effect is not
catastrophic across sampling regimes, nothing finer.

## Implication for the model band (scoped per GATE 2)

- The anchor-reuse accounting in `spec_speedup_model.md` is **not invalidated on
  acceptance grounds**: this offline proxy does not show a collapse. The original
  concern ("the drafter cannot draft from stale hidden") is **not borne out as a large
  effect** in any tested regime.
- **Lead 05 (verifier engineering) can proceed on the acceptance axis** — this falsifier
  does not block it — **but** the full realizability of the −0.9% optimistic edge remains
  **unproven and contingent** on verifier-side factors this test does NOT cover:
  1. **verify-produced-hidden equivalence**: the test feeds `main_hidden[pos−1]` from a
     clean greedy decode; a real verify forward produces hiddens over a partially-rejected
     suffix under IQ2XXS, where batched-prefill vs one-at-a-time-decode hiddens may differ.
     (The test assumes they coincide — a scope boundary, GATE 1.)
  2. **residual cycle overhead** (~15–19 ms readback / rollback / first-miss waste the
     model sets to zero).
  3. **a better-powered K=4 acceptance bound** — a prompt-clustered non-inferiority test
     with margin tied to the speedup budget (ΔE[a|4] ≳ −0.03), which needs more/larger
     prompts than the 10 short (~67–165 tok) cells here.
- Net: treat the −0.9% at K=4 exactly as `spec_speedup_model.md` already labels it — the
  **optimistic edge of a band**, now with the *acceptance-axis* risk downgraded from
  "unverified and load-bearing" to "no large collapse seen, but non-inferiority not yet
  established." The pessimistic edge (−18.9%, shipped verifier) is unchanged.

## What was built (reusable)

- `dspark_oracle/measure_acceptance_bundle.py`: `--reuse-mode {none,lag,backfill}` added
  to `measure_bundle`; documented KV-window modeling decision; baseline mode unchanged
  (fidelity gate confirms bit-for-bit reproduction of retained summaries).
- `run_anchor_reuse_falsifier.py`: fidelity gate + both reuse modes + per-step &
  per-prompt-clustered bootstrap CIs + exact McNemar + two-tier verdict; `--temps` sweeps
  t0p0/t0p5/t1p0.
- Artifacts: `falsifier_result.json` (all temps/modes/aggregates/units), `per_prompt_*.csv`,
  `units_<temp>_<mode>.csv`, both codex reviews.

## Open questions (deferred to lead 05 / follow-ups)

- Does a real folded verifier produce hiddens at accepted positions that match
  `main_hidden[pos]` under IQ2XXS? (capture from a forced-rejection verify forward;
  codex GATE-1/GATE-2 #1 experiment.)
- A powered non-inferiority test (more prompts, margin tied to the speedup budget) to
  actually confirm/refute the −0.9% edge rather than "no large collapse."
- The per-position reshaping (early +, late −) is a real drafter behavioral signature of
  stale-hidden input; worth quantifying on real verifier trajectories.
