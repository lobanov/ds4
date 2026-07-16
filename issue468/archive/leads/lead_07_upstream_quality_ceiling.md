# Lead 07 - Native-hidden recoverability on IQ2XXS

Date: 2026-07-11. **Status: resolved & archived 2026-07-16 (moved from `issue468/pending/`).
Result: PIVOT — closed negative (codex-concurred).** Experiment 1 (the crossed FP/IQ2
oracle, run on a teacher-forced common trajectory) verdicted PIVOT; Experiment 2 is
**not warranted**. The full attribution + decision are below (§ Result). The original
proposal (Purpose → One-line verdict) is retained after it as provenance.

## Result — Experiment 1 verdict (2026-07-16)

**PIVOT — the native-vs-IQ2 drafter-acceptance gain is NOT a recoverable hidden-side
effect on IQ2XXS.** Run on a **teacher-forced common trajectory**: the IQ2XXS-vs-native
greedy trajectories diverge (~6% token agreement, no common prefix, no offset), so the
proposal's "common-prefix" alignment does not exist. It was created by teacher-forcing
the IQ2 model onto the FP trajectory via a new `teacher_force` ds4-spec-bench submode
(commit `bb9c01f`) — `ds4_session_argmax()` records the IQ2 argmax (Y_iq2_tf) *before*
`ds4_session_eval(forced)` drives the model through the FP greedy tokens (Y_fp), and
`DS4_DSPARK_DUMP_HIDDEN` captures H_iq2_tf. Fidelity gate PASS (A(D_f32,H_iq2_tf,Y_fp)
p1=0.891, sane band vs ceiling 0.922 / IQ2-native 0.852).

**Setup.** 59 prompts (lead3_corpus, codealpaca/dolly/jsonex × 0080–0099; 1 skipped —
too few anchors for the block), 5554 anchors, common FP trajectory. D_f32 torch drafter;
anchors always = Y_fp (the FP-trajectory context tokens). H_fp/Y_fp from the retained
Lead 04 Modal captures (not re-captured); H_iq2_tf/Y_iq2_tf from the teacher-force.
ds4 chat-templated prompt length == FP prompt_tokens for 0/60 (clean alignment);
Y_iq2_tf==Y_fp at 92.5% on the common trajectory (per-step label-drift 7.5%; the ~94%
full-trajectory divergence is compounding, not large per-step drift).

**The 2×2 (common FP trajectory):**

| cell | p=1 [CI95] | E[a\|4] |
|---|---|---|
| ceiling  `A(D_f32, H_fp,     Y_fp )`    | **0.849** [.828,.869] | **2.786** |
| baseline `A(D_f32, H_iq2_tf, Y_iq2_tf)` | 0.842 [.820,.863] | 2.660 |
| hidden   `A(D_f32, H_fp,     Y_iq2_tf)` | 0.835 [.812,.858] | 2.666 |
| label    `A(D_f32, H_iq2_tf, Y_fp )`    | 0.833 [.810,.855] | 2.660 |

**Attribution** (prompt-clustered bootstrap, n=59):
- lift (ceiling − baseline) p1 = **+0.007, CI[-0.002, +0.016]** — includes 0. Negligible.
- hidden-side main effect p1 = +0.005; label-side main effect p1 = +0.002 (both ~0).
- interaction p1 = +0.023; at E[a\|4] the interaction dominates (ceiling 2.786 is an
  outlier — crossing *either* H or Y drops E[a\|4] to ~2.66).
- **recoverable hidden-side (hidden − baseline = FP hidden + deployable IQ2 labels) =
  −0.0068, CI[-0.0137, −0.0004]** — significantly *negative*: FP-like hiddens would
  *hurt* (~0.7 pp) against the IQ2 target's own labels.
- IQ2-native reference (deployable IQ2 trajectory) p1 = 0.792.

**Why PIVOT (neither):**
1. On a common trajectory, FP and IQ2 hiddens are equivalent for the drafter (p1 lift
   +0.007, CI incl 0).
2. The recoverable hidden-side effect (the deployable case: FP hidden with IQ2 labels)
   is ~0 in p1 and significantly *negative* (−0.0068) — there is no hidden-side lever.
3. The FP ceiling's block advantage (E[a\|4] 2.786) requires *both* H_fp and Y_fp (the
   self-consistent diagonal) — an interaction / self-consistency effect, not a
   hidden-side main effect. Y_fp (the native-FP argmax) is **undeployable on IQ2XXS**.
4. The residual native-vs-IQ2 p1 gap (ceiling 0.849 vs IQ2-native 0.792 ≈ 0.057) is
   target-trajectory difficulty (the IQ2 model, run freely, generates a
   harder-to-draft trajectory), not hidden precision.

Per Lead 07's rule ("neither cell much better than IQ2 baseline → stop"), **Experiment 2
(the body-side adapter recovery test) is not warranted; Lead 07 closes negative.**

**Codex gate (gpt-5.5 xhigh — retained: `issue468/artifacts/dspark_codex_reviews/
2026-07-16_lead07_crossed_oracle_gate.md`).** C1–C4 sound; codex reproduced dolly_0090
exactly (baseline 0.90625/hidden 0.9375/label 0.890625/ceiling 0.921875, n=64) +
recomputed anchor-weighted lift +0.0043 CI[-0.0040,+0.0128]. Independently re-verified
by the dispatcher (dolly_0090 match + anchor-weighted lift match + the −0.0068 finding).
**Caveat (P1, load-bearing):** the PIVOT is conditional on the retained Lead 04 FP
captures being faithful (their `mhc_post` fidelity was never algebraically proven — the
F16-on-FP anomaly). Robustness: a subtle F32-masked error cannot manufacture a large
*positive* recoverable effect, so the qualitative PIVOT holds; the decisive resolution
(a same-stack native-FP recapture through ds4) is infeasible — ds4 only supports
IQ2XXS/Q2_K experts — so it reduces to the Lead-04 hidden-capture-fidelity follow-up.

**Resolves / leaves open.** *Resolved:* the Lead-04 native-hidden ceiling is NOT an
IQ2XXS-recoverable hidden-precision effect; the drafter is not hidden-input-limited
relative to FP on a common context; the +5–15% native ceiling is a target-trajectory +
FP-self-consistency artifact. *Open (other leads):* the runtime +20% remains Lead 08
(fused verify kernel); the target-trajectory difficulty (~5 pp) is a target/quantization
property, not a drafter-fixable hidden-side lever.

**Artifacts.** `issue468/artifacts/lead07_crossed_oracle/`: `run_crossed_oracle.py`,
`crossed_oracle_result.json` (cells, attribution, per-source, per-prompt rows),
`fidelity_gate.py`, `prep_recapture.py`, `verify_recapture.py`. Raw captures
(`fp_captures_raw/`, `tf_dump_full/`, `force_tokens/`) gitignored (reproducible from
Modal + `ds4-spec-bench teacher_force`). Commits `bb9c01f`/`37789b5`/`49b9284`.

Purpose: resolve the **remaining drafter-quality question left open after Stage 2 and
Lead 04**:

> The native DeepSeek-V4-Flash-DSpark target exposes a large **float32 acceptance
> ceiling** for the current drafter, but the current **F16/IQ2XXS deployment cannot
> use it**. How much of that native-vs-IQ2 gap is a real hidden-state lever that can
> be recovered on the local IQ2XXS path, and by what deployable route, if any?

This is **not** a general "upstream quality ceiling" program anymore. The broader
questions were already reduced by prior leads:

- drafter-weight precision is closed negative (`dspark_quantization_ceiling.md`);
- partial higher-precision target proxy is measured (`stage1_tap_precision.md`);
- non-expert head LoRA on IQ2XXS is closed negative (`stage2_finetune_result.md`);
- native served-precision target hiddens were measured and show a real float32
  ceiling, but attribution and recoverability on IQ2XXS remain unresolved
  (`spec_speedup_model.md`, Lead 04 Phase B/C).

## Why this lead still exists

The dossier now says two things at once:

1. **Current local deployment remains below baseline.**
   On IQ2XXS, realistic cycle-jump K=4 is ~0.982x with the current drafter
   (`summaries/spec_speedup_model.md`).
2. **A large native-hidden ceiling exists.**
   Against native FP4/FP8 target hiddens, the same drafter at float32 reaches
   p1 ~0.855 and cycle-jump `E[a|4] = 2.741`, implying roughly **+8-10%** at K=4.
   But the current **F16 deployment cannot use that gain**: FP hiddens + F16 drafter
   are worse than IQ2XXS.

So the remaining question is no longer "is there any upstream ceiling?" The remaining
question is:

> Is the measured native-hidden gain mostly a **hidden-state precision** effect that
> can be transferred back onto IQ2XXS by a deployable adaptation, or is it mostly
> label/trajectory drift or an undeployable float32-only effect?

This lead should advance the specific open caveat retained in
`summaries/spec_speedup_model.md`: a **crossed FP/IQ2 oracle** is still needed to
separate hidden precision from label drift before attributing the native ceiling to
recoverable target-hidden quality.

## What is already closed

These are **not** in scope for Lead 07 anymore:

1. **Vendored drafter weight precision ceiling.**
   `Q4_K ~= F16` for acceptance; no material gain from removing local GGUF
   quantization noise.
2. **Generic "is the target in the top-k neighborhood?" on the tiny exactness set.**
   Stage 0 already showed shallow p=1 misses; that justified Stage 2 and does not
   need to be re-proved as a standalone lead.
3. **Generic non-expert head-only fine-tuning on IQ2XXS labels.**
   Stage 2 already ran the bounded PoC with the specified loss and found significant
   harm on held-out p=1.
4. **The broad claim that missing BF16 / original checkpoints are the main blocker.**
   The official upstream DSpark artifact is already the relevant native served-precision
   checkpoint; no additional BF16 checkpoint is assumed to exist.

## Scope

Lead 07 now has **two sequential experiments only**:

1. **Crossed FP/IQ2 oracle on common prefixes** - attribution.
2. **One deployability-focused recovery test** - only if experiment 1 says the
   hidden-state lever is real.

Anything broader belongs in a different lead.

## Definitions

`A(D, H, Y)` = drafter acceptance when drafter `D` consumes hidden-state trajectory
`H` and is judged against target labels `Y`.

Use these concrete instances:

- `D_f16`: current deployable drafter path (the local F16/Q4_K-served drafter behavior).
- `D_f32`: float32 evaluation path used in the retained torch/oracle studies.
- `H_iq2`, `Y_iq2`: IQ2XXS hidden states and greedy labels.
- `H_fp`, `Y_fp`: native DeepSeek-V4-Flash-DSpark hidden states and greedy labels
  captured in Lead 04.

The important currently measured cells are:

- `A(D_f32, H_iq2, Y_iq2)` — current float32 reference on IQ2XXS.
- `A(D_f32, H_fp,  Y_fp )` — native float32 ceiling from Lead 04.
- `A(D_f16, H_fp,  Y_fp )` — deployability failure case from Lead 04.

The missing attribution cells are:

- `A(D_f32, H_fp,  Y_iq2)` — hidden-side effect with IQ2 labels.
- `A(D_f32, H_iq2, Y_fp )` — label/trajectory effect with IQ2 hiddens.

## Experiment 1 - Crossed FP/IQ2 oracle on common prefixes

### Question

How much of the native-vs-IQ2 float32 gain comes from:

1. **hidden/input-side precision** (`H_fp` helps even when labels stay IQ2), versus
2. **label/trajectory drift** (`Y_fp` is easier / different even when hiddens stay IQ2)?

This is the load-bearing unresolved caveat from Lead 04.

### Inputs

- Reuse Lead 04 native captures and the retained IQ2XXS captures.
- Restrict analysis to **common-prefix / aligned anchors** only, so comparisons are
  not contaminated by unrelated trajectory divergence.
- Primary carrier: `D_f32`, because Lead 04 established that `D_f16` on `H_fp`
  is numerically unstable and therefore not useful for attribution.

### Acceptance cells to measure

For the same aligned anchors:

| acceptance cell | interpretation |
|---|---|
| `A(D_f32, H_iq2, Y_iq2)` | local baseline |
| `A(D_f32, H_fp,  Y_iq2)` | hidden-side effect only |
| `A(D_f32, H_iq2, Y_fp )` | label-side effect only |
| `A(D_f32, H_fp,  Y_fp )` | native float32 ceiling |

### Metrics

- p=1 acceptance delta for each cell.
- If alignment depth supports it: prefix metrics `E[a|4]`, `S(4)`, and a cycle-jump
  replay on common-prefix slices.
- Input main effect, label main effect, and interaction.
- Per-source breakdown (codealpaca / dolly / jsonex).

### Decision rule

- **If `A(D_f32, H_fp, Y_iq2)` captures most of the `H_fp,Y_fp` lift:**
  the gain is genuinely on the hidden/input side; a recovery path aimed at
  hidden adaptation is justified.
- **If `A(D_f32, H_iq2, Y_fp)` captures most of the lift:**
  the measured native win is mostly label/trajectory drift, not an IQ2-hidden
  recoverable lever; Lead 07 should stop.
- **If both matter materially:**
  the gain is mixed; any recovery attempt must be framed honestly as partial.
- **If neither cell is much better than IQ2 baseline:**
  the apparent native ceiling is not actionable for the local path; stop.

## Experiment 2 - One deployability-focused recovery test

### Gate

Run this only if Experiment 1 shows a **real hidden-side lever** worth recovering.

### Goal

Test **one concrete route** for making some of the hidden-side gain usable on the
IQ2XXS local path. This is not another generic fine-tune sweep. It is a targeted
recoverability test chosen using the crossed-oracle result.

### Preferred route order

Choose exactly one:

1. **Body-side adapter with native auxiliary teacher** (preferred default).
   Train a small adapter on the IQ2XXS path, with IQ2 labels primary and native
   hidden/label information as auxiliary supervision. This directly tests the
   best Lead-04-derived hypothesis: a deployable adaptation that makes IQ2XXS
   features more native-like without requiring float32 deployment.
2. **Full-HC residual probe before adaptation.**
   If the crossed oracle suggests the mean-HC reduction may be leaving signal on the
   table, run a cheap locked-split probe comparing full `[4,4096]` HC residual
   against mean-HC features. If full-HC does not beat mean-HC, do not pursue that
   route further.
3. **Hidden dequantizer / mapper diagnostic.**
   Only if the paired `H_iq2 ↔ H_fp` data suggests a simple learned mapping is
   plausible. This is primarily diagnostic, not a product path.

### Success criterion

Any route is only interesting if it shows both:

1. **Held-out p=1 gain over the IQ2XXS deployable baseline**, and
2. **A credible move in prefix/cycle metrics toward the native-hidden ceiling**, not
   just a tiny first-token gain with no block-level effect.

If it cannot clear that bar, record the native-hidden ceiling as **real but not
recoverable on the current deployment path** and close the lead.

## Non-goals

Lead 07 does **not** include:

- a new broad fine-tuning campaign;
- expert tuning as an open-ended workstream;
- a search for hypothetical BF16/original DSpark checkpoints;
- verifier engineering (Lead 06);
- scheduler-only work (Lead 02);
- generic corpus expansion unless needed to stabilize the crossed-oracle estimate.

## Deliverables

1. A compact summary answering:
   - how much of the native-vs-IQ2 gain is hidden-side vs label-side;
   - whether that gain appears recoverable on IQ2XXS;
   - whether the local drafter-quality axis remains open or should be closed.
2. Machine-readable artifact(s) for the crossed oracle.
3. If Experiment 2 runs, one bounded recovery result with a clear stop/proceed verdict.

## Exit conditions

- **Proceed:** crossed oracle says the hidden-side lever is real, and the bounded
  recovery test shows deployable held-out improvement.
- **Stop:** crossed oracle says the native gain is mostly label/trajectory drift, or
  the bounded recovery test fails to pull IQ2XXS meaningfully toward the native
  ceiling.

## One-line verdict (proposal, pre-execution — superseded by § Result above)

Lead 07 was justified as a **narrow post-Lead-04 recoverability lead**: first attribute
the native-hidden gain with a crossed FP/IQ2 oracle, then test at most one deployable
hidden-side recovery path. **Outcome (§ Result): the oracle PIVOTs — no recoverable
hidden-side lever; the lead closes negative; Experiment 2 does not run.**
