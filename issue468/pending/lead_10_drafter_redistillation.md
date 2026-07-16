# Lead 10 — Drafter re-distillation for the IQ2XXS target (soft labels)

Date: 2026-07-16. Status: **proposed (not yet started).**

## Purpose

Resolve the one drafter-quality question left open after Lead 07 and Stage 2:

> The current dspark drafter was distilled against the **native (FP) target** and reaches
> p1 ≈ 0.79 when deployed on the **IQ2XXS** target. Is that 0.79 a **distribution mismatch**
> the drafter could be re-trained out of, or a **capacity / information ceiling** no drafter
> clears on the IQ2 trajectory?

This is the surviving drafter-quality axis. Lead 07 closed the *hidden-side* route (the native
ceiling is not a recoverable hidden-precision effect); Stage 2 closed the *head-only hard-label*
route. The untested route is **full-body re-distillation on IQ2 captures with soft labels** —
the same method that produced the current drafter, re-targeted to the IQ2 target.

## Why this lead exists (the gap after Lead 07 + Stage 2)

- **Lead 07 (archived, negative):** on a common (teacher-forced) trajectory, FP and IQ2
  hiddens are *equivalent* for the drafter (p1 lift +0.007, CI incl 0); the recoverable
  hidden-side effect is ~0 (actually −0.007). The native ceiling is a target-trajectory +
  FP-self-consistency artifact, not hidden precision. **The drafter is NOT hidden-input-limited.**
- **Stage 2 (closed negative):** non-expert **head** LoRA on IQ2XXS labels (**hard labels**)
  caused significant held-out harm.
- **What remains:** whether the drafter (trained on the FP target's distribution) is mismatched
  to the IQ2 target's distribution — and whether full-body re-distillation on **soft labels**
  (the IQ2 target's full next-token distribution) can raise the 0.79. The ~7.5% per-step
  Y_iq2≠Y_fp disagreement is a real distribution gap a re-distillation could in principle close.

The honest framing: this is a **genuine but uncertain bet**. The head-only, hard-label version
already failed (Stage 2). Full-body + soft labels is the strictly stronger, untested variant —
but the prior is negative, and the IQ2 target is a quantization-degraded distribution (a noisier
teacher), which cuts against easy headroom.

## What is already closed (not in scope)

1. **Hidden-side recovery** — Lead 07 (the native ceiling is not a recoverable hidden-precision lever).
2. **Head-only hard-label adaptation** — Stage 2 (significant held-out harm).
3. **Drafter weight precision** — Q4_K ≈ F16 for acceptance; no gain from removing quantization noise.
4. **The native-FP teacher** — distilling on FP captures reproduces the current drafter (it is
   already at the 0.85 FP ceiling); the teacher here is the **IQ2 target**, not FP.

## Scope

Lead 10 has **one experiment only**: full-body, soft-label re-distillation of the drafter on
IQ2XXS captures, validated held-out via the torch oracle. No sweep of architectures, no
expert tuning, no FP-target distillation.

## Definitions

- **D**: the dspark MTP drafter (body + head), the `drafter_body`/`drafter_head` torch modules.
- **The teacher**: the **IQ2XXS target**, accessed through its captures — H_iq2 (layers 40/41/42
  mean hidden, the drafter input) and the IQ2 next-token distribution (top-128 logits).
- **Soft labels**: the IQ2 target's full next-token distribution (top-k logits/probs); the loss
  is KL(drafter ‖ target). This is how the drafter was originally distilled — re-targeting to
  IQ2 is the apples-to-apples re-distillation.
- **Hard labels**: the IQ2 greedy token Y_iq2; cross-entropy. (Stage 2's carrier; not used here
  except as a fallback/comparison.)
- **Baseline**: the current drafter's IQ2-native acceptance, p1 ≈ 0.79 (the deployable reference).
- **Ceiling**: A(D_f32, H_fp, Y_fp) ≈ 0.85 (the current drafter on FP; an input ceiling, not a
  capacity ceiling — a better drafter might exceed even this).

## The experiment

### Question

Can a full-body, soft-label re-distillation of D on IQ2 captures beat the current 0.79 by a
decision-grade margin on **held-out** IQ2 prompts?

### Inputs (the teacher signal)

- **240 training prompts + 60 held-out eval prompts**, 3 sources (codealpaca / dolly /
  jsonex), freshly sampled (seed 42) from `issue468/data/corpus_source/`, constructed per
  `data/distill_corpus/manifest.json`:
  - codealpaca: `prompt` (instruction);
  - dolly: `instruction` + `context`;
  - jsonex: `instruction` + `text`.
  - dolly context capped at <1500 words; 80/20 train/eval split per source.
- Per prompt, captured via `ds4-spec-bench` (`mode=argmax`, `gen_tokens=128`, ds4 chat template):
  - **H_iq2** — `--dump-hidden-dir` (the existing mechanism);
  - **Y_iq2** — the greedy tokens (the bench output);
  - **IQ2 top-128 logits** — `--dump-logprobs` (the soft-label target). **Phase 0 must confirm
    the bench exposes this; if not, augment it** (mirror the Lead-04 Q2 top-128 capture).

### Training approach

- **Full-body**: fine-tune the drafter body + head (not head-only LoRA — that is Stage 2, closed).
- **Soft labels**: KL(drafter distribution ‖ IQ2 top-128 teacher distribution) per anchor.
- **Precision**: train at **F32 or mixed-precision (F32 master / F16 compute)**, then cast to F16
  for the dspark GGUF. Rationale: the drafter is **dtype-invariant on Q2 hiddens** (Lead 04:
  f32-Q2 == f16-Q2, drafts identical), so F32-train / F32-validate (torch) / F16-deploy all agree
  on IQ2 inputs — no precision gap (the FP-hidden F16 anomaly does not apply here). "Train F16" is
  equivalent on Q2 but risks gradient underflow; F32/mixed is the stable choice.
- **Fidelity gate (mandatory before any powered number):** the trained drafter, loaded into the
  torch oracle, must (a) reproduce the current 0.79 baseline bit-token-for-token on a known-good
  held-out slice BEFORE training (sanity), and (b) post-training, agree at F16 and F32 on Q2 (the
  invariance must survive fine-tuning).

### Validation

- Run the **torch oracle (D_f32)** on the **60 held-out eval** captures → p1 (and E[a|4]).
- Compare to the **0.79 baseline** (current drafter) + the **0.85 ceiling**.
- **Prompt-clustered bootstrap CI** (resample prompts, not anchors). Per-source breakdown
  (codealpaca / dolly / jsonex) — report stratified, do not pool into one headline.

### Metrics

- Primary: held-out **p1** (first-draft acceptance) Δ vs the 0.79 baseline.
- Secondary: **E[a|4]** (the cycle-jump; the runtime S(4) is out of scope — that is Lead 08).

### Decision rule (locked before measuring)

With 60 held-out prompts and σ_Δ ≈ 0.03 (from the Lead-07 crossed-oracle per-prompt Δ spread),
the eval is powered to detect **+2 pp conclusively** (≈14 prompts needed at 80% power; 60 is
comfortable) and ≈ +1 pp at the margin:

- **PROCEED:** held-out Δp1 ≥ **+2 pp** (p1 ≥ 0.81) AND the prompt-clustered CI excludes 0 (and
  ideally excludes +1 pp). → integrate the re-distilled drafter, re-bench the full M3 stack,
  update `spec_speedup_model.md`. Note: even PROCEED does not reach +20% alone (that remains
  Lead 08); a +2 pp p1 gain is a modest acceptance improvement.
- **MARGINAL:** +1 to +2 pp, or CI straddling the thresholds. → report honestly; do not integrate
  without a stronger result; consider a soft-vs-hard ablation or more data.
- **STOP:** Δp1 ≤ +1 pp OR held-out harm (negative, like Stage 2). → the drafter-quality axis is
  closed; the 0.79 is a capacity/information ceiling, not a mismatch; future effort goes to
  Lead 08 (verify) + the target-trajectory property.

## Power / sizing (why 60 eval is enough)

Prompt-clustered (the unit is the prompt, not the anchor — positions within a prompt are
correlated). Per-prompt Δ SD ≈ 0.03 (same-trajectory paired-drafter noise, from the Lead-07
crossed-oracle cells; ~4× smaller than Lead-04 Phase B's cross-trajectory σ because the
trajectory cancels). At 80% power / one-sided α=0.05: **+2 pp needs ≈14 prompts, +1 pp needs
≈56**. 60 eval covers the +1 pp bar. (The binomial floor is *above* the empirical σ → adding
anchors-per-prompt past ~100 buys nothing; more prompts is the only lever.) The ~240-prompt
training set matches Stage 2's scale.

## Codex gates + fidelity gate

1. **Codex gate A (setup):** the soft-label capture augmentation (if added), the KL-loss
   indexing/normalization over the top-128 (padding, the masked positions), the F32/mixed
   precision, and the train/eval disjointness — BEFORE trusting any trained number.
2. **Fidelity gate:** trained drafter reproduces the 0.79 baseline pre-training-style + F16==F32
   on Q2 post-training.
3. **Codex gate B (verdict):** the PROCEED/MARGINAL/STOP vs the data — honestly scoped
   (significance ≠ a large gain; +2 pp is modest), per-source stratified.

## Non-goals

- Another head-only or LoRA-only adapter (Stage 2, closed).
- Distillation from the FP/native target (reproduces the current drafter).
- Expert tuning; drafter architecture search; a bigger drafter body.
- The verify side (Lead 08) or the scheduler (Lead 02) — this is an acceptance/quality study.
- The +20% runtime target (out of scope; a +2 pp acceptance win is a modest contributor).

## Deliverables

1. The soft-label IQ2 capture (240 train + 60 eval): H_iq2 + Y_iq2 + IQ2 top-128.
2. The trained full-body drafter (F16 GGUF) + the training config/checkpoint.
3. The held-out p1 (+ E[a|4]) vs the 0.79 baseline + the 0.85 ceiling, per-source, with CIs.
4. The PROCEED/MARGINAL/STOP verdict + the codex-gate artifacts.

## Exit conditions

- **PROCEED:** +2 pp held-out p1 (CI excl 0) → integrate + re-bench the M3 stack.
- **STOP:** ≤ +1 pp or harm → the drafter-quality axis closes; 0.79 is a capacity/information
  ceiling. Effort moves to Lead 08 (the runtime +20%) + the target-trajectory property.

## One-line verdict (proposal)

Lead 10 tests whether the IQ2-deployment acceptance deficit is a **distribution mismatch** a
full-body soft-label re-distillation can close — the one untested drafter-quality route after
Lead 07 (hidden-side, dead) and Stage 2 (head-only hard-label, dead). It is a real but uncertain
bet with a negative prior; the decisive, powered test needs only ~60 held-out prompts.

## Worklog

### 2026-07-16 — corpus + capture input prepared (Phase 0 setup, pre-execution)

- Sampled 100/source (codealpaca `prompt`; dolly `instruction`+`context` capped <1500w;
  jsonex `instruction`+`text`) from `issue468/data/corpus_source/`, seed 42 → 300 prompts,
  80/20 split → `issue468/data/distill_corpus/{train,eval}/` (240 train / 60 eval) +
  `manifest.json`. Quality: capture-safe (max ~538w ≈ 700 tokens), 3 diverse task types, no
  degenerate/duplicates; caveat = codealpaca homogeneity (short coding prompts, inherent).
- Built `issue468/data/distill_corpus/capture_config.jsonl` (300 entries, `mode=argmax`,
  `gen_tokens=128`, ds4 chat template, `frontier_tokens` placeholder for `--rewrite-frontier`).
- **Open Phase-0 item:** confirm `ds4-spec-bench` exposes `--dump-logprobs` for the IQ2 top-128
  soft-label target; if not, augment the bench (mirror Lead-04 Q2 top-128 capture) before the run.
- Decision (user): **soft labels**; train F16-equivalent (F32/mixed, dtype-invariant on Q2),
  validate via the torch oracle (D_f32) on the 60 held-out eval.
