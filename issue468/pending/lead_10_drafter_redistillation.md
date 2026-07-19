# Lead 10 — Soft-label dense-LoRA drafter re-distillation for IQ2XXS (experts frozen)

Date: 2026-07-16. Status: **proposed (not yet started).** *Reframed 2026-07-16 after the
drafter parameter-count finding (it is a 19.85B MoE — full fine-tune is infeasible at any sane
corpus; the experiment is dense-LoRA with the routed experts frozen, not "full-body").*

## Purpose

Resolve the one drafter-quality question left open after Lead 07 and Stage 2:

> The current dspark drafter was distilled against the **native (FP) target** and reaches
> p1 ≈ 0.79 when deployed on the **IQ2XXS** target. Is that 0.79 a **distribution mismatch**
> the drafter could be re-trained out of, or a **capacity / information ceiling** no drafter
> clears on the IQ2 trajectory?

This is the surviving drafter-quality axis. Lead 07 closed the *hidden-side* route; Stage 2
closed the *head-only hard-label* route. The untested route is **soft-label dense-LoRA
re-distillation on IQ2 captures** — re-targeting the drafter to the IQ2 target's distribution.

## Why this lead exists (the gap after Lead 07 + Stage 2)

- **Lead 07 (archived, negative):** on a common (teacher-forced) trajectory, FP and IQ2
  hiddens are *equivalent* for the drafter (p1 lift +0.007, CI incl 0); the recoverable
  hidden-side effect is ~0 (actually −0.007). The native ceiling is a target-trajectory +
  FP-self-consistency artifact, not hidden precision. **The drafter is NOT hidden-input-limited.**
- **Stage 2 (closed negative):** non-expert + head LoRA on IQ2XXS labels (**hard labels**)
  caused significant held-out harm.
- **What remains:** whether the drafter (trained on the FP target's distribution) is mismatched
  to the IQ2 target's distribution — and whether **soft labels** (the IQ2 target's full
  next-token distribution, the loss the drafter was originally distilled with) let a dense-LoRA
  re-target generalize where Stage 2's hard-label version did not.

The honest framing: this is a **genuine but uncertain bet**. The hard-label dense/head LoRA
version already failed (Stage 2). Soft-label dense-LoRA is the untested variant — but the prior
is negative, and the IQ2 target is a quantization-degraded distribution (a noisier teacher),
which cuts against easy headroom.

## Drafter architecture (load-bearing for the training design)

Per `issue468/inventories/dsv4_flash_dspark_model.md` (the authoritative model inventory,
derived from the staged HF checkpoint), the dspark drafter is **3 MTP layers (`mtp.0/1/2`)
that mirror the decoder** — not a small dense model:

- **MoE: 256 routed experts + 1 shared expert per layer, 6 activated/token**, `moe_intermediate_size=2048`. The routed experts are **MXFP4**; the shared expert + attention + `main_proj` are FP8; norms/head/`markov_head`/`confidence_head` are BF16; the HC-mixing weights are F32.
- **~19.35B of the ~19.85B total params sit in the routed experts** (`ffn_{up,gate,down}_exps`, `[4096,2048,256]`); the dense parts (sparse-MLA attention, `main_proj [4096,12288]`, the heads, the HC mixing, the router) are only ~0.5B.
- Drafter input = `main_hidden [12288] = concat(mean(hc_ffn_post[40/41/42]))` (the mean over the 4 HC components, per layer, ×3).

**Implication for training:** full fine-tune is off the table. At 30k training anchors with
6 active/token, each ~25M-param routed expert sees only **~700 gradient updates** (30k × 6 / 256)
→ guaranteed per-expert overfitting; and MXFP4 training needs dequant→train→requant. So the
**routed experts are frozen** (as in Stage 2). The trainable surface is the **dense parts** via
LoRA — the always-active sparse-MLA attention, `main_proj`, the heads (`markov_head`,
`confidence_head`), the norms, the HC mixing, and optionally the shared expert (always-active,
sees all 30k anchors).

## What is already closed (not in scope)

1. **Hidden-side recovery** — Lead 07.
2. **Hard-label non-expert+head LoRA** — Stage 2 (significant held-out harm). (This lead's delta over Stage 2 is **soft labels** + a deliberate LoRA-target/rank choice.)
3. **Drafter weight precision** — Q4_K ≈ F16 for acceptance.
4. **The native-FP teacher** — distilling on FP captures reproduces the current drafter (already at the 0.85 FP ceiling); the teacher here is the **IQ2 target**.

## Scope

Lead 10 has **one experiment only**: soft-label **dense-LoRA** re-distillation of the drafter on
IQ2XXS captures (routed experts frozen), validated held-out via the torch oracle. No full
fine-tune, no expert tuning, no FP-target distillation.

## Definitions

- **D**: the dspark MTP drafter (`drafter_body`/`drafter_head` torch modules); architecture per `inventories/dsv4_flash_dspark_model.md`.
- **Dense-LoRA**: LoRA adapters on the drafter's always-active dense parts (attention + `main_proj` + heads + HC mixing, optionally the shared expert); routed experts **frozen**.
- **The teacher**: the **IQ2XXS target**, via its captures — H_iq2 (the `main_hidden` drafter input) + the IQ2 next-token distribution (top-128 logits).
- **Soft labels**: the IQ2 target's full next-token distribution; loss = KL(drafter ‖ target). This is the loss the drafter was originally distilled with.
- **Hard labels**: the IQ2 greedy token Y_iq2; cross-entropy (Stage 2's carrier; not primary here).
- **Baseline**: current drafter IQ2-native acceptance, p1 ≈ 0.79.
- **Ceiling**: A(D_f32, H_fp, Y_fp) ≈ 0.85 (an input ceiling, not a capacity ceiling).

## The experiment

### Question

Can a soft-label dense-LoRA re-distillation of D on IQ2 captures beat the current 0.79 by a
decision-grade margin on **held-out** IQ2 prompts — generalizing where Stage 2's hard-label
LoRA did not?

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
  - **IQ2 top-128 logits** — `--dump-logprobs` (the soft-label target). **Capture mechanism
    proven:** Lead 11's unified capture (`issue468/artifacts/lead11_unified_capture/`) already
    produced H_iq2 + Y_iq2 + IQ2 top-k for the **lead3 corpus** (`lead3_h.bin` +
    `lead3_logprobs.jsonl`; 60 prompts / 5,811 anchors; variable-length record format decoded,
    0 h.bin↔logprobs mismatches). Running it on the **distill_corpus** (240 train + 60 eval) is
    now a plain bench invocation, not an R&D item.

### Phase 0 — LoRA target selection (gradient probe + ablation)

The dense LoRA targets are **selected empirically, not guessed**:

1. **Gradient-magnitude probe (cheap, no training).** Load the D_f32 drafter with gradients
   enabled; on a few hundred IQ2 anchors (H_iq2 → drafter → draft distribution), compute the
   loss vs the IQ2 target (**KL on the IQ2 top-128 — directly available in Lead 11's unified
   capture** `lead3_logprobs.jsonl` for the lead3 corpus; cross-entropy on Y_iq2 as a fallback
   proxy) and backprop
   once. For each dense candidate weight W, record **‖∇W‖ / ‖W‖** (normalized so targets at
   different depths are comparable). Rank. The high-ratio targets are the LoRA candidates.
   - Candidates (per the inventory): per layer — `main_proj`, the sparse-MLA attention
     (`wq_a/wq_b/wkv/wo_a/wo_b`), the HC mixing (`hc_attn_fn`/`hc_ffn_fn`), `ffn.gate`,
     `ffn.shared_experts`, the norms; top-level — `hc_head_fn`, `markov_head`,
     `confidence_head`. (Routed experts are frozen — read their gradients only as a reference.)
2. **Per-target LoRA ablation (confirmatory).** Train a separate small LoRA on each top-gradient
   candidate (short run, held-out p1 measured). Pick the winners for the full Lead-10 LoRA.

**Why:** the IQ2-mismatch location is not obvious — Lead 07 showed the drafter is
hidden-input-invariant at the acceptance level, so the remaining signal could be in the input
projection, the context modeling, or the output heads. The probe tells us where before we
commit. **Caveat:** gradient magnitude ≠ generalization (a target can have signal and still
overfit, like Stage 2's head-LoRA) — the ablation confirms which winners generalize.

**Secondary read:** if NO dense target carries a meaningful gradient, the IQ2 mismatch is weak
→ temper the prior (a pre-STOP hint before spending the full training budget).

### Training approach

- **Routed experts frozen; dense-LoRA on the rest.** Full fine-tune is infeasible (per-expert
  overfitting at ~700 updates/expert; MXFP4 train complexity). LoRA targets: **selected by the
  Phase-0 probe above** from the dense candidates (the attention, `main_proj`, the heads, the HC
  mixing, optionally the shared expert). This is the feasible trainable surface (LoRA adapters =
  millions of params, not 19.85B).
- **Soft labels**: KL(drafter distribution ‖ IQ2 top-128 teacher distribution) per anchor (the
  proper MTP-distillation loss, re-targeted to IQ2).
- **Precision**: train at **F32 or mixed-precision (F32 master / F16 compute)**, then cast to
  F16 for the dspark GGUF. The drafter is **dtype-invariant on Q2 hiddens** (Lead 04:
  f32-Q2 == f16-Q2, drafts identical), so F32-train / F32-validate (torch) / F16-deploy all
  agree on IQ2 inputs.
- **Train-set size is MEASURED, not guessed — a learning curve.** The dense LoRA on the
  low-dimensional IQ2 shift (~7.5% per-step Y_iq2≠Y_fp) fits at a few thousand anchors, so
  ~240 prompts (30k anchors) is enough to *fit*; the open question is *generalization* (Stage 2
  overfit at ~28k anchors). So: train on 60 / 120 / 240 / 480 prompts and track **held-out** p1
  (not training loss). If held-out is still climbing at 240 → data-limited → scale the corpus
  (prefer source *diversity*). If held-out peaks then drops before 240 → overfitting → don't
  scale; regularize (lower LoRA rank, dropout, earlier stopping). The prepared 240-prompt set is
  the **starting point** for this curve, not a committed final size.
- **Fidelity gate (mandatory before any powered number):** the trained drafter, loaded into the
  torch oracle, must (a) reproduce the current 0.79 baseline bit-token-for-token on a known-good
  held-out slice with LoRA disabled (sanity), and (b) post-training, agree at F16 and F32 on Q2
  (the invariance must survive fine-tuning).

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
  without a stronger result; the learning curve then decides whether more data or more
  regularization is the lever.
- **STOP:** Δp1 ≤ +1 pp OR held-out harm (negative, like Stage 2). → the drafter-quality axis is
  closed; the 0.79 is a capacity/information ceiling, not a mismatch; future effort goes to
  Lead 08 (verify) + the target-trajectory property.

## Power / sizing

Two distinct questions, both addressed:

- **Eval (verify a gain):** prompt-clustered (the unit is the prompt). Per-prompt Δ SD ≈ 0.03
  (same-trajectory paired-drafter noise, from the Lead-07 crossed-oracle cells; ~4× smaller
  than Lead-04 Phase B's cross-trajectory σ because the trajectory cancels). At 80% power /
  one-sided α=0.05: **+2 pp needs ≈14 prompts, +1 pp needs ≈56**. **60 eval covers the +1 pp
  bar.** (Binomial floor is *above* the empirical σ → adding anchors-per-prompt past ~100 buys
  nothing; more prompts is the only lever.)
- **Train (make progress that generalizes):** with routed experts frozen, the trainable surface
  is LoRA adapters (millions of params) on a low-dimensional shift — ~240 prompts (30k anchors)
  is enough to **fit**. Generalization is the binding constraint (Stage 2 overfit at ~28k), so
  the train-set size is set by the **learning curve** (held-out vs train-set size), not by a
  guess. Start at 240; scale only if held-out is still climbing.

## Codex gates + fidelity gate

1. **Codex gate A (setup):** the KL-loss indexing/normalization over the IQ2 top-128 (padding,
   masked positions; the Lead-11 unified-capture `lead3_logprobs.jsonl` format — soft-label
   capture proven, no bench augmentation needed), the **Phase-0
   gradient-probe + per-target ablation** (the target selection is sound — the ‖∇W‖/‖W‖
   normalization, the candidate set, the ablation's held-out read), the **expert-freeze +
   LoRA-target choice**, the F32/mixed precision, and the train/eval disjointness — BEFORE
   trusting any trained number.
2. **Fidelity gate:** trained drafter reproduces the 0.79 baseline (LoRA disabled) + F16==F32 on
   Q2 post-training.
3. **Codex gate B (verdict):** the PROCEED/MARGINAL/STOP vs the data — honestly scoped
   (significance ≠ a large gain; +2 pp is modest), per-source stratified, + the learning-curve
   read (data-limited vs overfit) explicit.

## Non-goals

- **Full fine-tune of the 19.85B MoE** — infeasible at any sane corpus (per-expert overfitting;
  MXFP4 train complexity).
- **Routed-expert / router tuning** — the experts are frozen.
- Another hard-label head-only run (Stage 2, closed).
- Distillation from the FP/native target (reproduces the current drafter).
- Drafter architecture search; a bigger drafter body.
- The verify side (Lead 08) or the scheduler (Lead 02) — this is an acceptance/quality study.
- The +20% runtime target (out of scope; a +2 pp acceptance win is a modest contributor).

## Deliverables

1. The soft-label IQ2 capture (240 train + 60 eval): H_iq2 + Y_iq2 + IQ2 top-128.
2. The Phase-0 target-selection result (gradient-probe ranking + per-target ablation) — which dense parts carry the IQ2-mismatch signal.
3. The trained dense-LoRA drafter (F16 GGUF, routed experts frozen) + the training config/checkpoint + the learning curve.
4. The held-out p1 (+ E[a|4]) vs the 0.79 baseline + the 0.85 ceiling, per-source, with CIs.
5. The PROCEED/MARGINAL/STOP verdict + the codex-gate artifacts.

## Exit conditions

- **PROCEED:** +2 pp held-out p1 (CI excl 0) → integrate + re-bench the M3 stack.
- **STOP:** ≤ +1 pp or harm → the drafter-quality axis closes; 0.79 is a capacity/information
  ceiling. Effort moves to Lead 08 (the runtime +20%) + the target-trajectory property.

## One-line verdict (proposal)

Lead 10 tests whether **soft-label dense-LoRA** (routed experts frozen) can re-target the
drafter to the IQ2 distribution and generalize where Stage 2's hard-label LoRA did not — the
one untested drafter-quality route after Lead 07 (hidden-side, dead) and Stage 2 (hard-label,
dead). A real but uncertain bet with a negative prior; the train-set size is measured by a
learning curve, the +2 pp verdict needs only ~60 held-out prompts.

## Worklog

### 2026-07-19 — codex gate A (Phase-0 probe methodology) → corrections + decisive test

- Codex review (gpt-5.5 xhigh, retained: `artifacts/dspark_codex_reviews/2026-07-19_lead10_phase0_gateA.md`)
  + my independent verification:
  - **Sinkhorn caveat REFUTED (my error):** the head's `hc_head` is a SIGMOID reduce
    (`drafter_head.py:78`), not the body's iterative Sinkhorn (`hc_primitives.hc_split_sinkhorn`)
    → `head.hc_fn`'s dominance is a REAL local CE sensitivity (the 65k matrix gates the final
    4×4096 HC state before the output), not a gradient artifact. Corrected in `probe_summary.md`.
  - **Metric: raw ‖∇W‖ is size-confounded.** RMS re-rank (‖∇W‖/√n) keeps `head.hc_fn` #1
    (~100×) but DEMOTES `lm_head` (RMS 2.51e-5 — 530M params, inflated under raw norm).
    Verified by recomputing from the shapes.
  - **`lm_head` + `embed_w` are TARGET-loaded** (shared with the verifier —
    `drafter_body.py:206`, `drafter_head.build_head`) → `lm_head` DROPPED from the LoRA set
    (deployment problem).
- **Corrected target set:** train `head.hc_fn` directly; LoRA `main_proj` + the attention
  projections + the router + the body HC-mixing + the shared expert; **DROP `lm_head`**.
- **Decisive next step (codex):** run a `head.hc_fn`-only ablation now (the head LoRA exists
  from Stage 2) — if held-out p1 improves under soft labels, keep it #1; else demote the
  gradient spike and re-select targets by KL + top-r/RMS.

### 2026-07-19 — Phase-0 gradient probe run (the LoRA-target ranking)

- Probe on the Lead-11 unified capture (`lead3_h.bin`, 60 prompts, max_step=3, **F32** — F16
  backward NaN'd through the MoE/rmsnorm; max_step=3 to fit the ~9 GB/step MoE activation
  retention within the 182 GB MPS limit). Experts frozen (verified `exp_gate.grad is None`).
  Loss: **CE vs sel** (the soft labels were dropped in Lead-11's Exp-0a consolidation — a
  Phase-1 re-capture item; CE is a valid probe proxy). Result: `artifacts/lead10_phase0/{gradient_probe_result.json, probe_summary.md, gradient_probe.py}`.
- **Finding (raw ‖∇W‖, matrices):** `head.hc_fn` (17.4) ≫ `main_proj` (1.3) > `head.lm_head`
  (0.58) ≈ attn projections (0.55) > `shared_expert` (0.39) > router (0.21) > body
  `hc_attn_fn`/`hc_ffn_fn` (0.19/0.18). The ‖∇W‖/‖W‖ ratio inflated the tiny HC-scale weights
  (a known artifact — those are scalar/vector, excluded from the LoRA set). **The IQ2-mismatch
  gradient concentrates at the OUTPUT (head.hc_fn, lm_head) + the INPUT (main_proj) + the
  attention — NOT in the body's HC mixing.** Suggested set: train head.hc_fn directly (65k
  params); LoRA main_proj + attn + lm_head + shared_expert.
- **Caveat (for codex gate A):** head.hc_fn's dominance may be partly Sinkhorn-iteration
  gradient amplification — the ablation must confirm it generalizes.
- **Deferred:** the per-target LoRA ablation (Phase-0 part 2) needs the training setup (body
  LoRA not yet built; head LoRA exists from Stage 2) + the labels (soft labels missing) →
  folded into Phase 1's first training runs (which ablate the targets).

### 2026-07-19 — Lead 11 unified soft-label capture available → Phase 0 unblocked (pre-execution)

- Lead 11's unified capture (`issue468/artifacts/lead11_unified_capture/`) produced H_iq2 +
  Y_iq2 + IQ2 top-k for the **lead3 corpus**: `lead3_h.bin` (variable-length records
  `id_len·id·pos·tok·hidden[12288]`; 5,811 records, consumes the file exactly; hidden finite)
  + `lead3_logprobs.jsonl` (`{id,pos,sel,top}`; `sel` = Y_iq2, matches the h.bin `tok` on all
  5,811 records — 0 mismatches; `top` = the IQ2 top-k soft labels). 60 prompts / 5,811 anchors.
- **Resolves the Phase-0 `--dump-logprobs` open item** — the soft-label capture mechanism is
  proven; no bench augmentation needed. The Phase-0 gradient probe is immediately runnable on
  this capture (KL on the IQ2 top-128, not just the hard-label proxy).
- The distill_corpus (240 train + 60 eval) soft-label capture is still a separate run, but now
  a plain bench invocation (mechanism proven) — not a Lead-10 blocker.

### 2026-07-16 — added Phase-0 LoRA target-selection diagnostic (pre-execution)

Added a two-step Phase-0 to pick the dense LoRA targets empirically rather than guessing:
(1) a **gradient-magnitude probe** (‖∇W‖/‖W‖ per dense candidate, one backward pass; runnable
on the lead3 captures with a hard-label proxy before the soft-label capture lands);
(2) a **per-target LoRA ablation** to confirm which targets generalize. Threaded through the
training approach (targets now selected by the probe), Codex gate A, and the deliverables.
Secondary read: if no dense target carries signal, temper the prior (pre-STOP hint).

### 2026-07-16 — drafter architecture finding → dense-LoRA reframe (pre-execution)

- Parameter count via `index_gguf` on `dspark.gguf`: **19.85B params, ~19.35B in the routed
  experts** (256 experts × 3 layers, `ffn_{up,gate,down}_exps [4096,2048,256]`); dense parts
  ~0.5B. Cross-checked against `inventories/dsv4_flash_dspark_model.md` (256 routed + 1 shared
  expert/layer, **6 activated/token**, MXFP4 routed experts).
- **Implication:** full fine-tune is infeasible — at 30k anchors × 6 active / 256 experts, each
  ~25M-param routed expert sees ~700 updates → per-expert overfitting (+ MXFP4 train complexity).
  → routed experts **frozen**; trainable surface = **dense-LoRA** (attention + `main_proj` +
  heads + HC mixing, optionally shared expert). Reframed the lead from "full-body" to
  "soft-label dense-LoRA, experts frozen." The real delta over Stage 2 (non-expert+head LoRA,
  hard labels) is the **soft labels** + a deliberate LoRA-target/rank choice.
- Train-set sizing is now a **learning curve** (60/120/240/480 → held-out p1), not a guess:
  ~240 prompts fits the dense LoRA on the low-dimensional IQ2 shift; generalization (Stage 2's
  failure mode) is the binding constraint.

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
