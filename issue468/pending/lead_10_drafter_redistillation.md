# Lead 10 — Soft-label dense-LoRA drafter re-distillation for IQ2XXS (experts frozen)

*** *Re-opened 2026-07-20 for re-attempt (goal mrthg76m-800onm).* The original STOP was
predicated on an **unfaithful torch oracle**: a faithful-repro investigation this session
showed the torch port's body forward proper diverges from the live Metal drafter by ~60%
(rel|diff|), with rec0 (empty win_kv) at 101% — i.e. the body forward itself is wrong, not
the win_kv or the capture. The hc_pre Sinkhorn, the win_kv KV projection, and the window
size (128) all match in code; the divergence localizes to **_attn / hc_post / _moe**.
Consequently the original Lead 10 offline gain (+2.30 pp) was measured on a torch port whose
per-position acceptance shape is wrong (pos1 0.880 vs live 0.996; suffix inflated) — the
offline→live non-transfer is explained by the unfaithful surface, not by a real negative.

*Re-attempt plan (goal mrthg76m-800onm):* (1) fix the torch body forward → faithful repro
(per-position acceptance within ~1pp of live); (2) measure suffix-loss concentration on
the faithful port → pick the LoRA target; (3) re-train with a **REINFORCE objective
(reward = #accepted, positions coupled)** instead of the per-position KL; (4) live ds4 test
+ codex gates A/B. The prior STOP verdict below is superseded pending this re-attempt. ***

---

## Re-attempt status (2026-07-20/21) — faithful repro DONE; REINFORCE paused

**The faithful-repro prerequisite (tasks 1-4) is DONE: position-1 0.880 → 1.000, body rel|d|
0.60 → 0.005.** The offline→live transfer barrier that killed the original Lead 10 is broken.
Target = head.hc_fn (gradient probe, confirmed). Task-5 (REINFORCE) is paused for a
harness-approach decision. The chronological entry is in the [Worklog](#worklog) (2026-07-20/21);
the task-organized detail follows.

### task-2-fix-body — DONE (faithful repro)

Methodology (per the goal): online research (QLoRA → "a Q8_0/Q4_K matmul IS an F32 matmul after
dequant, so a correct dequant should match the live"), two codex bug hunts, + a GPU-dump-sync
discipline (every ds4.c intermediate dump flushed via end_commands/begin_commands —
`ds4_gpu_tensor_read` does NOT sync, which had caused a stale-buffer confound + a false
"860× explosion" bisection).

Four real fixes, in order:
1. **hc_post comb-transpose** (`drafter_body.py:66`). The torch port used the naive
   matrix-vector product `comb[a,b]*residual[b]`; the live `hc_post_one` reads
   `comb_buffer[dst+src*HC]` == `comb[src,dst]` in torch row-major = the **transpose**.
   Codex-verified bit-exact vs the live on a controlled sample (max_abs_delta 0.0 for the
   fix vs 42.0 for the old). Body rel|d| 0.46 → 0.34.
2. **The missing main_x token.** The live Metal attention attends over `n_real + 1 + block`
   tokens — the `+1` is the anchor's main-hidden KV (the `main_x`), stored at slot `n_real`
   during the attention. The torch port (ported from the CPU drafter) missed it. Including it
   + the correct win_kv (fed from the live post-loop kv_cache dump): 0.34 → position-1 0.926.
3. **The harness dump-timing fix.** Codex hunt #2 found the 0.34 residual was a HARNESS bug,
   not a torch bug: `DS4_DSPARK_DUMP_WINKV_POST` was inside the stage-0 branch but dumped all
   3 stages' kv_caches — at that point only stage 0 had stored its main_x; stages 1+2's
   main_x (slot n_real) was zero. Fix: `DS4_DSPARK_DUMP_WINKV_FINAL` dumps after the 3-stage
   loop. Body rel|d| 0.34 → 0.027, position-1 0.926 → 0.988. (Codex also confirmed: routed
   MoE rel=2.85e-05 vs live — sound; shared expert sound; Q8_0/Q4_K dequant bit-exact; Q8_0
   matmul F32-accumulate == torch F32.)
4. **FP8-KV simulation** (`drafter_body.py:fp8_kv_quantize_nope`). The live FP8-quantizes
   the draft-block KV (per-64-block E4M3FN, power-of-2 scale) before storing/attending; the
   torch used F32. Matching it: body rel|d| 0.027 → **0.005** (the FP8-KV noise floor).

Verified-matching components (code + fresh flushed dumps): hc_pre Sinkhorn, rope
(interleaved, freq 10000), swiglu clamp (=10), expert weights, win_kv KV projection,
window=128, head_rms_norm, softmax-with-sinks attention, MLA output-projection grouping,
MoE routing (topk on probs+bias, weights from probs, scale 1.5).

### task-3-verify-faithful — DONE

20-prompt / 506-cycle `body_diff_mainx.py` (live win_kv + main_x fed into the torch body,
vs the live final body dump): **body rel|d| = 0.005**; per-position draft agreement (torch
vs live) **p1=1.000, p2=0.990, p3=0.986, p4=0.982, p5=0.972**. Position-1 acceptance within
~0pp of live (1.000 vs 0.996); positions 2-5 within ~1-1.5pp (the autoregressive cascade +
the residual FP8-KV E4M3 rounding). The body forward is at the quantization noise floor.

### task-4-pick-target — DONE

Gradient probe (`grad_probe_suffix.py`) on the faithful port: suffix-CE (positions 2-5)
gradient RMS over 437 samples — **hc_fn=2.6e-1, markov_w2=6.2e-7, markov_w1=1.3e-7**.
head.hc_fn is the overwhelming #1 target (~6 orders of magnitude above the markov),
confirming Lead 10's Phase 0 finding now on the FAITHFUL port. The markov (inter-position
coupling) has a negligible suffix gradient. The dense-body is structural (the parallel-block
noise-placeholder at positions 2-5). **CHOSEN TARGET: head.hc_fn** (same as Lead 10); the
re-attempt differs by using the REINFORCE objective on the now-faithful port.

### task-5-reinforce-train — IN PROGRESS (PAUSED 2026-07-21 for a harness-approach decision)

REINFORCE LoRA on head.hc_fn (rank 32, reward = #accepted, positions coupled via the
autoregressive rollout), trained on the captured LIVE body (bit-exact) + the sel. The
rollout was verified correct (== head.forward == live drafts). Two harness issues surfaced
that are NOT about the faithful port:
1. **Alignment bug**: the body dump (506 records) ≠ the bench cycles (511) → the sel
   (targets) misalign → the held-out baseline E[a|K] reads ~0.5 instead of ~3.5. Fixable by
   reconstructing the sel from the body dump's own anchors+drafts+the bench accept-counts
   matched by anchor.
2. **Noisy sampling**: the REINFORCE reward (sampled, temp=0.7) is ~0.02 — the suffix is
   uncertain, so sampled drafts are mostly wrong + the policy gradient is noise-dominated
   (reward flat across 8 epochs; held-out ΔE[a|K]=0.009, CI includes 0). Fixable by dropping
   the sampling temp to ~0.1 so samples track the greedy.

**PAUSE** (user direction 2026-07-21): bank the faithful-repro win, document progress, and
decide the task-5 approach before sinking more time into the REINFORCE alignment. Options
under consideration: (a) fix the REINFORCE alignment + lower temp; (b) bank the faithful
repro + defer the LoRA; (c) first run the simplest transfer test (re-bake the existing
Lead 10 KL LoRA + measure live, now that the port is faithful); (d) skip offline LoRA + do
a direct live-ds4 LoRA search.

### Key artifacts (this re-attempt)
- Scripts: `issue468/artifacts/lead10_phase2/{body_diff_mainx,body_diff_stages,debug_attn,debug_rec0,grad_probe_suffix,reinforce_train}.py`.
- ds4.c dump envs (all GPU-flushed): `DS4_DSPARK_DUMP_BODY` / `_BODY_S0` / `_BODY_S1` / `_WINKV` / `_WINKV_POST` / `_WINKV_FINAL` / `_ATTN` / `_ATTNINT` / `_HCSPLIT`.
- Codex reviews: hunt #1 (the hc_post comb-transpose) + hunt #2 (the harness dump-timing + the MoE/FP8 verification), logs under `/var/folders/.../adversarial_codex_*.final.md`.
- Commits (drafterresearch branch): the hc_post fix, the FP8-KV simulation, the harness fixes — see `git log --oneline | head`.

---

Date: 2026-07-16. **Status: RE-OPENED 2026-07-20 for re-attempt (goal mrthg76m-800onm).**
The original STOP below was on an **unfaithful torch oracle** (the body forward diverged ~60%
rel|d| from the live; the +2.30 pp offline was measured on a wrong per-position shape — pos1
0.880 vs live 0.996) and is **superseded**. The re-attempt's faithful repro (position-1
0.880 → 1.000, body rel|d| 0.60 → 0.005) is DONE; the REINFORCE re-training is in progress
(paused for a harness-approach decision). See the Worklog (2026-07-20/21). Original STOP text
(the head.hc_fn LoRA gives
no live-ds4 gain).** The head.hc_fn-only dense-LoRA (KL vs IQ2 top-128, rank=32) gives
+2.30 pp held-out offline (the torch oracle) but **NO live-ds4 gain** (E[a|K] 3.494 vs 3.516,
t/s 35.73 vs 37.07) when baked into the dspark GGUF + run on the live ds4. The offline
drafter_head (the torch port) + the live ds4 drafter are different regimes — the codex gate
B's baseline-discrepancy concern is confirmed. Result: `issue468/archive/leads/lead_10_drafter_redistillation.md`.

*** *Reframed 2026-07-16 after the
drafter parameter-count finding (it is a 19.85B MoE — full fine-tune is infeasible at any sane
corpus; the experiment is dense-LoRA with the routed experts frozen, not "full-body").*

## Purpose

***Re-attempt framing (2026-07-20/21):* the original Lead 10 closed STOP because the
head.hc_fn LoRA gave +2.30pp offline but no live-ds4 gain. This re-attempt rests on a new
finding: that negative was measured on an **unfaithful torch oracle** (the body forward
diverged ~60% rel|d| from the live; the offline per-position acceptance shape was wrong —
pos1 0.880 vs live 0.996). The faithful-repro investigation (online research + 2 codex hunts
+ a GPU-dump-sync discipline) closed that gap: **position-1 0.880 → 1.000, body rel|d| 0.60 →
0.005** — the torch oracle is now a trustworthy training surface. The original question is
re-opened, this time with a **REINFORCE objective (reward = #accepted, positions coupled)**
instead of the per-position KL. The faithful-repro detail + the REINFORCE status are in the
[Worklog](#worklog) (2026-07-20/21).**

The original Lead 10 question (the FP-vs-IQ2 distribution mismatch):

> Resolve the one drafter-quality question left open after Lead 07 and Stage 2:

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

### 2026-07-20/21 — RE-OPENED for re-attempt (goal mrthg76m-800onm): faithful repro DONE, REINFORCE in progress

**Why re-open:** the original STOP (below) rested on an **unfaithful torch oracle**. A
faithful-repro investigation (online research + 2 codex hunts + a GPU-dump-sync discipline)
found the torch port's body forward diverged from the live Metal drafter by ~60% rel|d| — the
offline +2.30 pp was measured on a port whose per-position acceptance shape was wrong
(pos1 0.880 vs live 0.996; suffix inflated). The offline→live non-transfer is explained by
the unfaithful surface, not a real negative. Re-attempt plan: faithful repro → suffix-loss
measurement → REINFORCE (reward = #accepted) → live test.

**task-2 (faithful repro) — DONE.** Four real fixes, body rel|d| 0.60 → **0.005**:
1. **hc_post comb-transpose** (`drafter_body.py:66`). Torch used the naive matrix-vector
   product; the live `hc_post_one` reads `comb_buffer[dst+src*HC]` == the **transpose**.
   Codex-verified bit-exact (max_abs_delta 0.0 vs 42.0). 0.46 → 0.34.
2. **The missing main_x token.** The live Metal attention is over `n_real + 1 + block`
   tokens — the `+1` is the anchor's main-hidden KV at slot `n_real`, which the CPU-port torch
   missed. Including it + the correct win_kv: 0.34 → position-1 0.926.
3. **The harness dump-timing fix** (codex hunt #2). The 0.34 residual was a HARNESS bug, not
   a torch bug: `DS4_DSPARK_DUMP_WINKV_POST` captured only stage 0's main_x. Fixed via
   `DS4_DSPARK_DUMP_WINKV_FINAL` (post-loop). Codex also confirmed: routed MoE rel=2.85e-05,
   Q8_0/Q4_K dequant bit-exact, Q8_0 matmul F32-accumulate == torch F32. 0.34 → 0.027.
4. **FP8-KV simulation** (`fp8_kv_quantize_nope`). The live FP8-quantizes the draft-block KV
   (per-64-block E4M3FN, power-of-2 scale); the torch used F32. Matching it: 0.027 → **0.005**.
   Driver: a stale-GPU-buffer confound (`ds4_gpu_tensor_read` does NOT sync) had caused a
   false "860× explosion" bisection; fixed via end_commands/begin_commands flushes before
   every ds4.c intermediate dump.

**task-3 (verify) — DONE.** 20-prompt / 506-cycle `body_diff_mainx.py`: body rel|d|=0.005;
per-position draft agreement (torch vs live) **p1=1.000, p2=0.990, p3=0.986, p4=0.982,
p5=0.972**. Position-1 acceptance within ~0pp of live (1.000 vs 0.996); positions 2-5 within
~1-1.5pp (the autoregressive cascade + residual FP8-KV E4M3 rounding). Body forward at the
quantization noise floor. **The offline→live transfer barrier is broken.**

**task-4 (target) — DONE.** Gradient probe (`grad_probe_suffix.py`) on the faithful port:
suffix-CE gradient RMS — **hc_fn=2.6e-1, markov_w2=6.2e-7, markov_w1=1.3e-7**. head.hc_fn is
the overwhelming #1 target (~6 orders of magnitude above the markov), confirming Phase 0 now
on the faithful port. The markov (inter-position coupling) is NOT the suffix lever. The
dense-body is structural (the parallel-block noise-placeholder). **Target: head.hc_fn.**

**task-5 (REINFORCE) — IN PROGRESS, PAUSED 2026-07-21.** REINFORCE LoRA on head.hc_fn
(rank 32, reward = #accepted, positions coupled via the autoregressive rollout), trained on
the captured LIVE body (bit-exact). The rollout was verified correct (== head.forward == live
drafts). Two harness issues (NOT about the faithful port): (1) an alignment bug — the body
dump (506 records) ≠ the bench cycles (511) → the sel misaligns → the baseline reads ~0.5
not ~3.5; (2) noisy sampling — the reward (temp 0.7) is ~0.02 (the suffix is uncertain), so
the policy gradient is noise-dominated (reward flat over 8 epochs; held-out ΔE[a|K]=0.009,
CI includes 0). Both fixable. **PAUSED** for a decision on the harness approach (fix the
REINFORCE alignment + lower temp / bank the faithful win / simplest-transfer-test first /
direct live-ds4 LoRA search).

Key artifacts: `artifacts/lead10_phase2/{body_diff_mainx,body_diff_stages,debug_attn,debug_rec0,grad_probe_suffix,reinforce_train}.py`;
ds4.c dump envs (`_BODY/_BODY_S0/_BODY_S1/_WINKV/_WINKV_POST/_WINKV_FINAL/_ATTN/_ATTNINT/_HCSPLIT`, all GPU-flushed);
codex reviews (hunt #1 the hc_post transpose, hunt #2 the harness/MoE/FP8 verification).

### 2026-07-20 — ds4 integration (SC7): NO live gain → STOP (the decisive test)

- Baked the head.hc_fn LoRA (trained on all 60 lead3, KL, rank=32) into the dspark GGUF
  (the mtp.2.hc_head_fn, at the correct absolute offset `gdoff + off = 11357400928`, F16).
  Two baking bugs found + fixed: (1) the offset was `off` (relative) not `gdoff + off`
  (absolute) — the delta went to the wrong file location; (2) the byte order (the
  `_gguf_ne_to_torch` does `reshape(reversed(dims))`, a simple C-order flat, not a
  transpose). Verified: the torch oracle with the baked weight reproduces the gain
  (+6.18pp in-sample — the LoRA trained on all 60, so higher than the +2.30pp held-out).
- **The live ds4 (the decisive test):** baseline E[a|K]=3.516, t/s=37.07. LoRA-baked
  E[a|K]=3.494 (−0.022), t/s=35.73 (−1.34, −3.6%). **NO live-ds4 gain.**
- **The offline +2.30pp (held-out) / +6.18pp (in-sample) does NOT transfer to the live ds4.**
  The same baked hc_head_fn weight → the torch oracle shows the gain, the live ds4 doesn't.
  The offline drafter_head (the torch port) + the live ds4 drafter are **different regimes**.
  The codex gate B's baseline-discrepancy concern (the phase2 cache 0.8440 vs the combined300
  0.7945) is confirmed: the offline regime ≠ the deployment regime.
- **Verdict: STOP.** The head.hc_fn LoRA (the #1 gradient target, +2.30pp held-out offline)
  does not deploy — no live-ds4 acceptance or throughput gain. The offline-trained LoRA
  (on the drafter_head's regime) doesn't transfer to the live ds4's regime.
- **Implication:** the drafter-quality axis (Lead 10) is closed. The offline torch port +
  the live ds4 must be reconciled before any further offline-trained LoRA can be trusted.
  The +20% remains Lead 08 (the verify side).

- Codex gate B (gpt-5.5 xhigh, retained: `artifacts/dspark_codex_reviews/2026-07-20_lead10_phase2_gateB.md`):
  - **C1 (+2.30pp):** validly derived (CI [+0.0161, +0.0301]) but conditional on one stochastic run
    (no torch.manual_seed) + hparam/target selection already used lead3 (in-sample bias).
  - **C2 (the baseline): LIKELY-WRONG.** The phase2 cache baseline (0.8440) does NOT match
    the combined300 for the SAME 0080–0099 IDs (combined300 mean ~0.7945). Specific prompts
    disagree hard (codealpaca_0080: phase2 0.852 vs combined300 0.762; dolly_0090: 0.680 vs
    0.852). So the phase2 cache (the drafter_body forward) + the combined300 (the Lead-03
    measure) are DIFFERENT regimes — the +2.30pp is on the phase2 cache, not the deployment.
  - **C3 (F16):** approximate (not a deployment proof; the decisive test is the live ds4).
  - **C4 (PROCEED): questionable** — the p1 clears the rule, but the baseline mismatch +
    F16 + E[a|4] gaps undermine the deployment PROCEED.
- **Verdict revised: MARGINAL.** The p1 signal is real-looking (the Δp1 CI excludes 0, all
  sources positive) but the baseline/eval regime is questionable (the phase2 cache may not be
  the deployed 0.79 regime) + the F16/E[a|4]/head-only gaps. The decisive test is the **ds4
  integration (SC7)**: bake the trained LoRA into the dspark GGUF + run the live ds4 on the
  combined300 or the fresh distill eval → the row-matched p1 + E[a|4] + the F32/F16/ds4
  agreement. NOT archived (the MARGINAL → the integration pending).

### 2026-07-20 — Phase-2 held-out validation: +2.30 pp (PROCEED on the phase2 cache)

- **Root cause** (read from `ds4.c:28286 dspark_hc_head_one`): the ds4's hc_head computes the
  RMS norm (`rms_norm_no_weight`) + the matvec (`matvec_f16`) + the sigmoid **all in F32**
  (the `float *` signatures; `matvec_f16` returns an F32 `pre`). The torch oracle's
  `drafter_head.hc_head` computed the rsqrt + the matmul **in F16** → overflowed on the large
  body-output values (up to 1721 → x² > 65504 → F16 inf → the rsqrt + the matmul-result
  cast collapse → p1 0.55 vs F32 0.85).
- **Fix:** made `drafter_head.hc_head` **F32-internal** (cast the input + the weights to F32,
  compute the rsqrt + the matmul + the sigmoid in F32, cast the output to the input dtype) +
  the `p1_scores` norm in F32. Faithful to the ds4's F32-internal hc_head.
- **Re-check:** **F16==F32 |diff| = 0.00000 → PASS.** The trained head at F16 gives
  p1=0.8742 == F32 0.8742 (the +2.4 pp holds at F16). The dtype-invariance is restored.
- **Learning curve** (with the fix): 10/20/40 → +1.6/+2.1/+2.4 pp (steady climb; the
  20→40 gain +0.34 pp → **data-limited**, scale to the distill_corpus).
- **ENCOURAGING** → the ds4 integration (SC7) is in scope. [codex review/bug-hunt of the fix
  next, per the f16-oracle-fix task.]

### 2026-07-19 — learning curve + the F16 fidelity complication (PAUSED for direction)

- **Learning curve** (head.hc_fn KL, nested train 10/20/40, 20-prompt eval): ~+2.0–2.6 pp
  but **noisy** — the 20→40 climb is inconsistent across runs (within the training noise; the
  LoRA init + the shuffle vary the held-out by ~0.5–0.9 pp). Train-set sizing is **inconclusive
  at 40 prompts** (the noise dominates the 20→40 signal). Result:
  `artifacts/lead10_phase1/learning_curve_result.json`.
- **F16 fidelity FAILS — but it's a torch-oracle artifact, not a LoRA issue.** The trained head
  at F16 gives p1=0.55 vs F32 0.87. **The UNTRAINED F16 head ALSO gives 0.55** → not a LoRA
  issue (the LoRA delta ‖B@A‖ is identical F16/F32 = 0.6104). Root cause: the torch oracle's
  `hc_head` overflows at F16 — the body output x0 has values up to **1721** (~2e-4 of elements
  > 255 → x0² > 65504 → F16 inf → the `rsqrt` + the matmul accumulation collapse). The ds4
  deployment (the live decoder) works at 0.79 (it handles the overflow differently — likely a
  F32 rsqrt / accumulation / clamping in the C/Metal path). **So the torch oracle's F16 is NOT
  faithful to the ds4's F16** for the hc_head's overflow handling.
- **Implication:** the +2.4–2.6 pp (F32) is solid, but the **F16 deployment fidelity can't be
  checked via the torch oracle** (the overflow artifact). The proper F16 check (the ds4 with the
  trained LoRA) is the **integration** — out of scope here. **Paused for the user's decision** on
  how to proceed:
  - (A) accept the F32 result (+2.4 pp → PROCEED on the F32 metric) + note the F16-deployment
  caveat (the integration verifies);
  - (B) fix the torch oracle's F16 `hc_head` (compute the rsqrt + the matmul accumulation in
  F32 internally) + re-check the F16 fidelity;
  - (C) integrate the trained LoRA into the ds4 + the live F16 check (a separate goal).

### 2026-07-19 — head.hc_fn KL ablation (codex exp #2): +2.58 pp, CONFIRMS the CE result

- Re-captured the lead3 soft labels (`ds4-spec-bench --dump-logprobs-jsonl`, top-128) →
  `artifacts/lead10_phase1/lead3_logprobs_recapture.jsonl` (5,811 anchors, `{id,pos,sel,top}`;
  **0 mismatches vs lead3_h.bin** — the recapture is aligned with the original trajectory).
- The head.hc_fn KL ablation (`head_hcfn_kl_ablation.py`, rank=32, 12 epochs, KL vs the IQ2
  top-128): baseline p1 = 0.8501; trained p1 = 0.8759; **delta = +2.58 pp**. The KL (the goal's
  specified loss) **confirms** the CE result (+2.41 pp) — head.hc_fn generalizes under both.
  The codex's concern (the KL might rank differently) didn't materialize (the IQ2 teacher is
  peaked — top token ~99.75% — so KL ≈ CE for the dominant positions; the KL edges slightly
  higher, +2.58 vs +2.41).
- **Net:** head.hc_fn-only clears the +2 pp PROCEED bar under BOTH CE + KL (on the 20-prompt
  eval). The decisive-test + its KL confirmation are both positive.

### 2026-07-19 — head.hc_fn ablation (codex's decisive test): +2.41 pp, GENERALIZES

- The head.hc_fn-only LoRA ablation (`artifacts/lead10_phase1/head_hcfn_ablation.py`): rank=32,
  12 epochs, hard-label CE vs sel, 40 train / 20 eval (seed 42) from the lead3 captures.
  Precomputed the body features no_grad (sidesteps the MoE activation-retention issue).
- **Result:** baseline (no LoRA) p1 = 0.8501; trained (hc_fn LoRA) p1 = 0.8742; **delta =
  +2.41 pp** → **GENERALIZES**. The codex's decisive test is POSITIVE — head.hc_fn (the #1
  gradient target) is a real, trainable lever, not just a large gradient.
- **Fixed:** the F16 body forward overflowed (inf in x0 → NaN training); switched the precompute
  to **F32 no_grad** (fits: experts 77 GB + ~9 GB/step, no retention; 0 inf-prompts).
- **Baseline note (fidelity):** the torch-oracle baseline = 0.8501, NOT the 0.79 (the combined300
  full-59 mean). Per-prompt the torch oracle MATCHES combined300 (dolly_0090: 0.852 ≈ the
  ablation). The 20-prompt eval subset is easier than the full corpus (0.85 vs 0.79). The
  +2.41 pp delta is internally consistent (same metric, same eval) but on the easy 20-subset.
- **Caveats:** hard-label CE (the KL version needs the soft-label re-capture); head.hc_fn-ONLY
  (the full multi-target LoRA — main_proj + attn + router + body HC + shared_expert — is the
  next step, needs body-LoRA); 20-prompt eval (the 60-prompt validation is phase 2).
- **Net:** head.hc_fn clears the +2 pp bar on a preliminary 20-prompt hard-label test → proceed
  to the full multi-target LoRA + the soft-label (KL) re-training + the 60-prompt validation.

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
