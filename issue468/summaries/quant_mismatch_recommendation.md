# Quantization-mismatch investigation — Stage 2 recommendation

Date: 2026-07-06. Status: **recommendation = NARROW (proceed to a bounded Stage 2
proof-of-concept; do not commit to a full fine-tuning pipeline).** Inputs:
`summaries/quant_mismatch_diagnostic.md` (Stage 0) and `summaries/stage1_tap_precision.md`
(Stage 1), both adversarial-codex-reviewed. This note is the deliverable for the
issue468 quant-mismatch goal.

## What was tested (Stage 0 + Stage 1, both codex-reviewed)

- **Stage 0 (offline diagnostic, robust):** the DSpark drafter's p=1 misses vs the
  IQ2XXS target are **shallow and in the right neighborhood** — median target-rank
  on misses = 1.0; 100% within the drafter's top-10; top-2 coverage 0.8125→0.9125.
  This is the regime where a fine-tune has maximum leverage. Causal link to *quant*
  is NOT proven (only ~27% of misses are genuine Q2 near-ties; median Q2 gap 2.15 nat).
- **Stage 1 (Q4-tap A/B, underpowered):** raising layers 37–42 routed experts to
  Q4_K did **not materially improve p=1 acceptance** (0.8125→0.7875, McNemar p≈0.77;
  E[a|5block] +0.10 with bootstrap 95% CI [−0.16, +0.41] — not proven noise, study
  underpowered; one-sided deeper-position trend p3–p5, rollout-confounded). This
  variant cannot isolate tap precision from lower-layer Q2, so it falsifies only
  "this expert-only Q4-tap variant helps," not "quant mismatch" in general.

## Recommendation: NARROW — a bounded Stage 2 proof-of-concept, not a full pipeline

**Proceed to Stage 2 only as a small, bounded PoC**, gated on a measurable p=1 lift.
Rationale:

- **For:** Stage 0 robustly shows recoverable-shape (shallow) error with real
  headroom (top-2 coverage 0.91 at p=1). Fine-tuning addresses the deficit
  *regardless of root cause* (quant or calibration) by teaching the served target's
  argmax ordering. And it is the **last unexplored lever with supporting evidence**:
  drafter precision is exhausted (`dspark_quantization_ceiling.md`), the verifier is
  bandwidth-bound (`mtp_verifier_bench_results.md`), trees don't help
  (`spec_speedup_model.md` Q3), and DFlash is a worse drafter (`dflash_oracle_investigation.md`).
- **Against / cautions:** the deficit's *cause* is unknown (lower-layer Q2,
  calibration, or irreducible capacity). If it is capacity, fine-tuning has a low
  ceiling. n=10 prompts / temp=0 only — the shallow-error finding needs broader
  confirmation before a large investment. Fine-tuning is explicitly out of scope
  for cycle 1 (GOAL.md), so this is a scope-extension decision the user should make
  consciously.

**Realistic ceiling — set expectations honestly (secondary gate, not primary):**
Even a fine-tune that captured every current top-2 case (p=1 → ~0.91) reaches
"beat baseline / match --mtp" at K=4 per the speedup model, but **not** the +20%
primary gate (needs ~0.94 at K=4 / ~0.89 at K=5). The +20% gate is a **compound
stretch**: it needs fine-tuning *and* the unbuilt anchor-reuse verifier
(`spec_speedup_model.md` flags the shipped `--mtp` verifier pays a redundant anchor
decode, ~−18 pp at K=4) *and* acceptance near the top-2 ceiling. So: fine-tuning
most plausibly delivers the **secondary** gate; the primary gate is an outside shot
requiring multiple independent things to land.

## Recommended next experiments (ranked; cheap ones first)

1. **Crossed hidden/label oracle (cheap, reuses existing artifacts).** Combine
   Q4-tap hidden states with baseline target tokens, and baseline hidden states
   with Q4-tap tokens, then re-measure. Separates **input-shift** (hidden-state
   representation — fixable via the drafter's input projection) from **label-shift**
   (argmax drift). Tells the fine-tune *where* to focus. No ds4 recapture.
2. **Broader / temp-varied corpus for the Stage 0 diagnostic.** Re-run the rank
   diagnostic on >100 prompts at temp 0/0.5/1.0 to confirm the shallow-error
   finding and narrow the CIs before any ML spend.
3. **Bounded Stage 2 PoC (the actual fine-tune test).** Small LoRA on a modest
   Q2-distillation corpus, measured on the existing harness. Decision rule: a
   **+3–5 pp p=1 lift** ⇒ the direction has legs (scale up); **flat** ⇒ the deficit
   is closer to irreducible capacity than calibration ⇒ stop.
4. **(Decisive but high-effort, needs a model not on disk)** Full Q8/BF16 target
   top-k on the exact p=1 miss set — directly separates quant from calibration.
   Codex's proposed decisive mechanism test.

## Fine-tuning-setup insights (from this investigation)

- **Corpus = (served-target tap hidden states → served-target greedy argmax) pairs.**
  The existing capture tooling (`run_exactness_small_bundles.py`) already produces
  exactly these (layer-40/41/42 hidden states + greedy tokens). Corpus gen reuses
  it — just needs more / longer prompts and a train/eval split.
- **The drafter's input is the concat of layers 40/41/42 (3·DIM=12288).** A minimal
  viable fine-tune targets `main_proj` (input projection) + the 3 drafter dense
  layers + the **markov head** (`markov_w1`/`markov_w2`). The markov head directly
  controls the top-of-distribution ordering where Stage 0 showed the errors
  concentrate (target usually rank 2); it must be in scope.
- **Low LoRA rank may suffice.** The error is a consistent one-rank mis-ordering,
  not a deep representation failure — suggesting a small parameter delta could move
  the argmax. Start with a small rank and measure.
- **Eval on the exactness corpus, exact-output preservation required.** Stage 2's
  success bar is a p=1 lift that preserves greedy exactness on the retained
  exactness bundles (the GOAL's exactness constraint). The harness for this exists.
- **Risk to watch:** if the deficit originates in lower-layer Q2 (Stage 1 could not
  rule this out), the fine-tune is learning to *compensate* for a representation
  shift at its input — learnable, but the gain magnitude is unknown until measured.

## What would change this recommendation

- **Upgrade to "proceed (full)":** the crossed oracle (exp 1) shows the deficit is
  predominantly input-side label-shift, AND a broader-corpus Stage 0 confirms
  shallowness holds, AND the PoC (exp 3) clears +5 pp p=1.
- **Downgrade to "stop":** the crossed oracle or a Q8 target test shows the deficit
  is irreducible drafter capacity (target rarely in drafter top-k on a larger
  corpus), OR the PoC is flat, OR temp>0 deployment collapses E[a] faster than
  fine-tuning can recover.

## One-line verdict

The quant-mismatch hypothesis, as originally framed (FP-trained drafter vs Q2
target, fixable by precision or fine-tuning), is **partially falsified** — tap
precision doesn't help — but the **underlying shallow-error finding is robust and
keeps a bounded drafter fine-tune in play as the last plausible lever**, most
realistically targeting the secondary gate. NARROW.
