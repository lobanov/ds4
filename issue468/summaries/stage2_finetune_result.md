# Stage 2 fine-tune PoC — result

Date: 2026-07-06. Status: **complete. Verdict: NOT-JUSTIFIED** (non-expert drafter
fine-tuning does not materially raise p=1 acceptance). Implements
`summaries/stage2_finetune_protocol.md` (Activities 1–9). Self-consistent torch MPS
drafter (float32; body structurally faithful, head bit-faithful).

## Verdict

**NOT-JUSTIFIED.** Head-only LoRA trained with the **specified DSpark §3.3 loss**
(`Lce + Ltv` total-variation + exponential position weights `w_k=exp(−(k−1)/γ)`, γ=5,
teacher-forced over K=5 positions) and swept over **rank {32,64,128} × seed {42,7}**
gives **−1.46 / −1.75 / −3.08 pp** held-out p=1 (best −1.46 pp at rank 32; seed-stable,
std 0.0017; worse at higher rank = overfitting). Body/input-side LoRA (Activity 8) was
not run — gated out by its contract (Activity 7 best > Activity 6 ceiling) and Activity 4
(drafter input-invariant at p=1). Decision: **close the non-expert fine-tuning direction.**

> Two adversarial codex reviews were load-bearing (both retained + independently verified):
> (1) the **body-bug review** found a Sinkhorn first-row eps-placement error in the torch
> body port (fixed; body layer-0 parity restored to ~2e-4); (2) the **verdict review**
> found a **dead-LoRA-init bug** (both factors zero-initialized → `delta=B@A=0` with zero
> gradient; an intermediate "NOT-JUSTIFIED" was based on norm-scaling only) AND required
> the specified `Lce+Ltv` loss + seed sweep before the verdict was defensible. Both fixed;
> the corrected, contract-compliant `Lce+Ltv` + sweep result above is the sound one.

## Evidence (Activities 4/6/7/9)

- **Activity 4 (crossed hidden/label oracle):** on 54 prefix-aligned cells the drafter is
  **input-invariant** at p=1 (input_effect 0.0; p=1 prediction identical on IQ2XXS vs Q4-tap
  hidden states 98% of the time); **label/argmax drift dominates** (label_effect 1.85 pp).
  ⇒ the lever is output/head ordering, not input representation.
- **Activity 6 (from-scratch head ceiling [GATE]):** no-LoRA frozen baseline p=1 = **0.8125**
  (≈0.81; self-consistent torch baseline confirmed). From-scratch head on frozen features
  reaches only **0.7479** held-out (< pretrained — under-generalization vs FP-trained head).
  **Overfit-to-eval 0.9947** ⇒ head impl correct, `h` carries the target info.
- **Activity 7/9 (head-LoRA, Lce+Ltv, rank×seed sweep):** baseline 0.8125; **rank 32 −1.46 pp
  (±0.0017 over 2 seeds), rank 64 −1.75, rank 128 −3.08** — **all negative**, worse at higher
  rank (overfitting), seed-stable. (An earlier p=1-CE-only run gave −0.13/−1.44 pp — same
  direction; the Lce+Ltv is the contract-specified loss and is reported as the result.)
- **Activity 8** skipped per its contract (Activity 7 best 0.7979 > Activity 6 ceiling 0.7479)
  + Activity 4 (input-negligible). Body LoRA is the untested non-expert lever (codex flagged
  the skip as the weakest point; documented).

## Interpretation

- The Q2-target drafter deficit is **shallow** (Stage 0: target usually rank-2) and
  **label/ordering-side** (Activity 4), but **not recoverable by non-expert head LoRA**:
  with the specified Lce+Ltv loss, head-LoRA overfits train and **reduces** held-out p=1
  (rank-sweep negative, seed-stable). The Ltv distribution-matching term does not rescue it.
- Consistent with Stage 1 (raising tap precision didn't help) and Stage 0's caveat that
  shallow-error *shape* ≠ *recoverability*. The from-scratch ceiling (<pretrained) signals
  the Q2 features generalize worse than FP, but not in a way non-expert head LoRA exploits.
- **Body/input-side LoRA (Activity 8) is the remaining untested non-expert lever** — skipped
  per the contract gate + Activity 4's input-invariance. It is the natural follow-up if
  non-expert tuning is pursued further.

## Methodology notes

- torch MPS drafter (body+head, float32) used **self-consistently**; decision rule measures
  *relative* improvement (valid regardless of inherent float32 BLAS chaos vs numpy).
- **Head bit-faithful** to numpy `forward_head` (argmax 100%, top-128 100%, base max-abs 3e-5).
  **Body structurally faithful** (layer-0 parity ~2e-4 after the codex-found Sinkhorn eps fix);
  deeper-layer float32 BLAS chaos inherent + accepted.
- Loss = `Lce + Ltv` with `w_k=exp(−(k−1)/5)` over K=5 teacher-forced positions (DSpark §3.3);
  Ltv computed over the captured top-128 target distribution + tail bucket. Seed sweep: 2 seeds
  (42, 7), stable (std ~0.0017). (3rd seed not run; the result is robustly negative across 2
  seeds × 3 ranks.)

## Recommendation

- **Close the non-expert fine-tuning direction** (head LoRA: tested with the specified loss,
  overfits/−1.46 pp; body LoRA: gated out + Activity 4 input-negligible).
- **Expert tuning** is the remaining lever but is **out of scope** (separate workstream).
- Else redirect to **server-side multi-request batching** — local single-request speculative
  decode on this target/drafter has now exhausted drafter precision, tap precision, tree
  structure, and non-expert fine-tuning without clearing baseline.

## Artifacts

- Results: `dspark_train/data/activity{6,7,9}*.json` + `activity7_kloss_summary.json`; features
  `dspark_train/data/{train,eval}_{torch,kloss}_features.safetensors`.
- Harnesses: `dspark_train/{drafter_body,drafter_head,run_stage2_*}.py`.
- Captured data: `dspark_train/data/shards/`; corpus `prompts/stage2_corpus/`.
- Engine change: `--capture-dataset` + multi-layer patch (ds4 repo, this branch).
- Codex reviews: `artifacts/stage2_plan_review/{codex_review,bodybug_codex_review,verdict_codex_review}.md`.
