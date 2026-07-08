You are a skeptical senior ML engineer doing an ADVERSARIAL review of a PoC VERDICT.
The investigator concluded NOT-JUSTIFIED (non-expert head-LoRA gives only +0.6 pp).
Challenge whether that verdict is sound and find methodology holes that could flip it.
Be terse; decisive claims must be runnable/citable (dispatcher re-verifies).

WORKTREE: /Users/lobanov/Projects/ds4-dspark-research. Venvs: `issue468/dspark_train/.venv`
(torch+numpy), `issue468/dspark_oracle/.venv` (numpy).

READ FIRST (the verdict + evidence):
- `issue468/summaries/stage2_finetune_result.md`   (the verdict + evidence + methodology)
- `issue468/dspark_train/data/activity9_final.json`, `activity7_summary.json`, `activity6_summary.json`
- `issue468/dspark_train/run_stage2_head_lora.py`, `run_stage2_head_ceiling.py`, `run_stage2_final_eval.py`
- `issue468/dspark_train/drafter_head.py`, `drafter_body.py`
- `issue468/artifacts/stage2_crossed_oracle/summary.json`

VERDICT UNDER REVIEW: NOT-JUSTIFIED. Head-only LoRA (rank 32/64/128) on the pretrained
drafter head, trained on 21,543 frozen pre-head features (torch MPS body) to predict the
p=1 target, raises held-out p=1 from 0.8118 to 0.8179 (+0.60 pp; McNemar p=0.012, real
but tiny). Rank sweep is FLAT across rank. Body/input-side LoRA (Activity 8) NOT run:
its trigger (Activity 7 < Activity 6 from-scratch ceiling) is unmet (0.8186 > 0.7479),
and Activity 4 (crossed oracle) shows the drafter is input-invariant at p=1.

CLAIMS TO CHALLENGE (verify, do not trust):
- V1 The +0.6 pp is the real head-LoRA ceiling (not under-trained). Training was 12
  epochs AdamW lr 3e-4, p=1 CE only. Could more epochs / higher lr / the full K-position
  teacher-forced loss (DSpark §3.3) / more data materially raise it? Is the rank-sweep
  flatness really "information-limited" vs a training-config artifact?
- V2 The from-scratch ceiling 0.7479 < pretrained 0.8118 means under-generalization
  (not an info deficit). Overfit-to-eval 0.9947 says h is informative. Sound?
- V3 Activity 8 (body LoRA) skip is justified (7>6 ceiling + Activity 4 input-invariant).
  Is skipping body LoRA premature? The 7>6-ceiling trigger is an odd proxy — body LoRA
  could help even if head LoRA beats the from-scratch ceiling. Does Activity 4's
  input-invariance (at p=1, frozen drafter) actually rule out body-LoRA gains?
- V4 Self-consistency + float32 BLAS chaos is a valid basis for the RELATIVE comparison
  (frozen→finetuned, both torch). Does the chaos (+ the 3.75pp torch-vs-numpy offset)
  threaten the +0.6 pp signal or the verdict?
- V5 The McNemar (p=0.012) with a +0.6 pp effect is correctly interpreted as "real but
  not material" (below the +1pp / +5pp gates).
- V6 The p=1-only training/eval is a fair proxy for acceptance (the drafter predicts K=5;
  training only p=1 might understate or misstate the gain).

YOUR MANDATE — find anything that could flip NOT-JUSTIFIED to REAFFIRM or INCONCLUSIVE.
Prioritize runnable checks (re-train with different config on the existing features;
inspect the loss curves; check if the head-LoRA overfit-to-eval being only 0.82 is a
capacity bug). Not exhaustive:
1. Re-train head-LoRA with a stronger config (more epochs, higher lr, or the full K-loss)
   on the existing features — does p=1 move materially? (The features are on disk; the
   harness is run_stage2_head_lora.py.)
2. Is the rank-sweep flatness a TRAINING bug (e.g., LoRA not actually receiving grad, or
   the markov LoRA delta saturating)? Verify the LoRA params change during training.
3. Could the chaos/float32 offset make the +0.6 pp a noise artifact? Check the per-seed
   variance (train 2-3 seeds).
4. Is Activity 8 (body LoRA) worth running despite the 7>6 trigger? The body is already
   ported; a quick body-LoRA rank-32 run on the existing features would settle it.
5. Is the from-scratch ceiling under-trained (25 epochs)? Does it rise with more epochs,
   changing the 7-vs-6 comparison?

CONSTRAINTS: read-only; run python -c / inline scripts. The features + harnesses are on
disk. GGUFs at /Users/lobanov/Projects/ds4/gguf/.

OUTPUT (terse, evidence-based):
## Verdict soundness (V1..V6: sound / questionable / likely-wrong + 1-line why)
## Highest-risk hole that could flip the verdict (the one test that would settle it)
## Concrete experiments to run (ranked; each: what, expected signal, effort)
## Bottom line: does NOT-JUSTIFIED survive, or is a key experiment missing?
