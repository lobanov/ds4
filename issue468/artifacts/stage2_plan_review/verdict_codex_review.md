## Verdict soundness

V1 **likely-wrong**: current “head-LoRA” is effectively norm-only. Both LoRA factors are zero-init in [drafter_head.py](/Users/lobanov/Projects/ds4-dspark-research/issue468/dspark_train/drafter_head.py:49), then used as `B @ A`; zero/zero gives zero gradients. Runnable check output: all `lora_*_A/B grad_max=0 delta_max=0`; only `lora_norm` moved. Rank flatness is therefore a training bug, not evidence of an information ceiling.

V2 **questionable**: Activity 6 is internally inconsistent. Saved gate says `"h lacks info -> consider 8"` in [activity6_summary.json](/Users/lobanov/Projects/ds4-dspark-research/issue468/dspark_train/data/activity6_summary.json:5), while the verdict says proceed to 7 / not info deficit in [stage2_finetune_result.md](/Users/lobanov/Projects/ds4-dspark-research/issue468/summaries/stage2_finetune_result.md:32). Overfit-to-eval proves fit/memorization capacity, not held-out recoverability.

V3 **likely-wrong**: skipping Activity 8 is premature. No body-LoRA runner exists; [drafter_body.py](/Users/lobanov/Projects/ds4-dspark-research/issue468/dspark_train/drafter_body.py:83) is frozen and `build_body` has no LoRA args. A4 is only 54 aligned cells at frozen p=1 in [summary.json](/Users/lobanov/Projects/ds4-dspark-research/issue468/artifacts/stage2_crossed_oracle/summary.json:11), not a proof body adaptation cannot help.

V4 **questionable**: self-consistent torch relative eval is basically valid, but scripts use unseeded `torch.randperm` in [run_stage2_head_lora.py](/Users/lobanov/Projects/ds4-dspark-research/issue468/dspark_train/run_stage2_head_lora.py:51). Protocol required a data/seed sweep before NOT-JUSTIFIED in [stage2_finetune_protocol.md](/Users/lobanov/Projects/ds4-dspark-research/issue468/summaries/stage2_finetune_protocol.md:32).

V5 **sound numerically**: McNemar p recomputes to `0.0120915`, net `+43/7137`; effect is real but below gates. But JSON text saying “McNemar not significant” is wrong in [activity9_final.json](/Users/lobanov/Projects/ds4-dspark-research/issue468/dspark_train/data/activity9_final.json:15).

V6 **questionable**: p=1 is preregistered primary, but the intended DSpark-style loss is K-weighted `Lce+Ltv` in [stage2_finetune_protocol.md](/Users/lobanov/Projects/ds4-dspark-research/issue468/summaries/stage2_finetune_protocol.md:70). Actual features/harness keep only `target_p1` and train `x[:,0]` CE.

## Highest-risk hole that could flip the verdict

Fix LoRA init, rerun head-LoRA. Standard init should be one factor random and the other zero, not both zero. Current +0.6 pp is not a head-LoRA ceiling; it is mostly `lora_norm` tuning.

## Concrete experiments to run

1. Correct LoRA init, rerun rank 32/64/128 p=1 CE, 3 seeds. Signal: if >1 pp, current NOT-JUSTIFIED fails; if >5 pp, REAFFIRM. Effort: low.
2. Add a param-delta assertion to the harness. Signal: all LoRA A/B factors must have nonzero delta after first epoch. Effort: trivial.
3. Run K=5 teacher-forced weighted CE + top-128 TV from shards. Signal: p=1 plus prefix hist / expected accepted tokens. Effort: medium.
4. Implement/run body LoRA despite the 7>6 proxy. Signal: tests the actual A6 “consider 8” path. Effort: medium/high.
5. Recheck from-scratch ceiling with seed/lr/epoch sweep. Signal: separates undertraining from real held-out ceiling. Effort: medium.

## Bottom line

Current NOT-JUSTIFIED does **not** survive adversarial review. The key head-LoRA experiment is invalidated by dead LoRA factors, and Activity 8 was skipped despite the saved A6 gate pointing toward it. The defensible verdict is **INCONCLUSIVE pending corrected LoRA + seed/loss sweep**.