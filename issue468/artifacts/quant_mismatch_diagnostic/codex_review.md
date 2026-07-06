## Verdict Per Claim

- **C1 FIDELITY: questionable.** The 0 mismatch claim is true from CSV vs retained summaries, but it is not independent: `run_stage0` reuses the same model/drafter loaders and forward path as `measure_acceptance_bundle`.
- **C2 MARKOV EXACT: sound internally.** `base_logits + markov_w1[prev] @ markov_w2.T` matches `forward_head`’s own logic at [forward.py](/Users/lobanov/Projects/ds4-dspark-research/issue468/dspark_oracle/forward.py:101); artifact completion implies the 400-position assert passed.
- **C3 ANCHORS: sound arithmetic, weak validation.** Recomputed: p1 = 65/80 = 0.8125; k=1 prefix hist `{0:15,1:13,2:18,3:11,4:10,5:13}` gives E=2.3375. But both derive from the same retained acceptance corpus.
- **C4 RANK COMPUTATION: mostly sound.** `count(score > score[target])` is correct 0-index strict rank; all target ids are in-vocab, no match/rank contradictions. But full scores are not persisted, so one miss cannot be independently recomputed from artifacts.
- **C5 ALIGNMENT: sound.** Verified all `target_topk.steps[s].selected.id == target_selected_tokens[s]`, top1 matches selected for temp=0, and row target equals `selected[step+p]`. This matches [measure_acceptance_bundle.py](/Users/lobanov/Projects/ds4-dspark-research/issue468/dspark_oracle/measure_acceptance_bundle.py:199) and [run_stage0_quant_mismatch.py](/Users/lobanov/Projects/ds4-dspark-research/issue468/run_stage0_quant_mismatch.py:169).
- **C6 INTERPRETATION: questionable / overclaimed.** Shallow p1 ranks are real, but “therefore quant-flip-like and fine-tuning-recoverable” is not established. Only 4/15 p1 misses have Q2 gap <1 nat; median Q2 gap is 2.1506, which cuts against a pure near-tie quant-flip story.
- **C7 COUNTERFACTUAL: questionable semantics, sound arithmetic.** k=1 exactly equals current retained prefix distribution; k=2 E=2.875 is verified. But “greedy-spine top-k” is not a true acceptance ceiling for fine-tune or tree deployment.

## Premise Sensitivity

- **P1:** Under FP-distillation premise, shallow Q2 misses motivate testing mismatch. If false, the quant-mismatch narrative collapses; the data only says “Q2 target often appears near the drafter top.”
- **P2:** Under IQ2XXS + temp=0 premise, this is a valid argmax diagnostic. If deployment includes sampling/real post-rejection trajectories, headline acceptance weakens: retained E[a|4] drops from 2.175 at temp=0 to 1.925/1.9125 at temp 0.5/1.0.
- **P3:** Under speedup-model validity, the anchors align. If it fails, the rank-shape diagnostic survives, but “clears toward speedup” does not.

## New Experiments To Try

1. **FP-vs-Q2 topk on the exact p1 misses.** Run FP teacher logits for the same `step+1` positions; check whether drafter pick matches FP top1 while Q2 picks target. Expected signal: directly separates quant flip from drafter error. Effort: medium.
2. **Independent production drafter checksum.** Export top-10/full-rank checks from the actual ds4/Metal drafter for a few retained positions. Expected: kills the shared-bug risk. Effort: medium/high.
3. **Persist full decision scores or top-64 scores.** Add artifact for `base`, `bias`, `full_score topN`, and exact target score. Expected: lets reviewers recompute ranks without loading the model. Effort: low.
4. **Stage 1 tap-precision acceptance.** Run the planned precision variant and compare p1, E[a|4], and p1 miss rank conversion. Expected: tests localization, not full quant-causality. Effort: medium.
5. **Larger/temp-varied corpus.** Repeat on >100 prompts and temp 0/0.5/1.0. Expected: narrows CI and exposes prompt/domain instability. Effort: medium/high.

## Leading Hypothesis

This looks like a **mixture**: some genuine shallow/near-tie mismatch, plus ordinary or systematic drafter error. The median rank=1 is real, but the large median Q2 gap says many misses are not obvious quant flips.

Decisive test: **FP teacher topk on the exact p1 miss set**. If FP top1 equals the drafter pick and Q2 top1 equals the retained target, quant mismatch is confirmed. If FP and Q2 both prefer the retained target, the drafter is just wrong.

## Bottom Line

“Proceed to Stage 1” survives as a cheap next experiment.  
“Stage 0 clears because the error is overwhelmingly perturbative / fine-tuning-recoverable” does **not** survive as stated. Downgrade it to: p1 misses are shallow on this small temp=0 corpus, but causality and recoverability remain unproven.