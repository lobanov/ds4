## Verdict Per Claim P1..P8

- **P1: likely-wrong** — `<+1pp ⇒ irreducible by drafter fine-tuning` overclaims; it only falsifies this bounded frozen-expert LoRA recipe.
- **P2: questionable** — 2a is a good cheap probe, but 2b confounds “input-side adaptation” with “more trainable capacity.”
- **P3: likely-wrong** — frozen top-2 coverage is **not** a head-only ceiling. Stage 1 already says it is headroom, not a hard ceiling: [stage1_tap_precision.md](</Users/lobanov/Projects/ds4-dspark-research/issue468/summaries/stage1_tap_precision.md:73>).
- **P4: likely-wrong** — frozen experts cannot falsify “drafter fine-tuning” broadly; plan itself freezes MoE bulk: [stage2_finetune_protocol.md](</Users/lobanov/Projects/ds4-dspark-research/issue468/summaries/stage2_finetune_protocol.md:66>).
- **P5: questionable** — few-file storage is fine in principle, but comma-list dump is not “trivial+safe”; dumps synchronize/restart command batches: [ds4.c](</Users/lobanov/Projects/ds4/ds4.c:10816>), current layer gate is single `strtoul`: [ds4.c](</Users/lobanov/Projects/ds4/ds4.c:10827>).
- **P6: likely-wrong** — argmax-token parity is necessary, not sufficient; training losses need logits/top-k/markov-score parity.
- **P7: questionable** — planned ~3.6k is fine for +5pp; “≥1k detects” is only robust under low churn.
- **P8: questionable** — hard CE is reasonable for greedy argmax, but label smoothing 0.1 is suspect; DSpark training uses CE + distribution matching/TV and position weights: [DSpark_paper.md](</Users/lobanov/Projects/ds4-dspark-research/issue468/ref/DSpark_paper.md:175>).

## Open-Questions Recommendations

1. **Soft vs hard:** use hard CE primary with **0 or ≤0.02 smoothing**, plus top-k KL/TV as co-primary ablation. Confidence: high.
2. **Frozen experts:** accept only as “dense/head LoRA boundary,” not “any fine-tune.” Add Stage 2c only if you need a real stop decision. Confidence: high.
3. **LoRA rank:** rank 32 is okay as first run, but pre-register rank 64/128 rescue before declaring null. Confidence: medium.
4. **Crossed oracle:** mandatory before training; it is cheaper and more diagnostic than 2b. Confidence: high.
5. **Generation length:** prefer more eval prompts over longer continuations; if capture is cheap, use 60 eval prompts or 30×256. Confidence: medium.
6. **K weighting:** p=1-heavy or exponential decay, not equal; p=1 is decision metric and highest leverage. Confidence: high.
7. **Temp>0:** temp=0 primary is fine for exactness, but add temp>0 eval if deployment samples; prior temp 0.5/1.0 lowers prefix length: [exactness_small_bundles_and_oracle_acceptance.md](</Users/lobanov/Projects/ds4-dspark-research/issue468/summaries/exactness_small_bundles_and_oracle_acceptance.md:77>). Confidence: medium-high.
8. **Throughput coupling:** keep acceptance-only scope; no throughput reaffirm without anchor-reuse verifier, which is unbuilt and load-bearing: [spec_speedup_model.md](</Users/lobanov/Projects/ds4-dspark-research/issue468/summaries/spec_speedup_model.md:48>). Confidence: high.

## Methodological Bugs / Risks Found

1. **Conclusive falsification overclaim.**  
   Fix: change falsify wording to “bounded frozen-expert LoRA did not justify scale-up”; require rank/loss/data/seed sweeps before “stop.” Effort: small.

2. **False head-only ceiling.**  
   A retrained head can move rank-5→rank-1 if frozen features contain the information.  
   Fix: train a from-scratch linear/markov head on frozen `h` to convergence and report held-out upper bound. Effort: small-medium.

3. **2b does not isolate input-side.**  
   Fix: ablate `head-only`, `main_proj-only`, `dense-block-only`, `head+main_proj`, `head+blocks`, with similar rank budgets. Effort: medium.

4. **Metric too narrow for deployment.**  
   p=1 is clean, but deployment depends on rollout prefix, `E[a|K]`, verifier cost, and post-rejection states.  
   Fix: report p=1 primary plus prefix hist / `E[a|4,5]` secondary under same speed model. Effort: small.

5. **Storage schema incomplete for soft losses.**  
   `topk_logits` alone is weaker than existing `--dump-logprobs`, which emits logprob after full-vocab normalization: [ds4_cli.c](</Users/lobanov/Projects/ds4/ds4_cli.c:755>).  
   Fix: store `topk_logprobs`, selected token, seed row, prompt offsets, and explicit no-cross-prompt windowing. Effort: small.

6. **Capture parity risk.**  
   Current proven path runs layers separately: [run_exactness_small_bundles.py](</Users/lobanov/Projects/ds4-dspark-research/issue468/run_exactness_small_bundles.py:131>).  
   Fix: golden test new multi-layer capture vs old per-layer dumps on 2 prompts, all layers/positions/tokens/top-k. Effort: small-medium.

7. **Fidelity gate too weak.**  
   Fix: require torch vs numpy full-score/top-128 parity, not just argmax; no-LoRA eval must reproduce baseline p=1/prefix hist. Effort: small.

8. **Corpus representativeness.**  
   Dolly/CodeAlpaca/json covers some exactness genres, but misses local chat, math/reasoning, long-context, and multi-turn.  
   Fix: make eval deployment-shaped; keep exactness corpus as locked legacy. Effort: small-medium.

## Cheaper / More-Decisive Alternatives

- Mandatory crossed hidden/label oracle before training.
- Frozen-feature from-scratch head upper-bound before LoRA.
- Broader Stage-0 rank diagnostic on the new eval corpus before any training.
- If available, Q8/BF16 target top-k on p=1 misses.
- Verifier anchor-reuse falsifier before interpreting acceptance as throughput.

## Bottom Line

As written, the plan can produce a useful **reaffirm** signal if held-out p=1 moves +5pp. It does **not** produce a trustworthy broad falsification; a null only says this small frozen-expert LoRA setup did not work on this corpus.

Single highest-value change before execution: add the frozen-feature from-scratch head upper-bound and make it a gate before LoRA. That directly fixes the false top-2 ceiling, tests whether `h` contains the missing information, and makes any later 2a/2b result much easier to interpret.