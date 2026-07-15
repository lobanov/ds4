**Findings**

- **Medium: threshold gate is env-based, not actual-path based.**  
  [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28577) makes `dspark_schedule_threshold()` return `0.15` whenever `DS4_DSPARK_DRAFT_METAL_STS` is set. But the actual Metal drafter path is separately gated by `draft_metal = dspark_draft_metal_enabled() && dspark_verify_batched_enabled() && draft_eval_n > 0` at [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:29721). So if `DS4_DSPARK_DRAFT_METAL_STS=1` is set without the Metal drafter actually running, the CPU scheduled drafter can get the Metal threshold. That violates the stated “CPU keeps 0.08” invariant. The fix would be to choose the threshold from the active scheduling path, not just the STS env flag.

- **Low: b6 diagnostics are capped at 5 entries.**  
  `ds4_spec_bench.c` serializes exactly five `conf_logits` and five `draft_ids` at [ds4_spec_bench.c](/Users/lobanov/Projects/ds4-dspark-research/ds4_spec_bench.c:1351) and [ds4_spec_bench.c](/Users/lobanov/Projects/ds4-dspark-research/ds4_spec_bench.c:1386). In `long_full_b6.jsonl`, cycles with `drafted=6` still emit arrays of length 5. This does not invalidate the timing result, but it means the b6 trace cannot fully audit the sixth token’s draft id from JSON alone.

**Gate Result**

- Block revert looks correct: `DS4_DSPARK_BLOCK` is back to `5` at [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:435), and `DS4_DSPARK_MAX_BLOCK` is back to `5` at [ds4.h](/Users/lobanov/Projects/ds4-dspark-research/ds4.h:58). Current five-entry STS temp table is consistent with that at [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:24078).

- Draft-6 conclusion is sound. I recomputed the long-context artifacts: mean t/s is `36.01` for block 5 vs `35.10` for block 6; paired prompts are 6 losses / 3 wins. In b6, `verify_n=6` occurs `107/413 = 25.9%`, while `verified=6` occurs `43/413 = 10.4%`, matching your claim. It pays the sixth verification much more often than it gets the sixth accepted token.

- Threshold conclusion is directionally sound but marginal. The cost argument is correct: lower fixed cycle overhead reduces the value of over-verifying to amortize the cycle, so a higher threshold is plausible. But the artifacts present here only include `full_stack.jsonl` and `full_stack_recal.jsonl`, not the named `full_thr*.jsonl` sweep files. The present rebench shows `39.85` mean t/s for recal vs `40.04` for prior full stack, so I would keep the conclusion as “not robust / marginal,” not “proven win.”

Net: no block-revert bug found. The one actionable issue is tightening the threshold gate so `0.15` is used only when the Metal STS path is actually active.