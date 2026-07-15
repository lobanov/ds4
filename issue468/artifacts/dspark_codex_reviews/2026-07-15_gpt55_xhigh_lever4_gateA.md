**Findings**

1. **The net loss is not inherent.** The expensive path is partial-prefix replay, not just “anchor moved into batched verify.” In [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:29929), partial accepts fall back to replaying `drafts[0..commit_n-1]` with `metal_graph_eval_token_raw_swa_top`. With anchor reuse, almost every miss after the anchor becomes a partial accept, so the anchor often gets batch-verified and then re-decoded for commit.

2. **Evidence from the artifact:** full accepts are fine. `verify_n=5, verified=5` is about 71 ms in both `metal_sts.jsonl` and `metal_anchor_sts.jsonl`. The bad cases are partial accepts: anchor `verify_n=5, verified=3/4` is about 149/176 ms, matching batched verify plus ~25.5 ms per replayed token. Replacing partial-replay timings with same-`verify_n` full-accept timings drops cycle `verify_ms` from ~81.8 ms to ~56.5 ms.

3. **`verify_n=3.57` is correct.** The Metal STS path schedules continuation length, then adds `anchor_off` in [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:29787). The continuation scheduled length is ~2.57, so total ~3.57. I do not see evidence of meaningful STS over-verify from stale conf logits.

4. **Refresh is not running on the full verify span.** `metal_graph_dspark_refresh_verified_rows` is called with `verified`, not `verify_n`, at [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:29980). However, the batch hidden capture runs for all `verify_n` rows during target verify, including anchor and rejected rows. That cost exists, but it is not the main doubling.

5. **There is a likely anchor-reuse state bug in Metal `dspark_n_real` advancement.** The post-cycle update still assumes “anchor is always decoded standalone” in [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:30122). With anchor reuse, `verified` already includes the anchor, but the code advances `metal_base_real + 1 + verified` in [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:30128). I would expect `metal_base_real + verified` for anchor-reuse. Target output remains verified, but drafter KV position/state can drift and hurt future drafts.

6. **Guard removal is output-correct, but incomplete.** The `anchor_off` array handling looks right, and the target verifier preserves generated-token correctness. But cold/rebuilt prefill does not clearly populate `dspark_metal_main_hidden`; it is populated by standalone decode capture in [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:17474) or refresh `keep_last_hidden` in [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:29363). Since anchor reuse skips standalone decode, the first Metal draft after reset may use stale drafter hidden until a refresh fixes it.

**Answer**

Refute the “inherent net loss” claim. The observed loss is mostly an implementation cost: anchor reuse converts many cheap reject-0 cases into partial accepts, and the current partial-commit path replays accepted tokens sequentially.

A fix can plausibly make it a win. The best lever is cheap partial-prefix commit for `verify_n >= 3` via the existing prefix-checkpoint path, plus fixing the anchor-reuse `g->dspark_n_real` advancement. Skipping anchor refresh is not the main lever; refresh is small and the anchor hidden is needed because the standalone decode was skipped.