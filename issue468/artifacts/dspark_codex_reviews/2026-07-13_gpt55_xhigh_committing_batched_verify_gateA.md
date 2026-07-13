**Verdicts**

C1: sound. `top_rows = n_tokens - 1`; `row_tops[i-1]` is logits after `drafts[i-1]`, so it verifies `drafts[i]`.

C2: questionable. Happy-path branches are recognizable, but restore failures are ignored and partial replay trusts approximate batched accept length.

C3: mostly sound. Batch row copy into `dspark_capture_hc` plus CPU fallback refresh is coherent; GPU push failure is unresolved perf risk.

C4: sound. Capture happens after `batch_cur_hc`/`batch_next_hc` swap, local rows `0..n_tokens-1`.

C5: likely-wrong as a general claim. Existing dist-probe artifact has `argmax_flips=1` on a verified accepted cycle; the new path uses those batched logits as continuation state.

**Premise Sensitivity**

P1: load-bearing. If committed-output exactness must hold beyond smoke, current evidence is insufficient and likely already contradicted by the retained flip artifact.

P2: survives. Scheduler only chooses `verify_n`; the indexing/control issues are independent.

P3: survives as framing, but performance projection is not validated. CPU fallback push plus extra memory can erase the intended gain.

**Bugs Found**

1. Critical: [ds4.c:28776](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28776), [ds4.c:28795](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28795). Batched `spec_logits` become authoritative next-token logits, but verifier is known non-bit-exact. Decisive evidence: [verify_dist_probe_exactness.jsonl:3](/Users/lobanov/Projects/ds4-dspark-research/issue468/artifacts/rejection_acceptance/verify_dist_probe_exactness.jsonl:3) has `code_topk`, cycle 0, `verified=2`, `accepted=3`, `argmax_flips=1`. Decisive test: run that prompt with `DS4_DSPARK_VERIFY_BATCHED=1` and token/text tracing; expect divergence after the flipped continuation.

2. High: [ds4.c:28805](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28805). General partial replay decodes `commit_n` tokens without rechecking exact `target_top` between drafts. Decisive test: force/mock a batched false-accept on row 0; exact replay should stop before draft 1, current code will not.

3. High, failure-path: [ds4.c:28825](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28825), [ds4.c:28842](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28842). `spec_frontier_restore` return is ignored on reject/fallback. Decisive test: fault-inject restore copy failure; code continues with corrupted target cache.

4. Medium: [ds4.c:11472](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:11472), [ds4.c:11566](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:11566). New `dspark_batch_capture_hc` buffers allocate unconditionally and are not included in allocation `ok`. Cost is about 768 MiB at prefill 4096, 1536 MiB at 8192.

**New Experiments**

1. Reproduce `code_topk` line 3 with committing batch enabled, no dist probe, and emitted-token tracing. Signal: exact first divergence. Effort: one model load plus trace plumbing if bench lacks token output.

2. Add debug-only shadow exact replay after every batched commit, compare continuation argmax and accepted prefix. Signal: direct false-commit count. Effort: moderate.

3. Compare DSpark hidden/window state for sequential vs batched accepted tokens using hidden/window dumps. Signal: capture/push correctness independent of final text. Effort: moderate.

4. Run retained bench with `DS4_DSPARK_TIMING=1 DS4_DSPARK_VERIFY_BATCHED=1`. Signal: whether CPU fallback push erases sublinear verifier gain. Effort: one bench pass.

**GPU Push Hypothesis**

Not unsupported GGUF types: `dspark.gguf` metadata shows `mtp.0.main_proj.weight` and stage `attn_kv.weight` are Q8_0. Leading hypothesis: `metal_graph_push_dspark_hidden` itself is failing in a Metal op or command buffer, and normal sequential DSpark also silently falls back.

Decisive test: temporary stage logging or strict mode around [ds4.c:19911](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:19911), run one normal DSpark push and one batch push, record the first failing op.