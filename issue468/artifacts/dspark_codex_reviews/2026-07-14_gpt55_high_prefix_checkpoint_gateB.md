## Verdict Per Claim

- F1: sound. Flag exists, reset at verify start, cleared in both aligned attn/index branches, and gates prefix commit. [ds4.c:10604](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:10604), [ds4.c:21757](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:21757), [ds4.c:18291](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:18291), [ds4.c:18609](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:18609), [ds4.c:29001](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:29001)
- F2: mostly sound, one caveat. Nonzero per-token compressed paths capture attn/index slots; aligned paths invalidate. Zero-prefix prefill skips capture without clearing, but normal DSpark verify starts from an existing frontier, not `pos0==0`. [ds4.c:18457](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:18457), [ds4.c:18748](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:18748), [ds4.c:18210](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:18210)
- F3: sound. `spec_prefix_capture_valid` is reinitialized per `metal_graph_verify_suffix_tops` call before the layer loop; no cross-cycle leakage. [ds4.c:21748](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:21748), [ds4.c:21761](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:21761)

- B1: sound with caveat. Matched artifacts show verify_ms cycle-weighted drops `82.89 -> 60.34`; cycles only `2445 -> 2402`, so not mainly fewer verifies. Acceptance moved slightly: prompt-mean verified `2.50 -> 2.56`, verify_n `3.59 -> 3.64`.
- B2: sound as measured. Short matched subset is genuinely slower vs plain: mean prompt t/s `27.95 / 38.20 = 0.73x`; 8k note reports `0.97x`. [progress:213](/Users/lobanov/Projects/ds4-dspark-research/issue468/summaries/dspark_runtime_milestone_3_progress.md:213), [progress:217](/Users/lobanov/Projects/ds4-dspark-research/issue468/summaries/dspark_runtime_milestone_3_progress.md:217)
- B3: relative speedup valid; absolute corpus claim biased. Same 93 OK prompts across configs, so `27.95 / 22.04 - 1 = +26.8%` is valid for that subset. But matched prompts are much shorter: mean prompt tokens `62` vs unmatched `163`; absolute t/s is not representative of all 176/300 attempted prompts. Error mechanism visible in JSONL: frontier > actual. [300p_plain.jsonl:2](/Users/lobanov/Projects/ds4-dspark-research/issue468/artifacts/dspark_m3_bench/large_corpus/300p_plain.jsonl:2)

## Fix-Completeness Trace

Aligned-cycle coverage is correct for production DSpark: block is 5, so aligned compressed verify means ratio-4 with `verify_n=4`. [ds4.c:435](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:435) Attn and index use the same condition, `pos0 % ratio == 0 && n_tokens % ratio == 0`; either one clearing the shared flag is enough. [ds4.c:18291](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:18291), [ds4.c:18609](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:18609)

Fall-through is correct: partial accept rewinds `checkpoint.len` to `bstart`; if prefix commit is skipped, `spec_frontier_restore(&bfrontier, s)` restores pre-batch compressor frontiers, then sequentially re-verifies each accepted draft. [ds4.c:28990](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28990), [ds4.c:29019](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:29019)

Caveat: zero-prefix compressed prefill path does not capture and does not invalidate. I do not see that reachable from normal DSpark verify, but the function itself is not locally self-protecting. [ds4.c:18210](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:18210), [ds4.c:18535](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:18535)

## Bench Soundness

The +26.8% is real for the matched subset. The strongest evidence is category timing: partial cycles drop `99.3ms -> 61.6ms`, while full cycles stay roughly flat `58.1ms -> 59.9ms`. That is exactly the expected replay-elimination signature.

The 0.73x vs plain is also real for the short matched subset, but the subset is biased short because frontier estimation rejected longer prompts. Treat the 0.73x as “short-context retained subset,” not “300-prompt corpus.”

## Leading Hypothesis + Decisive Test

Hypothesis: fix closes the stale-KV aligned path; remaining output/acceptance movement is inherited batched-verifier divergence, not stale checkpoint state.

Decisive test: add temporary counters for `aligned_invalidated`, `prefix_commit_taken`, and `replay_taken`, run one `DS4_DSPARK_VERIFY_K=4` prompt. Expected invariant: every aligned partial cycle has `aligned_invalidated=1`, `prefix_commit_taken=0`, `replay_taken=1`, and matches replay output for that cycle.