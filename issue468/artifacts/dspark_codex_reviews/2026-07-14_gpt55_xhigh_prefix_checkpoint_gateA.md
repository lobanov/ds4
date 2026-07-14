## Verdict per claim

- C1: sound. Slot math is correct: capture `slot=prefix_len-1` and commit `slot=accepted-1` align. [ds4.c:13592](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:13592), [ds4.c:24870](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:24870)
- C2: questionable. Timing is correct only in the per-token path; aligned compressor paths have no prefix capture calls. [ds4.c:18453](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:18453), [ds4.c:18290](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:18290)
- C3: conditional. Restore logic is right if the slot is fresh; it can restore stale/uninitialized ratio-4 slots after aligned `verify_n=4`. [ds4.c:24879](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:24879)
- C4: structurally sound. Branch order/gate are as claimed, but prefix path bypasses replay’s exact re-verification. [ds4.c:28985](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28985), [ds4.c:29016](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:29016)
- C5: mostly sound. Default-off is true; caveat: capture caches env once, commit checks env live. [ds4.c:21749](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:21749), [ds4.c:28446](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28446)
- C6: likely wrong. There is both a stale-slot path and an exact-target wrong-token path when batched row tops diverge.
- C7: questionable. The 20Q gate is score-level; it does not prove same committed tokens as replay.

## Correctness trace

No off-by-one in the intended path: after token `t`, code increments `layer_n_comp` then captures prefix `t+1` into slot `t`; commit of `accepted=a` restores slot `a-1`, so `n_comp` and attn/index state represent the prefix after token `a-1`. `spec_logits_row(a-1)` is also the logits after that prefix. Checkpoint rewind to `bstart` then pushing `a` drafts is consistent. [ds4.c:18453](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:18453), [ds4.c:13599](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:13599), [ds4.c:28987](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28987)

But stale KV is real for aligned ratio-4 batches: `aligned_chunk = pos0 % ratio == 0 && n_tokens % ratio == 0` runs batch compressor and never calls `metal_graph_capture_prefix_*`. With `DS4_DSPARK_BLOCK=5`, `verify_n=4`, `pos0%4==0` is reachable; artifacts show aligned partial examples, e.g. `pos=108, verify_n=4, verified=1`. [ds4.c:435](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:435), [ds4.c:18290](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:18290), [ds4.c:18607](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:18607), [benchB_batched_throughput.jsonl:7](/Users/lobanov/Projects/ds4-dspark-research/issue468/artifacts/dspark_m3_bench/benchB_batched_throughput.jsonl:7)

Wrong-token risk: old general-partial replay re-checks `target_top != drafts[i]` before committing; prefix-checkpoint commits `commit_n` from batched `b_row_tops` directly. Existing dist-probe shows nonzero batched-vs-exact argmax flips: 1/156 compared positions. [ds4.c:28963](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28963), [ds4.c:29018](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:29018), [distprobe_batched_vs_exact_fresh.jsonl:3](/Users/lobanov/Projects/ds4-dspark-research/issue468/artifacts/dspark_m3_bench/distprobe_batched_vs_exact_fresh.jsonl:3)

Fallback on logits-read failure is okay: prefix commit may mutate state, but failed `&&` falls to `spec_frontier_restore`, with `checkpoint.len` already rewound. [ds4.c:28997](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28997), [ds4.c:29015](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:29015)

Fresh smoke was blocked by sandbox, not code: `failed to open lock file /tmp/ds4.lock: Operation not permitted`.

## New experiments to try

1. Add a temporary per-slot valid/stamp assert in `spec_frontier_commit_prefix`; run `VERIFY_K=4` with prefix checkpoint. Expected: trips on aligned ratio-4 partial. Low effort.
2. Force per-token compressor path for prefix capture, rerun K=4 prefix-on vs replay-off token dumps. Expected: stale-slot divergence disappears. Medium.
3. Compare prefix-on vs replay path with exact sequential reverify on retained exactness corpus. Expected: residual divergences only from batched false accepts. Medium.
4. Log `commit_n`, `pos0%4`, and captured slots per layer. Expected: missing ratio-4 slot captures for aligned K=4. Low.

## Leading hypothesis

The port is off-by-one correct, but not correctness-safe. Two fixes are needed: make prefix slots valid for aligned compressor paths or disable prefix-commit there, and decide whether prefix-checkpoint is allowed to inherit batched verifier non-exactness. The decisive test is the slot-valid assert under forced `DS4_DSPARK_VERIFY_K=4`.