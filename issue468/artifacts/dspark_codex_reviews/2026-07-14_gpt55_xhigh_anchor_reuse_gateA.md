## Verdict Per Claim

C1: sound. Guard checks envs + batched verify + non-EOS + room + exact live argmax comparison at [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28630); callers generally pass `ds4_session_argmax/sample` from `s->logits` ([ds4_spec_bench.c](/Users/lobanov/Projects/ds4-dspark-research/ds4_spec_bench.c:1099), [ds4_eval.c](/Users/lobanov/Projects/ds4-dspark-research/ds4_eval.c:3862)). Forced/non-greedy tokens fall back.

C2: sound. `anchor_off` clamps continuation writes to `DS4_DSPARK_BLOCK - anchor_off` before passing `drafts + anchor_off` ([ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28677), [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28682)); helpers write `0..draft_n-1` only.

C3: sound, with one corner. In normal reuse verify, `drafts[0]=first_token` and `target_top=first_token`, so `commit_n` starts at 1 ([ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28678), [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28749), [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28809)). Corner: `DS4_DSPARK_VERIFY_K=0` returns `0` before verify/commit under reuse ([ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28654)).

C4: sound. I find no path where reuse commits anchor to output/KV but misses DSpark window push. See trace below.

C5: questionable. 8k acceptance/speed claim is artifact-supported: reuse continuation accepted ~= `4.1809-1=3.1809` vs batched verified `3.2495`. Gate claim is not independently verifiable from retained artifacts; no reuse ds4-eval trace is present, only prose. Q9 pass, if true, is divergent-output luck, not exactness evidence.

C6: sound for bounds. `dspark_conf_logits` is length 5 ([ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:26768)); scheduler reads `i < max_n` only ([ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28386)); reuse passes continuation count `draft_eval_n <= 4` ([ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28726)). Telemetry `last_cycle.conf_logits` is shifted/stale for `drafts[0]`, but scheduling is not.

C7: sound for drafter failures. The three helper-fail branches do standalone `ds4_session_eval(first_token)` then return anchor ([ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28687), [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28694), [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28701)); normal eval pushes DSpark hidden at [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:27835).

## Window-Consistency Trace

Full accept: batch verifier captures layers 40-42 per row ([ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:19680)); full accept pushes each accepted row via `dspark_session_push_batch_hidden(s,i)` before appending to `accepted[]` ([ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28813), [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28820)).

Prefix-1 partial: `commit_n==1 && verify_n==2` commits prefix-1 cache, reads row-0 logits, pushes batch hidden row 0, then records anchor ([ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28833)). Window OK; dump labels may be misleading because push precedes `token_vec_push`.

General partial: restores snapshot, replays accepted prefix sequentially, `token_vec_push`, then `dspark_session_push_graph_hidden` per token ([ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28842), [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28847), [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28850)).

Reject: `commit_n==0` only restores frontier and commits nothing ([ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28863)); unreachable for reuse unless memory/state is already broken.

Sequential fallthrough: if batched verify fails non-mutatingly, it restores then sequentially decodes each accepted draft, pushes token, and pushes graph hidden ([ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28874), [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28887), [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28905)).

## Experiments To Try

1. Add a read-only debug counter/assert: on every reuse cycle, assert `pushes_verify >= accepted` and `draft_ids[0]==anchor_id`; run tiny `ds4-spec-bench`. Decisive for hidden-window leak. Low effort.

2. Re-run first-20 ds4-eval with trace saved for reuse. Expected signal: prove or disprove Q9 pass and zero pass-to-fail. Medium effort, slow.

3. Run reuse with `DS4_DSPARK_VERIFY_K=0`. Expected: current code returns 0/stalls; confirms corner bug. Low effort.

4. Run fixed `VERIFY_K=1` reuse smoke. Expected: anchor-only batch verify, no OOB, push count 1/cycle. Low effort.

I attempted the requested live smoke, but this sandbox blocks the binary at `/tmp/ds4.lock` creation. I did run code parsing the retained JSONL artifacts.

## Leading Hypothesis

Anchor reuse is structurally window-safe; the bigger risk is not missing anchor hidden, but relaxed batched-verify divergence being laundered as “gate improvement.”

Decisive test: a reuse ds4-eval first-20 trace plus per-cycle assertion that every returned anchor increments DSpark push count exactly once.