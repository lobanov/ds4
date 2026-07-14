## Verdict Per Claim C1-C5

C1: sound. `ds4-eval` parses `--dspark` into config, passes it to engine options, but generation samples one token then calls plain `ds4_session_eval()` at [ds4_eval.c:3872](/Users/lobanov/Projects/ds4-dspark-research/ds4_eval.c:3872). `rg` found no `ds4_session_eval_speculative_argmax` / `DS4_DSPARK_VERIFY_BATCHED` in `ds4_eval.c`.

C2: sound, with fresh-run caveat. `ds4-spec-bench` `speculative_argmax` calls `ds4_session_eval_speculative_argmax()` at [ds4_spec_bench.c:1108](/Users/lobanov/Projects/ds4-dspark-research/ds4_spec_bench.c:1108). Batched verify is env-gated at [ds4.c:28332](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28332) and calls `metal_graph_verify_suffix_tops()` at [ds4.c:28762](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28762). Retained engagement log has 303 `dspark timing drafted=...` lines and all detail lines show `verify_decode=0.000`, e.g. [benchmark_q_batched_engagement.err:12](/Users/lobanov/Projects/ds4-dspark-research/issue468/artifacts/dspark_m3_bench/benchmark_q_batched_engagement.err:12). My fresh run was blocked: after weight warm/load, this sandbox reports `Metal device not available`.

C3: sound for the decisive parts I could re-diff. From retained token dumps: benchmark_q total is `435/768 = 56.6%`; first three prompts: `q00 53/64`, `q01 50/64`, `q02 32/64`. `code_topk` first 48 is `45/48`, first diff index `3`, `3696 -> 4085`; full retained 64-token dump is `61/64`.

C4: sound by code path, not freshly rerun here. With batched off, code falls through to exact sequential verification at [ds4.c:28848](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28848), checks `target_top != drafts[i]` before commit at [ds4.c:28851](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28851), then advances with target decode. Retained summary says seq-vs-plain is 0%, but I could not create fresh dumps due Metal unavailability.

C5: sound. Divergent outputs are readable continuations, not garbage. `code_topk` plain is a docstring/import-plan fragment; batched is a plausible self-test/example plus Python function start at [plain.txt:1](/Users/lobanov/Projects/ds4-dspark-research/issue468/artifacts/dspark_m3_bench/code_topk/plain.txt:1) and [batched.txt:1](/Users/lobanov/Projects/ds4-dspark-research/issue468/artifacts/dspark_m3_bench/code_topk/batched.txt:1).

## Is The 92Q Byte-Identical With DSpark Engaged?

No. The 92Q `ds4-eval` result is not evidence with speculative DSpark engaged. It is plain decode with drafter loaded. After stripping timestamps/elapsed fields, the two traces compare equal, but `ds4-eval` never enters the speculative call path.

## Inherent Batched Flip Or Wiring Bug?

Leading hypothesis: inherent batched-verify argmax/logit flip, not accept-prefix wiring. Row mapping is correct: `top_rows = n_tokens - 1` at [ds4.c:21660](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:21660), and caller compares `drafts[i]` to `row_tops[i-1]` at [ds4.c:28769](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28769). The decisive test: shadow every batched commit with exact sequential verify/logits, then rerun with exact fallback on any row-top/logit near-tie mismatch. If byte identity returns, this is the batched flip.

## Final Verdict

NO: committing batched verify is not functionally equivalent to plain decode when actually engaged. Confidence: medium-high. Main caveat: I could not fresh-rerun Metal in this sandbox, so empirical divergence is re-derived from retained token dumps plus code, not newly generated output.