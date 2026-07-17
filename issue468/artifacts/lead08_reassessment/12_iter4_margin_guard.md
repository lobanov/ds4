# Lead 08 iteration 4 - margin-guarded exact fallback probe

Date: 2026-07-17. Status: **NO-GO; do not implement the margin fallback.** Data:
`12_iter4_margin_guard.csv`.

## Question

Can the existing fast but numerically divergent batch verifier preserve exact greedy output by
falling back to the exact sequential verifier only when the batch top-1/top-2 logit margin is small?
The predeclared thresholds were 0.25, 0.5, 1.0, and 1.75.

## Instrumentation and protocol

`DS4_DSPARK_VERIFY_DIST_PROBE=1` already runs the batch verifier non-committing beside the exact
sequential verifier on the same drafts. The retained probe now also records, for every relevant
compared row:

- the batch top-1/top-2 margin;
- whether each threshold would guard the row and trigger a cycle fallback;
- whether an observed batch-vs-exact argmax flip is above the threshold and would be missed.

The replay used the same ten exactness prompts and run parameters as
`dspark_m3_bench/distprobe_batched_vs_exact_fresh.jsonl`, reconstructed directly from that artifact.
The current run produced 163 speculative cycles; 158 ran a batch probe, 91 had an exact row to
compare, and 157 rows were compared. `DS4_DSPARK_TIMING=1` measured exact replay cost in the same
run. Raw logs remain in `/tmp`; only this compact decision artifact is retained.

Replay from the retained workload metadata:

```sh
jq -c '{id,mode,prompt_file,system,frontier_tokens,gen_tokens:.gen_tokens_requested,exclude_eos,seed,temperature,top_k,top_p,min_p}' \
  issue468/artifacts/dspark_m3_bench/distprobe_batched_vs_exact_fresh.jsonl \
  > /tmp/lead08_margin_guard_config.jsonl
DS4_DSPARK_VERIFY_DIST_PROBE=1 DS4_DSPARK_TIMING=1 ./ds4-spec-bench \
  --bulk-config /tmp/lead08_margin_guard_config.jsonl \
  --jsonl-out /tmp/lead08_margin_guard_full.jsonl \
  --model /Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --dspark /Users/lobanov/Projects/ds4/gguf/dspark.gguf --metal
```

The replay reproduces the prior tail: **one argmax flip**. Its batch margin is **0.375505**. Across
all compared rows, mean batch margin is 3.376 and minimum margin is 0.0100.

## Results

| threshold | guarded rows | guarded cycles | missed flips | optimistic added ms/all cycle | projected t/s | vs plain |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 11/157 | 11/158 (7.0%) | **1** | 2.304 | 38.75 | +1.6% |
| **0.50** | **27/157** | **27/158 (17.1%)** | **0 observed** | **7.006** | **36.37** | **-4.7%** |
| 1.00 | 50/157 | 49/158 (31.0%) | 0 observed | 13.665 | 33.45 | -12.3% |
| 1.75 | 69/157 | 63/158 (39.9%) | 0 observed | 17.271 | 32.06 | -16.0% |

The added cost sums `verify_decode_ms + push_verify_ms + logits_read_ms` only on guarded cycles and
divides by all 163 speculative cycles. It is optimistic: it excludes the GPU top-2/guard primitive,
readback or synchronization, rollback, and policy plumbing. The projection applies that added cost
to the live M3 stack (69.38 ms/cycle, 40.04 t/s) and compares against plain 38.16 t/s.

## Decision

**NO-GO.** Threshold 0.25 retains most of the speed but misses the known flip, so it fails exactness.
Threshold 0.5 is the smallest tested value that catches the observed flip, but exact replay on 17.1%
of probed cycles adds at least 7.0 ms/cycle and projects the stack below plain. Higher thresholds are
worse.

This is also not proof of exactness at threshold 0.5: only one flip was observed, and no analytic
bound shows that future batch-reduction errors cannot flip a row with a larger margin. A guaranteed
bound based on the observed global logit-error tail would be much more conservative. Therefore the
margin guard is neither economically viable nor proof-grade for exact greedy preservation. Retain
the extended dist-probe as instrumentation, but do not implement a committing fallback policy.
