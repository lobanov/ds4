You are a skeptical senior engineer doing an ADVERSARIAL independent review. A
previous investigator reached a provisional conclusion about the current DSpark
runtime path; you must challenge it. Re-derive claims from the code and summary;
run read-only commands and inline scripts where useful. Be terse.

WORKTREE: /Users/lobanov/Projects/ds4-dspark-research

READ FIRST:
- issue468/summaries/dspark_runtime_initial_benchmark.md

ALSO READ:
- ds4.c
- issue468/summaries/confidence_scheduled_verification.md
- issue468/summaries/spec_speedup_model.md
- issue468/run_dspark_exactness_compare.py
- issue468/run_dspark_temp_distribution_compare.py
- issue468/run_dspark_corpus_bench.py
- issue468/run_dspark_phaseA_profile.py

RECAP:
The current implementation adds a DSpark speculative path behind `--dspark`,
with CPU drafter body/head, confidence scheduling, and a recent GPU-first hidden
capture / `main_proj` / stage-KV push slice. Greedy exactness and a conservative
temp>0 logit-parity gate pass, but runtime is still below baseline. The current
summary says the path is now a credible optimization substrate and recommends a
DSpark-specific GPU drafter body/head as the next major step.

THE INVESTIGATOR'S CLAIMS (verify, do not trust):
- The current implementation has moved past "mostly pointless CPU waste"; fixed
  `verify_k=1` around ~29 t/s and scheduled ~23-26 t/s mean the next major
  lever is a DSpark GPU drafter body/head path rather than more CPU cleanup.
- The current bottleneck picture is: target decode still ~28 ms, fixed-k draft
  ~9-10 ms, and scheduled verifier often ~27-56 ms once the path survives
  deeper.
- The best GPU migration path is to keep the exact verifier path intact and move
  DSpark hidden/state update, 3-stage block body, and head/confidence/argmax to
  GPU behind parity gates against the CPU drafter.
- Confidence scheduling as documented in
  `issue468/summaries/confidence_scheduled_verification.md` should remain part
  of the DSpark design even though the original MTP drafter lacks a confidence
  head.

PREMISES / SCOPE:
- Treat retained greedy exactness and conservative temp>0 parity artifacts as
  genuine for this review; you may still challenge their scope.
- Focus on optimization opportunities, bottleneck attribution, and best path
  for GPU migration. Do not spend effort on product/UX concerns.

YOUR MANDATE:
1. Reconstruct the actual runtime pipeline from code. Name where the CPU/GPU
   boundaries are today and which timing buckets include which work.
2. Find optimization opportunities that are still open in the current code,
   including CPU-side work that may still matter before or alongside GPU
   migration.
3. Challenge the current "best next step" recommendation. If a narrower or
   different GPU migration slice is better, say so and explain why.
4. Assess how confidence scheduling should shape the GPU design. Distinguish
   what needs GPU support from what can remain host policy.
5. Identify any benchmarking/methodology weaknesses that could mislead the
   optimization roadmap.
6. Prioritize recommendations by expected impact and implementation risk.

CONSTRAINTS:
- Do not modify tracked files.
- You may run read-only commands and inline scripts.

OUTPUT (terse, evidence-based):
## Verdict per current claim (sound / questionable / likely-wrong + 1-line why)
## Runtime decomposition as implemented now
## Ranked optimization opportunities
## Best GPU migration path
## Confidence-scheduling implications for design
## Benchmark / methodology risks
## One recommended next implementation step
