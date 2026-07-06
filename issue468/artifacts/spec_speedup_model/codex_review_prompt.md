ADVERSARIAL REVIEW — speculative-decode speedup model

You are reviewing a speculative-decoding speedup model for the "DSpark" drafter on
the "ds4" LLM engine (DeepSeek-V4-Flash target, MoE, Apple Silicon / Metal). Your
job is to be ADVERSARIAL: find concrete numerical/logical errors, load-bearing
unverified assumptions, and missing factors that could flip the conclusions. Do
not be polite, do not hedge, do not summarize what's right — attack it. Assume the
author is wrong until you can prove a step is correct.

FILES (read them; read-only):
- issue468/model_spec_speedup.py            — the model code
- issue468/summaries/spec_speedup_model.md  — the conclusions (Q1/Q2/Q3)
- issue468/summaries/mtp_verifier_bench_results.md        — empirical verifier timings used
- issue468/summaries/mtp_verifier_bandwidth_binding.md    — verifier bandwidth claim
- issue468/summaries/dspark_quantization_ceiling.md       — "acceptance is at the ceiling"
- issue468/summaries/exactness_small_bundles_and_oracle_acceptance.md — acceptance source

THE MODEL (corrected cycle accounting):
- Baseline decode: 26 ms/token (from 38.49 t/s plain baseline).
- DSPark draft: 10 ms/cycle (given assumption).
- Verifier cost verify_ms(K) = cross-prompt MEDIAN of a measured MTP bench on
  three 8k prompts: K=2:43.6, K=3:59.7, K=4:65.8, K=5:74.5, K=6:79.6 ms.
- The verify forward produces the correction token at the rejection point, which
  IS the next anchor. A fresh decode is needed ONLY on full-block acceptance.
  So E[cost/cycle] = 10 + verify_ms(K) + 26*S(K), where S(K)=P(first K drafts all
  match). E[tokens/cycle] = E[a|K] + 1. speedup = (E[a|K]+1)*26 / E[cost].
- Acceptance: greedy (temp=0) prefix histogram over a 5-token block from the
  numpy oracle on 10 prompts x 8 steps (80 cells). Survival S(1..5) =
  0.8125, 0.65, 0.425, 0.2875, 0.1625; E[a|K] = 1.4625/1.8875/2.175/2.3375 (K=2..5).

THE CONCLUSIONS TO STRESS-TEST:
- Q1: beating baseline needs ~79% per-position acceptance at K=4 (current ~77%,
  "only ~2 pp short"); the +20% gate needs ~94% (K=4) / ~89% (K=5).
- Q2: K=4 is at break-even (−0.9%) with the current drafter + an optimized
  verifier; ~2 pp more acceptance flips it positive.
- Q3: a 4-node draft tree cannot help — ceiling +2.2% (perfect acceptance),
  dominated by a linear chain; hedging is "doubly penalized" (raises the
  full-accept decode penalty).
- Headline claim: "the current drafter is ~2 pp of per-position acceptance away
  from beating baseline at K=4, IF the verifier reuses the verify-produced
  anchor."

ATTACK THESE SPECIFIC POINTS (and anything else you find):
1. "Verify produces the anchor" — is this actually realizable on the ds4 Metal
   graph? The note itself admits it's unverified. If the verifier does NOT expose
   a usable anchor hidden state at the rejection point, cost reverts to
   decode+draft+verify (the ~−19% model). How load-bearing is this, and is there
   code evidence either way? (read ds4.c: ds4_session_eval_speculative_argmax,
   metal_graph_verify_decode2_exact, metal_graph_verify_suffix_tops.)
2. The "+1 bonus token" — on rejection it comes from the verify forward. But does
   producing the target argmax at the rejection position require the OUTPUT HEAD
   (a ~545 MB Q8 vocab matmul) at that position? Is that output-head cost inside
   the measured verify_ms, or an ADDITIONAL cost the model omits? If omitted, by
   how much does it shift the thresholds?
3. verify_ms transferability — it was measured on the MTP path with
   DS4_MTP_TIMING. Does that timer include the output head, sampling, KV-cache
   writes, snapshot/replay, and command-buffer overhead? Is a DSpark verifier
   (batched suffix verify) genuinely the same cost as the MTP verifier? Could the
   real DSpark verify be cheaper (tree/batch attention) or more expensive (extra
   state capture)?
4. Acceptance measurement bias — the oracle drafts from ONE anchor and greedily
   rolls 5 ahead, comparing to the target's next 5. But in the real loop, a cycle
   that follows a REJECTION starts from the target's correction token (which the
   drafter did NOT predict). Acceptance from a "surprise"/mismatched token may be
   LOWER than from a drafter-predicted token. Does E[a|K] from the single-anchor
   measurement overstate per-cycle acceptance? Is there a first-draft-miss effect
   (the bench shows 17-50% first-draft misses) that this model ignores?
5. Geometric-p simplification — real acceptance decays faster than geometric with
   depth. The "current p≈0.77" and "need p≈0.79" are geometric fits. Does the
   non-geometric shape move the beat-baseline threshold? By how much?
6. 10 prompts at temp=0 — is the acceptance distribution stable enough to quote
   "2 pp short" to that precision? What is the across-prompt spread in E[a|K]?
   (check the exactness_small_bundles per-prompt numbers.)
7. decode_ms=26 constant — decode cost grows with context (KV grows, SWA window
   fills). The bench baselines were 33.8-38.5 t/s across prompts. Does using a
   flat 26 ms bias the model? Does the verify cost also grow with context?
8. The tree "doubly penalized" claim and the ceiling table — re-derive the
   depth-3 tree speedup yourself. Is P(full accept of tree) = S(2)*q3 correct?
   Is verify_ms(4) the right cost for a 4-node tree (tree attention mask vs
   linear causal batch)? Could a tree ever win if verify were cheaper per node?
9. Throughput vs latency — the +20% gate is throughput. Does anything change if
   the real requirement is per-token latency (queueing, tail)?
10. Is there a factor ENTIRELY missing that dominates (e.g., memory-bandwidth
    contention between the drafter forward and the verifier in the same cycle;
    cache/thrashing; the cost of drafting itself scaling with K)?

OUTPUT (be specific and quantitative):
A. CONCRETE ERRORS — any equation/number that is wrong, with the corrected value.
B. LOAD-BEARING UNVERIFIED ASSUMPTIONS — ranked by how much each could move the
   K=4 speedup. For each, state the direction and approximate magnitude of the
   bias if the assumption is wrong.
C. VERDICT — which of {Q1, Q2, Q3, the "~2 pp from baseline" headline} survive
   your scrutiny and which are fragile or overturned. Be decisive.
D. THE ONE EXPERIMENT — single best falsifier of the headline claim
   ("~2 pp from beating baseline at K=4 with an optimized verifier"). Describe it
   concretely enough to run.
