# Milestone-3 bench-batched divergence (2026-07-13)

Throughput (warm model, exactness_small_corpus, 10 prompts, frontier=48, gen=32):
- plain (argmax): 37.20 t/s
- seq (exact sequential verify): 16.29 t/s  (verified_mean=1.03, verify_n=2.00)
- batched (DS4_DSPARK_VERIFY_BATCHED=1): 20.19 t/s  (verified_mean=1.21, verify_n=2.11)
Attribution (1 cycle, code_topk, DS4_DSPARK_TIMING=1):
- seq:    decode 26.4  draft 28.8  verify 52.4  total 107.6 ms
- batched: decode 26.0  draft 21.2  verify 43.4  total  90.7 ms   (verify sublinear: 52 -> 43 ms)

Committed-output divergence (plain vs batched, flip-detecting token diff):
- ds4-eval benchmark (92 questions, greedy, --nothink, --tokens 200): 0% (BYTE-IDENTICAL)
- exactness_small_corpus (code/grounded/synthesis, 10 prompts, gen=48): 40% (7/10 prompts, 192/480 tokens)
  - code_topk: first divergence at token index 3 (plain[3]=3696 "docstring..." vs batched[3]=4085 "check for self-testing..."); 61/64 tokens differ. Both outputs are sensible text.
- seq (exact) dspark vs plain: 0% (byte-identical) -> the divergence is the batched flip, not a bug.

Conclusion: practical-exactness is CORPUS-DEPENDENT (0% benchmark / 40% code). Code generation has more near-tie argmax decisions -> more of the batched verify's 0.64%-level flips fire on verify decisions -> committed-output divergence.

## CORRECTION (2026-07-13, user challenge): ds4-eval does NOT engage DSpark

The earlier "0% divergence on the ds4-eval benchmark (92 Qs)" was INVALID: ds4-eval uses plain
decode (ds4_session_eval, ds4_eval.c:3872) — `--dspark` only loads the drafter, generation never
calls ds4_session_eval_speculative_argmax. So the 92Q "batched" run was plain-vs-plain (trivially
byte-identical), and DS4_DSPARK_VERIFY_BATCHED=1 had no effect. (This also means the milestone-2
"ds4-eval 10/10 byte-identical" was plain-vs-plain.)

With the batched verify ENGAGED (ds4-spec-bench speculative_argmax + DS4_DSPARK_VERIFY_BATCHED=1;
engagement proven by DS4_DSPARK_TIMING lines):
- BENCHMARK reasoning/math (12 extracted ds4-eval questions, frontier=32, gen=64): 12/12 prompts
  diverge, 435/768 tokens = 56.6%. (303 dspark-timing lines in the batched run.)
- exactness corpus (code/grounded/synthesis, 10 prompts): 40% (7/10).
- code_topk: 45/48 differ (first diff @ token 3).
- seq (exact) speculative_argmax vs plain: 0% (byte-identical) -> the divergence is the batched flip.

CONCLUSION (corrected): the committing batched verify is NOT practically exact on ANY tested
workload (reasoning 56.6% / mixed 40% / code 45-of-48) when engaged. Both plain + batched outputs
are sensible text (a real flip + greedy cascade, not garbage). The "practically exact" premise
(resting on the invalid ds4-eval plain-vs-plain tests) is falsified.
