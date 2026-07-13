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
