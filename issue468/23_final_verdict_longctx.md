# DSpark speculative decoding — FINAL VERDICT: UNACHIEVABLE (terminal outcome B)

Date: 2026-06-28. Supersedes the interim verdicts (issue468/18 short-ctx, 19 long-ctx
baseline, 20 draft-cost). This is the terminal long-context verdict with MEASURED
(not projected) Metal drafter acceptance.

## TL;DR

The >20% speedup gate **FAILS at every context, including 64k**, with decisive
measured evidence. The Metal drafter's real acceptance (measured on production
Q4_K kernels with persistent KV) is **~1.5-2.0 committed tokens/cycle**, far below
the **~2.82 needed at 64k** for the gate. Even the absolute best case (64k,
optimistic B2 = 2.2) is 0.95× (5% slower than plain decode). Terminal outcome B.

## The decisive measurement

Built a greedy-acceptance sweep probe (DS4_DSPARK_PROBE_ACCEPT) that runs the
complete Metal drafter forward (input stage + 3 DSparkBlocks + output head +
sequential Markov head) with PERSISTENT growing window KV across 19 decode steps
on the IQ2XXS target's captured main_hidden, comparing drafts to the target's true
greedy continuation. This is the production-representative measurement (real
dspark.gguf, real ds4 Metal Q4_K kernels, real target hidden states).

**Metal greedy acceptance: 1.53 avg prefix (33.7% match), 19 steps.**

vs the numpy oracle on the SAME data: **2.79 avg prefix (57.9% match)** (issue468/15).

The Metal drafter's real acceptance is roughly **half** the oracle's.

## Root cause: Q4_K precision × sequential Markov head error compounding

- ds4's routed-expert Metal kernel supports ONLY Q4_K / Q2_K / IQ2_XXS
  (`ds4_gpu_routed_mm_pipeline`; Q8_0 returns nil). Q4_K is the MAXIMUM precision
  available (issue468/22).
- The Metal fused Q4_K GEMM (BF16 accumulation) vs the oracle's F32-dequant-then-
  F32-matmul produces ~0.979/block FFN output correlation (error/signal corr=0.15,
  proven noise not a bug; attn sub-block validated at 0.99997).
- BUT the DSpark drafter uses a **sequential Markov head**: each draft token
  conditions the next. Autoregressive drafting compounds per-step errors
  MULTIPLICATIVELY in the prefix length. So 0.979/block corr → accepted prefix
  collapses from 2.79 (oracle) to 1.53 (Metal). Per-step breakdown confirms: the
  first draft token is usually correct, but a single borderline flip at position
  1+ sends the rest of the prefix off-target.
- This is INHERENT to the Q4_K precision + sequential-head architecture. No
  higher-precision option exists in ds4. The oracle's F32 path (2.79) is NOT
  achievable in production.

## Speedup with measured Metal acceptance

`committed` ≈ greedy_prefix + ~1 (B2 resample bonus). Realistic B2 ~1.8-2.2.
Gate at 64k needs committed > 2.82 (= 1.2 × 75 / (35.5 − ... ) break-even).

| ctx | plain ms | committed | ms/tok | speedup | gate (>20%) |
|---|---|---|---|---|---|
| 8k  | 31.3 | 1.53 (greedy)  | 53.7 | 0.58× | FAIL |
| 32k | 32.9 | 1.53           | 53.7 | 0.61× | FAIL |
| 64k | 35.5 | 1.53           | 53.7 | 0.66× | FAIL |
| 64k | 35.5 | 1.80 (B2 low)  | 45.7 | 0.78× | FAIL |
| 64k | 35.5 | 2.20 (B2 high) | 37.4 | 0.95× | FAIL |
| 64k | 35.5 | **2.82 (needed)** | 29.1 | 1.22× | (the bar) |

Cost basis (all measured): draft 7.2ms (5.0 backbone + 1.5 lm_head + 0.7 overhead),
verify 75ms (doc 06), plain decode 31.3→35.5ms/tok over 7k→55k (+11%, SWA).

## Which gate failed

**Acceptance (draft quality).** The Metal drafter, constrained to ds4's Q4_K
routed-expert kernels, cannot produce drafts accurate enough for the sequential
Markov head. Acceptance (~1.5-2.0) is far below the ~2.82 needed at 64k. This is
NOT a draft-cost or verify-cost problem (those were favorable at long context per
issue468/20); it is a draft-QUALITY problem caused by quantization precision.

## Verifier-optimization headroom (for a potential second pass)

The bottleneck is NOT verifier cost (75ms is fine at long context). It is draft
quality from Q4_K. Two knobs that WOULD change the economics (out of scope this pass):
1. **Higher-precision routed experts** (Q8_0/F16): ds4's Metal kernel doesn't
   support them for routed experts. Adding Q8_0 routed-expert support would let
   the drafter match the oracle's 2.79 → gate would PASS at 64k (+22%).
2. **A non-sequential / parallel draft head**: the sequential Markov head is what
   amplifies the Q4_K errors. A parallel head (each position independent) would
   not compound errors → acceptance would track the per-position 0.85 rate
   (issue468/18) → ~3.0+ committed. But this is an architecture change, out of scope.

## What was achieved (reusable artifacts)

Despite the negative verdict, substantial validated infrastructure was built:
- **Phase 3**: dspark.gguf (10.71 GiB, 81 tensors, 47/47 byte-exact crosscheck).
- **Phase 4**: complete Metal drafter forward (input stage corr 1.0, attn sub-block
  corr 0.99997, output head + Markov head) running end-to-end on Metal. Non-causal
  batched attention primitive added. Numpy oracle (validated algorithm reference).
  Multiple converter/kernel bugs found+fixed (F16 hc_fn, F32 norms, inverse-rope,
  dspark Metal map registration, swiglu_limit clamp).
- **Measurement harness**: backbone-timing probe, input/attn/block/token probes,
  and the persistent-KV greedy-acceptance sweep — all env-gated, reusable.
- **Cost curves**: decode 31→35ms/tok, verify 75ms (context-flat), draft 7.2ms —
  all measured.

## Final verdict

**Terminal OUTCOME B — UNACHIEVABLE.** The >20% speedup gate fails at every
context with measured evidence: the Metal drafter's real acceptance (~1.5-2.0
committed/cycle) is far below the ~2.82 needed at 64k, because ds4's Q4_K
routed-expert precision (the maximum available) is insufficient for the DSpark
drafter's error-amplifying sequential Markov head. No in-scope knob closes the
gap; the two that would (Q8_0 routed-expert kernel support; non-sequential draft
head) are out of scope. The conversion, loader, validated Metal drafter forward,
and measurement harness stand as reusable artifacts.
