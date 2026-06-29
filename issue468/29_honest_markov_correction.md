# Honest correction — Markov head cost was missing from draft timing

Date: 2026-06-29. Corrects issue468/28's draft cost (7.4ms) which excluded the
sequential Markov head (runs on CPU after the output head).

## What was missing

The draft forward timing (7.39ms median, measured via the accept probe) covers:
input stage + 3 DSparkBlocks + output head (hc_head + norm + lm_head).

It does NOT cover the **sequential Markov head** (5 positions, each conditioning
on the previous draft token):
- markov_w1[prev] embedding lookup (trivial)
- markov_w2 @ markov_w1[prev] = [vocab, 256] @ [256] = 129280 dot products

Measured (numpy BLAS sgemv): **0.85ms/position × 5 = 4.3ms** for the full head.

## Corrected speedup table

| ctx | plain | CPU-Markov draft (11.7ms) | GPU-Markov draft (7.5ms) |
|---|---|---|---|
| 8k | 31.3 | 1.23× (+23%) PASS | 1.30× (+30%) |
| 32k | 32.9 | 1.30× (+30%) PASS | 1.36× (+36%) |
| 64k | 35.5 | 1.40× (+40%) PASS | 1.47× (+47%) |

**Gate passes at all contexts even with the pessimistic CPU Markov cost.**

## Production note

The Markov head SHOULD be ported to GPU for production:
- markov_w2 [vocab, 256] BF16 = 66MB → memory-bound at 800GB/s ≈ 0.08ms
- 5 sequential positions ≈ 0.4ms total (vs 4.3ms CPU)
- Saves ~4ms/cycle → pushes 64k from +40% to +47%

The GPU port is a small matmul (ds4_gpu_matmul_f16_tensor on markov_w2) — trivial
to wire once B2 is integrated into the decode loop. Not needed for the gate to pass.
