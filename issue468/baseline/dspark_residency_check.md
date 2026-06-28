# Phase 3 — RAM Residency Check (recipe gate)

Date: 2026-06-28 (corrected 2026-06-28 after full 3-layer conversion). Node:
Apple M5 Max, 128 GB unified memory.

## Budget
- Target GGUF resident (DeepSeek-V4-Flash-IQ2XXS-...-imatrix.gguf): **80.76 GiB**
  (86,720,111,488 bytes on disk)
- macOS + ds4 process + Metal driver + buffers (conservative): **~6 GiB**
- KV + graph scratch @ 8192 ctx (ds4 reports "context buffers 495.92 MiB"
  target-only at 8192; add generous margin): **~1 GiB**
- **Fixed cost: ~87.8 GiB**, leaving **~40 GiB** for the drafter before the
  8 GiB-minimum-headroom line at 128 GiB.

## Drafter footprint by quant recipe
DSpark drafter HF source = 10.1 GB across 3 layers (shards 46-48). Routed
experts dominate (9.0 GB fp4 = 89% of the drafter); everything else <1.1 GB.

| recipe | drafter | total | headroom | verdict |
|---|---|---|---|---|
| **Q4_K experts (iter-1 recipe, MEASURED)** | **10.71 GiB** | **98.5 GiB** | **29.5 GiB** | **OK (chosen)** |
| Q2_K experts (fallback) | ~7.5 GiB | 95.3 GiB | 32.7 GiB | OK (not needed) |
| Q6_K experts | ~14.5 GiB | 102.3 GiB | 25.7 GiB | OK |

> **Correction (2026-06-28):** the iter-1 residency draft listed "6.18 GiB" for
> the Q4_K drafter. That figure was an estimate made before the full 3-layer
> conversion; the **measured** on-disk size of the converted `dspark.gguf` is
> **10.71 GiB** (11,501,064,576 bytes; 81 tensors: 9 packed Q4_K expert tensors
> × 1.2 GiB each = 10.1 GiB of experts + ~0.6 GiB Q8_0 attn/shared/main_proj +
> <0.1 GiB F32/BF16 small tensors). The Q4_K recipe still fits with 29.5 GiB
> headroom (3.7× the 8 GiB minimum), so the iter-1 decision is unchanged; only
> the documented size was wrong.

## Decision
**Use Q4_K routed experts** (iter-1 recipe) — fits with 29.5 GiB headroom
(3.7× the 8 GiB minimum), and Q4_K is the established quality choice in the
existing target/MTP GGUFs. Accuracy-critical small tensors (markov_head,
confidence_head, hc_fn, norms — <0.1 GiB total) kept BF16/F32 per the constraint.

Q2_K (~7.5 GiB drafter) is the documented fallback if a future change (larger
ctx, F16 debugging build, second model resident) overruns; not needed now.
