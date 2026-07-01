# Drafter Drift — Definitive Root Cause: F16-Input Q8_0 MMA

Date: 2026-07-01. Sixteenth productionization handoff note. Identifies the
definitive root cause of the mtp.2 attention divergence (issue468/53-56).
Doc-only research record.

## 0. The exact mechanism

The drafter's q_b matmul (in_dim=1024, out_dim=32768, n_tok=5) takes the Q8_0
batch fallback kernel `kernel_mul_mm_q8_0_f32` (ds4_metal.m:12878, because
n_tok=5 < the NAX threshold of 32). That kernel (metal/dense.metal:1600) is:

```
kernel_mul_mm<half, half4x4, simdgroup_half8x8,   // weight tile A: half
              half, half2x4, simdgroup_half8x8,   // activation tile B: half
              block_q8_0, 2, dequantize_q8_0,      // q8_0 dequant
              float, float4x4, float, float2x4>    // accumulator/output: float
```

The Q8_0 weights are dequantized (dequantize_q8_0, dense.metal:766) INTO
`half4x4` tiles, stored in `half` threadgroup memory, and the matrix-multiply
uses `simdgroup_half8x8` for BOTH input tiles (A and B), accumulating into
`simdgroup_float8x8`. So the MMA inputs are F16, even though accumulation is
F32. The "_f32" in the kernel name refers to output/accumulation, NOT input
tile precision.

## 1. Why this diverges for mtp.2 only

The F16 input tiles round each dequantized weight (int8 × F16 block-scale d)
to F16 before the MMA. For most weight distributions (mtp.0, mtp.1 q_b) this
F16 rounding is negligible (cosine 1.0 — verified, issue468/55). For mtp.2's
specific int8 value patterns, the F16 rounding diverges from numpy's pure-F32
dequant+matmul reference, compounding across the 32768-element reduction to a
4-6% query divergence → 9-11% MHSA output divergence → 10-25% block output
→ 81% argmax base_logits → the +8.27% acceptance gap (issue468/51).

This is NOT a bug in the F16 kernel (it's a deliberate speed/accuracy tradeoff
standard in llama.cpp-style inference) — it's that the drafter, being a small
speculative drafter conditioned on the target's hidden state, is unusually
sensitive to q precision at mtp.2 (the last stage feeding the head).

## 2. The fix (in-scope, code-only, frozen-drafter — but deep)

An F32-input Q8_0 MMA variant: `kernel_mul_mm<float, float4x4, simdgroup_float8x8,
float, float2x4, simdgroup_float8x8, block_q8_0, 2, dequantize_q8_0, float,
float4x4, float, float2x4>` — dequant to F32, F32-input MMA. Tradeoffs:
- **Threadgroup memory doubles** (float4x4 vs half4x4 tiles) — may exceed the
  6144/8192-byte threadgroup allocation, requiring tiling changes.
- **F32 MMA is ~2× slower** than F16 MMA on Apple Silicon (half the throughput).
- **Shared kernel**: this same `kernel_mul_mm` is the target model's prefill
  matmul. A global change risks target-model regression (the baseline 39 t/s).
  A drafter-only dispatch (separate pipeline + a code path that selects it for
  the drafter) is the safe approach but requires new kernel plumbing.

This is the "deep Metal kernel work, high regression surface" flagged in
issue468/56 §3. It is in-scope (code-only, drafter weights byte-identical) but
NOT a quick win — it's a kernel-precision change with real perf cost and a
broad regression surface (shared matmul).

## 3. Decision

The bisection has achieved its complete purpose: PROVEN the +8.27% headroom is
real + localized the divergence to F16-input Q8_0 MMA on mtp.2's weights +
identified the exact template parameter that would change. Implementing the
F32-input variant is a multi-hour Metal kernel change with regression risk to
the shared target-model matmul — a scope/risk investment that warrants user
direction (the A/B fork from the prior checkpoint).

The safe, in-scope recovery (FP8-off, +2.6%/+3.9%) is already landed.
Pivoting to the deterministic finalize work (gguf-tooling, quality-eval,
finalize-selfcontained) is the responsible use of this checkpoint — those
deliverables are required regardless of whether the deep kernel fix is pursued.
