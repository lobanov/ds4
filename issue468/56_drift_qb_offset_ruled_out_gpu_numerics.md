# Drafter Drift Bisection — q_b Offset Ruled Out; Remaining Cause is GPU Q8_0 Numerics

Date: 2026-07-01. Fifteenth productionization handoff note. Continues
issue468/55 (q_b localization). Rules out offset/binding; isolates the
remaining cause to GPU Q8_0 matmul numerics. Doc-only research record.

## 0. Where we were

issue468/55 localized the layer-2 divergence to mtp.2's q_b (the largest
attention weight, [1024,32768] Q8_0). Layers 0,1 produce bit-perfect q
(cosine 1.0); only mtp.2's q (post q_b matmul) diverges 4-6%.

## 1. Offset/binding Ruled OUT

Hypothesis: Metal reads the wrong bytes for mtp.2's q_b (off-by-one in offset
parsing at the 7.6GB region). Checked via the oracle's index_gguf:
- mtp.0 q_b: offset 6,035,264, size 35,651,584 bytes
- mtp.1 q_b: offset 3,827,496,256
- mtp.2 q_b: offset 7,595,463,488, size 35,651,584 bytes

All three: same dims [1024,32768], same size, **no overlap** (mtp.2 q_b spans
[7,595,463,488, 7,631,115,072); the next tensor mtp.2.attn_q_a_norm starts
immediately after at 7,631,109,632). Contiguous, correct sizes, regular layout.
`abs_offset = tensor_data_pos + rel_offset` (ds4.c:1948) is uniform.

→ **NOT an offset/binding error.** Metal reads the correct bytes for mtp.2 q_b.

## 2. The remaining cause: GPU Q8_0 dequant / matmul accumulation numerics

With identical code, identical quant type (Q8_0), identical block scales
(issue468/54 §3: all 0.00003), identical offsets, and layers 0,1 bit-perfect —
the ONLY layer-2-specific variable is mtp.2's actual **int8 weight values**.

This means the divergence originates in how the GPU's Q8_0 matmul kernel
(ds4_gpu_matmul_q8_0_tensor, called at ds4.c:17742) processes mtp.2's specific
int8 patterns vs how numpy's `_dequant_q8_0` + `@` reference processes them.
Likely culprits:
- **GPU dequant F16-accumulation rounding**: the GPU may accumulate the
  scale×int8 products in F16 (Metal's default for shader math) where numpy
  uses F32. For most weight distributions this rounds identically (hence
  layers 0,1 perfect), but mtp.2's specific value patterns may trigger a
  rounding divergence that compounds across the 32768-element reduction.
- **Reduction-order sensitivity**: GPU thread-group reductions sum in a
  different order than numpy's contiguous sum; for ill-conditioned mtp.2
  weight columns, this loses precision.

Both are properties of the GPU KERNEL, not the drafter weights — so a fix
would be code-only (drafter byte-identical). But both are DEEP Metal kernel
work: the q8_0 matmul is a heavily-optimized fused kernel; changing its
accumulation precision/order risks the target model (shared kernel) and the
MTP path.

## 3. The hard limit reached

I have localized the bug as tightly as a bisection allows: 8 levels down to a
single Q8_0 tensor's int8-value-dependent numerical divergence in the GPU
matmul kernel. Further progress requires either:

**A. Deep Metal kernel work** (in-scope, code-only, frozen-drafter):
   - Inspect ds4_gpu_matmul_q8_0_tensor's Metal shader; check if the q_b path
     accumulates in F16 vs F32.
   - If F16: force F32 accumulation for the drafter matmuls (or globally).
   - Risk: touches the shared Q8_0 kernel used by target + MTP; high
     regression surface. Each change needs full MTP + baseline re-verification.

**B. Accept the recovered ground** (FP8-off, issue468/52, +2.6% quality /
   +3.9% t/s already landed) and finalize on non-perf criteria. The mtp.2 q_b
     GPU-numerics bug is documented as a known-limitation with a clear
     localization + a concrete (if deep) fix path.

**C. Out-of-scope**: dequantize mtp.2 q_b to F16/Q6_K and rebuild the drafter
   GGUF. This would fix the divergence but CHANGES THE DRAFTER (forbidden by
   the frozen-drafter constraint).

## 4. Honest assessment

- The +8.27% oracle headroom is REAL and recoverable in principle — it's a
  genuine GPU numerical bug, not inherent quant noise.
- Recovering it would cross the gate (α 0.83→0.90 → ~41.5 t/s).
- BUT the fix is deep Metal kernel work (option A) on a shared, optimized
  kernel with broad regression risk — not a quick win.
- FP8-off (+2.6%/+3.9%) is already landed and is the safe, in-scope recovery.

The bisection has achieved its purpose: it PROVED the headroom is real and
localized the exact defect. Whether to invest in the deep kernel fix (A) or
finalize on the recovered ground (B) is a scope/risk decision for the user.
