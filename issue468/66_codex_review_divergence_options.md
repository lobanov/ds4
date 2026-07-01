# Codex Review (gpt-5.5 xhigh) — Divergence Options + Code Review

Date: 2026-07-01. Twenty-sixth productionization handoff note. Records the
independent codex review of the MoE divergence root cause (issue468/63-65) and
the verified follow-up analysis. Doc-only research record.

## 0. Review setup

Ran `codex exec -m gpt-5.5 -c model_reasoning_effort=xhigh -s workspace-write`
with a prompt containing the full op-level bisection data (issue468/63-65),
the specific Metal kernel source locations, and two questions: (1) is there a
code-level fix before re-quanting? (2) what are the realistic recovery options?

Full codex output: /tmp/codex_divergence_final.md (32 lines).

## 1. Codex code review findings (6 items)

### Finding #1 (path mismatch) — VERIFIED, important for understanding

The drafter MoE (n_expert=256, n_tokens=5) does NOT take the fused group
pipeline (`kernel_mul_mv_group_q4_K_pair_swiglu_f32`). Codex identified the
dispatch conditions:
- `direct_down_sum` requires `n_tokens == 1` (ds4_metal.m:22695) → FALSE (drafter has 5)
- `q4_batch_expert_table` requires `n_total_expert == 384` (ds4_metal.m:25028) → FALSE (drafter has 256)

So the drafter uses the **ID-matvec path**: separate `kernel_mul_mv_id_q4_K_f32`
matvecs for gate/up/down + separate SwiGLU + F32 six-expert sum. This matters:
the divergence is in the ID-matvec kernel's accumulation order, not the fused
group pipeline.

### Finding #2 (Q4_K accumulation order) — CONFIRMED as the likely source

Codex: "the kernel accumulates packed partial sums, applies d/dmin after grouped
reductions, then uses simd_sum. This is algebraically equivalent, but not
roundoff-equivalent to 'fully dequantize Q4_K to F32 then dot in F32'. Layer 2
can be uniquely sensitive even if layer 0 is clean."

This aligns with the bisection finding (L0 cos 1.0, L2 cos 0.996-0.998 on
identical input). The SIMD reduction order is the mechanism. Testable with a
slow exact-Q4 Metal kernel that dequants to F32 before accumulating.

### Finding #3 (il=0 cache aliasing) — REFUTED upon verification

Codex flagged that DSpark passes `il=0` for all 3 MTP blocks (ds4.c:18006),
which is forwarded as `layer_index` to the routed MoE. Codex warned about
streaming expert cache layer-index-keyed aliasing.

**Verified REFUTED**: the Q4 expert table cache is keyed by `model_map +
tensor_offset` (ds4_metal.m:11973), NOT by layer index. mtp.0/1/2 expert
weights have different GGUF tensor offsets (6GB / 3.8GB / 7.6GB ranges), so
their cache entries cannot collide. The `il=0` is correct for the drafter
(hash routing is off for compress_ratio=0). No cache aliasing.

### Finding #4 (half d precision) — RULED OUT

Codex: "Q4_K stores d/dmin as FP16 by format. Normal FP16 values promote exactly
to FP32. Subnormal flush is possible but should not be layer-2-specific."
Verified: the F16→F32 promotion is exact for all normal values, and layer 0's
cos 1.0 proves no dequant-level precision issue.

### Finding #5 (SwiGLU fusion) — RULED OUT

Codex: "In the default batch path [the drafter's ID-matvec path] SwiGLU is
already a separate F32 kernel. Gate/up are F32-materialized before activation."
No fusion-level precision loss in the drafter's dispatch path.

### Finding #6 (cross-token SIMD) — RULED OUT

Codex: "Batch pair work indexes token via iid1, down/sum uses tgpig.y, the F32
sum6 kernel adds slots 0..5 in order. DSpark n=5 does not use F16 mid;
use_mid_f16 requires use_mm_id which requires n_tokens >= 32." No cross-token
numerical interactions.

## 2. Options analysis (codex, ranked by recovery potential)

### Option 1: Exact-Q4 layer-2 diagnostic kernel (code-only, no re-quant)

The only code-only path that could plausibly recover most of the +8.27%
headroom without changing the GGUF. A slow Metal kernel that dequantizes each
Q4_K element to F32 individually, then dots in a deterministic F32 pairwise
sum (matching numpy's approach exactly).

**Tradeoff**: too slow for production (full dequant is 4× the compute), but
diagnostic-only: confirms whether Q4_K accumulation order IS the fixable issue.
If the layer-2 MoE cosine jumps from 0.996 to ~1.0 with exact-Q4, the
accumulation order is confirmed, and the question becomes whether a
production-speed variant is feasible.

### Option 2: Layer-2-only Q8_0 re-quant (+3.22GB, ID matvec exists)

Re-quantize ONLY mtp.2's routed experts (ffn_gate_exps, ffn_up_exps,
ffn_down_exps) from Q4_K to Q8_0. Q8_0's block scale is F32 (not F16 like
Q4_K's half d), eliminating the main accumulation-order sensitivity. The
Q8_0 ID matvec kernel (`kernel_mul_mv_id_q8_0_f32`) already exists and
dispatches through the same routed MoE path.

**Size**: mtp.2 routed layer ~6.44B weights. Q4_K ~3.62GB → Q8_0 ~6.85GB
(+3.22GB). Total drafter: 11.5GB → ~14.7GB. Feasible on M5 Max.
**Code**: needs tensor-type acceptance change (ds4.c:3278 validates routed
experts only for IQ2_XXS/Q2_K/Q4_K; Q8_0 would need adding) + the quantizer
to emit Q8_0 for mtp.2 experts specifically. ~1 day of code work.

### Option 3: Layer-2-only Q6_K re-quant (+1.66GB, needs new kernel)

Smaller size increase (+1.66GB) but Q6_K routed expert support doesn't exist
in the Metal dispatch (ds4_metal.m:20140 maps only IQ2_XXS/Q2_K/Q4_K for routed
experts). Would need a new Q6_K routed expert kernel + validation changes.
~2-3 days of code work.

### Option 4: Full-drafter re-quant — PREMATURE

Codex: "Layers 0/1 are already clean in the bisection, so paying 3× the
layer-2 size/perf cost is not justified until layer-2-only is measured."
Layers 0/1 at cos 1.0/0.9999 prove Q4_K is sufficient for their weight
distributions. Full re-quant wastes ~7GB for zero gain on layers 0/1.

### Option 5: Tensor-selective layer-2 re-quant

Re-quant only gate+up (not down) or only down — each is ~1/3 of the routed
layer. The existing dump hooks (ds4.c:19942) can identify which substage
dominates the divergence. Could halve the size cost if only one substage is
sensitive. Untested but promising.

## 3. My verified follow-up analysis

### The ID-matvec path matters

Codex's Finding #1 is critical: the drafter uses the **ID-matvec** path
(`kernel_mul_mv_id_q4_K_f32`), NOT the fused group pipeline. This means:
- The divergence is in `kernel_mul_mv_q4_K_f32_impl` (metal/moe.metal:476),
  the Q4_K matvec core, as called by the ID-matvec wrapper
- The grouped pair SwiGLU fusion (which I was examining) is NOT on the
  drafter's path
- Any fix must target the ID-matvec Q4_K accumulation

### Why layer 2 is uniquely sensitive

The Q4_K kernel's accumulation (sumf += d * grouped_acc - dmin * sumy) is
algebraically correct but numerically non-identical to numpy's dequant→dot.
Layer 0's weights produce well-conditioned partial sums (roundoff ≈ 1e-7,
cos 1.0). Layer 2's weights have extreme scale ratios (large d × small scale,
or large partial sums with cancellation), amplifying roundoff to 2-3.5%.
This is a weight-distribution property, not a kernel bug — hence not fixable
by changing the kernel code alone (the kernel IS correct for layer 0).

### The il=0 non-issue

I verified the Q4 expert table cache key includes the tensor_offset
(ds4_metal.m:11973: `NSString *key = [NSString stringWithFormat:@"%p:%llu:%llu:
%llu:%u:%llu:%u", model_map, model_size, tensor_offset, ...]`). Each MTP stage's
expert tensors are at different offsets (mtp.0 ~1.5GB, mtp.1 ~3.8GB,
mtp.2 ~7.6GB in the GGUF). The cache entries are disjoint. il=0 is correct.

## 4. Recommended next steps (ranked)

1. **Exact-Q4 diagnostic** (code-only, no re-quant, ~hours): add a DS4_DSPARK_EXACT_Q4
   env gate that routes the drafter's Q4_K routed expert matmuls through a slow
   F32 dequant path (dequant each element to F32, pairwise F32 sum). If the
   deterministic probe's layer-2 MoE cosine jumps to ~1.0, accumulation order is
   confirmed. This is the gate-crossing proof-of-concept.

2. **Layer-2-only Q8_0 re-quant** (requires un-freezing drafter, ~1 day code):
   if exact-Q4 confirms, re-quant mtp.2's routed experts to Q8_0. The ID matvec
   kernel exists; only validation + quantizer changes needed. Q8_0's F32 block
   scale eliminates the F16 d sensitivity. Expected: layer-2 cosine → ~1.0,
   recovering most of the +8.27% headroom.

3. **Tensor-selective re-quant** (if full layer-2 Q8_0 is too large): re-quant
   only the substage (gate/up/down) that dominates the divergence, identified
   via the existing FFN debug dump hooks.

4. **Full-drafter re-quant**: only if layer-2-only proves insufficient
   (unlikely, given layers 0/1 are already clean).

## 5. Significance

The codex review confirms the bisection's conclusion (MoE accumulation order
for layer 2) while identifying a concrete code-only diagnostic path (exact-Q4)
that could prove the hypothesis without re-quanting. The ID-matvec path
discovery (vs the fused group pipeline I was examining) narrows the fix to a
specific kernel. And the layer-2-only Q8_0 re-quant — with existing ID matvec
support — is a targeted, justified production fix if the diagnostic confirms.

The path to the gate is: exact-Q4 diagnostic → Q8_0 layer-2 re-quant → re-measure.
Both are in-scope if the drafter freeze is partially lifted for layer-2-only
mixed precision.
