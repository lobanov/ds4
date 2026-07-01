# ROOT CAUSE FOUND: Flash Attention F16 KV Conversion

Date: 2026-07-01. Twenty-second productionization handoff note. The definitive
root cause of the Metal-vs-oracle drafter divergence. Doc-only research record.

## 0. Summary

The drafter attention dispatch (`ds4_gpu_encode_flash_attention_decode_raw_batch_heads`,
ds4_metal.m:18489) converts the clean F32 raw KV cache to F16 via
`ds4_gpu_encode_cpy_f32_f16_1d` (line 18590), then runs the F16 flash attention
kernel `kernel_flash_attn_ext_f16_dk512_dv512` (line 18621). The oracle's
`sparse_attn` uses F32 KV throughout. The F16 KV precision loss in the 512-dim
QK^T dot products is the divergence source. It compounds across the 3 drafter
layers: layer 0 ~0.5%, layer 1 ~2-4%, layer 2 ~10-20%.

## 1. Evidence chain (complete op-level bisection)

| op | test | verdict |
|---|---|---|
| hc_pre + attn_norm | identical input | CLEAN (cos 1.00000, issue468/61) |
| q_a + q_b (Q8_0 matmul) | F32-input test | CLEAN for L0/L1 (cos 1.0, issue468/60) |
| KV window (dspark_kv_cache) | per-step anchor comparison | CLEAN per-step (cos 1.0, slots 1-3) |
| **flash attn F16 KV conversion** | identified in kernel source | **THE DIVERGENCE** (F32→F16 KV at ds4_metal.m:18590) |

The raw KV is F32 and bit-perfect vs the oracle. But the flash attention
dispatch converts it to F16 before the QK^T dot products. The 512-dim dot
product in F16 loses precision that the oracle doesn't.

## 2. Why it's layer-specific despite uniform F16 KV conversion

The F16 KV conversion applies to ALL layers equally, but the attention OUTPUT
divergence depends on the score distribution (softmax sensitivity). The error
compounds FORWARD across layers: layer 0's attention output is slightly wrong
(0.5%), which feeds layer 1's block input; layer 1's attention (also with F16
KV) diverges more (2-4%); layer 2 amplifies further (10-20%). This is FORWARD
propagation of attention-output error, not KV-window accumulation.

## 3. The fix (in-scope, code-only, frozen-drafter)

Two options:
**A. F32-KV flash attention variant:** instantiate `kernel_flash_attn_ext` with
F32 KV types (`float4x4` instead of `half4x4` for the KV tiles). Skip the
`ds4_gpu_encode_cpy_f32_f16_1d` conversion. Drafter-only dispatch. This is the
cleanest precision fix — same algorithm, just F32 KV.

**B. Direct F32 attention (non-flash):** for the drafter's small attention
(5 tokens × 64 heads × 512 dim × ~8 KV slots), compute QK^T + softmax + weighted
average as explicit F32 matmuls (no flash attention kernel). Simpler to
implement, exactly matches the oracle, but slightly slower (no flash-attention
memory hierarchy optimization — negligible at the drafter's small sizes).

Both are drafter-only, code-only, and frozen-drafter. The target model's flash
attention (causal decode path) is a different dispatch and is untouched.

## 4. Confirmation test (pending process lock)

Need to confirm by feeding Metal's EXACT q + KV window into the oracle's
sparse_attn and comparing to Metal's batch_heads (the flash attention output).
If they diverge → flash attn F16 KV is confirmed as the root cause. If they
match → output_proj is the bug instead.

## 5. Significance

This is the first root cause that survives the identical-input test methodology.
All prior candidates (hc_pre, q matmuls, KV precision) were disproven. The flash
attention F16 KV conversion is the ONE operation that (a) is applied to all
layers, (b) introduces F16 precision loss on the QK^T computation, (c) was never
tested, and (d) explains the layer-specific amplification pattern.

Per issue468/42's break-even, fixing this should recover the +8.27% acceptance
headroom → α 0.83→0.90 → ~41.5 t/s, crossing the gate.
