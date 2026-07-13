# Lead 08 — exactness-gap characterization (making the batch verify bit-exact)

Date: 2026-07-13. Task: `exactness-gap`.

> **⚠ GATE A CORRECTION (2026-07-13):** the central claim below — "the dominant
divergence is F16 (batch) vs F32/Q8_0 (decode), closable by kernel swap" — is **WRONG**.
`metal_graph_matmul_plain_tensor` (ds4.c:16615) dispatches on `w->type`, and `hc_attn_fn`
is `DS4_TENSOR_F16` (ds4.c:3655), so **decode ALSO uses `ds4_gpu_matmul_f16_tensor`** for
the HC projection; the attention compressor is F16 on **both** paths (ds4.c:15451). The
divergence is therefore **reduction/path/order, NOT dtype**, and "closable by F32 swap" is
unfounded. The real divergence source is unidentified; the gate/up "already bit-exact"
claim is also **unverified** (the measured 6.2–7.6 gate/up divergences are REAL — codex
recomputed the dumps — not a layout artifact). Treat the per-stage table below as a
kernel-difference MAP only, not a validated fix-scope. The decisive check is an
identical-input MoE kernel-equality test (see `floor_clearance_verdict.md`).

Inputs: the diagnose-divergence map (`stage_divergence_map.json`, layer 40 / pos 61) +
read-only kernel analysis. Premise: gate/up are already bit-exact (both paths use
`_impl`); the gaps are the OTHER stages. "Making the batch verifier bit-exact" = make
`verify_suffix_tops`'s reductions match the decode (`encode_decode_layer`) reductions.

## Per-stage fix scope

| Stage | Decode kernel | Batch kernel | Divergence (max_abs) | Bit-exact fix | Effort | Trade-off |
|---|---|---|---:|---|---|---|
| hc_attn / hc_ffn projection | `matmul_plain_tensor` | `matmul_f16_tensor` | 0.078–0.086 | use plain/F32 matmul (decode's) | LOW | F32 act ≈ same bw (acts small) |
| attention compression (comp_kv/comp_sc → KVcur) | `matmul_q8_0` | `matmul_f16_tensor` | KVcur 0.125 (largest) | use Q8_0/F32 (decode's) | LOW–MED | F32 ≈ same bw; biggest single win |
| attention kernel (flash, post-compression) | `attention_decode_heads` (per-token) | `encode_layer_attention_batch` (batched) | attn_out 0.064 (mostly upstream F16?) | bit-exact batched flash OR per-token | MED–HARD | per-token loses attention sharing |
| down + sum6 | `id_q2_k` down + `id_q2_k_sum6` | `addr_q2_k_sum6` (+ addr down) | ffn_moe_out 0.018 | use `id_q2_k` down+sum6 (decode's) | MED | verify Q2_K id vs addr reduction match |

## Key findings

1. **The bulk of the divergence is F16 vs F32/Q8_0, not exotic kernel reductions.** The
   batch path uses `ds4_gpu_matmul_f16_tensor` for the hc projections AND the attention
   compression (`batch_comp_kv`, `batch_comp_sc`); decode uses plain/Q8_0. The F16
   activation precision loss is the dominant source (hc 0.078–0.086 + KVcur 0.125 — the
   largest single divergence). These are **closable by kernel swap** (use the decode
   plain/Q8_0 matmul), low–medium effort. The activation is small relative to weights, so
   the bandwidth cost of F32 activation is likely minor (TBD by measurement).

2. **The attention kernel is the uncertain term.** The batched flash attention
   (`encode_layer_attention_batch`) differs structurally from per-token decode attention.
   BUT the measured `attn_out` divergence (0.064) likely inherits most of the upstream
   `KVcur` (0.125) F16-compression error — so the *residual* batched-attention divergence
   after F32 compression is **unmeasured** and could be small. Two paths: (a) if the
   batched flash reduction matches per-token once the compression is F32 → near-bit-exact
   for free; (b) if not → use per-token attention (exact, like `decode2_exact`), which
   **loses the attention sharing** (the attention goes linear, 2× decode). The
   feasibility hinges on measuring the residual — that is a cheap follow-on measurement,
   not weeks of kernel work.

3. **down/sum6 is a minor, closable gap.** `id_q2_k` vs `addr_q2_k` for Q2_K. If the Q2_K
   id and addr kernels share their reduction (as the IQ2_XXS gate/up share `_impl`), this
   is a kernel swap (low effort); ffn_moe_out divergence is only 0.018.

## Feasibility verdict (per stage)

- **hc projections + attention compression (F16→F32/Q8_0):** FEASIBLE, low–medium effort,
  high impact (these are the dominant divergences). Main open question is the F32 speed
  cost (likely small — activations are small vs weights).
- **attention kernel:** FEASIBLE-IF the residual batched-flash divergence (after F32
  compression) is small (a cheap measurement decides this); otherwise per-attention
  (exact but loses sharing). This is the one genuine uncertainty.
- **down/sum6:** FEASIBLE, medium effort, low impact.

## What this means for the full bit-exact sublinear verifier

A fully-bit-exact sublinear verifier is **plausibly achievable without novel IQ2_XXS
kernels**: the dominant gaps are F16→F32/Q8_0 matmul swaps (the decode path already has
these kernels), not new fused kernels. The one real risk is the batched-attention residual
(unmeasured). The gate/up — the biggest weight load and the original "fusion" target — are
already bit-exact. So the exactness work is mostly **kernel selection** (use the decode
reductions in the batch path), not novel-kernel authorship — a substantially lower-effort
path than the Phase-B single-stage-kernel plan assumed.

## Open measurement (cheap, decides the attention question)

Re-run the diagnose-divergence dump-tag harness with the batch path's compression forced
to F32 (or compare the batched-attention output with F32 compression vs decode) to measure
the residual `attn_out` divergence. If small → attention is feasible for free; if large →
attention needs per-token (loses sharing).
