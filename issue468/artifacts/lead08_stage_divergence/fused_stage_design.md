# Lead 08 Phase B — single-stage fused routed-expert M=2 kernel: design + fidelity-gate spec

Date: 2026-07-13. Task: `design-fused-stage`. Chosen stage: **routed IQ2XXS experts**
(see `stage_divergence_map.json`). Recorded BEFORE implementation.

## Goal of the prototype

`decode2_exact` is exact but LINEAR: it calls `ds4_gpu_routed_moe_one_tensor` twice
(once per token), reloading + redequanting the selected IQ2XXS expert weights each time.
The prototype replaces those two calls with ONE bit-exact M=2 dispatch that **shares the
expert weight dequant across both tokens** while keeping each token's reduction identical
to M=1. Result: exact (same numerics as decode) + cheaper (expert weights loaded once for
the union). The rest of the layer (attention, hc projections, router, shared expert, down)
stays per-token (`decode2_exact`-style) in this milestone — full-path fusion is follow-up.

## Bit-exactness mechanism (why M=2 == M=1 by construction)

Reference kernel: `kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32` (`metal/moe.metal:1022`),
N_R0_IQ2_XXS = 4 rows per expert. Per weight block (ib32) it:
1. loads the 32-float activation tile `yl[32]` (from the ONE token's `src1`);
2. dequants the gate/up IQ2XXS block (grid lookup `svalues`, signs `ssigns`, scales
   `dg`/`du`) — **weight-only, independent of the token**;
3. MACs `sg += v * grid[j] * sign`, `su += v * grid[j] * sign` per row, accumulates
   `sumg[row] += dg*sg`, `sumu[row] += du*su`;
4. reduces per row with `simd_sum(...)` then `* 0.25f`.

M=2 variant: extend the inner loop to TWO activation tiles (`yl_a[32]`, `yl_b[32]`) from
two tokens' `src1`, with TWO accumulator sets (`sumg_a/sumu_a`, `sumg_b/sumu_b`). The
dequant (step 2) is computed ONCE and reused for both tokens; steps 3–4 run per token with
the IDENTICAL operation order. Because each token's accumulator sees the same FP operations
in the same order as M=1, each token's gate/up output is **bit-identical** to a standalone
M=1 call. This is exactness-by-construction (strategy A, locked in `lock-exactness`).

## What must be fused vs kept per-token

- **Fused (shared dequant), M=2:** the gate+up paired IQ2XXS matmul
  (`kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32`) — the dominant expert weight load.
- **Down projection (expert_mid → out_dim):** also a weight load. Options — (a) also fused
  M=2 with the same shared-dequant + per-token-reduction pattern (full cost saving), or
  (b) kept per-token (exact, no sharing) for the first prototype to de-risk, share later.
  **Decision for the prototype: option (a) if the down kernel is the same family and
  straightforward; otherwise (b) and note the un-fused cost.** Revisit at build time.
- **Per-token (unchanged, like decode2_exact):** attention (MLA), hc projections, router,
  shared expert, norms — all already exact via the decode kernels.

## Union-expert handling (mirrors `routed_moe_batch`)

The two tokens may select different experts. The host computes the **union** of the two
tokens' top-6 selections (≤12 experts), dispatches the M=2 kernel over the union, and
routes each expert's per-token output to the token(s) that selected it (× that token's
router weight), then sums. This is the same structure `routed_moe_batch_tensor`
(`ds4_metal.m:24622`) already uses; the difference is the M=2 kernel uses the M=1
reduction (bit-exact) instead of the batched reduction.

## Host wiring

New `ds4_gpu_routed_moe_pair_tensor(...)` (or an `n_tokens` parameter on
`routed_moe_one_tensor`) taking 2 token activations + 2 selection/weight sets, dispatching
the M=2 kernel over the union. Plug-in site: a new env-gated branch
(`DS4_DSPARK_FUSED_ROUTED_M2=1`) inside `metal_graph_verify_decode2_exact`
(`ds4.c:21808`) that, for the routed-expert stage, calls the M=2 dispatch once instead of
`routed_moe_one` twice. Everything else in `decode2_exact` stays per-token.

## Fidelity-gate spec (MUST pass before any powered number)

**Question:** is the fused M=2 routed-expert output bit-identical to the M=1 decode output,
per token?

**Method (reuses the diagnose-divergence harness):**
1. Run `code_topk` with `DS4_DSPARK_VERIFY_DIST_PROBE=1` + the new `DS4_DSPARK_FUSED_ROUTED_M2=1`,
   dump-tagged (`b_` = the fused-M2 verify path; compare against the existing sequential
   decode reference `s_`), at a clean-input position (e.g. pos 61) on layer 40 (and a
   second layer, e.g. 20, for robustness).
2. Compare `dump_b_ffn_moe_out` / `dump_b_routed_out` (M=2 fused) vs `dump_s_ffn_moe_out`
   (M=1 decode) for the SAME token row.
3. **Pass criterion: `max_abs == 0.0` (bit-for-bit) on the held-out baseline.** Also
   re-run the full exactness corpus (greedy, temp=0) and require 0 argmax flips on the
   fused path vs plain decode (the existing `mtp_exactness_compare` harness).
4. If `max_abs != 0`: do NOT trust any timing number. Dispatch an adversarial-codex-review
   bug-hunt on the kernel reduction; fix; re-gate. (Hard abort per the goal if it cannot be
   made bit-exact after the bug-hunt.)

**Gate artifact:** the dump comparison + the exactness-corpus 0-flip result, paths recorded
in the worklog.

## Open build-time questions (to resolve at impl)

1. Per-expert tensor layout for `ffn_moe_gate_clamped`/`up_clamped`/`down` — confirm whether
   the large point divergences (6.7–331) seen in the diagnose map are real IQ2XXS
   amplification or a [expert,token] vs [token,expert] layout artifact. The fidelity gate
   resolves this empirically (if M=2 == M=1 bit-for-bit, the layout/numerics are correct).
2. Whether the down projection kernel family supports the same shared-dequant M=2 pattern
   cleanly (decides option a vs b above).
3. Whether the M=2 kernel should be a new `kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32_m2`
   or a runtime `n_tokens` parameter on the existing kernel (prefer the parameterized
   single-family form to maximize M=1/M=2 reduction identity).

## 2026-07-13 spec-read refinement (post design, pre impl)

Reading the actual kernels changed the picture materially:

- **The gate/up IQ2XXS paired reduction is ALREADY SHARED.** Both the decode `id` kernel
  (`kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32`, moe.metal:1022, inline loop) and the batch
  `addr` kernel (`kernel_mul_mv_addr_iq2_xxs_pair_swiglu_f32`, moe.metal:1257) use the
  SAME reduction — the `addr` kernel calls `kernel_mul_mv_iq2_xxs_pair_f32_impl`
  (moe.metal:680), which is bit-identical to the `id` kernel's inline loop (same dequant,
  same MAC order, same `simd_sum`+`*0.25`). So **gate/up are bit-identical batch-vs-decode**.
- Therefore the large point divergences measured for `ffn_moe_gate_clamped`/`up_clamped`/
  `down` (6.7–331) in the diagnose map were almost certainly a **per-expert tensor-layout
  artifact** (row-0 extraction mismatched an expert-major layout), NOT real. The reliable
  per-token outputs (ffn_moe_out 0.018, ffn_moe_weighted_swiglu 0.030, ffn_moe_logits 0.023)
  are the trustworthy numbers and are all SMALL.
- f16-mid is **ruled out**: `request_mid_f16 = ... && !use_iq2_batch_selected_addr`, so the
  IQ2_XXS batch path uses f32 mid (same as decode).
- The real routed-expert OUTPUT divergence (ffn_moe_out 0.018) lives in the **down+sum6**
  stage (id down+sum6 kernels vs addr/batch down+sum6 kernels) plus the tiny route-weight
  difference (ffn_moe_weights_scaled 0.0006) — NOT the gate/up.
- The expert weight LOAD is NOT shared across the 2 tokens in either path (each loads per
  (token,expert)), so the cost saving is still unrealized.

**Revised implementation focus:** the M=2 fused work is (a) **share the expert weight LOADS**
(gate/up + down) across the 2 tokens — the actual cost saving — via a kernel that loads
each union expert once and applies to both tokens; and (b) make the **down+sum6**
bit-exact with the decode (M=1) down+sum6 (the gate/up already are, via `_impl`). The
fidelity gate (M=2 ffn_moe_out == M=1, max_abs==0) validates both. The prototype's primary
measurable contribution is the verify_ms cost saving (shared loads), since the gate/up are
already near-exact.
