# Lead 08 — prototype-build plan: the M=2 fused routed-expert kernel (single-stage)

Date: 2026-07-16. Goal: `mrmkwnp6-6n9z9x`. Status: **GO (post sub-lead 1); build plan for the multi-day implementation.**
Design source: `issue468/artifacts/lead08_stage_divergence/fused_stage_design.md` (Phase B) + the live code read 2026-07-16.

## The architecture (what gets fused)

`metal_graph_verify_decode2_exact` (ds4.c:21808) runs `metal_graph_encode_decode_layer` (ds4.c:15198) **×2** (one per token). Each calls `ds4_gpu_routed_moe_one_tensor` (ds4_metal.m:22442), which runs:
1. **gate+up+swiglu**: `kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32` (moe.metal:1022) — per-(token, expert), IQ2XXS dequant + MAC + `simd_sum`×0.25 + swiglu → `dst_mid[token,expert]`.
2. **down+sum6**: `kernel_mul_mv_id_q2_K_sum6_f32` (moe.metal:2599) — reads `dst_mid[token,*]`, q2_k down projection, **sums over the 6 experts** → `routed_out[token]`.

The M=2 prototype fuses the **routed-expert stage** for the 2 tokens (the rest of the layer — attention, HC, norms — stays per-token, `decode2_exact`-style, exact).

## The bit-exactness subtlety (discovered 2026-07-16)

The **per-expert** gate+up+swiglu output is bit-identical to M=1 by construction (same dequant, same MAC order per token). BUT the **expert sum** in down+sum6 is FP-non-associative → the sum ORDER must match M=1 (the token's selection order) or `routed_out` diverges. So:
- The M=2 gate+up+swiglu may be **union-ordered** (load each union expert once, apply to selected tokens) — bit-exact per-expert.
- The M=2 down+sum6 must be **per-token, selection-ordered** (each token's 6 experts in ITS order) — to match M=1's sum.

## Refined build — gate+up fused FIRST; down per-token (de-risked)

**Phase 1 (this prototype):** fuse the **gate+up+swiglu** only (the bigger load — gate_up 26.6 ms vs down 18.1 ms at K=4, ~60% of the routed-MoE). Run the **down+sum6 per-token** (2× the existing `id_q2_K_sum6`, selection-ordered → bit-exact, no extra exactness risk). This measures the **gate+up de-dup prize** (the main term) + proves bit-exactness, with one new kernel.
- Pros: simpler (one new kernel: the M=2 gate+up+swiglu); bit-exact (down stays M=1-ordered); measures the main prize.
- The down-fusion (extra de-dup on the q2_k down weights) is a **follow-up** if Phase 1 lands the prize.

## The M=2 gate+up+swiglu kernel design

`kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32_m2` — processes one **union-expert slot**, applied to the selected tokens:
- **Inputs**: `src0_gate/src0_up` (expert weights, indexed by union-expert id), `src1_a/src1_b` (2 tokens' ffn_norm activations), `ids` (union expert ids per slot), `sel_a/sel_b` (per-slot masks: which tokens selected this union expert), `weights_a/weights_b` (route weights per slot), `dst_mid_a/dst_mid_b`.
- **Grid**: `tgpig.z = union_slot` (≤12 union experts for 2 tokens × top-6); `tgpig.x/y` = the output row tile (as M=1).
- **Body**: load the expert's IQ2XXS weights (grid/sign/scale) **once** (shared dequant). For the inner block loop, load `yl_a[32]` (token A) + `yl_b[32]` (token B); MAC into `sumg_a/sumu_a` + `sumg_b/sumu_b` (identical op-order per token as M=1 → bit-identical). Reduce (`simd_sum`×0.25) + swiglu per token; write `dst_mid_{a,b}[expert]` only for the selected token(s) (the mask).
- **Bit-exactness**: each token's per-expert gate/up is bit-identical to a standalone M=1 call (same dequant reused, same MAC order). ✓

## The host dispatch — `ds4_gpu_routed_moe_pair_tensor`

Mirror `ds4_gpu_routed_moe_one_tensor` for 2 tokens:
1. Compute the **union** of the 2 tokens' top-6 expert selections (≤12) + per-token selection masks + remap the route weights to the union slots.
2. Dispatch the M=2 gate+up+swiglu kernel over the union (shared load) → `dst_mid_a/b`.
3. Dispatch the existing `id_q2_K_sum6` down kernel **twice** (once per token, selection-ordered) → `routed_out_a/b`. (No down-fusion in Phase 1.)

## The wiring (env-gated)

In `metal_graph_verify_decode2_exact` (or a paired-variant of `encode_decode_layer`), when `DS4_DSPARK_FUSED_ROUTED_M2=1`: instead of running `encode_decode_layer` twice (each calling `routed_moe_one_tensor`), run the per-token stages (norm, attention, HC) for both tokens, then call `routed_moe_pair_tensor` **once** for the routed-expert stage. Everything else per-token (exact). Env-gated default-off.

## Phased implementation + test plan

1. **M=2 gate+up+swiglu kernel** → compile-clean (moe.metal). [the core]
2. **`routed_moe_pair_tensor` dispatch** (union + masks + the 2 kernels). [ds4_metal.m]
3. **Wiring** (`DS4_DSPARK_FUSED_ROUTED_M2` branch in decode2_exact). [ds4.c]
4. **Fidelity gate** (the exactness secondary): `DS4_DSPARK_VERIFY_DIST_PROBE`-style harness — compare the M=2 `routed_out`/`ffn_moe_out` vs the M=1 (decode2_exact) reference at clean-input positions. **Pass = `max_abs==0` bit-for-bit** (per-expert bit-exact by construction; the down sum-order is preserved by the per-token down). + 0 argmax flips on the exactness corpus (temp=0). On failure → codex bug-hunt on the reduction.
5. **Cost gate** (the viability bar): `verify_ms(K)` (M=2 vs the batch verifier) via ds4-spec-bench at K=2..5, warm, bootstrap CI. Pass = strictly < the batch verifier at matched K. Record the end-to-end delta (stacked on +4.9%).
6. **codex gate B** (post-prototype) → verdict-propagate.

## Realistic expectation
The gate+up de-dup prize ≈ 28.5% of the gate_up portion (~26.6 ms inflated → ~16 ms real at K=4) ≈ **~4.5 ms** (gate+up only; the down stays per-token). That's **~+13% over plain** — meets the primary bar ("faster than batch"), short of +20% (which needs the down-fusion too, Phase 2). The cost gate will measure the actual.
