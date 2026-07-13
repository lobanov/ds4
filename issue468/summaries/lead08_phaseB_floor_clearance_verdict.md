# Lead 08 Phase B — floor-clearance characterization (go/no-go verdict)

Date: 2026-07-13. Canonical summary of the Lead 08 Phase B **characterization milestone**
(reframed from "build a single-stage fused kernel" to "decide go/no-go before committing to
multi-week kernel work"). Decision-grade, codex-gated (A + B).
Artifacts: `issue468/artifacts/lead08_stage_divergence/` (`stage_divergence_map.json`,
`headroom_decomposition.md`, `exactness_gap.md` [corrected], `fused_stage_design.md`,
`floor_clearance_verdict.md`); reviews `issue468/artifacts/dspark_codex_reviews/2026-07-13_gpt55_xhigh_lead08_gate{A,B}_*.md`.

## Notation

`verify_ms(K)` target-verify wall time; `decode_ms`=26 (baseline 38.5 t/s); `draft_ms`=10
(GPU drafter); cycle-jump `E[a|4]`=2.198, `S(4)`=0.34; anchor-reuse accounting. Speedup =
`26·3.198/(10 + verify_ms(4) + 8.84)`.

## Verdict: HOLD / INCONCLUSIVE

A fully-bit-exact sublinear verifier **plausibly clears** the +20% gate, but the
characterization cannot confirm it without one end-to-end measurement. **Do not commit to
the multi-week novel-kernel build yet.** The single decisive test (gate B): **profile a
bit-exact K=4 verifier's `verify_ms(4)` under deployed accounting** — ≤ 50.5 ms → GO; >
50.5 ms → NO-GO.

## Robust findings (post both codex gates)

1. **Gate threshold:** +20% clears at **`verify_ms(4) ≤ 50.5 ms`** (sound; both gates
   re-derived 50.45).
2. **Decode bandwidth resolved (gate B, independently verified):** decode effective
   bandwidth is **~410–450 GB/s** (bytes/token = dense 8.20 + experts 1.70 + KV ≈ 10.4 GiB
   at decode_ms 26). The Lead 08 doc's "300 GB/s" was conservative; gate A's "≤250 cliff"
   is not real.
3. **Verify floor at decode bandwidth:** K=4 verify bytes ≈ 15.5 GiB → **~39 ms at 428 GB/s
   → 1.44× (clears comfortably)**; break-even ~329 GB/s. So the cost side is **favorable
   IF a fused/tuned verifier approaches decode bandwidth** (the batch path currently runs
   its expert stream at only ~190 GB/s — batch-kernel inefficiency; the K3→K4 vs K4→K5
   slope is unstable, so "190" is a rough marginal).
4. **Headroom location:** the ~21 ms verify headroom is **GPU `layer_execute` bandwidth
   inefficiency**, not host overhead (~3 ms) or readback (~0).
5. **Stage divergence (layer 40/pos 61):** expert SELECTION identical; gate/up reduction is
   source-identical to decode (`_impl`); the reliable per-token output divergences are
   attention/hidden (attn_out 0.064, KVcur 0.125) > FFN outputs (ffn_moe_out 0.018).

## Corrected claims (gates A+B — the draft overclaimed)

- **"F16 (batch) vs F32 (decode), closable by swap" — WRONG.** Decode ALSO uses
  `ds4_gpu_matmul_f16_tensor` for HC (`matmul_plain_tensor` dispatches on `w->type`;
  `hc_attn_fn` is F16); compressor is F16 on both paths. The divergence is
  reduction/path/order, **not dtype**.
- **"gate/up already bit-exact" / "large divergences are a layout artifact" — unverified /
  false.** The 6.2–7.6 gate/up divergences are REAL (recomputed); pos-61 is NOT clean-input
  (inputs differ before MoE), so the divergence is plausibly inherited, not a proven
  reduction diff. An identical-input MoE kernel-equality test would settle it.
- **"Phase-B targeted the wrong thing" — too strong.** Expert-load bandwidth sharing is
  still relevant; only an exactness-only gate/up target would be "wrong."
- **"Two decisive measurements" — false.** They are necessary probes, not sufficient; the
   single decisive test is the bit-exact K=4 verifier profile.

## Implication for the work plan

The decisive test (bit-exact K=4 verifier profile) **likely does NOT need novel IQ2_XXS
kernels** — it needs the batch path's HC/compressor/attention on the decode reductions
(kernel-selection/config swaps) + a profile. That is substantially cheaper than the
original Phase-B single-stage-kernel plan and is the right next slice.

## What is NOT established (→ the decisive follow-up)

Whether a bit-exact K=4 verifier profiles ≤ 50.5 ms: the exactness fixes (HC/compressor/
attention on decode reductions) may cost speed, and it is unproven that a K=4 fused
verifier reaches ~340+ GB/s. That single end-to-end measurement decides GO vs NO-GO.
