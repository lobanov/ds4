# Lead 08 Phase B — floor-clearance characterization + decisive measurements (FINAL verdict)

Date: 2026-07-13 (final). Canonical summary of the Lead 08 Phase B milestone: a
measurement-driven go/no-go on whether a fully-bit-exact sublinear verifier can clear the
+20% gate, BEFORE committing to multi-week kernel work. Codex-gated A + B + C.
Artifacts: `issue468/artifacts/lead08_stage_divergence/` (`stage_divergence_map.json`,
`headroom_decomposition.md`, `exactness_gap.md` [corrected], `fused_stage_design.md`,
`decisive_measurements.md`, `floor_clearance_verdict.md` [v4]); reviews
`issue468/artifacts/dspark_codex_reviews/2026-07-13_gpt55_xhigh_lead08_gate{A,B,C}_*.md`.

## FINAL VERDICT: NO-GO via swaps → bounded build attempt with a hard exit gate

- **There is no kernel-selection/config swap that yields a gate-clearing bit-exact verifier.**
  The only existing bit-exact verifier (linear sequential) is **~0.85× baseline on the 8k
  corpus** (30.81/31.69/28.85 vs 36.34/37.46/33.65; full corpus 0.84×) — NO-GO.
- A **sublinear bit-exact verifier does not exist** and cannot be produced by swaps
  (`verify_suffix_tops` is sublinear-but-not-exact; `decode2_exact` is exact-but-N=2-linear;
  no config reroutes the batch HC/compressor/attention onto decode reductions) → it requires
  **novel kernels**.
- The cost side is **favorable but unproven**: decode bw ~410–450 GB/s; a sublinear verifier
  at decode bandwidth floors at ~39–43 ms → ~1.34–1.44× — *if* built sublinear+exact, *if* it
  reaches decode bandwidth, *and if* the stack (GPU drafter + anchor reuse) is in place.
  None established.
- **Decision: attempt the sublinear bit-exact batch-path build as a BOUNDED effort with a
  hard exit gate.** GO is **unconfirmed** until an end-to-end K=4 bit-exact verifier profiles
  `verify_ms(4) ≤ 50.5 ms`. This is NOT "gate cleared" or "commit to the build as sufficient."

## The three decisive measurements (swap-only, per the constraint)

1. **MoE-equality (gate A #2):** gate/up are **bit-exact given identical inputs — consistent
   with** the source (both use `_impl`: same MAC/simd_sum/*0.25) + a consistency check (the
   6.7–7.6 gate/up divergence is a ~480× matmul amplification of the 0.014 *inherited* ffn_norm
   input; routed output 0.018 near-exact via cancellation). (Gate C: "consistent with," not
   "confirmed" — a literal identical-input harness is the confirmatory follow-up.)
2. **Attention-residual (gate A #3):** **no config swap** forces the batch HC/compressor/
   attention onto the decode reduction (`--quality` is N=2-only) → needs a code change
   (dispatch reroute) = beyond swaps. (Gate C confirmed: no missed swap.)
3. **Bit-exact K=4 verifier (gate B — decisive):** existing bit-exact verifier (DSpark
   sequential) = **~0.85× baseline (8k)** → NO-GO; no sublinear-exact swap → needs novel
   kernels. (Gate C: an earlier code_topk 0.41× probe was dist-probe-inflated — not
   representative.)

## Robust findings (post gates A/B/C)

- +20% gate threshold: **`verify_ms(4) ≤ 50.5 ms`** (sound).
- Decode effective bandwidth **~410–450 GB/s** (byte-data resolved; the "300" was conservative).
- K=4 verify floor at decode bw **~39–43 ms → 1.34–1.44×** (clears *if* built sublinear+exact).
- ~21 ms verify headroom = GPU `layer_execute` bandwidth inefficiency, not host overhead (~3 ms).
- gate/up bit-exact given identical inputs (consistent with source + consistency).
- The existing bit-exact verifier (linear) is NO-GO (~0.85×).
- **No swap yields a sublinear bit-exact verifier** → novel kernels required.

## Corrections through the gate chain (the draft overclaimed at each stage)

- Gate A: the divergence is NOT F16-vs-F32 (decode uses F16 too) → reduction/path/order; the
  gate/up large divergences are real (not a layout artifact); the 300 GB/s decode figure is
  load-bearing (later resolved by gate B).
- Gate B: decode bw resolved (~410–450 GB/s); HOLD confirmed, reason sharpened; the decisive
  test = bit-exact K=4 verifier profile.
- Gate C: moe "confirmed"→"consistent with"; the code_topk 0.41× was dist-probe-inflated (use
  8k ~0.85×); "commit to novel-kernel build"→"bounded build attempt with hard exit gate."

## Implication for the work plan (the gated follow-up)

The correct build target is a **sublinear bit-exact batch path** (HC/compressor/attention on
decode reductions + batched load sharing) — NOT the Phase-B "single-stage gate/up kernel"
(gate/up are already bit-exact). Attempt it as a bounded effort; the hard exit gate is
`verify_ms(4) ≤ 50.5 ms` on an end-to-end K=4 bit-exact verifier. Until that profiles, GO is
unconfirmed.
