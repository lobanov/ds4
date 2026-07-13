# Lead 08 — floor-clearance verdict (go/no-go on the full-fusion verifier)

Date: 2026-07-13. Task: `floor-clearance-verdict`. **v4 — FINAL, after the three decisive
measurements + codex gates A/B.** Integrates `headroom_decomposition.md` + `exactness_gap.md`
(corrected) + `decisive_measurements.md` + `spec_speedup_model.md`.

## The question

Can a fully-bit-exact sublinear verifier reach the bandwidth floor and clear the +20%
primary gate — justifying the multi-week full-fusion kernel work?

## Gate-clearance threshold (sound; codex-verified)

With the concrete runtime stack (anchor-reuse + GPU drafter draft_ms=10 + oracle acceptance
E[a|4]=2.198, S(4)=0.34; decode_ms=26): **+20% gate clears at `verify_ms(4) ≤ 50.5 ms`**
(speedup = 26·3.198/(10 + verify_ms(4) + 8.84) ≥ 1.20).

## The three decisive measurements (the swap-only test the user required)

1. **MoE-equality (gate A #2):** gate/up ARE bit-exact given identical inputs — codex
   source-verified both paths use `_impl` (same MAC/simd_sum/*0.25), and the consistency
   check shows the 6.7–7.6 gate/up divergence is a ~480× matmul amplification of the 0.014
   *inherited* ffn_norm input (not a reduction mismatch); routed output 0.018 near-exact.
2. **Attention-residual (gate A #3):** **no config swap** forces the batch HC/compressor/
   attention onto the decode reduction (`--quality` is N=2-only) → needs a code change
   (dispatch reroute) = beyond swaps → part of the novel-kernel work.
3. **Bit-exact K=4 verifier (gate B — decisive):** the existing bit-exact verifier (DSpark
   sequential decode verify) = **~0.85× baseline on the 8k corpus** (30.81/31.69/28.85 vs
   36.34/37.46/33.65; full corpus 32.673/39.020 = 0.84×) → **NO-GO.** No config swap yields a
   *sublinear* bit-exact K=4 verifier → **ABORT per the constraint: a gate-clearing bit-exact
   verifier requires NOVEL kernels.** (Gate C: an earlier code_topk 0.41× probe was
   dist-probe-inflated — not representative; the clean 8k ~0.85× is the reference.)

## VERDICT (gate-C-corrected): NO-GO via swaps → bounded build attempt with a hard exit gate

**There is no swap-only path to a gate-clearing bit-exact verifier.** The only existing
bit-exact verifier (linear sequential) is ~0.85× on the 8k corpus (NO-GO); a sublinear
bit-exact verifier does not exist and cannot be produced by kernel-selection/config swaps
(gate C confirmed: no missed swap).

The cost side is **favorable but unproven** (gate C): decode bw ~410–450 GB/s; a sublinear
verifier at decode bandwidth floors at ~39–43 ms → ~1.34–1.44× — **if** built
sublinear+exact, **if** it reaches decode bandwidth, **and if** the stack (GPU drafter +
anchor reuse) is in place. None established (the headroom doc calls K=4 decode-bw reachability
unproven).

**Decision-grade outcome (gate C framing):**
- **NO-GO via swaps** survives.
- The gate is clearable *in principle*, so a **sublinear bit-exact verifier build is
  plausible and worth a BOUNDED attempt** — with a **hard exit gate**: GO is **unconfirmed**
  until an end-to-end K=4 bit-exact verifier profiles `verify_ms(4) ≤ 50.5 ms`.
- This is **not** "gate cleared" or "commit to the build as sufficient." It is "attempt the
  build with a hard exit gate." Correct target: a **sublinear bit-exact batch path**
  (HC/compressor/attention on decode reductions + batched load sharing); the Phase-B
  "single-stage gate/up kernel" was mis-aimed (gate/up already bit-exact).

## What is now ESTABLISHED (robust, post-measurements + gates A/B/C)

- +20% gate threshold `verify_ms(4) ≤ 50.5 ms` (sound).
- Decode effective bandwidth ~410–450 GB/s (byte-data resolved).
- K=4 verify floor at decode bw ~39–43 ms → 1.34–1.44× (clears if built sublinear+exact).
- gate/up are bit-exact given identical inputs (source + consistency).
- The existing bit-exact verifier (linear) is NO-GO (0.41–0.78×).
- **No swap yields a sublinear bit-exact verifier** → novel kernels are required.

## What remains (the gated follow-up, NOT this milestone)

- Build a sublinear bit-exact batch verifier (HC/compressor/attention on decode reductions +
  batched load sharing) + profile verify_ms(4) ≤ 50.5 ms (the GO confirmation) + the stack
  (GPU drafter + anchor reuse). This is the multi-week novel-kernel work the characterization
  was convened to decide whether to commit to. **Verdict: commit.**
