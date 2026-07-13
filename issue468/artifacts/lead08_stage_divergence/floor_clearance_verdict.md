# Lead 08 — floor-clearance verdict (go/no-go on the full-fusion verifier)

Date: 2026-07-13. Task: `floor-clearance-verdict`. **v3 — final, after codex gates A + B**
(gate A caught that the CONDITIONAL-GO draft overclaimed; gate B resolved the cost side
and sharpened the HOLD reason). Integrates the corrected analyses + both codex gates.

## The question

Can a fully-bit-exact sublinear verifier reach the bandwidth floor and clear the +20%
primary gate — justifying the multi-week full-fusion kernel work?

## Gate-clearance threshold (sound; both gates verified)

With the concrete runtime stack (anchor-reuse + GPU drafter draft_ms=10 + oracle
acceptance E[a|4]=2.198, S(4)=0.34; decode_ms=26):

```
speedup(4) = 26 × 3.198 / (10 + verify_ms(4) + 8.84)
+20% gate clears at  verify_ms(4) ≤ 50.5 ms        (gates re-derived 50.45 ms ✓)
```

## Cost side — DECODE BANDWIDTH RESOLVED (gate B, independently verified)

Gate A flagged the "300 GB/s decode" figure as asserted + fragile (≤250 → fails). **Gate B
resolved it from existing byte data, and I independently verified:**

```
decode bytes/token ≈ dense 8.20 GiB + selected experts (6/256 × 72.56) 1.70 GiB + KV
  ctx 8k:  ~10.38 GiB at decode_ms 26 ms → ~429 GB/s (decimal)
  ctx 16k: ~10.85 GiB                   → ~448 GB/s
```

**Decode effective bandwidth is ~410–450 GB/s — comfortably ≥ 300 (the Lead 08 "300" was
conservative).** So the decode-bandwidth premise is NOT fragile; the gate-A ≤250 cliff is
not real.

**Verify floor at decode bandwidth:** K=4 verify bytes ≈ dense 8.20 + union 5.39 + KV(4×)
≈ 15.5 GiB. At decode's ~428 GB/s → **~39 ms → 1.44× (clears +20% comfortably)**. At 300
GB/s → 55.5 ms (fails); the break-even is ~329 GB/s. So the cost side is **favorable IF the
K=4 fused verifier approaches decode bandwidth** — the verify path currently runs its
expert stream at only ~190 GB/s (batch-kernel inefficiency), so the engineering question is
whether a fused/tuned verifier reaches ~340+ GB/s. (The K3→K4 vs K4→K5 slope instability
gate-A flagged means the "190 GB/s" itself is a rough marginal, not a stable measurement.)

## Exactness side — the F16 story was WRONG (gate A, verified); real source unidentified

- Draft claimed "F16 (batch) vs F32 (decode), closable by swap." **False:** decode ALSO uses
  `ds4_gpu_matmul_f16_tensor` for HC (`matmul_plain_tensor` dispatches on `w->type`;
  `hc_attn_fn` is F16) and the compressor is F16 on both paths. The divergence is
  **reduction/path/order, not dtype** — "closable by F32 swap" is unfounded.
- "gate/up already bit-exact": source-identical reduction (no MAC/simd_sum/*0.25 divergence
  — gate A), BUT the measured gate/up divergences (6.2–7.6) are **real, not a layout
  artifact** (gate A recomputed the dumps). Gate B clarified the pos-61 comparison is **not
  clean-input** (attn_norm/hc_attn_pre/KVcur/ffn_norm all differ before MoE), so the gate/up
  divergence is **plausibly inherited from upstream, not a reduction diff** — but this is
  NOT proven. An identical-input MoE kernel-equality test would settle it.

## VERDICT: HOLD / INCONCLUSIVE

**HOLD is the right call (both gates agree), but for a sharper reason than the draft:**

- The **cost side is favorable** (decode ~428 GB/s resolved; floor at decode bw ~39 ms →
  1.44×; clears if the fused verifier approaches decode bw).
- The **barrier is the unmeasured cost of a bit-exact K=4 verifier**: the exactness fixes
  (HC/compressor/attention on the decode reduction, NOT simple dtype swaps) may cost speed,
  and it is unproven that a K=4 fused verifier reaches ~340+ GB/s.
- This is **not a GO** (MoE-equality alone doesn't prove full exactness; exact-path speed
  cost unmeasured) and **not a NO-GO** (cost side is favorable). 

**Do NOT commit to the multi-week novel-kernel build.**

## The ONE decisive test (gate B)

**Build/config a bit-exact K=4 verifier and profile `verify_ms(4)` under deployed
accounting.** ≤ 50.5 ms → GO; > 50.5 ms → NO-GO. This likely does NOT need novel IQ2_XXS
kernels — it needs the batch path's HC/compressor/attention on the decode reductions
(config/kernel-selection swaps) + a profile. That is far cheaper than the Phase-B plan and
IS the decisive measurement (gate B: the separate decode-bw + MoE-equality probes are
necessary but not sufficient — only the end-to-end bit-exact K=4 timing decides it).

## Corrections to earlier overclaims (gates A+B)

- "Phase-B targeted the wrong thing" — **too strong** (gate B). If Phase-B targeted
  expert-load bandwidth sharing, that's still relevant; only an exactness-only gate/up
  target would be "wrong."
- "Two decisive measurements" — **false** (gate B); they're necessary probes, not
  sufficient. The single decisive test is the bit-exact K=4 verifier profile.
- "F16 vs F32, closable by swap" — **wrong** (gate A); F16 is on both paths.
- "gate/up already bit-exact" / "layout artifact" — **unverified/false** (gate A); the
  divergences are real; pos-61 isn't clean-input.

## What this milestone DOES establish (robust, post-gates)

- Gate threshold verify_ms(4) ≤ 50.5 ms (sound).
- Decode effective bandwidth ~410–450 GB/s (resolved from byte data).
- Headroom is GPU `layer_execute`, not host overhead (~3 ms).
- The verify floor at decode bandwidth (~39 ms) clears the gate IF a fused verifier
  approaches decode bandwidth.

## What it does NOT establish (→ the decisive follow-up test)

- Whether a bit-exact K=4 verifier profiles ≤ 50.5 ms (the exactness-speed tradeoff is
  unmeasured). That single test decides GO vs NO-GO.
