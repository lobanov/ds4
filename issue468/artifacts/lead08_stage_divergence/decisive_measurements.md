# Lead 08 — decisive measurements (codex-suggested, gate A #2/#3 + gate B)

Date: 2026-07-13. Tasks: `moe-equality-test`, `attention-residual-exact`, `bit-exact-k4-profile`.
Turns the HOLD characterization into a concrete GO/NO-GO. Constraint: bit-exact K=4
verifier via kernel-selection/config **swaps only**; abort + report if novel kernels required.

## 1. MoE kernel-equality (gate A #2) — gate/up ARE bit-exact given identical inputs

Codex source-verified both paths use the same reduction (`kernel_mul_mv_iq2_xxs_pair_f32_impl`
/ inline duplicate: same `yl`, MAC order, `simd_sum`, `*0.25`). Consistency check on the
pos-61 dumps (batch row0 vs seq) confirms the divergence is **inherited, not a reduction mismatch**:

| stage | dim | max_abs diff | rms diff | amplification vs ffn_norm input |
|---|---:|---:|---:|---:|
| ffn_norm (MoE input) | 4096 | 0.0141 | 0.00388 | (input) |
| ffn_moe_gate_clamped | 12288 | 6.74 | 1.33 | ~477× (matmul amplification, consistent) |
| ffn_moe_up_clamped | 12288 | 7.55 | 1.25 | ~535× |
| ffn_moe_weighted_swiglu | 12288 | 0.0300 | 0.00155 | ~2.1× |
| ffn_moe_out (routed output) | 4096 | 0.0177 | 0.00444 | ~1.2× |

The ~480–535× gate/up internal amplification of the 0.014 input is what a 4096→2048 IQ2XXS
matmul produces (rms weight magnitude ~5) — i.e. the divergence is fully explained by the
inherited upstream input difference. The routed **output** (ffn_moe_out 0.018) is near-exact
via downstream cancellation. **Verdict: gate/up are bit-exact given identical inputs — consistent with the source
+ consistency check** (a literal identical-input kernel-equality harness is the
confirmatory follow-up; gate C flagged "confirmed" is too strong without it).

## 2. Attention-residual with forced decode-reduction (gate A #3) — NOT achievable as a swap

Searched the batch-path config surface (`encode_layer_attention_batch`/`encode_layer_ffn_batch`):
**no env/config forces the batch HC/compressor/attention onto the decode reduction.**
`--quality`/`DS4_MTP_STRICT` selects `decode2_exact` **for N=2 only** (a separate exact
verifier, not a batch-path reduction swap). Forcing decode reductions in the batch path needs
a **code change (dispatch reroute)** — beyond "kernel-selection/config swaps." So the
attention/compressor divergence is **not closable via a config swap**; it is part of the
novel-kernel work. (Finding, not a measured residual.)

## 3. Bit-exact K=4 verifier profile (gate B — the decisive test) — NO-GO via swaps

The existing bit-exact verifier is the DSpark default path (sequential decode verify,
`metal_graph_eval_token_raw_swa_top` per draft — uses the decode kernels → bit-exact).
Clean reference (NO dist-probe overhead) = the retained 8k exact-reuse numbers
(`summaries/mtp_verifier_engineering_and_phaseA.md`):

- exact reuse K=4: **30.81 / 31.69 / 28.85 t/s** vs baseline **36.34 / 37.46 / 33.65** =
  **0.848 / 0.846 / 0.857 (~0.85×)**; full 300-prompt corpus 32.673 / 39.020 = **0.837×**.

Both are **NO-GO** (below baseline, far below the +20% gate = ~46.8 t/s).

> ⚠ Gate C correction: an earlier code_topk probe run with `DS4_DSPARK_VERIFY_DIST_PROBE=1`
> showed 0.41×, but that probe runs the batched verifier **in addition** to the real
> sequential verify (`ds4.c:28660`), inflating the time — **not representative**. The clean
> 8k exact-reuse ~0.85× is the correct reference.

There is **no config swap that yields a sublinear bit-exact K=4 verifier** — the batch path
(sublinear) is not exact, and the only exact path is the linear sequential verifier
(decode2_exact is exact but N=2-only + linear). **→ ABORT per the constraint: a gate-clearing
bit-exact K=4 verifier requires NOVEL kernels** (a sublinear+exact verifier).

## VERDICT (gate-C-corrected): NO-GO via swaps → bounded build attempt with a hard exit gate

The decisive measurement is unambiguous: **there is no swap-only path to a gate-clearing
bit-exact verifier.** The existing bit-exact verifier (linear sequential) is ~0.85× on the
8k corpus (NO-GO); a sublinear bit-exact verifier does not exist and cannot be produced by
kernel-selection/config swaps (gate C confirmed: no missed swap).

The cost side is **favorable but unproven** (gate C): decode bw ~410–450 GB/s; a sublinear
verifier at decode bandwidth would floor at ~39–43 ms → ~1.34–1.44× — **if** built
sublinear+exact, **if** it reaches decode bandwidth, **and if** the stack (GPU drafter +
anchor reuse) is in place. None of those is established (the headroom doc itself calls K=4
decode-bandwidth reachability unproven).

**Decision-grade outcome (gate C framing):**
- **NO-GO via swaps** survives.
- The gate is clearable *in principle*, so a **sublinear bit-exact verifier build is
  plausible and worth a BOUNDED attempt** — but with a **hard exit gate**: GO is **unconfirmed**
  until an end-to-end K=4 bit-exact verifier profiles `verify_ms(4) ≤ 50.5 ms`.
- This is **not** "gate cleared" or "commit to the build as sufficient." It is "attempt the
  build with a hard exit gate." The correct build target is a **sublinear bit-exact batch
  path** (HC/compressor/attention on decode reductions + batched load sharing); the Phase-B
  "single-stage gate/up kernel" was mis-aimed (gate/up are already bit-exact).
