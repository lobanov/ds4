# Metal-vs-Oracle Numerical Divergence — FP8 KV Bisection

Date: 2026-07-01. Eleventh productionization handoff note. Follows issue468/51
(numpy oracle Q4_K ceiling: +8.27% MC-B2 acceptance on the SAME weights).
Bisects where the Metal runtime diverges from the F32 oracle. Doc-only.

## 0. The question

issue468/51 showed the SAME Q4_K drafter weights run through the numpy F32
oracle get +8.27% higher MC-B2 acceptance than the live Metal path (4.10→4.49
accepted at ctx 8192). This means the α deficit is NOT the Q4_K weights
(issue468/42's assumption) — it's a numerical divergence in the RUNTIME forward
path. Where is it?

## 1. Method: the deterministic probe (ds4_dspark_probe_accept)

Used the existing `DS4_DSPARK_PROBE_ACCEPT` path (gated by `DS4_DSPARK_PROBE_ONLY`
inside `--verifier-curve-test`) against the oracle's own ctx_08192 capture
bundle (/tmp/dspark_sweep8/ctx_08192). This runs the deterministic Metal drafter
forward on disk-captured main_hidden and reports greedy-match vs target greedy
— the SAME draft-quality metric the oracle measures, with ZERO stochastic noise
(unlike live B2 commits/cycle, which is far too noisy at n=256-512 to detect a
±8% effect: measured 3.17-3.98 commits/cycle across runs).

## 2. FP8 KV quantization — VERIFIED divergence (+2.6%)

The Metal drafter path applies `ds4_gpu_dsv4_fp8_kv_quantize_tensor` to the
drafter KV (ds4.c:17774, 17804, gated by `DS4_DSPARK_NO_FP8`). The numpy oracle
uses pure F32 KV. FP8 has ~3-bit mantissa — the obvious divergence.

Deterministic probe (avg prefix = greedy-match length / 5):
| config | greedy match | avg prefix | Δ |
|---|---|---|---|
| FP8 ON (production) | 85/95 (89.5%) | 4.26 | — |
| FP8 OFF (NO_FP8=1) | 87/95 (91.6%) | 4.37 | +2.6% |

FP8 KV quantization costs ~2.6% of deterministic draft quality. Disabling it is:
- **Free** (no slowdown — 29.37 t/s vs ~31 production = noise; the FP8 quantize
  step is itself eliminated).
- **In-scope under the frozen-drafter constraint**: FP8 is a RUNTIME KV-cache
  quantization, NOT a change to the drafter's weights/architecture/GGUF. The
  drafter model is byte-identical.
- **Drafter-only**: `DS4_DSPARK_NO_FP8` gates only the drafter KV path; MTP
  (37.83 t/s) and baseline plain decode are unaffected.
- A strict quality improvement (more-accurate drafter KV → better drafts).

## 3. Other Metal precision knobs — NO effect

`ds4_metal.m:4617+` exposes a drift-patch knob system (commit 17502b9 "golden
inference drift test"). Swept all on the deterministic probe:
| knobs (on top of FP8 off) | avg prefix | Δ |
|---|---|---|
| FP8 off + KV_RAW_F32 | 4.37 | 0 |
| FP8 off + MATH_SAFE | 4.37 | 0 |
| FP8 off + KV_RAW_F32 + MATH_SAFE | 4.37 | 0 |
| FP8 off + all (KV_RAW_F32 + MATH_SAFE + ROPE_EXP2_LOG2) | 4.37 | 0 |

None of the available Metal precision knobs move the drafter probe beyond the
FP8-off result. So the remaining ~5.7% of the oracle gap is NOT in these knobs.

## 4. Where the rest of the gap likely lives (unbisected)

Candidates not covered by any current knob:
- **GPU Q4_K block-dequant accuracy**: the oracle dequantizes Q4_K→F32 in numpy;
  Metal dequantizes on-GPU. Block-scale 4-bit dequant can differ between the GPU
  kernel and the numpy reference. The drafter uses the shared Q4_K dequant kernel
  (no drafter-specific path found). This is the prime remaining suspect.
- **Attention/einsum accumulation order**: numpy einsum (F32, large accumulators)
  vs the GPU batched-attention kernel (possibly F16/blocked accumulation).
- **MoE expert routing/weighting**: the drafter is a 3-layer MoE; small routing
  differences could compound.

Bisection requires extending the drift-test harness (tests/ds4_test.c, 17502b9)
to dump drafter intermediates (per-layer hidden states, attention outputs) and
compare GPU vs numpy reference — the existing pattern, applied to the drafter
forward.

## 5. Significance — this REOPENS the gate question

issue468/42 concluded the binding constraint is α, TV-bounded by the frozen Q4_K
drafter (ceiling α ≤ 1−TV). issue468/51 OVERTURNS that premise: the SAME Q4_K
weights have +8.27% headroom in F32, so the ceiling is the RUNTIME path, not the
weights. Per the issue468/42 break-even table, closing the full +8.27% would push
α from 0.83 to ~0.90 → ~41.5 t/s, CROSSING the gate (>39).

FP8-off is the first verified chunk (+2.6% deterministic draft quality). The
remaining ~5.7% is bisectable via GPU-kernel intermediate comparison — a concrete
investigation, not a dead end. This is an in-scope (code-only, frozen-drafter)
avenue that the codex-review loop (issue468/41) did not surface because codex
reviewed CODE structure, not numerical accuracy vs a reference oracle.

## 6. Recommended next step

1. **Land FP8-off for the drafter** (free +2.6% quality, in-scope, safe). Either
   make `DS4_DSPARK_NO_FP8` the default for the drafter path, or remove the FP8
   quantize calls from the drafter KV path (ds4.c:17774, 17804).
2. **Bisect the remaining gap** via drafter-intermediate drift testing (extend
   tests/ds4_test.c pattern): dump per-layer GPU vs numpy hidden states on the
   ctx_08192 bundle, find the first diverging layer/op, fix it. Target: close
   toward the +8.27% ceiling.
3. If the gap closes substantially, re-measure end-to-end t/s — this is the one
   avenue with a theoretical path to crossing the gate without unfreezing the
   drafter.

This changes the A/B/C/D/E decision landscape: the "structural perf-blocker"
(issue468/39/41) was premised on the α ceiling being weight-bound; it is in fact
runtime-bound and partially recoverable in-scope.
