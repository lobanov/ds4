## Verdict Per Claim

1. **190 GB/s slope: questionable.** Arithmetic works only for K3→K4: `1.140 GiB / 6.136 ms = 185.8 GiB/s` (`199.5 GB/s decimal`) from [summary.json](/Users/lobanov/Projects/ds4-dspark-research/issue468/artifacts/mtp_phaseA_profile/summary.json). K4→K5 is `0.560 GiB / 10.048 ms = 55.7 GiB/s`; that contradiction makes K3→K4 an optimistic marginal, not a stable “expert stream bandwidth.”

2. **Headroom decomposition: questionable.** `layer_execute` includes dense HC/QKV, attention/KV, shared FFN, routed MoE, and output work. `selected_gib` is routed-expert bytes only. Using its marginal slope to rescale all K4 `layer_execute` is not a sound isolation unless non-expert terms are proven flat; K4→K5 disproves that simplification.

3. **300 GB/s decode floor: asserted, load-bearing.** With their rounded 190 basis: 300 GB/s → `42.98 ms`, `1.345x`; 250 → `50.90 ms`, `1.192x`; 200 → `62.80 ms`, `1.019x`. At 250 the +20% gate already fails or is cliff-edge depending on exact unit rounding.

4. **Gate/up bit-exact: mostly source-supported, not measured.** Batch `addr` calls `_impl` at [moe.metal:1308](/Users/lobanov/Projects/ds4-dspark-research/metal/moe.metal:1308); decode `id` is an inline duplicate at [moe.metal:1022](/Users/lobanov/Projects/ds4-dspark-research/metal/moe.metal:1022), not a call. I found no reduction-order divergence: same `yl`, same MAC order, same `simd_sum`, same `*0.25` as `_impl` [moe.metal:680](/Users/lobanov/Projects/ds4-dspark-research/metal/moe.metal:680). But “both call same `_impl`” is false.

5. **“F16 is dominant divergence source”: likely wrong / at least inferred.** Code proves batch HC uses `ds4_gpu_matmul_f16_tensor` [ds4.c:17766](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:17766), [ds4.c:19216](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:19216). But decode HC uses `metal_graph_matmul_plain_tensor` [ds4.c:15246](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:15246), which maps F16 weights to the same F16 matmul [ds4.c:16623](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:16623); target layout expects HC F16 [ds4.c:3655](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:3655). Compressor is also F16 on decode [ds4.c:15451](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:15451), [ds4.c:15467](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:15467) and batch [ds4.c:18122](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:18122). Confound is reduction/path/order, not just dtype.

6. **“Layout artifact” dismissal: likely wrong.** I recomputed the dump deltas and reproduced the map. Simple per-expert permutation does not clear the large deltas: gate best max `6.22`, up `7.55`, down still `331.20`. `ffn_moe_out` is small (`0.01765`), but that proves cancellation/downstream insensitivity, not that internals were layout artifacts.

7. **Projection arithmetic: sound; conclusion fragile.** Formula gives threshold `verify_ms <= 50.45`. Speeds: `43 ms → 1.345x`, `47 → 1.263x`, `50 → 1.208x`, `55 → 1.126x`. The “47–55 ms borderline” framing is honest; the “bit-exact F32 47–50 ms” estimate is not measured.

## Premise Sensitivity

Baseline: threshold/floor-speed = `50.45 ms / 1.346x`.

| One-at-time change | -25% | -10% | +10% | +25% |
|---|---:|---:|---:|---:|
| `E[a|4]` | `38.54 / 1.114x` | `45.69 / 1.253x` | `55.21 / 1.438x` | `62.36 / 1.577x` |
| `S(4)` | `52.66 / 1.395x` | `51.33 / 1.365x` | `49.57 / 1.327x` | `48.24 / 1.299x` |
| `decode_ms` | `35.34 / 1.047x` | `44.41 / 1.229x` | `56.50 / 1.459x` | `65.56 / 1.624x` |
| `draft_ms` | `52.95 / 1.402x` | `51.45 / 1.368x` | `49.45 / 1.324x` | `47.95 / 1.293x` |

Tightest model inputs: `decode_ms` and `E[a|4]`. Tightest external premise: the asserted 300 GB/s decode bandwidth.

Readback is not literally zero in the deployed path: profile `logits_read_ms` is zero when `row_logits == NULL` inside `metal_graph_verify_suffix_tops` [ds4.c:21748](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:21748), but the normal path later reads one logits row [ds4.c:29323](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:29323). Probably small, but not accounted by that profile field.

## New Experiments To Try

1. **Direct decode bandwidth measurement.** Run decode stage/profile on the same M5 Max/IQ2XXS target and compute effective bytes/ms. Signal: if decode is ≤250 GB/s, the headline projection is at/below gate. Effort: low.

2. **Identical-input MoE kernel equality harness.** Feed the same dumped `ffn_norm`, topk ids, and weights through `routed_moe_one_tensor` and `routed_moe_batch_tensor`/addr; dump gate/up/mid/down/out. Signal: zero diff settles “already bit-exact”; nonzero kills it. Effort: low-medium.

3. **Force exact HC/compressor/attention path, then rerun dist-probe.** Expected signal: `hc_*`, `KVcur`, `attn_out` residuals collapse or remain. Effort: medium.

4. **K=4 profile with exactness swaps enabled.** Measure `verify_ms(4)` after exactness fixes, before full fusion. Signal: if >50.45 ms, no +20% gate. Effort: medium.

## Leading Assessment

I would not accept “CONDITIONAL GO” as decision-grade yet. The arithmetic ceiling is real, but the methodology relies on a cherry-picked bandwidth slope, an unmeasured 300 GB/s decode premise, an inferred F16 divergence story contradicted in places by source, and an unsupported layout-artifact dismissal.

Decisive GO/NO-GO test: a retained K=4 run that is both bit-exact and profiles `verify_ms <= 50.45 ms` under the actual deployed verifier accounting. Without that, this is a hold, not a multi-week kernel commitment.