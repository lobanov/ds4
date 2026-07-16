## Verdict Per Claim

- **verify_ms(K) curve: sound.** I re-derived file-level means: K2 `37.76/41.36 ms`, K3 `48.26/52.34`, K4 `54.95/59.14`, K5 `63.85/68.64` for `layer_execute/verify`. Matches the probe.
- **linear fit / 33.5 ms expert portion: questionable.** Fit is descriptive, not causal: `layer_execute = 21.48 + 4.99 ms/GiB * physical`, R² `0.993`, but the same fit is basically `~21.47 + 8.50 ms*K`. Physical bytes are collinear with token count.
- **~208 GB/s: numerically plausible, not robust.** Depending on inclusion of short `tokens<K` rows: slope is `4.99` to `5.15 ms/GiB`; drop K2 gives `4.58 ms/GiB`. K4 “expert” becomes `31.2-35.1 ms`, not a measured `33.5`.
- **selected GiB is union bytes: sound.** Code uses `unique * per_expert_bytes` at [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:12454); dispatch still runs pairs via `nei0*nei1` at [ds4_metal.m](/Users/lobanov/Projects/ds4-dspark-research/ds4_metal.m:20948).
- **physical-pairs ⇒ DRAM cache-miss: questionable.** Current kernel is per-pair, yes: [metal/moe.metal](/Users/lobanov/Projects/ds4-dspark-research/metal/moe.metal:1273). But K4→K5 also adds one token’s dense/attention/shared work.

## Cache-Miss Inference

Not decisive. K4→K5 `+8.91 ms` vs `+1.70 GiB physical` is suggestive, but K2→K3/K3→K4/K4→K5 physical slopes are inconsistent: about `6.17`, `3.93`, `5.24 ms/GiB`. It proves “not union-only”; it does **not** prove redundant expert loads are DRAM misses.

Tightest no-new-kernel separator: rerun K4/K5 with existing batch-stage profiling, especially `DS4_METAL_LAYER_STAGE_PROFILE=1` and `DS4_METAL_MOE_STAGE_PROFILE=1`, aggregate `routed_moe` / `gate_up+down`. These profilers add sync, so compare deltas/share, not absolute latency. Instruments is still needed for the actual L2/DRAM hit-rate claim.

## Realistic Prize Range At K4

K4 redundant byte ceiling is real: `6.804 - 4.862 = 1.94 GiB`, `28.5%`.

But max byte prize is only `~9.7 ms`. Every `1 ms/token` of dense/KV/shared marginal reduces that by `~1.14 ms`. To clear +20% over plain needs about `8.7 ms` saved, leaving only `~0.85 ms/token` budget for all dense/KV confound plus fused-kernel overhead.

Realistic range:
- **Low:** `5-6 ms`
- **Central skeptical:** `7-8 ms`
- **High/optimistic:** `9.5-10 ms`

It clears +20% only at the optimistic end, not robustly.

## Recommendation

**HOLD.** Do not build the fused prototype yet.

One test that flips me to GO: K4/K5 stage-profile rerun shows routed MoE stages alone account for most of the K4→K5 delta, e.g. `≥8 ms` of the `+8.91 ms`, and K4 routed stage projects `≥9 ms` net de-dup saving after overhead. If routed delta is `≤6 ms`, this is a NO-GO for the fused-kernel prize as stated.