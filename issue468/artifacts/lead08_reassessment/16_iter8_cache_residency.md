# Lead 08 iteration 8: cache-residency upper bound and carrier test

Date: 2026-07-17. Verdict: **NO-GO for the current DSpark verifier; the large replay prize exists
only on an incompatible SSD selected-address path.**

## Question

Iteration 2 measured 10-11 ms/layer for the cold SSD selected-address routed path at the realistic
K=4, 18-unique/24-pair shape, while matched gate+up plus down accounted for only about half. Could
cross-cycle expert residency remove the remainder and supply a genuinely different Lead 08 lever?

## Protocol

Three controlled measurements separate the upper bound from the deployable path:

1. `DS4_METAL_STREAMING_PREFILL_BATCH_SELECTED_ADDR_PROFILE=1` decomposes cold selected-address
   preparation into selected-ID readback and cache wrap/load time.
2. `DS4_LEAD08_CACHE_REPLAY_PROBE=1` runs the same 18-unique selection sequence cold and then
   immediately replays it without clearing residency. It runs four all-43-layer rounds and directly
   compares cold versus replay gate, up, weighted mid, routed output, and argmax before timing.
3. The same replay probe runs without SSD streaming, matching the mapped-weight batch path that is
   compatible with the current DSpark full stack. The routed-expert profiler was also extended to
   ingest actual batched-verifier selections; one 64-token full-stack prompt produced 22,188 expert
   accesses across 3,698 layer records for a per-layer LRU simulation.

The exact SSD experiment uses `--ssd-streaming --ssd-streaming-cold`. Attempting to compose those
flags with `--dspark` fails explicitly: `--ssd-streaming is not compatible with --dspark yet`.

## Fidelity

Both SSD and mapped-path cold/replay comparisons have zero differing F32 bits through gate, up,
weighted mid, and routed output, with zero argmax flips.

## Replay result

| path | cold ms/layer | replay ms/layer | replay delta | cold wrap/load | replay wrap/load |
|---|---:|---:|---:|---:|---:|
| SSD selected address | 11.094 | 1.353 | **-87.8%** | 5.407 ms | 0.001 ms |
| current non-SSD mapped path | 0.566 | 0.566 | **-0.04%** | n/a | n/a |

Selected-ID readback on the SSD path is only 0.00027 ms/layer. The cold term is resource loading
and first-use residency, not selection synchronization. Exact replay proves a large SSD upper bound,
but the mapped path already has no measurable cold/replay gap.

## Actual verifier locality

The full-stack prompt emitted 64 tokens in 33 cycles with mean verify_n 2.61. Every routed layer
recorded 86 speculative rows. Mean adjacent top-6 overlap is 0.370 (about 2.22 experts) and the
per-layer LRU hit curve is:

| capacity | hit rate |
|---:|---:|
| 8 | 37.2% |
| 16 | 60.5% |
| 32 | 76.0% |
| 64 | 81.4% |
| 128+ | 83.5% |

This locality would be useful if an SSD-compatible speculative runtime existed. It does not create
a current-path saving: the mapped-path replay result is already parity, and the full-stack verifier
cannot enter the selected-address cache path.

## Decision

**Do not port or tune selected-address expert caching under Lead 08.** The 87.8% result is an exact
upper bound for a different runtime regime, not evidence of a deployable verifier speedup. Building
DSpark+SSD compatibility is a separate Lead 05/runtime project and must be justified against its own
baseline. For the current Lead 08 target, residency misses the material gate by construction: cold
and replay differ by 0.04%.

Retain `DS4_LEAD08_CACHE_REPLAY_PROBE` and the batched-verifier expert-profile extension for
reproduction/diagnosis. Compact values are in `16_iter8_cache_replay.csv` and
`16_iter8_actual_expert_lru.csv`; raw `/tmp` logs and profiles are not retained.
