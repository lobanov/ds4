# Lead 08 iteration 9: batched-verifier locality profiler correction

Date: 2026-07-17. Verdict: **iteration-8 locality values superseded; cache-carrier NO-GO unchanged.**

## Finding

The iteration-8 extension read `batch_router_selected` and `batch_router_weights` on the CPU inside
the all-layer verifier encode loop. The producing Metal command buffer was still uncommitted, and
the same two buffers are reused by every layer. Those reads therefore observed previously completed
contents rather than the layer being encoded.

The stale signature is decisive on a repeated 64-token `code_topk` run: 43 router layers produced
only 11 distinct top-16 expert/count vectors and three distinct adjacent-overlap values. Most layers
shared the same top expert and count despite layer-specific router weights. This invalidates only the
iteration-8 LRU/overlap numbers, not its independent cold/replay timing or fidelity results.

## Correction

When `DS4_EXPERT_PROFILE`, `DS4_MTP_VERIFY_PROFILE`, or
`DS4_MTP_VERIFY_EXPERT_PROFILE` is active, the graph now allocates small per-layer selected-ID and
route-weight capture tensors. Immediately after each router dispatch it encodes ordered GPU copies
into the layer's capture slice. The CPU reads and records all slices only after the normal verifier
command buffer completes. No capture tensors or copies exist in default inference.

The same prompt now produces 43 distinct layer vectors and 38 distinct overlap values. Its complete
cycle/draft/verify/accept trajectory is unchanged; both versions hash to
`6fa3d319ad89ef8afddc7f89c65145fdbd0214c7326a92aaa438c2a5b34f3481` after timing fields are removed.

## Corrected locality

The retained result expands to three prompt families (`codealpaca_0060`, `dolly_0090`, and
`jsonex_0000`), 64 emitted tokens each. It records 12,212 layer records / 73,272 selections and has
mean adjacent top-6 overlap 0.373.

| capacity | hit rate | weighted hit rate |
|---:|---:|---:|
| 8 | 38.6% | 44.4% |
| 16 | 57.3% | 61.8% |
| 32 | 72.3% | 75.6% |
| 64 | 81.5% | 84.0% |
| 128 | 87.9% | 89.8% |

Profiling and profiling-disabled controls have identical full cycle trajectories across all three
prompts. Aggregate throughput is 35.40 versus 35.66 t/s (-0.7%), bounding the diagnostic-copy cost.

## Decision

The corrected trace still establishes temporal expert locality, but it does not reopen caching for
the current verifier. Iteration 8's compatible mapped path remains cold/replay parity at
0.566/0.566 ms per layer; only the DSpark-incompatible SSD selected-address path has a residency
miss to remove. Retain the corrected profiler for future SSD/runtime work and do not treat its hit
curve as a Lead 08 speedup mechanism.

Compact stale-versus-corrected evidence is in `17_iter9_profiler_correction.csv`. The canonical
current curve replaces `16_iter8_actual_expert_lru.csv`; raw `/tmp` profiles and Xcode traces are not
retained.
