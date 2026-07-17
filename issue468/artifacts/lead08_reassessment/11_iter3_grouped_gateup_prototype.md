# Lead 08 iteration 3 - grouped gate+up prototype

Date: 2026-07-17. Status: **NO-GO; grouped gate+up branch stopped.** Data:
`11_iter3_grouped_gateup_prototype.csv`.

## Prototype

The bounded prototype is enabled only by `DS4_LEAD08_GROUPED_GATEUP_PROBE=1`. It changes the
production IQ2XXS batch address-table gate+up stage into two research-only dispatches:

- singleton experts retain the production physical-row reduction and occupancy;
- cache preparation emits a compact unique-expert/physical-row map without another GPU readback;
- the dispatch grid contains exactly one logical group per unique expert;
- each quantized gate/up tile is loaded once and applied to every matching token;
- one activation vector is live at a time; activation tiles and per-token partial sums spill to
  about 26 KiB of threadgroup memory;
- the existing selection-ordered Q2_K down/sum6 stage is unchanged.

The default path is unchanged when the environment variable is absent. The retained K=4 harness
adds `DS4_LEAD08_GROUPED_GATEUP_FIDELITY=1` for a direct reference comparison.

## Fidelity

The harness ran production and prototype sequentially on layer 0 for all controlled overlap
shapes (6/12/18/24 unique experts, 24 physical pairs). It compared every F32 element of gate, up,
weighted SwiGLU mid, and routed output.

| tensor | max absolute difference | differing F32 bits |
|---|---:|---:|
| gate | 0 | 0 |
| up | 0 | 0 |
| weighted mid | 0 | 0 |
| routed output | 0 | 0 |

Per-token routed-output argmax flips: **0**. The prototype is bit-exact for the isolated routed
stage across singleton, doubleton, and four-token-overlap cases.

## Performance

Matched runs used the artifact-10 protocol: K=4, 24 physical pairs, 43 layers, four balanced-order
rounds, resident-cache reset before every all-layer sample, and
`DS4_METAL_MOE_STAGE_PROFILE_FILTER=gate_up`. The profiler synchronizes between stages, so only
matched ratios are decision evidence.

| unique / 24 pairs | production ms/layer | prototype ms/layer | prototype delta |
|---:|---:|---:|---:|
| 6 | 3.139 | 5.276 | +68.1% |
| 12 | 3.350 | 5.211 | +55.6% |
| **18** | **4.869** | **5.986** | **+22.9%** |
| 24 | 4.993 | 4.947 | -0.9% |

The predeclared gate required at least a 15% improvement at 18 unique / 24 pairs. Instead the
prototype regressed 22.9%. Removing every layer-0 sample, which is sensitive to first-use effects,
strengthens the target-shape regression to 35.8% (3.200 -> 4.346 ms/layer).

The disjoint case is effectively parity (-0.9%; +0.4% with layer 0 removed), proving that the
compact expert-major map removed the provisional physical-grid/filter overhead. Performance gets
worse as more work moves into the grouped kernel: the 6- and 12-unique shapes regress 68.1% and
55.6%. This isolates the failure to threadgroup activation/partial-sum spill and barriers, which
cost more than the saved duplicate dequant/load work on this M5 Max kernel shape.

## Decision

**NO-GO. Stop the grouped gate+up branch.** The prototype passes fidelity but fails the cost gate
with the wrong sign. Per the bounded plan, do not integrate it into the verifier and do not iterate
on threadgroup-spill variants. Retain the env-gated implementation only to reproduce the negative
result; it is not a production optimization.

The final Lead 08 integration gate (`verify_ms(4) <= 50.5 ms`, exact greedy output) was not run
because the prerequisite stage gate failed. The controlled evidence withdraws the remaining
load-sharing prize for this design; any future verifier work needs a different mechanism rather
than another grouped IQ2XXS gate+up variant.

## Counter availability

A post-decision audit checked whether the separate coalescing/bandwidth hypothesis could be tested
without another kernel. This host has no `xctrace` or offline `metal` utility in its active developer
toolchain. `DS4_METAL_COUNTER_INVENTORY=1` reports only the Metal `timestamp/GPUTimestamp` counter
set; no DRAM, cache, occupancy, or instruction counters are exposed through the current API.
Artifact 13 subsequently ran the distinct fixed-work production timing probe: contiguous, same-set
permuted, and slab-wide strided placement differ by <2%, closing simple software address remapping.
Hardware bandwidth-vs-compute attribution remains unmeasured and would require full Xcode tooling.
