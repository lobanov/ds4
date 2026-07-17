# Lead 08 iteration 7: production address-kernel row tile

Date: 2026-07-17. Verdict: **NO-GO; alternate tiles have no repeatable material gain.**

## Question

Could the K=4 verifier recover the routed gate+up gap by changing the number of output rows each
SIMDgroup accumulates? This tests the register-pressure versus activation/weight-reuse tradeoff left
after the iteration-6 SIMDgroup-count sweep. It retains NSG 2, the same 24 physical `(token, expert)`
rows, weight addresses, per-row reduction order, and output arithmetic.

## Implementation and protocol

The research-only `DS4_LEAD08_ADDR_NR0` override selects row tile 2, 4, or 8 for
`kernel_mul_mv_addr_iq2_xxs_pair_swiglu_f32`; the default and production entry point remain tile 4.
Tiles 2 and 8 use separate Metal specializations so loop bounds and host grid coverage agree.
`DS4_LEAD08_ADDR_NR0_PROBE=1` reuses the controlled production K=4 geometry harness:

- 24 physical rows, 18 unique experts, and fixed duplicate multiplicities;
- NSG 2 and a cold resident-expert cache before every case;
- all 43 routed layers, two repetitions of four ascending/descending total-path rounds;
- direct layer-0 bit comparison against tile 4 before any candidate is timed.

The unprofiled runs measure total routed-MoE wall time. A matched run adds
`DS4_METAL_MOE_STAGE_PROFILE=1` and `DS4_METAL_MOE_STAGE_PROFILE_FILTER=gate_up`. The profiler
synchronizes between stages, so only matched ratios are compared. Both runs use
`DS4_M2_FIDELITY_TEST=1` and the retained Lead 08 model/configuration; the legacy harness exits
nonzero after completing because its older fast-math comparison is not bit-exact.

## Fidelity

| row tile | gate/up/mid differing F32 bits | routed-output differing F32 bits | argmax flips | eligibility |
|---:|---:|---:|---:|---|
| 2 | 0 | 0 | 0 | pass |
| 4 | 0 | 0 | 0 | reference |
| 8 | 0 | 0 | 0 | pass |

Both alternate tiles are bit-exact through the production gate, up, weighted-mid, routed-output,
and argmax checks.

## Performance

| row tile | total ms/layer | delta vs 4 | gate+up ms/layer | delta vs 4 | gate+up excluding layer 0 | delta vs 4 |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 11.064 | +1.0% | 4.998 | -1.3% | 3.292 | +1.3% |
| **4** | **10.954** | baseline | 5.063 | baseline | **3.249** | baseline |
| 8 | 10.982 | +0.3% | **4.938** | **-2.5%** | 3.251 | +0.1% |

Each total mean contains 344 layer samples across eight rounds. The matched full-stage mean contains
172 samples; the layer-0-excluded view has 168. The two total repetitions changed tile 8's sign
relative to tile 4 (+1.35% then -0.81%), confirming noise-scale sensitivity; their combined result
is +0.25%. Tile 8's apparent -2.5% profiled gain is also a layer-0/startup effect: it becomes +0.05%
excluding layer 0. Tile 2 is 1.01% slower total and 1.33% slower in steady-state gate+up. No
alternate approaches the predeclared >=15% stage gate.

## Decision

**Stop row-tile tuning and retain production tile 4.** The two exact alternatives are parity or
slower under steady-state and total-path views. Together with iteration 6, this closes the cheap
address-kernel launch/tiling geometry branch; neither SIMDgroup count nor rows per SIMDgroup is a
material verifier lever.

The env-gated specializations and sweep remain only for reproduction. Default execution is unchanged
when the variables are absent. Compact retained values are in `15_iter7_addr_row_tile.csv`; raw
`/tmp` stdout/stderr are not retained.
