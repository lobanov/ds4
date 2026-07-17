# Lead 08 iteration 6: production address-kernel SIMDgroup geometry

Date: 2026-07-17. Verdict: **NO-GO; geometry tuning is exact but far below the cost gate.**

## Question

Could the K=4 verifier recover the routed gate+up gap by changing only the production physical-row
kernel's SIMDgroups per threadgroup? This is distinct from grouped de-dup and address remapping: the
same `(token, expert)` rows, weight addresses, reduction order within each SIMDgroup, and output
arithmetic are retained.

## Implementation and protocol

The research-only `DS4_LEAD08_ADDR_NSG` override selects 1, 2, 4, or 8 SIMDgroups for
`kernel_mul_mv_addr_iq2_xxs_pair_swiglu_f32`; the default remains 2. The IQ2 lookup-table preload was
generalized for research specializations, while the literal original preload remains compiled for
the default NSG 2 path. `DS4_LEAD08_ADDR_NSG_PROBE=1` runs a balanced sweep in the retained K=4
production harness:

- 24 physical rows, 18 unique experts, and fixed duplicate multiplicities;
- cold resident-expert cache before every case;
- all 43 routed layers, four ascending/descending rounds;
- direct layer-0 bit comparison against NSG 2 before any candidate is timed.

The unprofiled run measures total routed-MoE wall time. A matched run adds
`DS4_METAL_MOE_STAGE_PROFILE=1` and `DS4_METAL_MOE_STAGE_PROFILE_FILTER=gate_up`. The profiler
synchronizes between stages, so only matched ratios are decision evidence. Both runs use
`DS4_M2_FIDELITY_TEST=1` and the retained Lead 08 model/configuration; the harness's legacy nonzero
exit after completion is expected.

## Fidelity

| NSG | gate/up/mid differing F32 bits | routed-output differing F32 bits | argmax flips | eligibility |
|---:|---:|---:|---:|---|
| 1 | 0 | 0 | 0 | pass |
| 2 | 0 | 0 | 0 | reference |
| 4 | 0 | 0 | 0 | pass |
| 8 | 0 | 0 | 0 | pass |

All three alternate geometries are bit-exact. The harness initializes the complete lookup tables for
each research specialization and enforces the bitwise gate before timing candidates.

## Performance

| NSG | total ms/layer | delta vs 2 | gate+up ms/layer | delta vs 2 | gate+up excluding layer 0 | delta vs 2 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 11.076 | -2.0% | 5.094 | -0.2% | 3.361 | +0.3% |
| **2** | **11.300** | baseline | **5.102** | baseline | **3.351** | baseline |
| 4 | **10.943** | **-3.2%** | 5.160 | +1.1% | **3.313** | **-1.1%** |
| 8 | 11.099 | -1.8% | **4.998** | **-2.0%** | 3.327 | -0.7% |

Each total/full-stage mean contains 172 layer samples; the layer-0-excluded stage view has 168.
NSG 4 is the best unprofiled result (-3.2%), but its isolated gate+up result is +1.1% slower overall
and only -1.1% faster excluding layer 0. NSG 8's best isolated result is -2.0%, shrinking to -0.7%
without layer 0. These small, inconsistent effects are run-noise/low-single-digit scale and do not
approach the predeclared >=15% stage improvement.

## Decision

**Stop the SIMDgroup-count branch.** Do not change the production NSG 2 default for a noisy 1-3%
microbenchmark effect. Threadgroup repacking alone cannot supply the roughly 8.7-10 ms verifier
saving required by the project gate.

The env-gated override and sweep remain only for reproduction. Default behavior and geometry are
unchanged when the variables are absent. Compact retained values are in
`14_iter6_addr_nsg_geometry.csv`; raw `/tmp` stdout/stderr are not retained.
