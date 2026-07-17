# Lead 08 iteration 5: controlled expert-address locality probe

Date: 2026-07-17. Verdict: **NO-GO for address reordering/contiguous packing as a verifier lever.**

## Question

Could the remaining routed-MoE cost be recovered by changing expert-address locality, independently
of the iteration-3 grouped/de-dup design? This is the controlled bandwidth microbenchmark authorized
after hardware-counter discovery found only `GPUTimestamp` on this host.

## Protocol

The env-gated `DS4_LEAD08_ADDRESS_LOCALITY_PROBE=1` mode exercises the production K=4
`routed_moe_batch` address path. Every case holds fixed:

- 24 physical token-expert rows, 18 unique experts, and the same duplicate multiplicities;
- four tokens and six routed rows per token;
- all 43 routed layers, four rounds, balanced ascending/descending case order;
- cold resident-expert cache at the start of every case.

Only physical expert address placement/order changes:

- `contiguous`: logical experts map to one contiguous 18-expert span;
- `permuted`: the exact same 18-expert set maps through `(logical * 5) % 18`, changing order only;
- `strided`: the 18 logical experts map across the slab with a coprime stride of 17.

The production weights and arithmetic are unchanged. The unprofiled run measured total routed-MoE
wall time. A matched second run used `DS4_METAL_MOE_STAGE_PROFILE=1` and
`DS4_METAL_MOE_STAGE_PROFILE_FILTER=gate_up`. Both used the retained lead-08 configuration/model and
`DS4_M2_FIDELITY_TEST=1`; the harness's legacy nonzero exit after completion is expected.

## Results

| pattern | total ms/layer | delta | gate+up ms/layer | delta | gate+up excluding layer 0 | delta |
|---|---:|---:|---:|---:|---:|---:|
| contiguous | 9.766 | baseline | 4.381 | baseline | 2.608 | baseline |
| permuted, same set | 9.953 | +1.9% | 4.450 | +1.6% | 2.653 | +1.8% |
| strided | 9.835 | +0.7% | 4.422 | +0.9% | 2.645 | +1.4% |

Each total and full stage mean contains 172 layer samples. The layer-0-excluded stage mean contains
168 samples and removes the four first-use layer-0 observations. The two views agree: changing from
contiguous placement to same-set permutation or slab-wide striding moves cost by less than 2%, with
contiguous modestly fastest.

## Decision

**Stop the address-layout branch.** The best possible result among these layouts is the contiguous
baseline, and the adverse layouts cost only 0.7-1.9%. Reordering or packing expert addresses therefore
cannot provide the required >=15% stage gain or the roughly 8.7-10 ms verifier saving needed for the
project gate. Do not build an expert-remapping/packing optimization from this hypothesis.

This does **not** prove that routed MoE is compute-bound or replace DRAM/cache/occupancy counters.
It rules out sensitivity to software-visible row order and broad address locality at the fixed
production 18-unique/24-pair shape. Full Xcode counters remain useful only if a genuinely different
mechanism is proposed; they are not authorization to reopen grouped de-dup, margin fallback, or
address reordering.

Raw stdout/stderr were temporary `/tmp` run products and are not retained. Compact values are in
`13_iter5_address_locality.csv`.
