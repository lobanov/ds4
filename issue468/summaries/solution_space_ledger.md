# Issue 468 solution-space ledger

Date: 2026-07-17. Purpose: make experiment selection explicit, comparable, and auditable. This is
the live decision ledger; `issue468/STATUS.md` remains the canonical narrative.

## Goal contract and common currency

Primary success requires all of the following on one realistic local machine/backend:

- at least 20% greedy throughput over target-only ds4;
- exact greedy output preservation against that target-only reference;
- no pathological memory/replay cost;
- a concrete runtime composition, not an isolated kernel projection.

Current common currency on the retained M5 Max / Metal / IQ2XXS setup:

| Quantity | Current value | Use |
|---|---:|---|
| target-only reference | 38.16 t/s on the 176-entry M3 corpus | primary denominator |
| best composed DSpark stack | 40.04 t/s, +4.9% | current nonexact best |
| remaining raw cycle prize | about 8.7 ms/cycle | minimum M3 saving for +20% |
| modeled K=4 verify | 65.8 ms | pre-M3 common verifier reference |
| K=4 +20% integration gate | <=50.5 ms | hard Lead 08 economic gate |
| estimated K=4 byte floor | about 39-43 ms | optimistic feasibility bound |

Numbers from different harnesses are not interchangeable. Every experiment must name its reference,
corpus, exactness contract, and whether timing is decision-grade or diagnostic-only.

## Status vocabulary

| Status | Meaning |
|---|---|
| `OPEN` | Unfalsified mechanism with a concrete next experiment |
| `P0` | Immediate dependency; no downstream experiment is interpretable until resolved |
| `DEFERRED` | Valid question dominated by a higher-value dependency |
| `MARGINAL` | Real effect, insufficient alone, retained only for composition |
| `BLOCKED` | Opportunity exists but cannot compose with the current runtime |
| `CLOSED` | Decisive negative or completed positive; do not repeat without new mechanism |
| `INTERIM` | Useful research result that fails the goal contract |

## Mechanism ledger

| ID | Family / mechanism | Maximum plausible contribution | Exactness | Compatibility | Cheapest decisive falsifier | Current evidence | Status / priority |
|---|---|---|---|---|---|---|---|
| D1 | Drafter weight precision | Small after Q4_K parity | Neutral | Composes | F16/F32/Q4 acceptance parity | Q4_K ~= F16 ~= F32; +0.38% net prefix from expert dequant | `CLOSED` |
| D2 | Non-expert head LoRA | Acceptance only | Neutral | Composes | Held-out p1 by rank | Hurts p1 by 1.5-3.1 pp | `CLOSED` |
| D3 | Target-hidden precision / crossed oracle | Previously estimated +8-10% ceiling | Neutral | FP labels undeployable on IQ2 | Common-trajectory crossed oracle | Recoverable IQ2 hidden-side effect is negative | `CLOSED` |
| D4 | DFlash drafter | Acceptance only | Neutral | Composes | Same-corpus accepted prefix | About 2.5x worse prefix than DSpark | `CLOSED` |
| D5 | Soft-label full-body redistillation | Unknown acceptance lift | Neutral | Training work; does not fix verifier exactness | Powered held-out +2 pp p1 test | Untested; negative prior after D1-D4 | `DEFERRED` behind verifier viability |
| A1 | Cycle-jump acceptance measurement | Corrects model optimism | Neutral | Complete | Powered trajectory simulation | E[a\|4]=2.198, S(4)=0.340 | `CLOSED` positive measurement |
| A2 | Anchor reuse | Removes redundant target step | Exact in retained path | Composes | Reuse falsifier + runtime integration | Survives; implemented | `CLOSED` positive |
| A3 | Confidence scheduling / STS | Low-single-digit conditional gain | Depends on verifier | Composes | Frozen-threshold held-out run | Marginal under reuse; cannot close gap | `MARGINAL` retained |
| R1 | GPU Metal drafter | Draft 45 -> about 7.6 ms | Does not determine target exactness | Composes | Runtime timing + draft parity | Implemented in M3 | `CLOSED` positive |
| R2 | Prefix checkpoint / fast commit | Avoids sequential replay | Inherits batch verifier numerics | Composes | Runtime timing and trajectory | Implemented in M3 | `CLOSED` positive |
| V1 | Existing committing batch verifier | First local win, +4.9% stack | Fails strict output contract | Composes | Exactness corpus | Score-neutral but nonexact | `INTERIM`, not goal success |
| V2 | Exact sequential verifier | Correct reference | Exact | Composes | End-to-end throughput | About 0.85x baseline | `CLOSED` economic NO-GO |
| V3 | Margin-guarded exact fallback | Could repair rare flips | No universal observed bound | Composes | Margin coverage + replay cost | Safe observed threshold guards 17.1%, adds >=7 ms/cycle | `CLOSED` |
| V4 | Shared batch M=1/M=K operation family | Exactness by shared family if M-invariant | Fails | Composes | Matched target rebaseline + corpus | 5/10 exact; M1 -13.0%; stack -1.3% vs shipped | `CLOSED` |
| V5 | Exact hybrid: row-wise exact state transitions plus invariant batch sharing | Only remaining measured Lead 08 route to floor | Unproven | Intended to compose | Stage-by-stage first-divergence movement plus cumulative cost bound | Original design never built | `OPEN`, highest-value branch |
| V6 | Row-wise batch-M1 Q/KV projection at layer 0 | Resolves first observed V5 correctness debt | Exact through Q/KV norm and KV path for captured row | Diagnostic patch | Exactify Q/KV; repeat position-104 capture; measure delta | Frontier moves to Qcur; noisy naive all-layer delta +0.56 ms mean with downstream-work confound | `CLOSED` positive |
| V7 | Row-wise production Q-b at layer 0 | Resolves next observed V5 correctness debt | Exact through inverse RoPE for captured row | Diagnostic patch | Row-wise Q-b; repeat capture and cost discipline | Frontier moves through `kqv_back`; naive all-layer whole-graph delta +7.47 ms | `CLOSED` positive |
| V8 | Attention-output low/final projection at layer 0 | Next observed V5 correctness debt | Unproven | Diagnostic patch | Branch-neutral production capture, then row-wise first differing projection | V7 is exact through inverse RoPE; final logits still flip | `OPEN`, rank 1 |
| K1 | Grouped routed gate/up | Proposed expert-load sharing | Bit-exact on identical inputs | Composes | Production 18/24 prototype | 22.9% slower | `CLOSED` |
| K2 | Expert address remapping / packing | Proposed coalescing | Exact | Composes | Fixed-work placement sweep | Less than 2% sensitivity | `CLOSED` |
| K3 | Address-kernel SIMDgroup / row-tile geometry | Low-single-digit possible | Exact | Composes | Production fixed-work sweep | At most about 1-3%, inconsistent | `CLOSED` |
| M1 | Cross-cycle expert residency on SSD selected-address path | Large isolated prize | Exact replay | DSpark rejects SSD streaming | Cold/resident replay plus compatibility check | -87.8% isolated, incompatible | `BLOCKED`, separate runtime lead |
| M2 | Cross-cycle residency on mapped current path | Small | Exact | Composes | Cold/resident replay | 0.566 -> 0.566 ms/layer | `CLOSED` |
| T1 | Standalone GPU timestamp decomposition | Attribution only | Neutral | Available | Matched timestamp/no-timestamp control | Only GPUTimestamp exposed; no independent decision mechanism | `DEFERRED`, supporting evidence only |

Canonical evidence index (summary names are under `issue468/summaries/`; reassessment artifacts are
under `issue468/artifacts/lead08_reassessment/`):

- D1-D5: `dspark_quantization_ceiling.md`, `stage2_finetune_result.md`,
  `stage2_crossed_oracle.md`, `dflash_oracle_investigation.md`, and
  `issue468/pending/lead_10_drafter_redistillation.md`.
- A1-A3: `spec_speedup_model.md`, `anchor_reuse_falsifier.md`, and
  `confidence_scheduled_verification.md`.
- R1-R2 and V1-V2: `dspark_runtime_milestone_3_progress.md` and `spec_speedup_model.md`.
- V3-V8: Lead 08 reassessment artifacts `12_iter4_margin_guard.md`,
  `18_iter10_batch_m1_target.md`, `lead08_phaseB_floor_clearance_verdict.md`, and
  `19_iter11_same_frontier_localization.md`, `20_iter12_rowwise_qkv.md`, and
  `21_iter13_rowwise_qb.md`.
- K1-K3, M1-M2: Lead 08 reassessment artifacts `11_iter3_grouped_gateup_prototype.md`,
  `13_iter5_address_locality.md`, `14_iter6_addr_nsg_geometry.md`,
  `15_iter7_addr_row_tile.md`, and `16_iter8_cache_residency.md`.
- T1: `07_leadid_candidate_synthesis.md` and the current Lead 08 worklog. Timestamp evidence is
  supporting-only until attached to a mechanism experiment.

## Lead 08 correctness frontier

The exact-hybrid search advances only at the first unresolved operation boundary. A downstream
difference inherited from an earlier stage is not evidence that the downstream stage is independently
M-dependent.

| Pipeline boundary | Same-frontier evidence | Independent status | Required next action |
|---|---|---|---|
| token / HC input | Row-0 input implied common; not separately retained | not fully isolated | retain only if Q/KV reroute exposes ambiguity |
| layer-0 `hc_attn_pre` | bit-identical row 0 in post-fix iteration 11 dump | exact for captured row | none |
| layer-0 `attn_norm` | bit-identical row 0 | exact for captured row | none |
| layer-0 Q-a / KV projection | bit-identical after V6 row-wise M1 projection | exact for captured row | none |
| Q/KV normalization | bit-identical after V6 | exact for captured row | none |
| Q-b + head norm + Q RoPE | bit-identical after V7 row-wise Q-b | exact for captured row | none |
| KV RoPE / store / common raw cache | bit-identical after V6 | exact for captured row | none |
| compressor / indexer | not exercised in the layer-0 raw-attention case | unknown | defer to first compressed-layer boundary |
| attention reduction + inverse RoPE | bit-identical through `kqv_back` after V7 | exact for captured row | none |
| attention-output low/final projection | final logits still diverge; production internals not retained | first unresolved composite boundary | V8 branch-neutral capture, then first projection only |
| HC post | divergent downstream | unknown independently | test next first boundary only |
| routed MoE gate/up | bit-exact in identical-input isolated prototype | exact in that harness; economics closed | retain production path; do not rebuild grouped K1 |
| routed down / sum and shared FFN | not isolated end-to-end | unknown | test only after attention frontier moves |
| output head | same batched function in V4 but inherited state differs | unknown independently | last boundary after layer path |

## Experiment contract

No new experiment starts without these fields in its worklog or artifact:

1. Hypothesis ID and one mechanism being changed.
2. Reference and candidate compositions.
3. Exactness invariant and how failure is detected.
4. Economic metric and predeclared threshold.
5. Cheapest falsifier and expected effort.
6. Outcome tree for pass, fail, and ambiguous results.
7. Required retained evidence and reproduction command.
8. Pre-experiment challenge of assumptions and post-experiment red-team audit.

For exact-hybrid stages, correctness and performance are coupled:

`candidate lower bound = proven-unavoidable exactified work + optimistic remaining batch floor + runtime overhead`

The branch stops as soon as that lower bound cannot meet `verify_ms(4) <= 50.5 ms`. Moving the
correctness frontier without measuring incremental cost is diagnostic progress, not authorization for
a full build.

## Ranked queue and outcome rules

| Rank | Experiment | Why now | Pass | Fail / stop |
|---:|---|---|---|---|
| P0 | Enforce `spec_frontier_restore` success in the same-frontier probe and rerun iteration 11 | Audit found the core invariant was unchecked | **Resolved:** hard failure plus 267/267 successful restores; corpus and dumps reproduce | Repeat audit passed; proceed to V6 |
| 1 | V8 branch-neutral attention-output capture, then row-wise first differing projection | V7 is exact through inverse RoPE | First divergence moves; incremental cost leaves credible <=50.5 ms path | Divergence remains, or proven-unavoidable lower bound exceeds gate |
| 2 | Repeat first-divergence isolation at the newly exposed boundary | Delta-debug the pipeline, not guess stages | Frontier moves within budget | Stop V5 only when proven-unavoidable work exhausts budget |
| 3 | Full-corpus exactness plus end-to-end K=4 timing | Only after all boundaries pass | Exact stream and <=50.5 ms verify, then >=20% composed run | Close V5 |
| 4 | D5 soft-label redistillation | Only if an exact economic verifier survives and acceptance remains limiting | Powered p1 lift composes into >=20% model | Close drafter axis |

P0, V6, and V7 are complete and their repeat audits passed. No
GPU-timestamp-only, grouped-MoE, layout, geometry, mapped-residency, or margin-policy experiment may
preempt V8 without new evidence that changes the ledger's upper bounds.
