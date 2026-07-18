# Issue 468 solution-space ledger

Date: 2026-07-18. Purpose: make experiment selection explicit, comparable, and auditable. This is
the live decision ledger; `issue468/STATUS.md` remains the canonical narrative.

## Goal contract and common currency

Primary success requires all of the following on one realistic local machine/backend:

- at least 20% greedy throughput over target-only ds4;
- exact greedy output preservation against that target-only reference;
- no pathological memory/replay cost;
- a concrete runtime composition, not an isolated kernel projection.

Lead 08 also has an explicit `INTERIM_BOUNDED` contract as of 2026-07-18. It may pursue the same
tradeoff admitted for Milestone 3: task-quality non-regression at temperature 0 and close
distributional agreement with exact sequential verification above temperature 0. This is an
engineering/research admission gate only; it does not satisfy the exact-output primary goal.

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
corpus, quality contract, and whether timing is decision-grade or diagnostic-only. Under
`INTERIM_BOUNDED`, a fresh same-binary plain control and M3 optimization-off control are mandatory;
historical M3 values are outer caps rather than substitutes for matched controls.

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
| `INTERIM_BASELINE` | Frozen quality/performance envelope for bounded-track comparisons; not primary success |

## Mechanism ledger

| ID | Family / mechanism | Maximum plausible contribution | Exactness | Compatibility | Cheapest decisive falsifier | Current evidence | Status / priority |
|---|---|---|---|---|---|---|---|
| D1 | Drafter weight precision | Small after Q4_K parity | Neutral | Composes | F16/F32/Q4 acceptance parity | Q4_K ~= F16 ~= F32; +0.38% net prefix from expert dequant | `CLOSED` |
| D2 | Non-expert head LoRA | Acceptance only | Neutral | Composes | Held-out p1 by rank | Hurts p1 by 1.5-3.1 pp | `CLOSED` |
| D3 | Target-hidden precision / crossed oracle | Previously estimated +8-10% ceiling | Neutral | FP labels undeployable on IQ2 | Common-trajectory crossed oracle | Recoverable IQ2 hidden-side effect is negative | `CLOSED` |
| D4 | DFlash drafter | Acceptance only | Neutral | Composes | Same-corpus accepted prefix | About 2.5x worse prefix than DSpark | `CLOSED` |
| D5 | Soft-label full-body redistillation | Unknown acceptance lift | Neutral | Training work; does not fix verifier cost | Powered held-out +2 pp p1 test | Untested; negative prior after D1-D4 | `DEFERRED`; conditional later composition |
| A1 | Cycle-jump acceptance measurement | Corrects model optimism | Neutral | Complete | Powered trajectory simulation | E[a\|4]=2.198, S(4)=0.340 | `CLOSED` positive measurement |
| A2 | Anchor reuse | Removes redundant target step | Exact in retained path | Composes | Reuse falsifier + runtime integration | Survives; implemented | `CLOSED` positive |
| A3 | Confidence scheduling / STS | Low-single-digit conditional gain | Depends on verifier | Composes | Frozen-threshold held-out run | Marginal under reuse; cannot close gap | `MARGINAL` retained |
| R1 | GPU Metal drafter | Draft 45 -> about 7.6 ms | Does not determine target exactness | Composes | Runtime timing + draft parity | Implemented in M3 | `CLOSED` positive |
| R2 | Prefix checkpoint / fast commit | Avoids sequential replay | Inherits batch verifier numerics | Composes | Runtime timing and trajectory | Implemented in M3 | `CLOSED` positive |
| V1 | Existing committing batch verifier | First local win, +4.9% stack | M3 bounded envelope; fails strict output contract | Composes | Fresh M3 control plus fixed quality gates | 61/92 vs 60/92; 83/92 same verdict; mean TV 0.0104; nonexact | `INTERIM_BASELINE`, not primary success |
| V2 | Exact sequential verifier | Correct reference | Exact | Composes | End-to-end throughput | About 0.85x baseline | `CLOSED` economic NO-GO |
| V3 | Margin-guarded exact fallback | Could repair rare flips | No universal observed bound | Composes | Margin coverage + replay cost | Safe observed threshold guards 17.1%, adds >=7 ms/cycle | `CLOSED` |
| V4 | Shared batch M=1/M=K operation family | Exactness by shared family if M-invariant | Fails | Composes | Matched target rebaseline + corpus | 5/10 exact; M1 -13.0%; stack -1.3% vs shipped | `CLOSED` |
| V5 | Exact hybrid: row-wise exact state transitions plus invariant batch sharing | Possible route back to primary exact contract | Unproven | Intended to compose | Stage-by-stage first-divergence movement plus cumulative cost bound | Original design never built; V6-V12 retain diagnostic value | `DEFERRED` behind bounded track |
| V6 | Row-wise batch-M1 Q/KV projection at layer 0 | Resolves first observed V5 correctness debt | Exact through Q/KV norm and KV path for captured row | Diagnostic patch | Exactify Q/KV; repeat position-104 capture; measure delta | Frontier moves to Qcur; noisy naive all-layer delta +0.56 ms mean with downstream-work confound | `CLOSED` positive |
| V7 | Row-wise production Q-b at layer 0 | Resolves next observed V5 correctness debt | Exact through inverse RoPE for captured row | Diagnostic patch | Row-wise Q-b; repeat capture and cost discipline | Frontier moves through `kqv_back`; naive all-layer whole-graph delta +7.47 ms | `CLOSED` positive |
| V8 | Attention-output low/final projection at layer 0 | Next observed V5 correctness debt | Exact through output-B for captured row | Diagnostic overlay | Capture then row-wise output-B | Output-A low is exact; frontier moves to HC expansion; redundant all-layer delta +6.99 ms | `CLOSED` positive |
| V9 | HC expansion input localization at layer 0 | Resolve whether expansion or latent input differs | Expansion not independently implicated | Diagnostic capture | Capture all expansion inputs | Block/residual exact; HC mix and consumed split state differ | `CLOSED` redesign |
| V10 | Row-wise HC attention mixer at layer 0 | First latent state debt exposed by V9 | Exact through attention HC post for captured row | Diagnostic patch | Row-wise F16 mixer, recapture mix/split/HC post | Frontier moves to FFN HC path; timing shows no obvious penalty but is route-confounded | `CLOSED` positive |
| V11 | FFN HC input localization at layer 0 | First captured difference after V10 | Mixer is first differing operation | Diagnostic capture | Capture FFN residual/flat/mix/split/pre | Residual and flat exact; mix differs 14/24 before split/pre | `CLOSED` redesign |
| V12 | Row-wise FFN HC mixer at layer 0 | Exact-input/different-output boundary exposed by V11 | Exact through FFN norm for captured row | Diagnostic patch | Apply existing F16 rows-as-M1 helper only to `hc_ffn_fn` | Frontier moves to router logits; local profiled envelope -0.029 ms | `CLOSED` positive |
| V13 | Generic same-accumulation batched-F16 design/cost inventory | Exact-track response to three exposed low-K F16 mismatches | Design only | Must preserve batch sharing | Inventory attention/FFN/router shapes, kernel route, shared mechanism, isolated cost carrier | Not run; unnecessary for M3-bounded admission | `DEFERRED` exactness track |
| V14 | M3-quality-bounded verifier acceleration | Must save >=8.7 ms/cycle to close the live gap | No worse than matched M3 task/distribution envelope | Must preserve M3 composition and integrity | Bind a separator to the live mapped kernel, then one >=15% fixed-work prototype | U1 closes arithmetic below 4.74 ms; iteration-25 audit finds no independent composable >=8.7 ms mechanism | `CLOSED`; bounded track exhausted |
| K1 | Grouped routed gate/up | Proposed expert-load sharing | Bit-exact on identical inputs | Composes | Production 18/24 prototype | 22.9% slower | `CLOSED` |
| K2 | Expert address remapping / packing | Proposed coalescing | Exact | Composes | Fixed-work placement sweep | Less than 2% sensitivity | `CLOSED` |
| K3 | Address-kernel SIMDgroup / row-tile geometry | Low-single-digit possible | Exact | Composes | Production fixed-work sweep | At most about 1-3%, inconsistent | `CLOSED` |
| M1 | Cross-cycle expert residency on SSD selected-address path | Large isolated prize | Exact replay | DSpark rejects SSD streaming | Cold/resident replay plus compatibility check | -87.8% isolated, incompatible | `BLOCKED`, separate runtime lead |
| M2 | Cross-cycle residency on mapped current path | Small | Exact | Composes | Cold/resident replay | 0.566 -> 0.566 ms/layer | `CLOSED` |
| T1 | Metal counter attribution tied to V14 | Selector only; no direct speedup | Neutral | CLI profiles lack useful M5 counters | Populated-counter and <=5% perturbation gate | Only zero-valued `RT Unit Active`; no shader intervals; +7.5% timing perturbation | `BLOCKED`; custom GUI template could reopen |
| T2a | SSD selected-address ALU-headroom separator | Attribution only | Neutral | SSD path cannot compose with DSpark | Production plus companion-zero control; compiler-retained dependent FMA sweep at fixed bytes/rows/geometry | Layer-0 pattern bit-exact; control within 2%; monotonic response, but rounds 32: +5.41% total/+10.10% gate-up falls in the predeclared ambiguous band | `CLOSED` ambiguous; incompatible carrier |
| T2b | Live mapped `tiny_pair_mv` ALU-headroom separator | Attribution only; may select arithmetic V14 mechanism | Neutral | Current K=4 `kernel_mul_mv_id_iq2_xxs_pair_f32` family | Literal production plus companion-zero and retained-FMA arms on two 18-unique selection strata; require all-layer dispatch/AIR/bitwise gates | Fidelity/AIR/dispatch pass and response is strong, but three control intervals escape +/-2%; overlap R32 lower CI is 24.958% | `CLOSED` invalid / near-threshold |
| T2c | Control-stabilized mapped pair replay | Attribution only | Neutral | Same K=4 mapped pair carrier | Direct gate/up replay with GPU command-buffer timestamps, production duplicate, companion-zero, R32/R128, two strata | All controls/fidelity/AIR gates pass; R32 lower CI is +45.65%/+45.66% vs zero and R128 is monotonic | `CLOSED` positive selector; one-sided arithmetic/issue sensitivity |
| U1 | Maximum-removable mapped-pair arithmetic upper bound | At most the approximately 10.74 ms/cycle direct carrier | Neutral research kernel | Same mapped carrier; not production semantics | Retain packed loads/addressing/geometry/stores, remove maximum arithmetic with AIR proof, require paired saving lower bound >=8.7 ms in both strata | Corrected address-faithful floor saves 4.708/4.724 ms with upper bounds below 4.74 ms | `CLOSED` economic STOP |

Canonical evidence index (summary names are under `issue468/summaries/`; reassessment artifacts are
under `issue468/artifacts/lead08_reassessment/`):

- D1-D5: `dspark_quantization_ceiling.md`, `stage2_finetune_result.md`,
  `stage2_crossed_oracle.md`, `dflash_oracle_investigation.md`, and
  `issue468/pending/lead_10_drafter_redistillation.md`.
- A1-A3: `spec_speedup_model.md`, `anchor_reuse_falsifier.md`, and
  `confidence_scheduled_verification.md`.
- R1-R2, V1-V2, and V14 contract: `dspark_runtime_milestone_3_progress.md`,
  `spec_speedup_model.md`, and reassessment artifact `27_iter19_bounded_divergence_reframe.md`.
- V3-V12: Lead 08 reassessment artifacts `12_iter4_margin_guard.md`,
  `18_iter10_batch_m1_target.md`, `lead08_phaseB_floor_clearance_verdict.md`, and
  `19_iter11_same_frontier_localization.md`, `20_iter12_rowwise_qkv.md`, and
  `21_iter13_rowwise_qb.md`, `22_iter14_rowwise_attn_out_b.md`, and
  `23_iter15_hc_input_localization.md`, `24_iter16_rowwise_hc_attn_mix.md`, and
  `25_iter17_ffn_hc_input_localization.md`, and `26_iter18_rowwise_hc_ffn_mix.md`.
- K1-K3, M1-M2: Lead 08 reassessment artifacts `11_iter3_grouped_gateup_prototype.md`,
  `13_iter5_address_locality.md`, `14_iter6_addr_nsg_geometry.md`,
  `15_iter7_addr_row_tile.md`, and `16_iter8_cache_residency.md`.
- T1-U1: `07_leadid_candidate_synthesis.md`, reassessment artifacts
  `27_iter19_bounded_divergence_reframe.md` and `28_iter20_metal_counter_capability.md`, and the
  current Lead 08 worklog. `29_iter21_ssd_addr_alu_headroom.md` closes T2a ambiguous; T1 is blocked and
  T2b closes invalid in `30_iter22_mapped_alu_separator.md`; T2c closes positive as a one-sided
  selector in `31_iter23_direct_gpu_replay.md`; U1 is the only authorized follow-up.

## Deferred Lead 08 exactness frontier

This frontier is retained for an eventual return to the primary exact contract, but it no longer
drives the bounded-track queue. The exact-hybrid search advances only at the first unresolved
operation boundary. A downstream
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
| attention-output low/final projection | output-A low exact; output-B exact after V8 overlay | exact for captured row | clean nonredundant API only after frontier survives |
| attention HC mixer/split state | exact after V10 row-wise mixer | exact for captured row | none |
| HC expansion after attention | exact inputs and output after V10 | exact for captured row | none |
| FFN HC residual + flat RMS | bit-identical after V10 in V11 capture | exact for captured row | none |
| FFN HC mixer | exact flat input; exact after V12 row-wise mixer | exact for captured row | none |
| FFN HC split + weighted sum + norm | bit-identical after V12 row-wise mixer | exact for captured row | none |
| FFN router projection | exact norm input, 186/256 differing logits after V12 | first captured unresolved operation | defer with V13; do not patch on bounded track |
| routed MoE gate/up | bit-exact in identical-input isolated prototype | exact in that harness; economics closed | retain production path; do not rebuild grouped K1 |
| routed down / sum and shared FFN | not isolated end-to-end | unknown | test only after attention frontier moves |
| output head | same batched function in V4 but inherited state differs | unknown independently | last boundary after layer path |

## Experiment contract

No new experiment starts without these fields in its worklog or artifact:

1. Hypothesis ID and one mechanism being changed.
2. Reference and candidate compositions.
3. Quality/divergence invariant and how failure is detected; exactness when the candidate claims it.
4. Economic metric and predeclared threshold.
5. Cheapest falsifier and expected effort.
6. Outcome tree for pass, fail, and ambiguous results.
7. Required retained evidence and reproduction command.
8. Pre-experiment challenge of assumptions and post-experiment red-team audit.

Under `INTERIM_BOUNDED`, candidate, fresh M3-control, fresh plain-control, and exact sequential
quality-anchor compositions must be explicit. The fixed gates and reproduction protocol live in
`artifacts/lead08_reassessment/27_iter19_bounded_divergence_reframe.md`. Runtime integrity,
acceptance, scheduler, expert work, and correction-token changes are mandatory outputs.

For deferred exact-hybrid stages, correctness and performance remain coupled:

`candidate lower bound = proven-unavoidable exactified work + optimistic remaining batch floor + runtime overhead`

The branch stops as soon as that lower bound cannot meet `verify_ms(4) <= 50.5 ms`. Moving the
correctness frontier without measuring incremental cost is diagnostic progress, not authorization for
a full build.

### T2b predeclared protocol (iteration 22)

The independent preflight returned **REDESIGN**: T2b is a one-sided arithmetic sensitivity
separator, not a two-way compute-versus-bandwidth classifier. Literal production selector 0 is
compared with one companion pipeline at actual rounds `{0,8,32,128}`. Every arm holds K=4, 24
physical pairs, 18 unique experts, mapped buffers/offsets, inputs, IDs, route weights, NSG 2, NR0 4,
grid, threadgroup memory, F32 gate/up outputs, separate activation/down work, and dispatch counts
fixed. The two fixed selection strata are contiguous-overlap and coprime-strided/dispersed.

Validity requires: all 43 eligible layers bit-identical for gate, up, weighted mid, routed output,
and output argmax; an explicit IQ2 `tiny_pair_mv` dispatch record for every measured layer; runtime
Metal/AIR with no probe load/backedge in production and two dependent FMA instructions plus a
rounds-dependent backedge in the companion; identical reported pipeline width/max-threads/static
threadgroup memory; and the production/companion-zero paired 95% interval contained within +/-2%.
Each stratum uses 20 samples per arm under four complete Latin/reversed rotations. Report unprofiled
all-layer wall time and synchronized `gate_up`, median/mean/MAD, deterministic bootstrap median and
paired intervals, and Theil-Sen slope.

Against companion-zero, monotonic R32 `gate_up` upper CI below 5% with R128 lower CI above zero is
**spare instruction headroom** and closes arithmetic reduction as the immediate mechanism. Monotonic
R32 `gate_up` lower CI at least 25%, with same-sign total movement, is **compute/issue sensitive** and
may authorize exactly one instruction/dequant reduction prototype. Intermediate, stage/total
disagreement, or stratum-dependent results are **AMBIGUOUS**. A weak response never authorizes a
traffic prototype; that requires a separately preflighted byte/latency separator or captured-live
replay.

### T2c predeclared protocol (iteration 23)

The independent preflight returned **REDESIGN**. T2c directly replays the existing mapped K4 pair
encoding and returns after gate/up. Each layer gets one command buffer; post-completion Metal
`GPUStartTime`/`GPUEndTime` is the primary timing, with kernel and wall intervals supporting. No
activation/down, stage profiler, or per-layer log is in the timed interval. Each sample sums 43
layers with fixed per-layer IDs in both overlap and coprime-dispersed strata.

Arms are literal production, identical-production duplicate, companion actual-zero, R32, and R128.
Discard two complete ten-round Latin/reversed schedules, then measure 30 rounds per stratum; each arm
occupies each order position six times. Full routed fidelity remains bit-exact across 43 layers times
two strata, direct gate/up must match, every sample must contain 43 valid finite GPU intervals, and
AIR retention remains mandatory. Production/duplicate paired 95% intervals must fit +/-1% and
production/companion-zero intervals +/-2% in both strata. Any validity failure permanently closes
this separator family.

After valid controls, R32 lower CI at least 25% in both strata plus positive monotonic R128 is
arithmetic/issue-sensitive; R32 upper CI below 5% in both plus positive R128 lower bounds is spare
headroom; otherwise close ambiguous. Positive classification authorizes only one maximum-removable
arithmetic upper-bound kernel that retains packed-byte loads/addressing with AIR proof. Its saving
lower bound must reach 47% of the 18.4 ms gate/up stage; otherwise close arithmetic reduction.

**Outcome.** T2c passes every validity gate. Production/duplicate paired intervals are
`[-0.064,+0.099]%` and `[-0.071,+0.154]%`; companion-zero/production intervals are
`[+1.149,+1.341]%` and `[+1.054,+1.195]%`. R32 versus companion-zero is +45.762%
`[+45.652,+45.934]` on overlap and +45.795% `[+45.660,+46.014]` on dispersed IDs; R128
is about +222% in both. The result is a valid one-sided arithmetic/issue-sensitive selector, not
proof that production is dequant-compute-bound. Direct production is only about 10.74 ms across 43
layers, so the unchanged absolute 8.7 ms prize is now an approximately 81% carrier-reduction gate.
U1 must use that stricter measured bound rather than the iteration-22 profiler percentage.

### U1 predeclared protocol (iteration 24)

Independent preflight returned **GO**, with U1 explicitly narrowed to an optimistic packed-weight
access/output-store floor rather than arithmetic-only attribution. The research kernel loads every
8-byte IQ2 q payload and 2-byte scale at the exact production gate/up expert-row/subblock addresses,
accumulates a live unsigned checksum per row/lane, performs a SIMD reduction, and overwrites both
production-shaped output destinations. It retains the ID-derived expert dependence, mapped weight
bases, runtime loops, NSG 2, NR0 4, 24-pair grid, and stores. It deliberately removes LUT
initialization/barrier, activation loads, dequantization, and floating dot arithmetic.

Arms are production, identical production duplicate, U1, and identical U1 duplicate. Use fixed IDs
by layer in both overlap and coprime-dispersed strata, discard 16 balanced warmup rounds, then
measure 32 Latin/reversed rounds per stratum with one command buffer per layer. Every sample must
contain 43 positive finite GPU intervals. Production/duplicate paired relative intervals must fit
+/-1%; U1/duplicate must fit +/-3% and +/-0.10 ms. U1 duplicates must overwrite outputs with exact
bits. AIR must retain ID, both mapped weight-base q/scale loads inside runtime row/subblock loops,
SIMD reduction, and both output stores, while containing no floating FMA/multiply, dequant/LUT work,
or threadgroup barrier.

The primary statistic is the same-round paired absolute median saving with 20,000 deterministic
bootstrap draws. A saving lower bound below 8.7 ms in either stratum, or stratum disagreement,
closes arithmetic reduction without tuning. Passing only establishes that this deliberately
optimistic floor is large enough; it requires ledger reassessment and another preflight and does not
automatically authorize a production prototype. Repeated control, AIR, path, or output validity
failure closes U1 invalid.

**Outcome.** The first capture omitted the production first-row weight offset and was discarded.
Corrected U1 passes validity controls but saves only 4.708104 ms `[4.683041,4.724125]` on overlap
and 4.723667 ms `[4.715042,4.735250]` on dispersed IDs. Even the upper bounds miss 8.7 ms by about
4 ms. Close arithmetic/dequant reduction without tuning or semantic prototype.

## Ranked queue and outcome rules

| Rank | Experiment | Why now | Pass | Fail / stop |
|---:|---|---|---|---|
| P0 | Redirect decision | Lead 08 bounded track is exhausted on the current runtime | Lead 05 compatibility falsifier if SSD scope is acceptable, otherwise deferred V13/V5 exact inventory | Do not reopen bounded Lead 08 without new >=8.7 ms independent evidence |
| 2 | V14 matched quality gate plus scheduled integration | Only after stage economics pass | No worse than fresh M3 control and historical caps; `verify_ms(4)<=50.5 ms`; save >=8.7 ms/cycle | Close candidate on any conjunctive failure |
| 3 | V14 full 176-entry composition | Decisive interim result | `>= max(45.8 t/s, 1.20 x fresh plain)` with all quality/integrity gates | Close candidate |
| 4 | V5/V13 exact track | Return path to primary goal after bounded path succeeds or is exhausted | Exact stream within economic gate | Defer or close based on cumulative lower bound |
| 5 | D5 soft-label redistillation | Only if verifier economics survive and acceptance remains limiting | Powered p1 lift composes materially | Close drafter axis |

T1 is blocked after its capability gate; no K1/K4 trace is authorized with the same template.
The old restore P0 and V6-V12 are complete and audited. They remain retained exact-track evidence.
No grouped-MoE, layout, geometry, mapped-residency, shared-family, top-r, or margin-policy experiment
may be repeated without new evidence that changes its upper bound. Counter attribution is not itself
a GO and must select exactly one V14 mechanism.
