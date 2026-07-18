# Issue 468 Status

The canonical current-state page for the branch. Single coherent narrative — the
bottom line first, then the investigation arc, then durable findings by axis. The
full artifact/tool inventory moved to `issue468/inventories/dossier_inventory.md`.

## Bottom line

**Goal:** validate whether DSpark-style speculative decoding can deliver **≥20 % greedy
decode throughput on ds4 (DeepSeek-V4-Flash, IQ2XXS)** with exact output preserved. Lead 08 now
also has an explicit `INTERIM_BOUNDED` track: M3-level task-quality/distribution non-regression is
acceptable for selecting and measuring a faster verifier, but cannot satisfy the primary goal.

**Current best (Milestone 3, 2026-07-15, COMPLETE):** the full DSpark stack **beats plain
ds4 by +4.9 %** (40.04 vs 38.16 t/s, full 176-entry corpus; +4.8 % on long-context) and is
**score-neutral** on the 92Q (61/92 vs 60/92, 90.2 % same verdict, net +1) — the first
config to beat plain locally. **+20 % is not reached**: verify still dominates ~80 % of the
cycle (~16 ms/token off the target).

| | |
|---|---|
| Goal | ≥20 % greedy throughput on ds4 IQ2XXS, exact-output-preserved |
| Current best | Full DSpark stack **+4.9 %** over plain (40.04 vs 38.16 t/s; score-neutral 61/92) — M3 |
| The gap | +20 % not reached; verify ≈80 % of the cycle |
| Open lever | **Lead 08 V15 Stage B2 seam/economic gate.** Stage B1 passes 10,535/10,535 direct M=2..8/seven-site cases and fourteen-kernel AIR topology. Capture C5, factor output-B nonredundantly, then require the M=4 all-layer delta upper bound <=7.5 ms |
| Closed (negative) | Drafter/input quality (Lead 07), drafter quant (Q4_K), non-expert finetune (Stage 2), DFlash, quant-mismatch |

## Investigation arc

Reverse-chronological. Each entry: what was tested → verdict → canonical record.

- **2026-07-18 - Lead 08 iteration 29 V15 Stage B1: PASS direct capability.** The shared Q8/F16
  family now has static M=2..8 specializations. All 10,535 all-layer/seven-site/C0-C4 direct cases
  match M unchanged M1 calls word-for-word; all fourteen AIR specializations pass conservative
  raw-load/grid/primary-recurrence topology checks. This is not a register/spill, speed, output-B
  seam, or final Stage-B result. C5/seam/timing remain P0 in Stage B2.
  `artifacts/lead08_reassessment/37_iter29_exact_smallm_direct.md`.

- **2026-07-18 - Lead 08 iteration 28 V15 Stage A: PASS capability.** One research-only host
  operation and shared structural Metal template cover Q8 Q-a and the F16 router at M=2 without
  changing graph callsites. All 430 real-offset/C0-C4 cases match two unchanged M1 references word
  for word. Runtime compilation and AIR show one logical format traversal, raw loads before the
  independent token arithmetic, and no token term in weight addresses; physical DRAM traffic is
  not claimed. Stage B's M=2..8 seven-site coverage, output-B seam, and <=7.5 ms economic gate are
  still untested. `artifacts/lead08_reassessment/36_iter28_exact_smallm_pair.md`.

- **2026-07-18 - Lead 08 iteration 27 V15 protocol: REDESIGN; staged GO.** A monolithic both-format
  M=2..8 build, seam refactor, exhaustive correctness, and timing iteration is not auditable. Stage A
  now admits only a research-only M=2 pair for Q8 Q-a and F16 router across all 43 offsets and five
  deterministic finite input classes, with literal M1 bit equality and source/AIR one-traversal
  proof. Only its audited both-format pass admits Stage B's seven-site M=2..8 gate, nonredundant
  output-B seam, and cumulative all-layer <=7.5 ms policy test. No graph edits are admitted.
  `artifacts/lead08_reassessment/35_iter27_exact_smallm_protocol.md`.

- **2026-07-18 - Lead 08 iteration 26 exact small-M dense inventory: DESIGN COMPLETE; V15
  conditional.** The initially proposed F16/router helper was too narrow. The exposed exactness debt
  spans four Q8 and three F16 sites totaling 77.875 MiB/layer. Four independent M1 traversals add
  9.810 GiB/43 layers, so the only admissible structure loads each weight once while retaining an
  independent literal M1 arithmetic DAG per token. Source seams are plausible, though output-B
  requires a nonredundant batch seam. Iteration 27 subsequently stages the implementation: a
  both-format M=2 capability spike precedes the retained all-seven M=2..8 and
  candidate-minus-current-ext all-43 delta upper-CI gate. No implementation or speed claim yet.
  `artifacts/lead08_reassessment/34_iter26_exact_smallm_dense_inventory.md`.

- **2026-07-18 - Lead 08 iteration 25 bounded-track audit: STOP / EXHAUSTED; corrected audit COMMIT.**
  The removable arithmetic saving is upper-bounded below 4.74 ms by U1; retained traffic is distinct but has no surviving
  realizable mechanism. Compatible residency, launch/host, grouping, layout,
  geometry, top-r, margin, and readahead are already too small or closed. Down/attention/dense lack
  both a fresh >=8.7 ms isolated carrier and a concrete bounded-quality mechanism. No further
  current-runtime bounded experiment is admitted; corrected audit COMMIT.
  `artifacts/lead08_reassessment/33_iter25_bounded_track_exhaustion.md`.

- **2026-07-18 - Lead 08 iteration 24 packed-weight floor: U1 economic STOP; corrected audit
  COMMIT.** A nonsemantic checksum kernel retains mapped IQ2 q/scale loads, ID addressing, grid,
  SIMD reduction, and stores while removing activation/LUT/dequant/dot work. All controls, 86-case
  store-liveness/duplicate checks, 11,008 timestamps, and AIR gates pass. Paired saving intervals
  are `[4.683,4.724]`/`[4.715,4.735]` ms, far below 8.7 ms. The first capture's
  missing row offset was discarded. Arithmetic reduction closes without a production prototype.
  The repeat audit reproduced corrected AIR, controls, evidence generation, and STOP economics.
  `artifacts/lead08_reassessment/32_iter24_weight_floor.md`.

- **2026-07-18 - Lead 08 iteration 23 direct mapped-pair replay: T2c VALID positive selector;
  audit COMMIT.** All selectors pass 86/86 full/direct bitwise cases; all 300 measured samples retain
  43 positive finite command-buffer GPU intervals; each arm occupies every order position six times.
  Duplicate/production intervals stay inside +/-0.16% and companion-zero/production inside +1.35%.
  R32 versus zero is +45.76%/+45.79% with lower bounds above +45.65%; R128 is about +222%.
  This establishes one-sided arithmetic/issue sensitivity, not dequant-compute dominance. Direct
  production is about 10.74 ms/cycle, making the 8.7 ms prize an approximately 81% carrier cut. Only
  one preflighted maximum-removable arithmetic upper-bound kernel is authorized. The audit
  regenerated all evidence and reproduced the code-path, AIR, statistics, and decision.
  `artifacts/lead08_reassessment/31_iter23_direct_gpu_replay.md`.

- **2026-07-18 - Lead 08 iteration 22 mapped ALU separator: T2b INVALID / near-threshold;
  audit COMMIT.** All companion arms are bit-exact on 86/86 all-layer/two-stratum cases, all
  dispatches stay on mapped IQ2 K4 `tiny_pair_mv`, and AIR retains the runtime FMA loop. R32 adds
  about 26-27% synchronized gate/up, but three production/companion-zero intervals escape +/-2%
  and the overlap R32 lower bound is 24.958%, just below the 25% positive threshold. No prototype;
  T2c control-stabilized direct replay preflight is P0.
  `artifacts/lead08_reassessment/30_iter22_mapped_alu_separator.md`.

- **2026-07-18 - Lead 08 iteration 21 selected-address ALU separator: T2a AMBIGUOUS; repeat audit
  COMMIT.** Literal production and companion-zero agree within 2%, and all companion arms are
  bit-exact on the layer-0 fidelity pattern. The response is monotonic, but 32 rounds add +5.41%
  total/+10.10% gate/up, inside the predeclared ambiguous band; 8-round intervals include zero. The kernel is exclusive to the SSD
  selected-address carrier, which cannot compose with DSpark. A follow-up path trace identifies all
  688 mapped K4 records as `tiny_pair_mv`; T2b on that live kernel is P0.
  `artifacts/lead08_reassessment/29_iter21_ssd_addr_alu_headroom.md`.

- **2026-07-18 - Lead 08 iteration 20 Metal counter capability: tooling BLOCKED; audit COMMIT.**
  Default GPU Counters rejects its profile. Metal System Trace exposes only zero-valued `RT Unit
  Active`, resolves 119 shader names but zero shader intervals, and perturbs the steady K4 verify
  median 62.082 -> 66.754 ms (+7.5%). No binding claim; T2 controlled separation becomes P0.
  `artifacts/lead08_reassessment/28_iter20_metal_counter_capability.md`.

- **2026-07-18 - Lead 08 iteration 19 bounded-divergence reframe: corrected audit COMMIT.**
  Exact output remains the project success gate, but Lead 08 may now optimize inside a matched M3
  task-quality/distribution envelope. V1 becomes the interim baseline; V5/V13 exact accumulation is
  deferred; failed performance mechanisms remain closed; counter-guided V14 becomes rank 1.
  The first audit caught four contract holes; the corrected paired-control, trajectory, throughput,
  and committed-divergence rules passed repeat audit.
  `artifacts/lead08_reassessment/27_iter19_bounded_divergence_reframe.md`.

- **2026-07-18 - Lead 08 iteration 18 row-wise HC FFN mixer: causal GO; audit COMMIT.**
  Mixer, split, HC pre, and FFN norm become exact for layer-0 row 0; router logits are the next
  boundary, while top-6 expert IDs remain exact. Isolated synchronized `ffn/hc_pre` timing shows no
  local penalty (-0.029 ms mean) but is diagnostic-only. The audit reproduced all correctness/timing
  values and endorsed V13: inventory a generic exact batched-F16 mechanism before any further row
  helper. `artifacts/lead08_reassessment/26_iter18_rowwise_hc_ffn_mix.md`.
- **2026-07-18 - Lead 08 iteration 17 FFN HC input localization: REDESIGN; corrected audit COMMIT.**
  Attention HC post, FFN-entry residual, and flat RMS input are exact for layer-0 row 0. The F16
  FFN mixer is the first difference at 14/24 values; split and `hc_ffn_pre` inherit that debt. V12
  row-wise FFN mixing is next. Capture-only, no timing claim. The first audit corrected the residual
  alias placement description; the repeat audit returned COMMIT.
  `artifacts/lead08_reassessment/25_iter17_ffn_hc_input_localization.md`.
- **2026-07-18 - Lead 08 iteration 16 row-wise HC attention mixer: causal GO; corrected audit COMMIT.**
  Exact mixer input now yields exact mix, consumed split state, and attention HC post for layer-0
  row 0. The first captured difference moves to the FFN HC path. Four matched all-layer supporting
  pairs show no obvious penalty, but their -1.25 ms sign is route-confounded and uninterpretable for
  mixer economics. The first audit required this wording correction; the repeat audit returned COMMIT.
  `artifacts/lead08_reassessment/24_iter16_rowwise_hc_attn_mix.md`.
- **2026-07-17 - Lead 08 iteration 15 HC expansion inputs: REDESIGN; audit COMMIT.** Block and
  residual inputs are exact, but the early HC mixer differs in 14/24 values and the split post/comb
  state consumed after attention differs in 15/20. Expansion is not independently implicated; the
  frontier returns to the F16 HC mixer. No timing claim. `artifacts/lead08_reassessment/23_iter15_hc_input_localization.md`.
- **2026-07-17 - Lead 08 iteration 14 row-wise attention output-B: causal GO; audit COMMIT.**
  Branch-neutral capture proves output-A low is exact. Output-B moves from 3,319/4,096 differing
  values to bit-identical; `hc_attn_post` is next. The final flip remains. The redundant all-layer
  overlay adds a confounded +6.99 ms and is supporting-only, not a feasibility lower bound.
  `artifacts/lead08_reassessment/22_iter14_rowwise_attn_out_b.md`.
- **2026-07-17 - Lead 08 iteration 13 row-wise Q-b: causal GO; audit COMMIT.** Metal's supposed
  fused Q-b hook is a stub, so V7 exactifies only the actual Q8 Q-b projection. With V6 enabled,
  layer-0 row 0 becomes bit-identical through head norm, Q RoPE, attention, and inverse RoPE. The
  cycle-24 flip remains, while max logit error falls 0.927542 -> 0.133129. Matched K=4 supporting
  timings put layer-0 cost at noise scale and the naive all-layer whole-graph delta at +7.47 ms with
  matched aggregate selected-byte/unique-count volume. Expert identity, locality, and downstream
  numeric effects remain confounders; this is an upper-bound estimate, not an unavoidable cost.
  `artifacts/lead08_reassessment/21_iter13_rowwise_qb.md`.
- **2026-07-17 - Lead 08 iteration 12 row-wise Q/KV: causal GO, audit COMMIT.** A verifier-scoped
  M=1 row projection makes layer-0 row-0 `q_lora`, `KVraw`, both normalizations, and the KV
  RoPE/storage path bit-identical. The first difference moves to production `Qcur` (fused Q-b +
  head norm + RoPE). Four order-balanced process-first K=4 samples observe noisy all-43-layer deltas
  of +0.56 ms mean / +0.69 ms median layer execution, with changed downstream work in the
  full-corpus pair. This is a naive implementation upper-bound estimate, not an isolated cost or
  unavoidable lower bound. `artifacts/lead08_reassessment/20_iter12_rowwise_qkv.md`.
- **2026-07-17 - Lead 08 iteration 11 same-frontier localization: corrected audit COMMIT.**
  A noncommitting M=K probe followed by batched-M1 replay finds
  2 argmax flips across 349 accepted-row comparisons, with 640 batched and zero raw singleton target
  evaluations. At a reproduced flip, row-0 `hc_attn_pre` and `attn_norm` are bit-identical; `q_lora`
  is the first captured difference (799/1024 F32 words, max 3.35e-8), which amplifies to post-FFN
  max 0.94 by layer 42. The first audit found restore failure was ignored. Restore failure now
  hard-fails the diagnostic; all 267 post-fix probe executions restored successfully, and the corpus
  and dumps reproduce. The repeat audit passed; the bounded Q/KV falsifier is next.
  `artifacts/lead08_reassessment/19_iter11_same_frontier_localization.md`.
- **2026-07-17 - Lead 08 iteration 10 shared batch M=1/M=K family: exactness and economics
  NO-GO.** Re-baselining target M=1 onto the existing layer-major batch graph yields only 5/10
  token-identical speculative prompts. Batch M=1 regresses target throughput 37.902 -> 32.989 t/s;
  M=K speculative reaches 37.407 t/s, only +13.4% over that slower target and -1.3% versus shipped
  decode. This closes the existing batch graph as the shared family, not the still-unbuilt hybrid
  exact-batch design. `artifacts/lead08_reassessment/18_iter10_batch_m1_target.md`.
- **2026-07-17 - Lead 08 iteration 9 profiler correction: iteration-8 locality superseded;
  verdict unchanged.** The iteration-8 extension read reused router buffers before the Metal command
  buffer completed. A stale run yielded only 11 distinct top-16 vectors across 43 router layers;
  per-layer GPU snapshots now yield 43/43 and preserve the complete cycle trajectory exactly. The
  corrected three-family LRU hit curve is 57.3%/72.3%/81.5%/87.9% at capacities 16/32/64/128.
  `artifacts/lead08_reassessment/17_iter9_profiler_correction.md`.
- **2026-07-17 - Lead 08 iteration 8 cache residency: current-path NO-GO.**
  Cold SSD selected-address preparation costs 5.407 ms/layer and bit-exact replay cuts routed time
  11.094 -> 1.353 ms (-87.8%). That carrier is unavailable: the runtime rejects `--ssd-streaming`
  with `--dspark`. The compatible mapped-weight path is already replay parity at 0.566 -> 0.566 ms
  (-0.04%). Corrected full-stack profiling shows 57.3%/72.3% per-layer LRU hits at capacities 16/32,
  exploiting them requires a separate SSD-compatible runtime, not another Lead 08 kernel change.
  `artifacts/lead08_reassessment/16_iter8_cache_residency.md`.
- **2026-07-17 - Lead 08 iteration 7 production row tile: NO-GO.**
  Held NSG 2 and the K=4 18/24 physical-row workload fixed while sweeping 2/4/8 output rows per
  SIMDgroup. Tiles 2 and 8 are bit-exact against production tile 4 but are +1.0%/+0.3% slower over
  eight total-path rounds; tile 8 changes sign across repetitions, confirming noise scale.
  Tile 8's apparent -2.5% matched gate+up result becomes +0.05% when startup-heavy layer 0 is
  excluded; tile 2 becomes +1.3%. The production tile remains 4 and the geometry branch closes.
  `artifacts/lead08_reassessment/15_iter7_addr_row_tile.md`.
- **2026-07-17 - Lead 08 iteration 6 production address-kernel geometry: NO-GO.**
  Held the K=4 physical-row algorithm and 18/24 shape fixed while sweeping 1/2/4/8 SIMDgroups per
  threadgroup. All variants are bit-exact. The best total result is NSG 4 at -3.2%, but its matched
  gate+up result is +1.1% overall / -1.1% excluding layer 0; NSG 8 is -2.0% / -0.7%. These noisy,
  low-single-digit effects are far below the >=15% stage gate, so the production default stays NSG 2.
  `artifacts/lead08_reassessment/14_iter6_addr_nsg_geometry.md`.
- **2026-07-17 - Lead 08 iteration 5 expert-address locality: software-layout NO-GO.**
  Held production K=4 work fixed at 24 physical rows / 18 unique experts and changed only expert
  placement: contiguous, same-set permuted, or slab-wide strided. Across four cold-cache rounds,
  permuted and strided total routed cost were only +1.9% and +0.7% versus contiguous; matched gate+up
  deltas were +1.6% and +0.9% (+1.8%/+1.4% excluding layer 0). Expert remapping/packing cannot supply
  the >=15% stage gain or 8.7-10 ms verifier prize. Hardware counters could still attribute the
  remaining cost, but this controlled result closes address reordering as a mechanism.
  `artifacts/lead08_reassessment/13_iter5_address_locality.md`.
- **2026-07-17 - Lead 08 iteration 4 margin-guarded exact fallback: exactness/cost NO-GO.**
  Extended the non-committing batch-vs-exact probe over the retained 10-prompt exactness workload.
  The known flip has batch top-2 margin 0.3755, so threshold 0.25 misses it. Threshold 0.5 catches
  the one observed flip but guards 27/158 probed cycles (17.1%); measured exact replay adds an
  optimistic 7.0 ms over all cycles and projects 40.04 -> 36.37 t/s, below plain 38.16. One flip
  cannot establish a universal margin bound. Do not implement the fallback policy.
  `artifacts/lead08_reassessment/12_iter4_margin_guard.md`.
- **2026-07-17 — Lead 08 iteration 3 grouped gate+up prototype: bit-exact -> performance NO-GO.**
  Built the one bounded expert-major/threadgroup-spill prototype. Direct production comparison has
  zero differing F32 bits in gate/up/weighted-mid/routed-output and zero argmax flips, but target
  18-unique/24-pair gate+up regresses **4.869 -> 5.986 ms/layer (+22.9%)** instead of improving
  >=15%; excluding layer 0 regresses 35.8%. The compact expert map makes disjoint execution parity,
  while heavier overlap regresses 56-68%, isolating spill/barrier costs. Stop this branch;
  do not integrate or run the final verifier gate. `artifacts/lead08_reassessment/11_iter3_grouped_gateup_prototype.md`.
- **2026-07-17 — Lead 08 iteration 2 production overlap gate: blocker resolved -> CONDITIONAL GO.**
  Replaced the cache-confounded M1x2 probe with K=4 `routed_moe_batch` measurements at a fixed 24
  physical pairs and 6/12/18/24 unique experts. Total cost tracks union-aware preparation, but the
  realistic partial-overlap gate+up stage (18 unique) is still **97% of fully disjoint** despite
  25% fewer unique experts; sparse doubletons remain a real grouped-kernel opportunity. Down is
  smaller. This authorized exactly one expert-major/threadgroup-spill prototype with a >=15% gate;
  iteration 3 subsequently failed it and stopped the branch. The prior 6-8 ms prize was not proven.
  `artifacts/lead08_reassessment/10_iter2_production_batch_overlap.md`.
- **2026-07-16 — Lead 08 iteration 1 simultaneous M=2 fusion: structural NO-GO; reported magnitude
  withdrawn.** Fidelity met the relaxed bar (max_abs≈4.8e-08, zero argmax flips), but the kernel
  computes 18 vs 12 token-expert streams and doubles live per-thread state. The reported 4.4x and
  derived 2.9x magnitudes used a cache-warm M1 comparator and are not cold-equivalent; do not reuse
  them. The simultaneous design stays dormant. Artifacts 05/06/09 + gate-B report.
- **2026-07-16 — Lead 07 (crossed FP/IQ2 oracle): PIVOT, closed negative.** The native-vs-IQ2
  drafter-acceptance gain is **not a recoverable hidden-side effect**. On a teacher-forced
  common trajectory the FP-vs-IQ2 p1 lift is +0.007 (CI incl 0); the recoverable hidden-side
  effect (FP hidden + deployable IQ2 labels) is −0.0068 (significantly *negative*); the FP
  ceiling's block advantage requires the FP target's undeployable labels. → the drafter/input
  quality axis closes; Experiment 2 not warranted. `archive/leads/lead_07_upstream_quality_ceiling.md`.
- **2026-07-15 — Milestone 3 COMPLETE: full stack +4.9 % over plain, score-neutral.** Levers:
  committing batched verify (sublinear, divergent-but-score-neutral), anchor-reuse,
  prefix-checkpoint, the GPU Metal drafter (draft 45→7.6 ms), anchor-reuse-for-Metal, STS
  threshold re-tune. +20 % not reached (verify dominates). `summaries/dspark_runtime_milestone_3_progress.md`.
- **2026-07-13/14 — M3 runtime levers.** Batched verify wired in (env `DS4_DSPARK_VERIFY_BATCHED`);
  the GPU drafter + anchor-reuse + STS composed. Each lever env-gated + codex-gated.
  `summaries/dspark_runtime_milestone_3_progress.md`.
- **2026-07-13 — Lead 08 Phase B: NO-GO via swaps → bounded build.** No swap-only config
  yields a sublinear bit-exact verifier; the existing bit-exact verifier is ~0.85× baseline.
  Cost side is favorable-but-unproven (K=4 verify floor ~39–43 ms at decode bw → 1.34–1.44×
  *if* built sublinear+exact). Attempt the fused-kernel build as a bounded effort with a hard
  exit gate. `summaries/lead08_phaseB_floor_clearance_verdict.md`.
- **2026-07-12 — Lead 08 Phase A: retained headroom → proceed to Phase B.** `verify_ms(K) −
  floor` ~19–22 ms/cycle (above the ~15 ms gate). `summaries/mtp_verifier_engineering_and_phaseA.md`.
- **2026-07-08 → 07-11 — the lead sequence that bounded the problem:**
  - **Lead 06** — anchor reuse implemented exactly (sequential-reuse substrate): +20.95 % over
    shipped `--mtp`, but −16.26 % vs baseline. `summaries/mtp_verifier_engineering_and_phaseA.md`.
  - **Lead 04** — native-FP ceiling capture: a real float32 ceiling (+8–10 % cycle-jump) but
    F16-deployment-blocked + capture-fidelity unresolved → HOLD. `archive/leads/lead_04_fp_ceiling_capture.md`.
  - **Lead 03** — acceptance statistical power + cycle-jump: realistic E[a\|4]=2.198, S(4)=0.340
    → 0.98× at K=4 (the honest pre-M3 band). `summaries/acceptance_statistical_power.md`.
  - **Lead 02** — confidence scheduling (STS): marginal, only a fragile conditional secondary
    under anchor-reuse. `summaries/confidence_scheduled_verification.md`.
  - **Lead 01** — anchor-reuse falsifier: survives (no collapse; non-inferiority de-risked,
    not fully powered). `summaries/anchor_reuse_falsifier.md`.
  - **Stage 2** — non-expert head-LoRA finetune: significantly HURTS (−1.5 to −3.1 pp p1).
    `summaries/stage2_finetune_result.md`.
  - **Stage 0/1** — quant-mismatch: drafter p1 misses are shallow/recoverable-shape, but the
    quant-flip attribution is unproven; Q4-tap didn't help → NARROW. `summaries/quant_mismatch_recommendation.md`.
  - **DFlash drafter** — ~2.5× worse accepted prefix than DSpark. `summaries/dflash_oracle_investigation.md`.
  - **Q4_K ceiling** — Q4_K ≈ F16 ≈ F32; drafter quant is not the bottleneck. `summaries/dspark_quantization_ceiling.md`.

## Findings by axis

### Drafter & input quality — CLOSED NEGATIVE

The drafter is **not** the lever, on four independent grounds:

- **Drafter weight precision:** Q4_K ≈ F16 ≈ F32 (the vendored drafter ships MXFP4; F16
  already captures the full dequant). Removing routed-expert quant yields +0.38 % net accepted
  prefix (noise). `summaries/dspark_quantization_ceiling.md`.
- **Non-expert head-LoRA finetune (Stage 2):** significantly HURTS — −1.46 / −1.75 / −3.08 pp
  held-out p=1 across rank 32/64/128; McNemar p=5.3e-5. `summaries/stage2_finetune_result.md`.
- **Native-hidden ceiling (Lead 04 → 07):** Lead 04 measured a real float32 native-FP ceiling
  (+8–10 % cycle-jump) but it is F16-deployment-blocked + its capture fidelity is unresolved.
  Lead 07's crossed oracle then **PIVOTed**: on a common trajectory the recoverable hidden-side
  effect is −0.0068 (negative); the ceiling requires the FP target's undeployable labels; the
  residual gap is target-trajectory difficulty, not hidden precision. **Not recoverable on
  IQ2XXS.** `archive/leads/lead_07_upstream_quality_ceiling.md`.
- **DFlash drafter:** ~2.5× worse accepted prefix than DSpark on this corpus. `summaries/dflash_oracle_investigation.md`.

### Verifier / cycle cost — GAP REMAINS; EXACT CAPABILITY GATE OPEN

Verify dominates ~80 % of the cycle; this is where the +20 % must come from.

- **Binding regime is not yet established empirically.** Host readback is not the bottleneck and
  dense Q8 paths are plausibly bandwidth-bound, but the production routed IQ2 verifier has not been
  separated among DRAM traffic, dequant/instruction throughput, occupancy/stalls, and dispatch
  gaps. The old roofline/source audit is a hypothesis, not a hardware-counter result.
  `summaries/mtp_verifier_bandwidth_binding.md`.
- **Milestone 2:** the sublinear batched verifier is break-even at oracle acceptance
  (crossover ~2.3 accepts at K=4); the runtime drafter is sound (live ≥ oracle). `summaries/dspark_runtime_milestone_2_progress.md`.
- **Milestone 3:** committing batched verify (sublinear, divergent-but-score-neutral) + the
  full stack → **+4.9 % over plain**, the first config to beat plain locally. `summaries/dspark_runtime_milestone_3_progress.md`.
- **Lead 08:** Phase A found retained headroom (~19–22 ms/cycle above the floor), but Phase B found
  no swap-only sublinear bit-exact verifier (~0.85× baseline). The controlled production probe then
  justified one grouped gate+up implementation. That prototype is bit-exact but **22.9% slower** at
  the target 18/24 shape (35.8% slower excluding layer 0), so the predeclared >=15% stage gate fails
  with the wrong sign. The grouped branch stops and the final `verify_ms(4) ≤ 50.5 ms` integration
  gate is not warranted. The separate margin-guard probe also fails: threshold 0.25 misses the known
  greedy flip, while 0.5 catches it only by triggering exact replay on 17.1% of probed cycles and
  adding at least 7.0 ms/cycle, which erases M3's advantage over plain. Finally, a fixed-work
  production probe varied only expert-address locality: same-set permutation and slab-wide striding
  changed total routed cost by just +1.9% and +0.7% versus contiguous, with matched gate+up movement
  below 2%. This closes software-visible address remapping/packing as the speculative coalescing
  mechanism, without claiming hardware bandwidth-vs-compute attribution. A follow-up production
  kernel-geometry and row-tile sweeps also fail: all variants are bit-exact, but matched effects
  remain within about 3% and are inconsistent between total and isolated-stage views. Exact replay
  then exposes an 87.8% SSD selected-address saving, but the DSpark-compatible mapped path is already
  at replay parity. Iteration 9 corrected the locality profiler without changing that carrier
  verdict. An independent audit then corrected the broader record: the full exact hybrid was never
  built. Iteration 10 tests the cheaper existing-batch shared-family strategy, but it diverges on
  5/10 prompts, regresses target M=1 by 13.0%, and delivers only +13.4% over that slower baseline.
  Iteration 11 then localizes the first same-frontier row-0 divergence to layer-0 attention
  projection, before routed MoE. Iterations 12-18 move that captured exact frontier through FFN
  normalization, but the bounded contract now defers further exactification. The failed grouped,
  layout, geometry, mapped-residency, existing batch-family, top-r, and margin mechanisms remain closed:
  relaxing exactness does not change their economic falsifiers. Iterations 20-25 then exhaust V14:
  counters are unavailable, controlled arithmetic sensitivity does not yield the required saving,
  and no independent composable >=8.7 ms mechanism survives. Iteration 26 returns to exactness and
  completes V13's seven-site cross-quant inventory. V15 is now the sole conditional capability
  gate; it is not yet implementation, speed, or full-path exactness evidence. The 43 ms value is a
  conservative policy base within a modeled 39-43 ms range, not a proved lower bound. Artifacts 10-34 under
  `artifacts/lead08_reassessment/`.

### Scheduler & acceptance

- **Lead 01 (anchor-reuse falsifier):** survives — 5/6 cells survive, 1 marginal, no collapse;
  the acceptance-axis risk is downgraded from "load-bearing/unverified" to "no large collapse,
  non-inferiority not fully powered." Anchor-reuse is now implemented exactly in the M3 stack.
  `summaries/anchor_reuse_falsifier.md`.
- **Lead 02 (confidence scheduling / STS):** marginal — a fragile conditional secondary only
  under anchor-reuse (frozen-threshold ~1.04× under reuse on fresh data); does not revive the
  gate under shipped economics. `summaries/confidence_scheduled_verification.md`.
- **Lead 03 (acceptance power + cycle-jump):** the realistic per-cycle trajectory gives
  E[a\|4]=2.198, S(4)=0.340 → 0.98× at K=4 (the honest pre-M3 acceptance currency; the full M3
  stack composes scheduler improvements on top to reach +4.9 %). Corpus-dependent.
  `summaries/acceptance_statistical_power.md`.

## What exists

Full artifact/tool/harness inventory (moved out of this file): **`issue468/inventories/dossier_inventory.md`**.
The speculative-speedup model: `summaries/spec_speedup_model.md` + `model_spec_speedup.py`.
The bench harness: `ds4-spec-bench` (`make ds4-spec-bench`) — use it, not the `ds4` CLI, for measurements.

## Next step

Experiment selection and stop rules are maintained in
`summaries/solution_space_ledger.md`; new work must enter that ledger before implementation.

**Lead 08:** T1's Metal-counter capability gate is blocked. T2a is ambiguous on the incompatible
SSD carrier and T2b is invalid. T2c now validly selects one-sided arithmetic/issue sensitivity on
the live mapped `tiny_pair_mv` kernel, without proving production is dequant-compute-bound. Its
literal direct carrier is about 10.74 ms/cycle. Corrected address-faithful U1 saves only 4.71-4.72
ms even after removing activation and almost all
semantics, so arithmetic/dequant reduction closes without a broad kernel sweep. Iteration 25 finds
no independent concrete >=8.7 ms mechanism, so Lead 08 bounded work is exhausted on this runtime.
V13 completes the cross-quant inventory and iteration 27 completes V15's executable protocol.
Iteration 28 passes Stage A, and iteration 29 passes Stage B1: all fourteen Q8/F16 M=2..8
specializations cover all seven real tensor sites, 43 offsets, and C0-C4 with literal M1 bits and
the required logical topology. Stage B2 is now P0: captured C5, the nonredundant output-B seam,
and the cumulative economic gate. This is not yet a speed result or full-path exactness.
Lead 05 SSD compatibility remains an alternate redirect if that scope becomes acceptable. Any future bounded-track admission additionally
requires `verify_ms(4) <= 50.5 ms` and `>= max(45.8 t/s, 1.20 x fresh plain)` on the 176-entry corpus, and no
regression from a fresh matched M3 control on the recorded task-quality and distribution gates.
V5 is active only behind the V15 P0; V13 is complete. Do not repeat the grouped, margin, layout,
geometry, mapped-residency, existing batch-family, or top-r experiments without evidence that changes their
upper bounds.

**Lead 10 — drafter re-distillation for IQ2XXS (soft labels)** is a proposed, lower-priority
drafter-quality follow-up: the one untested route after Lead 07 (hidden-side, dead) and Stage 2
(head-only hard-label, dead). It tests whether the 0.79 IQ2-native acceptance is a distribution
mismatch a full-body soft-label re-distillation can close, or a capacity ceiling. Powered to
detect +2 pp on 60 held-out prompts; corpus + capture input prepared (`issue468/data/distill_corpus/`).
A real but uncertain bet (negative prior); not yet started. Design in
`issue468/pending/lead_10_drafter_redistillation.md`.

## Canonicality rule

This file is the source of truth for the branch's current research state. Verdicts live here
(Findings by axis); the inventory is `inventories/dossier_inventory.md`; per-lead provenance is
`archive/leads/`. Any deeper note should support this file, not contradict it; when a result
supersedes an earlier one, rewrite the earlier framing rather than layering an update on top.
