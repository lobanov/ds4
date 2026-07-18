# Lead 08 — Fused low-K batch-verify kernel (close the verify-vs-floor gap)

> **Current verdict (2026-07-18): bounded work is exhausted; primary exact track resumed.**
> The prior `INTERIM_BOUNDED` contract remains a valid research envelope, but no current-runtime
> mechanism clears its 8.7 ms admission bar. V1 remains the interim baseline. V13 has completed a
> cross-quant exact small-M dense inventory and admits V15, one Q8/F16 family-level capability
> prototype. V15 Stage A now passes 430/430 all-layer M=2 word-exact cases with compiled AIR
> showing one logical weight traversal per format. Stage B's M=2..8, seven-site, output-B seam,
> and <=7.5 ms cumulative economic gates remain untested. T1 Metal counter attribution is tooling-blocked, T2a applies only
> to the incompatible SSD address kernel, and mapped-kernel T2b is invalid on its control gate.
> T2c validly selected one-sided arithmetic/issue sensitivity, but corrected address-faithful U1
> saves only 4.71-4.72 ms even after removing activation and almost all semantics. Arithmetic and
> dequant reduction are economically closed; V14 has no selected mechanism. Only a genuinely
> independent mechanism with a fresh >=8.7 ms upper bound may reopen bounded work. K=4 verify must
> reach <=50.5 ms, and the composed 176-entry run must reach
> `>= max(45.8 t/s, 1.20 x fresh plain)`.
> Separators select one mechanism; they are not themselves evidence of a speedup.
>
> **Active exact-track state:** iterations 12-14 move the captured layer-0 exactness frontier
> through Q/KV, Q-b, attention, inverse RoPE, and output-B. Iteration 15 finds the post-attention
> debt is inherited from the early HC mixer. Iteration 16 exactifies that state through attention
> HC post; iteration 17 localizes the next debt to the FFN HC mixer, and iteration 18 exactifies
> through FFN normalization. Iteration 26 shows that the required dense mechanism must span four Q8
> and three F16 sites; iteration 28 passes V15 Stage A for Q8 Q-a and the F16 router. Stage B is
> the next conditional family/economic gate. Iteration 10
> falsified the existing batch graph as a shared
> M=1/M=K exactness family; iteration 9 corrected the iteration-8 locality profiler;
> cache residency remains a current-path NO-GO; iterations 6/7
> production kernel geometry = NO-GO; iteration 5
> address locality = NO-GO; iteration 4 margin guard = NO-GO; iteration 3 grouped kernel = NO-GO.**
> Exact selection replay saves 87.8% on the SSD selected-address path, but `--ssd-streaming` is
> incompatible with `--dspark`. On the compatible mapped-weight path, cold and replay are both
> 0.566 ms/layer (-0.04%). Corrected three-family verifier profiling finds 57.3%/72.3% LRU hits
> at capacities 16/32, but that locality still has no current carrier. Artifacts:
> `issue468/artifacts/lead08_reassessment/16_iter8_cache_residency.md` and
> `issue468/artifacts/lead08_reassessment/17_iter9_profiler_correction.md`.
>
> `DS4_LEAD08_BATCH_M1_TARGET=1` re-baselines target decode onto the same layer-major graph used by
> the M=K verifier. On the retained 10-prompt/640-token corpus, M=K speculative output matches that
> target on only 5/10 prompts. The batch-M1 target is 32.989 t/s versus shipped decode 37.902
> (-13.0%), while speculative is 37.407 t/s: only +13.4% over the slower target and -1.3% versus
> shipped decode. The existing end-to-end batch graph/state transition is not M-invariant; this
> experiment does not localize the cause to a kernel. Artifact:
> `issue468/artifacts/lead08_reassessment/18_iter10_batch_m1_target.md`.
>
> With committing M=K disabled, a noncommitting M=K probe followed by batched-M1 replay finds 2
> argmax flips across 349 accepted-row comparisons (`640` batched / `0` raw singleton target
> evaluations). At a reproduced flip, layer-0 row-0 `hc_attn_pre` and `attn_norm` are bit-identical;
> `q_lora` is the first captured difference. Artifact:
> `issue468/artifacts/lead08_reassessment/19_iter11_same_frontier_localization.md`.
>
> Fixed-work SIMDgroup-count and output-row-tile sweeps preserve the physical-row algorithm and all
> variants are bit-exact. SIMDgroup effects stay within 1-3%; alternate row tiles are slower total
> and at steady state. Neither approaches the >=15% gate, so production remains NSG 2 / row tile 4.
> Artifacts: `issue468/artifacts/lead08_reassessment/14_iter6_addr_nsg_geometry.md` and
> `issue468/artifacts/lead08_reassessment/15_iter7_addr_row_tile.md`.
>
> The iteration-5 K=4 production probe holds 24 physical
> rows / 18 unique experts constant and varies only expert placement. Same-set permutation and
> slab-wide striding change total routed cost by just +1.9% and +0.7% versus contiguous; matched
> gate+up movement is below 2%. Expert remapping/packing cannot supply the required >=15% stage gain.
> Artifact: `issue468/artifacts/lead08_reassessment/13_iter5_address_locality.md`.
>
> The remaining cheap exactness fallback has also been measured. Threshold 0.25 misses
> the known batch-vs-exact argmax flip (margin 0.3755). Threshold 0.5 catches the one observed flip
> but guards 27/158 probed cycles and adds an optimistic 7.0 ms over all cycles, projecting the live
> stack from 40.04 to 36.37 t/s, below plain 38.16. One flip supplies no universal error bound, so
> the guard is neither proof-grade exact nor economically viable. Artifact:
> `issue468/artifacts/lead08_reassessment/12_iter4_margin_guard.md`.
>
> **Iteration 3 grouped gate+up prototype = NO-GO; branch stopped.**
> The one bounded expert-major/threadgroup-spill prototype is bit-exact against production across
> gate, up, weighted mid, and routed output (zero differing F32 bits; zero argmax flips), but fails
> the predeclared cost gate with the wrong sign. At the realistic 18-unique / 24-pair shape,
> production gate+up is **4.869 ms/layer** and the prototype is **5.986 ms/layer: 22.9% slower**,
> not >=15% faster. Excluding layer 0 strengthens the regression to **35.8%**. The 6/12-unique
> cases regress 68.1% / 55.6%; the 24-unique disjoint control is parity. Artifact:
> `issue468/artifacts/lead08_reassessment/11_iter3_grouped_gateup_prototype.md`.
>
> **Iteration 1 remains NO-GO, but its reported 4.4x magnitude is withdrawn.** M1x2 was warmed by
> its persistent selected-expert cache across three passes, while M=2 used a different uncached
> path. The structural diagnosis remains sufficient: M=2 computes 18 vs 12 token-expert streams
> in the test shape and holds both tokens' `yl` arrays/accumulators live, so simultaneous fusion is
> the wrong design. Do not reuse the derived 2.9x bound as a cold-vs-cold estimate. Artifacts 06/09.
>
> **Decision:** do not integrate or iterate on grouped IQ2XXS gate+up. A compact unique-expert map
> removes the disjoint control overhead, while heavier overlap gets sharply worse; activation and
> partial-sum spill plus barriers outweigh duplicate dequant/load savings. The final exact K=4
> `verify_ms(4) <= 50.5 ms` gate was not run because its prerequisite stage gate failed. Retain the
> env-gated code only as research instrumentation. Any future Lead 08 attempt requires a different
> mechanism, not another grouped gate+up variant.

Date: 2026-07-07 (refreshed 2026-07-18). Status:
**M3-bounded V14 exhausted -> V15 Stage A passed, Stage B pending; prior grouped, margin,
address-layout, production-geometry, and existing batch-family mechanisms remain NO-GO.**
This doc remains the Lead 08 provenance record. Phase A result:
`issue468/summaries/mtp_verifier_engineering_and_phaseA.md`; Phase B canonical
summary `issue468/summaries/lead08_phaseB_floor_clearance_verdict.md`.
**Two-phase: a cheap profiling gate first (DONE, passed), kernel work only if
the gate passes (it did — proceed to Phase B).**
Related to lead 06 but a distinct thesis: the verifier is above its *own*
bandwidth floor, independent of the redundant anchor decode.

> **2026-07-15 re-assessment (post-M3).** Milestone-3 delivered the runtime
> stack the Phase-B verdict assumed (anchor-reuse + GPU Metal drafter + STS) and
> the full DSpark stack now **beats plain by +4.9%** (40.04 vs 38.16 t/s). The
> re-assessment re-derives the prize from the LIVE M3 cycle (the pre-M3 "Prize
> at current acceptance" table below is superseded): the verify is now **89.5%
> of the cycle** (62.07 ms of 69.38 ms total at verify_n≈3.78), so a verify saving
> amplifies ~1:1 into throughput — **~10 ms off the verify = +20% over plain**
> (the prior "marginal" framing was wrong). The decisive gate before any kernel
> build is **sub-lead 1: the verify-cost-decomposition probe** (4 cheap
> measurements, NO novel kernel). See §"Re-assessment (2026-07-15)" below +
> artifacts `issue468/artifacts/lead08_reassessment/01_reorient_bar_redrivation.md`
> and `02_sublead1_fusion_prospect_and_probe.md`.

> **2026-07-12 refresh.** The milestone-2 model-on-Metal validation confirmed
> this lead is the **swing term**: recovering the ~19 ms verify headroom flips
> the model from 0.98× (below baseline) to ~1.26× (clears the +20% gate) at
> oracle acceptance with the runtime stack. The crossover measurement showed
> the *existing* batched primitive is only break-even, so a new fused kernel is
> required (not just `verify_suffix_tops`); the exactness measurement quantified
> the batch-vs-decode divergence (small, near-tie); and `decode2_exact` was
> ruled out as linear. These are folded into the rationale, Phase B item 6, and
> the success criteria below.

## Rationale

The bandwidth audit's verdict ("memory-bandwidth-bound") describes the binding
regime, not that the implementation achieves the bound. The dossier's own
measurements say it does not, by ~1.5–2×:

- **Wrong intercept.** Fitting the long-bench medians (K=3..6: 59.7/65.8/74.5/
  79.6 ms) gives verify(K) ≈ **40 ms fixed + 6.6 ms/K**. The slope is roughly
  the expert-union byte growth (physical). The fixed component is not: a verify
  pass's mandatory one-time traffic (one dense stream, ~16–20 ms of the 26 ms
  decode) should put the intercept *below* a decode. Decode itself sits at its
  achievable floor (~300 GB/s effective, tuned streaming/readahead); the batch
  verify path (self-described "production-shaped verifier *attempt*",
  `ds4.c:21117`) carries ~15–20 ms/cycle that is not bytes.
- **The floor is demonstrably reachable in-tree.** verify(K=2) measured at
  **27.3 ms** (code_4k bench, ≈1.05× decode — at the floor) and at **49.5 ms**
  (8k bench, decode2-exact path streaming weights twice by design). Same K, ~2×
  apart by code path: implementation, not physics, sets the current cost.
- M=2–6 is the untuned valley between the optimized decode path (M=1) and the
  compute-bound prefill path (363 t/s at large M).

What a fused kernel buys: dequant-once-apply-to-all-K (weight block stays in
registers/threadgroup memory across the M loop), collapsed per-layer
encode/launch/sync (61 layers × per-stage encodes), on-GPU argmax compare (no
per-position readback stalls), expert readahead overlapped with dense compute
as the decode path already does. What it cannot buy: the dense stream itself
and the expert-union bytes (the ~6.6 ms/K slope).

**Prize at current acceptance** (E[a|4]+1=3.175, S(4)=0.288, draft 10, decode 26),
fixed K=4:

| verify(4) | anchor-reuse accounting | shipped accounting |
|---:|---:|---:|
| 65.8 (today) | −0.9% | −18.9% |
| 55 | +14% | −4% |
| 45 | +32% | +7% |
| ~35 (byte floor) | +57% | +16% |

Three implications: (a) a floor-level verifier **revises the "acceptance-limited"
conclusion** — at verify(4) ≈ 40–45 ms the +20% primary gate clears at *current*
acceptance; (b) it is partially an *alternative* to anchor reuse, not only a
complement — even under shipped accounting a floor-level verify approaches the
gate, which matters if lead 01 falsifies reuse. A cheaper/flatter verify curve
also changes lead 02's scheduling economics (longer blocks cheaper, pruning less
valuable) — re-run that simulation against any new curve. (c) **The existing
batched primitive is insufficient on its own** (confirmed 2026-07-12): the
crossover measurement found `verify_suffix_tops` (66 ms at K=4) is only
break-even vs the short-circuiting sequential verify (~57 ms at model acceptance
E[a|4]≈2.2), because the sequential path stops at the first mismatch. So the
fused kernel's prize is hitting **~45 ms — below the sequential ~57 ms**, not
merely below the current 66 ms. That is why Phase B (a new kernel) is needed
rather than just reusing `verify_suffix_tops`.

## Re-assessment (2026-07-15, post-M3): the corrected prize + avenues to explore

Milestone-3 delivered the runtime stack the Phase-B verdict *assumed*
(anchor-reuse + GPU Metal drafter + STS) and the full DSpark stack now **beats
plain by +4.9%** (40.04 vs 38.16 t/s, full 176-entry corpus). This re-assessment
re-derives the Lead 08 prize from the **live M3 cycle** — the pre-M3 "Prize at
current acceptance" table above (draft=10/decode=26) is **superseded**. Full
detail in `issue468/artifacts/lead08_reassessment/01_reorient_bar_redrivation.md`
+ `02_sublead1_fusion_prospect_and_probe.md`.

> **2026-07-17 controlled-probe resolution.** Artifact 10 established that resident preparation
> already tracks unique experts while the physical-row gate+up kernel barely benefits from sparse
> doubletons. Artifact 11 then built the one allowed threadgroup-spill prototype. It is bit-exact,
> but regresses gate+up 22.9% at 18 unique / 24 pairs (35.8% excluding layer 0). The grouped branch
> therefore stops; the prior 6-8 ms estimate is falsified for this design. Artifact 13 separately
> held the same production work fixed while varying address locality. Total and gate+up sensitivity
> stayed below 2%, closing software-visible remapping/packing as the speculative coalescing mechanism.
> Artifact 14 then swept the production address kernel's SIMDgroup count. All alternates are
> bit-exact, but their inconsistent 1-3% effects are far below the >=15% stage gate; production
> therefore retains NSG 2. Artifact 15 swept the orthogonal rows-per-SIMDgroup tile at fixed NSG 2.
> Tiles 2/8 are bit-exact but +1.0%/+0.3% slower over eight total-path rounds; tile 8's apparent
> -2.5% profiled result
> disappears when layer 0 is excluded (+0.05%). Production retains tile 4 and geometry tuning stops.
> Artifact 16 then isolated cache residency. Cold SSD selected-address preparation costs 5.407
> ms/layer and exact replay removes 87.8% of total routed time, but that path cannot run with DSpark.
> The compatible mapped path changes only -0.04% on replay, so residency is not a Lead 08 lever.

### The live baseline (the bar to beat)

Measured from the M3 full-stack bench (`dspark_m3_bench/large_corpus/full_stack.jsonl`,
n=176): the current batch verifier (`metal_graph_verify_suffix_tops`) runs at
**62.07 ms/cycle at verify_n≈3.78 — 89.5% of the 69.38 ms cycle** (draft 6.64,
decode 0.67). The verify is now even more dominant than the Phase-B "~46% of
cycle" (M3 made draft+decode cheap). Reconciles with the Phase-B fit
`verify_ms(K) ≈ 40 + 6.6·K` (at K=3.78 ≈ 65 ms ✓ — the verify kernel didn't
change in M3).

### The corrected prize — a verify saving amplifies ~1:1 into throughput

Because the verify is 89.5% of the cycle, a saving X gives throughput gain
X/(69.38−X), stacking on +4.9%:

| verify saving X | over current full stack | vs plain (38.16) |
|---|---|---|
| 5 ms | +7.8% | +13.1% |
| 8 ms | +13.0% | +18.6% |
| **10 ms** | **+16.8%** | **+22.6%** ← clears **+20%** |
| 12 ms | +20.9% | +26.8% |
| 16 ms | +30.0% | +36.3% |

**Lead target: ~10 ms off the verify = +20% over plain.** The earlier "8–18% of
the verify = marginal" was wrong — it conflated % of verify with % of cycle.

### Avenues to explore — the two fusion mechanisms (tempered post-codex)

Traffic is weight-dominated (activation round-trips ~0.5 MB/token/layer,
negligible vs ~GB weights — codex confirmed the attention→FFN activation fusion is
sub-ms, ~5 MiB) → the register/cache reuse that matters is the **dequanted expert
weights**. Two mechanisms:

1. **Dequant-once-apply-to-all-tokens** (load-sharing / de-dup). Production
   preparation already loads each unique expert resource once, but the `addr` gate+up
   kernel still dispatches physical pair rows. At 18 unique/24 pairs its profiled stage
   is 97% of disjoint. The implementation test is now negative: the expert-major spill
   prototype is 22.9% slower at that shape and 56-68% slower under heavier overlap.
   Synchronization and spill traffic erase the theoretical
   traffic saving. Do not iterate on this grouped-kernel mechanism.
2. **Coalesced union access → decode bandwidth.** **SPECULATIVE — attribution
   unsupported (see "crux" below).** The fixed-work locality probe now falsifies the simple
   software-layout version: contiguous versus same-set permuted or slab-wide strided placement moves
   routed cost by <2%. A different hardware-level mechanism would need new evidence.

**Revised prize: no demonstrated deployable saving.** The bounded load-sharing design failed.
Speculative coalescing headroom beyond de-dup remains unestablished and is not a current build plan.

### The crux — why is the verify expert stream slow? (attribution UNSUPPORTED, codex-verified)

The prior "190 GB/s because scattered access" came from a confounded K sweep. Artifact 10
holds K and physical rows fixed; artifact 11 tests the resulting implementation hypothesis.
The grouped design loses despite exact arithmetic, so counters could diagnose the loss but cannot
rescue the failed predeclared gate. They are evidence for a genuinely different future mechanism,
not justification for iterating this prototype:

| hypothesis | decisive test | prize |
|---|---|---|
| sparse doubletons miss useful reuse in the pair-row kernel | controlled 18-vs-24 unique gate+up | confirmed, but grouped realization loses |
| grouped spill can convert that headroom into speed | bounded bit-exact prototype | falsified: +22.9% time at target shape |
| expert row order/placement causes a recoverable locality loss | fixed-work contiguous vs permuted vs strided production probe | falsified: <2% movement |
| production address-kernel threadgroup packing leaves occupancy headroom | fixed-work NSG 1/2/4/8 sweep with bitwise gate | falsified as material lever: all exact, only noisy 1-3% effects |
| output rows per SIMDgroup leave reuse/register headroom | fixed-work row-tile 2/4/8 sweep | falsified: exact alternatives have no steady-state or repeatable total gain |
| cross-cycle expert residency removes current verifier cost | identical-selection cold/replay on SSD and mapped paths | SSD upper bound -87.8%, but incompatible with DSpark; compatible path -0.04% |
| remaining cost is DRAM vs dequant/address work | Instruments after prototype miss | diagnostic only |
| a different hardware-level coalescing mechanism exists | counters: memory-bound + low L2 + poor coalescing, then a distinct controlled test | speculative; no current build |
| fundamental (batch-attn scattered KV / inherent dequant) | attention+dense share bounds it | TBD |

### Sub-lead 1 — decisive measurement gate: DONE

The K sweep and readahead tests bounded the stage but left K/unique/physical work confounded.
Artifact 09 exposed the invalid M1x2 cache ordering. Artifact 10 replaced it with a production K=4
batch path at fixed physical rows and controlled unique counts. Artifact 11 completed the bounded
follow-up: fidelity passes bit-exactly, but performance misses the >=15% gate with a 22.9% regression.
The grouped gate+up branch is complete and stopped. Down/sum6 remains deferred because it is a much
smaller term and cannot compensate for the failed prerequisite. Artifact 13 then completed the
separate controlled locality microbenchmark; it also failed the cost gate and stops address packing.
Artifacts 14/15 tested the remaining cheap production-kernel geometry knobs. SIMDgroup-count effects
stay within about 3%; alternate row tiles are slower total and parity/slower at steady state. The
defaults remain NSG 2 / row tile 4 and threadgroup repacking/tiling is stopped.
Artifact 16 then bounded cross-cycle residency: a large exact SSD-cache replay prize exists, but the
current DSpark runtime rejects SSD streaming and its compatible mapped path is already replay parity.
That mechanism moves to Lead 05/runtime scope rather than authorizing another Lead 08 build.

### The greedy-exactness secondary (unchanged from Phase B item 6)

Independent of the cost question: the fused kernel's reduction must match M=1
bit-for-bit (gate/up already bit-identical via the shared `_impl`; the divergence
is down/sum6 + attention). Fidelity gate: `max_abs==0` vs M=1 + 0 argmax flips
on the exactness corpus (temp=0). The single-stage expert prototype proves the
mechanism on one stage; **full** greedy-exactness needs all divergent stages
fused (the full build). Keeps the Phase-B abort condition (>5% re-verification).

## Content of work

**Phase A — profiling gate (1–2 days, no kernel code, start anytime):** DONE.

1. Split the verify cycle per stage with `DS4_METAL_GRAPH_TOKEN_PROFILE=1` /
   `DS4_METAL_LAYER_STAGE_PROFILE` (encode vs execute vs readback), K=2..6,
   same protocol as `run_mtp_verifier_bench_long.py`.
2. Capture achieved GB/s during verify vs decode (Instruments / GPU counters).
3. Log per-cycle expert-union sizes (shared instrumentation with lead 05) and
   compute the true byte floor: dense-once + union-experts + KV at the measured
   effective bandwidth.
4. Deliverable: a measured headroom number — `verify_ms(K) − floor_ms(K)` — and
   its decomposition (encode overhead / bandwidth inefficiency / readback).
   *(2026-07-12 note: Phase A delivered the headroom number — 19–22 ms/cycle at
   K=3..5 — but the encode-vs-bandwidth-vs-readback decomposition is still
   partially open; useful to finish for targeting Phase B, but the 19 ms number
   alone sets the target.)*

**Phase B — kernel work (weeks, Metal-specific):**

5. Fused dequant+GEMM micro-batch kernels (M=2..6) for the dense Q8_0 and
   IQ2_XXS expert paths, M-inner-loop over resident weight tiles; single
   command buffer per layer group; on-GPU suffix argmax compare.
6. **Exactness strategy — the actual hard part.** The fused kernel must be
   **both sublinear AND greedy-exact** — neither existing primitive qualifies:
   `verify_suffix_tops` is sublinear but flips greedy tokens, and
   `metal_graph_verify_decode2_exact` is exact but **linear** (runs two full
   decodes, ~2× decode, no amortization — correctness-only, not a verifier).
   Either (preferred) one kernel family serving both decode (M=1) and verify
   (M=K), making spec output equal target-only output *by construction*; or a
   margin-guarded fallback (re-verify near-ties with the exact path). The
   **2026-07-12 exactness measurement**
   (`issue468/artifacts/rejection_acceptance/verify_dist_probe_exactness.jsonl`)
   quantified the batch-vs-decode divergence: median TV 0.0035, **argmax flip
   rate 0.64%** (1/156 positions). Artifact 12 later disproved the inference that
   fallback frequency equals flip frequency: threshold 0.5 catches the observed
   flip but guards 17.1% of cycles and adds at least 7.0 ms/cycle. The margin
   fallback is now closed negative; it is not an exactness solution. Note:
   the single-family option changes baseline decode numerics; the exactness
   gate (spec == target-only, same build) still holds, but re-baseline the t/s
   denominator.
7. Measure with the retained bench protocol; feed the new verify curve back
   into the lead 02 simulation and the speedup model.

## Success criteria

- **Phase A gate (MET):** measured headroom at K=3..5 ≥ ~15 ms/cycle (verify
  demonstrably ≥1.4× above its byte floor). Came in at 19–22 ms/cycle → proceed.
- **Phase B target:** verify(4) ≤ ~45 ms on the 8k corpus (from 65.8), K=2
  batch-exact ≤ ~30 ms (from 49.5), with exact greedy output preserved on the
  full exactness corpus (hard requirement).
- **Net effect:** fixed-K=4 ≥ +25% with anchor reuse (or ≥ +5% under shipped
  accounting if lead 01 falsified reuse) at current acceptance — i.e., the
  kernel moves the primary gate from "needs a better drafter" to "needs no
  acceptance improvement." It still requires the **concrete runtime stack**
  (anchor reuse + GPU drafter, both engineering, not speculative) but **not** a
  better drafter / higher acceptance. The 2026-07-12 swing-term re-derivation
  independently confirms: verify(4) 66→47 ms flips the model 0.98×→~1.26×.
- **Abort condition:** exactness cannot be preserved by either strategy without
  reintroducing per-position exact re-verification on >~5% of positions (eating
  the gain) — record as the numerics falsification of the fused approach.

## Next steps

**2026-07-18 current:** corrected U1 closes arithmetic/dequant below 4.74 ms, and the iteration-25
mechanism audit finds no independent concrete >=8.7 ms package on the current in-RAM runtime.
Lead 08 `INTERIM_BOUNDED` is exhausted. V15 Stage A now passes both Q8/F16 M=2 capability gates:
430/430 all-layer C0-C4 cases are word-exact and source/AIR retain one logical traversal. Stage B
is next: M=2..8 at all seven sites, a nonredundant output-B seam, C5 scope, and an M=4 all-layer
candidate-minus-current-ext delta upper bound <=7.5 ms. No graph integration is authorized before
that passes. Lead 05 remains an alternate only if SSD compatibility is in scope.
Do not repeat grouped, margin,
address-layout, geometry, mapped-residency, existing batch-family, or top-r mechanisms without evidence that
changes their upper bounds.

**Milestone COMPLETE (2026-07-13): Phase B characterization + decisive measurements →
FINAL verdict = NO-GO via swaps → bounded build attempt with a hard exit gate
(codex-gated A+B+C; propagated).** Canonical summary:
`issue468/summaries/lead08_phaseB_floor_clearance_verdict.md`. STATUS.md +
spec_speedup_model.md updated.

**Historical gated follow-up (still unbuilt; prior completion wording was too broad):** attempt the **sublinear bit-exact
batch-path build** (HC/compressor/attention on decode reductions + batched load sharing) as
a BOUNDED effort with a **hard exit gate** — GO is unconfirmed until an end-to-end K=4
bit-exact verifier profiles `verify_ms(4) ≤ 50.5 ms`. NOT "gate cleared / commit as
sufficient." (Confirmatory: a literal identical-input MoE kernel-equality harness; a stable
verify-bandwidth slope re-measure.)

## Worklog

### 2026-07-18 - iteration-28 V15 Stage-A exact small-M M=2 -> PASS capability

Implemented one research-only Q8/F16 host operation and shared M-token structural Metal template,
with no production graph callsite changes. Q8 Q-a retains M1 NR0 2 / NSG 4 and the per-token
eight-Q sum then scale boundary; F16 router retains M1 NR0 2 / NSG 8 and four half4/float4 dots.
The candidate dispatch is one `(ceil(N/2),1,1)` grid; raw weights are loaded outside the two token
arithmetic chains. A dedicated filtered harness compares against two unchanged production M1 calls
in the same command buffer and emits explicit CSV evidence.

The retained decisive run covers 43 layers x two formats x C0-C4 = 430 unique cases. The
deterministic validator reconstructs every input and reports 215/215 cases per format, 86/86 per
corpus, zero differing F32 words, and zero max absolute difference. Runtime Metal compilation and
`metal -S` both pass. AIR contains one primary Q8 block-loop backedge and one primary F16 block-loop
backedge, with raw-weight loads before token arithmetic and no token term in weight addresses. This
is logical compiled-source evidence, not proof of physical DRAM transactions. Stage A establishes
capability only; Stage B's wider family, seam, and economics remain open. Artifact:
`artifacts/lead08_reassessment/36_iter28_exact_smallm_pair.md`.
The first red-team returned `FIX` because filtered subsets could exit as decisive `BIT_EXACT` and
odd N was unsafe under unconditional NR0 2. Both now fail closed; the repeat independently rebuilt,
reproduced 430/430 exact cases and fresh AIR, and returned `COMMIT`. Stage B is authorized.

### 2026-07-18 - iteration-27 V15 executable protocol -> REDESIGN / staged GO

Independent preflight rejected a monolithic both-format M=2..8 build plus seam refactor and timing.
Stage A is the only authorized implementation: one research-only API with both Q8 Q-a and F16
router M=2 specializations, literal M1 arithmetic, all 43 real offsets, C0-C4 bit gates, and
source/AIR proof of one logical weight traversal. No graph callsite changes. Only an audited
both-format pass admits Stage B: all seven sites at M=2..8, captured C5 inputs, a nonredundant
batch-low/output-B seam, and a balanced all-43 cumulative timing carrier whose delta upper bound
must be <=7.5 ms. Artifact:
`artifacts/lead08_reassessment/35_iter27_exact_smallm_protocol.md`.
The first red-team returned `FIX` on C5/output-B capture scope, M4 timing scope, deterministic input
constructors, Q8 reduction order, common-family structure, topology outcome, and numeric timing
validity; the protocol was corrected for repeat audit.
The repeat returned `FIX` on ledger C5 scope, format-specific AIR language, Stage-B reproduction,
validity-bootstrap details, and C3 site/sign construction; those were corrected for final audit.
The final repeat audit returned `COMMIT`.

### 2026-07-18 - iteration-26 cross-quant exact small-M inventory -> DESIGN COMPLETE / conditional V15

Returned to the primary exact-output track after bounded-track exhaustion. Independent preflight
rejected an F16/router-only patch and required one family spanning all seven exposed dense sites:
four Q8 projections and three F16 projections. Their weights total 77.875 MiB/layer (3.270 GiB per
43-layer traversal); four standalone M1 rows add 9.810 GiB, so row helpers cannot be production.
Literal M1 Q8/F16 bodies and integration seams make a load-once family structurally plausible, but
output-B needs a nonredundant batch seam and no implementation/economic result exists. V13 closes design-positive;
V15 conditionally admits one common Q8/F16 M=2..8 capability prototype, gated on direct bit equality
for all seven shapes and a candidate-minus-current-ext all-layer delta upper bound <=7.5 ms over the
conservative 43 ms policy base. Iteration 27 must predeclare the executable fixed-work timing protocol
before any code. The first red-team returned `FIX` on floor/lower-bound wording, protocol completeness,
output-B mechanism neutrality, four source lines, and stale family-closure wording; all were corrected
for repeat audit. The repeat returned `FIX` only on possible lower-bound double counting; the final
formula now permits one validated end-to-end lower bound or a sum of proved non-overlapping component
bounds, and policy rollout requires cumulative matched timing rather than additive isolated deltas.
The final repeat audit returned `COMMIT`.
`artifacts/lead08_reassessment/34_iter26_exact_smallm_dense_inventory.md`.

### 2026-07-18 - iteration-25 bounded-track reassessment -> STOP / EXHAUSTED

Independent preflight audited every mechanism after U1. None is both independent of U1 and backed
by a fresh composable >=8.7 ms upper bound. Close the current in-RAM M5 bounded track. Redirect to
Lead 05 only if SSD compatibility is in scope, or resume deferred V13/V5 exact work without a speed
promise. Artifact: `artifacts/lead08_reassessment/33_iter25_bounded_track_exhaustion.md`.

### 2026-07-18 - iteration-24 optimistic packed-weight floor -> economic STOP

U1 retains live mapped IQ2 q/scale loads, ID-dependent addressing, production geometry/SIMD
reduction/stores, while intentionally removing activation, LUT/barrier, dequant, and dot work. All
256 timing samples retain 43 intervals; sentinel-based 86-case duplicate checks and AIR gates pass.
After discarding a first capture that omitted the production first-row weight offset, the corrected
paired saving is 4.708104 `[4.683041,4.724125]` ms on overlap and 4.723667
`[4.715042,4.735250]` ms on dispersed IDs. Even this nonsemantic optimistic floor misses 8.7 ms,
so arithmetic/dequant reduction closes without tuning. Artifact:
`artifacts/lead08_reassessment/32_iter24_weight_floor.md`. The corrected repeat audit reproduced
address-faithful AIR, all controls, byte-identical CSV generation, and returned **COMMIT**.

### 2026-07-18 - iteration-23 direct mapped-pair GPU replay -> VALID positive selector

**Preflight and implementation.** The independent preflight returned REDESIGN for a final
control-stabilized separator. The harness directly replays the existing mapped K4 pair encoder and
returns after gate/up, committing one layer per command buffer. Metal GPU start/end timestamps are
primary; activation, down, stage profiling, and per-layer timed logging are excluded. Literal
production, identical production duplicate, companion actual-zero, R32, and R128 run across two
fixed-ID strata after 20 warmup rounds and 30 measured balanced rounds.

**Result.** All four nonbaseline selectors pass 86/86 full-routed plus explicit-direct bitwise cases.
All 300 samples contain 43 positive finite GPU intervals and every arm occupies every position six
times per stratum. Duplicate/production paired intervals are `[-0.064,+0.099]%` and
`[-0.071,+0.154]%`; zero/production intervals are `[+1.149,+1.341]%` and
`[+1.054,+1.195]%`. R32 versus zero is +45.762% `[+45.652,+45.934]` on overlap and
+45.795% `[+45.660,+46.014]` on dispersed IDs; R128 is about +222%. AIR preserves the separate
runtime argument, two FMA calls, and dynamic backedge only in the companion.

**Decision.** Close T2c **positive as a one-sided arithmetic/issue selector**. It neither proves
production dequant-compute dominance nor authorizes a production kernel. It authorizes exactly one
preflighted U1 maximum-removable arithmetic upper bound. Direct production is only about
10.73-10.76 ms across all 43 layers, so the unchanged 8.7 ms/cycle prize requires approximately 81%
removal; U1 must meet that absolute lower-bound gate in both strata or close arithmetic reduction.
Canonical artifact and retained layer/summary CSVs:
`artifacts/lead08_reassessment/31_iter23_direct_gpu_replay.md`. The independent audit regenerated
the layer, sample, and summary CSVs byte-identically, reproduced fidelity/AIR/statistics/economics,
confirmed the one-sided claim limit, and returned **COMMIT**.

### 2026-07-18 - iteration-22 mapped tiny-pair ALU separator -> INVALID / near-threshold

**Preflight and implementation.** Independent challenge returned REDESIGN: the retained-FMA test is
one-sided and a weak response cannot select traffic/latency. A separate mapped companion pipeline
preserves literal production selector 0 and holds K4, 24 pairs, 18 unique experts, mapped offsets,
NSG 2, NR0 4, F32 gate/up, activation/down work, and dispatch counts fixed. Two ID strata, all 43
layers, explicit dispatch identity, 20 balanced samples per arm, and dumped-source AIR gates replace
iteration 21's layer-0/incompatible-carrier limitations.

**Result.** Every companion arm passes 86/86 bitwise cases across gate/up/weighted-mid/routed output
and argmax. AIR retains two runtime dependent FMAs and the backedge only in the companion. R32 adds
+26.00% synchronized gate/up on overlap and +27.05% on dispersed IDs, with consistent positive wall
movement. However, three of four production/companion-zero paired intervals escape the predeclared
+/-2% band; the overlap R32 lower interval endpoint is also 24.958%, below the 25% positive gate.

**Decision.** Close this T2b measurement design **INVALID**, retaining the strong response only as
near-threshold diagnostic evidence. It authorizes neither arithmetic reduction nor a traffic
prototype. T2c, a freshly preflighted control-stabilized direct replay of the same mapped pair
carrier, is P0; stop the separator family if that replay cannot pass control. The measured gate/up
stage is about 18.4 ms/cycle, so a selected mechanism must credibly remove about 47% of the whole
stage to supply the required 8.7 ms prize. Canonical artifact:
`artifacts/lead08_reassessment/30_iter22_mapped_alu_separator.{md,csv}`. The independent audit
reproduced the schedule, 86-case fidelity matrix, dispatch/AIR proof, statistics, invalid gate, and
economic calculation and returned **COMMIT**.

### 2026-07-18 - iteration-21 selected-address ALU separator -> T2a AMBIGUOUS

**Preflight and implementation.** Independent challenge narrowed the claim from dequant/FMA binding
to ALU headroom. A research-only companion address kernel adds two dependent runtime identity-FMA
chains for rounds 8/32/128 while preserving bytes, addresses, 24 rows, geometry, reductions, and
outputs. Literal production plus a companion-zero arm bound specialization overhead. The existing
K4 harness supplies a layer-0 bitwise gate/up/mid/output/argmax pattern and 10 samples over all 43
layers under rotated/reversed order balance. Dumped runtime source compiled with `metal -S` retains both FMAs and the dynamic
backedge; the production specialization has zero probe-field loads or probe instructions.

**Result.** Production and companion-zero differ by -1.14% total/-0.19% gate/up, passing the 2%
control. Against companion-zero, rounds 32 add +5.41% total/+10.10% synchronized gate/up; rounds 128
add +23.72%/+51.38%. All companion arms are bitwise identical to production on the layer-0 pattern.
The response is monotonic, but rounds 32 fall in the predeclared 5-25% ambiguous band and 8-round
intervals include zero. It does not distinguish DRAM, cache, addressing, occupancy, unexcluded
control/codegen effects, or dequant cost.

**Scope correction and decision.** The initial mapped-path run was discarded because it never
dispatched the address kernel. The valid result is exclusive to the SSD selected-address carrier,
which cannot compose with DSpark. Close T2a as ambiguous evidence only; keep V14
unauthorized. A follow-up trace identifies all 688 mapped K4 stage records as `tiny_pair_mv`; T2b
on its `kernel_mul_mv_id_iq2_xxs_pair_f32` family becomes P0. Artifact and compact statistics:
`artifacts/lead08_reassessment/29_iter21_ssd_addr_alu_headroom.{md,csv}`.
The initial audit rejected the control and fidelity wording; after adding a literal-production arm,
a companion-zero arm, balanced rotations, and precise layer-0 scope, the repeat audit rejected only
an unsupported nonlinear-response claim. The corrected monotonic, approximately linear but
threshold-ambiguous classification passed the final repeat audit: **COMMIT**.

### 2026-07-18 - iteration-20 Metal counter capability gate -> tooling BLOCKED

**Preflight and contract.** Independent challenge returned **REDESIGN**: compare an identical
untraced fixed-K4 M3 control and Metal System Trace; require useful populated DRAM/cache, ALU, or
occupancy/stall counters, resolved shader intervals, and <=5% steady verify perturbation. K1/K4
same-path scaling and raw-M1 context were contingent on this capability gate.

**Result.** The default `Metal GPU Counters` profile is unsupported. Metal System Trace records
822,505 counter values, but its only counter is `RT Unit Active` and every value is zero. It resolves
119 ds4 shader names but exports zero shader interval rows. The steady 21-cycle K4 verify median moves
62.082 -> 66.754 ms (+7.526%); throughput moves 37.803 -> 35.703 t/s (-5.556%). Both the
discrimination and <=5% perturbation gates fail.

**Decision.** Make no bandwidth, compute, occupancy, or hot-kernel claim. Do not run K1/K4 under the
same template. T1 becomes `BLOCKED`; a custom GUI template with a supported M5 counter set may reopen
it. T2 same-kernel arithmetic-intensity separation becomes P0 and requires a fresh preflight before
implementation. The large trace/XML exports remain temporary; compact result and reproduction:
`artifacts/lead08_reassessment/28_iter20_metal_counter_capability.{md,csv}`. Mandatory audit is
**COMMIT** after independently reproducing the warning, counts, medians, and deltas.

### 2026-07-18 - iteration-19 bounded-divergence contract and ledger reset

**Decision.** The user accepts M3's interim tradeoff for Lead 08: task-quality non-regression at
temperature 0 and close batch-versus-exact distributions above temperature 0. This does not replace
the project's exact-output success criterion. The independent challenger returned **REDESIGN, then
PROCEED**, requiring fresh same-binary controls rather than unmatched historical comparisons.

**Reference/candidate.** Every V14 experiment uses fresh plain and frozen-M3 controls. Candidate is
the same M3 composition with exactly one verifier mechanism changed; exact sequential verification
is the fixed-frontier quality anchor. Scheduler, acceptance, `verify_n`, expert work, correction
tokens, and state/restore integrity are reported so speed cannot be attributed silently to less work.

**Quality invariant.** On 92Q, candidate score and same-verdict agreement must be at least both the
fresh M3 values and historical 61/92 and 83/92; plain-PASS to candidate-FAIL changes must be no more
than both fresh M3 and historical 4. The distribution probe runs candidate and M3 batch paths
non-committing beside exact sequential verification and hard-fails unless their exact-owned
trajectories and keyed cycle/row work match. Candidate metrics must be no worse than fresh M3 and
the historical envelope: flip rate <=1/156, mean / median / p90 / max TV <=0.0104 / 0.0035 / 0.0308 /
0.104, compared-cycle rate above TV 0.05 <=3/90, mean/max KL <=0.0022/0.054, and max absolute logit
delta <=4.56. Committed-token divergence is reported and investigated if worse than M3, but is not
an admission metric under the user-defined temperature-0 functional gate.

**Economics and outcome tree.** T1 first captures warm fixed-K4 M3 verification against M1 decode
with Metal counters. Bandwidth/cache, ALU/dequant, occupancy/barrier, or dispatch/fusion evidence
selects exactly one V14 mechanism. Ambiguous/unavailable counters redirect to a controlled separator,
not an attribution claim. A prototype needs >=15% on the selected hot stage and a credible >=8.7
ms/cycle composed saving. Integration requires `verify_ms(4)<=50.5 ms`, >=8.7 ms/cycle full-stack
saving, and `>= max(45.8 t/s, 1.20 x fresh plain)` on the 176-entry corpus with all quality gates
passing.

**Ledger reassessment.** V1 is `INTERIM_BASELINE`; V5 and V13 are deferred exact-track work; V6-V12
remain valid retained evidence. Previously failed grouped, margin, layout, geometry, mapped-residency,
existing batch-family, and top-r branches remain closed. D5 is only a conditional later composition. T1 is
P0 and V14 is the active rank-1 performance branch. Canonical contract and rationale:
`artifacts/lead08_reassessment/27_iter19_bounded_divergence_reframe.md`. Mandatory post-iteration
audit first returned **NO-COMMIT** on four contract holes. After making fresh-M3 quality gates
conjunctive, enforcing an exact-owned paired trajectory, requiring both the absolute and relative
throughput bars, and resolving committed-token divergence as a report-only metric, repeat audit
returned **COMMIT**.

### 2026-07-18 - iteration-18 V12 contract: row-wise HC FFN mixer

**Hypothesis and preflight.** V11 establishes exact FFN HC residual/flat input and 14/24 differing
F16 `hc_ffn_fn` outputs. The challenger returned **PROCEED** for the cheapest causal falsifier: reuse
the existing F16 rows-as-M1 helper only at this `16384 -> 24` projection. No split, normalization,
router, expert, attention, drafter, or output-head path changes.

**Reference/candidate.** Both arms retain V6-V10 at layer 0. The candidate additionally sets
`DS4_LEAD08_ROWWISE_HC_FFN_MIX_LAYERS=1`; the numeric cap is parsed, saved, set, announced, and
restored only inside multi-token suffix verification. Correctness compares the M=K row at position
104 with its restored batch-M1 replay. Existing branch-neutral router dumps extend the capture only
through logits/probabilities/top-k/weights; lowercase `ffn_out` and `ffn_shexp` remain unrequested.

**Outcome tree.** Upstream residual/flat must remain exact. Mixer, all split state, `hc_ffn_pre`, and
`ffn_norm` exact is a V12 causal pass. A remaining mix difference fails activation/helper/layout;
exact mix with differing split redirects to split/sinkhorn; exact split with differing pre redirects
to weighted sum; exact pre with differing norm redirects to RMS weight norm. Exact norm with differing
router logits exposes the F16 router projection; exact logits with later router-state differences
redirects to router select. A final-logit flip is not a V12 failure if the local frontier moves.

**Economics and evidence.** Correctness dumps are untimed. If retained, timing is restricted to
balanced process-first layer-0 `part=ffn stage=hc_pre` measurements with dumps/probing off. That local
envelope excludes expert routing and can describe only this naive row-loop realization, not production
verifier cost or the 50.5 ms bound. Omit all-layer timing because downstream state/routes diverge.
After V12, if the router projection is next, rank a generic same-accumulation batched-F16 design/cost
inventory above another reflexive row helper. Retain commands, activation/path counts, stage evidence,
challenger result, and mandatory post-iteration audit before commit.

**Result.** The candidate makes `ProdFFNHCMix`, all split state, `hc_ffn_pre`, and `ffn_norm`
bit-identical for the captured row. The first remaining difference is the F16 router projection:
186/256 logits differ, while top-6 expert IDs remain exact. Four balanced synchronized layer-0
`ffn/hc_pre` pairs all favor the row loop by 0.0293 ms mean, but this is diagnostic-only and not an
all-layer or production cost. Close V12 positive. Per preflight, rank a generic same-accumulation
batched-F16 design/cost inventory above another row-wise projection. The mandatory audit independently
reproduced all F32/I32 boundaries and timing arithmetic, validated scope/attribution, and returned
**COMMIT**.
Canonical artifact: `artifacts/lead08_reassessment/26_iter18_rowwise_hc_ffn_mix.md`.

### 2026-07-18 - iteration-17 V11 contract: FFN HC input localization

**Hypothesis and preflight.** V10 makes `hc_attn_post` exact but leaves `hc_ffn_pre` as the first
captured difference. The challenger returned **PROCEED** for capture-only localization and rejected
an immediate row-wise FFN mixer. The complete causal cone is the post-attention HC residual
(`16384`) -> plain row RMS (`16384`) -> F16 `hc_ffn_fn` mix (`24`) -> split state (`24`) ->
`hc_ffn_pre` (`4096`).

**Reference/candidate.** This iteration changes no compute. The existing restored-frontier comparator
retains V6-V10 at layer 0 and compares the M=K row at position 104 with its batch-M1 replay. New
case-sensitive aliases `ProdFFNHCResidual`, `ProdFFNHCFlat`, `ProdFFNHCMix`, and `ProdFFNHCSplit`
do not contain the lowercase legacy names that alter FFN branch selection. Flat and mix are
immediately post-producer; split follows its fused producer. Residual is retained at the FFN-entry
consumer boundary after read-only RMS and corroborated by producer-adjacent `hc_attn_post`.
Directional FFN steering remains disabled.

**Outcome tree.** A residual difference invalidates the V10/replay contract. Exact residual with a
flat difference implicates plain RMS; exact flat with a mix difference makes a row-wise FFN F16
mixer the next bounded falsifier; exact mix with a split difference redirects to split/sinkhorn;
exact split with differing `hc_ffn_pre` redirects to weighted sum. A synthetic combined FFN block is
not captured: the optimized path consumes routed F32 plus shared F16/F32 directly, and forcing
`ffn_out` would materialize a sum and change the branch.

**Evidence and economics.** Retain the exact command, activation/path counts, and compact stage CSV.
This is correctness-only instrumentation with no timing claim or change to the 50.5 ms gate. Run the
mandatory post-iteration audit before commit.

**Result.** `hc_attn_post`, `ProdFFNHCResidual`, and `ProdFFNHCFlat` are bit-identical.
`ProdFFNHCMix` is the first difference at 14/24 values, max 1.52587891e-5. Split state then differs
in 16/24 values and `hc_ffn_pre` in 752/4096, both inherited from the mixer. The run has 64
batch-M1 and zero raw-M1 evaluations; no legacy FFN path-debug name was requested. Close V11 as a
localization result and rank V12 row-wise FFN HC mixing next. No timing claim. Post-iteration audit
initially returned **NO-COMMIT** only because the residual alias was incorrectly described as
producer-adjacent. After correcting it to the FFN-entry consumer boundary, repeat audit returned
**COMMIT**. Canonical artifact:
`artifacts/lead08_reassessment/25_iter17_ffn_hc_input_localization.md`.

### 2026-07-18 - iteration-16 V10 contract: row-wise HC attention mixer

**Hypothesis and preflight.** V9 established exact normalized input but 14/24 differing F16 mixer
outputs. Metal dispatches `hc_attn_fn` (`16384 -> 24`) through the ordinary F16 matvec for M=1 and
the low-K `mul_mv_ext` kernel for M=2..8. The challenger returned **PROCEED** with a dedicated F16
row helper; the existing Q8 helper is not applicable.

**Reference/candidate.** Both arms retain V6-V8 at layer 0. The candidate additionally sets
`DS4_LEAD08_ROWWISE_HC_ATTN_MIX_LAYERS=1`, creating row-contiguous F32 input/output views and calling
the existing F16 primitive with one token per row. The numeric cap is saved/restored only inside
multi-token suffix verification. It applies only to attention `hc_attn_fn`, not the FFN mixer,
DSpark drafter, split/sinkhorn, normalization, attention, expansion, or output head.

**Exactness and outcomes.** `ProdHCFlat` must remain exact and `ProdHCMix` must become exact. If
consumed `ProdHCSplit[4:24]` and `hc_attn_post` also become exact, V10 passes and the first downstream
boundary is recaptured. Exact mix with differing split redirects to split/sinkhorn. Exact expansion
inputs with differing HC post reopens expansion despite its row-independent source. A remaining mix
difference with confirmed activation fails the helper or M1-family assumption. Final-logit flips do
not fail V10 if the local frontier moves.

**Economics and evidence.** Correctness dumps are untimed. Supporting process-first K=4 pairs hold
all prior caps fixed and compare mixer `0 -> 1` and `0 -> 43`, without dumps/probing. The all-layer
row loop repeats a roughly 0.75 MiB matrix per token/layer and is a naive implementation upper bound
with profiler and downstream-routing confounds; it cannot update the 50.5 ms feasibility lower bound.
Retain exact commands, stage/timing CSVs, activation logs, challenger result, and mandatory
post-iteration audit before commit.

**Result.** The candidate makes `ProdHCMix`, all consumed split state, and `hc_attn_post`
bit-identical for the captured row. The first captured downstream difference moves to `hc_ffn_pre`.
Four matched all-layer timing pairs have a -1.252 ms mean / -1.259 ms median layer-execution sign,
but changed accepted-token patterns and trajectories make that sign route-confounded and
uninterpretable for mixer economics. It shows only no obvious penalty in these instrumented runs.
V11 must capture FFN HC flat/mix/split inputs before any analogous compute change. The first audit
validated the implementation, correctness evidence, arithmetic, and V11 ordering but returned
**NO-COMMIT** on the timing attribution plus two ledger/date inconsistencies. Those findings were
corrected and the repeat audit returned **COMMIT**. Canonical artifact:
`artifacts/lead08_reassessment/24_iter16_rowwise_hc_attn_mix.md`.

### 2026-07-17 - iteration-15 V9 HC expansion inputs -> REDESIGN / audit COMMIT

**Preflight redesign.** The challenger rejected immediate row-wise HC expansion. M=1 and M=K use
the same `kernel_dsv4_hc_expand4`; token rows are independently indexed and the four-term
accumulation order is unchanged. Exact `hc_attn_pre` did not prove the full 24-value HC split row
was exact, although expansion later consumes its post gate `[4:8]` and combination matrix `[8:24]`.

**Cheapest falsifier.** With V6-V8 enabled at layer 0, capture branch-neutral `ProdHCBlock`
(post-steering output-B), `ProdHCResidual` (`batch_cur_hc`), `ProdHCFlat` (RMS-normalized mixer
input), `ProdHCMix` (all 24 mixer values), and
`ProdHCSplit` (all 24 split values) immediately before expansion. This is correctness-only and
changes no compute. If block, residual, and split `[4:24]` are exact but output differs, only then
test row-wise expansion and audit layout/races. If mix differs, return the frontier to the F16 HC
mixer projection. If mix is exact but consumed split differs, isolate split/sinkhorn. If residual
differs at layer 0, invalidate the prior input assumption. Retain dumps, exact command, preflight,
and mandatory audit before commit.

**Result.** The rebuilt V2 capture finds `ProdHCBlock`, `ProdHCResidual`, and normalized
`ProdHCFlat` bit-identical. `ProdHCMix` is the first exact-input/different-output boundary (14/24,
max 6.10e-5); consumed `ProdHCSplit[4:24]` differs 15/20 and `hc_attn_post` differs 6,269/16,384.
The JSON is status-ok with 64 emitted tokens, 27 cycles, 64 batch-M1 and zero raw-M1 evaluations;
the known single flip remains across 37 compared rows. HC expansion is not independently implicated
while its split input differs. V10 is row-wise F16 HC mixer projection, followed by mix/split/post
recapture. The first audit validated all binary/CSV values and this ordering but returned
**NO-COMMIT** until the worklog and stale-safe reproduction command were corrected.
The collision-safe reproduction fix passed repeat audit: **COMMIT**. Canonical artifact:
`artifacts/lead08_reassessment/23_iter15_hc_input_localization.md`.

### 2026-07-17 - iteration-14 V8 attention output-B -> CAUSAL GO / audit COMMIT

**Hypothesis and preflight.** With V6/V7 exact inputs, layer-0 output-A low projection should already
be M-invariant because Metal uses the same token/group-independent direct Q8 kernel below 32 rows.
The following output-B generic Q8 matmul changes between M=1 and multi-row kernels and is the likely
first difference. The challenger returned **PROCEED**: first capture branch-neutral `ProdOALow` and
`ProdOAOut`; only if low is exact and output differs, overwrite output-B rowwise.

**Reference/candidate and invariant.** Both arms set Q/KV and Q-b caps to one. The candidate sets
`DS4_LEAD08_ROWWISE_ATTN_OUT_B_LAYERS=1`; after the unchanged combined batch output call, it uses
the existing named M1 Q8 primitive once per row to overwrite only `batch_attn_out`. The cap is
saved/restored only inside multi-token suffix verification. Prompt prefill, the M1 comparator,
output-A, HC expansion, FFN, and the output head are unchanged. At position 104, `ProdOALow` must be
exact in both arms; `ProdOAOut` must become exact only in the candidate. `hc_attn_post` identifies
the next boundary.

**Economics and outcomes.** `N=1` is causal. `N=43` is diagnostic-only because it executes batch
output-B and then repeats output-B rowwise; its delta is redundant-work upper-bound evidence with
downstream-routing confounds, not a feasibility lower bound. If low differs, redesign around the
single-row low API before touching output-B. If low is exact and output-B becomes exact, V8 passes.
If output-B remains different, fail or debug the overlay. A successful F16 hook or directional
steering makes this Metal experiment ambiguous. Retain branch-activation stderr, stage CSV, exact
commands, any timing as supporting-only, preflight result, and mandatory post-iteration audit.

**Result.** `ProdOALow` is already exact; the row-wise overlay makes `ProdOAOut` exact (3,319/4,096
differences to zero) and exposes `hc_attn_post` as the first captured unresolved boundary. The same
cycle-24 flip remains; max error changes 0.133129 -> 1.00982 as the downstream M=K batch-error
pattern changes, while the canonical M1 replay/generated trajectory is unchanged. The redundant
all-layer overlay observes a confounded +6.990 ms and cannot update the feasibility lower bound.
The post-iteration audit returned **COMMIT**. Canonical artifact:
`artifacts/lead08_reassessment/22_iter14_rowwise_attn_out_b.md`.

### 2026-07-17 - iteration-13 V7 row-wise production Metal Q-b -> CAUSAL GO / audit COMMIT

**Hypothesis and mechanism.** V7 tests only whether the layer-0 `Qcur` difference exposed by V6
originates in the Q-b Q8 projection changing from the M=1 to the multi-row reduction. The
pre-experiment challenger returned **REDESIGN**: Metal's
`ds4_gpu_attn_q_b_f16_head_rms_rope_tail_tensor` is a stub that always returns zero, so the actual
production path is separate Q-b, head RMS norm, then RoPE. Treating that as one fused boundary would
conflate three mechanisms.

**Reference/candidate.** Both use the restored batch-M1 comparator and V6 row-wise Q/KV for layer 0.
The candidate additionally uses the existing Q8 primitive once per verifier row for Q-b at layers
`< DS4_LEAD08_ROWWISE_QB_LAYERS`; the reference sets that cap to zero. The cap is saved/restored only
inside `metal_graph_verify_suffix_tops`, engages only for multi-token suffix verification, and does
not alter prompt prefill or the M=1 comparator. Head norm, RoPE, KV/cache, attention, FFN, and output
head remain unchanged.

**Exactness and economics.** At position 104, layer-0 M=K row 0 and complete M=1 must have identical
`q_lora_norm` input. `ProdQB` must become bit-identical for V7 to pass causally. Branch-neutral
`ProdQHeadNorm` and `Qcur` captures identify norm or RoPE as the next boundary without requesting the
existing debug names that affect fused-hook dispatch. `N=1` is the causal test; `N=43` is only a
naive all-layer implementation-cost upper bound. No row-loop overhead is added to the 43 ms
optimistic floor unless it is proved unavoidable; the hard verifier gate remains 50.5 ms at K=4.

**Outcome tree and retained evidence.** If `ProdQB` remains different with exact input and confirmed
dispatch, V7 fails or the implementation is wrong. If it becomes exact but `ProdQHeadNorm` differs, norm
is next; if norm is exact but `Qcur` differs, RoPE is next; if all are exact, advance to the first
downstream boundary. An unexpectedly successful fused hook makes the run ambiguous for this design.
Retain the exact command/environment, stage-diff CSV, any matched timing CSV, source scope evidence,
the challenger result, and a mandatory post-iteration red-team verdict before commit.

**Result.** With V6 exact inputs, row-wise Q-b makes `ProdQB`, unchanged head norm/Q RoPE,
attention, and inverse RoPE bit-identical through `kqv_back` for the captured row. The final
cycle-24 flip remains, but max logit error falls from 0.927542 to 0.133129. The exact post-alias
timing rerun observes +0.140 ms for layer 0 and a confounded +7.465 ms all-layer whole-graph delta;
the latter is a naive repeated-dispatch upper-bound estimate, not an unavoidable kernel cost. The
initial audit's command/parser/wording/ledger blockers were corrected and the repeat audit returned
**COMMIT**. Canonical artifact: `artifacts/lead08_reassessment/21_iter13_rowwise_qb.md`.

### 2026-07-17 - iteration-12 V6 row-wise Q/KV projection -> CAUSAL GO / audit COMMIT

**Hypothesis V6.** The first row-0 M=K/M=1 difference is caused by the Q8 projection dispatch
changing from the M=1 matvec reduction to the 2-8-row extended kernel. Running Q-a and KV as
independent M=1 row views should make `q_lora` and `KVraw` bit-identical and move the first
divergence downstream without changing any other batch stage.

**Reference/candidate.** Reference is iteration 11's restored same-frontier M=K probe. Candidate
uses `DS4_LEAD08_ROWWISE_QKV_LAYERS=N`, saved/restored only inside the suffix verifier. `N=1` is the
causal capture; `N=43` is the naive all-layer cost probe. Prompt prefill, batch M=1, normalization,
Q-b, RoPE, cache/compressor, attention, FFN, and output head remain unchanged.

**Correctness invariant.** At `code_sort_pairs` position 104, M=K row 0 and the complete M=1 tensor
must retain bit-identical `attn_norm`; the first 1024 `q_lora` and 512 `KVraw` F32 values must become
bit-identical. Only row 0 is a same-frontier/identical-input claim. Dumps stop at `KVnorm`; requesting
`Qraw` would itself change the Q-b dispatch.

**Economic metric and threshold.** Matched fixed-K=4 runs without distribution probe, tensor dumps,
or per-stage synchronization compare `DS4_MTP_VERIFY_PROFILE` layer intervals for `N=0`, `N=1`, and
`N=43`. That profiler adds common per-layer router capture/statistics work, so paired deltas are
supporting evidence but absolute times do not compare directly with the 50.5 ms production gate.
`N=43` is an implementation upper bound, not an unavoidable-cost lower bound. Continue if the
correctness frontier moves and a credible optimized path to `verify_ms(4) <= 50.5 ms` remains.

**Cheapest falsifier and outcomes.** One verifier-scoped helper loops over row-contiguous views and
calls the existing named Q8 primitive with `n_tokens=1`. If either projection still differs after
the activation log confirms the path ran, V6 fails. If both match and `KVnorm` differs, V6 passes
causally and Q/KV normalization becomes the next boundary. If all captured stages match, extend the
capture only to the next boundary. A slow `N=43` closes only this naive row-dispatch implementation;
V5 stops only when costs proven unavoidable raise the retained feasibility lower bound above 50.5 ms.

**Evidence/audit contract.** Retain the one-prompt config, compact stage comparison, matched timing
summary, exact commands, and preflight challenge. The preflight returned **REDESIGN, then proceed**:
it required suffix-only scope, a numeric layer cap, minimal dumps, and upper-bound wording. Repeat
the independent red-team audit after results and commit only on its approval.

**Result.** With `N=1`, layer-0 row-0 `attn_norm`, `q_lora`, `KVraw`, both Q/KV normalizations,
KV RoPE/storage, and the common raw-cache prefix are bit-identical. `q_lora` improves from 799/1024
different words to zero and `KVraw` from 405/512 to zero. The first captured difference moves to
production `Qcur` (25,988/32,768 words, max 1.91e-6), the fused Q-b + per-head norm + RoPE stage.
The final cycle-24 logit flip remains, so this is boundary movement rather than verifier exactness.

Four order-balanced process-first K=4 samples put the noisy `N=1` layer delta at +0.080 ms mean /
+0.083 ms median (range -0.215 to +0.370) and naive `N=43` at +0.564 / +0.687 ms. The first
baseline total had a cold encode outlier; paired total medians are +0.081 and +0.810 ms. Full-corpus
`N=43` changes selected routed work, so these are observed whole-layer deltas and only an upper-bound
estimate for the naive implementation, not an isolated or proven-unavoidable Q/KV cost. The retained
~43 ms optimistic feasibility floor remains; the next boundary is row-wise production Q-b + norm +
RoPE. The ~56 ms absolute interval is profile-instrumented and is not compared directly to the
50.5 ms production gate. Artifact 20 + two compact CSVs.

The mandatory audit first blocked the timing record's unsupported thermal-control claim, incomplete
sequence reproduction, absolute profiler interpretation, and omitted selected-work confound. After
correction it returned **COMMIT**, validating source scope, row-view/primitive behavior, capture,
CSVs, flip, timing arithmetic, commands, and ledger. Residuals: no forced-error scope test; negative
layer text clamps to 43; correctness is one row/prompt/layer; timing is noisy profile-instrumented
supporting evidence.

### 2026-07-17 - iteration-11 same-frontier localization -> corrected audit COMMIT

Used `DS4_DSPARK_VERIFY_DIST_PROBE=1` with committing M=K disabled and the iteration-10 batched-M1
target enabled. The diagnostic computes M=K logits from a snapshot, then intends to restore before
committing accepted tokens through M=1. Across the first run, 176 cycles produced 349 accepted-row
comparisons with two argmax flips; all 640 singleton target evaluations are batched M=1 and none are
raw decode. Only row 0 is an identical-frontier/identical-input comparison.

Reproduced the `code_sort_pairs` flip (cycle 24, suffix start position 104) with existing tagged
tensor dumps. Layer-0 row-0 `hc_attn_pre` and `attn_norm` are bit-identical. The first captured
difference is `q_lora` (799/1024 F32 words, max 3.35e-8); the independent KV projection differs too
(405/512, max 2.98e-8). The error reaches max 1.72e-5 at attention output and propagates from
post-FFN max 6.85e-7 at layer 0 to 0.94 at layer 42. The dump run preserved the same flip cycle,
drafts, and max logit difference as the no-dump run.

The first mandatory red-team audit returned **NO-COMMIT** because `spec_frontier_restore` failure was
ignored, so the same-frontier invariant was not enforced across the corpus. It also corrected 267
probe-present cycles to 176 cycles with comparisons and required exact dump/scope reproduction.
The raw arithmetic, dump layout, stage/growth values, and bounded Q/KV rationale otherwise passed.
The diagnostic now hard-fails on restore failure. All 10 corrected corpus runs completed, so all 267
probe executions restored successfully; the 349-row/two-flip result and both localization captures
reproduced exactly. The repeat audit independently validated the code, arithmetic, captures, layouts,
commands, and ledger, and returned **COMMIT**. Its only nonblocking residual is that a hard restore
failure can leave the global debug filename tag as `b_`; computation cannot continue in that
session. Retained artifact 19 + three compact CSVs.

### 2026-07-17 - iteration-10 existing batch graph fails the shared M=1/M=K strategy -> STOP

Before implementing standalone GPU timestamp attribution, ran an independent red-team audit. It
found a material dossier overclaim: iteration 3 rejected only a grouped routed-MoE stage inside the
nonexact batch verifier, not Phase B's intended end-to-end exact hybrid. The cheapest still-open
exactness strategy was using one operation family for target M=1 and speculative M=K.

Added research-only `DS4_LEAD08_BATCH_M1_TARGET=1`. Normal Metal target evaluation and both DSpark
replay fallbacks use `metal_graph_verify_suffix_tops` with one token, so M=1 and M=K share the batch
layer and output-head functions. DSpark sessions capture layers 40-42 through the existing batch
buffers and push row 0 through the existing batch-hidden lifecycle. Default inference is unchanged
and the diagnostic rejects MTP rather than silently leaving its state inconsistent.

The mandatory post-implementation red-team audit rejected the first attempt because it used the
singleton prefill output head and left raw-decode fallbacks. That attempt was not committed. The
corrected benchmark JSON records 13 batched-M1 and zero raw-M1 evaluations across 282 speculative
cycles.

On the retained 10-prompt corpus (frontier 48, 64 emitted each), token-level comparison against the
batch-M1 target passes only 5/10 prompts. First divergences occur at emitted positions 29, 39, 21,
11, and 53 in `grounded_archive`, `grounded_observatory`, `synthesis_incident_json`,
`synthesis_ops_json`, and `synthesis_timeline_json`. A shared layer-major call graph therefore does
not make the end-to-end graph/state transition M-invariant; this run does not localize the cause to
a particular kernel or frontier update.

Economics also fail. Aggregate shipped decode is 37.902 t/s; batch-M1 target is 32.989 (-13.0%);
batch-M1 plus M=K DSpark is 37.407. That is only +13.4% over its re-baselined target, below the >=20%
gate, and remains 1.3% slower than shipped decode.

Verdict: **NO-GO for reusing the existing batch graph as the exact shared family.** Retain the env
path as a falsifier. This corrects the historical record rather than closing the original hybrid
build: decode-order HC/compressor/attention plus sharing only at proven-invariant stages remains
unbuilt. Retained artifact 18 + CSV.

### 2026-07-17 - iteration-9 corrected actual-selection profiling; cache verdict unchanged

Audited the iteration-8 batched-verifier profiler after its layer locality looked implausibly
repetitive. It read the reused router tensors on the CPU while their producing Metal command buffer
was still uncommitted. On a repeated 64-token `code_topk` run, the stale path produced only 11
distinct top-16 expert/count vectors across 43 router layers and three distinct adjacent-overlap
values, proving that those locality values were not layer-aligned.

The diagnostic path now encodes per-layer GPU snapshots immediately after each router dispatch and
reads them only after normal command-buffer completion. Capture tensors and copies are allocated
only when an expert/verify profile is active. The corrected same-prompt run produces 43 distinct
layer vectors and 38 overlap values. Timing-stripped complete cycle trajectories are identical
before and after the fix (SHA-256
`6fa3d319ad89ef8afddc7f89c65145fdbd0214c7326a92aaa438c2a5b34f3481`).

The corrected three-family run records 12,212 layer records / 73,272 selections. Mean adjacent
top-6 overlap is 0.373; per-layer LRU hit rates are 38.6%, 57.3%, 72.3%, 81.5%, and 87.9% at
capacities 8/16/32/64/128. A profiling-disabled control has the same full trajectory hash and
35.66 versus 35.40 t/s, bounding diagnostic overhead at about 0.7%.

Verdict: **iteration-8 locality numbers are superseded, but its carrier NO-GO is unchanged.** The
mapped path still measures 0.566/0.566 ms per layer cold/replay; only the DSpark-incompatible SSD
selected-address path has a material residency miss. Retained artifact 17 + CSV and replaced the
canonical artifact-16 LRU curve.

### 2026-07-17 - iteration-8 cache residency has an SSD-only prize -> current-path STOP

Profiled the preparation term outside the iteration-2 GPU stage boundaries. At the fixed K=4,
18-unique/24-pair shape, selected-ID readback is 0.00027 ms/layer while cold selected-address
cache wrap/load is 5.407 ms/layer. Added `DS4_LEAD08_CACHE_REPLAY_PROBE=1` to run an identical
all-layer selection sequence cold and immediately resident, with a direct fidelity gate.

The SSD selected-address path is bit-exact and drops 11.094 -> 1.353 ms/layer (-87.8%) on replay;
wrap/load drops 5.407 -> 0.001 ms. This is a real upper bound, not a deployable result:
`ds4-spec-bench` explicitly rejects `--ssd-streaming` with `--dspark`. Running the same probe on the
actual compatible mapped-weight path gives 0.566 -> 0.566 ms/layer (-0.04%), also bit-exact.

Extended the retained expert locality profiler to ingest actual batched-verifier selections.
Iteration 9 found that the initial implementation read reused router tensors before command-buffer
completion, invalidating this entry's original locality curve. The corrected three-family result is
57.3%/72.3%/81.5%/87.9% at capacities 16/32/64/128. The locality is real, but only an SSD-compatible
speculative runtime could exploit the measured cold-load gap.

Verdict: **NO-GO for current Lead 08; do not port selected-address caching into the mapped path.**
The compatible carrier misses the material gate, while building DSpark+SSD compatibility belongs to
Lead 05/runtime scope with its own baseline. Retained artifact 16 + two CSVs, the replay probe, and
batched expert-profile support for reproduction.

### 2026-07-17 - iteration-7 production row-tile sweep is exact but slower -> STOP

Added research-only tile-2/tile-8 specializations of the production
`kernel_mul_mv_addr_iq2_xxs_pair_swiglu_f32` path and selected them with
`DS4_LEAD08_ADDR_NR0`. `DS4_LEAD08_ADDR_NR0_PROBE=1` holds NSG 2, 24 physical rows, 18 unique
experts, routing, addresses, and downstream work fixed. The default remains the production tile 4.

The direct layer-0 gate finds tiles 2 and 8 bit-exact against tile 4 through gate, up, weighted mid,
routed output, and argmax: every differing-bit count and argmax-flip count is zero.

Across two four-round cold-cache balanced repetitions and all 43 routed layers, tiles 2/4/8 take
11.064/10.954/10.982 ms/layer total (+1.0%/baseline/+0.3%). Matched gate+up is
4.998/5.063/4.938 ms/layer (-1.3%/baseline/-2.5%), but excluding startup-heavy layer 0 changes the
comparison to 3.292/3.249/3.251 (+1.3%/baseline/+0.05%). The apparent tile-8 profile win is startup,
not steady-state improvement; its total-path sign also changes between repetitions (1.35% slower,
then 0.81% faster), confirming that the sub-percent effect is noise-scale.

Verdict: **NO-GO; retain row tile 4 and stop address-kernel geometry tuning.** Both exact alternates
miss the >=15% gate and provide no repeatable unprofiled gain. Retained artifact 15 + CSV and the
env-gated specializations/sweep only for reproduction; default execution is unchanged.

### 2026-07-17 - iteration-6 production address-kernel geometry has no exact speedup -> STOP

Added a diagnostic-only SIMDgroup-count override for the production
`kernel_mul_mv_addr_iq2_xxs_pair_swiglu_f32` path and a balanced K=4 sweep under
`DS4_LEAD08_ADDR_NSG_PROBE`. The experiment holds 24 physical rows, 18 unique experts, routing,
addresses, reduction order within each SIMDgroup, and all downstream work fixed. The default remains
NSG 2; its literal original IQ2 lookup-table preload is unchanged. Research specializations use a
complete table preload for NSG 1/4/8.

The direct layer-0 gate finds NSG 1, 4, and 8 bit-exact through gate, up, weighted mid, and routed
output, with zero differing F32 bits and zero argmax flips.

Across four cold-cache balanced rounds and all 43 routed layers, production NSG 2 is 11.300
ms/layer total. NSG 1/4/8 are 11.076/10.943/11.099 (-2.0%/-3.2%/-1.8%). Matched gate+up is
5.094/5.102/5.160/4.998 for NSG 1/2/4/8 (-0.2%/+1.1%/-2.0% versus 2); excluding layer 0 gives
+0.3%/-1.1%/-0.7%. The low-single-digit changes are inconsistent across views.

Verdict: **NO-GO; retain NSG 2 and stop the SIMDgroup-count branch.** No alternate approaches the
>=15% stage gate, and a noisy 1-3% microbenchmark effect does not justify changing production.
Retained artifact 14 + CSV and the env-gated override/sweep only for reproduction; default execution
is unchanged.

### 2026-07-17 - iteration-5 fixed-work expert-address locality is <2% -> STOP

Added `DS4_LEAD08_ADDRESS_LOCALITY_PROBE`, a diagnostic-only mode of the retained production K=4
batch-MoE harness. It fixes 24 physical token-expert rows, 18 unique experts, and duplicate
multiplicities, then compares contiguous placement, a permutation of the exact same 18-expert set,
and a coprime-strided placement across the expert slab. Every case clears the resident cache; four
rounds alternate case order and cover all 43 routed layers.

Unprofiled routed cost is 9.766 ms/layer contiguous, 9.953 permuted (+1.9%), and 9.835 strided
(+0.7%). A matched gate+up profile gives 4.381/4.450/4.422 ms/layer (+1.6%/+0.9%); excluding the
four first-use layer-0 samples gives +1.8%/+1.4%. The agreement between total and isolated stage
timing rules out a hidden large layout effect.

Verdict: **NO-GO for software-visible expert remapping/packing.** The result is far below the
predeclared >=15% stage gate and cannot supply the 8.7-10 ms verifier saving. It does not prove a
hardware bandwidth-vs-compute attribution; full Xcode counters remain diagnostic if a distinct
mechanism appears. Retained artifact 13 + CSV and the env-gated harness for reproduction.

### 2026-07-17 - iteration-4 margin guard catches the flip only above the economic limit -> STOP

Extended `DS4_DSPARK_VERIFY_DIST_PROBE` to record batch top-1/top-2 margins and simulate exact
fallback at the predeclared 0.25/0.5/1.0/1.75 thresholds. The change is diagnostic-only: verifier
selection and commit behavior are unchanged. Replayed the same ten exactness prompts as the retained
M3 batch-vs-exact artifact with per-cycle timing: 163 cycles, 158 batch probes, 91 compared cycles,
157 relevant rows, and the same one argmax flip.

The flip's batch margin is 0.375505. A 0.25 guard misses it. A 0.5 guard catches the observed flip
but triggers on 27/158 probed cycles (17.1%) and 27/157 rows. Measured exact replay plus state push
adds 7.006 ms per overall cycle before accounting for a GPU top-2 primitive, synchronization,
rollback, or policy overhead. Applied to the live M3 cycle, the optimistic projection is
40.04 -> 36.37 t/s, **4.7% below plain**. Thresholds 1.0/1.75 add 13.7/17.3 ms per cycle.

Verdict: **NO-GO; do not implement the margin fallback.** The only tested threshold that retains
most of the speed fails the observed exactness case; the smallest threshold that catches it erases
the speedup. Zero missed flips at 0.5 is observational, not a proof from one flip, because there is
no analytic reduction-error bound. Retained artifact 12 + CSV and the extended probe only for
reproduction. Lead 08 now has no demonstrated grouped-kernel or margin-fallback path.

### 2026-07-17 — iteration-3 expert-major spill prototype is bit-exact but slower -> grouped branch STOP

Implemented the single bounded follow-up from artifact 10 behind
`DS4_LEAD08_GROUPED_GATEUP_PROBE`. Singleton experts retain the production physical-row reduction;
cache preparation emits a compact unique-expert/row map, and duplicate groups run an expert-major
kernel that reuses quantized gate/up tiles across matching tokens while spilling activation tiles
and partial sums to about 26 KiB of threadgroup memory.
Default execution is unchanged. Added a direct production/prototype comparison under
`DS4_LEAD08_GROUPED_GATEUP_FIDELITY`.

Fidelity passes strongly on layer 0 across all four controlled overlap shapes: gate, up, weighted
mid, and routed output have max_abs=0 and zero differing F32 bits; routed-output argmax flips=0.
The matched 43-layer x 4-round stage profile fails the cost gate. At 18 unique / 24 pairs,
production is 4.869 ms/layer and the prototype 5.986 ms/layer, a **22.9% regression** instead of
the required >=15% improvement. Excluding layer 0 strengthens the regression to 35.8%. The
6/12-unique cases regress 68.1% / 55.6%, while the disjoint control is parity.

Verdict: **NO-GO; stop the grouped gate+up branch.** The compact map removes the disjoint overhead,
and heavier overlap shows directly that threadgroup spill/barrier costs overwhelm duplicate
dequant/load reuse. Per the bounded plan, do not integrate or iterate. The final exact
K=4 verifier gate is not run because the prerequisite stage gate failed. Retained artifact 11 +
CSV and the env-gated implementation solely for reproducibility.

Post-decision counter audit: the active developer tools provide neither `xctrace` nor the offline
`metal` utility, and the M5 Max Metal API advertises only `timestamp/GPUTimestamp`. No DRAM, cache,
occupancy, or instruction counters are available here. Artifact 13 subsequently supplied the
controlled fixed-work timing alternative and found <2% address-layout sensitivity. Hardware
bandwidth-vs-compute attribution remains unmeasured, but simple address remapping is now closed and
does not reopen the stopped grouped branch.

### 2026-07-17 — iteration-2 production batch overlap probe resolves the cold-sweep blocker -> conditional prototype GO

Replaced artifact 09's invalid M1x2 overlap comparator with a production-shaped K=4 harness around
`ds4_gpu_routed_moe_batch_tensor`. Held physical work at 24 token-expert rows, varied unique experts
6/12/18/24, cleared the resident cache before each all-layer sample, alternated case order over four
rounds, and separately profiled gate+up/down. Retained artifact 10 + CSV.

Result: total routed cost tracks unique experts (5.451/7.277/10.334/13.489 ms/layer), proving the
batch resource preparation is already union-aware. But at the realistic partial-overlap shape,
gate+up(18 unique)=4.642 is 97% of disjoint=4.787 despite 25% fewer unique experts: the physical-row
GPU kernel gets little benefit from sparse doubletons. Down is much smaller (0.695 vs gate+up 4.642
profile ms). This corrects both extremes: neither "production cost is physical-pair-only" nor
"existing caching already removes the grouped-kernel opportunity" is true.

Verdict: **CONDITIONAL GO for one bounded expert-major gate+up prototype** using M=1 register state
plus threadgroup-spilled per-token partial sums. Require >=15% gate+up improvement at 18 unique / 24
pairs; otherwise stop. The old 4.4x M2/M1 magnitude is withdrawn (warm-cache comparator), while the
iteration-1 simultaneous-fusion NO-GO remains structural (18-vs-12 work + doubled live state). The
prior 6-8 ms prize is downgraded from reliable to unproven until the prototype measures it.

### 2026-07-16 — codex avenues/experiments review of the re-assessed doc → CONDITIONAL HOLD; sub-lead-1 probe REVISED

Commissioned an adversarial codex review (gpt-5.5 xhigh, read-only) of the
folded re-assessment to find missed avenues + better experiments
(`dspark_codex_reviews/2026-07-16_gpt55_xhigh_lead08_reassess_avenues.md`).
Independently verified its decisive claims. **Verdict: CONDITIONAL HOLD**
(downgraded from the GO my synthetic-load probe implied). Confirmed: prize math
(89.5%, ~10 ms = +20%), per-pair de-dup duplication, dense-shared/experts-aren't.
Corrected: (a) the "190 GB/s → scattered access → recoverable" attribution is
UNSUPPORTED — the logged GiB is **unique-union bytes** (ds4.c:12401), not
physical per-pair; the slope is inconsistent (186/56/88–135 GB/s);
cache-hit-vs-miss unresolved; (b) the de-dup ceiling is only ~21–30% (high expert
overlap, union ≈19–21 of 24–30 pairs); (c) the synthetic-load microbench is not
runnable without a new kernel. Revised the prize: **reliable ~6–8 ms (de-dup,
borderline +20%); ~16 ms speculative (needs the bandwidth attribution)**. Revised
sub-lead 1: dropped the synthetic-load probe; the **free stage-profile sweep**
(K=2..5, the 4 existing flags) + the **pair-vs-unique correlation** are now the
decisive first experiments; added the readahead/overlap flag sweep (cheap
pre-kernel avenue), Instruments/GPU counters (the only bandwidth-vs-compute
settler), + the exactness margin probe. Codex also ranked missed avenues
(readahead/overlap first; STS re-tune; margin-guard fallback) + ruled out (shared
Q8 expert already batched; activation fusion sub-ms; on-GPU argmax already done).
Updated this doc (§"Re-assessment" — tempered mechanisms, revised crux + sub-lead-1)
+ artifact `02_sublead1` (revised).

### 2026-07-15 — re-assessment (post-M3): folded the re-derivation + sub-lead-1 probe into this doc as avenues; corrected the "marginal" mischaracterization

Milestone-3 delivered the runtime stack + the full DSpark stack now beats plain
+4.9% (40.04 vs 38.16 t/s). Re-derived the Lead 08 prize from the LIVE M3 cycle:
the verify is **89.5% of the cycle** (62.07 ms of 69.38 ms), so a verify saving
amplifies ~1:1 into throughput — **~10 ms = +20% over plain** (the prior "8–18%
of verify = marginal" was wrong; conflated % of verify with % of cycle).
Identified the two fusion mechanisms (dequant load-sharing ~5–8 ms + coalesced
union access → decode bw ~16 ms) + the crux (why is the verify at ~190 GB/s?).
Locked **sub-lead 1**: the 4-measurement probe (synthetic union-load bandwidth,
compute/bandwidth split, expert overlap, attention share) as the decisive
GO/NO-GO gate before any kernel build. Code-verified the load-sharing premise
(the batch expert kernel `kernel_mul_mv_addr_iq2_xxs_pair_swiglu_f32` moe.metal:1257
dequants per-(token,expert); the dense matmul IS shared via `r1_2..5`). Folded
into this doc (§"Re-assessment (2026-07-15)") + artifacts
`lead08_reassessment/01_reorient_bar_redrivation.md` +
`02_sublead1_fusion_prospect_and_probe.md`. Codex review of this doc
commissioned next (missed avenues/experiments).

### 2026-07-13 — cycle timing breakdown + strategic correction: the lever is anchor reuse + GPU drafter, NOT novel verify kernels

Measured the DSpark cycle timing breakdown (code_topk, 96 tok, 44 cycles, dist-probe OFF):
**draft 26.0 ms (37%) | verify 26.5 ms (38%) | decode(anchor) 26.3 ms (37%)**, total 70.5 ms
→ ~37 t/s. The anchor decode fires in **75% of cycles** (median 26 ms) — but the model's
anchor-reuse regime expects `decode·S(K)` (rare, only on full-accept; at this acceptance
S(K) ~10–15%). So the runtime is NOT in the anchor-reuse regime — it pays a fresh anchor
decode ~5× too often (the Lead 06 gap, live).

**Research-lead correction:** the anchor decode is a reuse/scheduling gap, not an inherent
cost. A scheduled verifier that verifies K+1 tokens (Lead 02) always produces the next
anchor's hidden → the fresh anchor decode never fires (carried hidden = anchor reuse).
That collapse is a state-plumbing/scheduling change, NOT a novel kernel. Corrected
"where to start":
1. anchor reuse + verify-K+1 → decode term → ~0; cycle ≈ draft+verify ≈ 54 ms/2.6 tok ≈
   48 t/s ≈ **1.25× baseline (clears +20% on its own)**.
2. + GPU drafter (draft 26→10) → ≈ 68 t/s ≈ **1.77× baseline**.
Both cheaper than novel IQ2XXS verify kernels; (1) alone projects to clear the gate. The
novel verify kernels drop to last priority (possibly unnecessary). This overturns the
earlier "weeks of novel verify kernels" framing — the binding levers are the anchor-reuse/
scheduled-verify gap + the GPU drafter.

### 2026-07-13 — ds4-eval practical comparison: batch-verifier vs plain decode (outputs identical; ~4% slower)

Ran `ds4-eval --plain --nothink --tokens 2048 --temp 0 --seed 1 --questions 10` baseline vs
DSpark with `DS4_DSPARK_SCHEDULE_BATCHED=1` (batch verifier confirmed engaged:
`batched_scheduled_draft = scheduled_verify && dspark_schedule_batched_draft()`, both true).
Artifact: `artifacts/lead08_stage_divergence/ds4eval_batch_vs_baseline.md`.

Result: **outputs byte-identical 10/10** (the verify-level 0.64% argmax-flip does NOT
propagate to the committed output on this ~5.8k-token sample); DSpark-batch ~37.0 t/s vs
baseline ~38.7 t/s → **~4% slower (no speedup)**; same 8/10 pass (2 AIME fails hit the
2048 cap). Interpretation: the current batch-verifier path is greedy-exact in committed
output but provides no speedup — consistent with the codex-gated verdict (a gate-clearing
verifier needs sublinear AND fast, not just exact). Open: a larger sample + flip-detecting
diff to confirm the committed-output-exactness holds generally.

### 2026-07-13 — decisive measurements + gate C + final verdict: NO-GO via swaps; bounded build attempt

Ran the three codex-suggested decisive measurements (swap-only constraint). (1) MoE-equality:
gate/up bit-exact given identical inputs — consistent with source (both use `_impl`) + a
~480× inherited-input amplification check (the 6.7–7.6 gate/up divergence is the matmul
amplification of the 0.014 inherited ffn_norm; routed output 0.018 near-exact via
cancellation). (2) Attention-residual: NO config swap forces the batch HC/compressor/
attention onto decode reductions (`--quality` is N=2-only) → needs a code change = beyond
swaps. (3) Bit-exact K=4 (decisive): the existing bit-exact verifier (DSpark sequential) =
~0.85× baseline on the 8k corpus (30.81/31.69/28.85 vs 36.34/37.46/33.65) → NO-GO; no swap
yields a sublinear bit-exact verifier → needs novel kernels (ABORT per constraint).

Codex gate C (gpt-5.5 xhigh; `artifacts/dspark_codex_reviews/2026-07-13_gpt55_xhigh_lead08_gateC_final.md`)
corrected: moe "confirmed"→"consistent with"; the code_topk 0.41× probe was dist-probe-
inflated (use 8k ~0.85×); "commit to novel-kernel build"→"bounded build attempt with hard
exit gate; GO unconfirmed until ≤50.5 ms"; confirmed no missed swap. Corrections applied to
`decisive_measurements.md` + `floor_clearance_verdict.md` (v4). Final verdict propagated to
`summaries/lead08_phaseB_floor_clearance_verdict.md`, STATUS.md, spec_speedup_model.md.
Committed on `dspark-research`.

### 2026-07-13 — propagate-verdict: milestone complete; HOLD verdict propagated to STATUS + spec_speedup_model

## Worklog

### 2026-07-13 — propagate-verdict: milestone complete; HOLD verdict propagated to STATUS + spec_speedup_model

Finalized the milestone. Wrote the canonical summary
(`summaries/lead08_phaseB_floor_clearance_verdict.md`) capturing the verdict (HOLD), the
robust findings (gate threshold verify_ms(4) ≤ 50.5 ms; decode bw ~410–450 GB/s resolved;
K=4 verify floor at decode bw ~39 ms → 1.44×; headroom = GPU layer_execute bandwidth
inefficiency), and the gate-corrected overclaims (F16-on-both-paths; gate/up divergences
real; pos-61 not clean-input; "two decisive measurements" → the one bit-exact K=4 profile).
Updated STATUS.md (Lead 08 Phase B characterization entry) + spec_speedup_model.md
(decode-bw resolution + the 50.5 ms gate threshold + the ~39 ms floor). Committed on
`dspark-research`. The lead worklog stays in `pending/` (the lead is not resolved — verdict
is HOLD pending the decisive bit-exact K=4 verifier profile).

## Worklog

### 2026-07-13 — codex gate A + verdict revision: CONDITIONAL-GO draft was not decision-grade; revised to HOLD

Codex gate A (gpt-5.5 xhigh; artifact
`artifacts/dspark_codex_reviews/2026-07-13_gpt55_xhigh_lead08_gateA_methodology.md`)
challenged the methodology; I independently verified the two decisive findings: (1) the
"F16 is the dominant divergence / closable by F32 swap" claim is WRONG —
`metal_graph_matmul_plain_tensor` (ds4.c:16615) dispatches on `w->type` and `hc_attn_fn`
is `DS4_TENSOR_F16` (ds4.c:3655), so decode ALSO uses `ds4_gpu_matmul_f16_tensor`; the
compressor is F16 on both paths → divergence is reduction/path/order, not dtype. (2) the
300 GB/s decode figure is load-bearing + unmeasured — recomputed: 300→1.346×, 250→1.193×
(under gate), 200→1.019× (fails); cliff-edge. Also: the 6.2–7.6 gate/up divergences are
REAL (codex recomputed the dumps; not a layout artifact); the 190 GB/s slope is
cherry-picked (K4→K5 contradicts at 56 GB/s).

**Verdict revised** (`floor_clearance_verdict.md`): from CONDITIONAL GO to **HOLD /
INCONCLUSIVE** — the gate-clearance is too fragile to commit (decode bw unmeasured; ≤250
→ fails) and the exactness story was wrong (F16 on both paths; real divergence source
unidentified; gate/up bit-exactness unverified). Exactness-gap artifact corrected (gate-A
correction prepended). Robust findings kept: gate threshold verify_ms(4) ≤ 50.5 ms;
headroom is GPU layer_execute not host overhead; Phase-B targeted the wrong thing. The
GO/NO-GO is deferred to two decisive measurements (decode bandwidth; identical-input MoE
kernel equality). Gate A did its job (caught the overclaims before propagation).

### 2026-07-13 — floor-clearance-verdict (DRAFT): CONDITIONAL GO; gate clears at verify_ms(4) ≤ 50.5 ms; Phase-B plan superseded

Produced the go/no-go verdict (artifact:
`artifacts/lead08_stage_divergence/floor_clearance_verdict.md`), integrating the headroom
decomposition + exactness-gap + `spec_speedup_model.md`. Gate-clearance threshold:
**verify_ms(4) ≤ 50.5 ms** clears +20% (stack: anchor-reuse + GPU drafter + oracle
acceptance). Sensitivity: current batch 66 ms → 0.98×; floor ~43 ms → 1.34×; bit-exact
F32 est. ~47–50 ms → 1.21–1.26× (borderline). **Verdict: CONDITIONAL GO** — both
barriers are tractable and lower-effort than the original plan: cost = bandwidth tuning
(recoverable, proven at K=2), exactness = F32/Q8_0 matmul swaps (decode kernels exist, no
novel IQ2_XXS kernels; gate/up already bit-exact). Conditional on two cheap measurements
(attention residual after F32 compression; F32 speed cost) before the heavy engineering.
**Supersedes the Phase-B single-stage-kernel plan** (it targeted the gate/up, already
bit-exact; the cost barrier is bandwidth, not expert-load-sharing). Not a guarantee the
gate is cleared — that needs the two measurements + the bandwidth engineering.

### 2026-07-13 — exactness-gap: dominant divergence is F16 vs F32/Q8_0 matmul swaps; gate/up already bit-exact

Characterized the bit-exactness fix scope per divergent stage (artifact:
`artifacts/lead08_stage_divergence/exactness_gap.md`, read-only kernel analysis + the
diagnose-divergence data). The batch path uses `ds4_gpu_matmul_f16_tensor` for the hc
projections AND the attention compression (`batch_comp_kv`/`batch_comp_sc`); decode uses
plain/Q8_0. **The F16 precision loss is the dominant divergence source** (hc 0.078–0.086,
KVcur 0.125 — the largest), and it is **closable by kernel swap** (use the decode
plain/Q8_0 matmul), low–medium effort, no novel kernels. The gate/up — the original
"fusion" target — are already bit-exact. down/sum6 (`id_q2_k` vs `addr_q2_k`) is a minor
closable gap (ffn_moe_out 0.018). The one real uncertainty is the **batched-attention
residual after F32 compression** (the measured attn_out 0.064 likely inherits most of the
upstream KVcur F16 error; residual unmeasured) — a cheap measurement decides whether
attention is feasible for free or needs per-token (loses sharing). **Implication:** a
fully-bit-exact sublinear verifier is plausibly achievable WITHOUT novel IQ2_XXS kernels —
mostly kernel selection (use decode reductions in the batch path), substantially
lower-effort than the Phase-B single-stage-kernel plan.

### 2026-07-13 — headroom-decomposition: headroom is GPU bandwidth inefficiency (~23 ms recoverable at K=4)

Decomposed the ~19–22 ms verify headroom from the retained Phase A
`DS4_MTP_VERIFY_PROFILE` data (`artifacts/mtp_phaseA_profile/summary.json`, code_8k
K=3,4,5; reuses the instrumentation, reproduces the 20.9 ms headroom at K=4). Artifact:
`artifacts/lead08_stage_divergence/headroom_decomposition.md`.

Finding: `layer_execute` (GPU) is ~95% of verify_ms; host-side launch/encode/readback is
only ~3–4 ms (~5%, readback ≈0). So the headroom is **not** host overhead — it is GPU
bandwidth inefficiency: the verify path runs the routed-expert stream at ~190 GB/s
(K3→K4 slope: 1.14 GiB / 6.14 ms) vs decode's ~300 GB/s (~63%). If a fused/tuned kernel
reached decode bandwidth, `layer_execute(4)` 62.6 → ~39.7 ms, `verify_ms(4)` ~43 ms
(**recover ~23 ms**, at/below the 45 ms floor target). The `code_4k` K=2 ≈ decode point
corroborates that the floor is reachable at low K. Load-bearing uncertainty for the
verdict: whether an M=K fused kernel hits 300 GB/s at K=4 (unproven); the 300 GB/s decode
figure is the Lead 08 doc's assertion (order-of-magnitude robustness noted).

### 2026-07-13 — build-single-stage scope check: novel kernel is multi-week; paused for approach decision

Assessed the realistic scope of `build-single-stage` after the spec-read refinement.
The novel M=2 routed-expert kernel requires: (1) a new IQ2XXS union-expert load-once
kernel (extends `_impl` to 2 tokens with shared weight load + per-token accumulators),
(2) a complex host dispatch (`routed_moe_pair`: union-expert computation, mmap'd weight
binding mirroring `routed_moe_one`'s ~250-line SSD-streaming dispatch), (3) wiring into
`decode2_exact`, then (4) bit-exactness iteration via build + 87GB fidelity-gate cycles
(likely + a codex bug-hunt). This is the "weeks of Metal work" the doc flags — confirmed.

The refinement also revealed a more-tractable alternative: `verify_suffix_tops` (batch)
is ALREADY sublinear (shared loads) and bit-exact in gate/up (both paths use `_impl`);
it diverges only in down+sum6 + attention. So the milestone's questions (can routed
experts be sublinear? bit-exact gate/up? what cost saving?) are largely answerable from
the existing paths by measurement, without a novel kernel. Paused for the user to choose
between (A) commit to the novel-kernel build, (B) reframe to characterize the existing
batch path + scope the down/sum6 fix, or (C) prove the mechanism on the simpler shared
Q8_0 expert first.

### 2026-07-13 — build-single-stage spec-read: gate/up reduction already shared; impl focus = shared loads + bit-exact down/sum6

Read the actual routed-expert kernels before implementing. **Gate/up IQ2XXS paired
reduction is already shared**: both decode (`kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32`,
moe.metal:1022, inline) and batch (`kernel_mul_mv_addr_iq2_xxs_pair_swiglu_f32`,
moe.metal:1257) use the SAME reduction — the `addr` kernel calls
`kernel_mul_mv_iq2_xxs_pair_f32_impl` (moe.metal:680), bit-identical to the `id` inline
loop (same dequant, MAC order, `simd_sum`+`*0.25`). So **gate/up are bit-identical
batch-vs-decode** → the large `ffn_moe_gate_clamped`/`up_clamped`/`down` divergences
(6.7–331) in the diagnose map were a **per-expert layout artifact** (row-0 extraction on an
expert-major tensor), not real. f16-mid ruled out (`request_mid_f16 = ... &&
!use_iq2_batch_selected_addr`). The real routed-expert output divergence (ffn_moe_out
0.018) is in the **down+sum6** stage + tiny route-weight diff, not gate/up. The expert
weight LOAD is not shared across tokens in either path → cost saving unrealized.

**Revised impl focus:** (a) share expert weight loads (gate/up + down) across the 2
tokens via a union-expert load-once kernel (reusing `_impl` for the gate/up MAC) — the
verify_ms cost saving; (b) make the down+sum6 bit-exact with decode (gate/up already
are). Fidelity gate (M=2 ffn_moe_out == M=1, max_abs==0) validates both. Recorded in
`artifacts/lead08_stage_divergence/fused_stage_design.md` (spec-read refinement section).
No kernel code written yet — impl-stage-kernel is the next concrete step.

### 2026-07-13 — design-fused-stage: bit-exact M=2 routed-expert kernel design + fidelity-gate spec recorded

Read the M=1 reference `kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32` (moe.metal:1022,
N_R0=4) and the host dispatch (`routed_moe_one_tensor` ds4_metal.m:22362,
`routed_moe_batch_tensor` :24622). Design recorded in
`artifacts/lead08_stage_divergence/fused_stage_design.md`.

**Bit-exactness mechanism (strategy A):** the M=2 variant shares the gate/up IQ2XXS
weight dequant (grid+sign+scale, weight-only) across both tokens and keeps TWO
separate per-token accumulator sets using the IDENTICAL MAC order + `simd_sum`+`*0.25`
reduction as M=1 → each token's gate/up output is bit-identical to a standalone M=1
call, by construction. Union-expert handling mirrors `routed_moe_batch` (≤12 union
experts, per-token router routing + sum). Plug-in: env-gated
(`DS4_DSPARK_FUSED_ROUTED_M2=1`) branch inside `metal_graph_verify_decode2_exact`
replacing the two `routed_moe_one` calls with one M=2 dispatch; rest of the layer
stays per-token (exact).

**Fidelity-gate spec:** reuse the diagnose-divergence dump-tag harness — compare the
M=2 fused `ffn_moe_out`/`routed_out` vs the M=1 decode reference at a clean-input
position (pos 61, layer 40 + a second layer); **pass = max_abs == 0.0 bit-for-bit**,
plus 0 argmax flips on the full exactness corpus (temp=0) vs plain decode. Gate must
pass before any timing number; on failure → codex bug-hunt, fix, re-gate.

**Open build-time questions:** down-projection fusion (same kernel family?) vs
keep-per-token initially; parameterized single-family vs a new `_m2` kernel (prefer
parameterized for reduction identity); confirm per-expert tensor layout
(gate/up/down large point divergences). The fidelity gate resolves the layout/
numerics question empirically.

### 2026-07-13 — diagnose-divergence (empirical map): routed IQ2XXS experts confirmed; attention is the larger per-output divergence

Built a stage-level divergence harness via a minimal dump-path tag in the dist-probe
(`ds4_metal_dump_path_tag`, ~10 lines in ds4.c: tags dumps `b_` around
`verify_suffix_tops` and `s_` around the sequential `eval_token_raw_swa_top` loop).
Ran `code_topk` (DS4_DSPARK_VERIFY_DIST_PROBE=1, layer 40, pos 61 = cycle 2 where both
paths ran from identical committed state). Artifact:
`artifacts/lead08_stage_divergence/stage_divergence_map.json` (+ `dumps/`).

Findings: **expert selection (topk) IDENTICAL** batch-vs-seq for token 0 (no router
selection divergence). Reliable per-stage max_abs: attention/hidden (KVcur 0.125,
hc_attn_pre 0.086, attn_out 0.064) > FFN output (ffn_out 0.019, ffn_moe_out 0.018,
ffn_shexp 0.014, router logits 0.023). Expert-internal (gate/up/down) show large point
divergences (6.7–331) — the IQ2XXS matmul divergence amplified through the down
projection (per-expert layout to confirm at build). Per-stage TVs (0.0003–0.004) are
**consistent with the known final-logit TV ~0.0035** → sanity-check PASS.

**Chosen fusion stage: routed IQ2XXS experts** (highest cost, self-contained,
thesis-central). Nuance recorded honestly: the routed-expert OUTPUT divergence is
already small (0.018, summation cancellation), so the prototype's measurable
contribution is primarily the verify_ms **cost saving** (shared dequant), not a
final-TV reduction; the attention path is the larger per-output divergence and will
need fusing too for full-path exactness (→ follow-up). Harness + tagging retained in
the build for the routed-expert fidelity gate.

### 2026-07-13 — diagnose-divergence (theoretical map): routed IQ2XXS experts chosen as the fusion stage; empirical harness next

Read-only stage-level kernel map (decode `encode_decode_layer` vs batch
`encode_layer_attention_batch`+`encode_layer_ffn_batch`), ranked by fusion value
(cost × divergence). Comparable dump points exist in both paths for: `hc_attn_pre`,
`attn_out`, `hc_ffn_pre`, `ffn_moe_logits/probs/weights_scaled` (router),
`ffn_moe_weighted_swiglu` + `ffn_moe_down` + `ffn_moe_out` (routed expert),
`ffn_shexp` (shared expert), `ffn_out`.

| Stage | Decode kernel | Batch kernel | Diverges | Cost |
|---|---|---|---|---|
| hc_attn / hc_ffn projection | `matmul_plain_tensor` | `matmul_f16_tensor` | yes (plain vs f16) | low–mid |
| Attention (MLA) | `decode_kv_store`+flash per-token | `encode_layer_attention_batch` | yes | mid |
| Router | `matmul_plain`+`router_select` | `matmul_f16`+`router_select_batch` | yes (selection-critical) | low |
| **Routed experts (IQ2XXS)** | **`routed_moe_one_tensor`** | **`routed_moe_batch_tensor`** | **yes (distinct kernels)** | **HIGHEST (~5 GiB sel / 72.56 full)** |
| Shared expert (Q8_0) | `matmul_q8_0` (single) | `matmul_q8_0` (n_tokens) | maybe | mid |

**Chosen fusion stage: the routed IQ2XXS experts.** Rationale: highest weight-load
cost (the dominant bandwidth term) AND a distinct batched kernel
(`routed_moe_batch` vs `routed_moe_one`) → highest fusion value (cost × divergence).
It is also the marquee stage for the Lead 08 thesis and self-contained (input:
`ffn_norm` hidden state + selected experts; output: `routed_out`). The single-stage
prototype = `decode2_exact`'s exact per-token structure with the `routed_moe_one`
call (×2) replaced by one bit-exact M=2 union-load kernel sharing the expert dequant.
Empirical per-stage divergence (to confirm + quantify, and sanity-validate vs the
known final TV ~0.0035) is the next step — harness design: compare decode vs batch
intermediate tensors at the anchor position (identical inputs there) on `code_topk`.

### 2026-07-12 — build-gate diagnostic: divergence is structural & spread across the layer; multi-backend surface

Mapped the decode vs batch per-layer kernel sequences (read-only) to scope the fused
N=2 work. **Decode path** (`metal_graph_encode_decode_layer`, ds4.c:15198) runs the
M=1 sequence inline: rms_norm → `matmul_plain_tensor` (hc_attn) → hc_pre/comb →
qkv norms → `decode_kv_store` → flash_attn → attn_out/hc_expand → rms_norm →
`matmul_plain_tensor` (hc_ffn) → ffn hc_pre → router (`matmul_plain`) → routed IQ2XXS
experts (`kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32`, single-token) + shared Q8_0 expert.
**Batch path** (`metal_graph_encode_layer_batch`, ds4.c:19647 → `attention_batch` 17699
+ `ffn_batch` 19166) uses DIFFERENT primitives: `ds4_gpu_matmul_f16_tensor`,
`ds4_gpu_router_select_batch_tensor`, `metal_graph_matmul_q8_0_named_tensor(...,n_tokens)`,
batched routed-expert dispatch, with Metal/ROCm/CUDA branches + SSD readahead.

Finding: the batch-vs-decode divergence is **structural and spread across the whole
layer** (hc projections, attention QKV/out, router, shared expert, routed experts) — not
concentrated in one kernel. decode2_exact (exact, linear, ~49.5 ms) uses the decode
kernels ×2; verify_suffix_tops (sublinear, ~27.3 ms, not exact) uses the batch kernels.
The fused N=2 target = batch's weight-sharing cost (~27–30 ms) + decode's bit-exact
reductions, i.e. M=2 variants matching the M=1 reduction across ALL stages. This is a
multi-week Metal/ROCm/CUDA engineering effort on a large production engine; pinning the
exact divergence source to design the minimal fusion needs either deeper kernel-reduction
reading or a stage-level measurement. **Paused to align on slicing** (diagnostic/design
vs single-stage prototype vs full multi-session commit) — see pause note.

### 2026-07-12 — lock-exactness: strategy A chosen; margin guard deferred (later NO-GO in artifact 12)

Locked the exactness strategy for the N=2 milestone BEFORE any kernel build.

**Chosen: strategy A — single-family fused kernel, bit-exact by construction.** The
fused M=2 kernel uses the identical dequant + GEMM reduction as the M=1 decode path
(same FP accumulation order per row), so per-row logits are bit-identical to decode →
argmax never flips → exact greedy AND distribution-exact. **Fidelity-gate target:
`max_abs_logit_diff = 0` and token-for-token greedy match vs the decode path.**

**Deferred: strategy B (margin-guarded fallback) — NOT used for N=2.** Empirical basis
(from `artifacts/rejection_acceptance/verify_dist_probe_exactness.jsonl`, 156 positions,
90 probed cycles): flip rate 0.641% (1/156), median TV 0.0035 — but the logit-space
divergence is large: `max_abs_logit_diff` median **0.28**, max **4.56**, and the single
flip occurred at a cycle with divergence **1.75 logits**. A margin-guard safe enough to
catch that flip would need threshold ≥ ~1.75 logits; since small top-2 margins are common
(`conf_logits` shows ~0.29-logit top-2 gaps), it would re-verify a large fraction of
positions → eating the gain. The artifact also lacks the per-position top-2 divergence
needed to derive a tight threshold. The later per-position top-2 measurement is artifact 12:
threshold 0.25 misses the flip and 0.5 triggers on 17.1% of cycles, so strategy B is now closed
negative.

**Threshold reference (empirical):** the gate target is `max_abs_logit_diff = 0`. The
current batched baseline (median 0.28 / max 4.56 divergence, 0.64% flip, median TV
0.0035) is the divergence being eliminated. The doc's >5% re-verification abort applies
if the fused kernel shows non-bit-exact residual divergence.

### 2026-07-12 — Phase B orientation (read-only): assets verified, kernel surfaces located

Read the lead + the Phase A canonical summary + the spec model it feeds. Verified all
claimed assets exist: `./ds4-spec-bench` binary; `issue468/run_mtp_verifier_bench_long.py`;
`artifacts/mtp_phaseA_profile/summary.json` (Phase A headroom, 19–22 ms/cycle);
`artifacts/rejection_acceptance/verify_dist_probe_exactness.jsonl` (flip distribution,
median TV 0.0035 / flip 0.64%); model GGUFs (target 86.7 GB IQ2XXS, drafter 11.5 GB).
Machine clean (~88 GB free, no heavy proc); on `dspark-research` branch.

Engine kernel surfaces located (file:line):
- `metal_graph_verify_suffix_tops` — `ds4.c:21626` (batched, sublinear, NOT exact:
dispatches `metal_graph_encode_layer_batch` → the `kernel_mul_mv_ext_q8_0_f32_r1_*`
multi-token dense kernels + batched expert path).
- `metal_graph_verify_decode2_exact` — `ds4.c:21808` (exact, LINEAR: runs
`metal_graph_encode_decode_layer` ×2 in one command stream, one per token). Its own
comment states the thesis: "the generic batch prefill path is fast, but…small row-wise
differences in HC/MoE/output kernels are enough to flip future greedy tokens."
- Decode path: `metal_graph_encode_decode_layer` (`ds4.c:15198`) → dense
`kernel_mul_mv_q8_0_f32` (`metal/dense.metal:181`, M=1) + expert
`kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32` (`metal/moe.metal:1022`, single-token, N_R0=4);
dequant `dequantize_iq2_xxs` (`metal/moe.metal:278`).
- **Multi-token Q8_0 dense kernels already exist** — `kernel_mul_mv_ext_q8_0_f32_r1_2..5`
(`metal/dense.metal:912–915`), using `dequantize_q8_0_t4`. So a fused dense N=2 may be
partly present; the open question is whether the M=2 reduction is bit-identical to M=1.

Key orientation findings: (1) a **known-good exact reference already exists** — the
Lead 06 anchor-reuse path is bit-exact (greedy 10/10 byte-for-byte, temp>0
`max_abs=0`), usable as the fidelity-gate reference. (2) The fused-N=2 challenge is
*adding exactness to the batched cost level*, not reaching it — `verify_suffix_tops`
already hits ~27.3 ms at K=2 (code_4k, near the ≤30 ms target) but flips greedy tokens.
(3) **Discrepancy for the build task:** `metal_graph_encode_layer_batch` (`ds4.c:19647`)
itself routes through `metal_graph_encode_decode_layer` in some branches (19723, 19787),
so the batch/decode split is conditional — the exactness-divergence source must be pinned
to the specific kernel/branch in the build task, not assumed. No new measurement started.

### 2026-07-12 — doc refreshed for Phase B launch

Tidied this lead doc for Phase B: corrected the stale "resolved & archived"
header to "Phase A resolved → Phase B active"; folded in the milestone-2 session
evidence — the crossover finding (existing batched primitive is only break-even,
so a fused kernel must hit ~45 ms, below the sequential ~57 ms, not just below
the current 66 ms), the exactness measurement (median TV 0.0035, flip 0.64%;
the later artifact 12 disproved the then-assumed cheap margin fallback), and the
`decode2_exact`-is-linear ruling (the fused kernel must be
both sublinear AND exact). Tightened the net-effect criterion to make the
dependency stack explicit (needs anchor reuse + GPU drafter, not a better
drafter). Core thesis and prize estimate unchanged and independently
re-confirmed by the swing-term re-derivation.

### 2026-07-11 — Phase A profiling gate PASSED; recommendation: proceed to Phase B

Implemented retained `DS4_MTP_VERIFY_PROFILE` instrumentation and measured the shipped
verifier on the long-context corpus for K=3..5. The resulting artifact
(`artifacts/mtp_phaseA_profile/summary.json`) shows verifier wall time dominated by layer
execution, not host readback: for K=4, verify medians were ~65.9/66.5/67.1 ms while the
initial layer-execute medians alone were ~62.6/63.0/64.0 ms on
`code_8k`/`synthesis_8k`/`grounded_8k`.

Selected routed-expert bytes at K=4 were only ~4.8–5.4 GiB against a full-routed
72.56 GiB layer set, and the measured `verify_ms(K) - floor_ms(K)` headroom remained about
19–22 ms/cycle at K=3..5. That clears the lead's `~15 ms` proceed gate. Recommendation:
**proceed to Phase B fused low-K kernel work** if verifier acceleration remains a live
research path. Canonical summary:
`issue468/summaries/mtp_verifier_engineering_and_phaseA.md`.
