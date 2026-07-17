# Speculative-decode speedup model — DSPark drafter on the ds4 target

> **M3 UPDATE (2026-07-15) — the headline below is SUPERSEDED.** The full DSpark stack
> (committing batched verify + anchor-reuse + prefix-checkpoint + the GPU Metal drafter +
> the STS) now **BEATS plain ds4 by +4.9 %** (40.04 vs 38.16 t/s on the full 176-entry
> corpus; +4.8 % on the long-context baseline_corpus), and is **score-neutral** on the 92Q
> (61/92 vs plain 60/92, 90.2 % same verdict, net +1). This is the first config to beat
> plain locally. The win came from attacking both the draft cost (Metal drafter 45→7.6 ms)
> and the verify overhead (batched + prefix-checkpoint + anchor-reuse-for-Metal). The old
> "does NOT beat plain" conclusion held for the *exact* sequential-reuse verifier (M2);
> the committing batched verify + the GPU drafter changed it. **+20 % is still not reached —
> the verify dominates (~80 % of the cycle, ~16 ms/token). Lead 08's grouped gate+up prototype is
> bit-exact but slower, its margin-guarded exact fallback erases the current speedup, and fixed-work
> expert address placement changes routed cost by <2%, and exact production threadgroup variants
> move it by only 1-3%, so +20 % has no demonstrated verifier mechanism.**
> See `dspark_runtime_milestone_3_progress.md` for the full M3 record.

Date: 2026-07-06 (cycle-cost model); **holistic integration 2026-07-08** of Lead 01
(anchor-reuse falsifier), Lead 02 (confidence-scheduled verification), and Lead 03
(powered acceptance + realistic-trajectory); **Lead 04 Phase B (FP hidden-precision
ceiling) integrated 2026-07-10**; **Lead 06 + Lead 08 Phase A measured 2026-07-11**.
Model: `issue468/model_spec_speedup.py`; powered data + cycle-jump:
`issue468/artifacts/acceptance_powered/`; outputs:
`issue468/artifacts/spec_speedup_model/{summary.json,model_inputs.json}`.

## Headline (current best understanding)

**Under the realistic per-cycle trajectory, DSpark speculative decoding still does NOT beat
plain ds4 decode locally on the measured corpus.** Lead 06 made the anchor-reuse verifier
real, but the correctness-safe implementation that survived regression is an **exact
sequential-reuse** path, not the cheap folded verifier assumed by the old optimistic model.
It materially improves over shipped larger-K `--mtp`, but remains below baseline on every
measured corpus: on the full modeled 300-prompt benchmark, baseline averages **39.02 t/s**,
shipped K=4 **27.15 t/s**, and exact anchor reuse **32.67 t/s** (**+20.95% over shipped,
−16.26% vs baseline**); on the retained long-context K=4 cells it reaches
**30.81 / 31.69 / 28.85 t/s** versus baselines **36.34 / 37.46 / 33.65**. Confidence
scheduling therefore remains only conditional secondary material: it helps only if a much
cheaper verifier exists than the exact Lead 06 path. Lead 08 Phase A now gives that next
recommendation directly: verifier wall time is dominated by GPU layer execution and retains
roughly **19–22 ms/cycle** of measured headroom at K=3..5, so a fused low-K verifier kernel
is justified. The remaining acceptance levers are still a better drafter (training) and —
per Lead 04 — target hidden-state precision.

## The question and the gates

Can a DSpark-style speculative path deliver a **material local decode speedup** on `ds4`
with exact greedy output preserved? Primary gate ≥+20% vs baseline; secondary gate beat or
match the shipped `--mtp` path. This document is the cycle-cost model plus the measured
fixed-K acceptance and adaptive-scheduling evidence that feed it. There are **two precision
dimensions** to the drafter's input, both now measured: **drafter-weight** precision is
closed (Q4_K≈F16, `dspark_quantization_ceiling.md` — not the lever); **target hidden-state**
precision is open and measured by Lead 04 Phase B (see "Target hidden-state precision" —
native FP4/FP8 hiddens lift p1 ~+5 pp at float32, but the F16 deployment result reverses
sign; resolved in Phase C — capture faithful, the lever is real at +8–10% float32 speedup, F16-deployment-blocked, IQ2XXS-recoverability unproven).

## Notation & definitions

**The two acceptance estimators** (Lead 03's core correction — see "Acceptance: two
estimators" below for the full reasoning):
- **Sliding** estimator — averages the accepted prefix over **every position** on the
  greedy spine (treats each position as an independent cycle start). Easy to compute;
  **optimistic / diagnostic only.**
- **Cycle-jump** estimator — simulates real decode: each cycle accepts `a` drafts, then
  **advances by `a + 1`** (the `a` accepted + 1 correction/bonus token) to the next
  anchor; averages `a` over the cycles a decode *actually runs* (real decode never
  starts a cycle at the positions it jumped over). This is the **model-currency /
  realistic** value, and is lower than sliding because it lands on post-rejection
  *correction-anchor* cycles that sliding under-weights.

**Symbols used in the tables and formulas:**
- **`K`** — draft block size: tokens the drafter speculates per cycle.
- **`a`** (0..K) — accepted prefix: how many of the K drafts the target accepts before
  the first mismatch in a cycle.
- **`decode_ms`** (26) — wall time of one plain target decode (= 1000/38.49 t/s).
- **`draft_ms`** (10) — wall time of one DSpark drafter forward (given).
- **`verify_ms(K)`** — wall time of one target verify forward over a K-token suffix
  (cross-prompt median; K2:43.6, K3:59.7, K4:65.8, K5:74.5).
- **`E[a|K]`** — expected accepted drafts **per cycle** at block size K (the acceptance
  currency). Always qualified by estimator: **sliding E[a|K]** or **cycle-jump E[a|K]**.
- **`S(K)`** = `P(full accept)` = P(the first K drafts all match) = P(prefix ≥ K) — the
  full-block-accept probability that triggers the extra anchor decode in the optimized-
  verifier cost. Also has sliding vs cycle-jump variants (tables label which).
- **`speedup(K)`** — `(E[a|K]+1)·decode_ms / (draft_ms + verify_ms(K) + decode_ms·S(K))`;
  `>1` beats plain decode. Quoted as the cycle-jump value unless marked "sliding".
- **`p`** — uniform per-position match probability under a geometric acceptance model;
  **`p beat`** / **`p +20%`** = the `p` needed for speedup 1.0 / 1.2.
- **`E beat`** / **`E +20%`** — the `E[a|K]` needed for speedup 1.0 / 1.2 (uses the
  measured sliding `S(K)` for the full-accept decode penalty).
- **dynamic break-even `E[a|4]`** — the `E[a|4]` at which `speedup(K=4)=1.0` given the
  measured `S(4)` (= `(draft+verify(4)+decode·S(4))/decode − 1`).
- **deficit** — current cycle-jump `E[a|4]` minus the dynamic break-even.
- **`P(speed<1)`** — bootstrap probability the cycle-jump speedup is below 1.0.
- **`CI [lo, hi]`** — prompt-clustered bootstrap 95% interval (resamples prompts, not
  steps).
- **predicted-anchor / correction-anchor cycle** — a cycle whose anchor the drafter
  predicted correctly / mispredicted (a correction-anchor cycle follows a rejection and
  has lower acceptance).
- **optimized-verifier / shipped verifier** — the modeled anchor-*reusing* regime
  (fresh decode only on full-block accept) vs the actual `ds4 --mtp` (decodes every
  cycle, ~−18.9 pp at K=4).

## The model (cycle-cost)

A speculative cycle drafts K tokens, the target verifies them, and accepts a prefix (0..K).
The verify forward produces the **correction/bonus token at the rejection point, which is
the next cycle's anchor** — so a fresh anchor decode is needed **only on full-block
acceptance** (the *optimized-verifier* regime):

```
cost(reject)      = draft + verify                 (verify yields the anchor)
cost(full accept) = draft + verify + decode
E[cost/cycle]     = draft + verify_ms(K) + decode * S(K)     # S(K)=P(first K all match)
E[tokens/cycle]   = E[a|K] + 1
speedup(K)        = (E[a|K] + 1) * decode_ms / E[cost]
```

Costs (measured): `decode_ms=26` (1000/38.49 t/s), `draft_ms=10` (DSpark, given),
`verify_ms(K)` = cross-prompt median of `mtp_verifier_bench_long` (K2:43.6, K3:59.7,
K4:65.8, K5:74.5, K6:79.6). The shipped `ds4 --mtp` does **not** reuse the anchor — it pays
`decode+draft+verify` every cycle (~−18.9 pp at K=4), so the optimized-verifier projection
below is the relevant ceiling, not the shipped reality.

## Acceptance: two estimators (Lead 03's core correction)

`E[a|K]`/`S(K)` are the model's currency, and **how they are estimated matters**. Both are
now measured on a **300-prompt powered corpus** (Stage 2's 240 dolly/codealpaca/jsonex + 60
newly captured via the committed `--capture-dataset`; 128-tok temp=0 greedy spines) with a
**torch/MPS drafter port precision-gated at 100% draft-token agreement vs the numpy oracle**
(the numpy oracle's per-expert re-dequant crashed the machine at scale; a codex bug-hunt
also found+fixed a real `hc_post` broadcast bug in the torch port).

- **Sliding (position-uniform greedy prefix).** The easy-to-compute estimator: average the
  accepted prefix over all positions. **Optimistic / diagnostic only** — it uniformly samples
  positions and under-weights post-rejection *correction* cycles.
- **Cycle-jump (per-cycle, realistic trajectory).** The model-currency estimator: simulate
  real cycles advancing by `accepted+1`, so per-cycle accepted is measured on the trajectory
  the decode actually follows (including correction cycles). This is what the speedup formula
  consumes.

The cycle-jump is **lower** than the sliding estimate by ~3 pp at K=4 — the *realistic-
trajectory bias* the model had flagged but never measured. Sliding gives the optimistic edge
of the band; cycle-jump the realistic edge.

## Current numbers (300-prompt corpus, optimized verifier)

| K | verify_ms | sliding E[a\|K] | sliding S(K) | sliding speedup | cycle-jump E[a\|K] | cycle-jump S(K) | **cycle-jump speedup** |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 2 | 43.6 | 1.442 | 0.652 | 0.900 (−10.0%) | 1.408 | 0.633 | **0.893 (−10.7%)** |
| 3 | 59.7 | 1.955 | 0.514 | 0.925 (−7.5%) | 1.886 | 0.483 | **0.912 (−8.8%)** |
| 4 | 65.8 | 2.337 | 0.382 | 1.012 (+1.2%) | 2.198 | 0.340 | **0.982 (−1.8%)** |
| 5 | 74.5 | 2.609 | 0.272 | 1.025 (+2.5%) | 2.422 | 0.232 | **0.983 (−1.7%)** |

- **K=4 cycle-jump (the headline):** speedup 0.982×, prompt-clustered bootstrap CI
  [0.971, 0.993], **P(speed<1) = 0.999**. The dynamic break-even E[a|4] at S(4)=0.340 is
  **2.256**; the deficit is **0.058 accepted drafts/cycle** (deficit > CI half-width, so
  "below baseline" is signed). Read the sliding column as the optimistic edge only. These
  are the unscheduled fixed-K reference points for the scheduling results below.

> **2026-07-13 Lead 08 Phase B floor-clearance update (codex-gated A+B).** Two robust
> refinements to the model's inputs, both decision-grade:
> - **Decode effective bandwidth ≈ 410–450 GB/s** (resolved from byte data: dense 8.20 +
>   experts 1.70 + KV ≈ 10.4 GiB/token at decode_ms 26). The earlier "~300 GB/s" assertion
>   (used to floor verify_ms) was conservative; the ≤250 GB/s cliff is not real.
> - **Gate-clearance threshold: `verify_ms(4) ≤ 50.5 ms`** clears +20%
>   (speedup = 26·3.198/(10 + verify_ms(4) + 8.84) ≥ 1.20). The current shipped verify_ms(4)
>   = 65.8 ms → 0.982× (below). The **K=4 verify floor at decode bandwidth ≈ 15.5 GiB /
>   428 GB/s ≈ 39 ms → ~1.44× (clears)**; break-even ≈ 329 GB/s.
>
> Verdict on the fused-verifier path (FINAL, post decisive measurements + gate C):
> **NO-GO via swaps → bounded build attempt with a hard exit gate.** The existing bit-exact
> verifier (DSpark sequential) is **~0.85× baseline on the 8k corpus** (NO-GO); gate/up are
> bit-exact given identical inputs; but **NO kernel-selection/config swap yields a sublinear
> bit-exact K=4 verifier** (verify_suffix_tops is sublinear-not-exact; decode2_exact is
> exact-N=2-linear; no config reroutes batch HC/compressor/attention onto decode reductions)
> → a gate-clearing verifier requires **novel kernels**. The cost side is favorable-but-
> unproven (floor ~39–43 ms → 1.34–1.44× *if* built sublinear+exact + stack). Decision:
> attempt the sublinear bit-exact batch-path build as a **bounded effort with a hard exit
> gate** — GO is **unconfirmed** until an end-to-end K=4 bit-exact verifier profiles
> `verify_ms(4) ≤ 50.5 ms`. See `summaries/lead08_phaseB_floor_clearance_verdict.md`.

> **2026-07-17 Lead 08 iterations 2/3: controlled opportunity -> prototype NO-GO.**
> Post-M3 the verify is **89.5 % of the cycle** (62.07 ms of 69.38 ms); the raw +20% threshold is
> about 8.7 ms of cycle saving, while `verify_ms(4) <= 50.5 ms` retains the hard integration gate.
> A controlled K=4 production `routed_moe_batch` probe fixed physical work at 24 pairs and varied
> unique experts 6/12/18/24. Total routed cost was 5.451/7.277/10.334/13.489 ms/layer: production
> preparation is already union-aware, so the earlier physical-pair-only attribution and **6-8 ms
> "reliable" prize are withdrawn**. However, profiled gate+up at the realistic partial-overlap
> shape was 4.642 ms/layer versus 4.787 fully disjoint (**97%**) despite 25% fewer unique experts;
> the physical-row GPU kernel barely benefits from sparse doubletons. Down was smaller (0.695 vs
> 0.815). This justified exactly one expert-major gate+up prototype with threadgroup-spilled
> partial sums. Artifact 11 records its result: all isolated routed tensors are bit-exact against
> production, but target-shape gate+up regresses **4.869 -> 5.986 ms/layer (+22.9%)**, rather than
> improving >=15%; excluding layer 0 regresses 35.8%. The grouped branch therefore stops, the final
> verifier integration gate is not run, and there is no demonstrated deployable load-sharing prize.
> Artifacts `lead08_reassessment/10_iter2_production_batch_overlap.md` and
> `lead08_reassessment/11_iter3_grouped_gateup_prototype.md`.
>
> Iteration 1's simultaneous M=2 design remains structurally NO-GO (18-vs-12 streams + doubled
> live state), but its reported 4.4x and derived 2.9x magnitudes are withdrawn: the M1 comparator
> was cache-warm and not cold-equivalent. Fidelity remained sub-ULP with zero observed argmax flips.
>
> **2026-07-17 Lead 08 iteration 4: margin-guard fallback NO-GO.** On the retained 10-prompt
> batch-vs-exact workload, the one greedy flip has batch top-2 margin 0.3755, so threshold 0.25
> misses it. Threshold 0.5 catches the observed flip but guards 27/158 probe cycles and measured
> exact replay adds an optimistic 7.0 ms/all cycle. Applied to the live stack, 40.04 t/s projects
> to 36.37 t/s, below plain 38.16; top-2/rollback overhead is not included. One flip also cannot
> prove that 0.5 catches future reduction errors. The policy is not implemented. Artifact
> `lead08_reassessment/12_iter4_margin_guard.md`.
>
> **2026-07-17 Lead 08 iteration 5: expert-address locality NO-GO.** A production K=4 probe held
> 24 physical rows, 18 unique experts, and duplicate multiplicities fixed, then compared contiguous,
> same-set permuted, and slab-wide strided expert placement. Versus contiguous, total routed cost
> changes only +1.9%/+0.7%; matched gate+up changes +1.6%/+0.9% (+1.8%/+1.4% excluding layer 0).
> This closes software-visible expert remapping/packing, not hardware bandwidth-vs-compute
> attribution. Artifact `lead08_reassessment/13_iter5_address_locality.md`.
>
> **2026-07-17 Lead 08 iteration 6: production address-kernel geometry NO-GO.** At the same fixed
> K=4 18/24 shape, NSG 1/4/8 are all bit-exact against production NSG 2. The best total effect is
> -3.2%, while matched gate+up effects remain between about -2% and +1% and shrink further excluding
> layer 0. These low-single-digit effects miss the >=15% gate, so the default remains NSG 2. Artifact
> `lead08_reassessment/14_iter6_addr_nsg_geometry.md`.
>
> **2026-07-17 Lead 08 iteration 7: production row-tile geometry NO-GO.** At fixed NSG 2 and the
> same K=4 18/24 workload, tiles 2/8 are bit-exact against production tile 4 but +1.0%/+0.3% slower
> total. Tile 8's -2.5% matched gate+up effect becomes +0.05% excluding layer 0; tile 2 becomes
> +1.3%. No steady-state gain exists and the >=15% gate is missed. Artifact
> `lead08_reassessment/15_iter7_addr_row_tile.md`.
>
> **2026-07-17 Lead 08 iteration 8: cache-residency carrier NO-GO.** Exact cold-to-resident
> replay cuts the SSD selected-address path from 11.094 to 1.353 ms/layer (-87.8%), but that path
> is explicitly incompatible with DSpark. On the current compatible mapped-weight path, cold and
> replay are both 0.566 ms/layer (-0.04%). Actual batched-verifier profiling confirms reusable
> expert locality, but there is no current-path residency miss to remove. Treat DSpark+SSD as a
> separate runtime lead, not a Lead 08 speedup. Artifact
> `lead08_reassessment/16_iter8_cache_residency.md`.
>
> **2026-07-17 Lead 08 iteration 9: locality-profiler correction.** Iteration 8's initial
> batched-selection profiler read reused router buffers before command-buffer completion. Per-layer
> GPU snapshots preserve complete cycle trajectories and correct the three-family LRU hit curve to
> 57.3%/72.3% at capacities 16/32. The mapped cold/replay result and speed model are unchanged, so
> the carrier remains a current-path NO-GO. Artifact
> `lead08_reassessment/17_iter9_profiler_correction.md`.
>
> **2026-07-17 Lead 08 iteration 10: existing batch shared-family NO-GO.** Re-baselining target M=1
> onto the layer-major batch graph does not make its end-to-end M=K state transition exact against
> batch M=1: only 5/10 retained
> prompts match token-for-token. Batch M=1 is 32.989 versus shipped decode 37.902 t/s (-13.0%);
> speculative reaches 37.407, only +13.4% over the slower target and -1.3% versus shipped decode.
> This closes the existing shared graph, not the unbuilt exact hybrid. Artifact
> `lead08_reassessment/18_iter10_batch_m1_target.md`.
>
> **2026-07-17 Lead 08 iteration 11: same-frontier localization.** A noncommitting M=K probe plus
> batched-M1 replay finds 2 argmax flips in 349 accepted-row comparisons. At a reproduced flip,
> layer-0 `hc_attn_pre` and `attn_norm` are bit-identical and `q_lora` is the first captured
> difference. The post-audit implementation hard-fails on restore failure and reproduced the corpus
> and captures; repeat audit passed, making bounded exact-Q/KV projection the next falsifier. The synchronized
> diagnostic's throughput is not economic evidence. Artifact
> `lead08_reassessment/19_iter11_same_frontier_localization.md`.
>
> **2026-07-17 Lead 08 iteration 12: Q/KV boundary moved at low naive cost.** Verifier-scoped
> row-wise M=1 Q/KV makes layer-0 row-0 Q/KV projection, normalization, and KV RoPE/storage
> bit-identical; `Qcur` is now the first captured difference. Four order-balanced process-first K=4
> samples observe a noisy naive all-layer delta of +0.56 ms mean / +0.69 ms median layer execution;
> full-corpus samples change downstream routed work. This is an upper-bound estimate, not an isolated
> cost or unavoidable addition to the ~43 ms optimistic floor. Artifact
> `lead08_reassessment/20_iter12_rowwise_qkv.md`.

- **Corpus-dependent** (cycle-jump K=4 speedup): jsonex **+2.3%**, codealpaca −1.9%, dolly
  −5.6%. The mix is on the easy side — the old 10-prompt code/synthesis exactness corpus was
  E[a|4]=2.175 (harder than all three families), so a code/synthesis-heavy deployment would
  be worse, not better.
- **Trajectory bias is real and structural:** correction-anchor cycles (drafter mispredicted
  the anchor) have E[a|4]=1.994 vs predicted-anchor 2.427; real decode interleaves them,
  pulling per-cycle acceptance from the sliding 2.337 down to the cycle-jump 2.198.

## Anchor reuse (Lead 01 + Lead 06) — assumption de-risked, cheap path still open

Lead 01 de-risked the **acceptance axis** of stale-hidden reuse: the drafter tolerates the
last-accepted-position hidden plus the correction token as anchor input, with no large
acceptance collapse observed (`summaries/anchor_reuse_falsifier.md`). Lead 06 then resolved
the **correctness** question on ds4: a real env-gated anchor-reuse path now exists and passes
both the greedy exactness corpus and the retained temp>0 logit/distribution parity gate
(`summaries/mtp_verifier_engineering_and_phaseA.md`).

What Lead 06 did **not** prove is the old optimistic economic assumption. The surviving path
is an exact sequential verifier substrate, not a cheap folded verifier. On the full modeled
300-prompt corpus it recovers about half of the shipped K=4 loss (+20.95% over shipped) but
still remains **−16.26% below baseline**, so the remaining gap is now squarely verifier cost,
not correctness or acceptance realizability.

## Q1 — acceptance required to beat baseline / clear +20% (sliding framing)

Using the sliding S(K) for the full-accept decode penalty (the model's Q-tables;
`E`=accepted drafts/cycle needed, `p`=uniform per-position match prob, −1=unreachable):

| K | E beat | E +20% | p beat | p +20% | cycle-jump E (real) |
|--:|--:|--:|--:|--:|--:|
| 2 | 1.713 | 2.256 | unreachable | unreachable | 1.408 |
| 3 | 2.194 | 2.833 | 0.890 | unreachable | 1.886 |
| 4 | 2.298 | 2.957 | **0.792** | 0.940 | **2.198** |
| 5 | 2.522 | 3.226 | 0.783 | 0.890 | 2.422 |

- The cycle-jump current E[a|4]=**2.198** is **below** the sliding beat-even E=2.298 (and
  below the cycle-jump dynamic break-even 2.256) — i.e. the realistic drafter is short of
  beating baseline by ~0.06–0.10 drafts/cycle, not the "~0.03" the old sliding/10-prompt
  framing implied.
- The **+20% gate** needs p≈0.94 at K=4 / 0.89 at K=5 (~+13–17 pp over current ~0.79) —
  reachable only with a substantially better drafter (training, not quantization).

## Q3 — can a 4-node draft tree help? No.

A verify of N nodes costs `verify_ms(N)` regardless of shape (bandwidth floor). Ceilings at
**perfect** acceptance: linear-4 +27.7% (current cycle-jump → 0.982×); best 4-node tree
(spine3 + repair) **+2.2% even at perfect acceptance**, doubly penalized (branches spend
verify nodes without extending the committed prefix, AND raising pos-3 coverage increases
the full-accept decode penalty). **Do not branch — use a linear chain;** no 4-node tree
beats the gate.

## Adaptive scheduling in the same model (Lead 02)

Lead 02 asks whether the existing confidence head can improve the local economics by
choosing a shorter verify span cycle-by-cycle. The policy evidence is fully offline:
confidence is extracted from the oracle / torch-MPS carrier, calibrated with train-fit STS,
thresholds are chosen on `eval`, and then replayed unchanged on fresh `lead3` under the
same two verifier accountings used elsewhere in this dossier.

| policy (fresh `lead3`) | shipped accounting | anchor-reuse accounting |
|---|---:|---:|
| fixed-K4 | 0.8123x | 0.9810x |
| **STS selected threshold** | **0.8646x** | **1.0375x** |
| **STS expected-opt** | **0.9205x** | **1.0523x** |
| oracle ceiling | 1.1297x | 1.2447x |

- Under **shipped** accounting, adaptive scheduling does **not** rescue the path. The
  table's `0.8646x` uses the anchor-reuse-selected threshold `0.08` held fixed across both
  accountings; even the threshold selected specifically for shipped costs (`0.52`) reaches
  only **0.8987x** on fresh `lead3`.
- Under **anchor reuse**, scheduling improves on unscheduled fixed-K4 but only into a
  narrow, fragile positive band. The clean frozen-threshold result is **1.0375x**, below
  Lead 02's predeclared `+5–10%` stacking tier. The **1.0523x** expected-opt figure is a
  useful diagnostic upper-ish policy, but it is not deployable evidence.
- The anchor-reuse positives are highly sensitive to residual folded-verifier overhead: the
  frozen-threshold gain has about **3.01 ms/cycle** of headroom and expected-opt about
  **3.89 ms/cycle**. Adding `+4 ms/cycle` drops them to about **0.988x** and **0.999x**.

So in the speedup model, scheduled verification is not an overlay or an independent route
to viability. It is best represented as a **conditional secondary increment** that matters
only if a much cheaper verifier than the current exact Lead 06 path exists.

## Target hidden-state precision — the FP ceiling (Lead 04 Phase B + C)

The acceptance numbers above are all measured on the **local IQ2XXS target** — i.e. the
drafter consumes IQ2XXS-degraded hidden states. Is part of the ~0.06 drafts/cycle deficit
attributable to IQ2XXS degrading the *target's* hiddens below their native (FP4/FP8)
precision? The DSpark drafter was distilled against native served precision
(`inventories/dsv4_flash_dspark_model.md`), so if native hiddens give higher acceptance
than IQ2XXS hiddens, target-hidden quantization is a recoverable lever.

**Method:** captured native FP4/FP8 target data (HC hiddens at layers 40/41/42 + greedy +
top-128 logits) on Lead 03's 300-prompt corpus via vLLM on Modal H200:2
(`run_lead04_modal/capture_hc_modal.py`); ran the local drafter on the FP hiddens; paired
FP p1 against the retained Q2 p1 per prompt; prompt-clustered bootstrap CI.

**Result (n=299; Δp1 = float32-FP − float32-Q2).** The Q2 reference is dtype-invariant
(float32-Q2 == F16-Q2 exactly, verified on 240 prompts, drafts byte-identical — the
drafter's argmax is robust to f16/f32 on Q2 hiddens), so the float32-FP-vs-f16-Q2
comparison IS the consistent-dtype comparison:

| | FP (native) | Q2 (IQ2XXS) | Δ | CI95 |
|---|---|---|---|---|
| mean p1 | 0.846 | 0.793 | **+5.28 pp** | [+3.81, +6.56] |
| codealpaca | | | +6.96 pp | [+4.3, +8.9] |
| dolly | | | +2.73 pp | [+1.3, +4.1] |
| jsonex | | | +6.15 pp | [+2.8, +8.8] |

**Phase C falsification (hidden-capture fidelity — RESOLVED to “likely faithful”).**
Code-reads on vLLM v0.24.0 (`/Users/lobanov/Projects/vllm`) verified the capture mechanics:
the decoder layer returns `(x, residual, post_mix, res_mix)` matching the hook’s unpack,
and `mhc_post_tilelang` writes to a fresh `out` (not in-place, so the hook’s `.clone()`s
are harmless). The F16/deployment anomaly (below) is **F16 numerical, not a capture bug**:
no NaN/inf, identical output magnitudes (head-out max 127973 for F16≈float32 on FP), and
the effect is argmax-flipping on near-tie logits — the FP-hidden distribution produces
closer logits that F16 rounding flips, while Q2 hiddens (the drafter’s deployed input) give
robust argmaxes. The capture is therefore likely faithful; the +5.3 pp is real at float32.

**Phase C cycle-jump (the full acceptance trajectory — the user’s hypothesis CONFIRMED).**
First-token p1 understates the FP benefit: the cycle-jump E[a|4] amplifies it. Measured on
the same 299-prompt corpus (Lead 03’s cycle-jump estimator; `dspark_oracle/analyze_phaseC.py`):

| | FP float32 | FP F16 (deploy) | Q2 (ref) |
|---|---|---|---|
| p1 | 0.855 | 0.596 | 0.793 |
| cycle-jump E[a\|4] | **2.741** | 1.693 | 2.198 |
| cycle-jump S(4) | ~0.57 (meas.) | — | 0.34 |
| speedup(K=4) | **~1.08–1.10× (+8–10%)** | ~0.83× | 0.982× |
| confidence separation | +0.247 (calibrated) | — | (Lead 02) |

At **float32, FP hiddens clear baseline by ~+8–10%** (E[a|4]=2.74, S(4)≈0.57 vs Q2
2.198/0.34) — the “later-position acceptance matters even if p1 doesn’t” hypothesis holds:
the cycle-jump gap (+0.54 drafts/cycle) is much larger than the p1 gap (+6 pp) suggests.
The drafter’s confidence head is calibrated on FP hiddens (accepted 0.87 > rejected 0.62,
separation +0.247), so confidence-scheduled verification (Lead 02’s framework) is viable
on FP. At **deployment precision (F16/Q4_K), FP hiddens are WORSE** (E[a|4]=1.69, 0.83×)
— the F16 drafter’s argmax-flipping on the FP distribution destroys the benefit.

**Verdict (Phase B+C, codex-amended): native served-precision hiddens expose a large
float32 acceptance ceiling (~+8–10% cycle-jump speedup, clearing baseline); the current
F16 deployment cannot use it; recoverability on an IQ2XXS target is UNPROVEN.** This is a
drafter-precision blocker, not a capture blocker: the captures are faithful (mechanics
verified), and a float32 drafter on native hiddens would deliver ~+8–10%. The local IQ2XXS
speedup verdict (0.98×) stands for the *current* deployment (IQ2XXS target + F16 drafter),
but target-hidden precision is now a **quantified live lever**. **Critical caveat (codex):**
the +8–10% compares a vLLM-native trajectory to a ds4-IQ2XXS trajectory — a **crossed
oracle** (drafter on FP-hiddens×IQ2XXS-labels, and vice versa, on common prefixes) is needed
to separate the hidden-precision gain from label/trajectory drift before attributing the
lever to hidden precision. Practical routes to capture some of the gain (target stays
IQ2XXS): (A) drafter body-LoRA adapter trained on IQ2XXS labels with native as an auxiliary
distillation teacher [codex’s most-promising pick, F16-deployable]; (B) feed the full
[4,4096] HC residual instead of the mean [needs probe first — the ceiling already used the
mean]; (D) a learned IQ2XXS→native-like hidden dequantizer [Lead 04 has the pairs] + a
float32 drafter. (C) float32-drafter-alone is useless on IQ2XXS (f32-Q2==f16-Q2). Caveats:
corpus is the easy side (dolly weakest); the +8–10% uses a 30-prompt S(4) estimate (full-300
S(4) unmeasured); F16 drafter cannot handle native hiddens (argmax-flipping).

## Verdict, levers, and open scope

**Verdict:** with the current drafter on the measured corpus, DSpark does not beat plain ds4
decode locally at any tested fixed K under the realistic trajectory (K4/K5 optimum ~0.98×,
significantly below baseline), and confidence scheduling does not change that under shipped
accounting. Under anchor reuse, scheduling rises only to a fragile low-single-digit positive
band on fresh replay, so the case for a local speedup still requires a stack of unverified
gains:

1. **Anchor-reusing verifier economics** (Lead 06 + Lead 08): correctness is now proven, and
   the measured exact-reuse path is worth about **+21% over shipped K=4** on the modeled
   300-prompt corpus, but it is still **−16% vs baseline** because it falls back to exact
   sequential verification. Phase A profiling shows the dominant remaining cost is verifier
   layer execution, with roughly **19–22 ms/cycle** of headroom at K=3..5, so the next
   question is not "can reuse be exact?" but "can verifier cost be driven toward the
   ~45 ms/K4 band without breaking exactness?".
2. **A materially better drafter** (Lead 07 upstream quality / training): the binding
   constraint is per-cycle acceptance; **drafter-weight** precision is exhausted (Q4_K≈F16),
   but **target hidden-state** precision is NOT closed — Lead 04 Phase B measured a ~+5 pp
   native-vs-IQ2XXS gap at float32 and, per Phase C, a **+8–10% cycle-jump speedup at
   float32** (E[a|4]=2.74 vs Q2 2.198) — the largest measured lever, but F16-deployment-
   blocked (the F16 drafter argmax-flips on native hiddens); see
   "Target hidden-state precision" above.
3. **Drafter-state pollution** (Lead 06): the cycle-jump measures anchor-token difficulty
   along a *linear* trajectory; it does NOT model the drafter's KV/state being polluted by
   rejected drafts, tree/branching, or adaptive policies — the realistic trajectory could
   be worse than measured.

**Scope / caveats:**
- `verify_ms(K)` is the MTP verifier; DSPark would reuse the same target verifier. The
  bandwidth floor (verify≈decode at K=2, super-linear past ~K=8 from MoE expert-union
  growth) is prompt-independent and unchanged.
- Corpus is dolly/codealpaca/jsonex (easy side); a code/synthesis deployment is harder.
  Acceptance is temp=0 greedy; temp 0.5/1.0 drops E[a|4] ~7 pp.
- The cheap optimized-verifier projection assumes a verifier much cheaper than the current
  exact Lead 06 path. Exact reuse is implemented; cheap reuse remains open.

Net: the realistic answer to "can DSpark beat baseline locally?" is still **no** with the
current drafter on this target. What changed is that the verifier question is now split in
two: **exact anchor reuse is real**, but **cheap verifier economics are not**. The
decision-grade open questions therefore move to Lead 08 Phase B style kernel work (can the
verifier be pushed down toward the measured headroom without breaking exactness?) and Lead 07
(can the drafter's per-cycle acceptance rise enough to clear the remaining deficit — and can
the Lead 04 +8–10% float32 FP-ceiling lever be realized, via a float32 drafter / body-LoRA /
hidden-dequant?).

## Milestone 3 update (2026-07-14): committing batched verify — measured + the exactness split

Measured (warm, exactness corpus, ds4-spec-bench): plain 37.20 / seq 16.29 / **batched 20.19
t/s**; `verify_ms` 52.4 -> 43.4 ms (sublinear, ~9 ms saved at `verify_n`~2). The committing
batched verify IS sublinear-faster than the sequential short-circuit at the STS `verify_n` (~2).

Exactness split (the key correction):
- The batched verify's 0.64%-level argmax flips **propagate to committed output when engaged**:
  56.6% token divergence on the benchmark, 40% on the exactness corpus. (The earlier "0% on the
  ds4-eval benchmark" was plain-vs-plain — ds4-eval didn't engage DSpark; now fixed.)
- **But score-neutral on the 92Q benchmark (greedy):** plain 60/92 (65.2%) vs DSpark-batched
  64/92 (69.6%), net +4, 89.1% same verdict (codex-verified: NO functional equivalence when
  engaged, but the divergence doesn't hurt the answer scores).
- **temp>0 distribution-exact** via the exact sequential verify (milestone-1: `max_abs=0`).

Implication: the "cheap verifier" question splits further — the batched primitive is sublinear +
score-neutral but NOT committed-output-exact. Lead 08 artifact 12 subsequently showed that the
first margin threshold catching the observed flip adds enough exact replay to fall below plain, so
strict exactness needs a different mechanism. The +20% gate still needs the full stack (batched verify +
anchor reuse + GPU drafter); levers 2/3 not started. Net: "can DSpark beat baseline locally?" is
still **no** on the delivered portion (batched 20.2 t/s = 0.54x baseline); the score-neutral
divergence makes relaxing greedy exactness a **viable interim strategy** pending verifier work.
GPU push (`metal_graph_push_dspark_hidden`) is a pre-existing general bug (CPU fallback, cheap) —
separate follow-up.
