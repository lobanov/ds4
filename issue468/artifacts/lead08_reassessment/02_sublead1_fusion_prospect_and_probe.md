# Lead 08 re-assessment - sub-lead 1: verify-cost decomposition (RESOLVED 2026-07-17)

Date: 2026-07-15; **revised 2026-07-16** after the codex avenues/experiments review
(`issue468/artifacts/dspark_codex_reviews/2026-07-16_gpt55_xhigh_lead08_reassess_avenues.md`).
Goal: `mrmkwnp6-6n9z9x`. Status: **resolved by artifacts 10-18.** The production overlap probe
authorized one grouped prototype; that prototype regressed 22.9%. The margin fallback was
uneconomic. A later fixed-work production microbenchmark changed only expert-address locality and
found <2% total/stage sensitivity, closing software-visible remapping/packing. The probe design below
is retained as historical rationale, not an open execution plan. Artifacts 14/15 additionally find
no material gain from repacking or retiling the production address kernel: all variants are exact,
but effects stay within a noisy 1-3% or regress. Artifact 16 finds an exact residency prize only in
the SSD selected-address runtime, which is incompatible with DSpark; the compatible mapped path has
no measurable cold/replay gap.
Artifact 17 corrects the original locality profiler using post-completion per-layer GPU snapshots;
the corrected hit curve is lower but does not change the carrier verdict.
Artifact 18 falsifies the existing batch graph as a shared M=1/M=K exactness family; the original
decode-order exact hybrid remains unbuilt.

**Why revised (codex, independently verified):** (a) the "190 GB/s → scattered access →
recoverable to decode bw" attribution is **unsupported** — the logged "selected GiB" is
unique-union bytes (ds4.c:12401 logs `unique` + `selected`), not physical per-pair traffic,
the kernel still reloads per-pair (ds4_metal.m:20945, `z = n_tokens × n_selected`), the slope
is inconsistent (K3→K4 ≈186 vs K4→K5 ≈56 vs others ≈88–135 GB/s), and cache-hit-vs-miss is
unresolved; (b) the de-dup ceiling is only **~21–30%** (high expert overlap: union ≈19–21 of
24–30 pairs at K4–5), not ~40%; (c) the proposed synthetic-load microbench is **not runnable
without a new kernel**, and the compute/bandwidth split needs Instruments/GPU counters.

## 0. Corrected framing (unchanged) — a verify saving amplifies ~1:1 into throughput

The verify is **89.5% of the cycle** (62.07 ms of 69.38 ms total, full corpus — verified).
A saving X → throughput gain X/(69.38−X), stacking on +4.9%:

| verify saving X | over current full stack | vs plain (38.16) |
|---|---|---|
| 5 ms | +7.8% | +13.1% |
| 8 ms | +13.0% | +18.6% |
| **10 ms** | **+16.8%** | **+22.6%** ← clears **+20%** |
| 12 ms | +20.9% | +26.8% |
| 16 ms | +30.0% | +36.3% |

**Target: ~10 ms off the verify = +20% over plain.** (Conservative: the STS picks verify_n
pre-verify from a fixed threshold, so this holds acceptance constant; a threshold re-tune after
cheaper verify would raise verify_n → more gain.)

## 1. The fusion prospect — two mechanisms (TEMPERED post-codex)

Traffic is weight-dominated (activation round-trips ~0.5 MB/token/layer, negligible vs ~GB
weights — codex confirmed the attention→FFN activation fusion is sub-ms, ~5 MiB) → the
register/cache reuse that matters is the **dequanted expert weights**.

1. **Dequant-once-apply-to-all-tokens** (load-sharing / de-dup). Saves the redundant
   per-(token,expert) dequant + fetch. **Reliable prize ~6–8 ms — BUT only if the redundant
   loads are DRAM cache-misses; if L2-hit, only the dequant compute is saved (smaller).** The
   de-dup ceiling is **~21–30%** of the routed cost (union ≈19–21 of 24–30 pairs at K4–5; high
   expert overlap — codex). Confirmed real: moe.metal:1257 dequants per-(token,expert); the
   dense matmul IS shared via `r1_2..5` (dense.metal:912–915).
2. **Coalesced union access → decode bandwidth.** The original attribution was unsupported.
   Artifact 13 later falsified its simple software-layout form: contiguous versus same-set permuted
   or slab-wide strided placement changed production routed cost by <2%. A different hardware-level
   mechanism would require new evidence.

**Revised prize: reliable ~6–8 ms (de-dup → +13–18.6% over plain, borderline +20%);
speculative up to ~16 ms (→ +36%) IF the bandwidth attribution holds. Clearing +20% is
uncertain until the bandwidth question is settled.**

## 2. The crux — why is the verify expert stream slow? (attribution UNSUPPORTED, codex-verified)

The prior "190 GB/s because scattered access" came from the K3→K4 slope of the logged "selected
GiB". Codex (verified): that GiB is **unique-union bytes**, not physical per-pair traffic; the
slope is **inconsistent** (186/56/88–135 GB/s); the instrumentation does NOT separate IQ2XXS
ALU/dequant, address divergence, occupancy, L2 hit rate, or physical memory traffic. Sharper
hypotheses + their decisive test:

| hypothesis | decisive test | prize |
|---|---|---|
| physical per-pair loads are DRAM cache-misses (bandwidth-bound, redundant) | L2 hit rate (Instruments); time tracks physical pairs | de-dup ~6–8 ms (mechanism 1) |
| L2 hits → dequant-COMPUTE-bound (the "186 GB/s" is misleading) | ALU utilization (Instruments); time doesn't track bytes | dequant load-sharing (smaller) |
| scattered access fixable by coalescing → decode bw | counters: memory-bound + low L2 + poor coalescing | ~16 ms (mechanism 2, speculative) |
| fundamental (batch-attn scattered KV / inherent dequant) | attention+dense share bounds it | TBD |

## 3. Sub-lead 1 — the REVISED probe (post-codex; 5 measurements)

The synthetic-load microbench is **dropped** (not runnable without a new kernel). The
stage-profile sweep is **FREE** (the 4 flags already exist: DS4_MTP_VERIFY_PROFILE,
DS4_MTP_VERIFY_EXPERT_PROFILE, DS4_METAL_LAYER_STAGE_PROFILE, DS4_METAL_MOE_ONE_STAGE_PROFILE).

1. **Fixed-K stage-profile sweep** (K=2..5, the 4 flags on the full-stack bench). **Bounds the
   real prize**: routed gate/up/down ms (isolated), attention share, dense/share cost. **The one
   decisive experiment to run first** (codex). FREE.
2. **Pair-vs-unique correlation**: from the profile's `unique` + `n_ids = verify_n × 6`,
   correlate routed stage time with physical pairs vs unique experts. **If time tracks pairs →
   de-dup helps (mechanism 1 real). If it tracks unique bytes poorly → coalescing/compute/
   counters needed.** (The decisive fork between mechanism 1 and 2.)
3. **Instruments/Metal GPU counters** on the batch `addr` kernel vs the decode `id` kernel: real
   memory throughput, **L2 hit rate**, occupancy, ALU utilization. **The ONLY way to settle
   bandwidth-vs-compute** (in-tree timings cannot). Counter inventory later found only
   `GPUTimestamp` on this host. Artifact 13's controlled locality timing closes address remapping,
   but does not supply this hardware attribution.
4. **Readahead/overlap flag sweep** (decode readahead ds4.c:14596; batch FFN hooks ds4.c:19466):
   toggle the selected-batch addr/readahead/shared-overlap flags, compare routed stage time. **A
   cheap pre-kernel avenue** (medium/high prize, low effort) — may win ms without any fused kernel.
5. **Exactness margin probe**: log top1/top2 margins for rows where batched vs exact disagree (or
   nearly). Simulate a margin-guarded fallback rate at thresholds 0.25/0.5/1.0/1.75. **May compose
   with the cost work better than full exact fusion** (revives the Phase-B margin-guard strategy).

   **Resolved 2026-07-17 (artifact 12): NO-GO.** Threshold 0.25 misses the known flip; 0.5 catches
   it but guards 17.1% of probed cycles and adds at least 7.0 ms/cycle, erasing M3's advantage.

## 4. Decision rule (revised)

- **Run 1+2 first (free)** → bound the routed-expert prize + settle the de-dup-vs-coalescing fork.
- **GO (build the M=2 grouped prototype, INCLUDING down/sum6)** if: routed gate/up/down is
  ~25–30 ms isolated AND (time tracks physical pairs [de-dup real] OR Instruments shows
  memory-bound + recoverable [coalescing real]).
- **Historical CONDITIONAL HOLD** (codex's verdict at this point): the prize appeared real but +20%
  clearance depended on bandwidth attribution. Artifacts 10-15 subsequently close the tested
  grouped, margin, address-layout, and production launch/tile-geometry mechanisms negative.
- **NO-GO / redirect** if: routed cost is dequant-compute-bound (not bandwidth) with a low de-dup
  ceiling → prize too small → redirect to the readahead/overlap sweep (4), STS re-tune, or the
  margin-guard exactness path (5). Subsequent artifacts close both readahead and margin guard
  negative.
- **Only then build** the M=2 grouped routed kernel — it MUST include down/sum6 or it overstates
  speed + understates exactness risk.

## 5. Codex review summary (2026-07-16, gpt-5.5 xhigh)

Verdict: **conditional HOLD.** Confirmed: the prize math (89.5%, ~10 ms = +20%), the per-pair
de-dup duplication, the dense-is-shared/experts-aren't split, the exactness-secondary scoping.
Corrected: the bandwidth attribution is unsupported; the de-dup ceiling is ~21–30% (not ~40%);
the synthetic-load probe isn't runnable. Missed avenues ranked: **readahead/overlap flag sweep**
(cheap, do first), STS re-tune after cheaper verify, **margin-guard exact fallback**; ruled out:
shared Q8 expert (already batched — not a lever), attention→FFN activation fusion (low prize,
~5 MiB/sub-ms), on-GPU argmax (already done). Full review:
`issue468/artifacts/dspark_codex_reviews/2026-07-16_gpt55_xhigh_lead08_reassess_avenues.md`.
