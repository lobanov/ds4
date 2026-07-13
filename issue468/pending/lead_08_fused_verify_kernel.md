# Lead 08 — Fused low-K batch-verify kernel (close the verify-vs-floor gap)

Date: 2026-07-07 (refreshed 2026-07-12). Status: **Phase A resolved 2026-07-11
→ Phase B active** (the fused-kernel work; this doc stays in `pending/` as the
active lead). Phase A result:
`issue468/summaries/mtp_verifier_engineering_and_phaseA.md`.
**Two-phase: a cheap profiling gate first (DONE, passed), kernel work only if
the gate passes (it did — proceed to Phase B).**
Related to lead 06 but a distinct thesis: the verifier is above its *own*
bandwidth floor, independent of the redundant anchor decode.

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
   rate 0.64%** (1/156 positions) — so the non-exactness is *small and
   concentrated on near-ties*, which both **validates the margin-guarded
   fallback as cheap** (it would trigger on only ~0.64% of positions) and lets
   the top1−top2 margin threshold (Q4-ceiling estimate ≲ ~0.5 logits) be **set
   empirically** from the measured flip distribution rather than assumed. Note:
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

**Milestone COMPLETE (2026-07-13): Phase B characterization + decisive measurements →
FINAL verdict = NO-GO via swaps → bounded build attempt with a hard exit gate
(codex-gated A+B+C; propagated).** Canonical summary:
`issue468/summaries/lead08_phaseB_floor_clearance_verdict.md`. STATUS.md +
spec_speedup_model.md updated.

**The gated follow-up (a separate lead/goal):** attempt the **sublinear bit-exact
batch-path build** (HC/compressor/attention on decode reductions + batched load sharing) as
a BOUNDED effort with a **hard exit gate** — GO is unconfirmed until an end-to-end K=4
bit-exact verifier profiles `verify_ms(4) ≤ 50.5 ms`. NOT "gate cleared / commit as
sufficient." (Confirmatory: a literal identical-input MoE kernel-equality harness; a stable
verify-bandwidth slope re-measure.)

## Worklog

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

### 2026-07-12 — lock-exactness: strategy A (bit-exact fused kernel by construction) chosen; margin-guard deferred

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
needed to derive a tight threshold. Strategy B is revisitable only if the fused kernel
proves unable to reach bit-exactness AND a later per-position top-2 measurement shows a
tight margin-guard is viable.

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
the current 66 ms), the exactness measurement (median TV 0.0035, flip 0.64% →
margin-guarded fallback is cheap and the margin threshold can be set
empirically), and the `decode2_exact`-is-linear ruling (the fused kernel must be
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
