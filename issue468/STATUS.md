# Issue 468 Status

The canonical current-state page for the branch. Single coherent narrative — the
bottom line first, then the investigation arc, then durable findings by axis. The
full artifact/tool inventory moved to `issue468/inventories/dossier_inventory.md`.

## Bottom line

**Goal:** validate whether DSpark-style speculative decoding can deliver **≥20 % greedy
decode throughput on ds4 (DeepSeek-V4-Flash, IQ2XXS)** with output preserved.

**Current best (Milestone 3, 2026-07-15, COMPLETE):** the full DSpark stack **beats plain
ds4 by +4.9 %** (40.04 vs 38.16 t/s, full 176-entry corpus; +4.8 % on long-context) and is
**score-neutral** on the 92Q (61/92 vs 60/92, 90.2 % same verdict, net +1) — the first
config to beat plain locally. **+20 % is not reached**: verify still dominates ~80 % of the
cycle (~16 ms/token off the target).

| | |
|---|---|
| Goal | ≥20 % greedy throughput on ds4 IQ2XXS, output-preserved |
| Current best | Full DSpark stack **+4.9 %** over plain (40.04 vs 38.16 t/s; score-neutral 61/92) — M3 |
| The gap | +20 % not reached; verify ≈80 % of the cycle |
| Open lever | **Verify cost remains the open +20 % lever** — Lead 08 fused-2-token-kernel thesis **NO-GO (4.4× slower, codex-confirmed)**; the de-dup is real but compute-fusion was occupancy-bound → **Lead 09** proposes the corrected mechanism (de-dup loads, sequential single-token compute) |
| Closed (negative) | Drafter/input quality (Lead 07), drafter quant (Q4_K), non-expert finetune (Stage 2), DFlash, quant-mismatch |

## Investigation arc

Reverse-chronological. Each entry: what was tested → verdict → canonical record.

- **2026-07-16 — Lead 08 (fused low-K verify kernel): NO-GO on cost, codex-gate-B-confirmed.**
  Built the single-stage M=2 routed-expert fused-kernel prototype (shared gate/up expert loads
  across 2 tokens = "de-dup" + per-token selection-ordered down) + a self-contained fidelity/cost
  unit-test harness (runs under `--dspark`; decode2_exact is MTP-only/unreachable). **Fidelity
  (secondary): relaxed bar met** — M=2 vs M=1 routed_out max_abs≈4.8e-08, argmax_flip=0 across 5
  layers (sub-ULP Metal fast-math noise, localized to the fused gate+up; NOT bit-exact). **Cost
  (primary): FAIL** — cold all-layers sweep M=2=13.3 ms/layer vs M=1×2=3.0 ms/layer → **4.4×
  slower**; the de-dup is overwhelmed by the fused kernel's register-pressure/occupancy overhead.
  Codex gate B (gpt-5.5 xhigh): NO-GO stands (found a real M=2 perf bug — 18 vs 12 streams — but
  even fixed the bound is ~2.9× slower; a real fix is a new kernel design, not bounded). The
  probe's "de-dup is real" (pair-vs-unique) holds, but fusing the per-token compute is the wrong
  mechanism — occupancy-bound here. M=2 code stays env-gated/dormant; do NOT productionize.
  Artifacts 05/06 + the gate-B report; `pending/lead_08_fused_verify_kernel.md` (verdict block).
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

### Verifier / cycle cost — THE OPEN +20 % LEVER (Lead 08)

Verify dominates ~80 % of the cycle; this is where the +20 % must come from.

- **The verifier is memory-bandwidth-bound** (not compute): verify(K=2) ≈ one decode; the
  shipped MTP path is net-negative at every K. `summaries/mtp_verifier_bench_results.md`.
- **Milestone 2:** the sublinear batched verifier is break-even at oracle acceptance
  (crossover ~2.3 accepts at K=4); the runtime drafter is sound (live ≥ oracle). `summaries/dspark_runtime_milestone_2_progress.md`.
- **Milestone 3:** committing batched verify (sublinear, divergent-but-score-neutral) + the
  full stack → **+4.9 % over plain**, the first config to beat plain locally. `summaries/dspark_runtime_milestone_3_progress.md`.
- **Lead 08:** Phase A found retained headroom (~19–22 ms/cycle above the floor); Phase B found
  no swap-only sublinear bit-exact verifier (~0.85× baseline) → the bounded **fused verify
  kernel** build, exit gate `verify_ms(4) ≤ 50.5 ms`. `summaries/lead08_phaseB_floor_clearance_verdict.md`.

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

**Lead 08 — the fused verify kernel** is the primary open path to +20 %. It is a bounded
sublinear bit-exact batch-path build (HC/compressor/attention on decode reductions + batched
load sharing) with a hard exit gate: an end-to-end K=4 bit-exact verifier must profile
`verify_ms(4) ≤ 50.5 ms`. GO is unconfirmed until that gate clears.

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
