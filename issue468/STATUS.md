# Issue 468 Status

## State

Framework established for a compact DSpark research dossier.

This branch starts from `main` and is intended to host a curated, low-noise version of the DSpark research record under `issue468/`.

## Current objective

Create a trustworthy active dossier that makes it easy to:

- understand the actual DSpark research goal,
- identify the current conclusions,
- retain useful tools and instrumentation,
- and prevent noisy experimental artifacts from dominating the branch.

## What exists now

- `issue468/README.md` — overview and directory contract
- `issue468/GOAL.md` — distilled research objective from `PLAN.md`
- this `issue468/STATUS.md` — canonical current-state page
- `issue468/AGENTS.md` — artifact hygiene rules
- dossier scaffold for summaries, inventories, artifacts, archive, and references
- retained DSpark numpy oracle with local smoke-checked venv setup:
  - code: `issue468/dspark_oracle/`
  - inventory: `issue468/inventories/numpy_oracle.md`
  - summary: `issue468/summaries/numpy_oracle_retention.md`
- retained exactness/debug bundles for the small exactness corpus across `temp=0/0.5/1.0`:
  - inventory: `issue468/inventories/exactness_small_bundles.md`
  - summary: `issue468/summaries/exactness_small_bundles_and_oracle_acceptance.md`
  - artifacts: `issue468/artifacts/exactness_small_bundles/summary.json`
  - artifacts: `issue468/artifacts/exactness_small_bundles/summary.csv`
- updated retained plain-baseline matrix for the later prompt corpus:
  - summary: `issue468/summaries/plain_baseline_matrix.md`
  - artifacts: `issue468/artifacts/plain_baseline_matrix/summary.csv`
  - artifacts: `issue468/artifacts/plain_baseline_matrix/summary.json`
- DSpark drafter quantization ceiling (Q4_K vs F16 vs F32) — measured and closed:
  - summary: `issue468/summaries/dspark_quantization_ceiling.md`
  - converter (research-scoped copy): `issue468/dspark_converter/`
  - bulk acceptance harness: `issue468/run_ceiling_bulk.py`
  - F16 ceiling drafter: `issue468/artifacts/dspark_ceiling/dspark_f16.gguf` (gitignored, reproducible)
  - acceptance results: `issue468/artifacts/exactness_small_acceptance/f16_ceiling/` and `q4k_baseline/`
  - diagnostics: `issue468/archive/diagnostics/`
- MTP bulk draft verifier bandwidth-binding audit (code audit, not a measurement):
  - summary: `issue468/summaries/mtp_verifier_bandwidth_binding.md`
- MTP verifier benchmark (empirical confirmation of the audit, K-sweep):
  - summary: `issue468/summaries/mtp_verifier_bench_results.md`
  - harness: `issue468/run_mtp_verifier_bench.py` (code_4k, K in {2,4,8,16}) and
    `issue468/run_mtp_verifier_bench_long.py` (8k prompts, K=2..6)
  - artifacts: `issue468/artifacts/mtp_verifier_bench/` and `mtp_verifier_bench_long/`
- Speculative-decode speedup model (numpy projection, answers the gate questions):
  - summary: `issue468/summaries/spec_speedup_model.md`
  - model: `issue468/model_spec_speedup.py`
  - artifacts: `issue468/artifacts/spec_speedup_model/{summary.json,model_inputs.json}`
- Anchor-reuse falsifier (Lead 01; offline realizability test of the model's anchor-reuse assumption):
  - summary: `issue468/summaries/anchor_reuse_falsifier.md`
  - harness: `issue468/run_anchor_reuse_falsifier.py` (reuse modes added to `dspark_oracle/measure_acceptance_bundle.py`)
  - artifacts: `issue468/artifacts/anchor_reuse_falsifier/` (falsifier_result.json, per_prompt/units csv, codex setup+verdict reviews)
  - lead (resolved, archived): `issue468/archive/leads/lead_01_anchor_reuse_falsifier.md`
- Acceptance statistical power + realistic-trajectory (Lead 03; powered measurement + cycle-jump correction):
  - summary: `issue468/summaries/acceptance_statistical_power.md` (canonical result)
  - worklog (resolved, archived): `issue468/archive/leads/lead_03_acceptance_statistical_power.md`
  - harness: `run_lead03_torch_measure.py` (torch/MPS), `run_lead03_cyclejump.py` (realistic trajectory), `run_lead03_aggregate.py`, `run_lead03_trajectory.py`, `run_lead03_sample_corpus.py`; store `dspark_oracle/stage2_capture_store.py` (multi-dir merge); torch port fix `dspark_train/drafter_body.py` (hc_post)
  - artifacts: `issue468/artifacts/acceptance_powered/` (combined300/{aggregate,cyclejump,trajectory}.json, stage2_torch_measure/, lead3_new_shards/ via `dspark_train/data/`, torch_measure/torch_precision_gate.json, codex_reviews/)
  - model: `model_spec_speedup.py` refreshed + `artifacts/spec_speedup_model/model_inputs.json` (powered sliding + `lead03_cyclejump_realistic`)
- Confidence-scheduled verification (Lead 02; confidence extraction + STS + adaptive replay):
  - summary: `issue468/summaries/confidence_scheduled_verification.md`
  - worklog (resolved, archived): `issue468/archive/leads/lead_02_confidence_scheduled_verification.md`
  - harness: `run_lead02_confidence_fidelity.py`, `run_lead02_torch_confidence_gate.py`, `run_lead02_measure_confidence.py`, `run_lead02_sts.py`, `run_lead02_replay.py`
  - artifacts: `issue468/artifacts/lead02_confidence_{fidelity,measure_smoke,measure_eval,measure_train,measure_lead3,sts_eval,sts_train,replay_eval,replay_eval_trainsts,replay_lead3_trainsts,torch_confidence_gate}/`
- DFlash oracle + accepted-prefix comparison vs DSpark (resolved):
  - summary: `issue468/summaries/dflash_oracle_investigation.md`
  - weights: `issue468/dflash_drafter/` (gitignored); forward + harness: `issue468/dflash_oracle/`
  - captures: `issue468/artifacts/dflash_capture/` (30 cells, gitignored); results: `issue468/artifacts/dflash_acceptance/`
  - capture driver: `issue468/run_dflash_capture.py`
- Quantization-mismatch investigation (Stage 0 + Stage 1, both codex-reviewed; **complete → recommendation: NARROW**):
  - recommendation: `issue468/summaries/quant_mismatch_recommendation.md`
  - Stage 0 summary: `issue468/summaries/quant_mismatch_diagnostic.md`; harness `issue468/run_stage0_quant_mismatch.py`; artifacts `issue468/artifacts/quant_mismatch_diagnostic/` (incl. `drafter_top64_scores.npz` for rank re-derivation, `codex_review.md`)
  - Stage 1 summary: `issue468/summaries/stage1_tap_precision.md`; capture `issue468/run_exactness_small_bundles.py` (Q4-tap variant); compare `issue468/run_stage1_q4tap_compare.py`; artifacts `issue468/artifacts/exactness_small_bundles_q4tap/` (incl. `comparison_vs_baseline/` + `codex_review.md`)
  - headline: drafter p=1 misses are **shallow / right neighborhood** (median target-rank 1.0; top-2 coverage 0.8125→0.9125) — recoverable shape, but causal link to quant unproven. Raising tap-layer (layers 37–42) precision to Q4 did **not** materially help p=1 (underpowered; CI straddles 0) — the original FP-vs-Q2 mismatch framing is partially falsified. Recommendation: bounded Stage 2 fine-tune PoC (not a full pipeline), targeting the secondary gate.
- **Lead 04 — FP ceiling capture (Phase A COMPLETE → HOLD on full Phase B):**
  - capture script: `issue468/run_lead04_modal/capture_hc_modal.py` (Modal TP=2 vLLM,
    V1 post-load `apply_model` hooks on layers 40/41/42, `mhc_post` on clones,
    `enforce_eager=True`, `enable_prefix_caching=False`); hook module: `dspark_hc_patch.py`
  - conversion: `convert_vllm_to_oracle.py` (`[n,4,4096]` → mean+concat → `[n,12288]`);
    fidelity: `validate_fidelity.py` (p1 from prefix_hist)
  - pilot artifacts: `issue468/artifacts/lead04_fp_pilot/` (5 prompts × drafter.json +
    oracle_inputs + bundle_manifest + target_tokens)
  - codex reviews: `issue468/artifacts/lead04_codex_reviews/` (6 retained: patch-design,
    path-review, smoke-diagnosis, gate-1, audit-review)
  - worklog + go/no-go: `issue468/pending/lead_04_fp_ceiling_capture.md`
  - **Verdict: HOLD (Phase B-limited, codex gate 2 re-review).** 299/300 native captures;
    consistent-dtype float32 Δp1=+5.28pp CI[+3.8,+6.6] (technically GO; the earlier codex
    dtype-confound hypothesis was REFUTED — f32-Q2==f16-Q2 exactly on 240 prompts). BUT not
    deployment-actionable: the F16 drafter on FP hiddens gives p1≈0.62 vs F16-Q2 0.79
    (deployment flips to STOP) — strong C1 evidence (mhc_post capture error exposed at F16,
    masked by float32). Plus CI lower ≈ kernel systematic (~4pp, thin), cross-engine
    confound (FP=vLLM vs Q2=ds4), easy corpus (dolly +2.7pp; exactness pilot −6pp). HOLD
    pending an algebraic vLLM-vs-ds4 hidden-equality proof (C1). Don't kill the native-
    hidden hypothesis (capture bug could explain F16) but STOP action on the F16 capture
    path. JIT caches (TileLang+DeepGEMM) wired to volume (526s→244s warm). (Phase A pilot
    HOLD superseded: that was n=5 underpowered; Phase B is the powered measurement.)
- Stage 2 bounded fine-tune PoC (Activities 1–9, doubly codex-reviewed; **complete → verdict: NOT-JUSTIFIED**):
  - result: `issue468/summaries/stage2_finetune_result.md`; protocol `issue468/summaries/stage2_finetune_protocol.md`
  - corpus/capture: `prompts/stage2_corpus/` (240 prompts); `run_stage2_capture.py` + ds4 `--capture-dataset` engine mode; shards `dspark_train/data/shards/`
  - torch MPS drafter (self-consistent): `dspark_train/{drafter_body,drafter_head}.py`; features `dspark_train/data/{train,eval}_{torch,kloss}_features.safetensors`; results `artifacts/stage2_results/{activity6,activity7,activity9_final,mcnemar_lce_ltv}.json`
  - codex reviews: `artifacts/stage2_plan_review/{codex_review,bodybug_codex_review,verdict_codex_review}.md`
  - headline: **non-expert head LoRA does NOT improve (significantly HURTS) acceptance** — with the SPECIFIED `Lce+Ltv` loss (DSpark §3.3, exp position weights) + rank 32/64/128 × 2-seed sweep: −1.46 / −1.75 / −3.08 pp held-out p=1 (seed-stable std ~0.0017); single-run McNemar p=5.3e-5 (significant HARM). (Earlier p=1-CE-only −0.13 pp was a precursor; Lce+Ltv is the contract result.) From-scratch ceiling 0.7479 < pretrained 0.8125 (overfit-to-eval 0.9947 → h informative); drafter input-invariant (Activity 4). Body LoRA (Activity 8) gated out (contract 7>6ceiling + Activity 4); expert tuning out of scope. Two load-bearing codex reviews (found a Sinkhorn-eps body bug + a dead-LoRA-init bug). Local single-request speculative decode on this target/drafter has now exhausted drafter precision, tap precision, tree structure, and non-expert fine-tuning without clearing baseline.

## Current conclusions

- **Q4_K is not the draft-quality bottleneck.** Removing routed-expert quantization
  (F16/F32 drafter from the vendored MXFP4 source) yields no material acceptance
  gain: 3.5% of draft tokens flip vs Q4_K, but net accepted-prefix change is
  +0.38% overall (within noise; sign inconsistent across temperatures).
- The vendored drafter ships as MXFP4 (4-bit); F16 already captures its full
  dequant (F16≡F32 at the weight level), so no available precision beats Q4_K.
- This falsifies the "Q4_K drafter quality causes the verifier-dominated cycle"
  hypothesis. The cycle overhead originates elsewhere (verification / KV-replay /
  scheduling), not draft precision. Research should redirect away from drafter
  quantization.
- This supersedes the sibling `ds4-dspark` dossier notes 22 and 51 (buggy oracle;
  did not measure the HF-source ceiling).
- **The MTP bulk draft verifier is memory-bandwidth-bound, not compute-bound,**
  across its entire operating range, now confirmed empirically: verify(K=2)
  median 27.3 ms ~= one decode (26.0 ms), and the MTP path is net-negative at
  every tested K (−5% at K=2, −70% at K=16). Total verify grows super-linearly
  with K beyond ~8 (MoE expert-union growth); acceptance peaks at K=4 then
  declines. Draft cost is negligible (2-6 ms for K=2-4). The path to a local
  speedup is not drafter precision (exhausted) and not larger K (super-linear
  verify + declining acceptance); it requires higher draft acceptance or
  server-side multi-request batching that amortizes the verify bandwidth floor.
  See `summaries/mtp_verifier_bench_results.md` (confirmed on three 8k prompts,
  K=2..6: net-negative at every cell; best case grounded_8k K=2 at −0.3%).
- **Acceptance statistical power + realistic trajectory (Lead 03): the K=4 verdict
  flips to BELOW baseline under the realistic cycle-jump trajectory; the prior sliding
  estimate was optimistic.** Powered the corpus to 300 prompts (Stage 2's 240 + 60 new;
  dolly/codealpaca/jsonex; 128-tok temp=0) via a torch/MPS port precision-gated at 100%
  vs the numpy oracle (`summaries/acceptance_statistical_power.md`). Two codex gates
  (GATE-1 caught an overclaim + a torch-body `hc_post` bug; GATE-2 corrected the
  break-even framing). Findings: (1) **sliding** E[a|4]=2.337 -> K=4 **+1.2%** (the
  model's old method; OPTIMISTIC, diagnostic only); (2) **cycle-jump** (realistic
  per-cycle trajectory) E[a|4]=**2.198**, S(4)=0.340 -> K=4 **0.982× (−1.8%)**,
  speedup CI [0.971,0.993], **P(speed<1)=0.999** — significantly below baseline. The
  dynamic break-even (at S(4)=0.340) is E[a|4]=2.256 (deficit 0.058), not the stale 2.203.
  Corpus-dependent (cycle-jump: jsonex +2.3%, codealpaca −1.9%, dolly −5.6%; old
  code/synthesis exactness was harder at 2.175). Sliding CI half-width 0.046 (<0.05 floor,
  met). **Net:** beating baseline at K=4 is NOT achieved on this corpus under the
  realistic trajectory; the honest band is sliding +1.2% (optimistic) to cycle-jump
  −1.8% (realistic, model currency). Scope: cycle-jump is linear-trajectory anchor-token
  difficulty, NOT drafter-state pollution / trees / residual overhead (Lead 06).
- **Anchor-reuse falsifier (Lead 01): no large acceptance collapse observed, but
  non-inferiority NOT established.** Offline test (`summaries/anchor_reuse_falsifier.md`)
  of whether the drafter's acceptance survives drafting from the last-accepted-position
  (stale) target hidden + correction token as embedding — the realizability test for
  the model's anchor-reuse assumption. Swapped `main_hidden[pos]→main_hidden[pos−1]` in
  the retained greedy-spine harness; two KV-window models (lag, backfill) × 3 temps = 6
  cells, 80 paired units each, paired bootstrap CI + exact McNemar. **Verdict: 5/6 cells
  SURVIVE, 1 MARGINAL (t0p0/backfill −0.075); no cell shows a significant collapse;
  all CIs straddle 0.** But (a) the corpus is ~10× too underpowered to confirm the
  fragile −0.9% edge (clustered 80% MDE ~0.2–0.4 vs ~0.03 E[a|4] budget), and (b) reuse
  reshapes the block: it helps early positions (1–2) and hurts late positions (4–5), so
  flat net E[a|5block] is not 'free'. Two codex gates (setup + verdict) passed, findings
  independently verified. **Implication:** anchor reuse is NOT invalidated on
  acceptance grounds; Lead 06 (verifier engineering) can proceed on the acceptance axis
  but must still prove verify-produced-hidden equivalence under IQ2XXS, residual cycle
  overhead, and a powered K=4 non-inferiority bound. The −0.9% edge stays the optimistic
  edge of the band, now with the acceptance-axis risk downgraded from 'unverified /
  load-bearing' to 'no large collapse seen, non-inferiority not yet established.'
- **Speculative speedup is acceptance-limited; with the realistic-trajectory (Lead 03)
  acceptance, the current drafter does NOT beat baseline at any K (~0.98× at K=4/5).**
  Corrected cycle model (`summaries/spec_speedup_model.md`, holistically integrated with
  Leads 01+03): the verify forward produces the next anchor, so a fresh decode is needed
  only on full-block acceptance (cost = draft+verify+decode·S(K)). Two acceptance
  estimators on the 300-prompt powered corpus: sliding (optimistic) E[a|4]=2.337 -> K=4
  +1.2%; **cycle-jump (realistic per-cycle trajectory) E[a|4]=2.198, S(4)=0.340 -> K=4
  0.982× (−1.8%), CI [0.971,0.993], P(speed<1)=0.999** — significantly below baseline;
  dynamic break-even E[a|4]=2.256 (deficit 0.058). Corpus-dependent (cycle-jump K=4:
  jsonex +2.3%, codealpaca −1.9%, dolly −5.6%). The shipped `--mtp` pays a redundant
  anchor decode every cycle (~−18 pp at K=4) — anchor reuse (acceptance-axis de-risked
  by Lead 01, verifier economics untested = Lead 06) is the largest lever, but it would
  feed a ~0.98× realistic edge, not the old sliding −0.9%. A 4-node draft tree still
  cannot help (ceiling +2.2%, dominated by a linear chain; hedging is doubly
  penalized). Levers: verifier-anchor reuse + a materially better drafter (training,
  not quantization) or server-side batching. **The headline is conditional on the
  anchor-reuse assumption** (the explicit modeling premise); an adversarial
  codex/gpt-5.5-xhigh review (retained in `summaries/spec_speedup_model.md` §
  Adversarial review, and `artifacts/spec_speedup_model/codex_review.md`) flags
  that assumption as the load-bearing unverified risk — the shipped verifier does
  NOT reuse the anchor (−18.9% at K=4) — plus ~15–19 ms residual per-cycle overhead
  set to zero in the model and prompt-level noise (sd ≈0.43 on E[a|4]).
  The under-assumption findings stand; treat them as the optimistic edge of a band
  whose pessimistic edge is the shipped-verifier reality.
- **Confidence-scheduled verification (Lead 02): useless under shipped economics; only
  fragile conditional secondary material under anchor reuse.** Canonical result:
  `summaries/confidence_scheduled_verification.md`. Confidence extraction was fidelity-gated
  against both the retained oracle and a torch-vs-numpy 3-source sample; the powered head is
  decision-relevant on IQ2XXS (per-position AUC ~0.77–0.82, cumulative AUC ~0.79–0.85 across
  train/eval/lead3), and STS modestly improves cumulative-prefix calibration. But the replay
  verdict is now out-of-sample: **fit STS on `train` (180 prompts), choose threshold on
  `eval` (60), evaluate on fresh `lead3` (60)**. On the fresh slice, the externally selected
  anchor-reuse-selected STS threshold `0.08` gives **0.8646×** under shipped accounting
  (the shipped-selected `eval` threshold `0.52` still reaches only **0.8987×** on fresh
  `lead3`) and **1.0375×** under anchor reuse (CI [1.013,1.064]); STS expected-opt is
  **0.9205×** shipped and **1.0523×** under anchor reuse. The clean frozen-threshold result
  is **below** Lead 02's predeclared `+5–10%` stacking tier, and the anchor-reuse positives
  have only about **3.0–3.9 ms/cycle** of overhead headroom before they disappear.
  Per-source fresh frozen-threshold speedup under anchor reuse: codealpaca **1.046×**,
  dolly **1.003×**, jsonex **1.063×**. **Decision:** Lead 02 does NOT revive the local gate;
  record it only as marginal conditional secondary material contingent on Lead 06 proving a
  genuinely cheap anchor-reuse verifier.
- **DFlash drafter comparison: DSpark is more attractive on this corpus.**
  DFlash oracle built and validated vs the MLX reference (<=0.08% rel); the only
  bug was a self-inflicted `d2t` token-mapping error (`d2t` is an offset, not an
  absolute map; correct decode `target_id = draft_idx + d2t[draft_idx]`), caught
  by an adversarial codex review after a brief mis-diagnosis as a representation
  blocker. On the same IQ2XXS target and offline per-step protocol, DFlash avg
  accepted prefix is 0.876 (7-token block) vs DSpark q4k's 2.171 (5-token block)
  — DSpark ~2.5x better. DFlash's parallel noise-block drafting rarely extends
  the prefix past position 1. DFlash pos1 acc 0.68 ≈ its val 0.74; positions 2-7
  underperform val (likely IQ2XXS effect on later positions / corpus). See
  `summaries/dflash_oracle_investigation.md`.
- **Quantization-mismatch hypothesis: partially falsified; drafter fine-tune kept
  in play as a bounded PoC (NARROW).** Stage 0 (codex-reviewed): DSpark drafter
  p=1 misses vs IQ2XXS are **shallow** (median target-rank 1.0; 100% within
  top-10; top-2 coverage 0.8125→0.9125) — recoverable shape, but the *quant-flip*
  attribution is unproven (only ~27% are Q2 near-ties; median Q2 gap 2.15 nat).
  Stage 1 (codex-reviewed): raising layers 37–42 routed experts to Q4_K did **not**
  materially improve p=1 acceptance (0.8125→0.7875; E[a|5block] +0.10, bootstrap
  CI [−0.16,+0.41] — underpowered) — so this expert-only Q4-tap variant does not
  materially help; tap-localized quant mismatch is **not demonstrated** as the
  mechanism (lower-layer Q2 vs calibration remains unresolved). Net: precision is not
  a usable lever; a bounded drafter fine-tune on the served target's distribution
  is the last plausible lever, realistically targeting the secondary gate
  (beat baseline / match --mtp), not the +20% primary gate. See
  `summaries/quant_mismatch_recommendation.md`.

## Next recommended steps

1. Create inventories for:
   - instrumentation in code
   - retained tools/scripts
   - retained artifacts
2. Add compact topic summaries for:
   - accepted findings
   - false leads / exhausted directions
   - current open questions
3. Move bulky or superseded material to archive references rather than keeping it active.
4. **Lead 04 — FP ceiling capture (Phase A COMPLETE → HOLD on full Phase B):**
   - Pilot: 5 exactness prompts measured (FP mean p=1=0.7143 GREEN; Δ vs Q2=−0.06,
     CI includes 0 → gap absent)
   - Full capture: NOT justified — deferred (pilot underpowered + mhc_post representation
     unproven; see go/no-go in the worklog)

## Canonicality rule

This file is the source of truth for the branch's current research state.
Any deeper notes should support this file, not contradict it.
