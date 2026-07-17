# Dossier inventory — what exists in `issue468/`

Reference catalog of the retained tools, harnesses, captures, and summaries in the
issue 468 dossier. This is the *inventory*; the **current state / verdicts live in
`issue468/STATUS.md`** (Findings by axis). Items grouped by topic. (Moved out of
STATUS.md on 2026-07-16 so STATUS reads as a single coherent narrative.)

## Scaffold

- `issue468/README.md` — overview and directory contract
- `issue468/GOAL.md` — distilled research objective from `PLAN.md`
- `issue468/STATUS.md` — canonical current-state page (bottom line + arc + findings)
- `issue468/AGENTS.md` — artifact hygiene rules
- dossier scaffold: `summaries/`, `inventories/`, `artifacts/`, `archive/`, `references/`

## DSpark numpy/torch oracle (offline drafter)

- code: `issue468/dspark_oracle/` (`forward.py` numpy D_f32; `analyze_phaseB_gap.py` torch D_f32 body+head; `measure_acceptance_bundle.py`)
- torch drafter body/head: `issue468/dspark_train/{drafter_body,drafter_head}.py`
- inventory: `issue468/inventories/numpy_oracle.md`
- summary: `issue468/summaries/numpy_oracle_retention.md`

## Captures / corpora

- exactness/debug bundles (small exactness corpus, temp=0/0.5/1.0): inventory `inventories/exactness_small_bundles.md`, summary `summaries/exactness_small_bundles_and_oracle_acceptance.md`, artifacts `artifacts/exactness_small_bundles/{summary.json,summary.csv}`
- plain-baseline matrix (later prompt corpus): summary `summaries/plain_baseline_matrix.md`, artifacts `artifacts/plain_baseline_matrix/`
- Stage-2 corpus (240 prompts): `prompts/stage2_corpus/`; capture `run_stage2_capture.py` + ds4 `--capture-dataset`; shards `dspark_train/data/shards/`
- Lead-3 corpus (60 prompts, codealpaca/dolly/jsonex 0080–0099): `prompts/lead3_corpus/`
- Lead-04 native-FP captures (Modal vLLM, 300 prompts): on the `lead04-captures` Modal volume; the 60 matched (`phaseB_<src>_0080–0099.npz`) re-downloaded under `artifacts/lead07_crossed_oracle/fp_captures_raw/` (gitignored, reproducible)
- DS4 env/instrumentation inventory: `inventories/ds4_env_and_instrumentation_inventory.md`

## Drafter precision / quality axis (CLOSED NEGATIVE — see STATUS)

- DSpark drafter quantization ceiling (Q4_K vs F16 vs F32): summary `summaries/dspark_quantization_ceiling.md`; converter `dspark_converter/`; bulk harness `run_ceiling_bulk.py`; F16 ceiling drafter `artifacts/dspark_ceiling/dspark_f16.gguf` (gitignored); results `artifacts/exactness_small_acceptance/{f16_ceiling,q4k_baseline}/`; diagnostics `archive/diagnostics/`
- Stage 2 bounded non-expert head-LoRA PoC (verdict NOT-JUSTIFIED): result `summaries/stage2_finetune_result.md`; protocol `summaries/stage2_finetune_protocol.md`; torch MPS drafter features `dspark_train/data/{train,eval}_{torch,kloss}_features.safetensors`; results `artifacts/stage2_results/`; codex reviews `artifacts/stage2_plan_review/`
- Quantization-mismatch (Stage 0 + Stage 1, NARROW): recommendation `summaries/quant_mismatch_recommendation.md`; Stage 0 `summaries/quant_mismatch_diagnostic.md` + `run_stage0_quant_mismatch.py` + `artifacts/quant_mismatch_diagnostic/`; Stage 1 `summaries/stage1_tap_precision.md` + `run_stage1_q4tap_compare.py` + `artifacts/exactness_small_bundles_q4tap/`
- Lead 04 FP ceiling capture (HOLD; capture-fidelity unresolved): capture `run_lead04_modal/{capture_hc_modal.py,dspark_hc_patch.py,convert_vllm_to_oracle.py,validate_fidelity.py}`; pilot `artifacts/lead04_fp_pilot/`; codex reviews `artifacts/lead04_codex_reviews/`; worklog `archive/leads/lead_04_fp_ceiling_capture.md`
- Lead 07 crossed FP/IQ2 oracle (PIVOT, closed negative): `archive/leads/lead_07_upstream_quality_ceiling.md` (verdict folded in); scripts + result `artifacts/lead07_crossed_oracle/` (`run_crossed_oracle.py`, `crossed_oracle_result.json`, `fidelity_gate.py`, `prep_recapture.py`, `verify_recapture.py`); codex gate `artifacts/dspark_codex_reviews/2026-07-16_lead07_crossed_oracle_gate.md`; ds4-spec-bench `teacher_force` submode (commit bb9c01f)
- DFlash drafter comparison (DSpark preferred): summary `summaries/dflash_oracle_investigation.md`; weights `dflash_drafter/` (gitignored); forward+harness `dflash_oracle/`; captures `artifacts/dflash_capture/` (gitignored); results `artifacts/dflash_acceptance/`; driver `run_dflash_capture.py`

## Verifier / cycle-cost axis (gap remains; tested Lead 08 branches closed)

- MTP verifier bandwidth-binding audit + bench: `summaries/mtp_verifier_bandwidth_binding.md`, `summaries/mtp_verifier_bench_results.md`; harnesses `run_mtp_verifier_bench.py`, `run_mtp_verifier_bench_long.py`; artifacts `artifacts/mtp_verifier_bench{,_long}/`
- Lead 06 verifier engineering + Lead 08 Phase A profiling: canonical `summaries/mtp_verifier_engineering_and_phaseA.md`; worklogs `archive/leads/lead_06_verifier_engineering.md`, `archive/leads/lead_08_fused_verify_kernel.md`; harnesses `run_mtp_{exactness_compare,temp_distribution_compare,phaseA_profile,corpus_bench}.py`; artifacts `artifacts/mtp_{exactness_compare,temp_distribution_compare,phaseA_profile,corpus_bench}/`
- Lead 08 Phase B floor-clearance verdict (NO-GO via swaps → bounded build): `summaries/lead08_phaseB_floor_clearance_verdict.md`; artifacts `artifacts/lead08_stage_divergence/`; codex reviews `artifacts/dspark_codex_reviews/2026-07-13_*`
- Lead 08 kernel/exactness iterations: worklog `pending/lead_08_fused_verify_kernel.md`; iteration artifacts `artifacts/lead08_reassessment/03_*` through `18_*`; retained self-contained harness `metal_graph_test_m2_fidelity_unit` in `ds4.c`, with production K=4 overlap mode `DS4_LEAD08_BATCH_OVERLAP_PROBE=1`; controlled opportunity result `10_iter2_production_batch_overlap.{md,csv}`; bit-exact but slower grouped-prototype NO-GO `11_iter3_grouped_gateup_prototype.{md,csv}`; uneconomic margin-fallback NO-GO `12_iter4_margin_guard.{md,csv}`; fixed-work address-locality NO-GO `13_iter5_address_locality.{md,csv}`; production address-kernel NSG NO-GO `14_iter6_addr_nsg_geometry.{md,csv}` reproduced by `DS4_LEAD08_ADDR_NSG_PROBE=1`; production row-tile NO-GO `15_iter7_addr_row_tile.{md,csv}` reproduced by `DS4_LEAD08_ADDR_NR0_PROBE=1`; cache-residency carrier NO-GO `16_iter8_cache_residency.md` plus compact replay/LRU CSVs, reproduced by `DS4_LEAD08_CACHE_REPLAY_PROBE=1` and actual batched selections under `DS4_EXPERT_PROFILE`; profiler correction and corrected three-family locality evidence `17_iter9_profiler_correction.{md,csv}`; existing-batch shared M=1/M=K exactness/economics NO-GO `18_iter10_batch_m1_target.{md,csv}` reproduced with `DS4_LEAD08_BATCH_M1_TARGET=1`; research-only grouped path `DS4_LEAD08_GROUPED_GATEUP_PROBE=1` plus direct comparator `DS4_LEAD08_GROUPED_GATEUP_FIDELITY=1`; retained batch-vs-exact margin instrumentation under `DS4_DSPARK_VERIFY_DIST_PROBE=1`
- DSpark runtime milestones: M2 `summaries/dspark_runtime_milestone_2_progress.md`; M3 `summaries/dspark_runtime_milestone_3_progress.md`; initial benchmark `summaries/dspark_runtime_initial_benchmark.md`; M3 bench artifacts `artifacts/dspark_m3_bench/`

## Scheduler & acceptance axis (see STATUS)

- Speculative speedup model (numpy projection): `summaries/spec_speedup_model.md`; model `model_spec_speedup.py`; artifacts `artifacts/spec_speedup_model/`
- Lead 01 anchor-reuse falsifier (survives): `summaries/anchor_reuse_falsifier.md`; harness `run_anchor_reuse_falsifier.py` (reuse modes in `dspark_oracle/measure_acceptance_bundle.py`); artifacts `artifacts/anchor_reuse_falsifier/`; worklog `archive/leads/lead_01_anchor_reuse_falsifier.md`
- Lead 02 confidence-scheduled verification (marginal): `summaries/confidence_scheduled_verification.md`; harnesses `run_lead02_*.py`; artifacts `artifacts/lead02_confidence_*/`; worklog `archive/leads/lead_02_confidence_scheduled_verification.md`
- Lead 03 acceptance statistical power + cycle-jump: `summaries/acceptance_statistical_power.md`; harnesses `run_lead03_*.py`; store `dspark_oracle/stage2_capture_store.py`; artifacts `artifacts/acceptance_powered/`; worklog `archive/leads/lead_03_acceptance_statistical_power.md`

## Bench harness (canonical)

- `ds4-spec-bench` (built via `make ds4-spec-bench`) — the bulk-config JSONL harness; modes argmax/sample/speculative_argmax/teacher_force; `--dump-hidden-dir`, `--force-tokens-dir`, `--rewrite-frontier`. Do NOT loop over the `ds4` CLI for measurements.
