# Dossier inventory — what exists in `issue468/`

Reference catalog of the retained tools, harnesses, captures, and summaries in the
issue 468 dossier. This is the *inventory*; the **current state / verdicts live in
`issue468/STATUS.md`** (Findings by axis). Items grouped by topic. (Moved out of
STATUS.md on 2026-07-16 so STATUS reads as a single coherent narrative.)

## Scaffold

- `issue468/README.md` — overview and directory contract
- `issue468/GOAL.md` — distilled research objective from `PLAN.md`
- `issue468/STATUS.md` — canonical current-state page (bottom line + arc + findings)
- `issue468/summaries/solution_space_ledger.md` — live mechanism/dependency/experiment ledger
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
- Lead 08 kernel/exactness iterations: worklog `pending/lead_08_fused_verify_kernel.md`; iteration artifacts `artifacts/lead08_reassessment/03_*` through `20_*`; retained self-contained harness `metal_graph_test_m2_fidelity_unit` in `ds4.c`, with production K=4 overlap mode `DS4_LEAD08_BATCH_OVERLAP_PROBE=1`; controlled opportunity result `10_iter2_production_batch_overlap.{md,csv}`; bit-exact but slower grouped-prototype NO-GO `11_iter3_grouped_gateup_prototype.{md,csv}`; uneconomic margin-fallback NO-GO `12_iter4_margin_guard.{md,csv}`; fixed-work address-locality NO-GO `13_iter5_address_locality.{md,csv}`; production address-kernel NSG NO-GO `14_iter6_addr_nsg_geometry.{md,csv}` reproduced by `DS4_LEAD08_ADDR_NSG_PROBE=1`; production row-tile NO-GO `15_iter7_addr_row_tile.{md,csv}` reproduced by `DS4_LEAD08_ADDR_NR0_PROBE=1`; cache-residency carrier NO-GO `16_iter8_cache_residency.md` plus compact replay/LRU CSVs, reproduced by `DS4_LEAD08_CACHE_REPLAY_PROBE=1` and actual batched selections under `DS4_EXPERT_PROFILE`; profiler correction and corrected three-family locality evidence `17_iter9_profiler_correction.{md,csv}`; existing-batch shared M=1/M=K exactness/economics NO-GO `18_iter10_batch_m1_target.{md,csv}` reproduced with `DS4_LEAD08_BATCH_M1_TARGET=1`; research-only grouped path `DS4_LEAD08_GROUPED_GATEUP_PROBE=1` plus direct comparator `DS4_LEAD08_GROUPED_GATEUP_FIDELITY=1`; retained batch-vs-exact margin instrumentation under `DS4_DSPARK_VERIFY_DIST_PROBE=1`
- Lead 08 exact-hybrid localization iteration: artifact `artifacts/lead08_reassessment/19_iter11_same_frontier_localization.md`, compact corpus/stage/growth CSVs and one-prompt config under `19_iter11_*`; reproduced with `DS4_LEAD08_BATCH_M1_TARGET=1`, `DS4_DSPARK_VERIFY_DIST_PROBE=1`, committing batch verify disabled, and existing tagged Metal tensor dumps
- Lead 08 row-wise Q/KV iteration: artifact `artifacts/lead08_reassessment/20_iter12_rowwise_qkv.md` plus compact stage/timing CSVs; reproduced with verifier-scoped `DS4_LEAD08_ROWWISE_QKV_LAYERS={1,43}`
- Lead 08 row-wise Q-b iteration: artifact `artifacts/lead08_reassessment/21_iter13_rowwise_qb.md` plus stage/timing CSVs; reproduced with verifier-scoped `DS4_LEAD08_ROWWISE_QB_LAYERS={1,43}` on top of the Q/KV gate
- Lead 08 attention output-B iteration: artifact `artifacts/lead08_reassessment/22_iter14_rowwise_attn_out_b.md` plus stage/timing CSVs; branch-neutral capture and verifier-scoped `DS4_LEAD08_ROWWISE_ATTN_OUT_B_LAYERS={1,43}`
- Lead 08 HC-input localization: artifact `artifacts/lead08_reassessment/23_iter15_hc_input_localization.md` plus CSV; branch-neutral capture only, no timing claim
- Lead 08 row-wise HC attention mixer: artifact `artifacts/lead08_reassessment/24_iter16_rowwise_hc_attn_mix.md` plus stage/timing CSVs; verifier-scoped `DS4_LEAD08_ROWWISE_HC_ATTN_MIX_LAYERS={1,43}`
- Lead 08 FFN HC-input localization: artifact `artifacts/lead08_reassessment/25_iter17_ffn_hc_input_localization.md` plus stage CSV; branch-neutral residual/flat/mix/split capture only, no timing claim
- Lead 08 row-wise HC FFN mixer: artifact `artifacts/lead08_reassessment/26_iter18_rowwise_hc_ffn_mix.md` plus stage/timing CSVs; verifier-scoped `DS4_LEAD08_ROWWISE_HC_FFN_MIX_LAYERS={0,1}` and branch-neutral router capture
- Lead 08 bounded-divergence reframe: artifact `artifacts/lead08_reassessment/27_iter19_bounded_divergence_reframe.md`; freezes the matched M3 quality envelope, defers the exact-hybrid track, and ranks Metal-counter-guided V14 acceleration first
- Lead 08 Metal-counter capability gate: artifact `artifacts/lead08_reassessment/28_iter20_metal_counter_capability.{md,csv}`; available CLI profiles lack discriminating M5 counters/shader intervals and perturb K4 timing, so T1 is blocked without a custom template
- Lead 08 selected-address ALU-headroom separator: artifact `artifacts/lead08_reassessment/29_iter21_ssd_addr_alu_headroom.{md,csv}`; research-only `DS4_LEAD08_ADDR_ALU_PROBE` plus production/companion-zero controls and `DS4_METAL_DUMP_SOURCE` prove retained dependent FMAs but yield a threshold-ambiguous response on the incompatible SSD address kernel; 688/688 mapped records identified the `tiny_pair_mv` carrier used by T2b
- Lead 08 mapped tiny-pair ALU separator: artifact `artifacts/lead08_reassessment/30_iter22_mapped_alu_separator.{md,csv}`; mapped-only companion, two 18-unique strata, 86-case all-layer fidelity, dispatch identity, and AIR retention pass, but the production/zero control interval fails, so the strong response is near-threshold evidence only and T2c direct-replay preflight is P0
- Lead 08 direct mapped-pair GPU replay: artifact `artifacts/lead08_reassessment/31_iter23_direct_gpu_replay.md`, aggregate summary and 12,900-row layer CSVs, plus deterministic validating parser; `DS4_LEAD08_MAPPED_ALU_DIRECT_REPLAY=1` times one gate/up command buffer per layer and closes T2c as a valid one-sided arithmetic/issue selector, authorizing only U1 maximum-removable arithmetic upper-bound work
- Lead 08 mapped packed-weight floor: artifact `artifacts/lead08_reassessment/32_iter24_weight_floor.md` plus validating parser and sample/layer/summary CSVs; corrected `DS4_LEAD08_MAPPED_ARITH_FLOOR=1` retains production row-addressed mapped q/scale access and stores but removes semantics, saving only 4.71-4.72 ms and closing arithmetic reduction below the 8.7 ms gate
- Lead 08 bounded-track exhaustion: `artifacts/lead08_reassessment/33_iter25_bounded_track_exhaustion.md`; systematic post-U1 audit finds no independent composable >=8.7 ms mechanism on the current in-RAM runtime
- DSpark runtime milestones: M2 `summaries/dspark_runtime_milestone_2_progress.md`; M3 `summaries/dspark_runtime_milestone_3_progress.md`; initial benchmark `summaries/dspark_runtime_initial_benchmark.md`; M3 bench artifacts `artifacts/dspark_m3_bench/`

## Scheduler & acceptance axis (see STATUS)

- Speculative speedup model (numpy projection): `summaries/spec_speedup_model.md`; model `model_spec_speedup.py`; artifacts `artifacts/spec_speedup_model/`
- Lead 01 anchor-reuse falsifier (survives): `summaries/anchor_reuse_falsifier.md`; harness `run_anchor_reuse_falsifier.py` (reuse modes in `dspark_oracle/measure_acceptance_bundle.py`); artifacts `artifacts/anchor_reuse_falsifier/`; worklog `archive/leads/lead_01_anchor_reuse_falsifier.md`
- Lead 02 confidence-scheduled verification (marginal): `summaries/confidence_scheduled_verification.md`; harnesses `run_lead02_*.py`; artifacts `artifacts/lead02_confidence_*/`; worklog `archive/leads/lead_02_confidence_scheduled_verification.md`
- Lead 03 acceptance statistical power + cycle-jump: `summaries/acceptance_statistical_power.md`; harnesses `run_lead03_*.py`; store `dspark_oracle/stage2_capture_store.py`; artifacts `artifacts/acceptance_powered/`; worklog `archive/leads/lead_03_acceptance_statistical_power.md`

## Bench harness (canonical)

- `ds4-spec-bench` (built via `make ds4-spec-bench`) — the bulk-config JSONL harness; modes argmax/sample/speculative_argmax/teacher_force; `--dump-hidden-dir`, `--force-tokens-dir`, `--rewrite-frontier`. Do NOT loop over the `ds4` CLI for measurements.
