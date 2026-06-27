# DSpark Feasibility — Issue 468: Phase 0 Research Dossier

Research artifacts for **Phase 0: Baseline Measurement and Instrumentation** of
`PLAN.md`. The goal of this phase is to build the measurement harness needed to
reason about DSpark-style speculative speedup *before* touching the engine for
DSpark itself.

> Phase 0 Stop/Go gate (from `PLAN.md`):
> *Do not implement DSpark execution until baseline timing clearly identifies
> where `ds4` spends time in speculative decode.*

## What is in this dossier

| File | Purpose |
|------|---------|
| `00_phase0_scope.md` | Objective, deliverables, success-criteria mapping, headline findings. |
| `01_existing_instrumentation.md` | **Reconnaissance.** Every timing/diagnostic surface that already exists, with `file:line` and the env var that activates it. |
| `02_gap_and_spec.md` | Gap analysis vs. `PLAN.md` Phase 0 work items, plus the **minimal instrumentation spec** (counters, CSV schema, exact code touchpoints). |
| `03_baseline_command_set.md` | The repeatable command set for local Metal: target-only and `--mtp`, plus machine/thermal protocol. |
| `04_run_matrix.md` | The run matrix and result-table templates to fill in once the baseline is executed. |
| `prompts/` | Fixed, version-controlled prompts so baseline runs are reproducible. |
| `baseline/` | Landing directory for raw run outputs (CSV/logs). Populated by executing `03_baseline_command_set.md`. |

## Status

- [x] Reconnaissance of existing measurement surfaces (`01`).
- [x] Gap analysis + instrumentation hand-off spec (`02`).
- [x] Repeatable command set + protocol (`03`).
- [x] Run matrix + result templates (`04`).
- [ ] **Execute** the baseline command set and fill `baseline/` + `04` tables.
- [ ] Implement the minimal instrumentation delta in `02` (only the parts not
      already covered by `DS4_MTP_TIMING` / `DS4_MTP_*`).
- [ ] Phase 0 sign-off: confirm we can see *where* `ds4` spends time in spec
      decode, then open Phase 1 (verifier cost curve).

## Headline findings (TL;DR)

1. **Most of the Phase 0 timing split already exists.** `DS4_MTP_TIMING` emits a
   per-cycle breakdown (draft / snapshot / verify / prefix / replay / total, in
   ms) for every verifier path in `ds4_session_eval_speculative_argmax()`
   (`ds4.c:27167`). `DS4_DECODE_PROFILE_DETAIL` gives layer-level target
   decode/prefill timing. See `01`.

2. **`ds4-bench` does not exercise the speculative path at all.** Its generation
   loop (`ds4_bench.c:625`) calls `ds4_session_argmax_excluding` +
   `ds4_session_eval` one token at a time and never passes `mtp_path` to the
   engine. So the only structured-CSV throughput harness we have today measures
   **target-only** decode. There is no bench harness for `--mtp`. This is the
   single biggest Phase 0 gap. See `02`.

3. **All MTP timing/counter output is unstructured stderr free-text.** It is not
   machine-parseable, there is no aggregate rollup, and there is no
   acceptance-by-position histogram or full/partial-accept tally. This is the
   second gap; the fix is small and localized. See `02`.

4. **A usable preliminary baseline can be captured with zero code change.** Run
   `ds4-bench` for target-only, and run the `ds4` CLI with `--mtp --mtp-draft 2`
   plus `DS4_MTP_TIMING`/`DS4_MTP_SPEC_LOG`/`DS4_MTP_CONF_LOG` for the spec path,
   using `DS4_MTP_SPEC_DISABLE` for the apples-to-apples target-only control
   through the same code path. See `03`.

## Conventions

- All `file:line` references are relative to the repo root
  (`/Users/lobanov/Projects/ds4-dspark`).
- Main model (explicit path — set as `MODEL` before running commands):
  `../ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`
  (the repo's `ds4flash.gguf` is a symlink to this file).
- MTP auxiliary (set as `MTP`):
  `../ds4/gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf`.
- Primary backend for this phase: **Metal** (local Apple Silicon).
