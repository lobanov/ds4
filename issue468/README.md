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
| `04_run_matrix.md` | The run matrix and result tables, **filled with measured baseline numbers + Phase 0 conclusion**. |
| `parse_spec_log.py` | Parser for `DS4_MTP_TIMING` logs → aggregate T3/T4/T5 tables (reusable). |
| `prompts/` | Fixed, version-controlled prompts so baseline runs are reproducible. |
| `baseline/` | Raw run outputs (CSV, logs, machine fingerprint, parsed summary). |

## Status

- [x] Reconnaissance of existing measurement surfaces (`01`).
- [x] Gap analysis + instrumentation hand-off spec (`02`).
- [x] Repeatable command set + protocol (`03`).
- [x] Run matrix + result templates (`04`).
- [x] **Execute** the baseline command set on M5 Max / Metal; fill `baseline/`
      + `04` tables.
- [ ] Implement the minimal instrumentation delta in `02` — **deferred**;
      existing `DS4_MTP_TIMING` + `parse_spec_log.py` are sufficient for Phase 1.
- [x] Phase 0 sign-off: we can see *where* `ds4` spends time in spec decode →
      **PROCEED to Phase 1, narrowed to the exact-verifier cost curve**.

## Headline findings (measured: Apple M5 Max / 128 GB / Metal, SHA `c7ef1bf`)

1. **The verifier owns the cycle.** Draft ≈ 2 ms, snapshot/prefix/replay ≈
   sub-ms; the verifier is 33 ms/cycle (fast batch) to 67 ms/cycle (exact),
   against a 28 ms baseline decode step. PLAN.md's "main risk" is confirmed.
2. **Default depth-2 MTP is only ~+6% (chat) / +7% (code)** — far under the 20%
   gate — and it achieves that by **not preserving exact greedy output**.
3. **The exact-greedy path (`--quality`) is a −28% regression.** The success
   criterion demands exactness, and exact verification is super-linear (67 ms
   for 2 tokens ≈ 2.4× a single step, worse than 2× sequential). Making a
   verifier that is *both cheap and exact* is the make-or-break problem.
4. **MTP suffix acceptance is weak.** Position-2 conditional acceptance is only
   **0.41–0.42** (DFlash ≈ 0.63–0.72 in paper Fig 2). Even a free verifier
   would cap gains at current acceptance.
5. **Most Phase 0 instrumentation already existed.** `DS4_MTP_TIMING` + a small
   parser (`parse_spec_log.py`) delivered the full T3/T4/T5 breakdown with zero
   engine change. The `DS4_SPEC_STATS` delta in `02` is deferred to Phase 4.
6. **`ds4-bench` does not exercise the speculative path** (still the biggest
   harness gap), and `DS4_DECODE_PROFILE_DETAIL` is **CPU-only** on this build —
   the decode-step denominator is taken from throughput. See `01`/`02`.

## Conventions

- All `file:line` references are relative to the repo root
  (`/Users/lobanov/Projects/ds4-dspark`).
- Main model (explicit path — set as `MODEL` before running commands):
  `../ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`
  (the repo's `ds4flash.gguf` is a symlink to this file).
- MTP auxiliary (set as `MTP`):
  `../ds4/gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf`.
- Primary backend for this phase: **Metal** (local Apple Silicon).
