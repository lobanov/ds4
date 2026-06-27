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
| `05_phase1_plan.md` | **Phase 1 execution plan — EXECUTED (Branch B selected).** Microbench design, placement, matrix, commands. |
| `06_phase1_results.md` | **Phase 1 results — verify(L) curves + exact-verifier breakdown + Branch B verdict.** |
| `07_branch_b_options.md` | **Branch B refinements: B1 margin-gated fallback vs B2 rejection sampling**, with the temp=0 non-determinism premise that tilts the choice. |
| `08_phase2_quality_results.md` | **Phase 2a quality baseline — EXECUTED.** batch=61/92 vs exact=67/92 (both reproducible): the batch verifier's drift is a real greedy-argmax regression → **confirms B2.** |
| `09_dspark_integration_plan.md` | **Phase 3+ plan: DSpark official-tensor integration on Branch B (B2)** — extract mtp.* from DeepSeek-V4-Flash-DSpark, port the drafter, implement rejection sampling, measure acceptance + speedup. |
| `run_quality_baseline.sh` / `diff_quality.py` | **Branch B precision baseline harness.** Runs batch + exact ×R, diffs against the reference target-only run via per-case consensus + run-to-run flip rates. Uses the **exact path (--quality, same decode kernel) as the Metal-non-determinism noise floor.** |
| `run_determinism_probe.sh` | **Determinism probe** (`07` §5.1): two identical target-only temp=0 runs, `cmp` — confirms/quantifies Metal temp=0 non-determinism at the token level, feeding the B1-vs-B2 choice. |
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
- [x] Update `PLAN.md` with Phase 0 outcome + deferred Branch A/B.
- [x] Prepare Phase 1 plan (`05_phase1_plan.md`).
- [x] **Execute Phase 1**: `ds4_engine_verifier_curve_test` + `--verifier-curve-test`
      + `DS4_VERIFY_CURVE_BREAKDOWN`; verify(L) for L=1..8 on all three kernels
      at ctx {2k,4k,8k}, chat + code (`06_phase1_results.md`).
- [x] **Phase 1 decision: Branch B.** Only the batch verifier is sub-linear;
      exact is linear-or-worse (95% of exact cost is the 2× layer dispatches).
      Phase 2 proceeds on Branch B; exact-greedy dropped from the primary gate.
- [x] Branch B refinement analysis (`07_branch_b_options.md`) — B1 vs B2, with
      the temp=0 non-determinism premise that tilts the choice.
- [~] **Branch B precision baseline** (`run_quality_baseline.sh` +
      `diff_quality.py`): `ds4-eval` wired to exercise the `--mtp` speculative
      path (it previously loaded MTP but never called it); runs batch + exact ×R
      against the reference target-only run (67/92). **Accounts for Metal temp=0
      non-determinism** by using the exact path as the noise floor — batch
      drift is Branch-B-attributable only if batch's flip rate exceeds exact's.
- [x] **Phase 2a result (`08`): batch=61/92 vs exact=67/92, both reproducible**
      → minimal Branch B (greedy-argmax on batch verifier) is a real regression,
      NOT Metal nondeterminism. **Confirms B2 (rejection sampling).**
- [x] **Phase 3+ plan (`09`): DSpark official-tensor integration on B2** —
      extract mtp.* from deepseek-ai/DeepSeek-V4-Flash-DSpark, port the 3-layer
      parallel-backbone + Markov drafter, implement rejection sampling over the
      batch verifier, measure acceptance + speedup.
- [ ] Determinism probe (`07` §5.1, `run_determinism_probe.sh`): running now
      (after the target control runs release the instance lock).
- [ ] Phase 3: drafter tensor extraction + GGUF conversion.
- [ ] Phase 4: DSpark drafter forward in ds4 (Metal).
- [ ] Phase 5: B2 rejection-sampling verification.
- [ ] Phase 6: empirical acceptance + speedup.

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
