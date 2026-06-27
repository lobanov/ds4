# Phase 0 — Scope, Deliverables, Headline Findings

## Objective (from `PLAN.md`)

Build the measurement harness needed to reason about speculative speedup **before
adding DSpark**. The output of Phase 0 is a decision-grade picture of *where*
`ds4` spends time during speculative decode, for:

- target-only greedy decode, and
- the current `--mtp` speculative path.

## Work items (from `PLAN.md`, Phase 0)

1. Add opt-in timing instrumentation for decode cycles.
2. Split cycle timing into:
   - target sample / top-token selection
   - target decode step
   - speculative draft time
   - verifier time
   - replay / commit time
3. Record speculative quality counters:
   - draft length
   - committed length
   - acceptance by draft position
   - full accept count
   - partial accept count
4. Add a machine-readable output mode for benchmarking.

## Deliverables (from `PLAN.md`, Phase 0)

- Repeatable benchmark command set for local Metal first.
- CSV or TSV output for later analysis.
- Baseline numbers for: target-only greedy decode, and current `--mtp` path.

## Stop / Go gate

> Do not implement DSpark execution until baseline timing clearly identifies
> where `ds4` spends time in speculative decode.

Phase 0 does **not** gate on a speedup number. The primary speedup gate
(`>= 20%` greedy throughput vs. baseline, exact greedy preservation) belongs to
Phase 5. Phase 0 only needs to prove the measurement harness can see the
relevant cost components.

## How the current code maps to the work items

| Work item | Status | Where |
|-----------|--------|-------|
| 1. Opt-in cycle timing | **Mostly done** | `DS4_MTP_TIMING` in `ds4.c:27167` (spec path); `DS4_DECODE_PROFILE_DETAIL` for layer-level target timing. |
| 2a. Target sample/top-token selection | **Not isolated** | Host-side argmax cost folded into the cycle; not separately timed. Tiny, but listed. |
| 2b. Target decode step | **Partial** | Covered as the implicit cost of `ds4_session_eval(first_token)`; `DS4_DECODE_PROFILE_DETAIL` gives layer detail but on a different channel. |
| 2c. Speculative draft time | **Done** | `draft=…ms` in every `DS4_MTP_TIMING` line. |
| 2d. Verifier time | **Done** | `verify=…ms` (+ `snapshot`, `prefix`, `exact_replay`, `replay` splits) per path. |
| 2e. Replay/commit time | **Done** | `prefix`/`replay`/`exact_replay` columns. |
| 3. Quality counters (draft len, committed len) | **Per-cycle only** | Logged as `drafted=N committed=M` in `DS4_MTP_TIMING`/`DS4_MTP_CONF_LOG`. No aggregate. |
| 3. Acceptance by position | **Missing** | Not collected. Highest-value counter gap. |
| 3. Full / partial accept counts | **Per-cycle only** | `DS4_MTP_SPEC_LOG` emits per-cycle accept/partial/miss; no tally. |
| 4. Machine-readable output | **Missing** | All output is stderr free-text. `ds4-bench` CSV has no spec columns. |

See `01_existing_instrumentation.md` for the full inventory and
`02_gap_and_spec.md` for the precise deltas to close the gaps.

## Key architectural facts that shape Phase 0

- The speculative path is **MTP-only** today. It is entered from `ds4_cli.c:483`
  (plain generate) and `ds4_cli.c:1154` (chat), gated on
  `temperature <= 0 && ds4_engine_mtp_draft_tokens(engine) > 1`. The engine entry
  point is `ds4_session_eval_speculative_argmax()` (`ds4.c:27167`).
- The production MTP depth is **2** (`--mtp-draft 2`; default `mtp_draft_tokens=1`
  in `ds4_cli.c:1397` means spec is off unless raised). Recursive drafting caps at
  `drafts[16]` (`ds4.c:27218`) and the engine clamps to 16 (`ds4.c:25561`).
- The verifier has multiple paths, all already individually timed under
  `DS4_MTP_TIMING`:
  - margin-skip 1-token early exit (`ds4.c:~27300`),
  - exact N=2 verifier `metal_graph_verify_decode2_exact` (`ds4.c:21218`),
  - layer-major batch verifier `metal_graph_verify_suffix_tops` (`ds4.c:21117`)
    with full-accept, prefix-1-capture, and snapshot+restore/replay sub-paths,
  - sequential exact fallback (`ds4.c:~27660`).
- `ds4-bench` builds its `ds4_engine_options` at `ds4_bench.c:516` **without**
  `mtp_path`, so even if a user passes `--mtp` flags they are ignored; the bench
  generation loop (`ds4_bench.c:625`) is strictly one-token `session_eval`.

## What "done" looks like for Phase 0

1. `issue468/baseline/` contains, for at least one Metal machine and one prompt
   class: a target-only CSV (`ds4-bench`) and an `--mtp` run whose
   `DS4_MTP_TIMING`/`DS4_MTP_SPEC_LOG`/`DS4_MTP_CONF_LOG` output is captured.
2. `04_run_matrix.md` tables are filled with measured prefill_tps / gen_tps /
   per-cycle draft/verify/total ms / mean committed length / full-accept rate.
3. The instrumentation gap in `02` is either (a) implemented behind a new
   diagnostic env var, or (b) explicitly judged unnecessary because the existing
   stderr logs + a small post-processor are sufficient for Phase 1.
4. A one-paragraph Phase 0 conclusion states where `ds4` spends speculative time
   and whether the verifier economics look plausible enough to start Phase 1.
