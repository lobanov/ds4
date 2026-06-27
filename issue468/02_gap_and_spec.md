# Gap Analysis and Minimal Instrumentation Spec

## 1. Gap analysis vs. `PLAN.md` Phase 0

| Phase 0 requirement | Current state | Gap |
|---|---|---|
| Opt-in cycle timing | `DS4_MTP_TIMING` covers the whole spec cycle | None for spec; target-decode host time not split (minor). |
| Split: draft / verify / replay | `draft`/`snapshot`/`verify`/`prefix`/`exact_replay`/`replay` columns exist | None. |
| Split: target sample / top-token selection | Not measured separately | Tiny host cost; defer unless `DS4_DECODE_PROFILE_DETAIL` shows it matters. |
| Quality counter: draft length | `drafted=N` per cycle | No aggregate. |
| Quality counter: committed length | `committed=M` per cycle | No aggregate / no distribution. |
| Quality counter: acceptance by draft position | **Missing** | Highest-value gap. Needed for DSpark break-even math. |
| Quality counter: full accept count | Per-cycle event only (`DS4_MTP_SPEC_LOG`) | No tally. |
| Quality counter: partial accept count | Per-cycle event only | No tally. |
| Machine-readable output | **Missing** | All spec output is stderr free-text; `ds4-bench` CSV has no spec columns. |
| `--mtp` baseline numbers | **Not collectable via bench** | `ds4-bench` cannot drive the spec path at all. |

**Two real gaps, both small and localized:**

1. **No structured `--mtp` benchmark.** `ds4-bench` does not load MTP and does
   not call the speculative path. The CLI *can* drive it, but only with
   free-text stderr timing.
2. **No aggregate quality counters / no acceptance-by-position histogram.** The
   per-event data is emitted, but nothing accumulates it.

Everything else in Phase 0 is a *collection* task, not a *build* task.

## 2. Minimal instrumentation spec (hand-off)

This is the smallest change that closes both gaps without disturbing the
production path. It is **opt-in via env vars**, in the same style as the
existing `DS4_MTP_*` diagnostics, and it touches only the speculative path and
the bench tool. It is intentionally *not* a new public API.

### 2.1 New diagnostic env var: `DS4_SPEC_STATS`

- **Purpose:** accumulate per-session speculative quality counters and emit a
  machine-readable per-cycle row + a final summary.
- **Activation:** presence of `DS4_SPEC_STATS` (optionally
  `DS4_SPEC_STATS_FILE=/path` to write the CSV there instead of stderr).
- **Default behavior unchanged** when unset.

### 2.2 Counter accumulator (private to `ds4.c`)

Add a small struct to the session (next to `mtp_probe_total` / `mtp_probe_hit`
at the `ds4_session` definition, near `ds4.c:23282`):

```c
typedef struct {
    bool     enabled;
    /* per-session tallies over all speculative cycles */
    uint64_t cycles;           /* cycles that attempted a draft suffix */
    uint64_t full_accept;      /* committed == drafted */
    uint64_t partial_accept;   /* 0 < committed < drafted (excludes first) */
    uint64_t miss_first;       /* first draft rejected, committed == 1 */
    uint64_t sum_drafted;
    uint64_t sum_committed;    /* includes the first target token */
    /* acceptance by draft position (index = position within drafts[]) */
    uint64_t pos_trials[16];   /* times position k was reached & verifiable */
    uint64_t pos_accepts[16];  /* times position k was accepted */
    /* latency sums, microseconds, for rollup (mirrors DS4_MTP_TIMING) */
    uint64_t us_draft_total;
    uint64_t us_verify_total;  /* snapshot+verify+prefix/replay, per path taken */
    uint64_t us_total_total;
    /* CSV sink */
    FILE    *csv;
} ds4_spec_stats;
```

The acceptance-by-position invariant is already computed inside
`ds4_session_eval_speculative_argmax` as `commit_drafts`
(`ds4.c:~27488`): for a cycle that drafted `draft_n` and committed
`commit_drafts`, record:

```
for k in [0, commit_drafts): pos_accepts[k]++, pos_trials[k]++
for k in [commit_drafts, draft_n): pos_trials[k]++   /* reached but rejected at k */
```

This is **the** DSpark-relevant metric: `pos_accepts[k]/pos_trials[k]` is the
empirical conditional acceptance at position k that the paper (§4.3.1, Figure 2)
shows DSpark must beat.

### 2.3 Emission points

Insert accumulation calls at the **5 commit points** where `accepted[]` is
finalized inside `ds4_session_eval_speculative_argmax` (each currently returns
`n_accept`):

| Commit point | Site | drafted / committed available as |
|---|---|---|
| margin-skip early exit | `ds4.c:~27300` | `draft_n` (=2), committed=1 |
| decode2 full accept | `ds4.c:~27360` | `draft_n`=2, committed=2 |
| decode2 partial (prefix-1) | `ds4.c:~27401` | `draft_n`=2, committed=1 |
| micro verifier full accept | `ds4.c:~27466` | `draft_n`, `commit_drafts=draft_n` |
| micro verifier partial (prefix-1 / replay) | `ds4.c:~27531`, `~27622` | `draft_n`, `commit_drafts` |
| sequential fallback | `ds4.c:~27695` | `draft_n`, `verified` |

At each, the values needed (`draft_n`, `commit_drafts`/`verified`, the ms
breakdown already computed for `DS4_MTP_TIMING`) are all in scope. The cleanest
factoring is one static helper:

```c
static void spec_stats_record(ds4_session *s,
                              int draft_n, int committed_drafts,
                              double us_draft, double us_verify, double us_total);
```

called once per finalized cycle, plus a `spec_stats_emit_summary(s)` hooked from
session free (or CLI exit). The latency values are already computed by
`now_sec()` at the existing `mtp_timing` checkpoints; reuse those `double`s when
`DS4_SPEC_STATS` is set rather than re-timing.

### 2.4 Per-cycle CSV row

```
cycle,drafted,committed,full,partial,miss_first,verifier_path,us_draft,us_verify,us_total
```

`verifier_path` is one of: `margin_skip`, `decode2_full`, `decode2_partial`,
`micro_full`, `micro_prefix1`, `micro_replay`, `seq`. This mirrors the existing
`DS4_MTP_TIMING` path tags so cross-referencing is trivial.

### 2.5 Final summary block (human + machine readable)

```
# ds4 spec stats
cycles=N full=A partial=B miss_first=C
drafted_total=D committed_total=E mean_committed=E/cycles
acceptance_by_position k=1..N: a1/t1 a2/t2 …
mean_us_draft=… mean_us_verify=… mean_us_total=…
```

`mean_committed` and the acceptance-by-position array are the two numbers that
feed directly into the Phase 2 offline feasibility simulator.

### 2.6 `ds4-bench` speculative mode (closes the structured-CSV gap)

Add three flags to `ds4-bench` (`ds4_bench.c` `parse_options`, after
`--quality`):

- `--mtp PATH` → set `c.engine.mtp_path` (new `bench_config` field)
- `--mtp-draft N` → set `c.engine.mtp_draft_tokens`
- `--spec` → drive the speculative path in the generation loop instead of
  one-token `session_eval`.

Then:
1. Forward these into the `ds4_engine_options` at `ds4_bench.c:516`.
2. In the generation loop (`ds4_bench.c:625`), when `cfg.spec` is set, mirror
   the CLI loop at `ds4_cli.c:477`–`520`: take `argmax_excluding(eos)` as
   `first_token`, call `ds4_session_eval_speculative_argmax`, and accumulate
   emitted tokens. The snapshot/restore around the loop already exists
   (`ds4_bench.c:613`/`645`) and is correct for this.
3. Extend the CSV header (`ds4_bench.c:584`) with spec columns, emitted as 0/blank
   for the target-only path so old rows stay comparable:
   ```
   …,gen_tps,kvcache_bytes,spec_path,spec_committed_tps,spec_mean_committed,spec_full_accept_rate
   ```
   `spec_committed_tps` = `sum_committed / gen_sec` is the number that should
   beat `gen_tps` for MTP to be worthwhile.

This is the only change that touches a *tool* rather than the engine. It is
roughly 40–60 lines in `ds4_bench.c`, plus the `ds4_spec_stats` readout accessor
on the session (a small `ds4_session_spec_stats_summary()` added to `ds4.h`).

## 3. Implementation order (recommended)

1. **Capture the preliminary baseline first, with zero code change**, using
   `03_baseline_command_set.md`. This validates that the existing
   `DS4_MTP_TIMING` stream is sufficient to see the cost components.
2. If the preliminary baseline is enough to answer "where does `ds4` spend
   speculative time", implement **only §2.1–2.5** (the counter accumulator) to
   get the acceptance-by-position histogram, since that feeds Phase 2 directly.
3. Implement §2.6 (`ds4-bench --spec`) only if the CLI+post-processor workflow
   proves too noisy for repeatable CSV comparison. It is the larger change and
   the one most likely to disturb the bench tool's existing users.

## 4. Risks and guardrails

- The accumulator must be **zero-cost when `DS4_SPEC_STATS` is unset** (no extra
  `now_sec()` calls, no branches on hot paths beyond a single `if (stats->enabled)`).
- Do **not** add a permanent semantic variant behind a flag (per `AGENT.md`):
  `DS4_SPEC_STATS` is a diagnostic switch that validates the one release path,
  not a new mode.
- Preserve exact greedy output: the counters are read-only observers of
  `commit_drafts`, which is already computed for correctness. No acceptance
  decision may depend on them.
- The change is confined to the speculative path and `ds4-bench`; it must not
  affect SSD streaming, CUDA, distributed, or the default Metal path. Per
  `AGENT.md`, run `make`, `make test`, and a before/after `ds4-bench` CSV after
  implementing.
