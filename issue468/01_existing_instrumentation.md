# Existing Instrumentation Inventory

Everything `ds4` can already tell us about decode-cycle timing and speculative
quality, with the env var that turns it on and the exact code site. **This
exists today; no code change is required to use any of it.**

The single most important fact: the speculative cycle already emits a full ms
breakdown under `DS4_MTP_TIMING`. Phase 0 is mostly a *collection and
structuring* problem, not a *build new instrumentation* problem.

## 1. Speculative-cycle timing — `DS4_MTP_TIMING`

**Site:** `ds4_session_eval_speculative_argmax()` at `ds4.c:27167`.
**Activation:** `DS4_MTP_TIMING` (checked at `ds4.c:27253`).
**Output:** stderr, one line per speculative cycle, format depends on the
verifier path actually taken. The columns are split precisely along the
`PLAN.md` Phase 0 work-item boundaries:

| `DS4_MTP_TIMING` column | PLAN.md Phase 0 component |
|-------------------------|---------------------------|
| `draft=…ms` | speculative draft time (item 2c) |
| `snapshot=…ms` | replay/commit cost: state snapshot before verify (item 2e) |
| `verify=…ms` | verifier time (item 2d) |
| `prefix=…ms` | replay/commit cost: prefix-1 commit on partial accept (item 2e) |
| `exact_replay=…ms` | replay/commit cost: exact replay of accepted prefix (item 2e) |
| `replay=…ms` | replay/commit cost: generic replay after restore (item 2e) |
| `total=…ms` | whole speculative cycle (draft + verifier + replay) |
| `drafted=N committed=M` | draft length + committed length (item 3) |

One line is emitted for each path actually executed, identified by a tag in the
message. Path tags observed in the source:

| Tag | Path | Site |
|-----|------|------|
| `mtp timing margin-skip` | margin-gated early exit, commit 1 | `ds4.c:~27300` |
| `mtp timing decode2 … committed=2` | exact N=2 full accept | `ds4.c:~27360` |
| `mtp timing decode2 … committed=1` | exact N=2 partial accept + prefix-1 | `ds4.c:~27401` |
| `mtp timing micro … committed=N` | batch verifier full accept | `ds4.c:~27466` (no `noreplay`) |
| `mtp timing micro … noreplay=1` | batch verifier, prefix-1 capture, no replay | `ds4.c:~27531` |
| `mtp timing micro … exact_replay` | batch verifier, exact replay debug | `ds4.c:~27510` |
| `mtp timing micro … replay` | batch verifier, snapshot+restore+replay | `ds4.c:~27622` |
| `mtp timing seq …` | sequential exact fallback | `ds4.c:~27695` |

**What it does NOT give us (gaps):** target sample/top-token host time is not
split out (it is tiny and folded into the cycle); the verifier `verify=` column
does not separate the first-token logits readback; and the output is free-text,
not parseable without a regex post-processor.

## 2. Per-cycle confidence / margin — `DS4_MTP_CONF_LOG`

**Site:** `ds4.c` (read alongside `DS4_MTP_TIMING`; `mtp_conf_log` flag at
`ds4.c:27252`). **Output:** one line per cycle:
```
ds4: mtp conf drafted=N committed=M mtp_top=T runner=R margin=… target_next=… draft_next=…
```
Useful for correlating acceptance with draft-token confidence — a direct proxy
for the "acceptance by draft position" analysis once aggregated. Computes the
logit top-1/top-2 margin of the *first* draft position only.

## 3. Per-cycle accept / partial / miss events — `DS4_MTP_SPEC_LOG`

**Site:** several `fprintf(stderr, "ds4: mtp spec …")` calls in
`ds4_session_eval_speculative_argmax()` and the sequential fallback
(`ds4.c:27257`, `ds4.c:~27699`, `ds4.c:~27753`, etc.). **Output:** event lines:
```
ds4: mtp spec miss first draft=…
ds4: mtp spec seq miss at=… draft=… base=… drafted=… accepted=…
ds4: mtp spec seq accept drafted=… accepted=…
ds4: mtp spec seq partial drafted=… verified=… accepted=…
```
This is the raw event stream from which full-accept / partial-accept / miss
tallies can be derived by post-processing.

## 4. Per-token draft accuracy in the *target-only* path — `DS4_MTP_PROBE`

**Site:** `ds4_session_eval_internal()` tail, `ds4.c:27110`. **Activation:**
`DS4_MTP_PROBE`. **What it does:** even when speculative decode is **off**, the
MTP block is run opportunistically each target step and its top-1 is compared to
the target's actual next token. Counters `s->mtp_probe_total` /
`s->mtp_probe_hit` accumulate over the whole session and are logged per step:
```
ds4: mtp probe token=… draft=… hit=H/T
```
This is the cleanest existing source of **position-1 acceptance rate** (the
single most important DSpark lever per the paper, §4.3.1). It runs without
committing any speculative state, so it is safe to enable alongside the
target-only baseline.

## 5. Apples-to-apples target-only control — `DS4_MTP_SPEC_DISABLE`

**Site:** `ds4_cli.c:483` (and `ds4_cli.c:1154` for chat). When set, the CLI
takes the `else` branch and calls `ds4_session_eval(token)` one token at a time,
even with `--mtp` loaded. Use this to measure the **target-only baseline through
the exact same CLI code path** as the `--mtp` run, eliminating harness noise.

## 6. Layer-level target decode/prefill profiling — `DS4_DECODE_PROFILE_DETAIL`

**Site:** `ds4.c:7816`, `ds4.c:7921`, `ds4.c:9693` (and additional prefill
profile prints around `ds4.c:20430`–`20882`). **Activation:**
`DS4_DECODE_PROFILE_DETAIL`. **Output:** per-layer encode/execute ms for
decode-step attention/FFN and for prefill. This is how we answer "what does a
single target decode step cost, layer by layer" — i.e. the denominator for any
speculative speedup claim. It is a different output channel from `DS4_MTP_TIMING`
and not cycle-aligned with the speculative stream, but it is sufficient for the
Phase 0 cost-curve sanity check.

## 7. Verifier-path knobs (select *which* verifier is measured)

These do not emit timings on their own, but they change which code path
`DS4_MTP_TIMING` reports. They are essential for Phase 0 because they let us
isolate the cost of each verifier strategy at depth 2:

| Env var | Effect | Site |
|---------|--------|------|
| `DS4_MTP_STRICT` | force exact (quality) verifier; preserves the one-token target stream | `ds4.c:27251` |
| `DS4_MTP_BATCH_VERIFY` | disable exact N=2, force the batch verifier | `ds4.c:27284` |
| `DS4_MTP_CAPTURE_PREFIX1` | force prefix-1 capture path for partial accepts | `ds4.c:~27470` |
| `DS4_MTP_FORCE_SNAPSHOT` | force snapshot+restore for every cycle | `ds4.c:~27470` |
| `DS4_MTP_EXACT_REPLAY` | exact-replay debug path | `ds4.c:~27470` |
| `DS4_MTP_MIN_MARGIN=<f>` | margin threshold for the early 1-token skip | `ds4.c:27255` |
| `DS4_MTP_FULL_LOGITS` | read back full-vocab logits each draft (cost diagnostic) | `ds4.c:27253` |
| `--quality` / `--mtp-margin` | CLI equivalents for strict + min-margin | `ds4_cli.c:1459` |

## 8. Structured throughput CSV — `ds4-bench`

**Site:** `ds4_bench.c:584` (header) and `ds4_bench.c:643` (row).
**Output columns:** `ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,kvcache_bytes`.
**Critical limitation:** the generation loop (`ds4_bench.c:625`) uses
`ds4_session_argmax_excluding(eos)` + `ds4_session_eval(token)` and the engine
options struct (`ds4_bench.c:516`) never sets `mtp_path`. So `ds4-bench`
**cannot produce a speculative CSV today**. It is an excellent target-only
harness and the natural place to add a `--mtp`/`--spec` mode (see `02`).

There is also `--dump-frontier-logits-dir` (`ds4_bench.c:353`) which writes
full-vocab logits JSON per frontier — useful later for DSpark draft-quality
offline analysis, not needed for Phase 0 timing.

## 9. Timing primitives

- `now_sec()` (`ds4.c:740`) — the `CLOCK_MONOTONIC` helper used by all the MTP
  timing prints.
- `bench_now_sec()` (`ds4_bench.c:54`) — the bench tool's equivalent.
- `cli_now_sec()` (`ds4_cli.c`) — the CLI's equivalent, used for the
  prefill/decode t/s summary line at `ds4_cli.c:532`.

## Summary: what we can measure today with zero code change

| Question | Answer using today's code |
|----------|----------------------------|
| Per-cycle draft / verify / replay ms for `--mtp`? | `DS4_MTP_TIMING` |
| Which verifier path each cycle took? | path tag in `DS4_MTP_TIMING` line |
| Position-1 acceptance of the MTP drafter? | `DS4_MTP_PROBE` (target-only path) |
| Per-cycle drafted vs committed length? | `drafted=/committed=` in timing + `DS4_MTP_CONF_LOG` |
| Full vs partial vs miss event stream? | `DS4_MTP_SPEC_LOG` |
| Per-layer target decode step cost? | `DS4_DECODE_PROFILE_DETAIL` |
| Apples-to-apples target-only baseline? | `DS4_MTP_SPEC_DISABLE` (CLI) or `ds4-bench` |
| Structured target-only throughput CSV? | `ds4-bench --csv …` |
| **Structured `--mtp` throughput CSV?** | **Not available — biggest gap.** |
| **Acceptance-by-position histogram?** | **Not available — second gap.** |
| **Aggregate full/partial/miss tallies?** | **Not available — derivable from logs only.** |
