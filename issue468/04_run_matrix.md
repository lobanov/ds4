# Phase 0 — Run Matrix and Result Tables

Fill these in after executing `03_baseline_command_set.md`. One row per
(prompt class × context × path). Keep raw logs in `issue468/baseline/`.

## Machine header (fill once)

| Field | Value |
|---|---|
| Machine / chip | |
| macOS version | |
| RAM / GPU | |
| Backend | Metal |
| Main GGUF | `../ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf` |
| MTP GGUF | `../ds4/gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf` |
| git SHA | |
| Power setting | `--power 100` |
| Ambient / thermal notes | |

Raw fingerprint file: `issue468/baseline/machine.txt`.

## T1 — Target-only throughput (structured CSV, via `ds4-bench`)

Source: `bench_target_only.csv`. One row per context frontier.

| ctx_tokens | prefill_tps | gen_tps | notes |
|---|---|---|---|
| 2048 | | | |
| 4096 | | | |
| 6144 | | | |
| 8192 | | | |

`gen_tps` here is the **denominator** every speculative number must beat.

## T2 — Apples-to-apples CLI throughput (target-only vs `--mtp`, same harness)

Fixed: `--temp 0 --seed 1 -n 256 -c 8192 --power 100 --mtp-draft 2`.
Throughput from the CLI summary line `ds4: … generation: X t/s`.

| Prompt class | Path | Env knobs | gen t/s | raw log |
|---|---|---|---|---|
| chat/general | target-only | `DS4_MTP_SPEC_DISABLE=1` | | `cli_target_only_chat.log` |
| chat/general | `--mtp` | `DS4_MTP_TIMING` etc. | | `cli_mtp_chat.log` |
| code | target-only | `DS4_MTP_SPEC_DISABLE=1` | | `cli_target_only_code.log` |
| code | `--mtp` | `DS4_MTP_TIMING` etc. | | `cli_mtp_code.log` |

Speedup of `--mtp` over target-only = `mtp_gen_tps / target_gen_tps` per row.

## T3 — Speculative cycle cost (from `DS4_MTP_TIMING` lines)

Mean over all cycles in the run. Parse the `mtp timing …` lines from the
matching `cli_mtp_*.log`. Columns map 1:1 to `PLAN.md` Phase 0 work items 2c/2d/2e.

| Prompt class | cycles | mean us_draft | mean us_verify | mean us_snapshot | mean us_replay | mean us_total |
|---|---|---|---|---|---|---|
| chat/general | | | | | | |
| code | | | | | | |

`us_verify` = the `verify=` column; `us_snapshot`/`us_replay` from the matching
path-tag columns. Use whichever path tag dominates the run (record the mix in
T5).

## T4 — Speculative quality (the DSpark-relevant counters)

`mean_committed` = accepted tokens per cycle **including** the first target
token (matches the paper's reported τ convention, footnote 3). Full/partial/miss
counts derived from `DS4_MTP_SPEC_LOG`. Position-1 acceptance from
`DS4_MTP_PROBE` `hit=H/T` aggregated over the run.

| Prompt class | mean drafted | mean committed | full accept | partial accept | miss-first | pos-1 accept rate |
|---|---|---|---|---|---|---|
| chat/general | | | | | | |
| code | | | | | | |

> Position-1 acceptance is the highest-leverage number for DSpark (paper §4.3.1:
> the parallel backbone's capacity advantage at position 1 is what makes the
> whole scheme viable). If MTP's pos-1 rate is already low on `ds4`, DSpark's
> parallel backbone must clear a higher bar.

## T5 — Verifier-path mix (which `DS4_MTP_TIMING` tag dominated)

Counts of each path tag per run. Tells us which verifier economics to model in
Phase 1.

| Prompt class | margin_skip | decode2_full | decode2_partial | micro_full | micro_prefix1 | micro_replay | seq |
|---|---|---|---|---|---|---|---|
| chat/general | | | | | | | |
| code | | | | | | | |

## T6 — Optional: per-verifier-path cost at depth 2 (Phase 1 preview)

From the isolated-knob runs (`DS4_MTP_STRICT`, `DS4_MTP_BATCH_VERIFY`,
`DS4_MTP_CAPTURE_PREFIX1`, `DS4_MTP_FORCE_SNAPSHOT`). Mean `us_total` per cycle,
chat prompt.

| Verifier path | mean us_total | mean committed | notes |
|---|---|---|---|
| default (unmodified) | | | reference |
| `DS4_MTP_STRICT` (exact N=2) | | | quality path |
| `DS4_MTP_BATCH_VERIFY` | | | |
| `DS4_MTP_CAPTURE_PREFIX1` | | | |
| `DS4_MTP_FORCE_SNAPSHOT` | | | worst-case replay |

## T7 — Target decode-step cost (denominator sanity, from `DS4_DECODE_PROFILE_DETAIL`)

From `target_decode_profile.log`: the per-layer encode/execute ms for a single
target decode step, summed. This is the `plain target decode cost per emitted
token` from `PLAN.md`'s working hypothesis.

| metric | value |
|---|---|
| sum of per-layer decode ms (1 step) | |
| inferred target-only t/s | |
| matches T1/T2? | yes/no — explain |

## Correctness gate

| Check | Result |
|---|---|
| `diff` target-only vs `--mtp` token stream (chat) | pass / FAIL |
| `diff` target-only vs `--mtp` token stream (code) | pass / FAIL |

Any FAIL blocks Phase 0 sign-off.

## Phase 0 conclusion (to write after the tables are filled)

One paragraph answering:

1. Where does `ds4` spend time in a speculative cycle (draft vs verify vs
   replay)?
2. Is the current `--mtp` path faster than target-only, by how much, on which
   prompt class?
3. Does the position-1 acceptance and mean committed length leave room for a
   DSpark parallel-backbone + Markov head to plausibly win (Phase 2 simulator
   input)?
4. Decision: **proceed to Phase 1** (verifier cost curve), **narrow scope**, or
   **stop**.
