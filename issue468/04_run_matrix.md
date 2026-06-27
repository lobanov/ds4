# Phase 0 — Run Matrix and Result Tables

Fill these in after executing `03_baseline_command_set.md`. One row per
(prompt class × context × path). Keep raw logs in `issue468/baseline/`.

## Machine header (fill once)

| Field | Value |
|---|---|
| Machine / chip | MacBook Pro (Mac17,6), **Apple M5 Max** |
| macOS version | 26.5.1 (build 25F80) |
| RAM / GPU | 128 GB unified |
| Backend | Metal |
| Main GGUF | `../ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf` |
| MTP GGUF | `../ds4/gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf` |
| git SHA | `c7ef1bf` |
| Power setting | `--power 100` |
| Ambient / thermal notes | single process, instance lock honoured, wall power |

Raw fingerprint file: `issue468/baseline/machine.txt`.

## T1 — Target-only throughput (structured CSV, via `ds4-bench`)

Source: `bench_target_only.csv`.

| ctx_tokens | prefill_tps | gen_tps | notes |
|---|---|---|---|
| 2048 | 464.97 | **37.40** | peak short-context decode |
| 4096 | 426.02 | 31.43 | |
| 6144 | 418.93 | 31.11 | |
| 8192 | 414.07 | 31.02 | long-context asymptote |

`gen_tps` here is the **denominator** every speculative number must beat.
Note: bench walks a long prompt (`promessi_sposi.txt`); the CLI runs below use
short prompts and land at ~35.6 t/s at 8192 ctx (prompt-class / KV-residency
difference, not a regression).

## T2 — Apples-to-apples CLI throughput (target-only vs `--mtp`, same harness)

Fixed: `--temp 0 --seed 1 -n 256 -c 8192 --power 100 --mtp-draft 2`.

| Prompt class | Path | Env knobs | gen t/s | vs target-only | raw log |
|---|---|---|---|---|---|
| chat/general | target-only | `DS4_MTP_SPEC_DISABLE=1` | **35.60** | — | `cli_target_only_chat.log` |
| chat/general | `--mtp` (default/fast) | `DS4_MTP_TIMING` etc. | **37.76** | **+6.1%** | `cli_mtp_chat.log` |
| chat/general | `--mtp --quality` (exact) | + `DS4_MTP_STRICT` via `--quality` | **25.65** | **−27.9%** | `cli_mtp_chat_strict.log` |
| code | target-only | `DS4_MTP_SPEC_DISABLE=1` | **35.78** | — | `cli_target_only_code.log` |
| code | `--mtp` (default/fast) | `DS4_MTP_TIMING` etc. | **38.21** | **+6.8%** | `cli_mtp_code.log` |

**Headline:** default depth-2 MTP gives only ~+6% locally on M5 Max, far below
the 20% gate. The exact-greedy (`--quality`) path is a **net 28% loss**.

## T3 — Speculative cycle cost (from `DS4_MTP_TIMING` lines)

Mean over all cycles in the run (parsed by `parse_spec_log.py`). Target-only
decode-step cost for reference: 1000/35.60 = **28.1 ms/token** (chat).

| Prompt class | path | cycles | mean us_draft | mean us_verify | mean us_snapshot | mean us_prefix/replay | mean us_total |
|---|---|---|---|---|---|---|---|
| chat | default | 88 | 2.17 | **33.12** | 0.00 | 0.02 | 35.34 |
| chat | strict (`--quality`) | 81 | 2.34 | **67.00** | 0.35 | 0.15 | 69.84 |
| code | default | 93 | 2.14 | **32.64** | 0.00 | 0.01 | 34.82 |

> **The entire speculative cost is in `verify`.** Draft is ~2 ms (negligible);
snapshot/prefix/replay are sub-ms. Verifier economics:
> - margin-skip verify (1 token) ≈ **26 ms** ≈ 0.93× one decode step
> - micro batch verify (2 tokens) ≈ **40 ms** vs 2×28=56 ms sequential → 0.71× (a win only on full accept)
> - decode2 **exact** verify (2 tokens) ≈ **67 ms** vs 56 ms sequential → **1.20× (slower than baseline)**

## T4 — Speculative quality (the DSpark-relevant counters)

`mean_committed` = accepted draft tokens per cycle; τ (paper convention, incl.
bonus) ≈ `mean_committed + 1`. Conditional acceptance at position k given
positions 0..k-1 accepted. Position 0 is 1.00 by construction (first draft is
pre-validated before timing).

| Prompt class | path | mean drafted | mean committed | τ (incl. bonus) | cond-accept pos-1 | cond-accept pos-2 |
|---|---|---|---|---|---|---|
| chat | default | 2.00 | 1.41 | ~2.41 | 1.00 | **0.41** |
| chat | strict | 1.99 | 1.57 | ~2.57 | 1.00 | **0.57** |
| code | default | 1.99 | 1.42 | ~2.42 | 1.00 | **0.42** |

> The default MTP drafter's **position-2 conditional acceptance is only ~0.41–0.42**.
> Paper Figure 2 (Qwen3-4B chat): DFlash pos-2 ≈ 0.63–0.72, DSpark ≈ stable.
> ds4's 1-layer MTP is materially weaker than DFlash at the suffix — this is
> the gap a DSpark parallel backbone + Markov head would need to close.

## T5 — Verifier-path mix (which `DS4_MTP_TIMING` tag dominated)

| Prompt class | margin_skip | micro | decode2 | seq |
|---|---|---|---|---|
| chat (default) | 46 | 42 | 0 | 0 |
| chat (strict) | 0 | 1 | 80 | 0 |
| code (default) | 50 | 43 | 0 | 0 |

## T6 — Per-verifier-path cost at depth 2 (chat, isolation knobs)

| Verifier path | env knob | gen t/s | mean us_total | mean us_verify | mean committed |
|---|---|---|---|---|---|
| default (reference) | — | **37.76** | 35.34 | 33.12 | 1.41 |
| batch verifier | `DS4_MTP_BATCH_VERIFY=1` | 38.04 | 35.00 | 32.80 | 1.41 |
| prefix-1 capture | `DS4_MTP_CAPTURE_PREFIX1=1` | 37.95 | 35.06 | 32.85 | 1.41 |
| snapshot+restore | `DS4_MTP_FORCE_SNAPSHOT=1` | 37.49 | 35.58 | 33.20 | 1.41 |
| exact N=2 | `--quality` (`DS4_MTP_STRICT`) | 25.65 | 69.84 | 67.00 | 1.57 |

> Replay/snapshot strategy is a **rounding error** (±0.5 ms, ±0.3 t/s). The only
> path that moves the needle is the exact verifier, and it moves it the wrong
> way (+34 ms/cycle). Phase 1 should focus on the *verifier kernel itself*, not
> on replay/snapshot plumbing.

## T7 — Target decode-step cost (denominator)

`DS4_DECODE_PROFILE_DETAIL` is **CPU-only** on this build (the Metal decode
path has no per-layer timing hook; the Metal layer prints are prefill-only).
Denominator is therefore derived from throughput, which is the cleaner number:

| metric | value |
|---|---|
| target-only gen t/s (chat, CLI) | 35.60 |
| **single decode-step cost** | **28.1 ms/token** |
| target-only gen t/s (bench, 2k ctx) | 37.40 |
| inferred short-ctx step cost | 26.7 ms/token |
| matches T1/T2? | yes |

## Correctness gate

| Check | Result |
|---|---|
| `cmp` target-only vs `--mtp` default (chat) | **DIFFER** (documented: fast batch verifier perturbs near-tied logits) |
| `cmp` target-only vs `--mtp` default (code) | **DIFFER** |
| `cmp` target-only vs `--mtp --quality` (chat) | **OK identical** |
| `cmp` target-only vs `--mtp --quality` (code) | **OK identical** |

> **Critical for PLAN.md success criteria:** the primary gate requires
> *exact greedy output preservation*. Only `--quality`/`DS4_MTP_STRICT`
> satisfies it today, and that path is **−28%** vs baseline. Any DSpark path that
> claims the primary gate must use exact verification — so DSpark's win must be
> large enough to overcome the exact-verifier tax, or the verifier must be made
> cheaper without losing exactness.

## Phase 0 conclusion

**Where does `ds4` spend a speculative cycle?** Almost entirely in the verifier.
Draft is ~2 ms (negligible); snapshot/prefix/replay are sub-millisecond
(±0.5 ms across all replay strategies). The verifier dominates: 33 ms/cycle
(default batch) to 67 ms/cycle (exact), against a 28 ms baseline decode step.
This directly confirms the PLAN.md working-hypothesis risk: *"`ds4`'s verifier
and state-management overhead may erase the benefit of longer accepted
prefixes."*

**Does current `--mtp` beat baseline?** Barely, and only by cheating on
exactness. Default depth-2 MTP is **+6% (chat) / +7% (code)** — well under the
20% primary gate. The exact-greedy path (`--quality`) that the gate requires is
a **−28% regression**. So on local Metal, MTP at depth 2 is not a real win under
the success criteria.

**Verifier economics (the Phase 1 input):**

| verify | cost | vs sequential |
|---|---|---|
| 1 token (margin-skip) | 26 ms | 0.93× a decode step (no real saving) |
| 2 tokens (batch) | 40 ms | 0.71× of 2 sequential (saves 16 ms **only on full accept**) |
| 2 tokens (exact) | 67 ms | 1.20× of 2 sequential (**slower than baseline**) |

For DSpark to win locally, a deeper draft block must be verifiable at a cost
that scales **sub-linearly** with suffix length. Today the exact verifier is
**super-linear** (67 ms for 2 ≈ 2.4× a single step, i.e. worse than 2×). The
batch verifier is sub-linear (40 ms for 2 ≈ 1.4×) but not exact. **Making a
batch-style verifier both cheap and exact is the single highest-leverage
engineering problem** for the whole effort.

**Draft quality (the other lever):** default MTP position-2 conditional
acceptance is only **0.41–0.42**, well below DFlash's ~0.63–0.72 (paper Fig 2).
Even with a free verifier, current acceptance would cap gains. A DSpark
parallel backbone + Markov head would need to (a) raise suffix acceptance
substantially and (b) do so in a draft pass that stays ≈ the current 2 ms.

**Decision: PROCEED to Phase 1 — narrowed.** The measurement harness clearly
identifies where `ds4` spends speculative time (the verifier), satisfying the
Phase 0 stop/go gate. But the data re-prioritizes the plan:

1. **Phase 1 (verifier cost curve) is now the make-or-break phase.** The exact
   verifier's super-linear cost is the dominant risk. Quantify `verify(L)` for
   L=1..5 *under the exactness constraint* before any DSpark loader work.
2. **Phase 2 simulator must use exact-verifier costs**, not the fast batch
   costs, because only exact verification satisfies the success criterion.
3. **Phase 3 (loader/format) can proceed in parallel** — it is independent of
   the verifier question — but no runtime integration (Phase 4+) until Phase 1
   shows a sub-linear *exact* verifier curve.
4. A fallback worth keeping open: if exact verification cannot be made cheap,
   reconsider whether the primary gate's exact-greedy requirement can be met by
   a **stitching** strategy (exact on the boundary token only), but that is a
   Phase 5+ question, not Phase 0.

### Deferred decision branches (revisit after Phase 1)

The exactness crux surfaced two strategic off-ramps. **Do not decide now —
capture so they reshape Phase 1's data collection, and revisit once the
Phase 1 verifier curve is in hand.**

**Branch A — find performance headroom in the exact verifier (keep exactness).**
Keep the `PLAN.md` primary gate (exact greedy output) as-is and attack the
exact verifier's cost instead. The cost is *not* uniform: per `01`/T6 the
exact path's ~58 ms verify is dominated by (i) two single-token layer dispatches
per layer instead of one fused batched pass (`metal_graph_encode_decode_layer`
×2 vs `metal_graph_encode_layer_batch` ×1, `ds4.c:21218`/`21261`/`19261`), and
(ii) two full-vocab host readbacks. Those are concrete headroom targets
(bit-stable batched reductions, single output-head pass with both vocab rows,
fewer command-buffer boundaries). This branch makes Phase 1's deliverable
stricter: **break `verify_decode2_exact` down by phase**, not just trace
`verify(L)`.

**Branch B — relax exact-greedy preservation, validate via task benchmarks.**
Drop exact-stream equality as a hard gate; accept the fast batch verifier's
near-tied logit drift and instead prove the speculative path does not measurably
hurt quality on real tasks (HumanEval/MBPP/MT-Bench/Arena-Hard-style evals, à la
the paper's §4 setup). This branch makes Phase 1 *broader*: it must also map the
**batch** verifier's curve (already sub-linear at L=2: ~40 ms ≈ 0.71× of
2×sequential, vs the exact curve's ~58 ms ≈ 1.04×), because that becomes the
load-bearing curve. It also promotes a benchmark-quality methodology from a
Phase 7 nicety to a Phase 2/5 first-class deliverable. Risk to record: the drift
is currently *token-different* (correctness gate: default `--mtp` differs from
target-only on both prompts), so Branch B has to show the drift is
quality-neutral, not merely small.

**Phase 1 should collect data that keeps both branches open** until the curve
is measured: trace `verify(L)` for L=1..5 for **both** the exact and the batch
kernel, plus an internal-phase breakdown of the exact verifier at L=2. Decide
A vs B vs proceed-as-planned once that curve exists.

**Instrumentation gap:** the existing `DS4_MTP_TIMING` + `parse_spec_log.py` are
sufficient for Phase 1. The `DS4_SPEC_STATS` aggregate-counter delta from
`02_gap_and_spec.md` is **not needed yet** — defer it until Phase 4 (draft-only)
needs the acceptance-by-position histogram at deeper draft lengths.
