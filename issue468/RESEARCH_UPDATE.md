# DSpark Feasibility — Research Update

**Status:** Phase 0 complete (executed) · Phase 1 prepared, paused for go-ahead
**As of:** 2026-06-27
**Scope:** local decode speedup of a DSpark-style speculative path on `ds4`/Metal
**Machine:** Apple M5 Max / 128 GB / Metal · git `c7ef1bf`

## TL;DR

The headline speedup target (≥20% greedy throughput **with exact greedy output
preservation**) is **not yet achievable** on local Metal, and Phase 0 isolated
exactly why: the **exact verifier is the binding constraint**, not drafting, not
state management, not the scheduler. The working hypothesis risk materialized
verbatim. The work is **proceeding — narrowed to the verifier-cost-curve
question** — with two off-ramps (keep-exactness vs relax-exactness) held open
until Phase 1 produces data.

## What Phase 0 measured

A full per-cycle breakdown of target-only and current `--mtp` decode, using the
pre-existing `DS4_MTP_TIMING` surface plus a small parser — **zero engine
change**. Sources: `00`–`04_run_matrix.md`, `baseline/parsed_summary.txt`.

## The four findings that reframe the plan

### 1. The verifier owns the cycle — confirmed, and worse than expected
Where one speculative cycle goes (vs a 28 ms baseline decode step):

| phase | cost |
|---|---|
| draft | ~2 ms (negligible) |
| snapshot / prefix / replay | sub-ms (±0.5 ms across all strategies) |
| **verifier** | **33 ms (fast batch) → 67 ms (exact)** |

The PLAN.md risk — *"verifier and state-management overhead may erase the
benefit"* — is **confirmed for the verifier and disproven for state management**.
Replay/snapshot strategy is a rounding error; the verifier *kernel* is the cost.

### 2. Exact verification is super-linear at L=2 — the binding constraint
| verify mode | cost (L=2) | vs 2× sequential (~56 ms) |
|---|---|---|
| batch, layer-fused (not bit-exact) | ~40 ms | 0.71× (sub-linear) |
| **exact fused N=2** (`--quality`) | **58–67 ms** | **1.04–1.20× (slower than baseline)** |

Worse: **the exact fused kernel exists only for N=2**. For L>2 there is no exact
fused kernel at all — the only exact option is the linear sequential path
(~28 ms × L). So exact verification is *both* super-linear at L=2 *and*
non-existent as a fused kernel beyond it.

### 3. The current `--mtp` path doesn't actually win under the success criteria
| path | chat t/s | vs target | exact greedy? |
|---|---|---|---|
| target-only | 35.60 | — | yes |
| `--mtp` default | 37.76 | **+6%** | **no** (perturbs near-tied logits) |
| `--mtp --quality` exact | 25.65 | **−28%** | yes |

The path that *beats* baseline does so by cheating on exactness; the path that
*satisfies* the gate loses 28%. **This is the crux of the whole effort.**

### 4. Draft quality is also weak — a second, compounding ceiling
Default MTP position-2 conditional acceptance is **0.41–0.42**, vs paper Figure
2's DFlash ≈ **0.63–0.72**. Even a *free* verifier would cap gains at current
acceptance. DSpark's parallel backbone + Markov head must raise suffix
acceptance substantially *and* keep the draft pass ≈ today's 2 ms.

## What this means for DSpark feasibility

- **The architecture is plausible; the verifier economics are not, yet.**
  DSpark's headline gain (60–85% per-user speedup) in the paper is a
  *server-concurrency* result, realized by routing idle batch compute. Locally,
  single-request, that lever does not exist — so the win must come purely from
  `(draft + verify) / accepted_tokens < plain decode`, which is exactly what the
  exact-verifier tax breaks.
- **Exactness, not drafting, is the gate.** The plan was right to gate on exact
  greedy output; Phase 0 showed that gate is currently a 28% tax. Any local win
  under the primary criterion must *first* make exact verification cheap.
- **Two levers, both must move:** (a) make `verify(L)` sub-linear under
  exactness, and (b) raise suffix acceptance toward DFlash/DSpark levels.

## Decision: PROCEED to Phase 1 — narrowed

Phase 0 cleared its stop/go gate (the harness *does* show where time goes).
Decision is **proceed**, with scope narrowed to one make-or-break question and
two deferred branches held open:

- **Branch A (keep exactness):** find headroom in the exact verifier —
  2× single-token layer dispatches, per-layer prefix-1 capture, 2× output-head
  + 2× full-vocab readbacks (`metal_graph_verify_decode2_exact`, `ds4.c:21218`).
- **Branch B (relax exactness):** accept the batch verifier's near-tied logit
  drift and prove it is *quality-neutral* on HumanEval/MBPP/MT-Bench/Arena-Hard.
  Risk to retire: default `--mtp` is currently *token-different*, so Branch B
  must show the drift is harmless, not merely small.
- **Stop local** if even the batch curve is cliffy at low L.

## What happens next (Phase 1, prepared, not yet executed)

A research-only verifier microbench (`ds4_engine_verifier_curve_test`,
mirroring the existing `*_test` family) producing `verify(L)` for L=1..5 across
**three** kernels (batch / exact-fused-N2 / sequential-exact), × {2k/4k/8k
context} × {chat, code}, plus the internal-phase breakdown of the exact
verifier at L=2. Output: a one-paragraph verdict (sub-linear / linear / cliffy)
that selects A / B / stop. Plan and placement in `05_phase1_plan.md`; paused at
"awaiting go-ahead" on four open questions before any engine code is touched.

Phase 3 (DSpark auxiliary GGUF loader/schema) can run in parallel — it is
independent of the verifier question.

## Net feasibility assessment

**Not disproven; materially harder than the paper's numbers suggest for local
single-request use.** The paper's gains are realized server-side by routing
idle batch capacity — a lever that is largely absent locally. On local Metal,
the binding constraint is exact verifier cost, and that constraint is currently
a net 28% loss. The honest read is: **feasibility is contingent on Phase 1
finding exact-verifier headroom (Branch A) or on a quality-neutrality result for
logit drift (Branch B).** If neither holds, the local case does not close and
the effort should refocus server-side only, where the paper's gains actually
live.
