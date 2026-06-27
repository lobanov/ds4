# DSpark Feasibility — Research Update

**Status:** Phase 0 + Phase 1 complete (executed) · **Branch B selected** → Phase 2
**As of:** 2026-06-27
**Scope:** local decode speedup of a DSpark-style speculative path on `ds4`/Metal
**Machine:** Apple M5 Max / 128 GB / Metal · git `d385d8b` (Phase 1 microbench)

## TL;DR

Phase 0 isolated the binding constraint (exact verifier cost) and Phase 1
resolved the branch decision: **only the batch verifier is sub-linear; every
exact path is linear-or-worse, and exact-fused N=2 is slower than plain
sequential.** The exact-verifier breakdown disproved the Branch A headroom
hypothesis (95% of exact cost is the 2× single-token layer dispatches; readback
overhead is ~2.5 ms / 5%). **Decision: Branch B** — drop exact-greedy
preservation from the primary gate, use the sub-linear context-flat batch
`verify(L)` curve as the simulator input, and validate the non-exact logit drift
is quality-neutral on real-world benchmarks. Branch A is deferred (would need
new bit-exact batched layer kernels). Full data: `06_phase1_results.md`.

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

## Decision: Phase 1 executed — Branch B selected

Phase 1 ran the verifier microbench (`ds4_engine_verifier_curve_test`, behind
`--verifier-curve-test`) for L=1..8 on all three kernels at ctx {2k,4k,8k},
chat + code. Verdict (`06_phase1_results.md`):

- **Batch verifier is strongly sub-linear AND context-flat** — per-token
  amortized 36 ms (L=1) → 12 ms (L=8); batch(L) nearly identical at 2k/4k/8k.
  Smooth, not cliffy. Batch beats sequential at every L≥2.
- **Every exact path is linear-or-worse.** Sequential ≈ 26–31 ms × L. The
  exact-fused N=2 kernel (56–66 ms) is *slower than plain sequential*
  (1.07–1.09×) — it dispatches two single-token decode-layer passes with
  capture overhead and **zero** actual fusion.
- **Branch A disproven by the breakdown:** layers = 53 ms @2k/4k, 64 ms @8k
  (95% of cost); output-head + readbacks ≈ 2.5 ms (5%). The hypothesized
  readback headroom is negligible; the cost is the 2× layer dispatches, which
  need new *bit-exact batched* kernels (the existing batch verifier is batched
  but not bit-exact) — a research kernel project. **Branch A deferred.**

**→ Phase 2 proceeds on Branch B:** the batch `verify(L)` curve is the
simulator input; benchmark-quality methodology (HumanEval/MBPP/MT-Bench/
Arena-Hard) is now load-bearing to prove the batch drift is quality-neutral.
Exact-greedy preservation is dropped from the primary gate. Scheduler rule
(grounded in data): verify at L=0 (plain decode) or L∈[2,γ]; **never L=1 via
batch** (batch L=1 is slower than sequential).

Phase 3 (DSpark auxiliary GGUF loader/schema) can run in parallel — it is
independent of the verifier question.

## Net feasibility assessment

**Not disproven; the local win now hinges entirely on Branch B's quality
neutrality.** Phase 1 established that the batch verifier's economics are good
(sub-linear, context-flat) but it is *not* bit-exact, and every exact path is a
loss. So the local case closes **only if** the batch verifier's logit drift is
shown quality-neutral on real tasks (Phase 2). The paper's headline gains
remain a server-concurrency result (routing idle batch capacity), largely absent
locally. If Branch B's validation fails, the fallback is Branch A (new bit-exact
batched layer kernels — uncertain) or refocusing server-side only, where the
paper's gains actually live.
