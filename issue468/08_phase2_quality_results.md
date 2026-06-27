# Phase 2a — Quality Baseline: Results (Branch B minimal-version verdict)

Status: **EXECUTED.** Captured the pre-DSpark precision baseline for Branch B,
accounting for Metal temp=0 nondeterminism. Result: **the minimal Branch B
(greedy-argmax on the batch verifier) is NOT quality-neutral — a reproducible
6-case regression. This routes to B2 (rejection sampling).**

Machine: Apple M5 Max / 128 GB / Metal, `--power 100`. Git `4774a2f` (ds4-eval
speculative-path wiring) + `d385d8b` family. Raw logs in
`issue468/baseline/quality/`.

## 1. What was measured

`ds4-eval`'s 92 built-in cases (GPQA Diamond, SuperGPQA, AIME 2025, COMPSEC),
greedy, `--nothink`, `-n 4096`, `--seed 1`. Three verifier regimes:

- **target** — non-MTP, exact decode kernel (ground truth; reference run from
  `reference_nothink_4096.log`, 2026-06-06, + 2 fresh control runs).
- **exact** — `--mtp --mtp-draft 2 --quality`: exact-fused N=2 verifier. **Uses
  the same single-token decode kernel family as target**, so its deviation from
  target IS the Metal nondeterminism noise floor.
- **batch** — `--mtp --mtp-draft 2`: the Branch-B batched verifier
  (`metal_graph_verify_suffix_tops`). Sub-linear/fast, NOT bit-exact.

R=2 runs each for batch and exact (variance control). Target control = 2 fresh
runs to confirm the reference is stable.

## 2. Headline numbers

| config | run | score | reproducible? |
|---|---|---|---|
| target (reference, 2026-06-06) | — | **67/92** | — |
| target | r1 (fresh) | **67/92** | ✓ |
| target | r2 (fresh) | **67/92** | ✓ |
| exact | r1 | **67/92** | ✓ |
| exact | r2 | **67/92** | ✓ |
| batch | r1 | **61/92** | ✓ |
| batch | r2 | **61/92** | ✓ |

Both configs are **perfectly reproducible across runs** (target=67/67/67, exact=67/67,
batch=61/61) — each kernel is internally deterministic (fixed reduction order),
so run-to-run flips are 0. **The difference is cross-kernel, not run-to-run
noise.** The 2 fresh target runs (67/67) also reproduce the 2026-06-06 reference
(67), ruling out model/code drift and confirming the reference is valid.

## 3. The decisive control

**target = 67/92 (reference + 2 fresh runs, all identical), exact = 67/92, batch
= 61/92.** The exact path uses the same single-token decode kernel as

target-only, so if the 6-case drop were Metal nondeterminism, exact would drop
too. It doesn't. The determinism probe further confirms target-only is
bit-reproducible. Therefore:

> **The 6-case regression is attributable to the batch verifier's logit drift,
> not Metal nondeterminism.** Minimal Branch B (greedy-argmax on the batch
> verifier) is **not** quality-neutral on this benchmark.

The per-case diff (`diff_quality.py`) shows the failures concentrate on the
long-chain AIME math cases (e.g. #6 target=26→batch=870, #15 target=82→batch=96,
#21 target=106→batch=374989): the classic greedy-argmax cascade — one near-tie
flip early in a multi-step derivation flips every downstream token, producing a
completely different (wrong) final answer. This is precisely the failure mode
`07_branch_b_options.md` §3 predicted for greedy-argmax: it is a *step function*
of the logits, so a 1e-7 drift near a tie is catastrophic.

## 4. Determinism probe (token-level confirmation)

`run_determinism_probe.sh` — two identical target-only temp=0 `--nothink` runs
of one chat prompt (512 tokens), `cmp`.

**RESULT: IDENTICAL (bit-reproducible).** Both runs produced the same 2017-byte
token stream. Combined with target.r1 = target.r2 = 67/92 (both reproducible at
the answer level), this establishes that **the target-only path is deterministic
on this M5 Max build** for these workloads.

The temp=0 nondeterminism caveat (Metal reduction order) was prudent to set up as
a control, but **empirically it does not manifest here.** Consequence: the 67
reference is a true ground truth, and the batch 61/92 is a **genuine systematic
regression with zero noise confound** — the cleanest possible signal for the
Branch-B decision. (Nondeterminism may still appear on other prompts/contexts;
the result is scoped to these 92 cases + the probe prompt.)

## 5. Branch decision — confirmed: B2

Per `07_branch_b_options.md` §4 decision logic: the minimal Branch B shows a
**distributed, reproducible quality regression** (not near-tie-recoverable, not
within noise). This routes unambiguously to **B2 — rejection sampling**:

- B1 (margin-gated exact fallback) needs a drift bound δ; but δ is
  unmeasurable in isolation because the greedy cascade makes failures
  non-local, and its "exactness" is undermined by baseline nondeterminism.
- B2 makes the batch verifier's drift **benign by construction**: rejection
  sampling is *smooth* in the logits, so the same ~1e-7 drift that flips a
  greedy token moves an acceptance probability by ~1e-7. It is the paper's own
  protocol (paper §4.1 "temperature set to 1.0"; `generation_config.json`
  confirms `do_sample: true, temperature: 1.0`).

## 6. Corroborating finding from the official DSpark inference code

The shipped `inference/generate.py` is a **plain autoregressive baseline** — it
only calls `model.forward()` and **never calls `forward_spec`** (the drafter).
The drafter is defined and exercised in `model.py`'s `__main__`, but its draft
output is discarded. **The open-source DSpark repo does not include the
verification/rejection-sampling loop** — that is left to the integrator. This is
exactly the B2 gap ds4 must fill, and it confirms the paper's results are under
stochastic rejection sampling, not greedy-argmax.

## 7. Caveats on this baseline

- **92 cases is a small, mixed set** (not a pure HumanEval/MBPP/MT-Bench pass).
  The 6-case signal is strong because it is fully reproducible, but a larger
  task benchmark remains the Phase-2 deliverable for B2 once the DSpark drafter
  is integrated.
- **The score is letter/number-answer extraction**, which is the right metric
  for Branch B (a synonym flip that still picks the right answer is
  quality-neutral). The greedy cascade hurts this metric maximally.
- **exact ≠ target bit-for-bit in general** (both are temp=0 on Metal), but
  both landed on 67/92 here. The 2 fresh target control runs (pending) quantify
  target's own run-to-run variance.
