# Branch B — Two Refinements: Margin-Gated Fallback (B1) vs Rejection Sampling (B2)

**Status:** refinement note for the **selected** Branch B.
Phase 1 (`06_phase1_results.md`) chose Branch B: drop "exact greedy output
preservation" from the primary gate, use the sub-linear batch verifier, and
validate that its logit drift is quality-neutral on real tasks. This note
captures **two refinements of how to realize Branch B** that become relevant
depending on (a) what the quality baseline (`run_quality_baseline.sh`) shows,
and (b) the determinism premise in §1. They are **B1** and **B2**.

This note does **not** decide between them. It states each precisely enough that
the quality-baseline results plus the determinism probe can choose.

## 0. Where these sit relative to the selected Branch B

The current Branch B plan is the *minimal* version: **keep greedy-argmax (temp 0)
on the batched verifier and validate its drift is quality-neutral on tasks**
(this is exactly what `run_quality_baseline.sh` tests: target_only / batch / exact
regimes across 92 eval cases). It makes no protocol change.

B1 and B2 are two refinements that become relevant *if* that minimal version is
insufficient, or as principled alternatives to it:

- **B1 — margin-gated exact fallback.** A quality *safety net* for the minimal
  Branch B. Stay on greedy-argmax, but on near-tie positions (where the batched
  argmax is most likely wrong) fall back to the exact sequential kernel. Recovers
  exactness only where it matters, cheaply. No contract change.
- **B2 — rejection-sampling protocol.** A *contract change* that makes the batch
  drift provably benign by construction. Switch from greedy-argmax to the
  textbook speculative-decoding rejection rule (the paper's own protocol). The
  safety net becomes unnecessary.

Both inherit Branch B's measured verifier economics (Phase 1 §3): the batch
verifier is smoothly sub-linear and context-flat, with a marginal verified-token
cost of ~4–8 ms (≈ a third of a ~26–31 ms decode step). That is the economic
basis both build on.

## 1. A premise that reshapes the choice: temp=0 decode is not deterministic

Branch B's whole rationale is that bit-exact verification is infeasible locally
(Phase 1 §4: 95% of exact cost is the 2× layer dispatches; making it sub-linear
needs a research-grade bit-exact batched layer kernel). But there is a deeper,
under-appreciated point: **even target-only decode at temp=0 is not entirely
deterministic run-to-run.** Sources include non-deterministic Metal reductions,
scheduling-sensitive accumulation order in quantized matmuls, and
command-buffer timing jitter.

This is not a minor footnote. It changes how B1 vs B2 reads:

1. **It questions the premise that "exact greedy output" is a well-defined
   object to preserve.** If the baseline itself is non-deterministic, then the
   Phase 0 correctness-gate row "`--quality` → OK identical" means `--quality`
   reproduces the *same-process realization* (same kernel sequence), **not** that
   either run is reproducible across invocations. That is a weaker property than
   the original gate implied — and it weakens the case for any approach (like B1)
   whose value is "recover exactness," because the reference being recovered is
   itself a moving target.

2. **It makes B1's drift bound δ unmeasurable in isolation.** B1 needs a
   provable bound on the logit drift between the batched and sequential kernels
   so the rule "margin > 2δ ⟹ same argmax" is rigorous. But if the sequential
   path itself jitters by some ε_baseline run-to-run, a measured batch-vs-
   sequential delta is actually δ + ε_baseline, inseparable without a
   deterministic reference — which is the thing in doubt. So non-determinism
   compounds B1's hard part; it does not merely add to it.

3. **It makes B2 strictly easier to validate.** B2 validates *distributionally*
   (output samples are indistinguishable in quality on real tasks), which does
   not require a deterministic reference at all. Baseline jitter is just one
   more source of output variance that a distribution-level test absorbs. So
   the premise that weakens B1 strengthens B2.

**Cheap experiment that feeds the whole choice (run before the quality baseline
settles it):** run the target-only chat prompt twice at temp=0, same seed, and
`cmp`/`diff`. If they differ, baseline non-determinism is confirmed at the token
level. Log under `issue468/baseline/determinism_probe.*`. (If it is already
confirmed for this engine/Metal, skip the probe — the decision logic below
applies directly.)

## 2. B1 — Margin-gated exact fallback

### Idea
Keep greedy-argmax, but don't trust the batched verifier uniformly. The batched
kernel (`metal_graph_verify_suffix_tops`, `ds4.c:21117`) already produces
per-position logits, so it also knows the **margin** between top-1 and top-2 at
each position. Rule:

- margin **large** → argmax is stable under any plausible drift → trust the
  batched result;
- margin **small** (near-tie) → the batched argmax might disagree with the true
  sequential argmax → re-verify **that one position** with the exact sequential
  kernel (`metal_graph_eval_token_raw_swa`, `ds4.c:19461`, ~26 ms/token at this
  context per Phase 1 §2).

Note: the *fused* exact kernel (`metal_graph_verify_decode2_exact`) is **dead**
per Phase 1 §4 (it is just sequential-decode ×2 plus capture overhead; `fused/seq
≈ 1.08`), so B1's fallback target is the plain sequential kernel, not any fused
path.

### What it buys
**Quality safety at sub-linear cost in expectation.** The exact kernel runs only
on near-tie positions, not on all L. Expected verify cost ≈ `batch_cost +
p(near-tie) · L · decode_step`. Phase 0's measured mean margins (~6.97 chat,
~6.64 code — `baseline/parsed_summary.txt`) suggest margins are *usually* large,
which is the favorable case for B1. The plumbing partially exists: the engine
already computes/logs `mean_conf_margin` and already has a margin-skip early-exit
branch in the spec path (`ds4.c:27350`).

This is the right refinement **if the minimal Branch B quality baseline shows
that the batch verifier's failures are concentrated on near-tie positions** —
B1 surgically fixes exactly those.

### The catch (and why the determinism premise hits it hardest)
To claim *true* exactness rather than "usually exact," B1 needs a **provable
bound δ** on the batch-vs-sequential logit drift, so "margin > 2δ ⟹ same argmax"
is rigorous. Two problems:

- On quantized Metal matmuls, deriving δ analytically is research-grade; it is
  not known for this build, and Phase 1 §4 already established that the bit-exact
  batched reductions needed to *reason* about δ are the unsolved hard part.
- Per §1, δ is not cleanly *measurable* either, because the sequential reference
  itself jitters by ε_baseline. You would be bounding δ + ε_baseline.

So B1's guarantee is, at best, empirical-and-conservative: "matches the baseline
realization within measured jitter." Defensible, but weaker than bit-exactness —
and if that is the actual guarantee, B1 has collapsed into "minimal Branch B with
a heuristic safety net," not a distinct exactness regime.

**Net:** highest product value *if* a drift bound works out and the quality
baseline shows near-tie-concentrated failures (keeps greedy-argmax, keeps the
CLI contract, keeps temp 0). Highest risk because its exactness guarantee is the
thing most threatened by the non-determinism premise. Single point of failure:
the δ bound.

## 3. B2 — Rejection-sampling protocol (the paper's own protocol)

### Idea
Replace greedy-argmax verification with the textbook speculative-decoding
rejection-sampling rule (Chen 2023; Leviathan 2023; paper §2.1): accept draft
token `x` (sampled from draft dist `q`) with probability `min(1, p(x)/q(x))`,
else resample from the corrected residual `max(0, p − q)` renormalized. This is
**provably exact in distribution** — it produces samples from the true target
distribution `p`, with no quality loss relative to plain target sampling.

### The most important framing point
**This is the protocol the DSpark paper itself uses.** Paper §4.1: "sampling
temperature set to 1.0." Every result in the paper — the 60–85% gains, Table 1's
accepted lengths, Figure 2's position-wise acceptance — is under stochastic
rejection sampling, *not* greedy-argmax. So Branch B's current "keep greedy-
argmax on the batch verifier" is actually *stricter* than what the paper
demonstrates. Adopting B2 is not "weakening" relative to the paper; it is
aligning with the paper's protocol and with the speculative-decoding field.

### Why it is robust where greedy-argmax isn't
Greedy-argmax is a **step function** of the logits: a 1e-7 drift near a tie flips
the token, and that flip propagates through the entire downstream stream. That is
exactly why the minimal Branch B has to validate the batch drift at all. Rejection
sampling is **smooth** in the logits: the same 1e-7 drift moves an acceptance
probability by ~1e-7 and the output distribution by ~1e-7. The batched verifier's
numerical drift is catastrophic under greedy-argmax but negligible under
rejection sampling. Under B2 the quality-neutrality question largely *dissolves*
— it reduces to "are the batched logits approximately distribution-correct,"
which they are by construction (same math, different reduction order).

### What it buys
It makes the **existing sub-linear batch verifier sufficient without a safety
net.** The Phase 1 §3 marginal economics (~4–8 ms per verified token, ≈ a third
of a decode step) become the load-bearing curve, and prediction accuracy / draft
quality becomes the real lever — exactly the regime where DSpark's draft
improvements (Markov head, higher suffix acceptance) can pay off. Combined with
the §1 premise, the validation target is clean: "output samples are quality-
indistinguishable from plain target sampling on real tasks," which does **not**
require a deterministic reference and is the *standard* way speculative decoding
is validated in the literature.

### The catches
1. **It abandons greedy determinism.** At temp=0, rejection sampling degenerates
   to greedy-argmax (point-mass distribution accepts iff draft == argmax), so B2
   requires temp>0 and stochastic output. For use cases that depend on a
   deterministic greedy stream (reproducible tests, exact `cmp` against baseline),
   that is a capability change. *Note from §1: if baseline temp=0 is already
   non-deterministic, this capability may be partly illusory already — the
   determinism probe quantifies how much B2 actually gives up.*
2. **It needs protocol plumbing** (rejection sampler, residual resampler, draft
   log-prob capture) that the current greedy-argmax path does not have. This is
   real engineering, but it is *bounded* and well-understood, not research-grade
   like the B1 drift bound or a Branch A bit-exact kernel.
3. **Quality validation still applies** (HumanEval/MBPP/MT-Bench/Arena-Hard),
   but a *far more forgiving* one than either bit-identical-token-stream (Branch
   A) or near-tie-argmax-stability (B1): you show samples are indistinguishable
   in quality. This is precisely what `run_quality_baseline.sh` is already set up
   to measure — the *same* harness validates B2 as validates minimal Branch B.

## 4. How the determinism premise and the quality baseline tilt B1 vs B2

| | minimal Branch B (current) | B1 (margin-gated fallback) | B2 (rejection sampling) |
|---|---|---|---|
| Contract | greedy-argmax, temp 0 | greedy-argmax, temp 0 | rejection sampling, temp>0 |
| Batch verifier usable? | As-is, drift validated on tasks | As-is, with exact fallback on near-ties | As-is, drift benign by construction |
| What must be proven | Drift is quality-neutral on tasks | Drift bound δ (hard on Metal) + tasks | Tasks (the standard spec-decode validation) |
| Hit by non-determinism? | Tolerated (pass-rate comparison) | **Hard** — δ becomes δ+ε_baseline | **Helped** — validates distributionally |
| Relaxes vs paper? | Stricter than paper | Stricter than paper | Matches paper's protocol exactly |
| New infra | None (existing harness) | Marginal (margin-skip branch exists) | Protocol plumbing + existing harness |
| Main risk | Drift may hurt quality on some tasks | Exactness guarantee may be unprovable → collapses to minimal B | Loses greedy determinism |

**Decision logic:**

- If `run_quality_baseline.sh` shows **batch ≈ target_only on tasks** → minimal
  Branch B is sufficient; neither B1 nor B2 is needed. Ship as-is.
- If the baseline shows **quality regressions concentrated on near-tie positions**
  → B1 is the surgical fix, *provided* the determinism probe (§1) and a drift-
  vs-margin scatter show δ is boundable. If δ is not boundable, B1 collapses and
  the choice is B2.
- If the baseline shows **distributed quality regressions**, or if the
  determinism probe confirms baseline non-determinism → **B2 is the principled
  choice**: it aligns with the paper, dissolves the drift question, and its
  validation harness already exists.

The blunt read: **B1's value is conditional on a drift bound that the
non-determinism premise makes hard to establish; B2's case is strengthened by
that same premise and aligns with the paper.** B1 is worth carrying because, if
its bound holds, it keeps the current contract and CLI — a real product
advantage. But it has a single point of failure (δ) that, if it doesn't hold,
routes the effort into B2 regardless. B2 has no such single point of failure;
its only cost is the deterministic-greedy capability, which the determinism
premise suggests may already be partly absent.

## 5. What to capture to choose between B1 and B2

1. **Determinism probe (§1).** Two target-only temp=0 runs, same seed, `cmp`.
   Cheap; run before settling B1/B2. If baseline is non-deterministic, the case
   for B2 hardens and B1's δ becomes δ+ε_baseline.
2. **Quality baseline (`run_quality_baseline.sh`).** Already staged. Decides
   whether minimal Branch B is sufficient at all, and whether any regressions are
   near-tie-concentrated (favoring B1) or distributed (favoring B2).
3. **Drift-vs-margin scatter (B1 input).** For a sample of cycles, record both
   the batched-verifier margin *and* the batched-vs-sequential logit delta per
   position. Gives the empirical δ distribution and the fraction of positions
   where margin > 2δ — i.e., how often B1 could trust the batched result without
   falling back. This single number decides B1's viability. (Reuse the
   `DS4_VERIFY_CURVE_BREAKDOWN` probe pattern from Phase 1 §7.)
4. **Acceptance-by-position counter (B2 input).** Promote the deferred counter
   from `02_gap_and_spec.md` (currently Phase 4): under B2, draft quality — not
   exactness — is the lever, so position-wise acceptance is load-bearing for the
   Phase 2 simulator's break-even math.

If (1) shows baseline non-determinism *or* (3) shows δ is hard to bound, B2 is
the clear choice. If (3) shows δ is small and tightly margin-separable *and* (2)
shows near-tie-concentrated regressions, B1 keeps the contract and wins. If (2)
shows minimal Branch B is already quality-neutral, neither refinement is needed.
