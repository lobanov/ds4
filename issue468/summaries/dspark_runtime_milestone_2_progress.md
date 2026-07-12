# DSpark runtime milestone 2 — model-on-Metal validation brief

Date: 2026-07-12. Status: **active / in progress**.

## Commander's intent

Milestone 2 validates or falsifies, on the Metal implementation, the
load-bearing assumptions of `issue468/summaries/spec_speedup_model.md` — using
the best-known modelled optimizations (anchor reuse, confidence-based
scheduling, batched drafting, batched verifying, and the rest) — **before
committing to other research leads.**

The point is not to make DSpark fast on CPU, and not to ship a product path.
The point is to decide, lever by lever, whether the speedup model's projected
gains **survive contact with the real runtime**, or whether the realistic
on-Metal band sits below the model's optimistic edge. The output of this
milestone is a per-lever verdict (`validated` / `partial` / `falsified` /
`not yet attempted`) plus a decision: which modelled levers are worth
committing further research to, and which should be closed.

The model's headline is conditional and stacked. It projects that DSpark can
beat baseline **only** by combining several assumed-but-unverified gains
(anchor-reusing verifier economics, a materially better drafter, and
conditional scheduling). Milestone 2's job is to replace each of those
assumptions with a measured Metal reality, then re-read the model with the
measured numbers.

## Scope: the modelled levers under validation

Each row is one assumption the speedup model depends on. "Model projection" is
what the model currently credits that lever; "Metal status" is the milestone's
current verdict on it. The levers are listed roughly in the order the model
ranks their leverage, not in execution order.

| lever | what the model assumes | model projection | Metal status |
|---|---|---|---|
| **Batched verify** | `verify_ms(K)` is one batched target forward over K drafts, sublinear in K (K2:43.6, K4:65.8 ms) | makes the optimized-verifier cost term real | **not yet attempted** — DSpark currently verifies K drafts as K sequential target decodes. Two batched primitives exist in-tree: `metal_graph_verify_suffix_tops` (prefill batch kernels; sublinear but **not greedy-exact**) and `metal_graph_verify_decode2_exact` (exact decode kernels, N=2; greedy-exact but **linear, no amortization**). **No existing primitive is both greedy-exact and sublinear.** Secondary: DSpark needs per-position layer-40/41/42 hidden (single-slot capture today). Next slice is a flip-rate probe, not integration. Current pivot. |
| **Anchor reuse** (optimized-verifier regime) | fresh anchor decode only on full-block accept; the verify forward yields the next anchor | removes the redundant per-cycle anchor decode (~−18 pp at K=4 vs shipped) | **partial** — Lead 06 made reuse exact and correctness-gated, but the surviving path is an exact sequential substrate, not the cheap folded verifier the model assumes. Worth +21% over shipped K=4 but still −16% vs baseline. Cheap reuse still open. |
| **Confidence-based scheduling** | a confidence head chooses verify span cycle-by-cycle, truncating low-confidence tails | conditional secondary, ~+3–5% under anchor reuse, ~0 under shipped accounting | **partial** — Lead 02 measured it: marginal under anchor reuse (frozen threshold ~1.04×, expected-opt ~1.05×), useless under shipped economics. Calibrated and decision-relevant, but below its predeclared stacking tier. |
| **Batched drafting** | the drafter emits a short block in one batched pass instead of token-serially | removes token-serial host draft overhead | **partial** — milestone 1 cycles 1–4: prefix-limited batched drafting at cap3 is a marginal local win over serial on the short sample, cap4 is the conservative pick, cap5 is a miss. Does not change acceptance; only draft-side cost. |
| **GPU drafter body/head** | `draft_ms ≈ 10` (the model's per-cycle draft cost) | the modeled speedup numbers assume a ~10 ms drafter forward, not the current ~20–27 ms CPU cost | **not yet attempted** — one GPU output-head cut line was tried and failed `--dspark-schedule-parity` on Metal (CPU Q8_0 vs GPU dequant argmax divergence). Not benchmarked. Whether any GPU cut line reaches parity is open. |
| **Target hidden-state precision** (Lead 04) | native FP hiddens lift acceptance vs IQ2XXS | +8–10% cycle-jump speedup at float32, the largest single lever | **falsified for current deployment / open as research** — float32 ceiling is real but F16-deployment-blocked (argmax-flipping); IQ2XXS recoverability unproven; crossed-oracle pending. This is upstream acceptance work, not runtime work, but it gates whether the runtime levers above can ever clear baseline. |

The milestone does **not** own drafter-training or hidden-precision research
(Lead 04 / Lead 07). It owns the runtime question: given whatever drafter and
hiddens we have, do the model's *runtime* assumptions hold on Metal?

## What "validate an assumption" means here

A modelled lever is `validated` only when all of the following are true on the
retained corpora:

1. the Metal implementation of that lever exists and is env-gated (default off
   until promoted);
2. it preserves the retained correctness gates (greedy exactness and the
   temp>0 logit/distribution gate);
3. it preserves the lever's *own* parity contract (draft ids, confidence
   logits, scheduled `verify_n`, accepted chunking, window-state, **and** —
   for verify-path changes — per-position argmaxes / correction tokens against
   the trusted reference);
4. it is benchmarked with full cycle-cost attribution; and
5. the measured economics are compared against the model's projection for that
   lever, and the gap is recorded.

A lever that preserves outputs but degrades acceptance, or that benchmarks
faster than the reference yet slower than the model projected, is `partial`,
not `validated`. A lever whose Metal cost is worse than the model assumed is
`falsified` for the speedup claim even if it is correctness-safe.

## Measurement methodology

Milestone 2 treats runtime work as a **two-axis benchmark problem**:
correctness of the final generated stream, and preservation of speculative
economics. The second axis matters because a runtime optimization can preserve
final tokens and sampled distributions while still degrading draft quality,
accepted-prefix length, or full-accept frequency enough to erase any speed
benefit.

The retained benchmarking contract has four layers.

1. **Final-output correctness gates remain mandatory.**
   Greedy output must remain byte-identical to baseline, and temp>0 parity must
   hold at the target logits / sampling-distribution level rather than only at
   emitted tokens.
2. **Draft-side quality must be checked separately from final-output parity.**
   Verifier exactness can hide a degraded drafter. For DSpark, runtime-valid
   optimizations therefore need draft-token parity, confidence-logit parity,
   scheduled `verify_n` parity, accepted-chunk parity, and DSpark window-state
   parity against the trusted reference, not just final text/logit agreement.
   Verify-path changes additionally need per-position argmax and correction-
   token parity.
3. **Acceptance metrics are first-class outputs.**
   Each retained bench reports at least drafted length, verify length,
   accepted/verified drafts per cycle, and where relevant the full-accept rate.
4. **Cycle-cost attribution must stay split by component.**
   Aggregate tokens/s is not enough to identify the next lever. The retained
   profile keeps `decode_ms`, `draft_ms`, `verify_ms`, `verify_decode_ms`,
   DSpark state-push time, and logits readback separate so verifier cost,
   drafter cost, and state-update cost do not get conflated. For batched-verify
   work the profile must distinguish *batched* verify cost from *sequential*
   verify cost on the same cycle shape.

Operational rules carried into this milestone:

- Use the same retained exactness, powered-corpus, and long-context corpora
  that already feed `issue468/summaries/spec_speedup_model.md`.
- Use `ds4_spec_bench.c` / `ds4-spec-bench` as the default DSpark benchmark
  substrate, extending it as needed rather than replacing it with many narrow
  drivers.
- Preserve batch submission from one bulk config, low-file-count output, and
  reuse of a single loaded engine while respecting the DS4 lock and using fresh
  sessions where needed.
- Do not validate an optimization by greedy exactness plus temp>0 parity alone
  if it changes draft acceptance, confidence values, scheduled verify lengths,
  or verify-path argmaxes.
- Always compare both a fixed low-K reference (`verify_k=1`) and the scheduled
  path, because the scheduled path can lose for two different reasons: verifier
  cost or drafter execution cost.

## Design constraints

Any milestone-2 change should preserve the following contract.

1. **Each lever is implemented env-gated, default off, behind its own parity
   gate.** No lever is promoted into the default runtime until it clears its
   full validation checklist.
2. **Scheduled-verification semantics stay authoritative where the model
   assumes them.** A migrated path must preserve draft ids, confidence logits,
   computed `verify_n`, accepted chunking, and final outputs against the
   trusted reference.
3. **The "confident prefix + 1 drafted token" policy remains intentional.**
   The point of scheduled DSpark is not to maximize full accepts; it is to
   draft one token past the confident prefix so the next cycle is less likely
   to pay a standalone anchor decode.
4. **Model assumptions are tested, not assumed.** When an optimization
   realizes a model assumption (e.g. batched verify realizes `verify_ms(K)`),
   the measured cost replaces the model's value for that term and the speedup
   projection is re-read with the measured number.
5. **The benchmark substrate and corpora stay fixed while internals move.**
   `ds4-spec-bench`, the retained corpora, and the parity gates remain the
   default validation harness across all levers.

## Carried-forward facts

These findings remain load-bearing.

1. The current DSpark runtime is correctness-gated and benchmarkable. Greedy
   exactness and the retained temp>0 distribution gate pass after the
   support-state and `main_proj` semantic fixes from milestone 1.
2. The verifier, not the drafter, is the binding runtime ceiling. With correct
   support-state advancement, setting draft cost to zero on the retained short
   sample still lands around **~37–38 t/s**, under the **~40 t/s** baseline.
   Drafter-side speedups alone cannot clear baseline on top of the current
   sequential verifier.
3. The current DSpark verify loop is **sequential**: one
   `metal_graph_eval_token_raw_swa_top` per draft, breaking on first mismatch.
   Two batched verify primitives exist in-tree (`verify_suffix_tops`, sublinear
   but not greedy-exact; `decode2_exact`, exact but linear), but **neither is
   both greedy-exact and sublinear** — see fact 6 and the 2026-07-12
   batched-verify scoping worklog. This is the largest unvalidated model
   assumption and the current pivot.
4. Fixed `verify_k=1` remains the cleanest local DSpark reference point, but it
   is still not a positive DSpark speedup claim.
5. The model projects that batched verify **plus** the current IQ2XXS drafter
   reaches only ~0.98×; clearing baseline requires batched verify **plus** a
   materially better drafter. The runtime milestone can validate or falsify the
   first half; it cannot deliver the second.
6. **The batched-verify lever is NOT a simple integration of an existing
   primitive.** Two batched verify primitives already exist in-tree, but neither
   is drop-in for DSpark:
   - `metal_graph_verify_suffix_tops` — one batched forward over K drafts via
     the **prefill batch kernels**. Fast, but the code comment at its sibling
     `metal_graph_verify_decode2_exact` states these kernels "are not a safe
     substitute for autoregressive decode: small row-wise differences… are
     enough to flip future greedy tokens." Used by MTP (with replay-on-partial).
     **Not greedy-exact**; would break DSpark's greedy gate.
   - `metal_graph_verify_decode2_exact` — N=2 only, encodes both tokens with
     the **exact decode kernels** interleaved per-layer in one command stream.
     Greedy-exact by construction (same kernels as sequential decode). Wired
     into MTP under `--quality`/`DS4_MTP_STRICT`; the MTP team labels it "not a
     speed win" for MTP.
   The binding sub-problem is **per-position hidden capture**: the DSpark
   drafter attends over a per-position KV window (`dspark_win_kv`, all
   `dspark_n_real` positions), so batched verify must retain the layer-40/41/42
   hidden at **every accepted position**, not just the boundary. The existing
   decode-layer capture (`dspark_capture_hc`, written at layers 40/41/42 during
   `metal_graph_encode_decode_layer`) is a **single-slot** buffer overwritten
   per token, so a batched verify only leaves the *last* row's hidden. MTP does
   not hit this because its anchor reuse needs only the boundary hidden
   (`metal_graph_copy_batch_hc_row`); DSpark needs per-position. So a greedy-
   exact batched DSpark verify requires **new per-row capture infrastructure**
   (retain all K rows' layer-40/41/42 hidden during the batched pass), not just
   routing through an existing primitive.
   The sharper blocker (reviewer-corrected): `decode2_exact` is **linear cost**
   (~2× decode, no amortization), so even with per-row capture it does not
   realize the sublinear `verify_ms(K)`; and `verify_suffix_tops` is sublinear
   but flips greedy tokens. **No existing primitive is both greedy-exact and
   sublinear.** The decisive next step is a measurement, not an integration:
   the greedy-token flip rate of `verify_suffix_tops` vs sequential decode on
   the DSpark acceptance positions. Flip rate ≈ 0 → the sublinear primitive is
   usable under the greedy gate (then per-row capture matters). Flip rate > 0 →
   the lever needs Lead 08 fused-kernel work (a hand-written decode microbatch
   preserving decode numerics across K rows) or is falsified for greedy-exact
   DSpark.

## Iteration protocol

Each optimization cycle follows the same order.

1. Implement one lever-relevant, parity-critical change behind an env gate.
2. Run smoke gates first: the lever's own parity gate (e.g.
   `--dspark-schedule-parity`, or a new `--dspark-verify-parity` for verify-path
   work), greedy exactness where applicable, and the temp>0 logit/distribution
   gate.
3. If smoke passes, commission adversarial review before benchmarking.
4. Benchmark with `ds4-spec-bench` on the retained corpora, with full
   cycle-cost attribution split by component.
5. Commission a second adversarial review on the measured result before
   summarizing it into the worklog and downstream summaries.
6. Update the lever-status table in this document with the verdict
   (`validated` / `partial` / `falsified`), the measured economics, and the gap
   to the model's projection.
7. Update `issue468/inventories/ds4_env_and_instrumentation_inventory.md` if the
   iteration added, removed, or changed any instrumentation or env-gated
   runtime variation.
8. Re-read `issue468/summaries/spec_speedup_model.md` with any measured term
   that replaces a modelled assumption, and record whether the headline moves.

## Retained artifacts

- greedy exactness: `issue468/artifacts/dspark_exactness_compare/summary.json`
- temp>0 logit/distribution parity:
  `issue468/artifacts/dspark_temp_distribution_compare/summary.json`
- short powered-corpus sample:
  `issue468/artifacts/dspark_corpus_bench/summary.json`
- long-prompt cycle profile:
  `issue468/artifacts/dspark_phaseA_profile/summary.json`
- scheduled batching cap sweep:
  `issue468/artifacts/dspark_corpus_bench/spec_sched_batchcap_v3_summary.json`
- adversarial optimization review:
  `issue468/artifacts/dspark_codex_reviews/2026-07-12_gpt55_xhigh_dspark_opt_review.md`
- retained batch-cap reviews:
  `issue468/artifacts/dspark_codex_reviews/2026-07-12_gpt55_xhigh_batchcap_prebench_review_v2.md`
  and
  `issue468/artifacts/dspark_codex_reviews/2026-07-12_gpt55_xhigh_batchcap_postbench_review.md`
- reusable bulk speculative bench binary:
  `ds4-spec-bench` from `ds4_spec_bench.c`

## Current lever status

| lever | status | measured (Metal) | model projection | gap / note |
|---|---|---|---|---|
| Batched verify | **measured (exactness-acceptable)** | argmax flip 0.64% (1/156); TV median 0.0035 mean 0.010 max 0.10; KL median 5e-5 | `verify_ms(K)` sublinear (K4:65.8 ms) vs current K sequential decodes | Sublinear `verify_suffix_tops` distribution is close to sequential (meas. 1 PASS). Usable for rejection sampling under the relaxed gate. Remaining ceiling is the drafter (~0.98× IQ2XXS), not the verifier. Per-position-hidden capture still needed to commit. |
| Anchor reuse | partial | +21% over shipped K=4, −16% vs baseline | removes ~−18 pp redundant anchor decode | exact sequential substrate only; cheap folded verifier open |
| Confidence scheduling | partial | ~1.04× frozen / ~1.05× expected-opt under anchor reuse | +3–5% conditional secondary | below predeclared tier; useless under shipped accounting |
| Batched drafting | partial | cap3 marginally beats serial on short sample | removes token-serial draft overhead | mixed; cap4 conservative, cap5 miss; acceptance unchanged |
| GPU drafter body/head | not yet attempted | n/a (GPU-head failed parity) | `draft_ms ≈ 10` | cut line open; first attempt diverged on argmax |
| Target hidden-state precision | falsified-for-deployment / open-research | +8–10% float32, F16-blocked | largest single lever | upstream acceptance work (Lead 04), not runtime |

## Next steps

This section is the current workload and recommendation snapshot. Update it
after each iteration.

1. **Batched verify (current pivot) — measurement-first after scoping** (see
   worklog 2026-07-12 batched-verify scoping). Code-reading found **no existing
   primitive is both greedy-exact and sublinear**: `verify_suffix_tops` is
   sublinear (`verify_ms(K)`) but flips greedy tokens; `decode2_exact` is exact
   but linear (~2× decode, no amortization — a correctness oracle, not a
   speedup). So the next slice is **not** an integration — it is a measurement:
   the greedy-token flip rate of `verify_suffix_tops` vs sequential decode on
   the DSpark acceptance positions. Flip rate ≈ 0 → the sublinear primitive is
   usable under the greedy gate, lever is alive (then solve per-position-hidden
   capture). Flip rate > 0 → lever needs Lead 08 fused-kernel work or is
   falsified for greedy-exact DSpark. A genuinely sublinear AND exact verifier
   is Lead 08 Phase B territory (hand-written decode microbatch preserving
   decode numerics across K rows), not an integration of current kernels.
2. **Re-read the model with the measured verify term.** If batched verify
   validates `verify_ms(K)`, the speedup projection's verify term becomes
   measured currency; the remaining gap to baseline is then cleanly isolable
   to drafter acceptance.
3. **Keep the CPU path as the semantic oracle.** Avoid broader CPU-local
   tuning unless the same structural change would still be wanted after the
   verifier and drafter move.
4. **Defer GPU drafter body/head work** until the verifier term is measured.
   The retained review already judges drafter-first "likely-wrong" as the next
   lever because the sequential verifier caps even a zero-draft path below
   baseline.
5. **Close or promote each lever explicitly.** After batched verify, decide
   whether confidence scheduling and batched drafting are worth re-measuring on
   top of the new verify economics, or whether they stay at `partial`.

## Worklog

### 2026-07-12 — milestone 2 reframed as a model-on-Metal validation brief

Rewrote this note around the commander's intent: validate or falsify, lever by
lever, whether the assumptions in `summaries/spec_speedup_model.md` survive
contact with the Metal runtime, before committing to other research leads. The
earlier "GPU drafter migration" framing is retained as one lever (GPU drafter
body/head) in the lever table, but it is no longer the milestone's primary
thrust — the carried-forward facts and the retained adversarial review both
identify the sequential verifier, not the drafter, as the binding ceiling. The
next iteration is the batched-verify pivot. The CPU implementation keeps its
role as correctness oracle, policy prototype, and benchmark reference.

### 2026-07-12 — first migration attempt: scheduled GPU-assisted output head failed parity

Tested an opt-in scheduled-drafter experiment that offloads the base output-head
projection from the CPU path under `DS4_DSPARK_SCHEDULE_GPU_HEAD=1`, while
retaining CPU-side confidence, Markov bias, and scheduling logic. The temp>0
logit/distribution parity gate still passed, but real `--dspark-schedule-parity`
on Metal failed: draft ids, confidence logits, computed `verify_n`, accepted
chunking, and final speculative progression diverged from the CPU reference.
This iteration is therefore not benchmarkable and does not yet justify review or
downstream model updates. Recorded as the GPU-drafter lever at status
`not yet attempted` (cut line open, first attempt diverged on argmax).

### 2026-07-12 — batched-verify scoping: no existing primitive is both exact and sublinear

Code-reading to scope the batched-verify pivot found two in-tree primitives and
a sharper obstacle than the brief's first cut assumed.

- `metal_graph_verify_suffix_tops` — one batched forward over K drafts via the
  **prefill batch kernels**. This is the only primitive that realizes the
  model's sublinear `verify_ms(K)` (batched expert selection / matmuls). But it
  is **not greedy-exact**: the sibling `metal_graph_verify_decode2_exact`
  comment states the batch kernels "are not a safe substitute for autoregressive
  decode: small row-wise differences… are enough to flip future greedy tokens."
  MTP only uses it in non-strict mode; strict / `--quality` mode falls back to
  `decode2_exact`.
- `metal_graph_verify_decode2_exact` — N=2, exact decode kernels interleaved
  per-layer in one command stream. Greedy-exact by construction, but it **runs
  two full decodes** (per-token layer work at full cost). Reviewer correction:
  this is **linear cost, not amortized** — wall time ≈ 2 × decode_ms, so it
  cannot realize the sublinear `verify_ms(K)` and is "not a speed win" (the
  MTP code comment's own verdict). It is a correctness oracle, not a
  batched-verify economics path, and should not be proposed as the viable
  primitive for this lever.

Net: **no existing primitive is both greedy-exact and sublinear.** The model's
`verify_ms(K)` sublinearity and the greedy-exactness gate (primary success
criterion) are in tension with the current kernels. A genuinely sublinear AND
exact verifier would need Lead 08 Phase B fused-kernel work (a hand-written
decode microbatch that preserves decode-kernel numerics across K rows).

Secondary sub-problem (real, but only matters once a usable verifier exists):
DSpark's drafter attends over the full per-position `dspark_win_kv` window, so a
batched verify must retain layer-40/41/42 hidden at **every accepted position**,
not just the boundary. The existing `dspark_capture_hc` is single-slot
(overwritten per token); MTP avoids this because its anchor reuse needs only the
boundary hidden. Per-row capture is a bounded extension of the current pattern
but is premature to build before the verifier question is settled.

Decisive cheap measurement before committing to anything: **the greedy-token
flip rate of `verify_suffix_tops` vs sequential decode on the DSpark acceptance
positions.** Flip rate ≈ 0 on the retained corpus → the sublinear primitive is
usable under the greedy gate and the lever is alive as an integration (plus
per-position-hidden capture). Flip rate > 0 → the lever needs Lead 08 kernel
work or is falsified for greedy-exact DSpark. This re-prioritizes the next
slice as a flip-rate probe, not an integration.

### 2026-07-12 — option A accepted; measurement 2 (RS acceptance) done: acceptance is neutral, not a windfall

Decision: pursue option A (anoint the batched verifier as the reference target;
switch DSpark verify from greedy-argmax to rejection sampling; amend the
primary gate from "exact greedy output" to "distribution-exact at temp>0 w.r.t.
the chosen reference + ds4-eval quality gate," with Lead 08 later reclaiming
token-exact). Measurement-first, per the protocol.

**Measurement 2 (rejection-sampling acceptance projection) — DONE.** Question:
does switching verification to rejection sampling raise E[a|K] over greedy-argmax
(the acceptance axis of option A)? Projected offline from the retained q4tap
bundles (target top-128 logprobs = p; drafter distribution q = softmax(base_logits
+ markov_bias) re-derived from retained hiddens via the oracle).

Harness: `dspark_oracle/measure_acceptance_bundle.py` extended with
`--emit-draft-dist` (writes per-position draft distribution sidecar) + new
`dspark_oracle/measure_rejection_acceptance.py` + one-load driver
`run_rejection_acceptance_all.py`. Environment: `issue468/.venv` +
`issue468/requirements.txt`. Artifact:
`issue468/artifacts/rejection_acceptance/q4tap_t0p0_summary.json`.

Result (pooled, 10 bundles, 80 blocks, 400 positions; p_mass=1.000, q_mass≈0.997
so the TV bound is essentially exact):

| block pos | greedy | rs (1-TV, sampled) | rs (min(1,p/q), greedy draft) |
|---:|---:|---:|---:|
| 0 | 0.787 | 0.804 | 0.819 |
| 1 | 0.688 | 0.684 | 0.709 |
| 2 | 0.525 | 0.537 | 0.551 |
| 3 | 0.412 | 0.356 | 0.409 |
| 4 | 0.287 | 0.245 | 0.276 |

| metric | E[a\|K=5] | Δ vs greedy |
|---|---:|---:|
| greedy-argmax | 2.4375 | — |
| rs, greedy draft | 2.5002 | **+0.0627 (+2.6%)** |
| rs, sampled draft | 2.2696 | **−0.1679 (−6.9%)** |

**Verdict: rejection sampling does NOT materially raise DSpark acceptance.** The
only positive variant (drafter keeps its greedy argmax, verify by min(1,p/q))
gains +0.063 E[a|K] (+2.6%) — right at the model's ~0.06–0.10 break-even
band, and on the small/hard exactness corpus, so likely within noise. The
sampled-draft variant is actively worse (−6.9%): the DSpark drafter's
distribution is meaningful mainly at the argmax and diverges at later block
positions (TV grows to ~0.65–0.76 by positions 3–4), so sampling from q
*lowers* acceptance.

**Implication for option A:** the value of switching to rejection sampling is
NOT an acceptance windfall — it is purely the **verifier-tolerance** axis
(unlocking the sublinear `verify_suffix_tops` by relaxing greedy-exactness to
distribution-exactness). Acceptance is roughly preserved (greedy-draft RS) or
worse (sampled RS). So option A = sublinear verifier at ≈greedy acceptance,
i.e. the model's ~0.98× (IQ2XXS) projection, still below baseline without a
better drafter. The drafter should keep proposing its argmax (greedy-draft RS);
sampling is a dead end. The decisive remaining question is measurement 1:
**does the batched verifier preserve the distribution closely enough that
rejection sampling against it is exactness-acceptable** (small TV vs sequential
decode)? If yes, the sublinear verifier is usable and option A clears the
verifier-exactness bar at neutral acceptance. Caveats: corpus is the small
(80-block) exactness set, on the hard side; per-position p1=0.787 matches the
dossier IQ2XXS p1≈0.79, confirming the pipeline is consistent with retained
acceptance data.

### 2026-07-12 — measurement 1 (batched-vs-sequential verifier distance) done: sublinear verifier is exactness-acceptable

**Question:** is the sublinear batched verifier (`metal_graph_verify_suffix_tops`,
prefill batch kernels) exactness-acceptable for rejection sampling — i.e. how
close is its per-position distribution to sequential decode? If close (small TV /
KL), rejection sampling against the batched verifier is "distribution-exact
within the kernel numerical floor" and option A clears the verifier-exactness
axis.

**Probe (built on ds4_spec_bench.c):** env `DS4_DSPARK_VERIFY_DIST_PROBE=1`
runs `verify_suffix_tops` non-committing (frontier snapshot → push drafts →
batched verify capturing full per-position logits → pop → restore) alongside
the real sequential DSpark verify (capturing per-position logits in-loop), then
computes per-position argmax-flip / max-abs-logit / TV / KL(seq‖batched) and
stores them in `dspark_last_cycle.verify_dist`, emitted per-cycle as
`verify_dist` objects in the ds4-spec-bench JSONL. Required a graph-alloc change:
`spec_logits` and the spec-frontier tensors were MTP-gated; now also allocated
when DSpark is enabled (ds4.c). Default runtime path unchanged (probe is
env-gated and non-committing).

**Result (10 exactness prompts, 90 cycles, 156 positions compared):**

| metric | mean | median | p90 | max |
|---|---:|---:|---:|---:|
| argmax flip rate | 0.64% (1/156) | — | — | — |
| TV(batched, seq) | 0.0104 | 0.0035 | 0.0308 | 0.104 |
| KL(seq‖batched) | 0.0022 | 5e-5 | — | 0.054 |
| max-abs logit | 0.61 | 0.28 | — | 4.56 |

3.3% of cycles (3/90) had TV > 0.05.

**Verdict: the sublinear batched verifier IS exactness-acceptable for rejection
sampling.** Its per-position distribution is very close to sequential decode
(median TV 0.0035 ≈ 99.65% overlapping mass; argmax agrees 99.36% of the time).
Rejection sampling against the batched verifier produces a distribution within a
small statistical distance of sequential-target — "distribution-exact within the
kernel numerical floor," which is the standard option A relaxed gate proposes.
The rare argmax flips (0.64%) confirm the earlier finding that batched reductions
*can* flip greedy tokens, but the rate is low and the underlying distributions
stay close, so rejection sampling (continuous in the logits) degrades gracefully
where greedy-argmax would flip. Artifact:
`issue468/artifacts/rejection_acceptance/verify_dist_probe_exactness.jsonl`.

**Option A verdict (both axes measured):**
- verifier-exactness axis (meas. 1): **PASS** — sublinear verifier usable, small
  distribution distance (median TV 0.0035).
- acceptance axis (meas. 2): **NEUTRAL** — RS preserves acceptance (greedy-draft
  +2.6%; sampled −6.9% → use greedy-draft).

Net: option A unlocks the sublinear verifier (the binding runtime lever) at
neutral acceptance and small distribution distance. The remaining baseline
ceiling is no longer the verifier or exactness — it is the drafter (model's
~0.98× on IQ2XXS), which is Lead 07 (drafter training) / Lead 04 (hidden
precision) territory, not runtime. Caveats: corpus is the small exactness set
(156 positions); the powered corpus would tighten the stats; the probe compares
only accepted positions (verified≥1), not first-mismatch positions; 3.3% of
cycles show larger divergence (TV>0.05), the tail risk for the relaxed gate.
Awaiting reassessment.

### 2026-07-12 — batched-vs-sequential verify COST measurement (crossover) + runtime-vs-oracle acceptance gap

**Question (validate the verifier modelling assumption):** at what acceptance does
the sublinear batched verifier (`verify_suffix_tops`) actually beat the
short-circuiting sequential verify, and is the oracle `E[a|K]` above that
crossover? Extended the dist-probe to also time the batched verify
(`verify_dist.batched_verify_ms`) alongside the clean sequential decode cost
(`verify_decode_ms`, which times only the real decodes — note the probe runs the
batched verify inside the `verify_ms` window, so `verify_ms` itself is
probe-contaminated and must not be used for this comparison).

**Result (10 exactness prompts, fixed `DS4_DSPARK_VERIFY_K=4`, 211 cycles,
K≈5):**

| accepts | n | clean seq decode | batched (fixed) | winner |
|---:|---:|---:|---:|---|
| 0 | 86 | 0.0 ms | 72.1 ms | sequential |
| 1 | 50 | 27.8 ms | 69.7 ms | sequential |
| 2 | 34 | 56.8 ms | 71.3 ms | sequential |
| 3 | 16 | 84.5 ms | 71.8 ms | batched |
| 4 | 25 | 112.4 ms | 72.3 ms | batched |

- clean sequential decode slope = **28.2 ms/accept** (scales with accepts).
- batched verify = **~72 ms (K=5) / ~66 ms (K=4, bench)** fixed, independent of
  accepts.
- **crossover ≈ 2.4–2.6 accepts.**

**Verdict on the verifier modelling assumption:** the oracle `E[a|4]` = 2.198
(cycle-jump) / 2.337 (sliding) sits **at/slightly-below the crossover**, so the
batched verifier is **break-even at oracle acceptance, not a clear win.** The
model's premise (batched verify replaces K decodes at sublinear cost) only holds
above ~2.4 accepts; below it, the sequential verify short-circuits more cheaply.
So routing DSpark through the batched verifier would NOT clear baseline even at
oracle acceptance — the committing batched-verify build is not justified by the
cost economics alone. Artifact:
`issue468/artifacts/rejection_acceptance/verify_dist_probe_vk4_timed.jsonl`.

**Secondary finding (larger than the verifier question): runtime-vs-oracle
acceptance gap.** This run's `verified` mean = **1.26** (effectively K=4 — see
the codex-amended note below), vs the oracle `E[a|4]` = **2.198** — the runtime
drafter realizes only **~57% of the oracle's acceptance**, and that is with the
full-span fixed verify (best case), not the confidence-truncated scheduled path. This gap is consistent with (and much
larger than) the milestone-1 runtime acceptance (~1.07–1.09) and points at
drafter-state pollution / runtime drafter underperformance vs the offline oracle
(the Lead 06 caveat: the cycle-jump measures linear-trajectory difficulty, not
runtime state pollution). Closing this gap is a far larger lever than verifier
shape: it would both push acceptance above the batched crossover AND raise
throughput directly on the acceptance axis.

**Implication for focus:** the verifier is break-even at best; the binding
runtime gap is acceptance (runtime 1.26 vs oracle 2.42). The next decision is
whether to (a) chase the runtime-vs-oracle acceptance gap (drafter state /
implementation — could be a real bug or the documented state-pollution effect),
or (b) accept the IQ2XXS ceiling and pivot to drafter quality (Lead 07) / hidden
precision (Lead 04). Awaiting reassessment.

### 2026-07-12 — adversarial codex review of the M2 verify/acceptance measurements (commissioned, verified)

Commissioned a gpt-5.5/xhigh adversarial review of C1–C5 above (the RS-neutral,
batched-exactness, crossover, runtime-gap, and “verifier-not-the-lever” claims).
Review retained at
`issue468/artifacts/dspark_codex_reviews/2026-07-12_gpt55_xhigh_m2_verify_dist_review.md`.
Findings INDEPENDENTLY re-verified before acceptance:

- **Confirmed — acceptance ratio was slightly overstated (52% → 57%).** The
  fixed-K run was effectively K=4: a clamp bug leaves `verify_n = draft_n`
  (=block=5) in the fixed non-scheduled path even though only `fixed_verify_n`
  (=4) rows are drafted, so the garbage 5th draft caps `verified` at 4
  (confirmed: max verified=4). The correct oracle comparison is K=4-vs-K=4:
  1.2607 / 2.198 = **57.4%**, not 52% (which used E[a|5]=2.42). The gap is
  still large; the headline (acceptance is the lever) is unchanged.
- **Confirmed — fixed-K `verify_n` clamp bug.** In the DSpark branch
  `verify_n = draft_n` is set before `draft_n` is reduced to `fixed_verify_n`,
  and the fixed non-scheduled path never re-clamps `verify_n`. Effect: fixed-K
  runs verify against (K..block) positions where drafts[K..] are uninitialized;
  in practice the garbage caps `verified` at K, so measurements are
  accidentally correct, but the code is sloppy and the batched probe paid for
  K=5 (not K=4) compute. To fix: clamp `verify_n = draft_n` after the
  `draft_n = draft_eval_n` assignment in the fixed path. Not yet fixed (noted).
- **Confirmed — `verify_decode_ms` understates full sequential cost.** It times
  only `metal_graph_eval_token_raw_swa_top`, omitting the per-accept
  `dspark_session_push_graph_hidden` and the final logits readback. Corrected
  slope: 28.12 → **28.88 ms/accept** (push+readback add ~0.76 ms/accept). Probe
  self-inflation (the probe makes `eval_token_raw_swa_top` read full logits per
  accept) is real but ~0.009 ms/accept — negligible.
- **Confirmed — batched committing overhead not measured.** `batched_verify_ms`
  wraps only `verify_suffix_tops`; snapshot/restore + per-position-hidden
  capture + RS resample are outside. So the crossover is optimistic for batched.
- **Confirmed — per-position TV not stored.** The artifact has cycle-mean TV/KL
  only, so the 0.64% flip / 0.0035 TV are cycle aggregates; “distribution-exact”
  is slightly stronger than the evidence (need per-position + mismatch-position
  coverage).
- **RS +2.6% is statistically thin.** Bundle-clustered 95% CI ≈ [−0.145,
  +0.270] (straddles 0) on n=80 blocks; “neutral” is the honest framing (which
  the worklog already uses), not “+2.6% real.”

**Revised crossover (after corrections):** K=4 break-even **2.28–2.31** accepts
(66 ms / 28.88); K=5 **2.52** (72 ms / 28.88); +3 ms committing overhead → K=4
~2.39, K=5 ~2.63. Sensitivity: with decode slope ±25% and batched ±10%, K=4
break-even ranges ~1.8–3.7. Tightest-margin input = decode slope.

**Conclusion survives (codex: “directionally sound, too absolute”).** C5 holds:
the verifier is break-even at oracle acceptance (oracle 2.2–2.34 vs crossover
2.28–2.4), not a clear win. Codex’s caveat: the margin is tight enough that a
low-overhead committing batched path could win at oracle SLIDING acceptance
(2.337) — so “verifier ruled out” is too absolute; “verifier marginal, not the
binding lever” is the defensible framing.

**Highest-value next experiment (codex + agreed):** the live-vs-oracle drafter
trace — log the live DSpark drafter’s q (draft logits/hiddens) at each cycle
start and compare to the offline oracle q on the SAME anchors. This decisively
separates “the runtime drafter is polluted/degraded” (fixable runtime bug or the
Lead 06 state-pollution floor) from “the offline oracle overstates acceptance”
(corpus/methodology). Either outcome re-ranks the levers. Prerequisite: fix the
fixed-K clamp so the trace runs at a clean K.

### 2026-07-12 — fixed-K clamp fixed + live-vs-oracle drafter trace: spines agree, live >= oracle on matched anchors (the "acceptance gap" looks like a capture-length artifact)

**(1) Fixed-K clamp fixed.** Added an unconditional `if (verify_n > draft_n)
verify_n = draft_n;` in the DSpark branch after `draft_n = draft_eval_n`, so a
`DS4_DSPARK_VERIFY_K` setting actually constrains the verify span (previously
`verify_n` stayed at the full block with garbage drafts beyond K). Verified:
`DS4_DSPARK_VERIFY_K=5` now yields `verify_n=5` cleanly (was block-overflow).

**(2) Live-vs-oracle drafter trace built.** Added `anchor_id` + `draft_ids[5]` to
`ds4_dspark_cycle_metrics` (populated per cycle, emitted in ds4-spec-bench JSON),
so the live runtime's actual draft tokens per cycle can be compared to the
offline oracle's drafts on the same anchors. Added a `think` run-config field to
ds4-spec-bench (the oracle captures were made with `ds4`'s default
`DS4_THINK_HIGH`; the bench hardcoded `DS4_THINK_NONE`, a 3-token chat-template
difference that changed the greedy spine — `think:"high"` reproduces the oracle
spine). Comparison script: `dspark_oracle/compare_live_vs_oracle.py` (position-
based alignment: live advances by verified+1 per cycle; match to oracle row at
the same spine step, sanity-checked on the anchor token).

**Result (q2 baseline captures — the deployment config, per reviewer correction
NOT q4tap; 10 exactness prompts, fixed K=5, think=high):**

| metric (matched anchors, n=20) | live | oracle |
|---|---:|---:|
| p1 (draft[0]==target) | 0.85 | 0.75 |
| mean accepted prefix | 2.60 | 2.10 |
| draft[0] agreement (live==oracle) | 0.90 | — |
| full-block draft agreement | 0.60 | — |

**The greedy spines AGREE** — at every matched position the live anchor equals
the oracle anchor (no divergence); the runtime drafter is on the correct
trajectory. And on those matched anchors, **live acceptance is at or ABOVE the
oracle**, not below. Artifacts: `artifacts/rejection_acceptance/live_trace_vk5_q2_think.jsonl`,
`live_vs_oracle_q2_think.json`, `q2_t0p0_rs_summary.json`.

**Critical caveat — the n=20 is bounded by the oracle's SHORT capture.** The q2
bundles only captured ~8 generated steps/prompt (generated_tokens=14); the live
run generates 48 tokens, so most live cycles land beyond the oracle's capture
("no oracle row"). The 20 matched pairs are the early-generation cycles. So this
is suggestive, not conclusive at scale.

**Implication (re-ranks the prior conclusion):** the "runtime realizes 57% of
oracle acceptance" finding (C4) looks like a **measurement artifact**, not a
runtime deficiency. The live mean-verified (1.3) is over a LONG generation (48
tok, including lower-acceptance later cycles); the oracle E[a] (2.2) is over
SHORT early-generation captures (8–14 tok). On matched early-generation anchors,
live >= oracle. So the runtime drafter is NOT systematically degraded vs the
oracle — the gap was largely capture-length + the earlier cross-encoding
(raw-48 / chat-70 vs oracle chat-67) mismatch. **This weakens the "acceptance gap
is the lever" conclusion (C5's secondary): if the runtime drafter is fine, the
binding question returns to the verifier economics (break-even) and the model's
IQ2XXS ceiling, not a runtime drafter bug.**

**Decisive follow-up to confirm:** re-capture the oracle with a LONG generation
(generated_tokens=64+) so live cycles have oracle rows to match across the full
trajectory, then re-run the comparison at n=100+. If live >= oracle holds at
scale, the acceptance-gap lever is closed and the focus returns to
verifier/drafter-quality (Lead 07/04), not runtime drafter forensics.

### 2026-07-12 — large-set re-capture (lead3, 60 prompts): live >= oracle CONFIRMED at n=946; the acceptance gap was a measurement artifact

Re-captured on a large set (lead3, 60 prompts) the way the user requested.
Instead of the slow per-prompt ds4 capture (3 loads/prompt), used an efficient
one-load approach: dumped the live runtime's `main_hidden` per committed token
during the live trace (`DS4_DSPARK_DUMP_HIDDEN`, forces a host refresh per
push), then converted those dumps into oracle bundles and ran the offline oracle
drafter on the LIVE greedy-spine hiddens. Because greedy speculative preserves
the greedy spine, the dumped hiddens ARE the clean capture, and the oracle is
**self-aligned to the live trajectory by construction** — no chat-template /
encoding alignment needed (the earlier 70-vs-67-token template mismatch is
moot). Tooling: `dspark_oracle/convert_dumps_to_bundles.py`, the bench
`--dump-hidden-dir` flag, and `compare_live_vs_oracle.py` (robust sequential
anchor alignment, handling the +1 index offset between live cycle and dump).

**Result (lead3, fixed K=5, n=946 matched anchors across 49 prompts):**

| metric | live | oracle |
|---|---:|---:|
| p1 (draft[0]==target) | **0.517** | 0.477 |
| mean accepted prefix | **1.64** | 1.21 |
| draft[0] agreement (live==oracle) | 0.71 | — |

**Verdict (decisive, large n): the runtime drafter is NOT degraded vs the
oracle. On the same greedy spine, live acceptance is at or above the oracle.**
This CONFIRMS the n=20 finding at scale and closes the question: **the
"runtime realizes 57% of oracle acceptance" gap (C4) was a measurement
artifact** of comparing across mismatched encodings + capture lengths, NOT a
runtime drafter deficiency. Artifacts: `live_trace_lead3_vk5.jsonl`,
`lead3_hidden_dumps/`, `lead3_oracle_bundles/`, `live_vs_oracle_lead3.json`.

Caveats: (a) absolute acceptance is depressed (oracle 1.21, live 1.64 << the
dossier's powered-corpus E[a|5]~2.4) because this run used a short context
(frontier_tokens=8 to fit all 60 prompts) — the short context lowers acceptance
for BOTH, but the RELATIVE live-vs-oracle comparison is valid; (b) draft[0]
agreement is 0.71 not ~1.0, i.e. the oracle-on-dumps and the live drafter do not
produce identical drafts despite the same hiddens — the oracle's window rebuild
from dumps differs subtly from the live's incremental sliding window, so the
live>oracle margin partly reflects oracle-construction differences; the decisive
point (live NOT worse) is unaffected.

**Net reassessment input:** the runtime drafter is sound. C4/C5's
"acceptance-gap-is-the-lever" is falsified as a runtime-bug hypothesis. The
binding question is back to verifier economics (break-even at oracle
acceptance) and drafter QUALITY (Lead 07 / Lead 04 hidden precision), not
runtime drafter forensics. A clean absolute-acceptance re-measurement would use
longer context (frontier >= full prompt) so the numbers are comparable to the
powered corpus.
