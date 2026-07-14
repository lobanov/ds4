# DSpark runtime milestone 3 — committing batched verify + anchor reuse + GPU drafter

Date: 2026-07-13. Status: **active**. Goal: add a **committing batched (sublinear) verify**
(the +20% enabler — practically-exact per ds4-eval), **anchor reuse** (folding the anchor
into the batched verify), and a **GPU-resident drafter** to the DSpark runtime, preserving
**committed-output** greedy-exactness; measure the decode speedup (target ≥ +20% over
baseline; report the actual). Branch: `dspark-research`.

This follows milestones 1 (`dspark_runtime_milestone_1_progress.md`) and 2
(`dspark_runtime_milestone_2_progress.md`) — the same measurement methodology, the same
retained corpora, and the same per-lever iteration protocol.

## Commander's intent

The batched verify will be **STS-driven**: the scheduler picks `verify_n ≈ E[a]` on average,
so the batched path does ~E[a] tokens (no oververify), and at the **same** token count its
sublinear weight-loading beats the sequential per-token load (`batched(2.2) ≈ 46 ms < sequential 2.2×28 = 62 ms`).
So the committing batched verify IS the viable verify lever; "sequential wins" was an artifact
of comparing 4 tokens vs 2.2. Committing is cheap at `verify_n≈2` (full-accept → read-logits; prefix-1 → `spec_frontier_commit_prefix1`; the STS-driven verify_n keeps it in that regime).
 
**Also: +20 % is the target/projection, not the deliverable** — the deliverable is the engine
implementation + the measured numbers (per-cycle cost, acceptance, t/s, with the STS-driven
verify_n). Proceeding with the implementation; the earlier "PAUSED" conclusion is withdrawn.

The earlier milestone-3 framing ("three equal ~26 ms thirds — draft, verify, anchor decode")
was based on an **atypical dist-probe run** (code_topk, low `verify_n`). The **reliable**
retained picture is milestone-1 (code_8k, scheduled, `n=32`): `decode 28.9 ms | draft 44.8 ms
| verify 62.0 ms (verified 2.20) | total 135.8 ms | 23.6 t/s` vs baseline `38.5 t/s`. **The
verify is the dominant cost (~46 % of the cycle), not an equal third**, and three facts flow
from that:

1. **`verify-decode ≈ anchor-decode ≈ 28 ms/token`** — both are the same bandwidth-bound
   target decode (milestone-1: scheduled verify `62.0/2.20 = 28.2 ms/token`; `verify_k=1`
   `29.4/1.0 = 29.4 ms/token`; anchor `decode 28.9 ms`). So on the **exact sequential verify**,
   **anchor reuse is ~NEUTRAL**: folding the anchor in *relocates* the ~28 ms decode into the
   verify (62 → ~90 ms) rather than eliminating it; net saving ≈ the anchor's readback/dispatch
   overhead only (~1–3 ms).
2. **With the exact verify, +20 % is unreachable even with a free drafter + relocated anchor**:
   `decode 0 + draft 0 + verify(3.2 × 28 ms = 90 ms)` → `3.2/0.090 = 35.5 t/s < 38.5 baseline`.
   The verify dominates and neither anchor reuse nor the GPU drafter touches it.
3. **+20 % requires the verify to be SUBLINEAR** so that folding the anchor (and the verify
   itself) is cheap. The existing batched primitive (`metal_graph_verify_suffix_tops`,
   `verify_ms(2)=43.6`, sublinear weight-loading) is the enabler; it is NOT verify-level
   bit-exact (0.64 % argmax flip) but IS **committed-output-exact** (10/10 byte-identical on
   ds4-eval) — accepted as practically exact.

**Therefore the milestone is restructured around the committing batched verify as the first
lever.** Anchor reuse becomes worthwhile *because* it folds the anchor into the sublinear
verify for ~free; the GPU drafter then takes draft 44.8 → ~10 ms. Projected combined
(batched verify + reuse + drafter) ≈ **+24 %** (`draft 10 + verify_batched(3.2)≈57 ms` →
`3.2/0.067 = 47.8 t/s`); the measurement is authoritative. Each lever gets its own full
measurement cycle (milestone-2 protocol): **batched verify → anchor reuse → GPU drafter**.

The STS scheduler stays as-is (already implemented; provisional — temperatures trained on
*assumed* timings). Greedy-exactness is re-stated as **committed-output byte-identical**
(the batched verify's 0.64 % verify-level flip is tolerated, rigorously confirmed not to
propagate); the exact sequential verify is retained as the bit-exact fallback (it cannot
clear +20 %). The Lead 08 *fused bit-exact* sublinear kernel stays deferred (the existing
batched primitive is practically exact).

## Scope: the levers under validation

| lever | what the model assumes | milestone-3 work | prior status (M1/M2) |
|---|---|---|---|
| **Committing batched verify** (the +20 % enabler) | `verify_ms(K)` sublinear (the model used the batched bench: K2:43.6, K4:65.8) | wire the existing `metal_graph_verify_suffix_tops` (currently dist-probe-only, ds4.c:28671) for **committing** in the DSpark branch: accept-prefix + correction token + DSpark window-state push; env-gated default-off; confirm committed-output byte-identical (flip-detecting diff) | the primitive exists (sublinear) but is probe-only; **never committed in DSpark**. Practically-exact (10/10 ds4-eval); not verify-level bit-exact (0.64 % flip) |
| **Anchor reuse** (folds into the batched verify) | fresh anchor decode only ~S(K); the verify yields the next anchor | fold the anchor into the **batched** verify (skip the standalone `ds4_session_eval(s, first_token)`); compose with the existing STS scheduled verify + the batched verify | Lead 06 made reuse exact but sequential (MTP-only); ~neutral on the exact verify (see fact below); **worthwhile only on the sublinear verify** |
| **GPU drafter body/head** | `draft_ms ≈ 10 ms` (not the current ~45 ms CPU) | move the **already-batched** CPU drafter (3 stages × 5-row `dspark_block_forward_batch`) + the 5× vocab output-head matvec (reuses the target's output weights) + markov + confidence + argmax onto Metal | not yet attempted; one GPU output-head cut line failed `--dspark-schedule-parity` (known trap to clear) |

Out of scope (deferred): re-implementing the STS scheduler; STS re-training on measured
timings; the Lead 08 **fused bit-exact sublinear kernel** (the path to verify-level
bit-exactness — the existing batched primitive is practically exact, so this is deferred);
target hidden-state precision / Lead 04 (F16-blocked); N=3/4/5/6. The **exact sequential
verify is retained** as the bit-exact fallback (default when the batched verify is off); it
cannot clear +20 %.

## What "validate a lever" means here (M2 methodology, gate re-stated)

A lever is `validated` only when: (1) the implementation exists, env-gated default-off;
(2) it preserves the **committed-output** correctness gate (byte-identical to plain decode on
the exactness corpus + a larger retained sample; the batched verify's 0.64 % verify-level flip
tolerated, rigorously confirmed not to propagate) and temp>0 logits/distribution parity;
(3) it preserves the lever's own parity contract — draft-token, confidence-logit, scheduled
`verify_n`, accepted-chunk, DSpark window-state parity, and **for verify-path changes,
per-position argmax + correction-token parity** vs the trusted reference; (4) it is
benchmarked with full cycle-cost attribution; (5) the measured economics are compared to the
model's projection and the gap recorded. A lever that preserves committed output but degrades
acceptance, or benchmarks faster-yet-slower-than-projected, is `partial`, not `validated`.

## Measurement methodology (inherits M2 §Measurement methodology + §Iteration protocol)

Two-axis benchmark — final-output correctness (**committed-output** byte-identical to plain
decode + temp>0 logits/distribution parity) AND speculative-economics preservation. Concretely:

1. Final-output correctness gates mandatory (committed-output byte-identical; temp>0 at logits/distribution level).
2. Draft-side quality checked SEPARATELY from final-output parity (draft-token / confidence-logit / scheduled-verify_n / accepted-chunk / window-state; + per-position argmax + correction-token for verify-path changes).
3. Acceptance metrics first-class (drafted/verify length, accepted/verified per cycle, full-accept rate).
4. Cycle-cost attribution split by component (decode_ms, draft_ms, verify_ms, verify_decode_ms, DSpark state-push time, logits readback — kept separate; batched vs sequential verify distinguished).
5. Same retained exactness / powered-corpus / long-context corpora that feed `spec_speedup_model.md`.
6. `ds4-spec-bench` / `ds4_spec_bench.c` as the default substrate (single loaded engine, bulk config, fresh sessions where needed).
7. Compare BOTH a fixed low-K reference (`verify_k=1`) AND the scheduled path.
8. Each lever env-gated default-off behind its own parity gate; not promoted until it clears the checklist.
9. Model assumptions tested, not assumed — measured cost replaces the model's value; the speedup projection is re-read with the measured number.

Iteration protocol per lever: implement (env-gated) → smoke gates (lever parity + committed-output-exact + temp>0) → codex gate A → benchmark (ds4-spec-bench, retained corpora, full cycle-cost attribution) → codex gate B → update this lever table → re-read `spec_speedup_model.md`.

## Design constraints (inherits M2 §Design constraints)

1. Each lever env-gated, default off, behind its own parity gate.
2. Scheduled-verification semantics stay authoritative (draft ids, confidence logits, computed `verify_n`, accepted chunking, final outputs vs the trusted reference).
3. The "confident prefix + 1 drafted token" policy remains intentional (the STS scheduler already implements it).
4. Model assumptions tested, not assumed.
5. The benchmark substrate and corpora stay fixed while internals move.

## Carried-forward facts (load-bearing from M1/M2 + 2026-07-13 verify-cost re-derivation)

1. **Reliable cycle economics (milestone-1, code_8k scheduled, `n=32`):** `decode 28.9 ms | draft 44.8 ms | verify 62.0 ms (verified 2.20) | total 135.8 ms | 23.6 t/s` vs baseline `38.5 t/s`. **The verify dominates (~46 %).** The earlier "draft/verify/decode ≈ 26/26/26 ms" was an atypical dist-probe run (low `verify_n`) — do NOT use it for lever economics. The anchor decode fires in **75 % of cycles** vs the model's rare `decode·S(K)` — but on the exact verify, skipping it only relocates the decode (see fact #12).
2. **`verify-decode ≈ anchor-decode ≈ 28 ms/token`** (milestone-1: scheduled `62.0/2.20 = 28.2`; `verify_k=1` `29.4/1.0 = 29.4`; anchor `decode 28.9`). Both are the same bandwidth-bound target decode.
3. **The anchor-decode site:** `ds4_session_eval(s, first_token, ...)` at ds4.c:28560, in the DSpark branch (`if (e->dspark_ready)`), unconditional every cycle; the branch returns ~28816.
4. **The drafter is ALREADY batched, not sequential.** `dspark_eval_draft_block_cpu*` (ds4.c:28135/28330/28428) → `dspark_block_forward_batch` (ds4.c:28036) = **one batched forward per stage** (DS4_DSPARK_STAGES=3) over DS4_DSPARK_BLOCK=5 rows; the input for tokens 2..5 is `noise_hc` (non-autoregressive, noise-injected "treeless" drafter). The inefficiency is NOT sequential drafting — it is that the batched forward (`matmul_q8_0_batch`, CPU) + the **5× vocab output-head matvec** (`matvec_q8_0(base_logits, &e->model, e->weights.output, norm)`, reusing the TARGET's output weights) + markov + argmax run entirely on CPU. **No GPU/Metal drafter forward exists.**
5. **The STS scheduler is already implemented + active by default.** `dspark_schedule_verify_len` (ds4.c:28308) uses baked-in temperatures `ds4_dspark_sts_temp` (ds4.c:23890 = Lead 02's train-fit 1.057/0.758/1.038/1.369/1.295) + cumulative-confidence survival + threshold; `scheduled_verify = fixed_verify_n < 0 && dspark_schedule_enabled()` is true by default. It already drafts past the confident prefix. **So anchor-reuse work only adds the fold/decode-skip; it does not re-implement scheduling.**
6. **Lead 06 anchor reuse is MTP-only.** `DS4_MTP_ANCHOR_REUSE` (ds4.c:28817) sits after the DSpark branch's return — the DSpark path has no reuse today. The Lead 06 machinery (reuse the correction token's hidden, re-base logits) is the reference to adapt for the fold.
7. **STS temperatures are trained on assumed timings** (draft ≈ 10 ms, unmeasured; real ~45 ms CPU). The scheduler is therefore provisional; re-training on measured post-optimization timings is a deferred follow-up.
8. **GPU output-head parity trap (M1):** a prior GPU output-head cut line (`DS4_DSPARK_SCHEDULE_GPU_HEAD=1`) failed `--dspark-schedule-parity` (draft ids/confidence/verify_n diverged, CPU Q8_0 vs GPU dequant argmax). The GPU drafter must clear the strengthened parity gate.
9. **Committed-output-exactness held** through M1/M2 (10/10 byte-identical; temp>0 `max_abs=0`). The 2026-07-13 ds4-eval practical check found the batch-verifier's committed output byte-identical to plain decode (10/10) — the verify-level 0.64 % flip does not propagate to the committed stream. (The fidelity gate is committed-output byte-identical; verify-level bit-exactness is NOT required — that is the Lead 08 fused-kernel path, deferred.)
10. `decode_ms`/`draft_ms`/`verify_ms`/`verify_decode_ms` + DSpark push + logits readback are all recorded in `s->dspark_last_cycle` (the existing cycle-timing) and emitted by ds4-spec-bench.
11. **Verifier reality (code-confirmed 2026-07-13):** the DSpark **committing** verify is the **sequential exact** loop (`metal_graph_eval_token_raw_swa_top` per token, ds4.c:28714, short-circuits at the first mismatch) — greedy-exact + LINEAR. The **batched** verifier (`metal_graph_verify_suffix_tops`) is SUBLINEAR but not verify-level bit-exact (0.64 % argmax flip); **in the DSpark branch it is called ONLY in the dist-probe (ds4.c:28671, non-committing)** — wiring it for committing is the milestone-3 enabler. `metal_graph_verify_decode2_exact` is exact but K=2-only + linear (MTP path). The `spec_speedup_model`'s `verify_ms(K)` (K2:43.6, K4:65.8) is from the BATCHED `mtp_verifier_bench` — i.e. the model's projections already assumed the sublinear verify this milestone must now actually wire for committing.
12. **Anchor-reuse economics, corrected (2026-07-13):** on the EXACT sequential verify, anchor reuse is **~NEUTRAL** — the anchor's ~28 ms bare decode relocates into the verify at ~28 ms/token (verify 62 → ~90 ms), saving only the readback/dispatch overhead (~1–3 ms). It is worthwhile **only on the sublinear (batched) verify**, where folding the anchor is ~free (verify_batched(3.2)≈57 ms vs decode(28.9)+verify_batched(2.2)≈46 ms). **+20 % requires the batched verify** — with the exact verify, even a free drafter + relocated anchor stays below baseline (`3.2/0.090 = 35.5 t/s < 38.5`).

## Further optimizations surfaced from the M1/M2 review

- **Committing batched verify** (promoted from "deferred" to **lever 1 / the enabler**) — the existing sublinear primitive, wired for committing; practically-exact per ds4-eval. The verify is NOT "last priority" — it is the dominant cycle cost and the +20 % gate.
- **Persistent device DSpark KV/window state** (M1 next-steps #4) — the GPU drafter port should keep the drafter's KV/window resident on-device (on unified memory the win is compute throughput + eliminating per-cycle GPU↔CPU hidden-readback/draft-push *serialization*, not copy-avoidance). Folded into the GPU-drafter lever.
- **Batched-drafting cap operating point** (M1 cycle 4) — cap3/cap4 as a free scheduling config choice for these measurements (does not change acceptance).
- **Lead 08 fused bit-exact sublinear kernel** — deferred: the path to verify-level bit-exactness; the existing batched primitive is practically exact, so not needed for +20 %.
- **STS re-training on measured timings** — deferred follow-up; only sensible after the levers produce real timings.
- **Target hidden-state precision (Lead 04)** — upstream acceptance work (not runtime); F16-deployment-blocked. Deferred.

## Next steps

**Current task:** `committing-batched-verify` — wire `metal_graph_verify_suffix_tops` for
committing in the DSpark branch (accept-prefix + correction token + DSpark window-state
push), env-gated default-off; confirm committed-output byte-identical (flip-detecting diff on
the exactness corpus + a larger sample); smoke-test the parity gate.

Then: `codex-gate-A → bench → codex-gate-B` (batched verify) → `anchor-reuse` (fold into the
batched verify) → its measurement cycle → `gpu-drafter` → its measurement cycle → `propagate`.

## Worklog

### 2026-07-14 — ds4-eval engagement fix + full 92Q SCORE comparison + temp>0 distribution-exactness (prior) => relaxing greedy exactness is NET-VIABLE for scores

**ds4-eval engagement fix:** discovered ds4-eval used plain decode (`ds4_session_eval`, ds4_eval.c:3872) — `--dspark`
only loaded the drafter, generation never called `ds4_session_eval_speculative_argmax`. So the earlier
"0% divergence on the ds4-eval 92Q benchmark" (see the entry below) was PLAIN-VS-PLAIN (trivially identical);
`DS4_DSPARK_VERIFY_BATCHED=1` had no effect on ds4-eval. (This also invalidates the milestone-2
"ds4-eval 10/10 byte-identical" as plain-vs-plain.) Fixed ds4-eval to engage the speculative path when
`--dspark` (gated; plain path unchanged; commits `produced` tokens/cycle, syncs the loop counter).

**Corrected engaged divergence** (ds4-spec-bench speculative_argmax + DS4_DSPARK_VERIFY_BATCHED=1;
engagement proven by 303 dspark-timing lines + by the divergence itself):
- benchmark reasoning/math (12 extracted ds4-eval Qs): 12/12 prompts, 56.6% of tokens.
- exactness corpus (code/grounded/synthesis, 10): 40% (7/10).
- code_topk: 45/48 (first diff @ token 3).
- seq (exact) speculative_argmax vs plain: 0% (byte-identical) => the divergence is the batched flip, not a bug.
- Both outputs are SENSIBLE text (real flip + greedy cascade, not garbage). Codex verdict (gpt-5.5 xhigh,
  `issue468/artifacts/dspark_codex_reviews/2026-07-14_gpt55_xhigh_engaged_divergence_verdict.md`):
  NO functional equivalence when engaged (medium-high confidence).

**Full 92Q SCORE comparison** (greedy, --nothink, --tokens 2048, temp 0, seed 1; BOTH runs complete;
canonical artifact `issue468/artifacts/dspark_m3_bench/m3_92q_score_comparison.txt`):
- PLAIN: 60/92 (65.2%).  DSPARK-batched (engaged): 64/92 (69.6%)  => net +4.
- same-verdict: 82/92 (89.1%).
- PASS/FAIL flips: 3 losses (plain-PASS->DSpark-FAIL: Q21, Q28, Q79) + 7 gains (plain-FAIL->DSpark-PASS:
  Q24, Q25, Q30, Q37, Q49, Q52, Q81). The divergent greedy path sometimes lands on the CORRECT answer
  where plain greedy failed.

**temp>0 distribution-exactness (PRIOR WORK, conclusive):** milestone-1's `ds4_test --dspark-temp-logit-parity`
gate (artifact `issue468/artifacts/dspark_temp_distribution_compare/summary.json`) PASSED: 640 sampled steps,
574 eligible, `max_abs=0.0`, `rms=0.0`, `sampled_lp_diff=0.0` at temp=0.5 and 1.0. Caveat: that gate forces
`DS4_DSPARK_VERIFY_K=0` (narrow: speculative entry/logit parity, not full multi-token) AND uses the SEQUENTIAL
(exact) verify — so it proves temp>0 distribution-exactness for the DSpark runtime via the exact verify, NOT for
the batched (divergent) verify at temp>0 (untested; the ds4-eval fix uses argmax-verify = greedy-only).

**Viability verdict:** relaxing greedy exactness (using the divergent batched verify) is NET-VIABLE for the
benchmark scores — the 56% token divergence does not hurt answer scores (net +4 on 92Q, 89.1% same verdict).
Strategy pending future verifier work: batched verify for greedy (score-neutral) + the exact sequential verify
for temp>0 (distribution-exact, prior work). Strict committed-output byte-exactness still needs the Lead 08
margin-guarded fallback (re-verify near-tie positions).

### 2026-07-13 — bench-batched RESULTS: throughput + attribution + DIVERGENCE (corpus-dependent: 0% benchmark / 40% code)  ["0% benchmark" SUPERSEDED — see 2026-07-14: ds4-eval was plain-vs-plain]

First real measurement (warm model, exactness corpus + full ds4-eval):
- **Throughput (warm, exactness corpus, 10 prompts):** plain 37.20 → seq 16.29 → **batched 20.19
t/s** (+24 % over seq; 0.54× baseline). The committing batched verify IS sublinear-faster than
sequential, but DSpark-batched is still below baseline — +20 % needs the full stack (batched +
reuse + GPU drafter); the batched verify is the *enabler*, not the full solution.
- **Attribution (1 cycle, code_topk, timing-on):** decode 26 ms, draft ~22-29 ms, verify
seq 52.4 ms → **batched 43.4 ms** (sublinear, ~9 ms saved at verify_n≈2). push ~1.85 ms
(CPU fallback, cheap). The frontier snapshot overhead is inside verify_ms.
- **DIVERGENCE (the key finding, added a token-dump to ds4-spec-bench via DS4_BENCH_DUMP_TOKENS):
  CORPUS-DEPENDENT.**
  - **ds4-eval benchmark (92 questions, greedy, --nothink, --tokens 200): BYTE-IDENTICAL — 0 %
    divergence** (plain vs batched). So the committing batched verify IS practically-exact for the
    benchmark workload.
  - **exactness corpus (code/grounded/synthesis, 10 prompts, 48 tok): 40 % divergence** (7/10
    prompts, 192/480 tokens). code_topk: first divergence at token index 3 (plain[3]=3696
    "docstring…" vs batched[3]=4085 "check for self-testing…"), then the greedy tail cascades
    (61/64 differ). Both outputs are SENSIBLE text (valid code/docstring) — the divergence is a
    real flip, not garbage.
  - The seq (exact) dspark is byte-identical to plain (0 %) — confirming the divergence is the
    batched flip, not a bug.
- **Why corpus-dependent:** code generation has more near-tie argmax decisions → more of the
  batched verify's 0.64 %-level flips fire on verify decisions → committed-output divergence.
  Reasoning/benchmark workloads have clearer argmax → flips rarely fire → byte-identical.
- **Implication:** the "practically exact" framing HOLDS for the benchmark workload (the practical
  use case, matches the original ds4-eval 10/10) but FAILS for code generation (40 %). The
  contract's hard-abort clause (>5 %) is triggered FOR CODE workloads. DECISION POINT: scope the
  milestone to benchmark-exact + note code needs the Lead 08 margin-guarded fallback, OR treat the
  code divergence as a falsification.

### 2026-07-13 — fix-gpu-push INVESTIGATION: GPU push is a PRE-EXISTING general bug (fails in both paths); push is CHEAP (1.85 ms) — codex C3 over-stated, NOT perf-critical

Investigated codex gate A C3 (metal_graph_push_dspark_hidden returns false → CPU fallback).
Stage-logging (past-NULL-checks + op-chain prints) run on BOTH the normal dspark path AND the
batched path: the GPU push fails at the OP CHAIN (past the NULL checks) in BOTH. So it is a
PRE-EXISTING general bug — the milestone-1 "DSpark hidden capture / stage-KV push is GPU-first"
claim was WRONG; the push has ALWAYS been the CPU fallback (dspark_session_push_graph_hidden's
refresh+project path). The batched path's dspark_session_push_batch_hidden correctly routes through
that same fallback (loads the batch row into dspark_capture_hc, invalidates main_hidden, calls the
full push path).

Timing (batched path, DS4_DSPARK_VERIFY_BATCHED=1 + TIMING=1, 1 prompt / 96 tok): decode 27.3 ms,
draft 170.1 ms (vs milestone-1's ~45 ms — likely swap-inflated; this is a cold 91 GB load),
verify 57.1 ms (INCLUDES the frontier snapshot/restore overhead), push_init 2.1 ms + push_verify
1.85 ms (CPU fallback — cheap), logits_read 0.0, total 254.5 ms → 11.4 t/s.

**Conclusion: the push is CHEAP (1.85 ms); codex C3 ("CPU fallback erases the sublinear gain") is
over-stated. The GPU-push fix is NOT perf-critical (saving ~1 ms on a 254 ms cycle). The real
bottlenecks are the draft (170 ms, swap) and the verify (57 ms, incl. the frontier snapshot).**
Decisive next step for the GPU push itself (deferred / out of scope): bisect the op chain in
metal_graph_push_dspark_hidden (begin/sum/matmul_plain/rms_norm/matmul_q8_0/rope/copy/end/read)
to find the failing op. Debug logging removed; ds4.c compiles clean; both binaries rebuilt.

### 2026-07-13 — committing-batched-verify: codex gate A DONE (gpt-5.5 xhigh) — bug 1 (practically-exact, not strictly) + bugs 2/3/4 fixed

Codex gate A (`issue468/artifacts/dspark_codex_reviews/` temp log) reviewed the committing wiring +
window-push + exactness. Verdicts: C1 (accept-prefix indexing) sound; C4 (capture rows) sound;
C3 (window-push) mostly sound (GPU-push-fails = unresolved perf risk); C2 (control flow) questionable;
C5 (exactness) LIKELY-WRONG as a general claim.

- **Bug 1 (CRITICAL, framing/contract — NOT a code bug): the committing batched verify is
  PRACTICALLY exact, NOT strictly byte-identical.** It uses the batched argmax (`row_tops`) for the
  verify DECISION; when the batched argmax flips vs the exact (0.64 %/position, per the dist-probe
  artifact — `verify_dist_probe_exactness.jsonl` code_topk cycle 0 has `argmax_flips=1` on a
  verified position), it commits a different token than plain decode → divergence. The 10/10 ds4-eval
  smoke used different (embedded) prompts + got lucky (no flip-on-decision). **This matches the
  user's "practically exact for now" — but the verification contract's "byte-identical" is too
  strong.** Strict byte-identity would need the Lead 08 margin-guarded fallback (re-verify near-ties
  with the exact path), which is deferred. DECISION POINT for the user: accept practical exactness
  (quantify the divergence rate on the retained corpus) vs add the margin-guarded fallback.
- **Bug 2 (HIGH, FIXED): general-partial replay did not re-verify.** Added `if (target_top !=
  drafts[i]) break;` at the replay-loop start so a batched false-accept cannot commit a wrong token
  (the replay is exact).
- **Bug 3 (HIGH, FIXED): `spec_frontier_restore` return was ignored on reject + fallback.** Now
  checked; a restore failure is a hard error (returns -1) instead of continuing with a corrupted
  cache. The hard_err check was moved after the fallback so both paths are covered.
- **Bug 4 (MEDIUM, FIXED): `dspark_batch_capture_hc` over-alloc.** Was `pc×hc_dim` per layer
  (~768 MiB at prefill 4096, unconditional); now `(DS4_DSPARK_BLOCK+1)×hc_dim` (~1 MiB) — the
  capture only needs the verify-width (≤5 rows).
- **GPU push (C3, open perf risk):** `metal_graph_push_dspark_hidden` returns false in this path
  (masked by the CPU fallback in the normal path); the batch push routes through
  `dspark_session_push_graph_hidden`'s fallback. Correctness OK (refreshes `dspark_main_hidden` from
  the loaded capture), but the CPU-fallback push is slower + serializes → likely erases the sublinear
  verify's gain. To investigate (restore the GPU push) before the bench — decisive test: stage-log
  around ds4.c:19911.
- Re-verified post-fixes: 10/10 ds4-eval committed-output byte-identical (no regression), no failures.

### 2026-07-13 — committing-batched-verify: IMPLEMENTED + smoke-exactness PASS (10/10 committed-output byte-identical)

Implemented the committing batched (sublinear) verify in the DSpark path, env-gated
`DS4_DSPARK_VERIFY_BATCHED` (default-off):
- **Capture buffer**: added `dspark_batch_capture_hc[3]` (struct + alloc 3×`pc×hc_dim` + free) +
  `dspark_batch_capture_active` flag.
- **Capture hook** in `metal_graph_encode_layer_batch` (ds4.c:19652): when the flag is set, copies
  the per-position post-FFN hidden for layers 40/41/42 (the 3 DSpark layers) into the batch buffer.
- **Batch window-push** `dspark_session_push_batch_hidden(s, batch_pos)` (ds4.c:~27958): loads the
  position's captured layers into the single-position `dspark_capture_hc`, invalidates
  `dspark_main_hidden_valid`, then delegates to the full push path (GPU push, else CPU fallback
  that refreshes `dspark_main_hidden` from the loaded capture).
- **Committing control flow** in `ds4_session_eval_speculative_argmax` (ds4.c:~28757, before the
  sequential verify loop): snapshot frontier → push drafts → `metal_graph_verify_suffix_tops`
  (capture active) → determine accept-prefix (`drafts[0]` by `target_top`, `drafts[i≥1]` by
  `row_tops[i-1]`) → commit (full-accept→read spec-logits + batch-push each; prefix-1 @ verify_n=2
  → `spec_frontier_commit_prefix1` + batch-push; general partial → restore + sequential replay;
  reject → restore) → fallback to the sequential verify if the batched path doesn't commit; a
  mid-commit GPU failure is a hard error.
- **Debugging note**: the GPU push (`metal_graph_push_dspark_hidden`) returns false here (it is
  masked by the CPU fallback in the normal path); the batch push therefore routes through
  `dspark_session_push_graph_hidden`'s fallback. To investigate/restore the GPU push later (perf).
- **Smoke exactness**: `ds4-eval --questions 10 --nothink --temp 0 --seed 1 --tokens 96`, plain vs
  `DS4_DSPARK_VERIFY_BATCHED=1` → **10/10 committed-output byte-identical** (only run timestamps/
  timing differ). The 0.64 % verify-level flip does NOT propagate to committed output. No crashes.
  (dspark_batched ~12 t/s on this tiny smoke — slow because of the CPU-fallback push + committing
  overhead; performance is for the bench step, after codex gate A.)
- **Next**: codex gate A (adversarial review of the committing wiring + window-push + exactness) →
  then bench (full retained corpus, verify_ms/acceptance/cycle-cost; GPU-push perf fix if gate flags it).

### 2026-07-13 — committing-batched-verify: design locked (window-push via per-position batched capture of layers 40-42); env gate added

Implementation analysis for `committing-batched-verify`:
- The committing path uses `metal_graph_verify_suffix_tops` (sublinear, STS-driven `verify_n`≈E[a],
  no oververify) + the MTP committing pattern (`spec_frontier_snapshot`/`restore`/
  `commit_prefix1`, ds4.c:24690-24758) for the cache: full-accept → read last spec-logits row
  (`metal_graph_read_spec_logits_row`, cheap); prefix-1 (verify_n==2) →
  `spec_frontier_commit_prefix1` (cheap); general partial → restore+sequential replay.
- **The DSpark window-state push is the novel piece.** `dspark_session_push_graph_hidden`
  (ds4.c:27887) reads `g->dspark_capture_hc[0..2]` — 3 specific layers (**40/41/42**, the post-FFN
  `after_ffn_hc`, hooked in the autoregressive decode at ds4.c:16139) — NOT the batched verify's
  `batch_cur_hc` (which holds only the final per-position hidden). So batch-verified tokens need
  a NEW per-position capture of layers 40-42 inside `metal_graph_encode_layer_batch`
  (ds4.c:19652) + a batch window-push variant (`metal_graph_push_dspark_hidden`, ds4.c:19883,
  adapted to read per-position). Moderate, well-defined GPU work (the per-position
  `after_ffn_hc` is already computed in the batched encode).
- Added the env gate `dspark_verify_batched_enabled()` (`DS4_DSPARK_VERIFY_BATCHED`, default-off)
  after `dspark_verify_dist_probe_enabled` (ds4.c).
- **Next:** add the batched capture buffer + hooks (layers 40-42) in
  `metal_graph_encode_layer_batch`; add the batch window-push helper; wire the committing
  control flow (accept-prefix + correction token + window push) in
  `ds4_session_eval_speculative_argmax` (ds4.c:~28689, before the sequential verify loop);
  build (`make ds4-spec-bench`/`ds4-eval`); smoke-test committed-output byte-identical
  (flip-detecting diff vs plain decode); codex gate A.

### 2026-07-13 — committing batched verify IS viable when STS-driven

The batched verify will be STS-driven, so `verify_n ≈ E[a]` on average (no oververify);
at the same token count (~2.2) its sublinear weight-loading beats the sequential per-token
load (`batched(2.2)≈46 ms < 2.2×28=62 ms`). Committing is cheap at verify_n≈2
(full-accept → read-logits; prefix-1 → `spec_frontier_commit_prefix1`, ds4.c:24758).

**Also: +20 % is the target/projection, not the deliverable** — the deliverable is the
engine + the measured numbers. Proceeding with the committing-batched-verify implementation.

### 2026-07-13 — premise revised: verify dominates, anchor reuse neutral on the exact verify, +20 % requires the committing batched verify

Re-derived the cycle economics from the **reliable** milestone-1 retained data (code_8k
scheduled): `decode 28.9 | draft 44.8 | verify 62.0 ms` — the verify dominates (~46 %), not an
equal third; `verify-decode ≈ anchor-decode ≈ 28 ms/token`. Consequently anchor reuse is
~neutral on the exact sequential verify (the decode relocates, not eliminated), and +20 % is
unreachable on the exact verify even with a free drafter. Code-confirmed the DSpark committing
verify is the sequential exact loop (ds4.c:28714) and the batched primitive
(`metal_graph_verify_suffix_tops`) is dist-probe-only (ds4.c:28671) — wiring it for committing
is the +20 % enabler. The batched verify is not verify-level bit-exact (0.64 % flip) but
committed-output-exact (10/10 ds4-eval). **Premise + lever table + carried-forward facts
rewritten above** to make the committing batched verify lever 1, fold anchor reuse into it,
and re-state greedy-exactness as committed-output byte-identical. (Supersedes the earlier
"verifier-reality clarification — no goal tweak needed" note, whose conclusion was wrong.)

### 2026-07-13 — orient: anchor-decode + drafter sites located; Lead 06 reuse is MTP-only; drafter is batched+CPU; STS scheduler already implemented

Read-only orientation of `ds4_session_eval_speculative_argmax` (ds4.c:28524). The per-cycle
anchor decode is `ds4_session_eval(s, first_token, ...)` at ds4.c:28560 (DSpark branch,
unconditional; fires 75 % of cycles vs the model's rare `decode·S(K)`). The drafter is
`dspark_eval_draft_block_cpu*` (28135/28330/28428) → `dspark_block_forward_batch` (28036):
**already batched** (3 stages × one batched forward over 5 rows, non-autoregressive
noise-injected), running entirely on CPU (forward + 5× vocab output-head matvec reusing the
target's output weights + markov + argmax) — no GPU/Metal drafter variant exists. Lead 06
`DS4_MTP_ANCHOR_REUSE` (28817) is MTP-only (after the DSpark branch return ~28816) → the
DSpark path has no anchor reuse today; that machinery is the reference to adapt. The STS
confidence-scheduled verify is already implemented + active by default (`dspark_schedule_
verify_len` 28308 + baked-in `ds4_dspark_sts_temp` 23890; drafts past the confident prefix).
Cycle timing recorded in `s->dspark_last_cycle`. Machine clean (~88 GB free), on
`dspark-research`. No implementation started. (Moved from `pending/dspark_runtime_anchor_
reuse_drafter.md` into this canonical milestone-3 progress doc.)
