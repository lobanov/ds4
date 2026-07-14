# DSpark runtime milestone 3 — committing batched verify + anchor reuse + GPU drafter

Date: 2026-07-13. Status: **active** (lever 1 implemented + measured; levers 2/3 pending).
Goal: add a **committing batched (sublinear) verify** (the +20 % enabler), **anchor reuse**
(folding the anchor into the batched verify), and a **GPU-resident drafter** to the DSpark
runtime; measure the decode speedup (target ≥ +20 % over baseline; report the actual).

**Exactness findings (batched verify, measured 2026-07-14):** the batched verify is **NOT**
committed-output-exact. At **temp=0** it diverges (56.6 % of committed tokens on the
benchmark; 40 % on the exactness corpus) — the earlier "10/10 ds4-eval byte-identical" was
plain-vs-plain (ds4-eval didn't engage DSpark; now fixed) — BUT it is **score-neutral** on the
92Q benchmark (plain 60 vs DSpark-batched 64, net +4, 89.1 % same verdict). At **temp>0** its
sampled distribution diverges (TV ~0.0104, 0.64 % argmax-flip, max_abs up to 4.56 — dist-probe).
The **exact sequential verify** remains the byte-exact (temp=0) / distribution-exact (temp>0,
TV=0) fallback. Branch: `dspark-research`.

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
   bit-exact (0.64 % argmax flip) and, when engaged, NOT committed-output-exact either (see
   the exactness findings above) — but it is score-neutral on the 92Q benchmark at temp=0
   (net +4) and its temp>0 distribution divergence is small (TV ~0.0104).

**Therefore the milestone is restructured around the committing batched verify as the first
lever.** Anchor reuse becomes worthwhile *because* it folds the anchor into the sublinear
verify for ~free; the GPU drafter then takes draft 44.8 → ~10 ms. Projected combined
(batched verify + reuse + drafter) ≈ **+24 %** (`draft 10 + verify_batched(3.2)≈57 ms` →
`3.2/0.067 = 47.8 t/s`); the measurement is authoritative. Each lever gets its own full
measurement cycle (milestone-2 protocol): **batched verify → anchor reuse → GPU drafter**.

The STS scheduler stays as-is (already implemented; provisional — temperatures trained on
*assumed* timings). The batched verify is accepted as a **score-neutral interim at temp=0**
(the 92Q net +4) + a small-divergence interim at temp>0 (TV ~0.0104); the exact sequential
verify is retained as the byte-exact / distribution-exact fallback (it cannot clear +20 %).
Strict committed-output byte-exactness needs the Lead 08 margin-guarded fallback (re-verify
near-tie positions) — deferred.

## Scope: the levers under validation

| lever | what the model assumes | milestone-3 work | prior status (M1/M2) |
|---|---|---|---|
| **Committing batched verify** (the +20 % enabler) | `verify_ms(K)` sublinear (the model used the batched bench: K2:43.6, K4:65.8) | wire the existing `metal_graph_verify_suffix_tops` (currently dist-probe-only, ds4.c:28671) for **committing** in the DSpark branch: accept-prefix + correction token + DSpark window-state push; env-gated default-off; measure committed-output divergence (temp=0) + distribution divergence (temp>0) + 92Q scores | **DONE (lever 1).** Env-gated `DS4_DSPARK_VERIFY_BATCHED`; codex gate A ran (3 bugs fixed). Measured (warm): plain 37.2 / seq 16.3 / **batched 20.2 t/s** (verify 52→43 ms, sublinear). NOT committed-output-exact when engaged (56.6 % temp=0 divergence — the 10/10 ds4-eval was plain-vs-plain) BUT **score-neutral on 92Q** (net +4, 89.1 % same verdict); temp>0 TV ~0.0104, 0.64 % argmax-flip. |
| **Anchor reuse** (folds into the batched verify) | fresh anchor decode only ~S(K); the verify yields the next anchor | fold the anchor into the **batched** verify (skip the standalone `ds4_session_eval(s, first_token)`); compose with the existing STS scheduled verify + the batched verify | **DONE (lever 2).** Env-gated `DS4_DSPARK_ANCHOR_REUSE` (requires `DS4_DSPARK_VERIFY_BATCHED`); skip the standalone decode when `first_token==argmax(logits)` + batched on; `drafts[0]=first_token`, continuation drafted from the stale window into `drafts[1..]` (Lead 01 de-risked). 8k bench (warm): **+10.5% over batched** (25.97 vs 23.50 t/s), decode 28.2→0.1 ms, verify 104.8→116.9 ms (sublinear), continuation acceptance held (~3.18 vs 3.25). 20-Q no-regression gate PASS (18/20, Q6+Q15 fail; flipped Q9 fail→pass, zero pass→fail). Still 0.73× plain (verify dominates). |
| **GPU drafter body/head** | `draft_ms ≈ 10 ms` (not the current ~45 ms CPU) | move the **already-batched** CPU drafter (3 stages × 5-row `dspark_block_forward_batch`) + the 5× vocab output-head matvec (reuses the target's output weights) + markov + confidence + argmax onto Metal | not yet attempted; one GPU output-head cut line failed `--dspark-schedule-parity` (known trap to clear) |

Out of scope (deferred): re-implementing the STS scheduler; STS re-training on measured
timings; the Lead 08 **fused bit-exact sublinear kernel + margin-guarded fallback** (the path
to verify-level bit-exactness AND committed-output byte-exactness — the batched primitive is
NOT exact: score-neutral interim at temp=0 / TV~0.0104 at temp>0); target hidden-state
precision / Lead 04 (F16-blocked); N=3/4/5/6. The **exact sequential verify is retained** as
the byte-exact (temp=0) / distribution-exact (temp=0, TV=0) fallback (default when the batched
verify is off); it cannot clear +20 %.

## What "validate a lever" means here (M2 methodology, gate re-stated)

A lever is `validated` only when: (1) the implementation exists, env-gated default-off;
(2) **temp=0**: committed-output **score-neutral** on the 92Q benchmark (plain vs DSpark-batched,
both engaging the speculative path via the ds4-eval engagement fix; net +4, 89.1 % same verdict —
NOT byte-identical: 56.6 % token divergence); **temp>0**: distribution similarity measured via the
dist-probe (TV ~0.0104, 0.64 % argmax-flip — NOT exact; the sequential verify is TV=0);
(3) it preserves the lever's own parity contract — draft-token, confidence-logit, scheduled
`verify_n`, accepted-chunk, DSpark window-state parity, and **for verify-path changes,
per-position argmax + correction-token parity** vs the trusted reference; (4) it is
benchmarked with full cycle-cost attribution; (5) the measured economics are compared to the
model's projection and the gap recorded. A lever that preserves committed output but degrades
acceptance, or benchmarks faster-yet-slower-than-projected, is `partial`, not `validated`.

## Measurement methodology (inherits M2 §Measurement methodology + §Iteration protocol)

Two-axis benchmark — final-output correctness (**temp=0 score match** on the 92Q benchmark + **temp>0 distribution similarity** via the dist-probe) AND speculative-economics preservation. Concretely:

1. **temp=0 correctness (score match):** the ds4-eval 92Q score comparison — plain vs DSpark-batched, BOTH engaging the speculative path (the ds4-eval engagement fix routes `--dspark` through `ds4_session_eval_speculative_argmax`; the earlier "10/10 byte-identical" was plain-vs-plain). Report the score (net +4, 89.1 % same verdict) + the pass/fail flips.
2. **temp>0 distribution similarity:** the dist-probe (`DS4_DSPARK_VERIFY_DIST_PROBE`) measures the batched-vs-exact-sequential logit divergence (TV, max_abs, argmax-flip) — logits are temp-independent, so this IS the temp>0 sampled-distribution divergence (TV ~0.0104, 0.64 % argmax-flip). The sequential verify is TV=0.
3. Final-output correctness gates mandatory (the temp=0 score match + the temp>0 distribution similarity above).
4. Draft-side quality checked SEPARATELY from final-output parity (draft-token / confidence-logit / scheduled-verify_n / accepted-chunk / window-state; + per-position argmax + correction-token for verify-path changes).
5. Acceptance metrics first-class (drafted/verify length, accepted/verified per cycle, full-accept rate).
6. Cycle-cost attribution split by component (decode_ms, draft_ms, verify_ms, verify_decode_ms, DSpark state-push time, logits readback — kept separate; batched vs sequential verify distinguished).
7. Same retained exactness / powered-corpus / long-context corpora that feed `spec_speedup_model.md`.
8. `ds4-spec-bench` / `ds4_spec_bench.c` as the default substrate (single loaded engine, bulk config, fresh sessions where needed); warm the model (`--warm-weights`) before timing.
9. Compare BOTH a fixed low-K reference (`verify_k=1`) AND the scheduled path.
10. Each lever env-gated default-off behind its own parity gate; not promoted until it clears the checklist.
11. Model assumptions tested, not assumed — measured cost replaces the model's value; the speedup projection is re-read with the measured number.

Iteration protocol per lever: implement (env-gated) → smoke gates (lever parity + temp=0 score match + temp>0 distribution similarity) → codex gate A → benchmark (ds4-spec-bench, retained corpora, full cycle-cost attribution) → codex gate B → update this lever table → re-read `spec_speedup_model.md`.

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
9. **Batched verify exactness (measured 2026-07-14, NOT exact):** the committing batched verify
   (env `DS4_DSPARK_VERIFY_BATCHED`), when engaged, is NOT committed-output-exact. temp=0:
   56.6 % committed-token divergence on the benchmark (40 % on the exactness corpus, 45/48 on
   code_topk) — the earlier "10/10 ds4-eval byte-identical" was plain-vs-plain (ds4-eval used
   `ds4_session_eval`, not the speculative path; now fixed). BUT score-neutral on the 92Q
   benchmark (plain 60 vs batched 64, net +4, 89.1 % same verdict). temp>0: sampled-distribution
   divergence TV ~0.0104, 0.64 % argmax-flip, max_abs up to 4.56 (dist-probe). The **exact
   sequential verify** is byte-exact (temp=0) / distribution-exact (temp>0, TV=0). Codex verdict:
   NO functional equivalence when engaged. (Strict committed-output byte-exactness needs the
   Lead 08 margin-guarded fallback — deferred.)
10. `decode_ms`/`draft_ms`/`verify_ms`/`verify_decode_ms` + DSpark push + logits readback are all recorded in `s->dspark_last_cycle` (the existing cycle-timing) and emitted by ds4-spec-bench.
11. **Verifier reality (code-confirmed 2026-07-13):** the DSpark **committing** verify is the **sequential exact** loop (`metal_graph_eval_token_raw_swa_top` per token, ds4.c:28714, short-circuits at the first mismatch) — greedy-exact + LINEAR. The **batched** verifier (`metal_graph_verify_suffix_tops`) is SUBLINEAR but not verify-level bit-exact (0.64 % argmax flip); **in the DSpark branch it is called ONLY in the dist-probe (ds4.c:28671, non-committing)** — wiring it for committing is the milestone-3 enabler. `metal_graph_verify_decode2_exact` is exact but K=2-only + linear (MTP path). The `spec_speedup_model`'s `verify_ms(K)` (K2:43.6, K4:65.8) is from the BATCHED `mtp_verifier_bench` — i.e. the model's projections already assumed the sublinear verify this milestone must now actually wire for committing.
12. **Anchor-reuse economics, corrected (2026-07-13):** on the EXACT sequential verify, anchor reuse is **~NEUTRAL** — the anchor's ~28 ms bare decode relocates into the verify at ~28 ms/token (verify 62 → ~90 ms), saving only the readback/dispatch overhead (~1–3 ms). It is worthwhile **only on the sublinear (batched) verify**, where folding the anchor is ~free (verify_batched(3.2)≈57 ms vs decode(28.9)+verify_batched(2.2)≈46 ms). **+20 % requires the batched verify** — with the exact verify, even a free drafter + relocated anchor stays below baseline (`3.2/0.090 = 35.5 t/s < 38.5`).

## Further optimizations surfaced from the M1/M2 review

- **Committing batched verify** (promoted from "deferred" to **lever 1 / the enabler**) — the existing sublinear primitive, wired for committing (**DONE**). Measured: sublinear (20.2 t/s, verify 52→43 ms); NOT committed-output-exact when engaged but **score-neutral on 92Q** (net +4) at temp=0; temp>0 TV ~0.0104, 0.64 % argmax-flip. The verify is NOT "last priority" — it is the dominant cycle cost and the +20 % gate.
- **Persistent device DSpark KV/window state** (M1 next-steps #4) — the GPU drafter port should keep the drafter's KV/window resident on-device (on unified memory the win is compute throughput + eliminating per-cycle GPU↔CPU hidden-readback/draft-push *serialization*, not copy-avoidance). Folded into the GPU-drafter lever.
- **Batched-drafting cap operating point** (M1 cycle 4) — cap3/cap4 as a free scheduling config choice for these measurements (does not change acceptance).
- **Lead 08 fused bit-exact sublinear kernel + margin-guarded fallback** — deferred: the path to verify-level bit-exactness AND committed-output byte-exactness. The batched primitive is NOT exact (diverges 56.6 % temp=0 / TV ~0.0104 temp>0); the margin-guarded fallback (re-verify near-tie positions with the exact path) would restore strict byte-exactness. Not needed for the score-neutral interim.
- **STS re-training on measured timings** — deferred follow-up; only sensible after the levers produce real timings.
- **Target hidden-state precision (Lead 04)** — upstream acceptance work (not runtime); F16-deployment-blocked. Deferred.

## Next steps

**Lever 1 (committing batched verify) DONE + measured.** Implemented (env-gated
`DS4_DSPARK_VERIFY_BATCHED`); codex gate A ran (3 bugs fixed); benched (plain 37.2 / seq 16.3 /
batched 20.2 t/s; verify 52→43 ms, sublinear); exactness characterized (temp=0: 56.6 % divergence
but score-neutral +4 on 92Q; temp>0: TV ~0.0104, 0.64 % argmax-flip); ds4-eval engagement fix
(`--dspark` now routes through the speculative path).

**Next:** levers 2 (anchor reuse, fold into the batched verify) + 3 (GPU drafter), each its
own codex-gate-A → bench → codex-gate-B cycle; then `propagate`. Open: accept the batched
verify as a score-neutral interim (temp=0 +4 / temp>0 TV~0.01) OR add the Lead 08 margin-guarded
fallback for strict committed-output byte-exactness.

## Worklog

### 2026-07-14 — 8k BASELINE refreshed + anchor reuse (lever 2) IMPLEMENTED + benched (+10.5%) + 20-Q gate PASS

**8k baseline (refreshed, warm, `DS4_DSPARK_TIMING=1`, frontier=3072, gen=128, 3 prompts code_8k/synthesis_8k/grounded_8k ~3.9-4.0k tokens):**
plain **35.50** / seq **19.96** / batched **23.50** t/s. seq cycle: decode 29.3 / draft 85.5 / verify 93.0 / total 207.9 ms, verified 3.10, verify_n 4.42.
batched cycle: decode 28.2 / draft 48.0 / verify 104.8 / total 181.1 ms, verified 3.25, verify_n 4.50. The longer context raises
draft+verify (DSpark window + KV grow). draft_ms differs seq vs batched with identical rows_computed (~4.7) — a CPU/GPU timing-overlap
artifact; total_ms is authoritative. **Recorded first-20 no-regression reference** (from the retained 92Q): plain 17/20 + batched 17/20
(identical: Q6/Q9/Q15 FAIL). Artifacts: `dspark_m3_bench/bench8k_{plain_seq,batched}.jsonl`.

**Anchor reuse (lever 2) — implemented** (env `DS4_DSPARK_ANCHOR_REUSE`, default-off; requires `DS4_DSPARK_VERIFY_BATCHED`):
skips the standalone `ds4_session_eval(s, first_token)` when `first_token==sample_argmax(s->logits)` (the precondition, mirroring
`DS4_MTP_ANCHOR_REUSE`) + batched on + max_tokens>1. `drafts[0]=first_token`; the drafter produces the continuation in `drafts[anchor_off..]`
(1 fewer continuation token; drafted from the stale window — Lead 01 de-risked). Verify span = continuation + anchor; `target_top = first_token`
(so `commit_n` starts at 1 — the reject path never fires for reuse). The anchor's hidden is pushed by the batched verify
(`dspark_session_push_batch_hidden(s,0)` on full-accept / the sequential replay's `push_graph_hidden` on partial). 5 edits in
`ds4_session_eval_speculative_argmax` (ds4.c ~28338 + ~28618-28713): env-gate fn + the anchor-decode guard + the offset-aware draft +
STS verify_n + target_top override. Falls back to the standalone decode if the drafter fails or the precondition fails. Smoke:
decode 26.5→0.1 ms (fold fires ~100% of cycles); reuse-without-batched → decode 26.7 (guard correctly does NOT engage).

**8k REUSE bench (warm):** 25.97 t/s vs batched 23.50 → **+10.5%**. decode 28.2→0.1 ms, verify 104.8→116.9 ms (sublinear +12 ms for the
folded anchor), total 181.1→162.2 ms. **Continuation acceptance held** (verified-minus-anchor ~3.18 vs batched 3.25) — on the 8k context
the one-position window staleness is amortized (the short-prompt smoke showed a drop, but 8k does not). Still 0.73× plain (35.5) —
the verify dominates at ~117 ms. Artifact: `dspark_m3_bench/bench8k_anchor_reuse.jsonl`.

**20-Q no-regression gate (ds4-eval, greedy, engaged):** reuse first-20 = **18/20 (Q6, Q15 FAIL)** vs the recorded reference
17/20 (Q6, Q9, Q15 FAIL). The reuse flipped Q9 fail→pass (a gain); **zero recorded-pass flipped to fail** → no regression, **gate PASS**.
(The ds4-eval exits 1 after grading on a teardown fault from the pre-existing 3.51-3.56 GiB model-mapping gap; all 20 verdicts are
complete + valid. The gap is non-fatal during generation — present in the ds4-spec-bench runs too. Clean --questions 5 re-runs exit 0.)

**Read on the +20% target:** the 8k baseline shows the batched verify dominates at ~105-117 ms (verify_n~4.5); reuse+drafter projects
~0.9× plain on exactness, ~0.73× on 8k. The +20% over plain is NOT reachable with levers 2+3 alone — the verify is the bottleneck
(Lead 08 territory). Lever 2 is a real win over the DSpark-batched baseline (+10.5%) + score-neutral on the gate.

### 2026-07-14 — lever 3 parity CODEX review: empty KV (dspark_n_real=0) + pos off-by-one (fixed, didn't help) + the commit lifecycle missing

**Codex review** (gpt-5.5 xhigh, report at `dspark_codex_reviews/2026-07-14_gpt55_xhigh_lever3_parity_codex.md`):
- **H1 (empty KV) — SOUND**: `metal_graph_dspark_refresh_verified_rows`/`_current_row` have NO call sites; the PR calls them on commit + advances `graph.dspark_n_real`. Our port is missing the entire commit lifecycle.
- **H2 (input path) — likely-WRONG as stated**: DSpark uses BOTH `main_proj(hidden40..42)` → `main_x` (the drafter's residual) + the target token embeddings for `[anchor, noise...]` → `batch_cur_hc`. Our CPU drafter ISN'T token-only — its push path computes `main_proj` too (ds4.c:28097). So the input paths are the same.
- **H3 (forward) — SOUND + an off-by-one**: we passed `pos=checkpoint.len` but the PR passes `checkpoint.len - 1`. **FIXED** (didn't help alone — the empty KV dominates).
- **H4 (output-head/markov) — likely-wrong as primary**: check later.
- **The KV fill gap**: the PR's verify captures the accepted hiddens into `dspark_verify_hidden` → the commit refresh projects them into the drafter KV (`refresh_verified_rows`) → advances `dspark_n_real`. Missing locally: the refresh is unused; the verify captures into `dspark_batch_capture_hc` (not `dspark_verify_hidden`); the commit only updates the CPU state.
- **Verified**: `dspark.gguf` metadata — `target_layer_ids=[40,41,42]`, `noise_token_id=128799` (< DS4_N_VOCAB=129280, valid), `main_proj=[12288,4096] Q8_0`.

**Ranked leads**: (1) wire the commit lifecycle (fill dspark_verify_hidden, call refresh, advance dspark_n_real) — medium; (2) fix pos (DONE); (3) unify the verified-hidden source (port the PR's batch capture OR adapt the refresh to consume dspark_batch_capture_hc) — small-medium; (4) build a parity harness (same state, compare Metal vs CPU non-scheduled) — medium; (5) inspect output-head/markov later.

**Status**: the pos fix (committed) didn't change the result (still verified=1, t/s=17.4). The first draft is ALSO wrong (the codex said it should work with n_real=0 — so there's an issue beyond the KV: the forward, the output-head, OR the main_x/batch_cur_hc flow). Next: wire the commit lifecycle (leads 1+3) + trace the forward's use of dspark_main_x + build the parity harness (lead 4).

### 2026-07-14 — lever 3 BREAKTHROUGH: Metal drafter ENGAGES (draft=6.9ms vs CPU 45ms) but drafts are WRONG (parity issue)

**Fixed the hidden-flow + the dspark model registration — the Metal drafter now ENGAGES** (committed `2005696`):
- **The capture** (`metal_graph_capture_dspark_metal_main_hidden`): ported the PR's `metal_graph_capture_dspark_main_hidden` — a `ds4_gpu_hc_weighted_sum_tensor` (hc→embd via `dspark_mean_weights`) for the 3 target layers (40/41/42). Wired into `metal_graph_encode_token_raw_swa` (the anchor decode, after the cur_hc swap — uses `cur_hc` = the post-FFN hc), NOT the verify (the PR captures in the anchor decode, not the batched verify).
- **The dspark model registration**: the `ds4_gpu_wrap_model_range` (the matmul's weight lookup) only finds registered `g_model_views` — the dspark model wasn't registered (only the target). Fixed: exposed `ds4_gpu_add_model_view_range` + called it for the dspark model after `model_open` (coexisting with the target's views, no clear) + `ds4_gpu_init()` before it (the device wasn't init at the engine-init stage → `maxBufferLength=0`).
- **Anchor-reuse disabled when Metal is on**: the Metal drafter needs the anchor decode (the capture), but the anchor-reuse skips it (chicken-and-egg). So `anchor_reuse = ... && !dspark_draft_metal_enabled()`.

**Result (smoke, short prompt):** the Metal drafter ENGAGES — `drafted=4.25`, **`draft=6.9 ms`** (vs the CPU's ~45 ms — the lever-3 draft reduction is real!), `cycles=12` (multiple tokens/cycle). BUT `verified=1` (low acceptance) → **the drafts are WRONG** (the Metal drafter's output doesn't match the target). The t/s=17.4 (low — the wrong drafts → rejections → the verify overhead).

**Next (the parity debugging):** the Metal drafter produces wrong drafts. The likely cause is in the capture (the hc→embd reduction, the timing), the main_proj matmul, the drafter's forward (the noncausal attention, the KV cache), the output-head, OR the markov bias. The drafter parity gate (draft-token + conf-logit vs the CPU reference, one-time) is the check. Need a focused comparison: run the Metal + the CPU drafter on the same input + diff the drafts.

### 2026-07-14 — lever 3 (GPU drafter) PORT + WIRE done; runs but FAILS at input_stage (hidden-flow gap)

**Ported + wired (builds clean, the branch fires):**
- **Graph extension** (4b94392): added the GPU drafter state — `dspark_main_x`, `dspark_kv_cache[DS4_DSPARK_STAGES]`, `dspark_verify_hidden`, `dspark_verify_main_x`, `dspark_n_real`, + `dspark_metal_main_hidden` (GPU; our `dspark_main_hidden` is a CPU `float*` capture buffer — kept separate to avoid the type conflict). Struct + alloc + free.
- **Noncausal batch attention** (9b39b38, `ds4_metal.m`): ported `ds4_gpu_fill_raw_decode_batch_all_mask` + the `noncausal` flag on `ds4_gpu_encode_flash_attention_decode_raw_batch_heads` (the 2 causal callers pass `false`) + the `ds4_gpu_attention_decode_raw_batch_heads_noncausal_tensor` wrapper. The drafter's prerequisite (every draft query sees all raw KV). The ONLY missing GPU fn — all other deps (`ds4_gpu_matmul_q8_0_tensor`, `metal_graph_encode_layer_ffn_batch`, etc.) our branch already had; no new `ds4_metal.m` shaders.
- **The 6 `metal_graph_dspark_*` functions** (be211a3, ~715 lines): ported `metal_graph_eval_dspark_draft_block` + `metal_graph_dspark_input_stage` + `metal_graph_dspark_encode_attention` + `metal_graph_dspark_refresh_main_rows`/`_verified_rows` + `metal_graph_encode_output_head_dspark_batch` + the `dspark_apply_markov_bias` helpers, adapted to our `ds4_dspark_weights` (`mtp->stage[s]`→`block[s]`; `mtp->main_proj`/`main_norm`/`norm`/`hc_head_*`→our fields; `mtp->dspark.block_size`→`DS4_DSPARK_BLOCK`; `mtp->dspark.n_mtp_layers`→`DS4_DSPARK_STAGES`; `mtp->dspark.noise_token_id`→`128799`; `g->dspark_main_hidden`→`g->dspark_metal_main_hidden`).
- **Wired** (aa47b66, env `DS4_DSPARK_DRAFT_METAL`): replaces the CPU drafter in `ds4_session_eval_speculative_argmax` (the Metal branch drafts the continuation into `drafts+anchor_off`, composing with anchor-reuse); fixed `verify_n = draft_eval_n + anchor_off` (no STS — the Metal drafter doesn't set `dspark_conf_logits`; the STS composition needs the conf, a follow-up).

**The Metal drafter RUNS (draft_eval_n=4, the branch fires) but FAILS at `metal_graph_dspark_input_stage`.** Root cause: `dspark_metal_main_hidden` (the target's 3-layer hiddens, which the input projection consumes) is allocated but **never filled**. **Hidden-flow gap:** the PR fills it via the GPU verify-capture (the target's verify writes the post-FFN **embd** into the GPU buffer); our branch captures the target's hiddens on the **CPU** (`dspark_capture_hc`, the post-attention **hc** = `DS4_N_HC × DS4_N_EMBD`, a different type + location). So the Metal drafter's input projection reads an empty/wrong buffer.

**Next (the hidden-flow reconciliation):** either (a) capture the target's **embd** on the GPU (port the PR's GPU capture into our verify — the target's verify writes the embd to `dspark_metal_main_hidden`), or (b) upload+convert our CPU `dspark_capture_hc` (hc) to the GPU `dspark_metal_main_hidden` (embd) before the Metal drafter (a CPU→GPU copy + the hc→embd extraction). (a) is cleaner (matches the PR) but touches the verify; (b) is a smaller adapter. Then: drafter parity + 20-Q gate + codex A → large-corpus bench → codex B.

### 2026-07-14 — lever 3 (GPU drafter) ORIENTATION + graph extension done; port plan

**Orientation (PR #502 `metal_graph_dspark_*`):** the Metal drafter is ~509 lines (`metal_graph_eval_dspark_draft_block` + `metal_graph_dspark_input_stage` + `metal_graph_dspark_encode_attention` + `metal_graph_dspark_refresh_main_rows`/`_verified_rows` + `metal_graph_encode_output_head_dspark_batch`) + it **reuses the existing Metal kernels** (the PR's `ds4_metal.m` has only 1 dspark ref — the drafter is built from `ds4_gpu_matmul_q8_0_tensor` / `ds4_gpu_rms_norm_weight_tensor` / `metal_graph_encode_layer_ffn_batch` / `metal_graph_upload_prompt_tokens`, all present in our branch). So **no new `ds4_metal.m` shaders** — the port is C-side.

**Our branch already has** most deps: `metal_graph_upload_prompt_tokens`, `ds4_gpu_matmul_q8_0_tensor`, `metal_graph_encode_layer_ffn_batch`, + the GPU buffers `dspark_mean_weights` + `dspark_stage_kv` (partial prior state). **Missing graph buffers (ADDED this slice, builds clean):** `dspark_main_x`, `dspark_kv_cache[DS4_DSPARK_STAGES]`, `dspark_verify_hidden`, `dspark_verify_main_x`, `dspark_n_real`, + `dspark_metal_main_hidden` (GPU; our `dspark_main_hidden` is a CPU `float*` capture buffer — kept separate to avoid the type conflict). Struct + alloc + free wired. Committed (this slice).

**Weights adaptation (our `ds4_dspark_weights` vs the PR's `ds4_mtp_weights`):** `mtp->stage[s]`→`dspark_weights.block[s]`; `mtp->main_proj`/`main_norm`/`norm`/`hc_head_*`→our fields (present); `mtp->dspark.block_size`→`DS4_DSPARK_BLOCK`; `mtp->dspark.n_mtp_layers`→`DS4_DSPARK_STAGES`; `mtp->dspark.noise_token_id`→a constant/lookup; `mtp->kind`→`DS4_MTP_DRAFT_DSPARK`. Mechanical.

**Port plan (remaining):**
1. Port the 6 `metal_graph_dspark_*` functions (adapted to `ds4_dspark_weights` + the `dspark_metal_main_hidden` rename) — ~509 lines.
2. Wire into `ds4_session_eval_speculative_argmax`: env-gated `DS4_DSPARK_DRAFT_METAL` (off=CPU drafter, on=Metal drafter); the CPU drafter stays as the one-time parity reference.
3. Drafter parity (draft-token + confidence-logit vs CPU, one-time) + 20-question no-regression gate.
4. codex A → large-corpus bench (ds4-spec-bench) → codex B.

**Projection:** the draft is **memory-bound** (`-t 8` vs `-t 14` flat) → the GPU drafter (shared unified memory) is marginal (~1.5–2× → draft 45→~25 ms). The full stack (reuse+prefix-checkpoint+Metal drafter) on 8k → ~39 t/s ≈ **1.1× of plain (+10%)** — the closest to +20%, but not there. On the short corpus, less (the verify dominates). refresh_verified_rows (the GPU fast-commit, replaces the CPU push) ports with lever 3 (it refreshes the drafter's GPU state).

### 2026-07-14 — verifier-improvements (prefix-checkpoint) IMPLEMENTED + codex gate A (stale-KV bug FIXED) + 20-Q gate EXACT match + large-corpus bench (+26.8% over reuse-alone; context-dependent vs plain)

**Prefix-checkpoint implemented** (env `DS4_DSPARK_VERIFY_PREFIX_CHECKPOINT`, default-off): generalized our EXISTING prefix-1 infra (`spec_prefix1_attn_state_kv` + `metal_graph_capture_prefix1_attn_state` + `spec_frontier_commit_prefix1`, single slot for n_tokens==2) to `block_size-1` slots. New: multi-slot buffers (`spec_prefix_attn_state_kv`, slots×bytes) + `metal_graph_capture_prefix_attn_state(g,il,prefix_len)` (slot=prefix_len-1, a `ds4_gpu_tensor_copy` — no custom kernel) + `spec_frontier_commit_prefix(s,accepted,draft_n)` (slot=accepted-1) + the verify hook (capture all prefix_len during `metal_graph_encode_layer_attention_batch`) + the wiring (partial-accept uses `spec_frontier_commit_prefix` instead of the sequential replay, env-gated). Committed f631fcc.

**Codex gate A (gpt-5.5 xhigh) found a REAL correctness bug** (report at `dspark_codex_reviews/2026-07-14_gpt55_xhigh_prefix_checkpoint_gateA.md`): the prefix capture is hooked only in the PER-TOKEN compressor path; the ALIGNED ratio-4 path (`pos0%4==0 && n_tokens%4==0`, verify_n=4) never captures → the prefix-commit restored STALE/uninitialized ratio-4 slots → potential wrong-token commits. **FIXED**: added `spec_prefix_capture_valid` flag (set false in the aligned branches 18291/18608; the prefix-commit checks it → falls to the replay when aligned → no stale KV). (C6's "no re-verify" is the batched verify's known score-neutral divergence — acceptable per the goal's frame.) Re-benched + re-gated after the fix.

**8k bench (warm, 3 prompts):** reuse+prefix-ckp **34.45 t/s** (verify 82.4) vs reuse-alone 25.97 → **+33%**; vs plain 35.5 → **0.97× of plain** (verify dropped 117→82 ms; the prefix-checkpoint avoided the partial-accept replay). Default-off unchanged (27.39 1-prompt ~ the retained 25.97; coff is 1-2 so the multi-slot buffers are tiny — no memory pressure).

**20-Q no-regression gate (re-run with the fix):** reuse+prefix-ckp = **17/20 (Q6, Q9, Q15 FAIL) — EXACT match to the recorded reference** (plain+batched first-20). The fix made the aligned cycles use the exact replay → Q9 matches the reference (pre-fix was 18/20 with Q9 diverged to pass). ZERO recorded-pass→fail. GATE PASSES (the best possible — exact match).

**Large-corpus bench (300-prompt corpus, n=93 matched, via ds4-spec-bench, bootstrap CI):** plain **38.20** [38.12,38.29] / reuse-alone **22.04** [21.21,22.88] (verify 83.9) / reuse+prefix-ckp **27.95** [27.00,28.86] (verify 60.9). **Prefix-checkpoint = +26.8% over reuse-alone** (verify 83.9→60.9 ms). But vs plain = **0.73×** (DSpark slower on the SHORT-prompt corpus). **Context-dependent:** 8k (long, decode ~28 ms) → 0.97× of plain (DSpark amortizes); 300-prompt (short, ~25 tok, decode ~26 ms) → 0.73× (the draft+verify overhead exceeds the amortization at verified~2.56). Consistent with the spec_speedup_model.md headline ("DSpark does NOT beat plain ds4 decode locally on the measured corpus") + the PR #502 on-hardware results (workload-dependent). (n=93 is a subset — 83 prompts errored on frontier>actual (tok_est overestimates); the 3 configs are matched on the same 93, so the relative speedup is apples-to-apples.)

**Read on the +20% target:** the prefix-checkpoint is a real win over the DSpark-batched+reuse baseline (+26.8% on the large corpus, +33% on 8k) + the 20-Q gate is an EXACT match. But vs plain, DSpark is still slower on short contexts (0.73×) + ~baseline on long (0.97×). The +20% over plain is NOT reached on the general corpus — the verify + draft overhead exceeds the amortization on short contexts. (The `refresh_verified_rows` GPU fast-commit is deferred to lever 3.)

**Codex gate B (gpt-5.5 high) — CONFIRMS the fix + the bench** (report at `dspark_codex_reviews/2026-07-14_gpt55_high_prefix_checkpoint_gateB.md`): F1/F3 sound (the flag is reset per-verify, cleared in both aligned branches, gates the commit — no cross-cycle leakage); F2 caveat (zero-prefix prefill path skips capture but isn't reachable from normal DSpark verify — theoretical). B1 sound (the verify_ms drop is the replay-elimination signature: partial cycles 99.3→61.6ms, full cycles flat 58→60 — NOT fewer verifies). **B3 caveat (fair):** the n=93 matched subset is BIASED SHORT (mean 62 tokens vs 163 unmatched) — the 0.73× vs plain is the "short-context retained subset," NOT the full 300-prompt corpus. The +26.8% over reuse-alone is valid (matched, apples-to-apples). Decisive test (not run): per-cycle counters (aligned_invalidated / prefix_commit_taken / replay_taken) under VERIFY_K=4.

### 2026-07-14 — GOAL TWEAK applied: add verifier-improvements lever (prefix-checkpoint) before lever 3; demote +20% to "report the actual"; large-corpus bench via ds4-spec-bench

**Tweak:** added a **verifier-side-improvements** lever (port the PR #502 prefix-checkpoint), ordered **before** lever 3 (the Metal drafter). M3's focus reframed to **measuring actuals on the best-possible practical implementation + re-assessing other leads — NOT clearing the +20% gate** (the verify dominates on Apple Silicon; +20% likely needs Lead 08). Bench must use a **sufficiently large prompt corpus** (spec_speedup_model.md-comparable fidelity) via **ds4-spec-bench** (not ds4 loops). Post-measurement lead re-assessment (Lead 08 etc.) added to the final task. `--dspark-schedule-parity` dropped (drafter parity instead).

**refresh_verified_rows is drafter-side (deferred to lever 3):** on reading `metal_graph_dspark_refresh_verified_rows` (PR ds4.c:20228) it delegates to `metal_graph_dspark_refresh_main_rows` with `g->dspark_verify_hidden`/`g->dspark_verify_main_x` + the **draft model** — so it refreshes the DRAFTER's GPU state, NOT the target verify. It's coupled to the Metal drafter (lever 3) + can't replace our CPU `dspark_session_push_batch_hidden` standalone (our drafter is CPU). **Confirmed with the user (goal_question):** verifier-improvements = prefix-checkpoint only (standalone); refresh_verified_rows ports with lever 3.

**Verifier-improvements = the prefix-checkpoint — and our codebase ALREADY has the prefix-1 infra to generalize:**
- `spec_prefix1_attn_state_kv[DS4_MAX_LAYER]` + `spec_prefix1_attn_state_score` + `spec_prefix1_index_state_{kv,score}` + `spec_prefix1_n_comp` + `spec_capture_prefix1` (ds4.c:10584-10593) — a SINGLE-slot prefix capture.
- `metal_graph_capture_prefix1_attn_state(g, il)` (13547) — copies `layer_attn_state_kv[il]` into the single slot (a `ds4_gpu_tensor_copy`, NOT a custom kernel).
- `spec_frontier_commit_prefix1(s)` (24775) — restores from the single slot.
- Hooked in the verify only for `n_tokens == 2` (21690-21726).
- The PR's prefix-checkpoint generalizes this to `block_size - 1` slots: `spec_prefix_attn_state_kv` (slots × layer_attn_state_bytes), `metal_graph_capture_prefix_attn_state(g, il, prefix_len)` (slot = prefix_len - 1), `spec_frontier_commit_prefix(s, accepted, draft_n)` (slot = accepted - 1), capture hooked for ALL prefix lengths during the verify.

**Port plan (verifier-improvements = generalize prefix-1 → multi-slot prefix-checkpoint):**
1. Graph struct: replace the single-slot `spec_prefix1_*` buffers with multi-slot `spec_prefix_*` (slots × bytes) + `spec_prefix_slots` + `spec_prefix_n_comp[slot][il]` + `spec_capture_prefix_tokens`. Alloc in the graph init.
2. Capture: `metal_graph_capture_prefix_attn_state(g, il, prefix_len)` — `ds4_gpu_tensor_copy` into slot (prefix_len-1).
3. Commit: `spec_frontier_commit_prefix(s, accepted, draft_n)` — restore from slot (accepted-1); replaces the general-partial sequential replay.
4. Verify hook: capture for all prefix_len (1..n_tokens-1) during `metal_graph_verify_suffix_tops` (not just n_tokens==2).
5. Wiring: in our committing batched verify, replace the general-partial replay (the `else if (spec_frontier_restore(...))` sequential replay branch) with `spec_frontier_commit_prefix(s, commit_n, verify_n)`. Keep the full-accept + prefix-1 paths.
6. Env-gate (extend `DS4_DSPARK_VERIFY_PREFIX_CHECKPOINT` or similar); default-off.
**Expected win:** ~12 ms (the partial-accept replay saved) on top of the 117 ms verify; stacks with the STS verify_n. Gate: 20-question no-regression (the commit path changes — verify parity vs the replay).

### 2026-07-14 — lever 3 VIABILITY FINDING: draft is MEMORY-BOUND; GPU forward port is marginal on Apple Silicon (shared unified memory); +20% needs Lead 08

**Batched output-head (CPU, DS4_DSPARK_OUTPUT_BATCHED) implemented + benched — MINOR win (+1.3%):** replaced the `draft_n` separate `matvec_q8_0` output-head calls with one `matmul_q8_0_batch` (reads the ~917 MB output weights once). 8k reuse+OB: draft 45.2→42.1 ms, t/s 25.97→26.31 (+1.3%). **The output-head is only ~3-5 ms** — NOT the dominant cost. My earlier "output-head dominates" reasoning was WRONG.

**The 3-stage forward dominates (~38-40 ms):** the drafter's ~10.7 GB weights (read once per cycle) are the bulk of the draft. Confirmed by the batched output-head experiment (the output-head is small) + the thread-count diagnostic.

**Thread-count diagnostic (decisive):** 8k reuse at `-t 8` vs `-t 14` → draft 38.4 vs 38.3 ms (**FLAT**). More CPU threads do NOT speed up the draft → the forward is **memory-bound** (bandwidth-limited, not compute-limited).

**Implication for the GPU port (lever 3):** on Apple Silicon the CPU + GPU share the unified memory pool. A GPU forward port draws from the SAME bandwidth → at best ~1.5× on the forward (draft ~25-30 ms), NOT the ~10 ms target. The GPU drafter is **marginal + big-effort** (a Metal implementation of the drafter's full transformer, or reusing the target's Metal graph for the drafter's weights) for a ~0.79×-of-plain result.

**Combined projection (levers 2+3):** reuse (0.73× plain) + GPU forward (marginal) ≈ 0.79× plain. The verify (~117 ms) dominates. **+20% over plain is NOT reachable with levers 2+3 on this hardware** — it needs Lead 08 (the fused verify kernel / verify-cost reduction), which is deferred.

**Decision surfaced (paused):** (a) skip lever 3 (not viable as specified — memory-bound on Apple Silicon) + run the final 92Q (plain vs batched+reuse) + propagate, reporting ~0.73× plain + the verify-dominates finding; (b) attempt the GPU forward port anyway (marginal, big effort, "report the actual"); (c) pivot to Lead 08 (the real +20% path). The batched output-head (CPU, +1.3%) is kept as a minor env-gated optimization regardless.

### 2026-07-14 — lever 3 (GPU drafter) ORIENTATION: output-head dominates (~95%); CPU batched output-head is a safer alternative to the GPU port

**Drafter cost breakdown (reasoned + scoped):** the draft_ms ~45 ms (8k) is dominated by the **output-head matvec** — the loop does `draft_n` SEPARATE `matvec_q8_0(base_logits, &e->model, e->weights.output, norm)` calls, each reading the target's full-vocab Q8_0 output weights (~7168×128k ≈ 917 MB). At ~100 GB/s, 5 reads ≈ 43 ms ≈ 95% of the draft. The 3-stage `dspark_block_forward_batch` forward is small (~2 ms); the markov/conf/argmax are small. So **moving the output-head is the lever-3 win**; the forward need not move.

**No `--dspark-schedule-parity` flag exists** in the current tree — the prior GPU-head cut line was removed/reverted, so lever 3 starts fresh. There is **no standalone Metal output-matvec** to reuse (the target computes logits inside the full Metal graph forward); a GPU output-head port needs a new Metal Q8_0 kernel + host glue (upload `norm`, run, download `base_logits`) + a Q8_0 GPU-vs-CPU parity check — the risk the objective flags.

**Safer alternative identified (CPU batched output-head):** `matmul_q8_0_batch` (ds4.c:5373, already used by the drafter forward) can replace the `draft_n` separate `matvec_q8_0` calls with ONE batched matmul (norms[draft_n×n_embd] × output → base_logits[draft_n×vocab]), reading the 917 MB weights **once** → ~5× less memory traffic → output-head ~9 ms (vs ~43 ms) → draft_ms ~11 ms (forward 2 + head 9 + markov/argmax ~0.5). This hits the lever-3 draft_ms target (~10 ms) **on the CPU, no GPU, no Q8_0 parity trap**. The markov (sequential, depends on prev) + conf + argmax stay per-row; only the output matvec is batched.

**Decision point (surfaced to the user):** the objective specifies the GPU port ("onto Metal", "GPU-resident"), but the CPU batched output-head achieves the same draft_ms target with far less risk + is a natural prerequisite for a later GPU port (batch first, then move the batched matmul to Metal). The verify still dominates (~117 ms), so either way levers 2+3 reach ~0.9× plain, NOT +20% (Lead 08 territory).

### 2026-07-14 — anchor reuse (lever 2): codex gate A DONE (gpt-5.5 xhigh) — sound + 1 corner-case bug FIXED (VERIFY_K=0)

**Codex gate A** (review of the fold logic; report at `dspark_codex_reviews/2026-07-14_gpt55_xhigh_anchor_reuse_gateA.md`):
- C1 (guard), C2 (drafts bounds), C4 (window push), C6 (verify_n bounds), C7 (drafter-fail fallback): **sound.** The window-consistency trace confirmed the anchor hidden IS pushed in all 4 batched-verify outcomes (full-accept / prefix-1 / general-partial / reject) + the sequential fallthrough — no missing-hidden leak.
- C3 (commit_n=1): sound, with **one corner-case bug**: `DS4_DSPARK_VERIFY_K=0` + reuse returned 0 tokens (the reuse guard skipped the standalone decode, but VERIFY_K=0 disables the verify that would fold the anchor). **FIXED**: the `fixed_verify_n==0` early-exit now does the standalone decode + commits the anchor when `anchor_reuse` (smoke: VERIFY_K=0+reuse now emits 24 tokens, not 0).
- C5 (the 20-Q gate): "questionable" — fairly noted that the Q9 fail→pass flip is **divergence-driven luck** (the batched verify itself is divergent-but-score-neutral), NOT an exactness/improvement signal. The gate (no recorded-pass flipped to fail) holds; the reuse's correctness frame is the SAME as the batched verify (divergent, score-neutral), not byte-exactness. The 8k bench's verified-minus-anchor ~3.18 vs batched 3.25 already shows the window is consistent — a missing anchor hidden would crash acceptance, not hold it.
- Verdict: anchor reuse is structurally window-safe. The reuse ds4-eval first-20 trace is retained (20 verdicts: 18 PASS / Q6+Q15 FAIL; the run's exit-1 is the pre-existing model-mapping-gap teardown fault, post-grading).

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
the batched (divergent) verify at temp>0. **Measured the batched verify's temp>0 distribution
divergence via the dist-probe** (`DS4_DSPARK_VERIFY_DIST_PROBE`, fresh run 2026-07-14, exactness
corpus; artifact `issue468/artifacts/dspark_m3_bench/distprobe_batched_vs_exact_fresh.jsonl`):
90 cycles / 156 compared positions — **TV ~0.0104 (mean), argmax-flip 0.64%, max_abs up to 4.56**
(KL~0.0025). So the batched verify is NOT distribution-exact at temp>0 (small but non-zero
divergence), vs the sequential verify's TV=0. (The milestone-1 parity test can't measure this — it
forces VERIFY_K=0, so the verify never runs; the dist-probe is the analog that compares the batched
vs exact logits. Logits are temp-independent, so this IS the temp>0 sampled-distribution divergence.)

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
