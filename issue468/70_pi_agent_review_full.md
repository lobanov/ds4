# Independent pi-Agent Review — DSpark Perf-Gate Recovery Leads

Date: 2026-07-01. Thirtieth productionization handoff note. Fulfills the
`pi-agent-review-fallback` task contract: independent, fresh-eyes review of ALL
issue468/32–69 + the codebase, with leads **independently verified by running**
(not read-only), viable ones **implemented** (B2/MTP/baseline-safe), and outcomes
recorded. Supersedes the earlier read-only draft of this note. Doc-only research
record; the one code change (env-gated n_real cap) is committed to `dspark`.

---

## 0. TL;DR

1. **One real, implemented, B2-exactness-safe win: the `dspark_n_real` window cap.**
   The drafter's non-causal attention over a growing historical-anchor window
   (n_real → 128) **collapses** draft acceptance on long generations (committed/
   cycle 4.11 → 2.89, full-accept 27 % → 0 % across a 384-token run). Note 40
   found this but KEPT the growing-window "fix" because it matched the 19-step
   probe — which never exposed large-n_real degradation. Implemented
   `DS4_DSPARK_NREAL_CAP` (ds4.c:~30146). Measured +0 to +6 t/s (prompt-dependent;
   catastrophic-degeneration cases recover most). Env-gated, default unchanged.

2. **The research record's single headline number is not the whole picture.**
   Live t/s is **highly prompt-dependent** (≈19–36 t/s across prompts) and has
   **±3–5 t/s run-to-run variance** (M5 Max thermal). The "29.7 t/s" / "3.66
   committed" framing is roughly typical for prose, but a code prompt sustains
   ≈31–32 t/s (cap=5) and hits 36.7 t/s early. No prompt/config tested crosses 39.

3. **The dominant acceptance gap is live-vs-probe (0.71), NOT probe-vs-oracle
   (0.12).** Notes 53–69 (the entire MoE/precision bisection) chased the SMALL
   0.12 gap (4.37→4.49). The BIG gap (live 3.66 vs probe 4.37) is the
   probe-vs-live discrepancy (batch-p, live main_hidden, window fill) — the n_real
   cap addresses the window-fill piece of it.

4. **Verdict: the perf gate (DSpark > 39) is NOT crossed by any in-scope safe
   lever.** Best sustained ≈31–33 t/s vs flat 38.6 baseline. The ONE identified
   path that *could* cross — eliminating the partial-accept correction decode
   (~26 ms × ~80 % of cycles) via true batch↔decode KV compatibility (the P0
   lever P0 only half-finished) — is high-risk and, combined with the cap, lands
   at ≈38–40 t/s (within noise of the gate, not safely above it). With the 10th
   blocker precondition now satisfied, this is a defensible structural negative
   result. **Recommend the user decide between (A) accept the perf-blocker and
   ship the quality-neutral implementation, or (B) authorize the high-risk
   correction-decode / custom-verifier work, or (C) an out-of-scope lever
   (draft trees / un-freeze drafter).**

---

## 1. Method

- Read issue468/32–69 in full + ds4.c (drafter forward, `ds4_session_eval_dspark_b2`,
  exact-Q4 path), ds4_metal.m, ds4_cli.c, ds4_eval.c, gguf tensor types.
- **Ran** the live B2 path with `DS4_DSPARK_B2_DEBUG=1` to get the REAL cycle
  breakdown on the current build (not the assumed 108/130 ms from old notes).
- **Ran** baseline + DSpark sweeps at ctx 8192/16384/32768, multiple prompts,
  n=96–384, to characterize prompt/variance behavior.
- Confirmed the on-disk `dspark.gguf` is the **Q4_K** config (mtp.2 routed experts
  q4_k; 81 tensors), i.e. the good 4.37-prefix config — the Q8_0 re-quant
  (issue468/69) was reverted.

## 2. The REAL cycle breakdown (measured, current build, ctx=8192)

From `DS4_DSPARK_B2_DEBUG` per-cycle lines:

| phase | ms | when |
|---|---|---|
| anchor decode (`ds4_session_eval` + capture readback/mean/upload) | ~26.5 | every cycle |
| drafter forward (input+3 blocks+head+Markov) | ~9 | every cycle |
| verify (`metal_graph_verify_suffix_tops`, 5-pos batch) | ~70 | every cycle |
| B2 accept (CPU softmax over vocab ×2 ×5) | ~2 | every cycle |
| correction decode (`ds4_session_eval`, labelled `kv`) | ~27 | partial-accept (~80 %) |

- Full-accept cycle ≈ **108 ms**, 6 committed → 18 ms/tok (efficient).
- Partial-accept cycle ≈ **133 ms**, committed 2–5.

**Reconciliation with note 32:** note 32's "+48–60 %, 83 ms cycle" projection
OMITTED the anchor (26 ms) and correction (27 ms) single-token forwards — it
modeled only drafter + verify. The real cycle is ≈125–132 ms. Note 42 §2.3
correctly identified the "3-forward cycle" as the handicap; this note confirms it
empirically. **Consequence:** the note-32 break-even (committed 3.24 crosses) is
wrong; the real break-even committed at a 132 ms cycle is **5.08** — and the real
break-even cycle at committed 4.1 is **~105 ms**.

## 3. The three-way acceptance gap (the key reframe)

| metric | value | protocol |
|---|---|---|
| Live B2 committed/cycle | **3.66** (default) / **4.1** (cap=5) | Metal drafter, in-GPU main_hidden, batch verify p, B2 |
| Probe avg prefix | **4.37** | Metal drafter, disk main_hidden, sequential greedy match, growing window |
| Oracle avg prefix | **4.49** | numpy F32, same Q4_K weights |
| Oracle MC-B2 ceiling | **+8.27 %** vs Metal | issue468/51 |

The **dominant** gap is **live ↔ probe (0.71 committed)**, NOT probe ↔ oracle
(0.12). Notes 53–69 (MoE bisection, exact-Q4, Q8_0, hc_pre) chased the 0.12
precision gap. The 0.71 live gap decomposes (per note 70 §1.3, which I confirm):
(a) batch verify p ≠ sequential p at positions 1–4; (b) live in-GPU main_hidden;
(c) **window fill (n_real grows in live → ~70–128; probe caps at ~19)**.

**(c) is the piece I verified is real, severe, and fixable.**

## 4. LEAD #1 (implemented): `dspark_n_real` window cap

### What it is
`g->dspark_n_real` grows 1 → 128 across a live generation (the Bug #1 "fix"). The
drafter's non-causal block attention attends over `[0..n_real+block]` historical
anchors. As n_real → 128, draft acceptance collapses.

### Evidence (measured this review, ctx=8192, n=384, default cap=127)
Per-third-of-generation trajectory:

| third | full-accept | zero-accept | committed/cycle | t/s |
|---|---|---|---|---|
| first (n_real small) | 27 % | 10 | 4.11 | 32.5 |
| mid | 3 % | 13 | 3.27 | 24.4 |
| last (n_real large) | **0 %** | 19 | 2.89 | 21.6 |

This is unambiguous (far outside the ±3–5 t/s run-to-run noise). 42/112 cycles
were zero-accept (drafter's first draft rejected) late in generation.

### Why it was set aside (and why that was wrong)
Note 40 measured pre-fix (stale n_real=1) at 4.06 commits/31.5 t/s vs post-fix
3.66/28.6, but KEPT the growing-window fix because it "restores the documented
research intent" and matches the **19-step** probe (n_real ≤ 19). **The probe
never exercises n_real > 19, so it cannot see the collapse that only the live
multi-cycle path reaches.** The "intent" argument is documentary, not
perf-grounded; the live measurement is authoritative and contradicts it. Note 70
(correctly) flagged this as its Lead A/E but did not implement or run it.

### B2-exactness argument (why it's safe)
B2 rejection sampling samples from the **target** p regardless of the drafter q
(`accept w.p. min(1,p/q)`; correction `argmax(p-q)`). Changing the drafter's
window (and thus q) changes **efficiency only**, never the output distribution.
MTP/plain are untouched (separate code path). Baseline untouched. Default
(no env) is byte-identical to prior behavior.

### Implementation
`DS4_DSPARK_NREAL_CAP` env gate at ds4.c:~30146 (replaces the hard-coded
`DS4_N_SWA - 1u` cap). Range 1..127; default unchanged.

### Measured outcome (ctx=8192, n=384, Fibonacci prompt — favorable case)
| cap | t/s |
|---|---|
| 1 (stale) | 29.6 |
| 5 | **31.9** ← peak |
| 13 | 30.1 |
| 31 | 27.1 |
| 127 (default) | 25.7 |

Multi-prompt (n=300): cap helps significantly on prompts that degenerate
(robot story: +5 t/s), is neutral-to-slightly-negative on prompts that don't
(gradient descent: −1). Because of ±3–5 t/s variance, only the
catastrophic-degeneration cases give a reliable signal. **Net: a real option for
long generations; not a universal default.** Kept env-gated; documented.

### Does it cross the gate? **No.** Best sustained ≈31–33 t/s < 39.

## 5. LEAD #2 (identified, high-risk, NOT implemented): eliminate the correction decode

### What it is
On partial accept (~80 % of cycles), the cycle pays a 4th forward — the
correction decode (~27 ms) — to (a) advance KV to the correction position in
**decode-path format** and (b) set `s->logits` for the next sample. P0
(issue468/40) already cut the prior O(k+1) replay to this single O(1) decode, but
did NOT achieve true batch↔decode KV compatibility, so the single correction
decode remains.

### Why it could cross (the math)
At cap=5: committed ≈4.1, cycle ≈132 ms → 31 t/s. Eliminating the correction
(27 ms × 0.80 ≈ 22 ms/cycle avg): cycle → ~110 ms → **4.1/0.110 ≈ 37 t/s**
(with cap). Combined with any small acceptance uptick → ~38–40 t/s, brushing the
gate. This is the only in-scope lever with a plausible crossing trajectory.

### Why I did NOT implement it
- It requires the deep batch↔decode KV-compatibility bridge that P0 explicitly
  struggled with ("compressed KV cache overflow", issue468/32 §5). High
  correctness risk to the **target** decode path (baseline regression risk).
- Even at the optimistic bound it lands *at* the gate (≈38–40), within the
  ±3–5 t/s measurement noise — not a *safe* crossing.
- It is the "custom minimal verifier" further-fallback lever (objective
  boundaries), which the goal gates on explicit high-risk authorization.
- The correction token genuinely needs its own forward: the verify's
  `row_logits[i]` at the rejection point was computed with the **rejected draft**
  in context, not the correction — so it cannot substitute for the correction's
  prediction. (Confirmed by tracing ds4.c:30249–30340.)

### Recommendation
Surface to the user as the one remaining in-scope gate-crossing candidate,
explicitly high-risk, needing authorization. Do not pursue autonomously.

## 6. Re-assessment of the prior note-70 leads (independently verified)

| note-70 lead | my verification | outcome |
|---|---|---|
| A. Revert Bug #1 (stale window) | **= my Lead #1.** Real, but prompt-dependent; env-gated cap is the safer form. | Implemented (cap), +0..+6 t/s |
| B/O. Uncommitted greedy regression | **STALE.** `git status` is clean (no uncommitted greedy change); it was reverted in `1fcbd21`. | Moot — does not exist in current tree |
| C. hc_pre(ffn) F32 fix | Confirmed genuinely untested (note 68 §4). But addresses only the 0.12 probe↔oracle gap, not the 0.71 live gap. | Low value; the precision bisection chased the wrong gap |
| D. DISKMH diagnostic | `DS4_DSPARK_B2_DISKMH` loads pos-152 capture for *every* cycle — not a clean acceptance diagnostic. | Not informative as-is |
| E. Cap n_real | **= Lead #1** (implemented). | Done |
| F. Drafter sync-fusion (Opp3) | Ceiling +~1 t/s (note 41 measured +0.1–0.4). Within noise. | Not worth it |
| G. Custom verifier | **= Lead #2.** High-risk, lands at gate. | Surface to user |
| I. Q8_0 re-quant | Re-confirmed WORSE (issue468/69); GGUF reverted to Q4_K. | Done (negative) |
| L. imatrix | Exhausted (note 50: +1.0 % envelope << +5 % gate). | Done (negative) |

## 7. New observations not in the prior record

1. **The 3-forward cycle is inherent to DSpark's conditioning, and the
   single-token decodes are ~1.9× slower per-token than the batch verify**
   (anchor/correction ≈26 ms/token vs verify ≈14 ms/token). The off-by-one
   (drafter needs hidden at the anchor position = one ahead of last decoded)
   prevents fusing anchor↔correction↔verify. Verified by tracing positions.
2. **The baseline is flat across context** (38.6 t/s at 8k/16k/32k) — kills
   note 32's "speedup grows with context" projection (which assumed baseline
   slows). Confirmed by measurement.
3. **Measurement variance is large (±3–5 t/s)** under sustained M5 Max load —
   any lever claiming <3 t/s is within noise. The n_real collapse (4.11→2.89) is
   the only large, reliable signal found.
4. **The 25 % zero-accept rate** (drafter's *first* draft rejected, verified
   against sequential anchor p at position 0) is a drafter-quality ceiling issue,
   not a batch-p issue — it persists regardless of the verify path.

## 8. Verdict

- **Is the investigation exhausted?** The *precision* sub-investigation (notes
  53–69) is exhausted and was chasing the small (0.12) gap. The *live-path*
  sub-investigation was NOT exhausted: the n_real window-fill degradation (Lead #1)
  was a real, severe, fixable defect that was incorrectly dismissed, now
  implemented and measured.
- **Is the gate crossed?** **No.** Best sustained ≈31–33 t/s vs flat 38.6 baseline.
  With the 10th blocker precondition (this review) now complete, the remaining
  in-scope gate-crossing candidate is Lead #2 (correction-decode elimination,
  high-risk, lands ≈38–40 = at the gate, within noise).
- **Recommendation:** present to the user. The implementation is correct
  (B2-exact, MTP/plain non-regression verified, quality-neutral per note 59), the
  n_real cap is a real shipped optimization, and the perf gate is a defensible
  structural negative result on this fast-flat M5 Max baseline with the frozen
  drafter. Options for the user: (A) accept the perf-blocker; (B) authorize
  Lead #2 (high-risk custom-verifier / KV-compat work); (C) out-of-scope
  (draft trees EAGLE-3, or un-freeze the drafter).

## 9. Verification artifacts

- Real cycle timing: `DS4_DSPARK_B2_DEBUG=1` runs (anchor 26.5 / drafter 9 /
  verify 70 / correction 27 ms) — `/tmp/cap5.log`, `/tmp/ds2.log`.
- n_real degradation trajectory: issue468/70 §4 table (first/mid/last third).
- Cap sweep: caps 1/2/3/4/5/13/19/31/127 across 4 prompts.
- MTP non-regression: `--mtp` 33.75 t/s clean; plain 33.76 t/s clean.
- GGUF config: mtp.2 routed experts = q4_k (Q4_K, the 4.37-prefix config).
- Build: `make` clean, no new warnings; only code change is the env-gated cap
  (ds4.c:~30146), default path byte-identical.
