# DSpark Productionization — Regression Check & Structural Finding

Date: 2026-06-30. Second productionization handoff note (parallel to impl work in
`dspark-impl`). Supersedes the perf-blocker concern raised in the kickoff (doc 33)
by ruling out a self-inflicted regression and identifying the true structural
reason the end-to-end gate is hard on this hardware.

---

## 1. Baseline regression check (CLEAN — no regression)

Before concluding the perf gate cannot be met, the user directed a definitive
test: re-measure plain-decode baseline with ALL DSpark changes rolled back to the
upstream impl base vs current HEAD, at identical conditions.

**Method:** throwaway git worktree at the upstream base commit `cc33048`
(pre-port, all DSpark changes rolled back) built `ds4`; compared against current
`dspark-impl` HEAD (`17f51f0`, all DSpark work). Both run plain decode (no
`--dspark`, no `--mtp`), same model, same ctx/temp/prompt/seed.

| condition | upstream (cc33048) | HEAD (17f51f0) | gap |
|---|---|---|---|
| ctx=8192, n=256, t=0 (avg 3 runs) | 39.08 t/s | 39.01 t/s | 0.18% (noise) |
| ctx=32768, n=128, t=0 | 39.44 t/s | 39.41 t/s | 0.08% (noise) |

**Verdict: NO self-inflicted regression.** Plain decode is byte-for-byte identical
in performance within run-to-run noise. This is consistent with the code: when
`--dspark` is not passed, `e->dspark_ready = false`, so `enable_dspark = false` at
`metal_graph_alloc_raw_cap`, and no DSpark allocations (`spec_prefixN` buffers,
`dspark_kv_cache`) or DSpark hot-path code run during plain decode. The
P0 capture code in the verify loop is gated by `g->spec_capture_prefixN` (only
true during DSpark B2 verify). Plain decode is unaffected.

The handoff's headline baseline (39.18 t/s) reproduces exactly on this M5 Max.

## 2. Structural finding: baseline is FLAT across context (projection premise fails)

The handoff's speedup projection (§4.6) rests on a premise: baseline decode
slows with context (31.3 ms/tok at 8k → 35.5 ms/tok at 64k), while the DSpark
verify stays context-flat (75ms). Under that premise, DSpark wins at all contexts.

**Measured on this M5 Max (upstream binary, the clean reference):**

| ctx | baseline gen t/s | baseline ms/tok |
|---|---|---|
| 8192 | 39.40 | 25.4 |
| 16384 | 39.37 | 25.4 |
| 32768 | 39.40 | 25.4 |

**Baseline is FLAT (~25.4 ms/tok, 39.4 t/s) across context on this hardware.** It
does NOT slow with context (this M5 Max's memory subsystem sustains the attention
scan at constant rate to at least 32k). The projection's premise does not hold
here, which is the structural reason DSpark's verify overhead cannot amortize —
not a bug, not a regression.

Note: the handoff is internally consistent with this — §4.1 measures the same
~39 t/s baseline (25.5 ms/tok); only the §4.6 projection TABLE assumed a slowing
baseline that this hardware does not exhibit.

## 3. P0 result (delivered, working, +17% at ctx=4096 — but below baseline at 8k+)

P0 (KV replay elimination) is committed (`7f2320f`) and verified:
- Output clean/coherent (n=200, no garble/duplicates), deterministic.
- B2 exactness preserved (argmax(p−q) correction unchanged).
- ctx=4096: 28.25 → 32.93 t/s (+17% vs the 0.71× ported baseline).
- MTP non-regression: --mtp runs clean (39.5 t/s), matches plain greedy.

But at ctx≥8192 (the gate), DSpark is still below baseline:

| ctx | DSpark t/s | baseline t/s | ratio |
|---|---|---|---|
| 8192 | 28.29 | 39.24 | 0.72× |
| 16384 | 28.60 | 39.67 | 0.72× |
| 32768 | 28.49 | 39.61 | 0.72× |

## 4. Cycle breakdown (where the time goes — verify dominates)

Per-phase timing (`DS4_DSPARK_B2_DEBUG`, ctx=8192, committed `17f51f0`):

| phase | partial accept (drafts=2) | full accept (drafts=5) |
|---|---|---|
| anchor decode (+ capture mean readback) | 26 ms | 26 ms |
| drafter forward (+ Markov head CPU) | 12 ms | 12 ms |
| **verify (batch, 5 positions)** | **69 ms** | **69 ms** |
| B2 accept (CPU) | 1.5 ms | 1.9 ms |
| correction decode (KV) | 26 ms | 0 ms |
| **total cycle** | **135 ms** | **110 ms** |

The verify (69ms, ~48% of every cycle) is the dominant cost and is touched by
neither P0 nor P1-P4. It is the batch target forward over 5 draft positions
through all 61 layers. On this hardware the baseline decode is so fast (25.4
ms/tok, flat) that the speculation overhead cannot amortize at the achieved
acceptance (~4.5 committed/cycle).

## 5. Next: P1-P4 GPU ports

Proceeding with the GPU ports per the original goal ("as feasible"):
- **P3** in-GPU capture mean (eliminates the ~5ms CPU readback+mean+upload) —
  safest, no type-mismatch risk.
- **P1** Markov head to GPU (eliminates the ~4.3ms CPU loop; the [vocab,256]@emb
  matmul is memory-bound at ~0.1ms on GPU). Requires care: the weight is BF16 and
  `ds4_gpu_matmul_f16_tensor` reads F16 — must use a BF16-aware path (the same
  class of type mismatch that caused the ffn_gate_inp bug).
- **P2** B2 acceptance to GPU (~1.5ms → ~0.1ms).
- **P4** +1 bonus token — already landed in P0 (full-accept `s->logits` fix).

Even with all ports done, the structural analysis (§4) suggests the gate is hard
on this hardware: the theoretical best with verify unchanged is ~30-43 t/s
(depending on acceptance mix), vs baseline 39.4. The ports narrow but may not
reliably cross the line. We measure after each port and report honestly.
