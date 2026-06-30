# DSpark Productionization — P1-P4 Final Results & Perf-Blocker Report

Date: 2026-06-30. Fourth productionization handoff note. Final results after
P1-P4 were attempted, and the perf-blocker report (per the goal's blocker rule:
regression check clean + P1-P4 attempted).

## 1. P1-P4 outcome

| port | status | result |
|---|---|---|
| **P1** Markov head → GPU | **DONE** (commit 9a8649a) | drafter phase 12.4→9.7ms (-2.7ms); DSpark 28.29→30.12 t/s (+6.5%). Bit-exact (F32). Reuses `ds4_gpu_matmul_f32_tensor` with mw2 converted BF16→F32 at load into a registered auxiliary model map. |
| P2 B2 accept → GPU | assessed, **skipped** | CPU B2 accept = ~1ms/cycle (0.8% of cycle). A GPU softmax/accept kernel would save <1ms for non-trivial complexity/risk. Negligible ROI. |
| P3 in-GPU capture mean | assessed, **skipped** | capture readback+mean+upload = **0.05ms/cycle** (measured) — NOT the ~5ms the handoff claimed. The 192KB readback + HC=4 mean is free next to the anchor decode. Negligible. |
| P4 +1 bonus token | **already in P0** | the full-accept `s->logits = row_logits[block-1]` fix (P0) already produces the bonus token as the next cycle's anchor. It cannot be a free *extra* commit — the bonus position's KV requires a decode regardless, which is exactly the next anchor. No additional win. |

So P1 is the only real perf win; P2/P3 negligible; P4 was already handled.

## 2. Final measured numbers (the three the blocker rule requires)

ctx=8192, n=256, temp=0.0, identical prompt/seed:

| build | gen t/s | ms/tok |
|---|---|---|
| baseline, rolled-back (cc33048, all DSpark changes reverted) | **39.08** | 25.6 |
| baseline, current HEAD (17f51f0 + 9a8649a, plain decode) | **39.06** | 25.6 |
| **DSpark, current HEAD (P0 + P1)** | **29.02** | 34.5 |

- **No self-inflicted baseline regression** (39.08 vs 39.06 = noise).
- **DSpark is 0.74× baseline** (29.02 / 39.06). The gate (DSpark > baseline) is NOT met.

## 3. Cycle breakdown after P0+P1 (ctx=8192, why it can't win here)

Per-phase (`DS4_DSPARK_B2_DEBUG`), avg over many cycles, ~3.56 committed/cycle:

| phase | ms | % cycle | reducible? |
|---|---|---|---|
| verify (batch target, 5 positions) | 68 | 52% | NO — full target forward |
| correction decode (partial accept) | 26 | 20% | NO — B2 correction ≠ draft; KV must be decoded |
| anchor decode (+ 0.05ms capture) | 25 | 19% | NO — baseline-equivalent token decode |
| drafter forward + GPU Markov (P1) | 9.7 | 7% | P1 done; remainder is GPU forward |
| B2 accept (CPU) | 1 | 1% | negligible |
| **total** | **~130** | | |

The three target forwards (anchor + verify + correction = 119ms) are **91% of the
cycle and structurally irreducible** in the B2 design. To beat baseline
(25.4 ms/tok), the cycle must be < 25.4 × committed:

- full accept (6 committed): 102.7ms / 6 = 17ms/tok ✓ (wins)
- partial k=3 (5): 130/5 = 26ms/tok ≈ baseline
- partial k=2 (4): 130/4 = 32.5ms/tok ✗
- partial k=0 (2): 130/2 = 65ms/tok ✗✗

DSpark wins only on high-acceptance cycles (k≥3); the achieved avg (~3.56
committed, ~1.5 drafts/cycle) is dominated by low-k partial accepts that lose.

## 4. Root cause (structural, two factors — neither is a bug)

1. **Baseline is fast and FLAT** on this M5 Max: 25.4 ms/tok (39.4 t/s), constant
   across 8k–32k context. The handoff's projection (§4.6) assumed the baseline
   slows with context (31→35.5 ms/tok at 8k→64k); that premise does NOT hold on
   this hardware. Each accepted token saves only 25.4ms, which cannot amortize the
   68ms verify.

2. **Acceptance is moderate** (~3.56 committed/cycle, ~31% of the 5-draft block).
   Low acceptance means the 68ms verify (paid for all 5 positions) is mostly
   wasted. Acceptance is fixed by drafter-model quality, not by code.

The DSpark implementation is structurally correct and P0/P1-optimized; the gate
fails specifically because this M5 Max's baseline decode is too fast and flat.
On hardware where the baseline IS slower (31+ ms/tok) and/or slows with context,
the same code WOULD win per the handoff projection.

## 5. What IS delivered (production-ready, correct, self-contained)

- Clean port of all validated research code (f220ebe); 0 warnings.
- P0 KV-replay elimination — partial accept is now O(1) correction decode, was
  O(k+1) replay. Clean/deterministic output, B2 exactness preserved (7f2320f).
- Full-accept `s->logits` fix (the +1-bonus / P4) — eliminates duplicates.
- P1 GPU Markov head — bit-exact, +6.5% (9a8649a).
- Self-contained GGUF tooling — converter + quantizer + Makefile + public-bucket
  download (65d9a35, 328f003, a479ac7).
- MTP non-regression — `--mtp` runs clean, matches plain greedy.
- Baseline non-regression — plain decode identical to upstream (doc 34).
- B2 exactness preserved — output verifier-guaranteed correct.

The branch is correct and self-contained. The perf gate is the only unmet
criterion, and it is structurally blocked by this hardware's fast flat baseline.

## 6. Perf-blocker report (per goal blocker rule)

The perf-blocker is raised after BOTH preconditions are met:
1. Baseline-regression check: CLEAN (39.08 vs 39.06 t/s).
2. P1-P4 GPU ports: attempted (P1 done; P2/P3 negligible; P4 in P0).

Measured: DSpark 29.02 t/s < baseline 39.06 t/s (0.74×). Blocking technical
reason: the three target forwards (anchor 25ms + verify 68ms + correction 26ms =
119ms, 91% of cycle) are structurally irreducible, and on this M5 Max the
baseline is too fast (25.4ms/tok) and flat (doesn't slow with context) for the
speculation to amortize at the achieved ~3.56 committed/cycle.

Awaiting user direction on how to proceed (see pause note).
