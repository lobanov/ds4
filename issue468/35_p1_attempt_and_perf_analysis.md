# DSpark Productionization — P1 Attempt & Definitive Perf Analysis

Date: 2026-06-30. Third productionization handoff note. Documents the P1 (Markov
head) GPU-port attempt and the definitive structural analysis of why DSpark
cannot beat baseline on this M5 Max.

## 1. P1 (Markov head) attempt — bandwidth-bound on CPU, GPU needs custom kernel

The Markov head CPU loop is the biggest CPU cost in the B2 cycle (5.3ms/cycle,
measured via the timing diagnostic). It computes, per position:
  q_row[v] = base_logits[v] + sum_r mw1[prev,r] * mw2[v,r]   (BF16 weights)

**Attempted optimization:** pre-convert mw1/mw2 from BF16 to F32 once at first
call (lazy static), eliminating the per-element BF16→F32 bit-shift in the inner
loop (165M conversions/cycle).

**Result: REGRESSED (5.3ms → 7.5ms).** The Markov head is **CPU memory-bandwidth-
bound**, not compute-bound. mw2 is [vocab=129280, rank=256]:
- BF16 inline: reads 64MB/position × 5 = 320MB/cycle (2-byte elements)
- F32 pre-converted: reads 129MB/position × 5 = 645MB/cycle (4-byte elements)

The 2× bandwidth increase exceeds the conversion savings. The original BF16
inline loop is already near-optimal for CPU (bandwidth-bound at 64MB/position).

**GPU port requirement:** a custom Metal BF16 matmul kernel. The existing
`ds4_gpu_matmul_f16_tensor` reads F16 (5 exp bits); BF16 has 8 exp bits — using
it on BF16 data would misread weights (same class of bug as ffn_gate_inp). No
BF16→F32 GPU cast primitive exists. Potential GPU savings: 5.3ms → ~0.4ms
(800GB/s GPU vs ~10GB/s CPU bandwidth). Reverted the CPU attempt.

## 2. Definitive cycle breakdown (ctx=8192, n=64, 18 cycles, 3.56 committed/cycle)

| phase | cost | % cycle | reducible? |
|---|---|---|---|
| verify (batch target, 5 pos) | 69 ms | 51% | NO — batch target forward |
| anchor decode | 25 ms | 19% | NO — baseline decode (unavoidable) |
| correction decode (partial) | 26 ms | 19% | NO — B2 correction ≠ draft, needs its own KV |
| drafter forward + Markov CPU | 12 ms | 9% | partially (Markov 5.3ms → needs GPU BF16 kernel) |
| B2 accept (CPU) | 1 ms | 1% | yes but tiny |
| **total** | **134 ms** | | |

The three target forwards (anchor + verify + correction = 120ms) are **90% of the
cycle and irreducible**. They process 7 positions for 3.56 committed.

## 3. Why DSpark cannot beat baseline on this M5 Max (structural, not a bug)

Baseline: **39.4 t/s = 25.4 ms/tok, FLAT across context** (8k–32k, verified on
upstream binary — doc 34).

For DSpark to beat baseline: cycle_time / committed < 25.4 ms.
- Full accept (6 committed): 120/6 = 20.0 ms/tok ✓ (beats baseline)
- Partial k=4 (6 committed): 134/6 = 22.3 ✓
- Partial k=3 (5 committed): 134/5 = 26.8 ✗ (barely loses)
- Partial k=2 (4 committed): 134/4 = 33.5 ✗
- Partial k=0 (2 committed): 134/2 = 67.0 ✗✗

DSpark only wins on high-acceptance cycles (k≥3). The measured avg committed
(3.56, i.e. avg ~1.5 drafts accepted) means most cycles are low-k partial
accepts that lose badly. The full-accept cycles (which win big) are too rare to
compensate.

**Even with all feasible GPU ports done** (Markov→GPU 5ms + capture 0.5ms +
accept 0.1ms = 5.6ms saved): 134→128ms → 128/3.56 = 36.0 ms/tok = 27.8 t/s.
Still 0.71× baseline. The verify (69ms, 51%) dominates and is untouched.

## 4. Root structural cause (two independent factors)

1. **Baseline is fast and flat** (25.4 ms/tok, doesn't slow with context on M5
   Max). The handoff's projection assumed 31→35.5 ms/tok (slowing) — that premise
   doesn't hold here. Each accepted token saves only 25.4ms, which can't amortize
   the 69ms verify.

2. **Acceptance rate is moderate** (3.56 committed/cycle, ~31% of the 5-draft
   block). Low acceptance means the 69ms verify (paid for all 5 positions) is
   mostly wasted. This is fixed by drafter model quality, not by code changes.

On hardware where the baseline IS slower (31+ ms/tok) and/or slows with context,
the same implementation WOULD win (per the handoff's projection). The DSpark code
is structurally correct and P0-optimized; the gate fails specifically because
this M5 Max's baseline is too fast and flat.

## 5. What IS delivered (production-ready, correct, self-contained)

- Clean port of all validated research code (commit f220ebe).
- P0 KV replay elimination — working, clean output, deterministic, +17% at
  ctx=4096 (commit 7f2320f). Partial-accept is now O(1) correction decode.
- Full-accept `s->logits` fix (the +1 bonus / P4) — eliminates duplicates.
- Self-contained GGUF tooling — converter + quantizer + Makefile + download (65d9a35, 328f003).
- MTP non-regression — --mtp runs clean, matches plain greedy.
- Baseline non-regression — plain decode identical to upstream (doc 34).
- B2 exactness preserved — output is verifier-guaranteed correct.

## 6. Quality gate (next)

B2 is exactness-preserving (the verifier guarantees output correctness regardless
of draft quality), so DSpark@temp=1.0 should match target@temp=1.0. The quality
gate (≥65/92) is independent of the perf gate and should pass.
