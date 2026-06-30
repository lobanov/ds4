# Theoretical Drafter Acceptance — Research Synthesis & DSpark Break-Even Analysis

Date: 2026-06-30. Tenth productionization handoff note. Web-research synthesis
applying the speculative-decoding theoretical literature to DSpark's measured
numbers, to inform the pending perf-gate decision (A/B/C/D). Doc-only.

## 1. Theoretical foundations (from the literature)

### 1.1 Expected-acceptance formula (Leviathan et al. 2023, "Fast Inference from Transformers via Speculative Decoding")

For a draft block of γ tokens with per-position acceptance probability α:
```
E[accepted drafts per cycle] = (1 - α^(γ+1)) / (1 - α) - 1   (excl. bonus token)
E[total commits per cycle]    = (1 - α^(γ+1)) / (1 - α)       (incl. bonus)
```
Speedup over baseline decode ≈ E[commits] / (verify_cost_in_single_decode_units + draft_overhead).
The BentoML/NVIDIA empirical rule: "at α ≥ 0.6 and γ ≥ 5, spec decoding achieves 2-3× speedups" — **on compute/memory-bound baselines**.

### 1.2 Acceptance ceiling (the fundamental bound; Zhou et al.; "Disparate Impacts"; LANTERN)

The per-token acceptance probability is theoretically bounded by the **total-variation distance** between the drafter q and target p:
```
α ≤ 1 - TV(p, q) = 1 - ½·Σ_x |p(x) - q(x)|
```
TV is fixed by drafter-model quality. **A frozen drafter has a fixed α ceiling.** This is why the codex review concluded "no code-only lever raises α" — the bound is model-quality, not code.

### 1.3 What raises EFFECTIVE α WITHOUT changing the drafter (the research opening)

The literature documents several **algorithmic** methods that raise the effective commits-per-verify without re-quantizing/retraining the drafter:
- **Draft trees (EAGLE-2/3, Li et al.)**: instead of one linear γ-token draft, propose a TREE of candidate continuations; verify ALL tree paths in ONE batched target pass. Raises accepted-tokens-per-verify. EAGLE-3 reports state-of-the-art acceptance via dynamic confidence-based tree pruning. **The single most-cited training-free lever.**
- **Lossy / loose acceptance (FLy, "Optimal Lossy Variant", Annealed Relaxation)**: relax verification to accept semantically-equivalent or resampled drafts → raises effective α. TRADES exact-distribution-match (conflicts with B2 exactness).
- **Steering pretrained drafters (Zhou et al. AAAI)**: runtime steering of a FROZEN drafter toward the verifier → **+35% accepted tokens, +22% throughput**, no retraining.
- **Multi-drafter ensemble / Not-a-Bandit (arXiv 2510.20064)**: pick the best drafter per context online; provably no-regret.
- **Adaptive early-draft-stopping (AdaEDL)**: stop drafting when confidence drops → fewer wasted verify positions.

## 2. Applied to DSpark's actual numbers (the decision-relevant part)

### 2.1 DSpark's acceptance is actually GOOD — not the absolute problem

Inverting E[commits] = (1-α⁶)/(1-α) for DSpark's measured **3.98 commits/cycle at γ=5** (issue468/37 sweep):
```
inferred per-position α = 0.832
```
For context, EAGLE-3 reports ~46% avg acceptance as "good" in many setups; DSpark's mtp.2 drafter at α=0.83 is already strong. The drafter is NOT under-performing.

### 2.2 DSpark is just BELOW break-even (only ~6 points short)

Break-even analysis (cycle_ms / E[commits] < baseline 25.6 ms/tok), using measured phases (anchor 25ms + drafter 9.7ms + verify 68ms; +correction 25ms on partial-accept):

| α | E[commits] | E[cycle] | ms/tok | t/s | gate? |
|---|---|---|---|---|---|
| 0.83 (current) | 3.96 | 118ms | 29.8 | 33.6 | ✗ |
| 0.85 | 4.15 | 117ms | 28.1 | 35.6 | ✗ |
| **0.88** | **4.46** | **115ms** | **25.7** | **39.0** | **break-even** |
| 0.90 | 4.69 | 113ms | 24.1 | 41.5 | ✅ crosses |
| 0.95 | 5.30 | 108ms | 20.5 | 48.9 | ✅ crosses |
| 1.00 (full) | 6.00 | 103ms | 17.1 | 58.4 | ✅ big win |

**Two findings:**
1. **The gate needs α ≈ 0.88+; DSpark is at 0.83 — only ~5-6 percentage points short.** This is NOT a structural impossibility; it is a near-miss.
2. **At full acceptance (α→1), DSpark would hit 58 t/s** (the 3-forward cycle still wins because verify is batched — 68ms for 6 tokens is efficient). The headroom exists; the implementation is not fundamentally unviable.

### 2.3 Why it loses despite a good drafter

The handicap is the **DSpark-specific 3-forward cycle** (anchor + verify + correction = 118ms, 91% of cycle), driven by two design choices:
- **Anchor decode**: the drafter is conditioned on the target's hidden state (main_hidden), requiring a standalone target forward to capture it each cycle (+25ms).
- **Correction decode**: on partial accept, the correction token needs its own forward (+25ms on ~85% of cycles).

Standard Leviathan spec decoding has only ONE forward (verify); the bonus token IS the next anchor. DSpark pays two extra forwards for drafter-conditioning + correction. This raises the break-even α from ~0.7 (standard) to ~0.88 (DSpark).

## 3. Implications for the perf-gate decision (A/B/C/D + new E)

Under the CURRENT goal scope, the codex-loop conclusion stands: no code-only lever on the frozen drafter crosses the gate, because raising α beyond the TV-bound requires either a better drafter (frozen) or an algorithmic change (excluded: "do not redesign the spec-decode algorithm").

The research, however, reveals a path that was NOT in the original A/B/C/D menu:

**Option E — re-scope to allow VERIFIER-side algorithmic changes (draft trees):**
- Draft trees (EAGLE-2/3 style) are the standout training-free lever. They change the VERIFY path (tree-structured candidate attention + multi-path acceptance), NOT the drafter — so the frozen-drafter constraint is preserved.
- Theoretical promise: raises accepted-tokens-per-verify, directly attacking the "verify is 52% of cycle for ~3.96 commits" problem. A draft tree that raises effective commits/verify from 3.96 → 4.7+ would cross the gate per the break-even table.
- Caveats: (a) substantial new implementation (tree attention mask, multi-path rejection sampling, shared with the MTP verifier — regression risk); (b) touches the goal's "B2 design is fixed" boundary, so it needs an explicit scope change; (c) the ~0.83→0.88 gap is small but draft trees don't raise per-position α, they raise accepted-tokens-per-verify by covering more candidates — so the exact gain is a measurement question.

**Refined decision menu:**
- **A** — accept structural perf-blocker, complete on non-perf criteria. (current recommendation if scope is fixed)
- **B** — un-freeze drafter + imatrix (out of current scope; generic path already measured-failed; acceptance-targeted variant untried, multi-day).
- **C** — custom microbatch verifier (codex: +1.5-4 t/s, high-risk, likely still <39 — does NOT raise α, only cuts overhead; the break-even table shows overhead-cutting alone is insufficient since α is the binding constraint).
- **D** — abandon the perf gate.
- **E (NEW)** — re-scope to allow draft-tree verification (the one training-free lever that attacks the binding α constraint without unfreezing the drafter; substantial implementation; theoretical promise but unmeasured).

## 4. Summary verdict

The literature says DSpark is a **near-miss, not a structural failure**: α=0.83 vs break-even 0.88, with 58 t/s headroom at full acceptance. The binding constraint is α (model-quality-bounded), and the only training-free lever that raises accepted-tokens-per-verify is draft trees (EAGLE-style) — an algorithmic change currently out of scope. Overhead-cutting (Option C, custom verifier) does NOT address the binding constraint and per the break-even table is insufficient. Unfreezing the drafter (Option B) raises the α ceiling but is multi-day with contrary evidence. Option E (draft trees) is the untried path most aligned with the theoretical headroom, but needs a scope change.
