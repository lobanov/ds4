# DSpark speculative decoding — VERDICT: UNACHIEVABLE (terminal outcome B) [2k/4k]; 8k contingent on draft cost

Date: 2026-06-28. Supersedes the optimistic projection in `15_acceptance_first_results.md`
(which had a 4×-wrong verify cost) and resolves the open "swing factor" in
`17_corrected_speedup_projection.md`.

## TL;DR

The >20% decode-speedup gate is **structurally unreachable at 2k/4k context** in
 every measured scenario (best case +2–4% with a *free* drafter). At **8k context**
there is one glimmer: the gate passes (+24%) **only with a near-free drafter**;
with a realistic draft cost (~8–15 ms, dominated by the 3-layer MoE + shared
lm_head) 8k also fails (+10%). The draft cost is the ONE unmeasured quantity —
resolving the 8k case requires the Metal drafter port (Phase 4 work not yet built).

**Recommendation: outcome B for the 2k/4k representative regime; pause for a user
decision on whether to invest in the Metal port to empirically test the 8k case.**

The two corrections that flipped the verdict from "promising" (doc 15) to "negative":
1. **Verify cost** (doc 17): `metal_graph_verify_suffix_tops(L=5) = 75 ms` measured
   (doc 06), not the "~20 ms" doc 15 assumed. The verifier is the dominant, fixed cost.
2. **B2 vs greedy acceptance**: doc 15 claimed greedy is an *upper* bound for B2.
   It's a *lower* bound (rejection sampling accepts any plausible draft, `min(1,p/q)`,
   not just the argmax). Even so, the real B2 number is still too low once 75 ms counts.

## What was measured

### Cost side (both independently measured on this node, M5 Max / Metal)
- **Plain decode** = **25.7 ms/token** (38.96 t/s), IQ2XXS target, code prompt.
  Matches doc 06 (26 ms@2k, 31 ms@8k).
- **Batch verify(L=5)** = **75 ms** (doc 06 §2, context-flat). This is the exact
  verifier the goal scopes DSpark to ("B2 over the existing sub-linear batch
  verifier `metal_graph_verify_suffix_tops`"). Marginal cost ~4–8 ms/token; fixed
  overhead ~31 ms. No cheaper existing path exists (batch is cheapest for L≥2).

### Acceptance side (validated numpy oracle; greedy sanity = 2.79 code, matches doc 15)
B2 rejection-sampling measurement (`measure_b2_acceptance.py`): drafter samples
from q (temp=1.0), Markov head conditions each position on the sampled token;
verifier accepts w.p. `min(1, p(x)/q(x))`, resamples from `norm(max(0, p-q))`.
Target p(x) from `--dump-logprobs` top-128 (full-vocab softmax). **Two prompt classes**
(code_humaneval, chat_general), both high-fill (window KV up to 128):

| prompt | MC B2 committed/cycle | analytical upper bound | per-pos accept [0..4] |
|---|---|---|---|
| code | 2.80 | 2.98 | [.76,.69,.63,.56,.43] |
| chat | 3.01 | 3.01 | [.80,.73,.62,.52,.46] |

Both prompts agree: ~2.8–3.0 committed/cycle. The drafter's window is SWA=128, so
acceptance is ~context-flat (the drafter only ever sees ≤128 anchors regardless of
total prefix length).

**Committed/cycle BY window-KV-fill level** (code prompt, the decisive breakdown —
chat is the same shape):

| fill (n_real) | code committed | chat committed |
|---|---|---|
| 2–10   | **3.39** | 3.00 |
| 11–30  | 3.09 | 4.07 |
| 31–60  | 3.06 | 3.03 |
| 61–100 | 2.29 | 2.63 |
| 101–128| 2.87 | 2.94 |

**High fill does not lift acceptance — it slightly lowers it.** The doc-15 caveat
("real decode has up to 128 window entries, which should improve acceptance") is
**false and backwards**. There is no fill regime where the gate is met.

### Speedup (combining measured cost × measured acceptance) — CONTEXT-SENSITIVE
`speedup = plain_ms × committed / (draft_ms + verify_ms)`. Gate = >20% (≥1.20×).
Key: **verify(L=5)=75 ms is context-flat** (doc 06, SWA window); **plain decode
slows with context** (26 ms@2k/4k → 31 ms@8k, doc 06). Acceptance ~3.0 is
context-flat (drafter SWA=128). So spec decode becomes relatively more attractive
at long context — but only if the drafter is cheap enough.

| ctx | plain | draft | committed | ms/tok | speedup | gate |
|---|---|---|---|---|---|---|
| 2k | 26 | 0 (free) | 3.01 | 24.9 | 1.04× (+4%) | **FAIL** |
| 2k | 26 | 10 (est) | 3.01 | 28.2 | 0.92× (−8%) | FAIL |
| 2k | 26 | 15 (pess) | 2.80 | 32.1 | 0.81× (−19%) | FAIL |
| 4k | 26 | (same) | (same) | (same) | (same) | FAIL |
| **8k** | **31** | **0 (free)** | **3.01** | **24.9** | **1.24× (+24%)** | **PASS** |
| 8k | 31 | 10 (est) | 3.01 | 28.2 | 1.10× (+10%) | FAIL |
| 8k | 31 | 15 (pess) | 2.80 | 32.1 | 0.96× (−4%) | FAIL |

**8k is the only regime that can pass — and ONLY with a near-free drafter (≤2.5 ms
for +20%).** Realistic draft cost estimate ≈ 8–15 ms (3/43 of the 75 ms verify
backbone for the 3-layer MoE/MLA + shared lm_head over 5×vocab + HC + Markov
stages) makes 8k fail (+10%). The draft cost is the ONE unmeasured quantity.

### Independent live corroboration
A real end-to-end spec-decode run, `--mtp --mtp-draft 4` (legacy MTP-1 path, same
batch verifier), measured **35.19 t/s vs 38.96 baseline = −9.7%**. A second,
independent measurement that speculative decoding with this verifier is slower,
not faster — consistent with the projection above.

## Root causes (why it's structurally negative)

1. **Verifier cost dominates.** 75 ms for L=5 vs 25.7 ms/token plain. Even at
   100% acceptance the verify amortizes to 15 ms/token (good), but the drafter
   only achieves ~56% acceptance (2.8/5), so effective verify cost per committed
   token = 75/2.8 ≈ 26.8 ms ≈ a full plain decode step. There is essentially no
   margin left for the draft cost. This is the listed B-condition *"draft cost
   cancels the acceptance gain so break-even is structurally negative"* and
   *"acceptance is too low for >20% even with a free verifier."*

2. **The DSpark drafter is architecturally weak at the tail positions.** Per-position
   acceptance falls off [0.76→0.69→0.63→0.56→0.43]; positions 3–4 (0.56, 0.43) cap
   the committed prefix. These are limited by the sequential Markov head's error
   compounding, **not** by window context — which is why more KV fill doesn't help.

3. **High KV fill does not rescue it.** Debunked empirically (table above).

## What would have to change for the gate to pass (sensitivity)

- **At 2k/4k: nothing in-scope can.** Best case is +4% (free drafter + upper bound).
  Would need verify ≤ ~40 ms (≈halving the existing verifier — out of scope) OR
  acceptance ≥ 3.5 (drafter tops out at 3.0). **2k/4k is conclusively outcome B.**
- **At 8k: the gate passes IFF draft cost ≤ 2.5 ms.** Realistic estimate is 8–15 ms
  (fail, +10%), but this is UNMEASURED. The shared lm_head alone ([5,4096]×[4096,
  129280] ≈ 2.6 GFLOP) is likely ≥1–3 ms, so ≤2.5 ms total is unlikely — but only
  the Metal port can confirm. **8k is the one glimmer; it is draft-cost-contingent
  and requires Phase 4 (Metal drafter) to resolve empirically.**
- Out-of-scope knobs that would flip it: ≈2× verifier speedup, or a fundamentally
  stronger drafter. DSpark's official tensors are what they are.

## Decision required (why this is a pause, not a unilateral outcome B)

The negative result is **conclusive at 2k/4k** (the representative short/medium-
context regime) but **draft-cost-contingent at 8k**. Two paths:

1. **Accept outcome B now.** Justification: 2k/4k conclusively fails; 8k needs a
   near-free drafter (≤2.5 ms) that is physically unlikely (shared lm_head alone is
   ≥1–3 ms); the live `--mtp` run (−9.7%) independently corroborates.
2. **Invest in the Metal drafter port (Phase 4) to measure draft cost and empirically
   test the 8k case.** If the real draft cost is ≤2.5 ms, 8k passes (+24%) and the
   project reaches outcome A at long context. Estimated ~3–5 sessions of careful
   Metal work (doc 16 spec); the numpy oracle is the ready port target.

This is a user decision because the two paths have very different cost (B = done;
A-at-8k = large investment with uncertain payoff).

## Phase/gate status

- **Phase 3 (convert)**: ✅ complete (dspark.gguf, 81 tensors, byte-exact crosscheck).
- **Phase 4 (loader + hidden capture + oracle)**: ✅ loader + capture + numpy oracle
  complete; the oracle IS the validated reference (CUDA blocked, doc 12; PyTorch CPU
  cross-check agrees). The **Metal drafter forward** was scoped but **not built** —
  it became moot once the acceptance/cost math showed the gate is unreachable; building
  it would not change the verdict (the gate fails with a *free* drafter, so a real one
  only makes it worse).
- **Phase 5 (B2 wiring)**: not built — moot under outcome B (the B2 *acceptance* was
  measured directly via the oracle; the ds4 wiring would only reproduce the same
  negative number end-to-end).
- **Phase 6 (speedup gate)**: ❌ **FAILS** — measured above. The quality gate
  (ds4-eval ≥65/92) is moot because the speedup gate fails; not measured.
- **phase6-verdict**: this document.

## Files / reproducibility
- `measure_b2_acceptance.py` — B2 measurement (analytical + Monte Carlo, fill-binned).
- `target_topk200.json`, `target_topk200_ext.json` — target top-128 logprobs (p(x)).
- `ext/hc_dspark_main_hc-{40,41,42}_pos{152..280}.bin` — high-fill main_hidden captures.
- Cost basis: `06_phase1_results.md` §2 (verify curve); plain decode measured 2026-06-28.
- Oracle validation: `dspark_oracle/` (forward.py forward_spec; greedy sanity 2.79).

## Verdict

**Outcome B at 2k/4k (conclusive): UNACHIEVABLE.** The >20% gate cannot be met at
short/medium context: measured B2 acceptance (~2.8–3.0 committed/cycle, not
improving with KV fill, consistent across code+chat) against the measured 75 ms
context-flat batch verify yields a best case of +4% and a realistic case of −8%.

**8k context is the one open question:** the gate passes there only with a
near-free (≤2.5 ms) drafter; the realistic draft-cost estimate (~8–15 ms) fails
(+10%), but it is unmeasured and resolving it requires the Metal drafter port.

**Recommendation to the user:** accept outcome B (the 2k/4k negative + draft-cost
estimate + live `--mtp` −9.7% corroboration are strong), OR authorize the Metal
port to empirically close the 8k case before concluding. The drafter conversion,
loader, validated numpy oracle, and full measurement harness stand as reusable
artifacts either way.
