# DSpark speculative decoding — VERDICT: UNACHIEVABLE (terminal outcome B)

Date: 2026-06-28. Supersedes the optimistic projection in `15_acceptance_first_results.md`
(which had a 4×-wrong verify cost) and resolves the open "swing factor" in
`17_corrected_speedup_projection.md`.

## TL;DR

The >20% decode-speedup gate is **structurally unreachable** on this hardware with
the in-scope verifier, in every measured scenario. Best case (a *free* drafter
and the analytical acceptance *upper bound*) is **+2%**; the realistic case is
**−20% (slower than plain decode)**. This is terminal outcome B.

The two corrections that flipped the verdict from "promising" (doc 15) to "negative":
1. **Verify cost** (doc 17): `metal_graph_verify_suffix_tops(L=5) = 75 ms` measured
   (doc 06), not the "~20 ms" doc 15 assumed. The verifier is the dominant, fixed cost.
2. **B2 vs greedy acceptance**: doc 15 claimed greedy is an *upper* bound for B2.
   It's a *lower* bound (rejection sampling accepts any plausible draft, `min(1,p/q)`,
   not just the argmax). Even so, the real B2 number (2.80 committed/cycle) is still
   far too low to clear the gate once the 75 ms verify is counted.

## What was measured

### Cost side (both independently measured on this node, M5 Max / Metal)
- **Plain decode** = **25.7 ms/token** (38.96 t/s), IQ2XXS target, code prompt.
  Matches doc 06 (26 ms@2k, 31 ms@8k).
- **Batch verify(L=5)** = **75 ms** (doc 06 §2, context-flat). This is the exact
  verifier the goal scopes DSpark to ("B2 over the existing sub-linear batch
  verifier `metal_graph_verify_suffix_tops`"). Marginal cost ~4–8 ms/token; fixed
  overhead ~31 ms. No cheaper existing path exists (batch is cheapest for L≥2).

### Acceptance side (validated numpy oracle; greedy sanity = 2.79, matches doc 15)
B2 rejection-sampling measurement (`measure_b2_acceptance.py`): drafter samples
from q (temp=1.0), Markov head conditions each position on the sampled token;
verifier accepts w.p. `min(1, p(x)/q(x))`, resamples from `norm(max(0, p-q))`.
Target p(x) from `--dump-logprobs` top-128 (full-vocab softmax).

| metric | low-fill (pos 152–175) | **high-fill (pos 152–280)** |
|---|---|---|
| MC B2 committed/cycle | 3.22 | **2.80** |
| analytical upper bound (greedy-cond) | 3.47 | **2.98** |
| per-position accept [0..4] | [.85,.84,.72,.54,.40] | [.76,.69,.63,.56,.43] |

**Committed/cycle BY window-KV-fill level** (high-fill run, the decisive breakdown):

| fill (n_real) | mean committed | n cycles |
|---|---|---|
| 2–10   | **3.39** | 1800 |
| 11–30  | 3.09 | 4000 |
| 31–60  | 3.06 | 6000 |
| 61–100 | 2.29 | 8000 |
| 101–128| 2.87 | 5000 |

**High fill does not lift acceptance — it slightly lowers it.** The doc-15 caveat
("real decode has up to 128 window entries, which should improve acceptance") is
**false and backwards**. There is no fill regime where the gate is met.

### Speedup (combining measured cost × measured acceptance)
`speedup = 25.7 × committed / (draft_ms + 75)`. Gate = >20% (≥1.20×).

| scenario | draft | committed | ms/tok | speedup | gate |
|---|---|---|---|---|---|
| best conceivable (free drafter + upper bound) | 0 | 2.98 | 25.2 | **1.02× (+2%)** | **FAIL** |
| realistic (15 ms drafter) | 15 | 2.80 | 32.1 | 0.80× (−20%) | FAIL |
| pessimistic (30 ms drafter) | 30 | 2.80 | 37.4 | 0.69× (−31%) | FAIL |

Break-even for the gate needs **committed ≥ 3.50 with a *free* drafter**
(`1.2 × 75 / 25.7`); the absolute upper bound across all fill levels is **2.98**.

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

## What would have to change for the gate to pass (sensitivity / out-of-scope)

- **Verifier ≤ ~50 ms at L=5** (at acceptance 2.80, free drafter → +44%). Requires a
  ~33% verifier/kernel speedup — explicitly out of scope (goal: reuse the existing
  batch verifier). *A 2× verifier optimization would flip the verdict.*
- **Acceptance ≥ 3.5 committed/cycle.** The 3-layer drafter tops out at 2.98 (upper
  bound); a fundamentally stronger drafter would be needed. DSpark's official
  tensors are what they are.
- Neither knob is movable within the goal's scope.

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

**Terminal outcome B — UNACHIEVABLE.** The >20% decode-speedup gate cannot be met
locally with the in-scope verifier: measured B2 acceptance (2.80 committed/cycle,
not improving with KV fill) against the measured 75 ms batch verify yields a best
case of +2% and a realistic case of −20%. No remaining in-scope knob (drafter
precision, block size, KV fill) closes the gap; the two that would (≈2× verifier
speedup, a fundamentally stronger drafter) are out of scope. The drafter
conversion, loader, and validated numpy oracle stand as reusable artifacts.
