# Long-context baseline: decode curve + break-even + draft-cost decomposition

Date: 2026-06-28. Baseline for the long-context gate (8k/16k/32k/64k) after the
goal pivot. All numbers measured on M5 Max / Metal, IQ2XXS target, code prompt.

## 1. Plain decode t/s vs context (measured)

Prefill ~N tokens (code prompt repeated), generate 64, read generation t/s.

| ctx (prompt_tok) | decode t/s | ms/tok |
|---|---|---|
| ~6.9k  | 31.98 | 31.3 |
| ~13.7k | 31.53 | 31.7 |
| ~27.5k | 30.41 | 32.9 |
| ~54.9k | 28.77 | 34.8 |

**Decode slows only +11% from 7k→55k (31.3→34.8 ms/tok).** This is the key
long-context-thesis test, and it is WEAK: ds4 uses SWA (window=128), so decode
attention is context-independent — the small slowdown is memory pressure from the
larger resident KV set, not algorithmic. The batch verifier uses the same SWA
kernels, so it is also ~context-flat (doc 06: 75ms at 2k/4k/8k; estimated ~83ms
at 64k by the same +11%). **The spec-decode speedup therefore grows only mildly
with context**, not dramatically.

## 2. RAM feasibility at 64k (constraint check)

context buffers at ctx=65536 = **1394 MiB** (raw_kv_rows + compressed_kv_rows scale
sub-linearly). At 64k plain: ~18 GB RAM free after model (80.76 GB). With drafter
(10.71 GB): ~7 GB free — **marginal vs the 8 GB headroom constraint**. 64k
*sweep point* likely fits; 64k with drafter loaded is borderline. 32k and below
fit comfortably. The long-context sweep is feasible up to ~32k with drafter; 64k
needs care (may need to measure 64k plain-decode-only and project the +drafter case).

## 3. Break-even draft budget for the >20% gate (measured decode × measured acceptance)

`speedup = plain_ms × committed / (draft_ms + verify_ms)`, committed ≈ 3.0
(context-flat; drafter SWA=128), verify ≈ 75 ms (doc 06; +11% at 64k → 83ms est).

Gate `speedup >= 1.20`  ⟺  `draft <= plain_ms × committed / 1.20 − verify_ms`
= `plain_ms × 2.5 − 75`:

| ctx | plain ms | break-even draft for +20% |
|---|---|---|
| ~7k  | 31.3 | **3.3 ms** |
| ~14k | 31.7 | 4.3 ms |
| ~27k | 32.9 | 7.2 ms |
| ~55k | 34.8 | **12.0 ms** |

The break-even draft budget GROWS with context (3.3ms@7k → 12ms@55k) — the only
reason long context is more forgiving. The user's gate ">20% at 8k AND
non-declining through 64k" binds at the **8k clause (draft <= 3.3ms)**.

## 4. Draft-cost decomposition (from MEASURED ds4 components)

The verifier `metal_graph_verify_suffix_tops(L=5) = 75 ms` (doc 06) drives 5
positions through all 43 layers + the shared output head (lm_head) + topk. doc 06
§4 measures the output head directly: `out0/out1 ≈ 1.25 ms/pass` (the lm_head is
memory-bound on its 1.06 GB weight, amortized over the batch; M5 Max unified
memory ~800 GB/s → ~1.3 ms, matching the measurement). So:
- verify(75ms) ≈ 43-layer backbone (73.5 ms) + output head (~1.5 ms)
- backbone per layer per 5-pos batch = 73.5 / 43 ≈ **1.71 ms/layer**
- drafter backbone (3 layers) = 3 × 1.71 ≈ **5.1 ms**
- drafter shared lm_head (5 pos, memory-bound) ≈ **1.5 ms** (measured floor, doc 06 §4)
- drafter-specific overhead (main_proj, embed, hc_head sigmoid, sequential Markov
  head): ≈ **2 ms** (small matmuls/lookups, not weight-bandwidth-bound)

**Draft-cost estimate ≈ 8–9 ms** (5.1 backbone + 1.5 lm_head + 2 overhead), now
grounded in MEASURED components (verify 75ms, output-head 1.5ms). The only
unmeasured piece is the ~2ms drafter-specific overhead (bounded; small).

## 5. Projected speedup at 8–9 ms draft (the realistic range)

`speedup = plain × 3.0 / (draft + 75)`:

| ctx | plain | @8ms draft | @9ms draft | gate (>20%) |
|---|---|---|---|---|
| ~7k  | 31.3 | 1.13 (+13%) | 1.12 (+12%) | **FAIL** |
| ~14k | 31.7 | 1.16 (+16%) | 1.15 (+15%) | FAIL |
| ~27k | 32.9 | 1.19 (+19%) | 1.18 (+18%) | FAIL / borderline |
| ~55k | 34.8 | 1.26 (+26%) | 1.25 (+25%) | **PASS** |

**At the realistic draft-cost range, only 55k clears +20%; the strict gate
(8k AND non-declining) FAILS** on the 8k magnitude clause (8k gets +12–13%),
though the speedup IS non-declining/growing and IS material at long context
(+25–26% at 55k).

## 6. What this means for the plan

1. **The long-context thesis holds weakly, not strongly.** Decode barely slows
   (SWA), so the spec-decode edge grows only mildly with context. This was NOT
   known at tweak time (doc 06 only had 2k/4k/8k); it materially tightens the
   projection.
2. **The strict ">20% at 8k" clause is in real jeopardy** — it needs draft
   <= 3.3ms, but the lm_head floor alone is ~2.65ms and the full drafter is
   estimated 8–10ms. Only 55k clearly passes at realistic draft cost.
3. **The draft cost is STILL the unmeasured swing factor.** The estimate (8–10ms)
   is grounded but ±50%; a 3ms drafter would pass 8k, a 12ms drafter fails 55k.
   Only the Metal port resolves it.
4. **Per the goal (tweak path 2), the Metal port is authorized and is the next
   concrete step.** Two outcomes from building it:
   - If real draft cost <= ~3.3ms → strict gate passes at 8k+ and grows (outcome A).
   - If real draft cost is ~8–10ms (likely) → outcome B with a MEASUREED draft
     cost and a documented "+23–26% at 55k but <20% at 8k/16k/32k" partial result.

## 7. Decision

Proceed to build the Metal drafter port (authorized) — it is the only way to get
the real draft cost and the real end-to-end speedup. The risk is real (strict 8k
gate likely fails per §5), but the goal explicitly accepted this as the path to a
measured (not projected) verdict. The numpy oracle (validated, B2 acceptance 2.8–3.0)
is the spec to port.
