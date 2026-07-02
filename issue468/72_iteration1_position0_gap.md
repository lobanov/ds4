# Iteration 1: Live-vs-Oracle Position-0 Gap — Batch-Verify Target Mismatch

Date: 2026-07-01. Iteration 1 of the live-vs-oracle investigation. Doc-only research record.

## 0. Measurement setup

Live B2 at temp=1, ctx=8192, n=256, seed=1, Fibonacci/code prompt. 66 B2 cycles.
DS4_DSPARK_B2_DEBUG=1 for per-position accept/reject logging.

## 1. Per-position accept rate: LIVE vs ORACLE

| position | live accept rate | oracle accept rate | gap |
|---|---|---|---|
| **pos 0** | **68.2%** | **95.5%** | **−27.3 pp** |
| pos 1 | 82.2% | 98.1% | −15.9 pp |
| pos 2 | 67.6% | 91.8% | −24.2 pp |
| pos 3 | 64.0% | 87.9% | −23.9 pp |
| pos 4 | 43.8% | 78.5% | −34.7 pp |

Every position is 15-35 percentage points lower in the live path than the oracle predicts. **Position 0 is the worst at −27.3pp.** Since position 0 uses `s->logits` (the target distribution from the anchor decode — NOT the batch verify), the gap at position 0 points to the **anchor decode's logits** diverging from the oracle's captured targets.

## 2. The three data sources for target distribution

- **MC oracle** (`measure_b2_acceptance.py`): uses `target_topk.json` — pre-captured target top-128 logprobs from a separate reference decode run. These are the "ground truth" target distribution.
- **Live B2 pos 0**: uses `s->logits` — the target model's logits after the anchor decode. Set by `ds4_session_eval(s, first_token, ...)` in the B2 function.
- **Live B2 pos 1-4**: uses `row_logits[i-1]` from `metal_graph_verify_suffix_tops` — the batch-verify path's target logits at each draft position.

## 3. Position 0 gap analysis

Position 0 compares the drafter's draft[0] (q distribution) against `s->logits` (p distribution). The 68% accept rate means the drafter is proposing tokens that the target assigns low probability to. But the oracle, using the SAME Metal base_logits, achieves 95.5% — so either:

(a) The drafter's q distribution in live differs from the probe's q distribution (the live drafter forward produces different base_logits because main_hidden or KV state differs), OR
(b) The target's p distribution (`s->logits`) in live differs from the oracle's captured targets (`target_topk.json`), OR
(c) The rejection sampling RNG sequence differs.

The RNG is seeded from --seed (Bug #3 fix). The oracle uses a fixed seed (20260630). So (c) is unlikely to explain a 27pp gap.

For (b): `s->logits` is set by the anchor decode's output head. The oracle's `target_topk.json` was captured from a reference run. If the anchor decode and the reference run used different KV state (the live path's growing drafter KV window might perturb the target's checkpoint), the logits could differ. But the target model's KV is independent of the drafter's KV — the drafter's KV window doesn't affect the target's forward.

For (a): the live drafter forward uses main_hidden captured from the ANCHOR DECODE (the target model's forward at the anchor position). The probe uses main_hidden captured from a pre-computed reference run. If the live anchor decode produces different main_hidden (because the target's checkpoint differs — e.g., different prompt, different context position), the drafter's base_logits would differ, producing worse drafts.

## 4. The key hypothesis: live main_hidden ≠ probe main_hidden

The MC oracle's high acceptance (4.10) is measured on main_hidden captured from a controlled reference decode (the sweep8 capture bundles). The live B2 path captures main_hidden from the actual anchor decode during generation. If the live anchor decode's hidden state at layers 40/41/42 differs from the reference capture — because the target model's internal state evolved differently during live generation vs the reference run — the drafter would produce different (potentially worse) drafts.

This is TESTABLE: dump the live main_hidden and compare to the probe's captured main_hidden at the same position.

## 5. Iteration report

| metric | live (temp=1) | oracle MC (Metal) | gap |
|---|---|---|---|
| accepted drafts/cycle | **1.97** | **4.10** | **2.13** |
| committed tokens/cycle | **2.97** | **5.10** | **2.13** |
| gen t/s | **29.5** | — | — |
| ms/token (1000/t/s) | **33.9** | — | — |
| baseline t/s | **~39** | — | — |
| pos 0 accept rate | **68.2%** | **95.5%** | **−27.3 pp** |

## 6. Next step

Investigate hypothesis (a): dump the live main_hidden (from the anchor decode at layers 40/41/42) and compare to the probe's captured main_hidden at the same generation position. If they differ, the live anchor decode is producing a different hidden state, and the drafter is consuming degraded input.
