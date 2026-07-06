# Stage 1 — tap-precision acceptance measurement (Q4-tap vs IQ2XXS)

Date: 2026-07-06. Status: **complete. Result: no significant acceptance change
from this Q4-tap variant (underpowered); the strong "tap-localized quant mismatch
falsified" framing was overstated and is softened below per adversarial review.**
Capture: `issue468/run_exactness_small_bundles.py` (model = Q4-tap variant, temp=0,
seed=2, same corpus); comparison tool: `issue468/run_stage1_q4tap_compare.py`.
Artifacts: `issue468/artifacts/exactness_small_bundles_q4tap/` (bundles +
`comparison_vs_baseline/{comparison.json,comparison.csv,codex_review.md}`).

## Question (gated by Stage 0)

Stage 0 showed drafter p=1 misses are **shallow** (median target-rank 1.0; 100%
within top-10). Stage 1 tests the specific mechanism: does **raising tap-layer
precision convert those rank-2 misses into rank-1 matches** — i.e. is the mismatch
localized to the drafter's tap layers (40/41/42) and quant-driven? (Codex flagged
this as a partial proxy for the decisive FP-teacher test, which needs a model we
don't have locally.)

## Setup — clean A/B

- **Baseline:** retained IQ2XXS bundles (`exactness_small_bundles/*__t0p0`).
- **Variant:** `DeepSeek-V4-Flash-Layers37-42Q4KExperts-...-imatrix-fixed.gguf` —
  same model, **layers 37–42 routed experts at Q4_K** (everything else IQ2XXS),
  same AProjQ8/SExpQ8/OutQ8 (so embed/lm_head quant unchanged).
- Same 10 prompts, temp=0, seed=2, ctx=4096, 14 generated tokens. Greedy is
  seed-independent, so tokenization matches (prompt_tokens identical per prompt).
- Re-captured layer 40/41/42 hidden states + greedy tokens + top-128 from the
  Q4-tap target, then re-measured drafter acceptance on the fresh captures. **The
  drafter, its weights, and embed/lm_head quant are unchanged** — only the target
  (hidden states the drafter sees + the ground-truth tokens) changes.

**Freshness verified (contract):** Q4-tap `main_hidden` differs from baseline
(max abs diff 59.5, mean 1.09, not identical); 33/140 target tokens differ — the
captures are genuinely from the Q4-tap run, not reused.

## Result

| metric | IQ2XXS baseline | Q4-tap | Δ |
|---|---:|---:|---:|
| **E[a\|5block]** (avg accepted prefix) | 2.3375 | 2.4375 | **+0.10** |
| **p=1 match rate** (clean signal, n=80) | 0.8125 | 0.7875 | **−0.025** |
| overall match% (all 5 positions) | 51.0% | 54.0% | +3.0 pp |
| per-prompt avg-prefix direction | — | — | 4 up / 3 down / 3 flat |

**The clean signal (p=1) is flat-to-slightly-negative** (65/80 → 63/80 matches;
McNemar discordants 7 base-only / 5 q4-only, p≈0.77; not significant). The +0.10
E[a|5block] is **statistically indistinguishable from zero** (paired t p=0.54;
per-prompt bootstrap 95% CI **[−0.16, +0.41]** — wide, straddling zero) — so it is
NOT proven noise; the study is underpowered to detect a small effect. There is a
**one-sided per-position trend**: Q4-tap is flat/slightly-negative at p=1
(−2.5 pp), unchanged at p=2, and increasingly positive deeper (p=3 +3.75, p=4
+5.0, p=5 +8.75 pp) — but deeper positions are rollout-confounded (after the first
miss the drafter's context diverges) and small-N, so this is an observation, not a
clean effect. **Net: this Q4-tap expert-only variant does not materially improve
the clean p=1 acceptance; any broader claim is limited by power and by the
variant's partial precision (see caveats).**

## Interpretation (softened per adversarial review)

- **This Q4-tap variant does not materially help p=1; broader falsification is
  overstated.** The drafter's input layers (40/41/42) were raised to Q4-K experts,
  yet p=1 acceptance did not improve. But this only falsifies *"this expert-only
  Q4-tap variant materially helps"* — **not** *"tap-localized quant mismatch"* in
  general: layers 40–42 still receive Q2-corrupted activations from below
  (layers 1–39 unchanged), so a flat result is ambiguous between (a) lower-layer
  Q2 effects, (b) drafter-vs-target calibration error independent of quant, and
  (c) a real but small effect the study was underpowered to detect.
- **Serving a higher-precision target is not demonstrated as a usable lever** at
  this precision level and sample size. (Resolving localization cleanly needs a
  full Q8/BF16 target — codex's proposed decisive test — not available locally.)
- **Fine-tuning is a plausible next lever, not a proven exclusive one.** The
  Stage 0 top-2 coverage (≈0.91 at p=1) is **headroom** (target currently sits at
  drafter rank ≤2 ~91% of the time), NOT a hard fine-tune ceiling — a fine-tune
  could do better or worse. It indicates the drafter is consistently in the right
  neighborhood, which is the regime where a fine-tune has leverage, but only an
  actual fine-tune (Stage 2) would measure the realized gain. Reframing the
  deficit as "calibration error" is an inference beyond the data (the data shows
  shallow disagreement, not its cause).

## Caveats

- The Q4-tap variant raises only layers 37–42 experts; lower layers stay IQ2XXS,
  so a genuinely flat result cannot distinguish "not quant" from "quant but in
  lower layers." Resolving that needs a full Q8/BF16 target (not available locally).
- n=10 prompts, 8 measure steps each (80 p=1 positions). The +0.10 E[a|5block] and
  −0.025 p=1 are both within noise; the robust claim is "no material effect," not
  a precise negative magnitude.
- Both runs are temp=0 (greedy). temp>0 deployment trajectories were not tested.

## What this means for the Stage 2 decision

Stage 0 + Stage 1 together: the acceptance deficit is **shallow and in the right
neighborhood (Stage 0, robust)**, and **not cleanly attributable to tap-layer
precision (Stage 1, underpowered + partial-precision variant)**. A drafter
fine-tune on the served target's distribution remains a **plausible** lever with
meaningful headroom (top-2 coverage ≈0.91 at p=1), but it is **not proven** as the
exclusive or sufficient path, and the realized gain is unknown until Stage 2.
See the final recommendation.

## Adversarial review (codex / gpt-5.5, xhigh) — retained

Read-only review via the `adversarial-codex-review` skill; prompt/final/trace
retained at
`artifacts/exactness_small_bundles_q4tap/comparison_vs_baseline/{codex_review_prompt.md,codex_review.md,codex_review_output.txt}`.
The dispatcher independently re-verified codex's decisive recomputes — all matched
exactly: per-position match deltas (p1 −2.5 → p5 +8.75 pp); E[a|5block] per-prompt
bootstrap 95% CI [−0.16, +0.41]; p=1 McNemar discordants 7/5; and codex's check
that `token_embd.weight`/`output.weight` SHA-256 are identical across the two
GGUFs (confirms embed/lm_head unchanged → clean A/B). Summary of verdicts:
- **Sound:** D1 (fresh capture, recomputed), D2 (clean A/B; embed/lm_head SHA
  identical; spot re-run reproduced stored rows).
- **Questionable → refined:** D3 (p1 flat but study underpowered — prompt-bootstrap
  CI [−12.5, +6.25] pp), D4 (+0.10 not proven noise — bootstrap CI straddles 0 with
  a one-sided deeper-position trend).
- **Likely-wrong as stated → softened:** D5 ("tap-localized quant mismatch
  falsified" overstates — only this variant is shown not to help; P1 unresolved),
  D6 ("fine-tuning the only lever / +10 pp ceiling" overstates — top-2 coverage is
  headroom not a hard ceiling; "calibration error" is an inference).
- **Bottom line (codex):** "No observed material Q4-tap effect" survives; the
  falsification (D5) and the fine-tune reframing (D6) are overstated and softened
  above. No conflict with the descriptive result.
