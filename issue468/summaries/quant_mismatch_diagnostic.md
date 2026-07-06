# Stage 0 — quantization-mismatch error-shape diagnostic

Date: 2026-07-06. Status: **complete; Stage 0 clears → proceed to Stage 1.**
Harness: `issue468/run_stage0_quant_mismatch.py` (research instrumentation, reuses
`dspark_oracle/`). Artifacts: `issue468/artifacts/quant_mismatch_diagnostic/{summary.json,per_position.csv,per_position.json}`.

## Question

Is the IQ2XXS quantization mismatch (drafter distilled on the FP teacher, served
against the Q2 target) a **material, recoverable** contributor to draft misses —
i.e. does the error look shallow/perturbative (fine-tunable) or deep (drafter
capacity)? Gating decision for whether to run Stage 1 (tap-precision acceptance)
and whether to recommend Stage 2 (drafter fine-tune).

## Method

Re-ran the DSpark drafter over the **10 temp=0 exactness bundles** (80 measure
steps; temp=0 isolates drafter-argmax vs Q2-argmax with no sampling noise).
Captured the drafter's **full per-position decision score** `base_logits +
markov_bias` (the stored `oracle_ref.npz` logits are base-only and single-step,
unusable for ranks), and recorded per (bundle, step, draft-position p=1..5):

- `rank_target_in_drafter` — rank of the Q2 target token in the drafter's full score
- `rank_drafter_in_q2` — rank of the drafter's pick in the Q2 target's own top-128
- `q2_top1_top2_gap` — Q2's own top1/top2 logit gap (near-tie indicator)
- top-k coverage and a greedy-spine top-k counterfactual prefix acceptance

**Fidelity (3 independent checks, all passed):**
1. Reproduced draft tokens == retained `oracle/acceptance_summary.json` at every step (0 mismatches) — the re-run forward is faithful to the retained measurement.
2. Markov-bias reconstruction exact: `argmax(base+bias) == rollout token` held at all 400 positions.
3. Two numeric anchors hit exactly: p=1 match-rate = **0.8125** ≡ the speedup model's S(1); k=1 counterfactual E[a|5block] = **2.3375** ≡ the model's E[a|5]=2.338 (and S5 full-accept 0.1625 ≡ S(5)).
4. Headline aggregates independently recomputed from `per_position.csv` match `summary.json`.

## Result — p=1 (the clean signal; conditioned only on the real anchor, 80 positions, 15 misses)

| metric | value |
|---|---|
| match rate (current top-1) | **0.8125** |
| **median rank of target on misses** | **1.0** |
| misses with target ≤ rank 2 / 3 / 5 / 10 | **73% / 80% / 87% / 100%** |
| misses with target > rank 50 | **0%** |
| top-k coverage k=1/2/3/4/5 | 0.8125 / **0.9125** / 0.95 / 0.9625 / 0.975 |
| drafter's pick in Q2 top-3 (on misses) | **80%** (median rank-in-Q2 = 1) |
| Q2 near-tie on misses (gap < 1 nat / < 2 nat) | 27% / 40% (median gap 2.15 nat) |

**The error is overwhelmingly shallow and in the right neighborhood.** When the drafter
misses at p=1, the correct token is its **rank-2 pick** in the median case (73% of
misses it is rank ≤ 2; 100% within top-10; 0% beyond rank 50). And in **12/15** p1
misses the Q2 target itself ranked the drafter's token within its own top-3 (in 10
of those the drafter picked Q2's exact rank-2 token) — the drafter and Q2 agree on
the top neighborhood and merely order the top-2/3 differently.

**Mechanism caveat (important — refined after adversarial review, see below):** the
shallow ranks are NOT by themselves proof of a quant *flip*. Only ~27% of p1 misses
(4/15) are genuine Q2 near-ties (gap < 1 nat); the median Q2 top1/top2 gap on
misses is **2.15 nat**, i.e. Q2 is usually *confident* about its pick. Classic
quant flips produce *small* near-ties, not large confident preferences — so most of
these misses look more like "drafter is one rank off in a neighborhood where Q2 (and
likely FP) is confident" than like "quant noise flipped a coin-flip." The strong
*causal* claim ("quant-flip-like") is therefore **downgraded**; what stands is the
*shallowness* signal. Crucially, shallowness carries the recoverability argument
regardless of root cause: whether the miss is a quant flip or a drafter-calibration
error, the target sits at drafter rank 1–2 and the drafter picks the wrong one of
the top few — exactly where a Q2 fine-tune has maximum leverage (it teaches the Q2
argmax ordering). The *cause* (quant vs calibration) is what Stage 1 tests; the
*recoverable shape* is what Stage 0 establishes.

## Per-position decay (p=2..5, supporting; rollout-compounded)

| p | match rate | median rank on miss | top-2 cov | note |
|---:|---:|---:|---:|---|
| 1 | 0.8125 | 1.0 | 0.9125 | clean (anchor-conditioned) |
| 2 | 0.6875 | 2.0 | 0.80 | still close |
| 3 | 0.4875 | 4.0 | 0.725 | degrading |
| 4 | 0.3625 | 49 | 0.39 | rollout-diverged |
| 5 | 0.20 | 40.5 | 0.39 | rollout-diverged |

Match rate and rank-on-miss degrade with depth. This is the **autoregressive
rollout-compounding** effect (after the first miss the drafter rolls on its own
wrong token, so its context diverges) — a separate mechanism from the single-step
quant mismatch, and expected. The single-step signal is p=1 (and p=2 is still
healthy). Caveat: deeper-position ranks are confounded by divergence, so the p=1
numbers are the load-bearing evidence.

## Greedy-spine top-k counterfactual (acceptance headroom, NOT realized speedup)

E[a|5block] if each position accepted the target whenever it is in the drafter's
top-k along its own greedy rollout:

| k | E[a\|5block] | S5 full-accept |
|---:|---:|---:|
| 1 (current) | **2.3375** | 0.1625 |
| 2 | **2.875** | 0.225 |
| 3 | 3.2125 | 0.2875 |
| 4 | 3.2875 | 0.30 |
| 5 | 3.3625 | 0.3125 |

**Interpretation guard-rails (important):**
- This is **acceptance headroom**, not realized throughput. Realizing it via a
  draft **tree** does *not* work: per `spec_speedup_model.md` Q3, trees are
  verify-cost-dominated (4-node ceiling +2.2%; the bandwidth floor is unchanged).
  So the +0.54 drafts/cycle at k=2 does **not** translate to speedup via trees.
- The counterfactual's relevance to the **fine-tuning** hypothesis is different:
  it shows the drafter is **consistently one rank off** (median 1, 100% within
  top-10 at p=1), which is precisely the regime where a fine-tune has maximum
  leverage (nudging the top of the ordering). Top-2 coverage of 0.9125 at p=1 is
  essentially the ceiling a good Q2-fine-tune could approach at the first position.

## Verdict: Stage 0 CLEARS (to Stage 1), on shallowness + headroom — not on proven quant causality

Against the goal's clearing criteria:
- **Low median rank on misses (shallow):** ✓ decisively — median 1.0, 100% within top-10, 0% beyond rank 50.
- **Top-k ceiling materially above current:** ✓✓ — +10 pp at k=2 alone (0.8125→0.9125), +16 pp at k=5.
- **Substantial near-tie fraction:** ◐ weak — only 27% < 1 nat (median gap 2.15 nat); the perturbative signature is real but the *quant-flip* attribution is not established.

None of the "does NOT clear" conditions (deep misses, rare near-ties, no headroom)
hold — the dominant signals (shallow rank, large headroom) are robust and
independently confirmed. The error shape is **consistent with** quant mismatch
being a recoverable contributor (necessary-but-not-sufficient) and is **decisively
shallow**, which is the property that makes fine-tuning effective. **Proceed to
Stage 1.** (An adversarial codex/gpt-5.5-xhigh review independently affirmed that
"proceed to Stage 1" survives; it correctly downgraded the overclaimed causal
framing — see below. It did not conflict with the clearing *decision*.)

## What Stage 1 should now test

Stage 0 shows the error is shallow (good for fine-tuning) **regardless** of
mechanism. Stage 1 (drafter acceptance against the `Layers37-42Q4KExperts`
tap-precision variant) is a **partial proxy** for the decisive causality test
(codex's proposed test: run the FP teacher's top-k on the exact p1 miss set and
check whether FP agrees with the drafter while Q2 agrees with the target — we lack
an FP/Q8 target locally, so Q4-at-tap is the available proxy). It tests whether
**raising tap-layer precision converts rank-2 misses into rank-1 matches**, i.e.
whether the perturbation is localized to the tap layers. Two outcomes:
- **Acceptance jumps** (p=1 → ~0.88+): mismatch localized to the tap region is
  supported → fine-tuning on Q2 hidden states is strongly motivated (input-shift
  mechanism (A) supported).
- **Flat**: the rank-2 flips originate in lower layers (still Q2 in the variant) or
  are drafter-calibration error — but Stage 0's shallowness *still* supports
  fine-tuning (the drafter need only learn the Q2 argmax ordering, wherever the
  perturbation enters). So Stage 1 is informative for *mechanism/localization*, but
  the fine-tuning case does not live-or-die on it. **Set expectations accordingly:
  a flat Stage 1 is ambiguous, not a disproof.**

## Adversarial review (codex / gpt-5.5, xhigh) — retained

A read-only adversarial review was run via the `adversarial-codex-review` skill;
prompt + final + full trace retained at
`artifacts/quant_mismatch_diagnostic/{codex_review_prompt.md,codex_review.md,codex_review_output.txt}`.
The dispatcher independently re-verified codex's two load-bearing recomputes
(k=1 prefix hist `{0:15,1:13,2:18,3:11,4:10,5:13}`→E=2.3375; p1-miss Q2 gaps 3/4/6
under 0.5/1/2 nat, median 2.1506) — both matched exactly. Summary of verdicts:
- **Sound:** C2 (markov reconstruction exact), C3 (anchor arithmetic, recomputed),
  C4 (rank formula correct, all targets in-vocab), C5 (alignment verified
  algebraically against `target_topk.steps[s].selected.id`).
- **Questionable / accepted-as-refinement:** C1 (crosscheck is real but shares the
  drafter forward with the retained tool → a shared bug would pass) — addressed by
  persisting `drafter_top64_scores.npz` (331/400 ranks now re-derivable from
  artifacts alone, 331/331 reproduced); C6 ("quant-flip-like" overclaimed given the
  large median Q2 gap) — accepted, causal claim downgraded above; C7 (greedy-spine
  counterfactual is acceptance headroom, not a realized-throughput ceiling) — kept,
  caveats sharpened.
- **Bottom line (codex):** "Proceed to Stage 1" survives; "clears *because*
  overwhelmingly perturbative" does not survive as stated → downgraded to "p1 misses
  are shallow on this small temp=0 corpus; causality/recoverability consistent but
  not proven." No conflict with the clearing *decision*.

## Caveats

- 10 prompts (temp=0), 80 measure steps, 400 draft positions. Small corpus; the
  p=1 miss set is n=15. The rank distribution is tight (IQR 1–2.5) so the
  "shallow" conclusion is robust, but exact fractions carry sampling noise.
- temp=0 only (by design — isolates argmax mismatch). temp>0 would add sampling
  noise to "miss" and is out of scope for this diagnostic.
- `rank_drafter_in_q2` is capped at the Q2 top-128 (13% of p=1 misses had the
  drafter's pick outside Q2's top-128 → these are the少数 genuine deep-neighborhood
  misses; even there, the *target* was within the drafter's top-10).
- The drafter's input is the Q2 layer-42 hidden state, so this diagnostic measures
  the **total** Q2 mismatch (input shift + label shift) the drafter suffers — which
  is the quantity fine-tuning on Q2 would address.
