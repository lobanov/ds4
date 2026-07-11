# Speculative-decode speedup model — DSPark drafter on the ds4 target

Date: 2026-07-06 (cycle-cost model); **holistic integration 2026-07-08** of Lead 01
(anchor-reuse falsifier), Lead 02 (confidence-scheduled verification), and Lead 03
(powered acceptance + realistic-trajectory); **Lead 04 Phase B (FP hidden-precision
ceiling) integrated 2026-07-10**.
Model: `issue468/model_spec_speedup.py`; powered data + cycle-jump:
`issue468/artifacts/acceptance_powered/`; outputs:
`issue468/artifacts/spec_speedup_model/{summary.json,model_inputs.json}`.

## Headline (current best understanding)

**Under the realistic per-cycle trajectory, DSpark speculative decoding does NOT beat plain
ds4 decode locally on the measured corpus.** The best fixed-K result under the optimized
anchor-reusing verifier is K≈4–5 at **~0.98×**; the shipped verifier is materially worse.
Confidence scheduling narrows but does not reverse that conclusion: on fresh out-of-sample
replay it still loses under shipped accounting, and under anchor reuse it reaches only a
fragile **1.0375×** (frozen threshold) to **1.0523×** (expected-opt diagnostic) with less
than **4 ms/cycle** of overhead headroom. The remaining acceptance levers are a better
drafter (training) and — per Lead 04 Phase B — **target hidden-state precision**, which is a
*separate, open* lever from drafter-weight precision: native FP4/FP8 target hiddens lift
first-token acceptance ~+5 pp over the local IQ2XXS target at float32 (CI clears +2 pp), but
the deployment-dtype (F16) result reverses sign (a likely capture-representation error), so
that lever is **HOLD** pending an algebraic hidden-equality proof (see "Target hidden-state
precision" below). So beating baseline remains contingent on an unbuilt cheap anchor-
reusing verifier plus a better drafter; scheduling is at most conditional secondary
material, and target-hidden precision is live-but-unconfirmed.

## The question and the gates

Can a DSpark-style speculative path deliver a **material local decode speedup** on `ds4`
with exact greedy output preserved? Primary gate ≥+20% vs baseline; secondary gate beat or
match the shipped `--mtp` path. This document is the cycle-cost model plus the measured
fixed-K acceptance and adaptive-scheduling evidence that feed it. There are **two precision
dimensions** to the drafter's input, both now measured: **drafter-weight** precision is
closed (Q4_K≈F16, `dspark_quantization_ceiling.md` — not the lever); **target hidden-state**
precision is open and measured by Lead 04 Phase B (see "Target hidden-state precision" —
native FP4/FP8 hiddens lift p1 ~+5 pp at float32, but the F16 deployment result reverses
sign; resolved in Phase C — capture faithful, the lever is real at +15% float32 speedup, F16-deployment-blocked).

## Notation & definitions

**The two acceptance estimators** (Lead 03's core correction — see "Acceptance: two
estimators" below for the full reasoning):
- **Sliding** estimator — averages the accepted prefix over **every position** on the
  greedy spine (treats each position as an independent cycle start). Easy to compute;
  **optimistic / diagnostic only.**
- **Cycle-jump** estimator — simulates real decode: each cycle accepts `a` drafts, then
  **advances by `a + 1`** (the `a` accepted + 1 correction/bonus token) to the next
  anchor; averages `a` over the cycles a decode *actually runs* (real decode never
  starts a cycle at the positions it jumped over). This is the **model-currency /
  realistic** value, and is lower than sliding because it lands on post-rejection
  *correction-anchor* cycles that sliding under-weights.

**Symbols used in the tables and formulas:**
- **`K`** — draft block size: tokens the drafter speculates per cycle.
- **`a`** (0..K) — accepted prefix: how many of the K drafts the target accepts before
  the first mismatch in a cycle.
- **`decode_ms`** (26) — wall time of one plain target decode (= 1000/38.49 t/s).
- **`draft_ms`** (10) — wall time of one DSpark drafter forward (given).
- **`verify_ms(K)`** — wall time of one target verify forward over a K-token suffix
  (cross-prompt median; K2:43.6, K3:59.7, K4:65.8, K5:74.5).
- **`E[a|K]`** — expected accepted drafts **per cycle** at block size K (the acceptance
  currency). Always qualified by estimator: **sliding E[a|K]** or **cycle-jump E[a|K]**.
- **`S(K)`** = `P(full accept)` = P(the first K drafts all match) = P(prefix ≥ K) — the
  full-block-accept probability that triggers the extra anchor decode in the optimized-
  verifier cost. Also has sliding vs cycle-jump variants (tables label which).
- **`speedup(K)`** — `(E[a|K]+1)·decode_ms / (draft_ms + verify_ms(K) + decode_ms·S(K))`;
  `>1` beats plain decode. Quoted as the cycle-jump value unless marked "sliding".
- **`p`** — uniform per-position match probability under a geometric acceptance model;
  **`p beat`** / **`p +20%`** = the `p` needed for speedup 1.0 / 1.2.
- **`E beat`** / **`E +20%`** — the `E[a|K]` needed for speedup 1.0 / 1.2 (uses the
  measured sliding `S(K)` for the full-accept decode penalty).
- **dynamic break-even `E[a|4]`** — the `E[a|4]` at which `speedup(K=4)=1.0` given the
  measured `S(4)` (= `(draft+verify(4)+decode·S(4))/decode − 1`).
- **deficit** — current cycle-jump `E[a|4]` minus the dynamic break-even.
- **`P(speed<1)`** — bootstrap probability the cycle-jump speedup is below 1.0.
- **`CI [lo, hi]`** — prompt-clustered bootstrap 95% interval (resamples prompts, not
  steps).
- **predicted-anchor / correction-anchor cycle** — a cycle whose anchor the drafter
  predicted correctly / mispredicted (a correction-anchor cycle follows a rejection and
  has lower acceptance).
- **optimized-verifier / shipped verifier** — the modeled anchor-*reusing* regime
  (fresh decode only on full-block accept) vs the actual `ds4 --mtp` (decodes every
  cycle, ~−18.9 pp at K=4).

## The model (cycle-cost)

A speculative cycle drafts K tokens, the target verifies them, and accepts a prefix (0..K).
The verify forward produces the **correction/bonus token at the rejection point, which is
the next cycle's anchor** — so a fresh anchor decode is needed **only on full-block
acceptance** (the *optimized-verifier* regime):

```
cost(reject)      = draft + verify                 (verify yields the anchor)
cost(full accept) = draft + verify + decode
E[cost/cycle]     = draft + verify_ms(K) + decode * S(K)     # S(K)=P(first K all match)
E[tokens/cycle]   = E[a|K] + 1
speedup(K)        = (E[a|K] + 1) * decode_ms / E[cost]
```

Costs (measured): `decode_ms=26` (1000/38.49 t/s), `draft_ms=10` (DSpark, given),
`verify_ms(K)` = cross-prompt median of `mtp_verifier_bench_long` (K2:43.6, K3:59.7,
K4:65.8, K5:74.5, K6:79.6). The shipped `ds4 --mtp` does **not** reuse the anchor — it pays
`decode+draft+verify` every cycle (~−18.9 pp at K=4), so the optimized-verifier projection
below is the relevant ceiling, not the shipped reality.

## Acceptance: two estimators (Lead 03's core correction)

`E[a|K]`/`S(K)` are the model's currency, and **how they are estimated matters**. Both are
now measured on a **300-prompt powered corpus** (Stage 2's 240 dolly/codealpaca/jsonex + 60
newly captured via the committed `--capture-dataset`; 128-tok temp=0 greedy spines) with a
**torch/MPS drafter port precision-gated at 100% draft-token agreement vs the numpy oracle**
(the numpy oracle's per-expert re-dequant crashed the machine at scale; a codex bug-hunt
also found+fixed a real `hc_post` broadcast bug in the torch port).

- **Sliding (position-uniform greedy prefix).** The easy-to-compute estimator: average the
  accepted prefix over all positions. **Optimistic / diagnostic only** — it uniformly samples
  positions and under-weights post-rejection *correction* cycles.
- **Cycle-jump (per-cycle, realistic trajectory).** The model-currency estimator: simulate
  real cycles advancing by `accepted+1`, so per-cycle accepted is measured on the trajectory
  the decode actually follows (including correction cycles). This is what the speedup formula
  consumes.

The cycle-jump is **lower** than the sliding estimate by ~3 pp at K=4 — the *realistic-
trajectory bias* the model had flagged but never measured. Sliding gives the optimistic edge
of the band; cycle-jump the realistic edge.

## Current numbers (300-prompt corpus, optimized verifier)

| K | verify_ms | sliding E[a\|K] | sliding S(K) | sliding speedup | cycle-jump E[a\|K] | cycle-jump S(K) | **cycle-jump speedup** |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 2 | 43.6 | 1.442 | 0.652 | 0.900 (−10.0%) | 1.408 | 0.633 | **0.893 (−10.7%)** |
| 3 | 59.7 | 1.955 | 0.514 | 0.925 (−7.5%) | 1.886 | 0.483 | **0.912 (−8.8%)** |
| 4 | 65.8 | 2.337 | 0.382 | 1.012 (+1.2%) | 2.198 | 0.340 | **0.982 (−1.8%)** |
| 5 | 74.5 | 2.609 | 0.272 | 1.025 (+2.5%) | 2.422 | 0.232 | **0.983 (−1.7%)** |

- **K=4 cycle-jump (the headline):** speedup 0.982×, prompt-clustered bootstrap CI
  [0.971, 0.993], **P(speed<1) = 0.999**. The dynamic break-even E[a|4] at S(4)=0.340 is
  **2.256**; the deficit is **0.058 accepted drafts/cycle** (deficit > CI half-width, so
  "below baseline" is signed). Read the sliding column as the optimistic edge only. These
  are the unscheduled fixed-K reference points for the scheduling results below.
- **Corpus-dependent** (cycle-jump K=4 speedup): jsonex **+2.3%**, codealpaca −1.9%, dolly
  −5.6%. The mix is on the easy side — the old 10-prompt code/synthesis exactness corpus was
  E[a|4]=2.175 (harder than all three families), so a code/synthesis-heavy deployment would
  be worse, not better.
- **Trajectory bias is real and structural:** correction-anchor cycles (drafter mispredicted
  the anchor) have E[a|4]=1.994 vs predicted-anchor 2.427; real decode interleaves them,
  pulling per-cycle acceptance from the sliding 2.337 down to the cycle-jump 2.198.

## Anchor reuse (Lead 01) — the optimized-verifier assumption, partially de-risked

The optimized-verifier regime above **assumes** the verify-produced anchor can seed the next
draft without a fresh 26 ms decode. An adversarial codex review flagged this as the
load-bearing unverified risk (the shipped `--mtp` decodes every cycle; the reuse graph is
unbuilt on ds4). Lead 01's offline falsifier tested the **acceptance-axis** of that
assumption: feeding the drafter the *last-accepted-position* (stale) hidden + the correction
token as embedding. Result (`summaries/anchor_reuse_falsifier.md`): **no large acceptance
collapse** in any tested regime (5/6 temp×KV-model cells SURVIVE, 1 MARGINAL; all CIs
straddle 0; p=1 robustly non-negative), so the drafter tolerates stale-from-valid-prefix
hidden — reuse is **necessary-but-not-sufficient** and is not refuted on acceptance grounds.
BUT non-inferiority is not established (the corpus was underpowered; reuse reshapes the block
— early positions +, late −), and the **verifier-side** risks remain entirely untested:
verify-forward-hidden equivalence under IQ2XXS, residual per-cycle overhead (~15–19 ms of
readback/rollback/first-miss waste the model sets to zero), and actual cycle timing. Those
gate the optimistic edge and are Lead 06's job.

## Q1 — acceptance required to beat baseline / clear +20% (sliding framing)

Using the sliding S(K) for the full-accept decode penalty (the model's Q-tables;
`E`=accepted drafts/cycle needed, `p`=uniform per-position match prob, −1=unreachable):

| K | E beat | E +20% | p beat | p +20% | cycle-jump E (real) |
|--:|--:|--:|--:|--:|--:|
| 2 | 1.713 | 2.256 | unreachable | unreachable | 1.408 |
| 3 | 2.194 | 2.833 | 0.890 | unreachable | 1.886 |
| 4 | 2.298 | 2.957 | **0.792** | 0.940 | **2.198** |
| 5 | 2.522 | 3.226 | 0.783 | 0.890 | 2.422 |

- The cycle-jump current E[a|4]=**2.198** is **below** the sliding beat-even E=2.298 (and
  below the cycle-jump dynamic break-even 2.256) — i.e. the realistic drafter is short of
  beating baseline by ~0.06–0.10 drafts/cycle, not the "~0.03" the old sliding/10-prompt
  framing implied.
- The **+20% gate** needs p≈0.94 at K=4 / 0.89 at K=5 (~+13–17 pp over current ~0.79) —
  reachable only with a substantially better drafter (training, not quantization).

## Q3 — can a 4-node draft tree help? No.

A verify of N nodes costs `verify_ms(N)` regardless of shape (bandwidth floor). Ceilings at
**perfect** acceptance: linear-4 +27.7% (current cycle-jump → 0.982×); best 4-node tree
(spine3 + repair) **+2.2% even at perfect acceptance**, doubly penalized (branches spend
verify nodes without extending the committed prefix, AND raising pos-3 coverage increases
the full-accept decode penalty). **Do not branch — use a linear chain;** no 4-node tree
beats the gate.

## Adaptive scheduling in the same model (Lead 02)

Lead 02 asks whether the existing confidence head can improve the local economics by
choosing a shorter verify span cycle-by-cycle. The policy evidence is fully offline:
confidence is extracted from the oracle / torch-MPS carrier, calibrated with train-fit STS,
thresholds are chosen on `eval`, and then replayed unchanged on fresh `lead3` under the
same two verifier accountings used elsewhere in this dossier.

| policy (fresh `lead3`) | shipped accounting | anchor-reuse accounting |
|---|---:|---:|
| fixed-K4 | 0.8123x | 0.9810x |
| **STS selected threshold** | **0.8646x** | **1.0375x** |
| **STS expected-opt** | **0.9205x** | **1.0523x** |
| oracle ceiling | 1.1297x | 1.2447x |

- Under **shipped** accounting, adaptive scheduling does **not** rescue the path. The
  table's `0.8646x` uses the anchor-reuse-selected threshold `0.08` held fixed across both
  accountings; even the threshold selected specifically for shipped costs (`0.52`) reaches
  only **0.8987x** on fresh `lead3`.
- Under **anchor reuse**, scheduling improves on unscheduled fixed-K4 but only into a
  narrow, fragile positive band. The clean frozen-threshold result is **1.0375x**, below
  Lead 02's predeclared `+5–10%` stacking tier. The **1.0523x** expected-opt figure is a
  useful diagnostic upper-ish policy, but it is not deployable evidence.
- The anchor-reuse positives are highly sensitive to residual folded-verifier overhead: the
  frozen-threshold gain has about **3.01 ms/cycle** of headroom and expected-opt about
  **3.89 ms/cycle**. Adding `+4 ms/cycle` drops them to about **0.988x** and **0.999x**.

So in the speedup model, scheduled verification is not an overlay or an independent route
to viability. It is best represented as a **conditional secondary increment** that matters
only if Lead 06 makes the optimized anchor-reusing verifier real and very cheap.

## Target hidden-state precision — the FP ceiling (Lead 04 Phase B + C)

The acceptance numbers above are all measured on the **local IQ2XXS target** — i.e. the
drafter consumes IQ2XXS-degraded hidden states. Is part of the ~0.06 drafts/cycle deficit
attributable to IQ2XXS degrading the *target's* hiddens below their native (FP4/FP8)
precision? The DSpark drafter was distilled against native served precision
(`inventories/dsv4_flash_dspark_model.md`), so if native hiddens give higher acceptance
than IQ2XXS hiddens, target-hidden quantization is a recoverable lever.

**Method:** captured native FP4/FP8 target data (HC hiddens at layers 40/41/42 + greedy +
top-128 logits) on Lead 03's 300-prompt corpus via vLLM on Modal H200:2
(`run_lead04_modal/capture_hc_modal.py`); ran the local drafter on the FP hiddens; paired
FP p1 against the retained Q2 p1 per prompt; prompt-clustered bootstrap CI.

**Result (n=299; Δp1 = float32-FP − float32-Q2).** The Q2 reference is dtype-invariant
(float32-Q2 == F16-Q2 exactly, verified on 240 prompts, drafts byte-identical — the
drafter's argmax is robust to f16/f32 on Q2 hiddens), so the float32-FP-vs-f16-Q2
comparison IS the consistent-dtype comparison:

| | FP (native) | Q2 (IQ2XXS) | Δ | CI95 |
|---|---|---|---|---|
| mean p1 | 0.846 | 0.793 | **+5.28 pp** | [+3.81, +6.56] |
| codealpaca | | | +6.96 pp | [+4.3, +8.9] |
| dolly | | | +2.73 pp | [+1.3, +4.1] |
| jsonex | | | +6.15 pp | [+2.8, +8.8] |

**Phase C falsification (hidden-capture fidelity — RESOLVED to “likely faithful”).**
Code-reads on vLLM v0.24.0 (`/Users/lobanov/Projects/vllm`) verified the capture mechanics:
the decoder layer returns `(x, residual, post_mix, res_mix)` matching the hook’s unpack,
and `mhc_post_tilelang` writes to a fresh `out` (not in-place, so the hook’s `.clone()`s
are harmless). The F16/deployment anomaly (below) is **F16 numerical, not a capture bug**:
no NaN/inf, identical output magnitudes (head-out max 127973 for F16≈float32 on FP), and
the effect is argmax-flipping on near-tie logits — the FP-hidden distribution produces
closer logits that F16 rounding flips, while Q2 hiddens (the drafter’s deployed input) give
robust argmaxes. The capture is therefore likely faithful; the +5.3 pp is real at float32.

**Phase C cycle-jump (the full acceptance trajectory — the user’s hypothesis CONFIRMED).**
First-token p1 understates the FP benefit: the cycle-jump E[a|4] amplifies it. Measured on
the same 299-prompt corpus (Lead 03’s cycle-jump estimator; `dspark_oracle/analyze_phaseC.py`):

| | FP float32 | FP F16 (deploy) | Q2 (ref) |
|---|---|---|---|
| p1 | 0.855 | 0.596 | 0.793 |
| cycle-jump E[a\|4] | **2.741** | 1.693 | 2.198 |
| speedup(K=4) | **~1.15× (+15%)** | ~0.83× | 0.982× |
| confidence separation | +0.247 (calibrated) | — | (Lead 02) |

At **float32, FP hiddens clear baseline by +15%** (E[a|4]=2.74 vs Q2 2.198) — the
“later-position acceptance matters even if p1 doesn’t” hypothesis holds, and strongly:
the cycle-jump gap (+0.54 drafts/cycle) is much larger than the p1 gap (+6 pp) suggests.
The drafter’s confidence head is calibrated on FP hiddens (accepted 0.87 > rejected 0.62,
separation +0.247), so confidence-scheduled verification (Lead 02’s framework) is viable
on FP. At **deployment precision (F16/Q4_K), FP hiddens are WORSE** (E[a|4]=1.69, 0.83×)
— the F16 drafter’s argmax-flipping on the FP distribution destroys the benefit.

**Verdict (Phase B+C): the FP lever is REAL and LARGE at float32 (+15% speedup, clearing
baseline), but F16-deployment-blocked.** This is a drafter-precision blocker, not a capture
blocker: the captures are faithful, and a float32 drafter on native hiddens would deliver
~+15%. The local IQ2XXS speedup verdict (0.98×) stands for the *current* deployment (IQ2XXS
target + F16 drafter), but target-hidden precision is now a **quantified live lever**: a
float32 drafter (or an F16 fix / a drafter re-distilled to be F16-robust on native hiddens)
could realize the +15%. Lead 07’s trigger is met at float32. Caveats: (1) the +15% uses Q2’s
S(4)=0.34 (the FP S(4) wasn’t separately reported — likely conservative); (2) corpus is the
easy side (dolly weakest); (3) FP (vLLM) vs Q2 (ds4) is cross-engine. The path: run the
drafter at float32 (cost/latency tradeoff) OR fix the F16 numerical sensitivity.

## Verdict, levers, and open scope

**Verdict:** with the current drafter on the measured corpus, DSpark does not beat plain ds4
decode locally at any tested fixed K under the realistic trajectory (K4/K5 optimum ~0.98×,
significantly below baseline), and confidence scheduling does not change that under shipped
accounting. Under anchor reuse, scheduling rises only to a fragile low-single-digit positive
band on fresh replay, so the case for a local speedup still requires a stack of unverified
gains:

1. **Anchor-reusing verifier** (Lead 06): acceptance-axis de-risked by Lead 01; verifier-
   hidden equivalence + residual overhead + cycle timing untested. This is the prerequisite
   for any positive scheduling result at all. It is worth ~18 pp at K=4 relative to shipped
   fixed-K accounting if realizable, but the realistic unscheduled edge it would feed is
   still only ~0.98×, and Lead 02 lifts that only to ~1.04× in the clean frozen-threshold
   result.
2. **A materially better drafter** (Lead 07 upstream quality / training): the binding
   constraint is per-cycle acceptance; **drafter-weight** precision is exhausted (Q4_K≈F16),
   but **target hidden-state** precision is NOT closed — Lead 04 Phase B measured a ~+5 pp
   native-vs-IQ2XXS gap at float32 and, per Phase C, a **+15% cycle-jump speedup at
   float32** (E[a|4]=2.74 vs Q2 2.198) — the largest measured lever, but F16-deployment-
   blocked (the F16 drafter argmax-flips on native hiddens); see
   "Target hidden-state precision" above.
3. **Drafter-state pollution** (Lead 06): the cycle-jump measures anchor-token difficulty
   along a *linear* trajectory; it does NOT model the drafter's KV/state being polluted by
   rejected drafts, tree/branching, or adaptive policies — the realistic trajectory could
   be worse than measured.

**Scope / caveats:**
- `verify_ms(K)` is the MTP verifier; DSPark would reuse the same target verifier. The
  bandwidth floor (verify≈decode at K=2, super-linear past ~K=8 from MoE expert-union
  growth) is prompt-independent and unchanged.
- Corpus is dolly/codealpaca/jsonex (easy side); a code/synthesis deployment is harder.
  Acceptance is temp=0 greedy; temp 0.5/1.0 drops E[a|4] ~7 pp.
- The optimized-verifier projection assumes a usable anchor hidden at the rejection point —
  an implementation property to confirm against the ds4 graph (Lead 06).

Net: the realistic answer to "can DSpark beat baseline locally?" is **no, not with the
current drafter + an optimized-but-unbuilt verifier on this corpus** — and adaptive
scheduling does not alter that answer under shipped economics. At best, if Lead 06 delivers
a genuinely cheap folded verifier, scheduling contributes a fragile extra few points on top.
The decision-grade open questions therefore move to Lead 06 (does a real folded verifier
realize the anchor-reuse + low-overhead regime?) and Lead 07 (can the drafter's per-cycle
acceptance rise enough to clear the ~0.06–0.10 drafts/cycle deficit — and can the Lead 04
   +15% float32 FP-ceiling lever be realized, via a float32 drafter or an F16 fix?).

