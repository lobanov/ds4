# Lead 02 — Confidence-scheduled verification (local, single-request)

Date: 2026-07-07. Status: resolved & archived 2026-07-08 (moved from
`issue468/pending/`). Result: see
`issue468/summaries/confidence_scheduled_verification.md`. Depends on: lead 01
for the anchor-reuse accounting (the shipped-verifier variant of this lead is
still informative if lead 01 falsifies, but only reaches ~break-even).

## Execution contract

Pursue this lead under the general research methodology in
`.pi/skills/research-lead/SKILL.md`:

- orient and verify all claimed assets before acting;
- lock the decision rule and estimator before trusting measurements;
- fidelity-gate any new confidence-extraction or replay path against retained
  oracle / powered-acceptance tooling;
- report prompt-clustered CIs and per-source results;
- propagate the canonical result through `summaries/` and `STATUS.md`; and
- archive the lead correctly on completion.

Treat this as decision-grade work with two adversarial review gates following
`.pi/skills/adversarial-codex-review/SKILL.md`:

- **setup gate:** after the extraction / calibration / simulation path exists,
  but before trusting measurements; and
- **verdict gate:** before recording the conclusion.

Each gate should use an independent `gpt-5.5` `xhigh` subagent review that
re-derives claims from code and artifacts. Any decisive subagent finding must be
independently verified before acceptance.

## Rationale

Scheduled verification is the third component named in `GOAL.md` and the only one
never explored. The DSpark confidence head exists in the vendored GGUF
(`mtp.2.confidence_head.proj.weight`) and `dspark_oracle/forward.py` already
computes its scores ("for completeness") — no experiment has ever used them.

Two measured facts make per-cycle scheduling valuable *locally*, independent of
the paper's batch-capacity argument:

- **First-draft misses are the largest single leakage**: 17–50% of cycles pay
  draft + verify(K) and accept zero drafts (`summaries/mtp_verifier_bench_results.md`).
  Under anchor-reuse accounting that is ~76 ms for one token vs 26 ms to decode it.
- **Verify is not flat in K** (~7–9 ms/position: 43.6/59.7/65.8/74.5 ms for
  K=2..5), so pruning doomed suffix positions has a real price attached. And
  under anchor reuse, rejection is the *cheap* outcome (correction token for
  free) while full acceptance forces a 26 ms decode — so the optimal length given
  expected survival `a` is ℓ = a+1 (harvest the correction).

**Oracle ceiling from retained numbers** (temp-0 prefix histogram
P(a=0..5) = .1875/.1625/.225/.1375/.125/.1625; verify_ms from the long bench;
draft=10, decode=26): an oracle scheduler yields 21.0 ms/token = **+23.8%** vs
baseline — the only configuration modeled anywhere in this dossier that clears
the +20% primary gate with the current drafter. Best fixed-K is −0.9%. Under
shipped-verifier accounting the oracle reaches only ~break-even (vs −18.9%
fixed), which is why lead 01 gates the headline. The realized number lands
somewhere in [−0.9%, +23.8%] depending entirely on the head's per-cycle
discrimination — the one unknown, and it is measurable offline.

Known risk: the head was trained/calibrated against the FP teacher; on the
IQ2XXS target its calibration is likely off (fixable — STS is a 1-D grid search)
and its discrimination may degrade (the killer; measured in step 1). Fallback if
the head is weak: the drafter's top1−top2 margin as a confidence proxy (Stage 0
showed misses concentrate at small margins).

## Content of work

All offline against retained artifacts; no ds4 engineering:

1. **Measure the head.** Extract per-position confidence scores across the 30
   bundles; label with actual accept/reject; report AUC and ECE per position;
   recalibrate with sequential temperature scaling on a held-out split.
2. **Per-cycle simulation.** Extend `model_spec_speedup.py` from aggregate S(K)
   to per-cycle replay: each bundle cycle scheduled by a threshold policy on the
   recalibrated cumulative confidence (skip-verify gate + adaptive ℓ, including
   the ℓ = â+1 correction-harvest rule), under both verifier accountings.
3. Report the realized-policy speedup with per-prompt spread, alongside the
   fixed-K and oracle bounds. Retain artifacts + summary in dossier format.

Estimated effort: ~2–4 days.

## Success criteria

- **Realized modeled speedup ≥ ~+10%** (anchor-reuse accounting): scheduling
  becomes a top engineering priority alongside lead 05, and the primary gate is
  live again on the current in-RAM setup.
- **+5–10%**: keep as a stacking lever (with fine-tune / lead 05), secondary-gate
  material.
- **< ~+5%**, or head AUC on the Q2 target materially below the paper's 0.81–0.90
  and the margin-proxy fallback also weak: drop the lead and record it as a
  closed direction.

## Worklog

### 2026-07-08 — execution contract: methodology and review gates locked

Added an explicit execution-contract section tying this lead to
`.pi/skills/research-lead/SKILL.md` and
`.pi/skills/adversarial-codex-review/SKILL.md`. This records that Lead 02 must
use the research-lead workflow, prompt-clustered / per-source reporting, a
fidelity gate on any new confidence-extraction path, and two independent
`gpt-5.5` `xhigh` adversarial subagent reviews (setup + verdict), with
independent verification of any decisive subagent findings before acceptance.

### 2026-07-08 — orientation: confidence-path extraction gap confirmed

Verified the first implementation blocker in the retained code paths: the
confidence-head weight is threaded through both the numpy oracle and the torch
head, but neither path currently exposes confidence scores to callers. In
`dspark_oracle/forward.py`, `forward_head(..., conf_proj, ...)` receives
`mtp.2.confidence_head.proj.weight` but returns only `output_ids` and base
`logits`; `measure_acceptance_bundle.py` and `run_stage0_quant_mismatch.py`
therefore discard confidence entirely. In `dspark_train/drafter_head.py`, the
torch port similarly returns only draft ids plus base logits and has no
confidence projection path at all.

Minimal first execution slice: extend both head implementations to emit
per-position raw confidence logits / sigmoid scores while preserving existing
draft-token outputs bit-for-bit, then add a small fidelity gate on the retained
temp-0 exactness bundles before any calibration or scheduler replay work is
trusted. The powered 300-prompt corpus from Lead 03 remains the preferred next
measurement carrier once this extraction gate passes.

### 2026-07-08 — implementation slice 1: confidence outputs exposed, ungated

Implemented the first extraction slice in code. `dspark_oracle/forward.py`
`forward_head()` now supports an opt-in confidence return path that emits
per-position raw confidence logits plus sigmoid scores while preserving the
existing `(output_ids, logits)` return for legacy callers. The torch head in
`dspark_train/drafter_head.py` now loads `mtp.2.confidence_head.proj.weight`,
reconstructs the confidence input as normalized hidden `h_k` concatenated with
the Markov embedding of the previous token, and can return the same
per-position confidence logits / scores behind an opt-in flag.

Also exposed a minimal retained-oracle harness surface:
`dspark_oracle/measure_acceptance_bundle.py` now has `--include-confidence` /
`include_confidence=True`, which appends `confidence_logits` and
`confidence_scores` to per-step rows without changing the default summaries.
Current status: syntax-checked (`python3 -m py_compile`) but **not yet
fidelity-gated** against retained exactness outputs, so these confidence values
are available for inspection but are not yet trusted for Lead 02 conclusions.

### 2026-07-08 — fidelity gate 1: retained oracle path PASSED

Added `run_lead02_confidence_fidelity.py` and ran it under the retained oracle
venv against the temp-0 exactness bundles. Result artifact:
`artifacts/lead02_confidence_fidelity/summary.json`. Gate condition: the
baseline retained-oracle summary must equal the confidence-enabled summary after
removing confidence-only fields. Outcome: **PASS on all 10 temp-0 bundles, 0
failures**; every row carried `confidence_logits` / `confidence_scores` of
length 5, and all previously trusted rows / prefix histograms / aggregate
summaries remained identical after stripping those fields.

Interpretation: the numpy-oracle confidence extraction path is now trusted as
observationally inert with respect to retained acceptance numerics. The torch
path is still ungated for confidence specifically; next decision is whether to
gate that path separately before using the powered 300-prompt corpus, or to use
the retained oracle as the canonical extraction path for calibration first.

### 2026-07-08 — fidelity gate 2: torch confidence path PASSED on 3-source sample

Added `run_lead02_torch_confidence_gate.py` to compare the torch confidence
path directly against the retained numpy oracle on `Stage2CaptureStore`
prompts, re-checking both draft-token agreement and confidence deltas. Ran the
gate on one prompt from each source family (`codealpaca_0000`, `dolly_0000`,
`jsonex_0000`) with `float32` on `mps`; artifact:
`artifacts/lead02_torch_confidence_gate/summary.json`.

Outcome: **PASS, 0 failures**. Draft-token agreement stayed at 100% for all
three prompts. Confidence deltas were tiny:

- codealpaca: logit MAE `6.57e-06`, score MAE `7.5e-07`
- dolly: logit MAE `8.97e-06`, score MAE `7.5e-07`
- jsonex: logit MAE `5.81e-06`, score MAE `6.3e-07`

Max absolute logit deltas stayed below `7.2e-05`; max score deltas below
`1.0e-05`. Interpretation: on this 3-source setup sample, the torch path is now
trusted as a powered-corpus carrier for Lead 02 confidence measurements, not
just for draft-token agreement.

### 2026-07-08 — measurement harness 1: powered-corpus confidence collection + smoke

Added `run_lead02_measure_confidence.py`, the first real Lead 02 measurement
harness over the powered `Stage2CaptureStore` carrier. It writes per-prompt
confidence rows (draft, target, prefix, per-position confidence logits/scores)
and an aggregate with per-position raw and cumulative AUC / ECE, using the
conditional label `1[prefix >= k]` at each draft position `k`.

Smoke run (MPS, float32) on one prompt per source family:
`codealpaca_0000`, `dolly_0000`, `jsonex_0000`. Artifact:
`artifacts/lead02_confidence_measure_smoke/aggregate.json`. Result: the raw
confidence head already shows meaningful discrimination on this tiny sample
(AUCs `0.75–0.84` across positions), but raw calibration is visibly weak at
deeper positions (raw ECE `~0.13–0.17`). The cumulative-prefix probabilities
improve calibration at positions 2–5 (e.g. p4 raw ECE `0.1747` -> cumulative
ECE `0.1244`; p5 `0.1329` -> `0.0778`) while preserving discrimination.

This is only a smoke, not a decision artifact, but it is the first direct
evidence that the confidence head is not degenerate on the served IQ2XXS target
and that STS is a live next step rather than speculative cleanup.

### 2026-07-08 — measurement harness 2: held-out eval run completed on MPS

Ran the same harness on the full held-out `eval` split (60 prompts, balanced
20/20/20 across `codealpaca` / `dolly` / `jsonex`) using the powered torch path
on `mps` with `float32`. Artifact:
`artifacts/lead02_confidence_measure_eval/aggregate.json`. Runtime after model
build settled at ~`13.8s/prompt`; aggregate coverage was **6897 steps**.

Held-out result: discrimination is consistently real across all five draft
positions, with raw per-position AUCs `0.7769 / 0.7711 / 0.7933 / 0.8039 /
0.8165` for p1..p5. Raw calibration degrades with depth (`raw_ece`
`0.0487 / 0.0977 / 0.1328 / 0.1510 / 0.1506`), but cumulative-prefix
probabilities materially improve both discrimination and calibration beyond p1:
for p2..p5, cumulative AUC rises to `0.7940 / 0.8141 / 0.8290 / 0.8399` and
cumulative ECE falls to `0.0746 / 0.0838 / 0.0792 / 0.0714`.

Interpretation: on held-out powered data, the confidence head remains
decision-relevant on the served IQ2XXS target. The main remaining problem is
calibration rather than collapse of ranking signal, which keeps sequential
temperature scaling as the correct next step before scheduler replay or verdict
claims.

### 2026-07-08 — STS fit 1: held-out cumulative calibration improved

Added `run_lead02_sts.py`, an offline calibration script that implements the
DSpark paper's left-to-right Sequential Temperature Scaling: at each draft
position `k`, perform a 1-D log-grid search over temperature `T_k` to minimize
the ECE of the cumulative-prefix probability `prod_{i<=k} sigmoid(logit_i / T_i)`
while keeping previously calibrated positions fixed. Artifact:
`artifacts/lead02_confidence_sts_eval/summary.json` plus
`artifacts/lead02_confidence_sts_eval/temperatures.json`.

Held-out fitted temperatures were `1.057 / 0.758 / 1.038 / 1.203 / 1.447` for
p1..p5. On the same 60-prompt `eval` set, cumulative-prefix calibration
improved at every position while discrimination stayed effectively unchanged:
overall cumulative ECE moved from `0.0487 / 0.0746 / 0.0838 / 0.0792 / 0.0714`
to `0.0465 / 0.0729 / 0.0790 / 0.0744 / 0.0643`, while cumulative AUC stayed
at roughly `0.777 / 0.794 / 0.814 / 0.829 / 0.840`.

Interpretation: the confidence path now has a concrete calibrated carrier for
scheduler replay. STS helps, but only modestly; the lead is still constrained
primarily by the head's underlying ranking quality and whatever acceptance gain
adaptive scheduling can actually harvest from these calibrated cumulative
probabilities.

### 2026-07-08 — replay path 1: adaptive policy only helps if anchor reuse is real

Added `run_lead02_replay.py`, which replays speculative decode on the realistic
cycle-jump trajectory (`next_step += accepted + 1`) using per-cycle verification
lengths chosen from either raw or STS-calibrated cumulative confidence. It
reports fixed-K baselines, an expected-speedup argmax policy, a
correction-harvest heuristic (`l = floor(E[a]) + 1` with skip fallback), a
threshold sweep over cumulative survival probability, and a per-cycle oracle
upper bound, each under both shipped and anchor-reuse verifier accountings.
Artifact: `artifacts/lead02_confidence_replay_eval/summary.json`.

Independent fixed-K sanity check passed: a separate direct cycle-jump traversal
over the held-out confidence rows reproduced the replay script's fixed-K
acceptance means, full-accept rates, and both accounting speedups exactly for
K=`2..5` (e.g. K=4: `E[a]=2.1772`, `S(4)=0.3358`, shipped `0.8115x`,
anchor-reuse `0.9773x`).

Held-out `eval` result (60 prompts): under **shipped** accounting, every tested
adaptive policy remains clearly below baseline; the best STS threshold sweep is
only `0.9079x` (CI `0.8853..0.9308`), and STS expected-opt is `0.9216x`. Under
**anchor-reuse** accounting, scheduling becomes meaningfully positive but still
modest: STS expected-opt reaches **`1.0586x`** (CI `1.0365..1.0824`), STS best
threshold `0.09` reaches **`1.0391x`** (CI `1.0160..1.0638`), and the
correction-harvest heuristic reaches `1.0144x`. The per-cycle oracle upper bound
on this same held-out slice is **`1.2483x`**, so the head harvests only part of
the available scheduling ceiling.

Interpretation: the new replay path says the same thing as the earlier rationale
but with end-to-end evidence. Scheduled verification is **not** a rescue under
the shipped verifier economics. If Lead 06's anchor-reuse verifier proves
realizable, confidence scheduling looks like a real but secondary stacking lever
on this held-out slice (`~+4%` to `+6%`, not gate-reviving by itself). This is
still only the 60-prompt eval carrier; broader-corpus replay and the setup
adversarial review gate remain before trusting a dossier-level verdict.

### 2026-07-08 — setup gate 1 (adversarial review): path is coherent, headline softened

Ran the required setup adversarial review with an independent `gpt-5.5` `xhigh`
subagent over the extraction, STS, and replay path, then independently verified
its decisive findings from code and artifacts. Confirmed locally: the STS
temperatures reproduce exactly; the replay's fixed-K cycle-jump traversal
matches an independent direct calculation; and the shipped-accounting oracle is
indeed `1.12967x`, so any wording must say **non-oracle adaptive policies**
remain below baseline, not "no shipped policy" without qualification.

The main gate outcome is methodological, not arithmetic: the current anchor-reuse
adaptive gains are **in-sample** on the held-out `eval` slice because STS
temperatures were fit on that same artifact and the threshold sweep selects on
that same slice. The replay artifact now records this explicitly. Verified
sensitivity check: STS expected-opt under anchor reuse (`1.0586x`) has only
`~4.42 ms/cycle` of headroom over baseline-equivalent cost on this slice, so a
`+5 ms/cycle` residual-overhead miss would erase it (`~0.993x`).

Interpretation: the setup gate **passes the implementation/estimator path for
continued research**, but it rejects any verdict-grade claim stronger than this:
on the current eval slice, scheduling is clearly useless under shipped
accounting and only a modest, potentially fragile stacking lever under
anchor-reuse accounting. Next requirement before any final verdict: split
fit/selection/eval or broaden to a fresh carrier so the policy result is
out-of-sample.

### 2026-07-08 — out-of-sample replay 1: train-fit / eval-select / lead3-eval stays modest

Completed the first proper split-eval replay path. Measured confidence rows on
the 180-prompt `train` split (`artifacts/lead02_confidence_measure_train/`),
fit STS there (`artifacts/lead02_confidence_sts_train/temperatures.json`), then
used those temperatures on the 60-prompt `eval` slice to choose a threshold and
on the fresh 60-prompt `lead3` slice to evaluate it. `train`-fit STS temperatures
were `1.057 / 0.758 / 1.038 / 1.369 / 1.295` for p1..p5. On `eval`, the
anchor-reuse STS threshold sweep chose **`0.08`** (`1.0395x` in-sample), while
STS expected-opt on the same validation slice was `1.0581x`.

Fresh `lead3` evaluation artifact:
`artifacts/lead02_confidence_replay_lead3_trainsts/summary.json`. Result:
under **shipped** accounting, the externally selected STS threshold `0.08`
stays clearly below baseline at **`0.8646x`** (CI `0.8374..0.8933`,
`P(speed<1)=1.0`). Under **anchor-reuse** accounting, the same externally chosen
threshold reaches **`1.0375x`** (CI `1.0130..1.0639`, `P(speed<1)=0.0015`).
Out-of-sample STS expected-opt on the same fresh slice is **`1.0523x`**
(CI `1.0310..1.0754`, `P(speed<1)=0.0` at 2000 bootstrap draws). The fresh
oracle ceiling is **`1.2447x`**.

Per-source on the fresh `lead3` slice, the out-of-sample STS threshold `0.08`
under anchor reuse is mixed but positive overall: `codealpaca 1.0462x`,
`dolly 1.0027x`, `jsonex 1.0628x`. STS expected-opt is similar:
`1.0653x / 1.0222x / 1.0687x`.

Interpretation: after removing the same-slice fit/selection loophole, the
confidence-scheduling picture weakens slightly but does **not** disappear.
Scheduled verification still fails decisively under shipped economics and still
looks like only a **small stacking lever** under anchor reuse (`~+3.8%` with a
frozen threshold, `~+5.2%` with parameter-free expected-opt), not a gate-reviving
local win on its own.

### 2026-07-08 — verdict gate 2 (adversarial review): final verdict softened to marginal conditional material

Ran the required verdict adversarial review with an independent `gpt-5.5`
`xhigh` subagent against the recorded Lead 02 conclusion, then independently
verified its decisive findings from code and artifacts. Confirmed locally:
the shipped-selected validation threshold (`0.52`) still loses on fresh `lead3`
at **`0.8987x`**, so shipped non-viability stands regardless of threshold
selection; and the fresh anchor-reuse positives have only **`3.01 ms/cycle`**
headroom for the frozen threshold and **`3.89 ms/cycle`** for expected-opt
before they disappear (`+4 ms/cycle` drives them to `0.988x` / `0.999x`).

Gate-2 change to the recorded conclusion: do **not** describe Lead 02 as a
clean stacking lever. The deployable-ish frozen-threshold result is only
`+3.75%`, below this lead's predeclared `+5–10%` stacking tier; the `+5.2%`
expected-opt figure is an offline policy diagnostic, not deployable evidence.
Final recorded verdict: shipped scheduling is closed as non-viable locally, and
anchor-reuse scheduling is only **marginal conditional secondary material**
contingent on Lead 06 proving a genuinely cheap folded verifier.
