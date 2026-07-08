# Lead 02 — confidence-scheduled verification: result

Date: 2026-07-08. Status: **resolved** (both codex/subagent gates passed; decisive findings
independently verified). Worklog:
`issue468/archive/leads/lead_02_confidence_scheduled_verification.md`. Core harnesses:
`run_lead02_measure_confidence.py`, `run_lead02_sts.py`, `run_lead02_replay.py`.
Key artifacts:

- `issue468/artifacts/lead02_confidence_measure_{train,eval,lead3}/`
- `issue468/artifacts/lead02_confidence_sts_{train,eval}/`
- `issue468/artifacts/lead02_confidence_replay_eval/summary.json`
- `issue468/artifacts/lead02_confidence_replay_eval_trainsts/summary.json`
- `issue468/artifacts/lead02_confidence_replay_lead3_trainsts/summary.json`

## Question

Can DSpark's confidence head make **scheduled verification** a real local speedup lever on
this dossier's target path, or is it at best a secondary stacking term after anchor reuse?

This lead evaluates that question offline by extracting confidence from the retained oracle /
torch carrier, calibrating it with Sequential Temperature Scaling (STS), and replaying
per-cycle adaptive verification on the **realistic cycle-jump trajectory**
(`next_step += accepted + 1`) under two verifier accountings:

- **shipped** — the current `ds4 --mtp` style accounting (pays decode every cycle)
- **anchor-reuse** — the optimized verifier accounting already used in `spec_speedup_model`

## Notation & definitions

- **`k`** — draft position within the 5-token DSpark block.
- **`prefix`** — realized accepted prefix length for a cycle (`0..5`).
- **`c_k`** — raw per-position confidence score from the confidence head.
- **cumulative survival** — `a_k = prod_{i<=k} c_i`, the estimated probability that the
  draft survives target verification through position `k`.
- **STS** — Sequential Temperature Scaling; fit temperatures `T_k` left-to-right so
  `sigmoid(logit_k / T_k)` improves calibration of the cumulative product.
- **fixed-K** — always verify exactly `K` draft positions.
- **expected-opt** — per-cycle offline policy that picks the `l in {0..5}` with highest
  **expected** local speedup from the confidence scores. Useful as a policy upper-ish
  diagnostic, not a deployable causality-proof scheduler.
- **selected threshold** — a threshold on cumulative survival chosen on one slice, then
  replayed unchanged on a fresh slice. This is the cleanest threshold-policy result here.
- **oracle** — per-cycle upper bound that chooses `l` from the **realized** accepted prefix.
  Not deployable; a ceiling on how much scheduling could extract from this corpus/accounting.
- **CI [lo, hi]** — prompt-clustered bootstrap 95% interval on speedup (resample prompts,
  not cycles).
- **`P(speed<1)`** — bootstrap mass below baseline decode throughput.

## Method

1. **Confidence extraction + gates.**
   Exposed confidence logits / scores in both the numpy oracle and the torch/MPS carrier.
   Retained-oracle inertness gate passed on all 10 temp-0 bundles. Torch-vs-numpy confidence
   gate passed on a 3-source sample with tiny score/logit deltas.
2. **Measurement.**
   Measured per-cycle confidence rows on the powered corpus:
   `train` 180 prompts, `eval` 60 prompts, `lead3` 60 prompts, all balanced across
   `codealpaca` / `dolly` / `jsonex`.
3. **Calibration.**
   Fit STS on `train`. Fitted temperatures (p1..p5):
   `1.057018 / 0.757858 / 1.037660 / 1.369200 / 1.295342`.
4. **Replay.**
   Used `train` STS on `eval` to choose a threshold, then replayed that frozen threshold on
   fresh `lead3`. Also reported fixed-K, expected-opt, harvest, and oracle under both
   accountings.

## Confidence-head measurement

The confidence head is **not degenerate** on the served IQ2XXS target.

Per-position discrimination / cumulative calibration:

| split | p1 AUC | p3 cum AUC | p5 cum AUC | p3 cum ECE | p5 cum ECE |
|---|---:|---:|---:|---:|---:|
| train | 0.780 | 0.809 | 0.843 | 0.090 | 0.075 |
| eval | 0.777 | 0.814 | 0.840 | 0.084 | 0.071 |
| lead3 | 0.795 | 0.814 | 0.848 | 0.086 | 0.077 |

STS improves cumulative-prefix calibration modestly but consistently on `train`:

- p2 cumulative ECE `0.0779 -> 0.0747`
- p3 `0.0900 -> 0.0846`
- p4 `0.0854 -> 0.0816`
- p5 `0.0750 -> 0.0705`

Interpretation: the head carries useful ranking signal, but calibration improvement is only
incremental; the main constraint is still how much cycle-level acceptance the scheduler can
harvest under realistic local verifier economics.

## Replay result

### Validation-side policy selection (`eval`, train-fit STS)

With `train` temperatures applied to `eval`:

| accounting | STS expected-opt | best STS threshold |
|---|---:|---:|
| shipped | 0.9212x | threshold `0.52` -> 0.9057x |
| anchor reuse | 1.0581x | threshold `0.08` -> 1.0395x |

The threshold `0.08` is the externally selected anchor-reuse threshold used for the fresh
evaluation below.

### Fresh out-of-sample evaluation (`lead3`, train-fit STS, threshold chosen on eval)

| policy | shipped | anchor reuse |
|---|---:|---:|
| fixed-K2 | 0.7837x | 0.8926x |
| fixed-K3 | 0.7803x | 0.9111x |
| fixed-K4 | 0.8123x | 0.9810x |
| fixed-K5 | 0.7987x | 0.9761x |
| **STS expected-opt** | **0.9205x** CI [0.8990, 0.9438] | **1.0523x** CI [1.0310, 1.0754] |
| **STS selected threshold `0.08`** | **0.8646x** CI [0.8374, 0.8933] | **1.0375x** CI [1.0130, 1.0639] |
| STS harvest | 0.9118x | 1.0133x |
| oracle ceiling | 1.1297x | 1.2447x |

Per-source fresh-eval result for the **frozen threshold `0.08` under anchor reuse**:

| source | speedup |
|---|---:|
| codealpaca | 1.0462x |
| dolly | 1.0027x |
| jsonex | 1.0628x |

Per-source fresh-eval result for **STS expected-opt under anchor reuse**:

| source | speedup |
|---|---:|
| codealpaca | 1.0653x |
| dolly | 1.0222x |
| jsonex | 1.0687x |

## Verdict

**Under shipped verifier economics, confidence scheduling is not a viable local lever.**
On the fresh out-of-sample `lead3` slice, the **anchor-reuse-selected** STS threshold `0.08`
gives **0.8646x** under shipped costs; even the **shipped-selected** threshold from `eval`
(`0.52`) reaches only about **0.8987x** on fresh `lead3`. STS expected-opt gives only
**0.9205x**. Non-oracle adaptive policies remain decisively below baseline.

**Under anchor-reuse accounting, confidence scheduling survives only as fragile,
conditional secondary material, not a gate-reviving win.** On fresh out-of-sample
evaluation:

- the clean frozen-threshold result is **1.0375x** (about **+3.8%**)
- STS expected-opt is **1.0523x** (about **+5.2%**)

The clean frozen-threshold result is **below the lead's predeclared `+5–10%` stacking
tier**, and the expected-opt value is an offline policy diagnostic rather than deployable
evidence. Both are also fragile to residual verifier overhead: on fresh `lead3`, the
frozen-threshold positive has only about **`3.01 ms/cycle`** of headroom, and expected-opt
about **`3.89 ms/cycle`**. Adding `+4 ms/cycle` drops them to about **`0.988x`** and
**`0.999x`**, respectively.

**Decision:** Lead 02 does **not** rescue the shipped local path and does **not** clear its
own stacking tier on the clean frozen-threshold policy. Record it as **marginal,
conditional secondary material**: worth remembering only if Lead 05 proves a genuinely
cheap anchor-reuse verifier, but not a standalone priority and not evidence that scheduled
verification revives the local gate.

## Scope / caveats

- The **frozen threshold** is the strongest threshold-policy evidence here because it is
  selected on `eval` and replayed unchanged on fresh `lead3`.
- **expected-opt** is still an offline policy diagnostic. It is informative about ceiling-ish
  value from the confidence scores, but it is not a proof of a deployable non-anticipating
  scheduler.
- All anchor-reuse positives remain **conditional** on Lead 05's verifier economics. On the
  fresh `lead3` slice, the frozen-threshold positive has only `3.01 ms/cycle` of headroom
  and expected-opt `3.89 ms/cycle`; a few milliseconds of real folded-verifier overhead
  would erase the gain.
- The corpus is still `codealpaca` / `dolly` / `jsonex`; a code/synthesis-heavy deployment
  could move the result.

## Implications

- **Lead 05 (verifier engineering):** Lead 02 no longer carries the burden of reviving the
  local gate. If anchor reuse proves real and *very* cheap, scheduling is at most a modest
  extra term, not the main win.
- **Lead 06 / upstream quality:** a better drafter remains the larger open lever; scheduling
  can only harvest what the confidence head can rank from the existing acceptance landscape.
- **spec_speedup_model:** the prior open interval for scheduling is now narrowed:
  shipped local path stays non-viable; anchor-reuse path gets only a few additional points
  from scheduling out-of-sample, not a primary-gate jump.
