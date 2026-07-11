# MTP verifier engineering and fused-verify Phase A

Date: 2026-07-11.
Scope: Lead 06 verifier engineering plus **Phase A only** of Lead 08.
Artifacts:
- `issue468/artifacts/mtp_verifier_bench_long/summary.{json,csv}`
- `issue468/artifacts/mtp_exactness_compare/summary.json`
- `issue468/artifacts/mtp_temp_distribution_compare/summary.json`
- `issue468/artifacts/mtp_phaseA_profile/summary.json`
- `issue468/artifacts/mtp_corpus_bench/summary.json`

## Headline

An **exact, env-gated anchor-reuse verifier path now exists and is benchmarkable**, but it
does **not** reach baseline decode speed. It materially improves on shipped `--mtp` at the
same larger K by removing the standalone anchor decode, yet on every measured corpus it
still loses to plain decode. On the retained long-context corpus at K=4, shipped `--mtp`
lands at **22.04 / 23.19 / 24.19 t/s** (`code_8k` / `synthesis_8k` / `grounded_8k`),
while exact anchor reuse reaches **30.81 / 31.69 / 28.85 t/s** against baselines
**36.34 / 37.46 / 33.65 t/s**. On the full modeled 300-prompt corpus, baseline averages
**39.02 t/s**, shipped K=4 **27.15 t/s**, and exact anchor reuse **32.67 t/s**:
**+20.95% over shipped**, but still **−16.26% vs baseline**.

Lead 08 Phase A also landed: the shipped verifier is dominated by layer execution, not
host readback. The measured `verify_ms(K) - floor_ms(K)` headroom at K=3..5 is roughly
**19–22 ms/cycle**, above the lead's `~15 ms` proceed gate. Recommendation:
**proceed to Phase B fused-kernel work** if verifier acceleration remains in scope.

## What was implemented

- Added an env-gated path controlled by `DS4_MTP_ANCHOR_REUSE=1`.
- The implemented path is **exact sequential anchor reuse**, not a fast exact batched
  verifier. It:
  - drafts from the committed hidden carried across cycles;
  - reuses the correction token as the next anchor token;
  - falls back to exact sequential target verification;
  - copies the last committed hidden row back into `cur_hc` so reuse stays live.
- Added verifier instrumentation under `DS4_MTP_VERIFY_PROFILE=1` to record upload,
  encode, execute, readback, selected-expert bytes, and expert-union counts.
- Added retained harnesses for:
  - exact greedy output comparison,
  - temp>0 logits/distribution parity,
  - Phase A verifier profiling,
  - resumable modeled-corpus benchmarking.

## Exactness gates

- **Greedy exactness:** PASS on the retained exactness corpus.
  - `issue468/artifacts/mtp_exactness_compare/summary.json`
  - 10/10 prompts matched baseline byte-for-byte at K=4.
  - Mean gen t/s: baseline **38.665**, exact anchor reuse **31.872**.
- **Temp>0 distribution parity:** PASS.
  - `issue468/artifacts/mtp_temp_distribution_compare/summary.json`
  - 640 sampled steps across the retained exactness corpus at `temp=0.5` and `1.0`.
  - 577 steps exercised the anchor-reuse speculative path directly.
  - `max_abs=0`, `rms=0`, `sampled_lp_diff=0`.

Interpretation: the current env-gated path is a correctness-safe verifier substrate, not a
speed win.

## Benchmark results

### Long-context corpus

Retained prompts: `code_8k`, `synthesis_8k`, `grounded_8k`; 64 generated tokens.

| prompt | baseline | shipped K=2 | shipped K=4 | exact reuse K=4 |
|---|---:|---:|---:|---:|
| code_8k | 36.34 | 34.72 | 22.04 | 30.81 |
| synthesis_8k | 37.46 | 35.40 | 23.19 | 31.69 |
| grounded_8k | 33.65 | 33.95 | 24.19 | 28.85 |

Key points:
- Best shipped cell remains K=2, near but mostly below baseline.
- Exact anchor reuse is **better than shipped at K=3..6 everywhere**.
- Exact anchor reuse is **still worse than baseline everywhere**.
- Its verifier cost is relatively flat across K because it pays the sequential exact path:
  overall median verify stays around **56.5–57.2 ms** for K=2..6.

### Modeled 300-prompt corpus

Full retained corpus: Stage 2 240 + Lead 03 60; 100 prompts each from `dolly`,
`codealpaca`, `jsonex`; 128 generated tokens.

| corpus | baseline | shipped K=4 | exact reuse K=4 |
|---|---:|---:|---:|
| full 300 | **39.020** | **27.153** | **32.673** |
| dolly | 38.987 | 26.517 | 32.503 |
| codealpaca | 39.234 | 27.818 | 32.870 |
| jsonex | 38.839 | 27.123 | 32.648 |

Relative to baseline / shipped:
- **shipped K=4 vs baseline:** **−30.41%** mean over 300 prompts
- **exact reuse K=4 vs baseline:** **−16.26%**
- **exact reuse K=4 vs shipped K=4:** **+20.95%**

Per-source mean relative deltas:
- `dolly`: shipped **−31.97%**, exact reuse **−16.63%**, exact reuse vs shipped **+23.15%**
- `codealpaca`: shipped **−29.09%**, exact reuse **−16.22%**, exact reuse vs shipped **+18.78%**
- `jsonex`: shipped **−30.16%**, exact reuse **−15.94%**, exact reuse vs shipped **+20.93%**

Interpretation: the Lead 06 path recovers about half the shipped K=4 penalty, but not
enough to make the verifier competitive with plain decode on realistic prefix lengths.

## Lead 08 Phase A profiling

The profiling harness measured the shipped K=3..5 verifier on the long-context prompts.

Representative medians:

| prompt | K | verify_ms | initial layer_execute_ms | replay layer_execute_ms | selected GiB | full GiB |
|---|---:|---:|---:|---:|---:|---:|
| code_8k | 3 | 59.51 | 56.46 | 35.43 | 4.25 | 72.56 |
| code_8k | 4 | 65.91 | 62.59 | 45.36 | 5.39 | 72.56 |
| code_8k | 5 | 76.66 | 72.64 | 45.89 | 5.95 | 72.56 |
| synthesis_8k | 4 | 66.47 | 63.04 | 55.06 | 4.82 | 72.56 |
| grounded_8k | 4 | 67.07 | 63.95 | 57.11 | 4.82 | 72.56 |

Findings:
- Verifier time is overwhelmingly in **layer execution**.
- Host-side upload/readback is negligible in comparison.
- Selected routed-expert bytes at K=4 are only **~4.8–5.4 GiB** versus a full-routed
  **72.56 GiB** layer set, but the current implementation still pays about
  **66–67 ms** wall time.
- Using the lead's floor framing (`decode_ms≈26`), the K=4 headroom is about
  **40 ms** versus baseline decode and about **19–22 ms** versus the lead's own
  `~45 ms` proceed target.

Verdict on the Phase A gate: **PASS**. The verifier is not at floor; there is enough
measured headroom to justify a fused low-K kernel investigation.

## Verdict

### Lead 06

**Resolved as a correctness-safe but performance-negative verifier implementation.**

What it proved:
- the anchor-reuse state machine can be made exact on ds4;
- exact greedy behavior survives;
- temp>0 distribution parity survives;
- removing the standalone anchor decode yields a real improvement over shipped larger-K MTP.

What it falsified:
- the current exact implementation is **not** the cheap folded verifier assumed by the old
  optimistic model framing;
- exact anchor reuse alone does **not** recover baseline throughput on realistic prefixes.

### Lead 08 Phase A

**Resolved with recommendation to proceed to Phase B.**

Reason:
- measured verifier headroom exceeds the lead's proceed gate;
- the dominant cost sits in the GPU verifier layer-execute path, exactly where fused low-K
  kernel work would attack it;
- the current exact reuse path already shows that verifier engineering, not acceptance
  alone, is the active performance bottleneck at this stage.
