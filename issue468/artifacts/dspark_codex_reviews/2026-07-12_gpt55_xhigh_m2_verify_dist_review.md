## Verdict Per Claim

- **C1: sound, but statistically thin.** Recomputed: greedy `2.4375`, RS-greedy-draft `2.5002` `(+0.0627, +2.57%)`, RS-sampled `2.2696` `(-0.1679, -6.89%)`. Tail handling is conservative. Bundle-clustered 95% CI for `+0.063` is roughly `[-0.145, +0.270]`, so “neutral” is better than “+2.6% real.”
- **C2: questionable.** Recomputed: `1/156 = 0.64%` flips, weighted TV mean `0.01039`, cycle median TV `0.00354`, max `0.10397`, cycle KL median `5.32e-5`. But artifact stores per-cycle aggregate TV/KL, not per-position samples; “distribution-exact” is stronger than the evidence.
- **C3: mostly sound direction, flawed framing.** Recomputed timed groups match: seq decode slope `28.12 ms/accept`; adding hidden push + final readback gives `28.88 ms/accept`. K5 batched mean `72.85 ms`, break-even `2.52` accepts. K4 batched mean/bench is `~65.8-66.7 ms`, break-even `2.28-2.31`.
- **C4: gap real, ratio overstated.** Runtime verified mean is `266/211 = 1.2607`. But this run mostly drafted 4 tokens while reporting `verify_n=5`; q4tap greedy `E[min(prefix,4)] = 2.2125`, so runtime realizes `57%`, not `52%`. Still a large gap.
- **C5: directionally sound, too absolute.** “Verifier not a clear lever” survives corrected costs. But the margin is tight enough that with higher true decode slope or a very low-overhead committing path, batched K4 could win at oracle sliding acceptance.

## Premise Sensitivity

- **Local IQ2XXS + current DSpark drafter:** headline survives. Change drafter quality or target mix and crossover can flip.
- **`spec_speedup_model.md` oracle as reference:** if offline oracle overstates live acceptance, batched looks worse; if easier corpus/live q improves, batched may become useful.
- **Option A relaxed gate:** if ds4-eval rejects TV tail behavior, batched verifier is dead regardless of cost. If accepted, RS gives verifier tolerance, not acceptance lift.

## Methodology Bugs Found

- **`verify_decode_ms` is not full sequential cost.** Code times only `metal_graph_eval_token_raw_swa_top`; hidden push and final logits read are outside it. Check: `nl -ba ds4.c | sed -n '28635,28660p;28728,28741p'`.
- **`verify_decode_ms` is also slightly probe-inflated.** With probe, `seq_cap` is non-null and `metal_graph_eval_token_raw_swa_top` reads full logits per accepted token. Check: `nl -ba ds4.c | sed -n '20401,20423p;28637,28639p'`.
- **Fixed verify K artifact is mislabeled.** `verify_n` is initialized before `draft_n` is reduced to `fixed_verify_n`; clamp does not run for fixed non-scheduled mode. Check: `nl -ba ds4.c | sed -n '28553,28587p'`.
- **Batched committing overhead not measured.** `batched_verify_ms` wraps only `metal_graph_verify_suffix_tops`; snapshot/restore is outside. Check: `nl -ba ds4.c | sed -n '28610,28628p'`.
- **Exactness artifact lacks per-position TV.** Struct/emission only has `mean_tv` per cycle. Check: `nl -ba ds4.h | sed -n '278,286p'` and `nl -ba ds4_spec_bench.c | sed -n '1273,1281p'`.

## Revised Crossover

Using timed artifact:

- Sequential decode-only slope: `28.12 ms/accept`.
- Sequential with DSpark push + final readback: `28.88 ms/accept`.
- K5 batched: `72.85 / 28.88 = 2.52` accepts.
- K4 batched: `65.8-66.7 / 28.88 = 2.28-2.31` accepts.
- Add committing overhead: break-even is `(batched_ms + overhead) / 28.88`; `+3 ms` moves K5 to `2.63`, K4 to `~2.39`.

Sensitivity: with slope `±25%` and batched `±10%`, K5 break-even ranges about `1.82-3.70`. Tightest-margin input is decode slope.

## New Experiments

1. **Fix `verify_n` clamp, rerun clean no-probe fixed K4/K5.** Signal: real runtime acceptance and clean sequential timing. Effort: medium.
2. **Emit per-position TV/KL, not cycle means.** Include decision-row identity and immediate-miss accounting. Signal: whether TV tail is real. Effort: low-medium.
3. **Prototype minimal committing batched RS timing.** Include snapshot/restore, hidden capture, accept/reject/resample, selective row read. Signal: real overhead. Effort: high.
4. **Live-vs-oracle drafter trace on same cycle starts.** Log live draft logits/hiddens and compare to offline q. Signal: state pollution vs oracle mismatch. Effort: medium.
5. **Scale C1/C2 to powered corpus.** Need hundreds to thousands of positions for flip/tail precision. Effort: medium-high.

## Leading Hypothesis

Runtime drafter state/trajectory mismatch is the main gap; verifier economics are marginal, not obviously useless. Decisive test: after fixing the fixed-K bug, run clean fixed K4 no-probe on the exactness corpus and simultaneously log live DSpark q for each cycle start; compare live q/prefixes to the offline oracle on the same anchors.