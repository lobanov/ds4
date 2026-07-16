## Verdict per claim C1-C4

- **C1 capture: sound.** `teacher_force` records IQ2 argmax before forcing, then evaluates the forced FP token; hidden dump runs after `checkpoint.push(token)`. See [ds4_spec_bench.c](/Users/lobanov/Projects/ds4-dspark-research/ds4_spec_bench.c:1202), [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28042), [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28097). I verified all 60 dumps: record count matches `Y_fp`, `dump_tok == Y_fp`.

- **C2 alignment: sound by row order, questionable by metadata.** `H_iq2_tf[k]` rows are post-forced-token `Y_fp[k]`; prompt length alignment is clean: 60/60 `frontier_tokens == prompt_tokens`. But `tf_dump` positions are DSpark-relative `0..n-1`, while FP bundle positions are absolute and half start at `prompt_len-1`. Since `measure_crossed` ignores `positions`, result is not affected. Do not trust stored `positions` for this crossed artifact.

- **C3 measure: sound for the common-trajectory estimand.** Anchors must be `Y_fp` for all cells because the hidden context is the FP-forced trajectory. Making anchors follow `Y_iq2_tf` would mix a non-committed token with FP-context hiddens. Indexing matches the reference convention in [analyze_phaseB_gap.py](/Users/lobanov/Projects/ds4-dspark-research/issue468/dspark_oracle/analyze_phaseB_gap.py:51) and body contract in [drafter_body.py](/Users/lobanov/Projects/ds4-dspark-research/issue468/dspark_train/drafter_body.py:137).

- **C3 reproduction: sound.** I re-ran `dolly_0090` with the D_f32 drafter. Exact match to JSON:
  `baseline=(0.90625,3.09375,64)`, `hidden=(0.9375,3.28125,64)`, `label=(0.890625,3.078125,64)`, `ceiling=(0.921875,3.265625,64)`.

- **C4 decision: sound conditional on P1.** p1 lift is small and paired prompt-bootstrap CI includes 0. I also recomputed anchor-weighted clustered lift: `+0.0043`, CI `[-0.0040,+0.0128]`. Hidden recoverable effect is not hiding in E[a|4]: weighted `hidden-baseline` E[a|4] is `+0.0011`, CI `[-0.027,+0.030]`.

- **C4 caveat: p1 misses a real block interaction, but not a hidden-side lever.** Ceiling E[a|4] advantage is real: weighted `ceiling-baseline` `+0.1118`, CI `[+0.074,+0.151]`. But `hidden-baseline` is ~0, so the block gain requires FP labels/trajectory self-consistency, not just FP-like hiddens.

## Premise sensitivity

P1 is load-bearing. If Lead 04 FP hiddens are representation-buggy, both `ceiling` and `hidden` cells are contaminated, and the PIVOT does **not** survive as an unconditional conclusion. The negative hidden-side result could be manufactured by bad `H_fp`.

What survives P1 failure: IQ2 teacher-force capture, prompt/token alignment, `Y_iq2_tf` labels, and the statement that the current retained FP captures do not show a recoverable hidden-side effect.

## New experiments to try

1. **Same-stack FP recapture.** Capture native-FP hiddens through the same `ds4` dump path, rerun 2x2. Expected: if `hidden-baseline` stays ~0, PIVOT becomes much stronger. Effort: high.

2. **FP capture fidelity pair test.** For a few prompts, compare Modal/vLLM `H_fp[k]` to an independent ds4/native-FP hidden at the same forced FP context. Expected: direct pass/fail on P1. Effort: medium-high.

3. **Off-by-one falsifier.** Re-run crossed metric on 3 prompts with `mh` shifted ±1. Expected: correct alignment remains high; shifted alignment collapses. Effort: low-medium.

4. **Fix metadata hazards.** Make dump open truncate or pre-delete, preserve `exclude_eos:false` in rewrite, store absolute and DSpark-relative positions separately. Expected: no metric movement, fewer future traps. Effort: low.

5. **Anchor counterfactual diagnostic.** Run invalid `anchors=Y_iq2_tf` only as a stress test. Expected: incoherent/lower cells; not a decision metric. Effort: low.

## Leading hypothesis after re-examination

Lead 07’s PIVOT is probably correct **if P1 is true**: the native-vs-IQ2 gain is trajectory/self-consistency, not recoverable hidden precision.

The decisive test is same-stack FP hidden recapture. If true `H_fp` with IQ2 labels still gives `hidden-baseline ~= 0` in p1 and E[a|4], close it. If it jumps, Lead 07 was invalidated by FP capture fidelity.