# ds4-eval: batch-verifier (DSpark SCHEDULE_BATCHED) vs plain decode — practical difference

Date: 2026-07-13. Method: `ds4-eval --plain --nothink --tokens 2048 --temp 0 --seed 1 --questions 10`,
two runs — baseline (no drafter) vs DSpark with `DS4_DSPARK_SCHEDULE_BATCHED=1` (the batch
verifier engaged: `batched_scheduled_draft = scheduled_verify && dspark_schedule_batched_draft()`,
both true). Single process each, M5 Max / Metal / IQ2XXS. Traces: `/tmp/eval_base_q10.json`,
`/tmp/eval_dbatch_q10.json`.

## Headline

**Tentative conclusion (research-lead): on a practical benchmark, the greedy-exactness
difference between the batch verifier and plain decode is immaterial.** Committed outputs
are byte-identical (10/10 questions, ~5,800 tokens); the verify-level 0.64% argmax-flip
(milestone-2 dist-probe) does not propagate to the committed generation. The binding gap is
**speed**, not exactness (DSpark-batch ~37 t/s vs baseline ~38.7 t/s, ~4% slower — no speedup).

*Why "tentative":* 10 Q / ~5.8k tokens is a moderate sample; the mechanism (commit re-decodes
exactly after rejects vs. flips not changing accept/reject) isn't pinned down; a different
workload could in principle surface a divergence. A larger sample + a token-level flip-
detecting diff would promote this to "confirmed."

*Implication for Lead 08:* the greedy-exactness bar is tentatively **met by the existing
path's committed output** — so the real challenge is making a sublinear path **faster than
baseline**, not making it exact. This sharpens (and partly relaxes) the final verdict's
"need a sublinear bit-exact verifier" framing: bit-exactness-in-committed-output appears
already in hand; speed is the binding constraint.

## Per-question (passed questions; gen tokens / gen s → t/s)

| Q | baseline (tok/s) | dspark-batch (tok/s) |
|---|---:|---:|
| 1 (GPQA) 385 tok | 38.9 (9.9s) | 37.0 (10.4s) |
| 2 (SuperGPQA) 60 | 40.0 (1.5s) | 35.3 (1.7s) |
| 3 (AIME) 288 | 38.9 (7.4s) | 37.4 (7.7s) |
| 4 (GPQA) 168 | 39.1 (4.3s) | 37.3 (4.5s) |
| 5 (SuperGPQA) 95 | 39.6 (2.4s) | 36.5 (2.6s) |
| 7 (GPQA) 573 | 37.9 (15.1s) | 37.2 (15.4s) |
| 8 (SuperGPQA) 60 | 37.5 (1.6s) | 37.5 (1.6s) |
| 10 (GPQA) 508 | 37.6 (13.5s) | 37.1 (13.7s) |
| **mean** | **~38.7** | **~37.0** |

- Both: 8/10 passed, 2 failed (Q6, Q9 = AIME, hit the 2048-token cap, no answer) — **same
  outcomes on both paths**.
- Q6/Q9 both generated the full 2048 tokens on both paths (identical token counts).

## Output identity (greedy-exactness in practice)

```
baseline outputs: 10, dspark-batch outputs: 10
identical outputs: 10/10
divergent questions: []
```

**All 10 MODEL_OUTPUT blocks are byte-identical** between baseline and DSpark-batch. So on
this sample the batch-verifier path commits the same greedy tokens as plain decode. (Open
question: whether this holds because the commit re-decodes exactly after rejects, or because
the 0.64% verify-level flips don't change accept/reject decisions — a larger sample + a
flip-detecting diff would settle it. 10 Q / ~5.8k tokens is moderate.)

## Throughput

- DSpark-batch mean ~37.0 t/s vs baseline ~38.7 t/s → **~4% slower** (no speedup).
- ds4-eval wall runtime: baseline 2m, DSpark-batch 3m (the extra minute is the drafter load
  + drafting overhead, not per-token gen).

## Interpretation for the Lead 08 question

The current batch-verifier path is **greedy-exact in committed output** (10/10) but provides
**no speedup** (~4% slower). This is consistent with the codex-gated verdict: the existing
(bit-exact-enough-in-practice) path doesn't clear the gate on speed; a gate-clearing verifier
needs to be both sublinear AND fast (the novel-kernel build), not just exact.
