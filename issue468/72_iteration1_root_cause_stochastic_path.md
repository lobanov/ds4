# Iteration 1: Root Cause — Live vs Probe Drafter Inputs Fundamentally Different

Date: 2026-07-01. Iteration 1 finding update. Doc-only research record.

## 0. The finding

Live B2 draft tokens at the first cycle: [1309, 304, 5085, 260, 15255]
Probe draft tokens at step 1 (same position): [201, 3465, 10150, 3913, 3395]
Target greedy at step 1: 36490

The live and probe drafters produce **completely different drafts** at the same
generation position. This means the drafter's INPUT (main_hidden) is different
between live and probe, producing different base_logits and different drafts.

## 1. Why the inputs differ

The MC oracle and the probe use main_hidden captured from a **greedy reference
decode** (target_greedy path). The live B2 path at temp=1 uses main_hidden
captured from the **actual stochastic anchor decode** (the main loop samples the
anchor from s->logits at temp=1, producing a different anchor token than greedy).

Different anchor token → different hidden state at L40/41/42 → different
main_hidden → different drafter forward → different drafts → different acceptance.

## 2. This is NOT how standard speculative decoding works

In standard spec decoding (Leviathan 2023), the drafter proposes drafts for the
target's NEXT token distribution. The drafter doesn't need a "special" hidden
state — it just needs to produce a distribution q close to the target's p. The
acceptance min(1, p/q) should work regardless of the sampling path.

But DSpark's drafter is CONDITIONED on the target's main_hidden (the hidden state
at layers 40/41/42 of the anchor decode). This is an architectural choice: the
drafter gets "inside information" from the target's forward pass. If the anchor
token is stochastic (temp=1), the target's hidden state is for a DIFFERENT token
than the greedy reference, and the drafter's conditioning is on a different
input than what it was trained on.

## 3. The per-position accept rate confirms this

Position 0 accept rate: 68.2% (live) vs 95.5% (oracle). The 27pp gap at
position 0 — the VERY FIRST draft position — means the drafter's q distribution
diverges from the target's p distribution much more on the stochastic path than
on the greedy path. This is the signature of a drafter that was trained on
greedy-path hidden states and doesn't generalize perfectly to stochastic-path
hidden states.

## 4. Can this be improved?

Two possible directions:

**A. Improve the drafter's generalization to stochastic paths (model training — OUT of scope).**
The drafter was likely trained/distilled on greedy-path main_hidden. At temp=1,
stochastic anchors produce out-of-distribution main_hidden inputs. Fixing this
requires retraining the drafter on stochastic-path data — out of scope (frozen drafter).

**B. Make the drafter's conditioning more robust (code-only).**
If the main_hidden capture has a quality issue (e.g., the F16 truncation of
cur_hc at layers 40/41/42 during the capture is lossy), improving the capture
precision could help the drafter generalize better. But issue468/52 already
landed F32 KV default and the capture was verified clean. The issue is more
fundamental: the drafter's architecture (conditioned on target hidden states)
makes it path-dependent.

**C. Use the drafter in a regime where it performs well (temp=0 / greedy).**
At temp=0 (greedy decoding), the live anchor IS the greedy anchor, and the
main_hidden matches the reference. The probe's 4.37 acceptance would translate
to live. BUT: B2 rejection sampling at temp=0 is degenerate (the user's original
observation). The solution: use GREEDY-MATCH acceptance (not rejection sampling)
at temp=0. This was tested in issue468/41 (temp-aware B2) and HURT because the
batch-verify argmax diverged from sequential-decode argmax. But that was before
the n_real cap fix.

## 5. Iteration report

| metric | live (temp=1) | oracle MC (greedy path) | gap |
|---|---|---|---|
| accepted drafts/cycle | 1.97 | 4.10 | 2.13 |
| committed tokens/cycle | 2.97 | 5.10 | 2.13 |
| pos 0 accept rate | 68.2% | 95.5% | −27.3pp |
| gen t/s | 29.5 | — | — |
| ms/token | 33.9 | — | — |
| baseline t/s | ~39 | — | — |

## 6. Lead for iteration 2

The gap is NOT fixable in the rejection-sampling acceptance at temp=1 — it's
inherent to the drafter's conditioning on target hidden states from stochastic
paths. The drafter performs best on the GREEDY path.

**Iteration 2 lead: revisit greedy-match acceptance at temp=0 with the n_real cap=5.**
At temp=0 with cap=5, the probe achieves 4.37 avg prefix. If greedy-match
acceptance (instead of rejection sampling) can be made to work at temp=0, the
live path would achieve ~4.37 acceptance on the GREEDY path (which IS the live
path at temp=0). The prior test (issue468/41) failed because the batch-verify
argmax diverged — but that was before the n_real cap fix. With cap=5, the
batch-verify might be more stable.
