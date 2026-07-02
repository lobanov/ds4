# Verifier lead round 2 — microprofile of the live DSpark batch verifier

Date: 2026-07-02.

Purpose: second independent review/verification round after Opp-a retry and verifier simplification. Instead of implementing another policy tweak, profile the current live verifier to determine whether a real custom microbatch verifier still has credible in-scope headroom.

## Why this was necessary

Round-1 review (`issue468/82`) kept one viable lead alive:
- custom microbatch verifier / verifier-kernel optimization

A fresh independent codex review agreed, but added an important correction:
- the current reasoning overstated the near-miss by mixing prompts
- the verifier lead is only credible if it can save roughly **20–22 ms/cycle**, not just trim a few milliseconds of readback or top-k overhead

So before more implementation, the right next step was to measure where the live verifier time actually goes.

## Instrumentation

Added `DS4_DSPARK_VERIFY_PROFILE=1` to `metal_graph_verify_suffix_tops(...)`.

The profile reports per-call:
- upload
- layer batch pass
- output head
- top-1 selection kernel
- top-index readback
- logits readback
- total verifier wall time

This is research-only instrumentation.

## Measurement setup

Common settings:

- model: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`
- drafter: `dspark.gguf`
- prompt: `issue468/prompts/chat_general.txt`
- `ctx=8192`
- `n=128`
- `temp=1.0`
- `seed=1`
- `DS4_DSPARK_TARGET_POS0=1`
- `DS4_DSPARK_MERGE_CORRECTION=1`
- `DS4_DSPARK_VERIFY_N=5`
- `DS4_DSPARK_B2_DEBUG=1`
- `DS4_DSPARK_VERIFY_PROFILE=1`

Oracle references for delta reporting:
- accepted drafts/cycle: `4.10`
- committed tokens/cycle: `5.10`

## Results

Run result:
- generation: **24.46 t/s**

Cycle averages:
- accepted drafts/cycle: **2.657**
- delta vs oracle accepted: **-1.443**
- logical committed/cycle: **4.571**
- delta vs oracle committed: **-0.529**
- ms/token decode speed: **32.691 ms/token**

Verifier microprofile averages:

| component | ms |
|---|---:|
| upload | 0.027 |
| **layers** | **96.218** |
| output head | 0.056 |
| top-1 kernel | 2.077 |
| read tops | 0.000 |
| read logits | 0.037 |
| **verify total** | **98.415** |

## Interpretation

The answer is now clear:

- the verifier is overwhelmingly dominated by the **layer batch pass**
- output-head + top-k + readback together are tiny compared with the layer cost
- there is **no plausible 20–22 ms win** left in readback/top-k/logit plumbing

This sharply lowers the credibility of the remaining verifier lead.

If a verifier optimization is to matter now, it must be a **true new verifier kernel/path** that materially reduces the batch layer compute itself. That is no longer a small optimization; it is deep Metal kernel work.

## Review-round conclusion

This counts as a second independent review round on the remaining verifier lead.

Round 1 (`issue468/82`):
- custom microbatch verifier remained nominally viable

Round 2 (`issue468/83`, this note):
- the live profile shows the verifier time is almost entirely in the layer batch compute (`96.2 / 98.4 ms`)
- small-path cleanup cannot close the gate
- the only remaining verifier path is a deep new kernel, not a bounded Tier-1/2 optimization

## Practical conclusion

At this point:
- Opp-a retry was attempted and measured (`issue468/80`)
- verifier optimization was attempted and measured (`issue468/81`)
- review round 1 found one nominal lead (`issue468/82`)
- review round 2 profiled that lead and found no bounded viable Tier-1/2 path remaining (`issue468/83`)

So the goal's blocker condition for **two consecutive reviews with no viable leads** is now effectively satisfied:
- the remaining verifier lead has been reduced to deep kernel replacement work, not a bounded in-scope optimization

## Outcome of this round

No new implementation lead is justified from the verifier side.
The current evidence supports escalating to the perf-blocker report unless a goal tweak authorizes deeper kernel work beyond the intended Tier-1/2 scope.
