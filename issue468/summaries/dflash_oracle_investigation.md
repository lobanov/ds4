# DFlash vs DSpark — accepted-prefix comparison

Date: 2026-07-05. Status: **resolved** (was briefly mis-diagnosed as blocked; see
"the d2t decoding bug" below). DFlash oracle built, validated, and measured.

## Goal

Build a numpy oracle for the **DFlash** drafter
(`inference-optimization/dflash-DeepSeek-V4-Flash-all-swa-muon-speculators-50k`,
~1.8B BF16, 5 dense Llama layers), feed it target hidden states captured from
ds4, and compare accepted prefix head-to-head against DSpark to judge whether
DFlash is a more attractive drafter for this target.

## Result (same IQ2XXS target, same offline per-step protocol, 10 prompts x 3 temps)

| metric | DFlash (7-token block) | DSpark q4k (5-token block) |
|---|---:|---:|
| avg accepted prefix (overall) | **0.876** | **2.171** |
| ... temp 0.0 | 0.886 | 2.337 |
| ... temp 0.5 | 0.886 | 2.087 |
| ... temp 1.0 | 0.857 | 2.087 |
| match % (overall) | 19.2 | 48.4 |

**Verdict: on this corpus DSpark is substantially more attractive — ~2.5x the
accepted prefix of DFlash under the same protocol.** DFlash's parallel
noise-block drafting is repetitive within a block and rarely extends the prefix
past position 1; DSpark's autoregressive markov rollout naturally builds longer
matching prefixes.

DFlash per-position marginal accuracy on this corpus:
`[0.68, 0.27, 0.13, 0.10, 0.07, 0.07, 0.02]`; DFlash's own val_metrics are
`[0.74, 0.51, 0.37, 0.27, 0.21, 0.17, 0.14]`. Position 1 matches val closely
(0.68 vs 0.74); positions 2-7 underperform val. Likely contributors: the IQ2XXS
2-bit target (DFlash was validated on BF16; degraded context hurts later
positions more) and/or corpus difficulty. This does not change the comparison,
which is fair (both drafters use the same IQ2XXS captures).

## The d2t decoding bug (how this was briefly mis-diagnosed as "blocked")

The drafter outputs logits over a **32000-token draft vocabulary** mapped to the
129280 target vocab via `d2t`/`t2d`. I initially decoded with
`target_id = d2t[draft_idx]` (treating `d2t` as an absolute map). That produced
common-token garbage (0% match), which I mis-attributed to a target-hidden
representation mismatch and paused the goal.

An adversarial review (codex, gpt-5.5 xhigh) caught it: **`d2t` is an offset, not
an absolute map.** Verified: `np.nonzero(t2d)[0] == d2t + arange(32000)`, so the
correct decode is `target_id = draft_idx + d2t[draft_idx]`, and `lm_head[i]`
matches the target's `output.weight[i + d2t[i]]`. The fix is one line; with it,
position-1 accuracy jumped from 0% to ~68%.

Lesson recorded: verify token-mapping semantics algebraically (against `t2d`)
before concluding a representation blocker. My IQ2XXS "precision/SNR at shallow
layers" hypothesis was a red herring — codex also noted layer-42's per-position
RMS is ~5.4, not the 799 I quoted (the 799 was outlier-inflated std).

## What was built and validated (reusable)

- **Numpy forward** (`issue468/dflash_oracle/forward.py`): port of
  `dflash_mlx/model.py`; validated vs the MLX reference to <=0.08% rel at small
  and real scale. Token decode now uses the correct offset mapping.
- **Target HC captures** at layers 3/13/23/32/42 for all 30 cells
  (`issue468/artifacts/dflash_capture/`), bit-identical to the DSpark-validated
  layer-42 capture. Representation confirmed (HC state [4,4096], hc-outer) against
  the HF `inference/model.py`.
- **Measurement harness** (`issue468/dflash_oracle/measure_dflash_acceptance.py`)
  + results (`issue468/artifacts/dflash_acceptance/`).
- Adversarial review prompt + codex log retained at `/tmp/codex_prompt.md`,
  `/tmp/codex_review.log`.

## Open questions (not needed for the verdict)

- How much of the DFlash pos2-7 gap vs val is IQ2XXS vs corpus? Would need a
  higher-precision (Q8/BF16) target for the shallow layers to test cleanly.
- DFlash's parallel drafting is inherently weaker at prefix length than an
  autoregressive drafter; a fairer "attractiveness" axis might be draft-throughput
  (one forward vs DSpark's per-token rollout), not accepted prefix alone.
