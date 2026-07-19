## Verdict Per Claim C1-C5
- **C1: questionable.** F32/60/max_step=3 are verified, but max_step=3 uses only 180/5451 available p1 anchors, all cold KV (`n_real=2..4`); CE-only is not validated as rank-preserving.
- **C2: likely-wrong.** Ratio does inflate tiny HC scale weights, but raw `||grad||` is also size-confounded and is not a LoRA metric.
- **C3: questionable.** `head.hc_fn` dominance is real in the saved CE gradient, but it is **not** a Sinkhorn-iteration artifact. Head has no Sinkhorn path.
- **C4: likely-wrong.** The target set follows only under raw norm. RMS re-ranking materially demotes `lm_head` and raises body HC/router/KV.
- **C5: questionable.** Full body ablations need body LoRA, but a cheap `head.hc_fn`-only/head-LoRA ablation should run now before trusting the #1 result.

## The head.hc_fn / Sinkhorn Question
`head.hc_fn` is not behind iterative Sinkhorn in this probe. The head path is flat -> `hc_fn` -> sigmoid -> HC weighted sum in [drafter_head.py](/Users/lobanov/Projects/ds4-dspark-research/issue468/dspark_train/drafter_head.py:78). The iterative Sinkhorn is only in body `hc_split_sinkhorn` at [drafter_body.py](/Users/lobanov/Projects/ds4-dspark-research/issue468/dspark_train/drafter_body.py:45), and the numpy oracle says head is “SIGMOID reduce” at [hc_primitives.py](/Users/lobanov/Projects/ds4-dspark-research/issue468/dspark_oracle/hc_primitives.py:86).

So: the Sinkhorn explanation for `head.hc_fn` is refuted by code. The spike is a real local CE sensitivity, probably because a 65k matrix gates the final 4x4096 HC state directly before norm/lm_head. It is still not proof of generalizing trainability.

Synthetic Sinkhorn-iteration check also did not show amplification: random `hc_split_sinkhorn` grad norm was ~18.9 at 1 iter and ~18.4 at 20 iters.

## The Metric Question
RMS re-rank changes the story. Re-derived from [gradient_probe_result.json](/Users/lobanov/Projects/ds4-dspark-research/issue468/artifacts/lead10_phase0/gradient_probe_result.json:2) plus tensor shapes:

| group | raw-style rank | group RMS grad |
|---|---:|---:|
| `head.hc_fn` | 1 | `6.78e-2` |
| body `hc_ffn_fn` | low | `3.10e-4` |
| body `hc_attn_fn` | low | `3.08e-4` |
| router `ffn_gate_inp` | low | `2.16e-4` |
| `main_proj` | 2 | `1.85e-4` |
| shared expert | mid | `1.55e-4` |
| attn matrices | mid | `1.35e-4` |
| `lm_head` | 3 raw | `2.51e-5` |

`head.hc_fn` stays #1 by a huge margin. But `lm_head` falls near the bottom per parameter. Raw norm is a full-rank update proxy, not a LoRA proxy. For LoRA, save gradients and rank by top-r singular energy, `sum σ_i^2`, or at minimum RMS.

Also: `lm_head` is loaded from the **target GGUF**, not dspark, at [drafter_head.py](/Users/lobanov/Projects/ds4-dspark-research/issue468/dspark_train/drafter_head.py:149). `embed_w` is also target-loaded at [drafter_body.py](/Users/lobanov/Projects/ds4-dspark-research/issue468/dspark_train/drafter_body.py:205). Training `lm_head` but freezing `embed_w` needs an explicit deployment story.

## New Experiments To Try
1. **Run `head.hc_fn`-only ablation now.** Train only that matrix/direct adapter on existing hard labels, then repeat with KL once labels exist. Signal: held-out p1 improves or the gradient spike is non-actionable. Effort: low.
2. **Re-run the gradient probe with aligned top-128 KL.** Current unified dir lacks `lead3_logprobs.jsonl`; only `lead3_h.bin` and capture metadata are present. Signal: whether CE overweights output-side weights. Effort: low-medium.
3. **Step-bucket gradient probe.** Compare steps `1..3` vs sampled later positions, e.g. 16/32/64. Signal: body HC/attention may rise outside cold KV. Effort: medium.
4. **Save gradients and compute top-r singular energy.** Do this for `head.hc_fn`, `main_proj`, `kv`, body HC, shared expert, `lm_head`. Signal: true rank-r LoRA potential. Effort: medium.
5. **Candidate audit.** Include router, markov heads, HC scale/base direct tuning, and confidence head with an actual confidence loss. Effort: low.

## Leading Hypothesis + Decisive Test
Hypothesis: `head.hc_fn` is a genuine first-order CE sensitivity, not Sinkhorn noise, but the proposed target set is too raw-norm/output-biased and may not generalize.

Decisive test: run a `head.hc_fn`-only held-out ablation. If it improves p1 under KL/soft labels, keep it #1. If it harms or only reduces train loss, demote the gradient spike and select targets by KL plus top-r/RMS metrics.