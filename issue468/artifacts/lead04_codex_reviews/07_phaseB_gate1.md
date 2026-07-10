## C1 Indexing

Off-by-one is correct for target-rank: `hidden_states[i] -> logprobs[i+1] -> g_{i+1}`.

Evidence: Stage2 says `main_hidden[p]` is post-token hidden and predicts `token[p+1]` ([stage2_capture_store.py](/Users/lobanov/Projects/ds4-dspark-research/issue468/dspark_oracle/stage2_capture_store.py:13)). The harness loads `mh0` only as KV init, then measured rows start at `step=1`, with anchor `target_tokens[step]` and target `target_tokens[step+1...]` ([measure_acceptance_bundle.py](/Users/lobanov/Projects/ds4-dspark-research/issue468/dspark_oracle/measure_acceptance_bundle.py:218), [measure_acceptance_bundle.py](/Users/lobanov/Projects/ds4-dspark-research/issue468/dspark_oracle/measure_acceptance_bundle.py:234)). vLLM logprobs are per output token, so `logprobs[0]` is the distribution that selected `g0`, not the next-token distribution after `h0` ([vLLM SamplingParams](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/sampling_params.py)).

Right alignment:
`h[i] = post(g_i)`, anchor `g_i`, p1 target `g_{i+1}`, target rank from `topk/logprobs[i+1]`. For K-position analysis use `logprobs[step+p]`, matching prior Q2 code ([run_stage0_quant_mismatch.py](/Users/lobanov/Projects/ds4-dspark-research/issue468/run_stage0_quant_mismatch.py:190)).

Blocker: current converter is sloppy. `capture_hc_modal.py` saves `n_gen=len(greedy)` while layer arrays are `n_dec` ([capture_hc_modal.py](/Users/lobanov/Projects/ds4-dspark-research/issue468/run_lead04_modal/capture_hc_modal.py:143)); `convert_vllm_to_oracle.py` uses `d["n_gen"]` to build `positions` without asserting it equals layer length ([convert_vllm_to_oracle.py](/Users/lobanov/Projects/ds4-dspark-research/issue468/run_lead04_modal/convert_vllm_to_oracle.py:39)). Fix to explicit `n_capture`, `n_generated`, `logprob_shift=+1`.

## C2 Estimator

Prompt-paired Δp1 with prompt bootstrap is the right estimator. Positions are correlated; resampling positions would be wrong.

Pilot 7-vs-8 anchors was not clean. Full 128-token capture should be comparable only if FP keeps full greedy labels: anchors are `len(tokens)-BLOCK-1`, usually `122` for 128 tokens. Q2 combined300 already has variable `n_steps` due early stops, mean `115.5`, 48 prompts not 122, so compare per-prompt rates, not raw counts.

Caveat: FP-vs-Q2 greedy trajectories may differ. That makes the estimand “native FP trajectory vs Q2 trajectory,” not pure hidden degradation. Use common-prefix/crossed-oracle as sensitivity if Δ is small.

## C3 Kernel Noise

Confound, not harmless noise. `max_model_len` changing p1 by ~4pp is deterministic instrumentation bias; prompt bootstrap will not capture it.

Control it before interpreting +2pp:
lock `max_model_len=2048` for smoke and full run, rerun a small paired sensitivity at the exact final config, and report kernel sensitivity as systematic error. A +2pp GO threshold is not credible if uncontrolled kernel drift is ~4pp.

## C4 p1 Metric

Correct: `p1 = 1 - prefix_hist[0] / total_anchors`.

`prefix=0` means first draft token missed; torch compact computes the same thing directly as `draft[0] == target[0]` ([run_lead03_torch_measure.py](/Users/lobanov/Projects/ds4-dspark-research/issue468/run_lead03_torch_measure.py:83)). `match_pct` is all-slot match rate, not p1.

## C5 Logits vs Logprobs

Logprobs are sufficient for rank and probability: rank from sorted ids, probability from `exp(logprob)`. Logit gaps within one distribution are also preserved by log-softmax differences.

Asymmetry is only a problem if you compare absolute raw logits FP-vs-Q2. Do rank/logprob-space analysis on both. Also mask padded `topk_ids == -1`; FP pads logprobs with `0.0`, which is dangerous if unmasked.

## C6 Decision Rule

Not well-specified enough.

If +2pp is the practical threshold, GO should require CI lower bound above +2pp, or at least above kernel systematic. Current “Δ≥+2pp and CI lower >0” can GO without proving the +2pp threshold. STOP should be `CI upper < +2pp`, even if point estimate is small positive. “and/or target-rank” also needs a predeclared quantitative rule.

## GO/NO-GO

NO-GO as written.

Fix C1 schema/alignment first, then run a 5-prompt logprobs smoke that asserts:
`n_logprobs == n_generated`, `n_capture == n_generated - 1`, `len(positions) == n_capture`, full greedy labels retained, and rank analysis uses `logprobs[step+p]`. Also lock/report kernel config sensitivity. After that, the 300-prompt capture is technically defensible.