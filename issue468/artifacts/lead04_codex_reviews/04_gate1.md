## Verdict per claim C1-C6

- **C1: questionable.** ds4’s `hc_ffn_post` is `after_ffn_hc` after FFN HC post, not `out[0]` alone: [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:16067). HF reference also builds DSpark `main_hidden` from target-layer `h.mean(dim=2)` with target layers `[40,41,42]` ([model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/blob/main/inference/model.py), [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/blob/main/config.json)). But the patch proves this only if vLLM’s layer hook output tuple is exactly `(ffn_out, after_attn_hc, ffn_post, ffn_comb)`. Smoke sanity is not semantic proof. Also dead `apply_patch()` path calls a nonexistent `mhc_post_fn=` kwarg.

- **C2: questionable.** `13` captures for `14` generated tokens is expected: first token is sampled from prefill logits; last generated token is never processed as input. Captured positions should be prompt_len..prompt_len+12. But [capture_hc_modal.py](/Users/lobanov/Projects/ds4-dspark-research/issue468/run_lead04_modal/capture_hc_modal.py:87) reports `n_gen=len(greedy)=14`, while saved `n_gen` is capture count `13`; the smoke pass check compares shapes to the wrong `n_gen`.

- **C3: sound, conditional on C1.** Converter mean axis and concat order match [build_main_hidden_from_captures.py](/Users/lobanov/Projects/ds4-dspark-research/issue468/dspark_oracle/build_main_hidden_from_captures.py:92): per layer mean over HC, concat layers `(40,41,42)`. [convert_vllm_to_oracle.py](/Users/lobanov/Projects/ds4-dspark-research/issue468/run_lead04_modal/convert_vllm_to_oracle.py:41) does `[n,4,4096].mean(axis=1)` then concat.

- **C4: likely-wrong.** The 69 vs 70 gap is not proven to be “BPE version diff.” ds4 pushes BOS/User/Assistant/`</think>` as special token IDs and BPE-tokenizes only text: [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:22304). vLLM runner passes one raw concatenated string: [capture_hc_modal.py](/Users/lobanov/Projects/ds4-dspark-research/issue468/run_lead04_modal/capture_hc_modal.py:81). This is a structural-tokenization risk and breaks exact D1.

- **C5: likely-wrong.** `0.6857 from 35 positions` is not p=1. `validate_fidelity.py` computes `total_match / total_positions`, and `total_positions = rows * BLOCK` in [measure_acceptance_bundle.py](/Users/lobanov/Projects/ds4-dspark-research/issue468/dspark_oracle/measure_acceptance_bundle.py:307). That is all-slot match rate, not first-token p=1. If treated anyway as 24/35 Bernoulli, Wilson 95% CI is ~`[0.52, 0.81]`; GREEN is not defensible.

- **C6: likely-wrong.** `3.43 >> Q2 2.34` compares one FP smoke to corpus-wide Q2. Local same-prompt `code_histogram__t0p0` Q2 avg_prefix is `3.125`, while exactness-small corpus avg is `2.3375`. With prompt_len mismatch and different greedy trajectory, this is only a non-garbage sanity signal, not FP-ceiling-above-Q2 evidence.

## The one thing to fix BEFORE the pilot

Fix the indexing/validation harness: keep `main_hidden` length = captured states, but keep **full greedy labels** in `target_selected_tokens.json`; compute real p=1 as `draft[0] == target[0]` over rows, not `total_match / total_positions`; rename/report `n_capture` vs `n_generated`.

## Is the pipeline READY for the 5-prompt pilot?

**fix-first / HOLD.** Capture may be basically working, but the current validator and D1/indexing can produce misleading GREEN results. Do not spend pilot runs until that harness is corrected and v9 is re-scored.