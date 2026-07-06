**A. Concrete Errors**

1. **The corrected cycle model is wrong for rejection.**  
   The verifier can produce the correction token’s logits, but it has not run the target on that correction token, so there is no valid KV/hidden state for the next anchor. Current ds4 does the opposite of the model: every speculative cycle starts by decoding `first_token` via `ds4_session_eval` before drafting ([ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:27202)). The verifier commits states only for draft tokens actually processed, not for an argmax correction token ([ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:27555)).

   Correct realizable K=4 cost is effectively:
   `cost = decode + draft + verify = 26 + 10 + 65.8 = 101.8 ms`, not `83.3 ms`.

   That changes K=4 current speedup from:
   `0.991 (-0.9%)` to `0.811 (-18.9%)`.

2. **The K=4 acceptance threshold is not ~79% if anchor reuse is unavailable.**  
   With decode every cycle, K=4 needs geometric `p = 0.877`, not `0.792`, just to break baseline. The +20% gate needs `p = 0.969`, not `0.940`.

3. **The output-head cost is not omitted from `verify_ms`.**  
   `metal_graph_verify_suffix_tops` calls the batched output head ([ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:21159)), and that head runs the Q8 vocab matmul ([ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:16207)). So the ~545 MB vocab projection is inside the measured verifier body. What is omitted is the dependent target decode needed to turn the correction token into valid next-cycle state.

4. **`verify_ms` is not full cycle overhead.**  
   The K=4 long benchmark has `mean_total_ms - mean_draft_ms - mean_verify_ms` around 40-45 ms across the three 8k prompts. That is not noise; it is state management, replay/readback, anchor decode, and first-miss behavior outside the median verifier number.

5. **The “2 pp short” precision is statistically indefensible.**  
   Temp=0 K=4 `E[a]` across the 10 prompts has mean `2.175`, sd `0.426`, rough prompt-level 95% half-width `±0.305`. The model’s break-even gap is only `0.028` accepted drafts/cycle. Prompt-level modeled K=4 speed ranges from `-18.5%` to `+10.6%`.

**B. Load-Bearing Unverified Assumptions**

1. **Verifier-produced anchor state exists: bias +18 pp at K=4.**  
   This is the load-bearing failure. Current code produces logits/top ids, not a processed correction-token hidden/KV state. If wrong, K=4 goes from `-0.9%` to `-18.9%`.

2. **MTP verifier timing transfers to DSpark: bias easily 5-15+ pp.**  
   The timer includes output head, but not the full speculative state machinery. Any extra +5 ms drops K=4 from `0.991` to `0.935`; +10 ms drops it to `0.885`.

3. **Acceptance corpus represents real cycles: bias several to 10+ pp.**  
   The oracle walks target-stream anchors from tiny prompts, 67-165 tokens, not real speculative trajectories ([measure_acceptance_bundle.py](/Users/lobanov/Projects/ds4-dspark-research/issue468/dspark_oracle/measure_acceptance_bundle.py:158)). Temp 0.5/1.0 lowers K=4 E from `2.175` to about `1.92`, roughly a 7 pp speed hit under the optimistic model.

4. **Batch verifier is exact enough.**  
   ds4 comments already admit batch verification can perturb greedy tokens; strict exactness exists for N=2, not a proven exact K=4 path. If exact K=4 requires replay or sequential decode, the speed model collapses.

5. **Draft cost = 10 ms constant.**  
   K=4 has only `-0.725 ms` of cost headroom even in the optimistic model. A +1 ms draft miss moves it from `-0.9%` to about `-2.0%`; +5 ms to `-6.5%`.

**C. Verdict**

- **Q1:** overturned for any currently realizable ds4 path. Baseline at K=4 needs ~88% geometric acceptance, not ~79%, unless someone proves correction-token state reuse.
- **Q2:** fragile to the point of useless. “K=4 is break-even” only holds under the unproven and likely false anchor-state assumption.
- **Q3:** mostly survives directionally. A 4-node tree is bad under measured verifier cost. But “cannot help” is too absolute: if 4-node tree verify fell from `65.8 ms` to `<=50.7 ms`, a perfect depth-3 tree could clear +20%.
- **Headline:** does not survive. The current drafter is not “~2 pp from baseline” on the ds4 Metal implementation. It is more like `~10.6 pp` geometric acceptance short at K=4 after correcting the missing decode-every-cycle cost.

**D. The One Experiment**

Force a rejection and test whether the alleged anchor exists.

Run a K=4 verifier microbench on one prompt, deliberately replace `draft[0]` or `draft[1]` with a wrong token, read the verifier correction token, then try to start the next DSpark/MTP draft without calling `metal_graph_eval_token_raw_swa(correction_token)`. Compare next-step logits/argmax against a baseline branch that does decode the correction token.

Pass condition for the headline: no extra target decode, exact next logits, valid next drafter hidden state.  
Expected result: impossible or divergent until the correction token is decoded. That single result falsifies the “verify-produced anchor” premise and forces the `decode + draft + verify` model.
