## Root Cause

**Primary: H1, missing/mismatched chat template.**  
[ capture_hc_modal.py](/Users/lobanov/Projects/ds4-dspark-research/issue468/run_lead04_modal/capture_hc_modal.py:78) calls `LLM.generate([prompt])` on raw text. vLLM `generate()` is completion-style; `chat()` is the path that renders messages through the tokenizer chat template and then calls generate. Source: [vLLM LLM source](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/entrypoints/llm.py), [vLLM renderer source](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/renderers/hf.py).

DS4 does not feed raw text. It wraps prompt as:

`BOS + system + <｜User｜> + prompt + <｜Assistant｜> + </think>`

Evidence: [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:22298), [ds4_server.c](/Users/lobanov/Projects/ds4-dspark-research/ds4_server.c:2312), default CLI system at [ds4_cli.c](/Users/lobanov/Projects/ds4-dspark-research/ds4_cli.c:1500). The 60 vs 70 token delta is exactly consistent with “raw vLLM prompt” vs “DS4 chat prompt with system + special tokens.”

**H2 is a real bug/risk, but not proven as the repeating-greedy cause.**  
[dspark_hc_patch.py](/Users/lobanov/Projects/ds4-dspark-research/issue468/run_lead04_modal/dspark_hc_patch.py:83) still calls `mhc_post_tilelang` inside the forward hook. Cloning only protects those input tensors; it does not rule out global workspace, stream, or kernel side effects. Also this is probably semantically wrong/unneeded: the older hook captures `out[0]` directly as the post-FFN HC residual in [capture_hc_residual.py](/Users/lobanov/Projects/ds4-dspark-research/issue468/run_lead04_modal/capture_hc_residual.py:87).

**H3: unproven.** No evidence yet that `enforce_eager=True` alone degenerates DeepSeek-V4-Flash. Test it only after fixing prompt and hook.

**H4: unlikely primary.** The observed token delta is explained by prompt rendering. Tokenizer-version mismatch remains possible only if DS4-rendered prompt token IDs still disagree.

## The Fix

1. **Use the exact DS4 chat prompt.** Prefer manual DS4 rendering or `llm.chat()` only if it renders byte/token-identical output: same system string, user role, `add_generation_prompt=True`, non-thinking `</think>`. Assert `prompt_len == 70` for this case.

2. **Remove `mhc_post_tilelang` from the hook.** Capture only:
   `hs = out[0]; hs.detach().flatten(1).float().cpu()`.  
   Do the HC mean/concat offline.

3. Only if clean templated runs still repeat, test `enforce_eager=False`.

## The Decisive Test

Run this matrix, same prompt, temp 0:

1. no hooks + raw `generate`
2. no hooks + DS4-rendered/chat-template prompt
3. copy-only hook + DS4-rendered prompt
4. current `mhc_post` hook + DS4-rendered prompt

If 1 repeats and 2 is sane: root cause is H1.  
If 2 sane and 4 repeats: H2.  
If 2 repeats with `enforce_eager=True` but not `False`: H3.  
If templated token IDs still differ from DS4: H4.

## Drafter Comparison

Yes. The drafter comparison must use the same DS4 chat template and non-thinking mode. Otherwise prompt length, absolute positions, hidden states, anchors, and greedy labels are from a different trajectory.