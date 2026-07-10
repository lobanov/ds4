## Verdict On The Design
1. **EAGLE3 mechanism: partial.** Yes, `extract_hidden_states` uses EAGLE3 aux plumbing, and the model must capture aux states in `forward`; the runner only unpacks `model_output` as `(hidden_states, aux_hidden_states)` when aux output is enabled. It does not hook layers itself. DeepseekV2 proves capture is inside `Model.forward`.  

2. **Cross-worker gather: not handled generically.** vLLM runner/proposer just slices/stacks/cats aux tensors. DeepseekV2 does its own `tensor_model_parallel_all_gather` inside model forward when needed. If DeepseekV4 HC residual is TP-sharded, the patch must gather it before returning aux states.  

3. **Existing MTP path: real, underused.** vLLM already has a DeepSeek V4-specific override: `get_mtp_target_hidden_states()` replaces target hiddens for the drafter path. That is likely the simpler integration point for final-layer HC residual, but it only gives the final pre-`hc_head` buffer today, not layers 40/41/42. ([raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.24.0/vllm/v1/worker/gpu_model_runner.py))

4. **Representation match: unproven.** The ds4 dump is explicitly `after_ffn_hc = HCPost(routed_out + shared_out, residual_hc, split)` and is dumped as `hc_ffn_post`. [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:15779), [metal/dsv4_hc.metal](/Users/lobanov/Projects/ds4-dspark-research/metal/dsv4_hc.metal:622). In vLLM, because `post_mix/res_mix` are carried and a final `mhc_post_tilelang(...)` happens after the loop, a naive “capture after `layer()` returns” may be one post step too early. Verify exact materialization point before coding.

5. **Converter: mostly sound, but only after shape/path fixes.** The local oracle builds `main_hidden` by `arr.reshape(4,4096).mean(axis=0)` per layer then concatenates layers 40/41/42. [build_main_hidden_from_captures.py](/Users/lobanov/Projects/ds4-dspark-research/issue468/dspark_oracle/build_main_hidden_from_captures.py:66). The drafter consumes `mh.reshape(..., 3 * DIM) @ main_proj.T`. [drafter_body.py](/Users/lobanov/Projects/ds4-dspark-research/issue468/dspark_train/drafter_body.py:137). Axis order must be `[layer40, layer41, layer42][hc0..3][dim]`.

6. **Monkeypatch deployment: possible, brittle.** Runtime protocol check is structural: `supports_eagle3()` uses `isinstance(model, SupportsEagle3)`, and the protocol requires `supports_eagle3` plus `set_aux_hidden_state_layers` and default-layer method semantics.  Patch must run in every TP worker before model load. Image/source patch or `sitecustomize` is safer than driver-only import.

## Correct EAGLE3 Aux Contract
The runner sets `use_aux_hidden_state_outputs=True` for `extract_hidden_states`, checks `supports_eagle3(self.get_model())`, reads `eagle_aux_hidden_state_layer_ids` from draft config, then calls `model.set_aux_hidden_state_layers(aux_layers)`. 

During forward, the model must return:

```python
hidden_states, aux_hidden_states
```

where `aux_hidden_states` is a list of tensors, one per requested layer. The runner does not call `get_eagle3_aux_hidden_states()`. It just unpacks the forward result. 

Critical problem: stock `ExtractHiddenStatesProposer` allocates `[max_tokens, num_layers, model_config.get_hidden_size()]` and documents each aux tensor as `[num_tokens, hidden_size]`. For DeepseekV4 config `hidden_size=4096`, raw `[tokens, 16384]` HC tensors will not fit without also patching the extractor/cache-only model path. 

## Simpler Alternative
Reuse the existing DeepSeek V4 MTP hidden override path if possible. vLLM already knows that DeepSeek V4 MTP needs the pre-`hc_head` residual and calls `get_mtp_target_hidden_states()`. Extending that buffer/capture to three layers plus a small custom dump connector is likely less invasive than pretending stock `extract_hidden_states` supports 16k aux states. ([raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.24.0/vllm/v1/worker/gpu_model_runner.py))

## Cross-Worker Gather Verdict
Not handled by generic EAGLE3/extract plumbing. If the captured HC tensor is local-sharded under TP=4, this patch is bigger: gather inside DeepseekV4 forward, and patch extractor buffer dimensions. First smoke must print per-rank `_mtp_hidden_buffer.shape` under TP=4.

## Ranked Risks
1. **Shape mismatch:** stock extractor expects 4096, proposed output is 16384.
2. **Wrong capture point:** must capture materialized post-FFN HC after `mhc_post`, not pending pre-post state.
3. **TP sharding:** no generic hidden-dim gather.
4. **Monkeypatch scope:** must execute in worker processes before load.
5. **Precision wording:** extracted activations are `model_config.dtype` from quantized kernels, not FP4/FP8 tensors.

One thing to verify before writing code: under TP=4, at layers 40/41/42, capture both raw loop output and post-`mhc_post` output for one token, compare layer42 post value to existing `_mtp_hidden_buffer`, and compare all three to ds4 `hc_ffn_post` within expected quantized-kernel tolerance.