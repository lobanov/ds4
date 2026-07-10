## Verdict Per Wall W1-W4

- **W1: confirmed.** `extract_hidden_states` forces aux hidden outputs, and GPU runner raises if `supports_eagle3(self.get_model())` is false. The extractor buffer is `[max_tokens, num_layers, hidden_size]`, with `hidden_size = model_config.get_hidden_size()`, so it cannot carry `[4,4096]` HC streams without surgery. Cite: runner EAGLE3 gate ([raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.24.0/vllm/v1/worker/gpu_model_runner.py)), extractor buffer ([raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/spec_decode/extract_hidden_states.py)). Nuance: newer/main DeepSeek-V4 code has an aux branch, but it does `aux_recon.mean(dim=1)`, destroying distinct HC components ([raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/models/deepseek_v4/nvidia/model.py)).
- **W2: confirmed, with nuance.** In V1 multiprocessing, the user-side `LLMEngine` only sets `model_executor` when `not multiprocess_mode`; otherwise model lives behind `EngineCoreClient` ([raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.24.0/vllm/v1/engine/llm_engine.py)). Direct hooks from the parent object are the wrong control plane.
- **W3: confirmed operationally.** TP=1 is a bad bet for 129 GB weights plus vLLM/KV/profiling overhead; source confirms init profiles model/KV memory before serving ([raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.24.0/vllm/v1/worker/gpu_worker.py)). TP=4 is the sane target.
- **W4: mostly confirmed.** vLLM’s quant registry includes `deepseek_v4_fp8`, not `deepseek_v4_hybrid_iq2` ([raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.24.0/vllm/model_executor/layers/quantization/__init__.py)); DeepSeek-V4’s own quant config names only `deepseek_v4_fp8` and handles FP4/FP8 via `expert_dtype` inside that method ([raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/models/deepseek_v4/quant_config.py)). No cheap DGX-Spark vLLM rehearsal.

## TP=4 Residual: Replicated Or Sharded?

**Replicated. No HC all_gather needed** under normal TP=4, PP=1, no sequence-parallel residual path.

Evidence: attention shards heads (`n_local_heads = n_heads // tp_size`) but projects back through `RowParallelLinear` `wo_b`, then returns `_o_proj(...)` ([raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/models/deepseek_v4/attention.py)) ([raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/models/deepseek_v4/attention.py)). MLP uses `MergedColumnParallelLinear` then `RowParallelLinear(... reduce_results=True)` by default ([raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/models/deepseek_v4/nvidia/model.py)). The model’s own PP intermediate tensor shape is full `[batch, hc_mult, hidden_size]`, and `_mtp_hidden_buffer` stores full flattened HC `[tok, hc_mult * hidden]` ([raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/models/deepseek_v4/nvidia/model.py)).

So a patch sees full `[tok,4,4096]` on every TP rank. Dump only TP rank 0 to avoid duplicates. Disable sequence-parallel/PP complications for the capture run.

## Cleanest Viable V1 Capture Mechanism

No stock return-hidden API exists for this. `SamplingParams` exposes logprobs and extras, not hidden-state output ([raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.24.0/vllm/sampling_params.py)).

There is a useful V1 control plane: `LLMEngine.collective_rpc` / `apply_model` delegates to EngineCore/executor, and multiproc workers can execute cloudpickled callables inside worker processes ([raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.24.0/vllm/v1/engine/llm_engine.py)) ([raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.24.0/vllm/v1/engine/core.py)) ([raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.24.0/vllm/v1/executor/multiproc_executor.py)). But it cannot recover transient per-layer tensors after forward. Use it to **arm/read a model-owned buffer**, not as the capture mechanism itself.

Cleanest path: extend the existing `_mtp_hidden_buffer` pattern to `_dspark_hc_buffer[:, 3, 16384]`. In `DeepseekV4Model.forward`, after layers 40/41/42, compute `mhc_post_tilelang(hidden_states,residual,post_mix,res_mix).flatten(1)` and `copy_` into the buffer. Existing final MTP buffer already does the same stable-address copy pattern ([raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/models/deepseek_v4/nvidia/model.py)), and `get_mtp_target_hidden_states()` exposes the final buffer ([raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/models/deepseek_v4/nvidia/model.py)).

Do **not** write files inside forward. Buffer on GPU, then `collective_rpc/apply_model` to fetch rank-0 CPU data after the request.

## Realism Verdict

Stock vLLM path is effectively infeasible for this target hidden. But **native-FP capture via a minimal DeepSeek-V4 model-forward buffer patch is realistic** for one decisive TP=4 run. The investigator overestimated gather complexity; TP hidden is replicated.

Risk is engineering/deployment, not math: patch must load inside EngineCore workers before model load, avoid CUDA graph side effects, and dump only rank 0.

## Recommendation

Run exactly one TP=4 Modal smoke test with the buffer patch.

Decisive test: one short prompt, capture layers `[40,41,42]`, then `collective_rpc` returns for each TP rank: shape, dtype, token count, and hash/norm of `_dspark_hc_buffer[:n]`. Pass if every rank reports `[n,3,16384]` and rank hashes match. Then write rank-0 buffer and proceed to DSpark. If shape is `[n,3,4096]`, HC streams were averaged; if rank hashes differ as shards, stop and reassess gather.