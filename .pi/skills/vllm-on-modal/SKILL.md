---
name: vllm-on-modal
description: Pitfalls and reliable patterns for running vLLM (esp. on Modal) to capture intermediate hidden states or run large / custom-architecture models. Use before any paid vLLM GPU run — to check EAGLE3/extract support, pick TP, avoid V1-engine walls, and stage free CPU-only probes first.
---

# vLLM on Modal — pitfalls and reliable patterns

Hard-won lessons from trying to use vLLM on Modal to extract intermediate-layer hidden
states from a large, custom-architecture model. None of this is model-specific; it is
the general vLLM/V1/Modal machinery that will bite any hidden-state-capture or
large-model run. Read it **before** spending GPU money.

## When to use

- You plan to run vLLM on Modal (or any host) to **capture intermediate-layer hidden
  states** (e.g., for a drafter/distillation dataset).
- The model is **large relative to one GPU**, or has a **custom/new architecture** or a
  **non-standard quantization**.
- You're about to `modal run` something on a paid GPU and want to derisk it for free first.

## Core lessons (the load-bearing ones)

1. **V1 runs the model in a subprocess — forward hooks from the `LLM` side don't fire.**
   In vLLM V1 (the default), the model lives in the **EngineCore process** (async
   scheduling). The `LLM`/`LLMEngine` object does **not** hold the model:
   `llm.llm_engine.model_executor` does not exist in multiproc mode. → Registering
   PyTorch `register_forward_hook` from the driver does nothing. → To touch the model,
   use **`LLMEngine.collective_rpc(method, ...)` / `apply_model`** (delegates into the
   EngineCore; runs cloudpickled callables in the worker processes). Use it to **arm or
   read a model-owned buffer**, not to recover transient per-layer tensors after forward.

2. **`extract_hidden_states` requires the model to implement EAGLE3.** Stock
   `speculative_config={"method":"extract_hidden_states",...}` +
   `eagle_aux_hidden_state_layer_ids=[...]` needs the model to implement the
   `SupportsEagle3` interface. Many models — especially new/custom arches — don't →
   `RuntimeError: Model does not support EAGLE3 interface`. **Check first:** does the
   model class define `get_eagle3_aux_hidden_states` / `supports_eagle3`? And note the
   extractor buffer is hardcoded to `[max_tokens, num_layers, hidden_size]` — it cannot
   carry a per-layer representation larger than `hidden_size` without surgery.

3. **Reuse the model's existing plumbing before patching.** Before writing a custom
   capture (an EAGLE3 patch, a hook), check whether the model **already computes** what
   you need — e.g. an MTP/draft hidden buffer (`_mtp_hidden_buffer`), an aux-hidden
   branch. Extending an existing, model-owned buffer (populated during `forward`, read
   via `collective_rpc` after) is far cheaper and more robust than a from-scratch patch,
   and it survives the V1 subprocess boundary.

4. **TP=1 OOMs once weights are a large fraction of one GPU.** vLLM's init profiles
   model + KV memory; weights + base overhead + KV-cache profiling don't fit one GPU when
   the model is ~90%+ of VRAM. Use **TP≥2**. Don't infer TP from a model card/blog GPU
   count — those recipes are often for bigger GPUs or **data-parallel** (DP replicates the
   *whole* model per GPU, which requires a GPU that fits the entire model).

5. **vLLM eager-loads weights — there is no engine-level SSD/expert streaming.** Unlike
   some research engines, vLLM has no `--ssd-streaming` that pages routed experts on
   demand. Its memory profiling tends to touch every layer, so weights go resident. A
   model larger than the GPU pool OOMs; mmap *may* page cold MoE experts but it is not
   guaranteed. **Do not assume** a model bigger than GPU memory will "stream."

6. **Quant-method and GGUF-arch are hard load gates.** vLLM loads only the quant methods
   in its registry (`fp8`, `gptq`, `deepseek_*_fp8`, …). A model with an unsupported
   `quantization_config.quant_method` (e.g. a custom hybrid quant) **fails to load** —
   check `vllm/model_executor/layers/quantization/__init__.py`. GGUF loading maps
   **known architectures only** (llama, qwen2, …); a novel/custom GGUF arch has no map
   and won't load. Don't assume a niche-quant HF checkpoint or a research-engine GGUF
   loads in vLLM.

7. **TP residual is usually replicated — don't assume a gather.** Under standard TP
   (attention shards heads then `RowParallelLinear` reduces; MLP uses
   `MergedColumnParallel` + `RowParallel(reduce_results=True)`), a layer's **output
   hidden state is full on every rank**. So a buffer captured in `forward` is complete on
   rank 0 — **no cross-worker gather needed**; dump rank 0 only. (Verify once: compare
   rank hashes.) This removes the scariest part of "capture under TP."

8. **aarch64+CUDA is supported.** vLLM ships aarch64 wheels
   (`manylinux_2_28_aarch64`) and `torch+cu130` aarch64 exist — so vLLM runs on
   Grace-Blackwell (e.g. DGX Spark). Plain `pip install vllm`.

9. **CUDA graphs break forward hooks.** `enforce_eager=False` (cudagraphs ON) gives
   ~2× speedup BUT **forward hooks don't fire during graph replay** → the capture
   list stays empty (zero captures). For hook-based capture, `enforce_eager=True` is
   **required** (~2 toks/s on TP=2 H200 for a 285B MoE; acceptable for small captures,
   slow for full-powered runs). A cudagraph-compatible capture would need a
   model-forward patch (write-to-buffer inside `forward`, part of the graph) — a
   deeper surgery for future optimization.

10. **The chat template is mandatory — raw text produces degenerate output.**
    `llm.generate([raw_text])` is completion-style and does **not** apply the chat
    template. A chat model given an unformatted prompt can produce a **degenerate
    repeating pattern** (e.g. the same 3 tokens every ~4 positions). Always use
    `llm.chat()` or manually render the model's chat format (BOS + system +
    `<｜User｜>` + prompt + `<｜Assistant｜>` + `</think>` for DeepSeek). **Diagnostic:**
    if the greedy repeats and 0%-matches a reference, check `prompt_len` — a mismatch
    vs the reference engine's tokenization = a template issue, not a model/hook bug.

11. **Clone inputs before computing in a forward hook.** A hook that calls a CUDA
    kernel (e.g. `mhc_post_tilelang`) on the layer's OUTPUT tensors **corrupts the
    forward** — those tensors are the next layer's INPUTS; an in-place kernel
    modification silently degrades every subsequent layer. Always `.clone()` the
    inputs before any computation inside a hook: `mhc_post_fn(hs.clone(),
    residual.clone(), ...)`. (Even with clones, verify the greedy is non-repeating
    and the drafter-sanity p=1 is in band.)

12. **The working capture pattern (validated): post-load `apply_model` hooks.** The
    reliable V1 capture = register PyTorch forward hooks on the model's layers
    **AFTER** the model is loaded (not at import), via `llm.apply_model(func)` —
    which runs `func(model)` **inside the EngineCore** where the model lives. Set
    `VLLM_ALLOW_INSECURE_SERIALIZATION=1` (cloudpickle for the function). Fetch the
    captured buffer afterward via a second `apply_model(lambda m:
    m.get_buffer())`. The buffer is replicated across TP ranks (lesson 7) — fetch
    rank 0 only. This pattern avoids the subprocess boundary (lesson 1), the EAGLE3
    requirement (lesson 2), and the CUDA-graph incompatibility (lesson 9).

## Pre-GPU-spend checklist (do these FREE first)

- **CPU-only Modal probes cost nothing.** A Modal function **without** `gpu=` that reads
  the **installed** vLLM source (grep the model class, its `forward`, the EAGLE3
  interface, the engine/`collective_rpc` path) is free. Do this to understand the model
  + engine **before** any paid GPU run. (Don't fight the local Mac for the vLLM source —
  it isn't installed locally; read it in the Modal container.)
- **Confirm the quant method + arch load**: is the model's `quant_method` in vLLM's
  registry? If GGUF, is the arch mapped? If "no", the GPU run will fail at load — don't
  pay to find out.
- **Confirm EAGLE3 support** (if you planned to use `extract_hidden_states`).
- **Compute the TP floor** from the model weight size vs GPU VRAM (leave headroom for
  vLLM base + KV); pick the **cheapest TP that fits** (TP=2 if 2×VRAM ≫ weights).
- **Mock the capture logic locally** (fake `forward` + dummy tensors) to prove the
  indexing/shape/buffer code before running the real model.
- **Stand-in test the V1 deployment mechanism** on a **small, loadable** model (e.g.
  Qwen3-1.7B): prove the patch-module loads in the EngineCore and `collective_rpc`
  fetches — this derisks the part that breaks (the subprocess boundary), without the
  large model.

## vLLM API gotchas

- `SamplingParams(logprobs=N)` is capped at **20** by default → raise with
  `LLM(max_logprobs=N)`.
- Some connectors/features require `enable_chunked_prefill=False`.
- `VLLM_ALLOW_INSECURE_SERIALIZATION=1` is required to pass Python functions (lambdas,
  closures) through `apply_model` / `collective_rpc` across the EngineCore subprocess
  boundary (uses cloudpickle). Without it: `TypeError: Object of type function is not
  serializable`.
- `modal.Image.add_local_file(path, remote_path)` must use `copy=True` if any `.env()` or
  build step follows it; otherwise: `InvalidError: an image tried to run a build step
  after using add_local_*`. Or place `add_local_*` calls last.
- `llm.chat()` applies the chat template; `llm.generate([raw_text])` does NOT. For
  chat-trained models, always use `chat()` or render the template manually to avoid
  degenerate output (lesson 10).
- `enforce_eager=True` skips torch.compile + CUDA graphs (faster bootstrap, slower
  inference) and is **required** by some paths (e.g. CacheOnly-style extraction).
- `kv_cache_dtype` may be **required** by certain attention layouts (an assertion fires
  if wrong) — read the error.
- `output.kv_transfer_params["hidden_states_path"]` is the offline retrieval surface for
  the `extract_hidden_states` KV-connector (a `.safetensors` with `token_ids` +
  `hidden_states`); the connector's write can **lag** `generate()` returning on a warm
  request — retry-read the file.

## Modal-specific

- **`modal run --detach` backgrounded with `&` is unreliable across CLI calls** — the
  local process can die between bash invocations, aborting the in-flight remote call.
  For a reliable long run, use **foreground with a generous timeout**, or a backgrounded
  **runner script** on the remote that waits + logs (then poll one log).
- **The model load dominates cost**, not inference. Each GPU Modal run's minimum cost is
  load + vLLM init (minutes for large models); the actual generate is seconds. Budget
  smokes so **one run answers one question**.
- Read HF-cache config/model code from the **Modal volume** (HF cache symlinks don't
  survive `modal volume get` to local — read server-side via a cpu function).

## Methodology

- **Dispatch an adversarial-codex-review at vLLM-internal walls.** When you hit an
  interface/TP-sharding/capture-mechanism wall, codex fetches the vLLM source and
  independently confirms the structure + finds the clean mechanism (e.g. it caught
  "residual is replicated, no gather" + "extend the MTP buffer"). Hours of solo
  source-spelunking miss what a fresh adversarial pass finds in one review.
- **Independently verify codex's decisive claims** before acting (re-derive the key one
  yourself).

## Guardrails

- **Never pay for a GPU run to learn something a free CPU-only probe could tell you**
  (quant-method support, EAGLE3 support, the model's `forward` structure, the engine
  path). Read the installed source first.
- **Never assume TP=1 fits** a model that's a large fraction of one GPU's VRAM.
- **Never assume a custom-quant HF checkpoint or a novel-arch GGUF loads in vLLM** —
  check the registry/arch map.
- **Never register forward hooks from the `LLM` side in V1 and expect them to fire** —
  the model is in a subprocess; use a model-owned buffer + `collective_rpc`.
- **Never reinvent a capture if the model already has the plumbing** (MTP/aux buffer) —
  extend it.
- **Never background a detached Modal run you need to stay alive** across calls without a
  runner script + log.

## Anti-patterns (from experience)

- **Trusting stock `extract_hidden_states`** on a new arch without checking EAGLE3 → the
  EAGLE3 RuntimeError, after a paid model load.
- **TP=1 on a near-full-GPU model** → OOM during KV-cache profiling.
- **Forward hooks from the driver in V1** → silently never fire (model in subprocess).
- **Guessing the inner-model path** (`llm_engine.model_executor...`) in V1 → AttributeError.
- **A niche-quant / custom-arch GGUF assumed to load** → load failure, wasted download.
- **From-scratch EAGLE3 patch** when the model already had an MTP hidden buffer → weeks
  of avoidable surgery.
