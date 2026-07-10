#!/usr/bin/env python3
"""Lead 04 — Modal TP=2 decisive smoke (v2): post-load patch via apply_model.

The native model LOADS on 2xH200 (confirmed v1). The patch is applied POST-load via
LLM.apply_model(_register_on_model) — which runs inside the EngineCore (where the model
lives), avoiding both the V1 hooks-from-LLM failure AND the sitecustomize-too-early
import issue. VLLM_ALLOW_INSECURE_SERIALIZATION=1 enables cloudpickle for the functions.
Fetch via apply_model(_fetch_buffer). Decisive: [n,3,16384] replicated across both ranks.
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import modal

MINUTES = 60
MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-DSpark"
GPU_CONFIG = "H200:2"

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=False)
output_vol = modal.Volume.from_name("lead04-captures", create_if_missing=True)

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .uv_pip_install("vllm>=0.18.0", "transformers>=5.6,<6", "huggingface-hub>=1.5.0,<2.0",
                    "numpy", "safetensors", "ninja")
    .add_local_file("dspark_hc_patch.py", "/opt/dspark_hc_patch.py", copy=True)
    .env({
        "PYTHONPATH": "/opt",
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",   # allow cloudpickle for apply_model funcs
        "TRITON_CACHE_DIR": "/output/triton_cache",
        "TORCHINDUCTOR_CACHE_DIR": "/output/inductor_cache",
    })
)
app = modal.App("lead04-decisive-smoke")


# ---- top-level functions (cloudpickle-serializable; run inside the EngineCore) ----
def _register_on_model(m):
    """Register the dspark hooks on the loaded inner model. Runs in EngineCore."""
    import dspark_hc_patch as P
    inner = m.model if hasattr(m, "model") else m
    P.register_dspark_hooks(inner)
    return f"registered on {type(inner).__name__}, {len(inner.layers)} layers"


def _fetch_buffer(m):
    """Fetch the _dspark_hc_buffer from the inner model. Runs in EngineCore."""
    inner = m.model if hasattr(m, "model") else m
    fn = getattr(inner, "get_dspark_hc_buffer", None)
    if fn is None:
        return None
    return fn().cpu()

def _reset_captures(m):
    """Clear the capture lists. Call before each prompt to avoid accumulation."""
    inner = m.model if hasattr(m, "model") else m
    fn = getattr(inner, "reset_dspark_captures", None)
    if fn:
        fn()
    return "reset"


@app.cls(image=vllm_image, gpu=GPU_CONFIG, cpu=2, timeout=45 * MINUTES,
         volumes={"/root/.cache/huggingface": hf_cache_vol, "/output": output_vol},
         secrets=[modal.Secret.from_name("huggingface-token")],
         scaledown_window=15 * MINUTES)
class DecisiveSmoke:
    @modal.enter()
    def start(self):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=MODEL_ID, trust_remote_code=True, kv_cache_dtype="fp8",
                       enforce_eager=True, max_model_len=512, tensor_parallel_size=2,
                       max_num_seqs=1, enable_prefix_caching=False)
        self.SamplingParams = SamplingParams
        print(f"[smoke] loaded {MODEL_ID} on {GPU_CONFIG}", flush=True)
        # register the hooks on the loaded model (inside the EngineCore)
        reg = self.llm.apply_model(_register_on_model)
        print(f"[smoke] hook registration: {reg}", flush=True)

    @modal.method()
    def capture(self, prompt: str, max_tokens: int = 14, prompt_name: str = "smoke") -> dict:
        import numpy as np
        self.llm.apply_model(_reset_captures)  # clear captures from previous prompts
        sp = self.SamplingParams(temperature=0, max_tokens=max_tokens)
        # Manually construct the DS4 chat format (matching ds4.c:22282 encode_chat_prompt
        # + ds4_cli.c:1402 default system): BOS + "You are a helpful assistant" +
        # <｜User｜> + prompt + <｜Assistant｜> + </think> (non-thinking mode).
        ds4_prompt = ("<｜begin▁of▁sentence｜>You are a helpful assistant"
                      "<｜User｜>" + prompt + "<｜Assistant｜></think>")
        res = self.llm.generate([ds4_prompt], sp)[0]
        greedy = list(res.outputs[0].token_ids)
        prompt_len = len(res.prompt_token_ids)
        bufs = self.llm.apply_model(_fetch_buffer)   # list[rank] of [n,3,16384] or None
        out = {"prompt_len": prompt_len, "n_gen": len(greedy), "greedy_head": greedy[:8],
               "n_ranks": len(bufs)}
        for r, b in enumerate(bufs):
            if b is None:
                out[f"rank{r}"] = None; continue
            arr = b.numpy() if hasattr(b, "numpy") else np.asarray(b)
            out[f"rank{r}_shape"] = list(arr.shape)
            out[f"rank{r}_hash"] = hashlib.md5(arr.tobytes()).hexdigest()[:16]
            out[f"rank{r}_mean"] = float(arr.mean()) if arr.size else None
            out[f"rank{r}_nan"] = bool(np.isnan(arr).any()) if arr.size else None
        if bufs and bufs[0] is not None:
            arr0 = bufs[0].numpy() if hasattr(bufs[0], "numpy") else np.asarray(bufs[0])
            n = arr0.shape[0]
            layers = {f"layer{L}": arr0[:, i, :].reshape(n, 4, 4096).astype(np.float32)
                      for i, L in enumerate([40, 41, 42])}
            np.savez(f"/output/pilot_{prompt_name}.npz",
                     **layers, layer_ids=np.array([40, 41, 42], dtype=np.int32),
                     n_gen=np.int32(n), prompt_tokens=np.int32(prompt_len),
                     greedy_tokens=np.array(greedy, dtype=np.int64))
            output_vol.commit()
            out["saved"] = f"/output/pilot_{prompt_name}.npz"
        return out

    @modal.exit()
    def stop(self):
        del self.llm


@app.local_entrypoint()
def main(prompt: str = "The capital of France is", max_tokens: int = 14):
    """Capture multiple exactness prompts in one warm session."""
    from pathlib import Path
    prompts = {}
    corpus_dir = Path.cwd() / ".." / "prompts" / "exactness_small_corpus"
    manifest = json.load(open(corpus_dir / "manifest.json"))
    for name in ["code_histogram", "code_sort_pairs", "code_topk", "grounded_observatory", "grounded_archive"]:
        info = manifest.get(name)
        if info:
            prompts[name] = (corpus_dir / Path(info["file"]).name).read_text()
    if prompt != "The capital of France is":
        prompts = {"custom": prompt}  # override for single-prompt smoke
    print(f"Capturing {len(prompts)} prompts...")
    s = DecisiveSmoke()
    for name, text in prompts.items():
        r = s.capture.remote(text, max_tokens, prompt_name=name)
        print(f"  {name}: n_gen={r.get('n_gen')} prompt_len={r.get('prompt_len')} "
              f"shapes={[v for k,v in r.items() if k.endswith('_shape')][:1]} "
              f"hash_match={len(set(v for k,v in r.items() if k.endswith('_hash')))==1}")
