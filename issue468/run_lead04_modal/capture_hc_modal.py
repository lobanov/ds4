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
MAX_MODEL_LEN = 2048   # corpus max prompt ~1236 tokens + template + 128 gen
TOP_K_LOGPROBS = 128   # top-128 logits+weights per decode step

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=False)
output_vol = modal.Volume.from_name("lead04-captures", create_if_missing=True)

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .uv_pip_install("vllm>=0.18.0", "transformers>=5.6,<6", "huggingface-hub>=1.5.0,<2.0",
                    "numpy", "safetensors", "ninja")
    .add_local_file("dspark_hc_patch.py", "/opt/dspark_hc_patch.py", copy=True)
    .add_local_file("convert_vllm_to_oracle.py", "/opt/convert_vllm_to_oracle.py", copy=True)
    .env({
        "PYTHONPATH": "/opt",
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",   # allow cloudpickle for apply_model funcs
        "TRITON_CACHE_DIR": "/output/triton_cache",
        "TORCHINDUCTOR_CACHE_DIR": "/output/inductor_cache",
        # Cache TileLang (28 custom mhc kernels ~5min) + DeepGEMM (FP4 expert kernels)
        # on the PERSISTENT volume so they compile once, not every run.
        "TILELANG_CACHE_DIR": "/output/tilelang_cache",
        "DG_JIT_CACHE_DIR": "/output/deepgemm_cache",
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


@app.cls(image=vllm_image, gpu=GPU_CONFIG, cpu=2, timeout=5 * 60 * MINUTES,
         volumes={"/root/.cache/huggingface": hf_cache_vol, "/output": output_vol},
         secrets=[modal.Secret.from_name("huggingface-token")],
         scaledown_window=15 * MINUTES)
class DecisiveSmoke:
    @modal.enter()
    def start(self):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=MODEL_ID, trust_remote_code=True, kv_cache_dtype="fp8",
                       enforce_eager=True, max_model_len=MAX_MODEL_LEN, tensor_parallel_size=2,
                       max_num_seqs=1, enable_prefix_caching=False,
                       max_logprobs=TOP_K_LOGPROBS)
        self.SamplingParams = SamplingParams
        print(f"[smoke] loaded {MODEL_ID} on {GPU_CONFIG}", flush=True)
        # register the hooks on the loaded model (inside the EngineCore)
        reg = self.llm.apply_model(_register_on_model)
        print(f"[smoke] hook registration: {reg}", flush=True)
        # Persist the TileLang/DeepGEMM/Triton JIT caches so the NEXT run skips
        # the ~5-9min recompilation (cold cache on first run, warm thereafter).
        output_vol.commit()
        print("[smoke] JIT caches committed to volume", flush=True)

    def _capture_impl(self, prompt: str, max_tokens: int = 14, prompt_name: str = "smoke",
                top_k: int = TOP_K_LOGPROBS, skip_if_exists: bool = False,
                prefix: str = "pilot") -> dict:
        """Capture one prompt: HC hiddens + greedy tokens + top-k logits+weights.

        Saves /output/{prefix}_{prompt_name}.npz with:
          layer{40,41,42} [n_dec,4,4096] f32, greedy_tokens [n_gen] i64,
          topk_ids [n_lp,top_k] i32 (-1 pad), topk_logprobs [n_lp,top_k] f32,
          topk_ranks [n_lp,top_k] i32, prompt_tokens i32, n_gen i32, layer_ids.
        """
        import numpy as np
        out_path = f"/output/{prefix}_{prompt_name}.npz"
        if skip_if_exists and Path(out_path).exists():
            return {"prompt_name": prompt_name, "status": "skipped", "saved": out_path}
        self.llm.apply_model(_reset_captures)  # clear captures from previous prompts
        sp = self.SamplingParams(temperature=0, max_tokens=max_tokens, logprobs=top_k)
        # Manually construct the DS4 chat format (matching ds4.c:22282 encode_chat_prompt
        # + ds4_cli.c:1402 default system): BOS + "You are a helpful assistant" +
        # <｜User｜> + prompt + <｜Assistant｜> + </think> (non-thinking mode).
        ds4_prompt = ("<｜begin▁of▁sentence｜>You are a helpful assistant"
                      "<｜User｜>" + prompt + "<｜Assistant｜></think>")
        res = self.llm.generate([ds4_prompt], sp)[0]
        o = res.outputs[0]
        greedy = list(o.token_ids)
        prompt_len = len(res.prompt_token_ids)
        # --- parse top-k logprobs (per generated-token distribution) ---
        lp_raw = o.logprobs or []   # List[Dict[token_id, Logprob]], len == n_gen
        topk_ids, topk_logprobs, topk_ranks = [], [], []
        for pos_d in lp_raw:
            items = sorted(pos_d.items(), key=lambda kv: kv[1].logprob, reverse=True)[:top_k]
            ids = [tid for tid, _ in items]; lps = [lp.logprob for _, lp in items]
            rks = [getattr(lp, "rank", i) for i, (_, lp) in enumerate(items)]
            pad = top_k - len(ids)
            topk_ids.append(ids + [-1]*pad)
            topk_logprobs.append(lps + [0.0]*pad)
            topk_ranks.append(rks + [-1]*pad)
        bufs = self.llm.apply_model(_fetch_buffer)   # list[rank] of [n,3,16384] or None
        out = {"prompt_name": prompt_name, "prompt_len": prompt_len,
               "n_gen": len(greedy), "n_logprobs": len(lp_raw),
               "greedy_head": greedy[:8], "n_ranks": len(bufs)}
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
            n_dec = arr0.shape[0]
            layers = {f"layer{L}": arr0[:, i, :].reshape(n_dec, 4, 4096).astype(np.float32)
                      for i, L in enumerate([40, 41, 42])}
            save_kwargs = dict(
                **layers, layer_ids=np.array([40, 41, 42], dtype=np.int32),
                n_gen=np.int32(len(greedy)), n_logprobs=np.int32(len(lp_raw)),
                prompt_tokens=np.int32(prompt_len),
                greedy_tokens=np.array(greedy, dtype=np.int64))
            if topk_ids:
                save_kwargs.update(
                    topk_ids=np.array(topk_ids, dtype=np.int32),
                    topk_logprobs=np.array(topk_logprobs, dtype=np.float32),
                    topk_ranks=np.array(topk_ranks, dtype=np.int32))
            np.savez(out_path, **save_kwargs)
            output_vol.commit()
            out["saved"] = out_path
        return out

    @modal.method()
    def capture(self, *args, **kwargs):
        """Remote-callable wrapper around _capture_impl (plain method, safe to call
        both via .remote() from the entrypoint AND internally from capture_corpus)."""
        return self._capture_impl(*args, **kwargs)

    @modal.method()
    def capture_corpus(self, prompts: dict, max_tokens: int = 128,
                       top_k: int = TOP_K_LOGPROBS, prefix: str = "phaseB") -> dict:
        """Batch-capture a corpus in one warm session. Checkpoints per-prompt
        (skip-if-exists for crash recovery). Returns a per-prompt summary dict."""
        results = {}
        for name, text in prompts.items():
            try:
                r = self._capture_impl(text, max_tokens=max_tokens, prompt_name=name,
                                 top_k=top_k, skip_if_exists=True, prefix=prefix)
            except Exception as e:   # don't let one prompt kill the batch
                import traceback
                tb = traceback.format_exc()[:800]
                print(f"  {name} ERROR: {e}\n{tb}", flush=True)
                r = {"prompt_name": name, "status": "ERROR", "error": str(e)[:200],
                      "traceback": tb}
                output_vol.commit()  # persist what we have so far
            results[name] = r
            print(f"  {name}: n_gen={r.get('n_gen')} lp={r.get('n_logprobs')} "
                  f"shapes={[v for k,v in r.items() if k.endswith('_shape')][:1]} "
                  f"status={r.get('status','ok')}", flush=True)
        return results

    @modal.exit()
    def stop(self):
        del self.llm


def _load_corpus(limit: int = 0) -> dict:
    """Load Lead 03's 300-prompt corpus (stage2_corpus + lead3_corpus) as {name: text}.
    Names match the Q2 prompt_ids (e.g. codealpaca_0000)."""
    from pathlib import Path
    prompts = {}
    root = Path(__file__).resolve().parent.parent
    for sub in ["stage2_corpus", "lead3_corpus"]:
        d = root / "prompts" / sub
        for f in sorted(d.glob("*.txt")):
            prompts[f.stem] = f.read_text()
    if limit:
        prompts = dict(list(prompts.items())[:limit])
    return prompts


@app.local_entrypoint()
def main(mode: str = "smoke", max_tokens: int = 14, limit: int = 0):
    """mode=smoke: 5 exactness prompts (fidelity check). mode=corpus: Lead 03's 300."""
    from pathlib import Path
    s = DecisiveSmoke()
    if mode == "corpus":
        prompts = _load_corpus(limit=limit)
        print(f"[corpus] {len(prompts)} prompts, {max_tokens} tokens each...")
        results = s.capture_corpus.remote(prompts, max_tokens=max_tokens)
        ok = sum(1 for r in results.values() if r.get("status") not in ("ERROR",))
        skipped = sum(1 for r in results.values() if r.get("status") == "skipped")
        errs = [n for n, r in results.items() if r.get("status") == "ERROR"]
        print(f"[corpus] done: {ok} ok ({skipped} skipped), {len(errs)} errors")
        if errs:
            print(f"  errors: {errs[:10]}")
    else:  # smoke: 5 exactness prompts
        prompts = {}
        corpus_dir = Path.cwd() / ".." / "prompts" / "exactness_small_corpus"
        manifest = json.load(open(corpus_dir / "manifest.json"))
        for name in ["code_histogram", "code_sort_pairs", "code_topk",
                     "grounded_observatory", "grounded_archive"]:
            info = manifest.get(name)
            if info:
                prompts[name] = (corpus_dir / Path(info["file"]).name).read_text()
        print(f"[smoke] {len(prompts)} prompts, {max_tokens} tokens, logprobs={TOP_K_LOGPROBS}...")
        for name, text in prompts.items():
            r = s.capture.remote(text, max_tokens, prompt_name=name, prefix="smoke")
            hashes = [v for k, v in r.items() if k.endswith("_hash")]
            print(f"  {name}: n_gen={r.get('n_gen')} lp={r.get('n_logprobs')} "
                  f"prompt_len={r.get('prompt_len')} "
                  f"shapes={[v for k,v in r.items() if k.endswith('_shape')][:1]} "
                  f"hash_match={len(set(hashes))==1}")
