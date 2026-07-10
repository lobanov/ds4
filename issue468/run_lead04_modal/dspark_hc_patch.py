#!/usr/bin/env python3
"""DeepSeekV4 HC-residual capture hook (VALIDATED — v7 smoke: p1=0.74 GREEN).

LESSONS LEARNED:
1. V1 model is in the EngineCore subprocess — register hooks POST-LOAD via
   llm.apply_model (not from the driver; not via sitecustomize-at-startup).
2. mhc_post_tilelang MUST be called on CLONES of the layer output tensors —
   calling it on the originals corrupts the next layer's forward (degenerate
   repeating greedy). Always clone: mhc_post_fn(hs.clone(), residual.clone(), ...).
3. out[0] alone is [tok,4096] (reduced FFN hidden), NOT the HC residual —
   mhc_post(out[0], residual, post_mix, res_mix) produces the [tok,4,4096] HC
   residual (ds4 after_ffn_hc). Don't skip mhc_post.
4. enforce_eager=True is REQUIRED — cudagraphs (enforce_eager=False) break
   forward hooks (they don't fire during graph replay → zero captures).
5. The captured buffer is replicated across TP ranks (RowParallel reduce) —
   fetch rank 0 only, no gather needed.

See .pi/skills/vllm-on-modal/SKILL.md for the full methodology.
"""
from __future__ import annotations

# ---- pure capture logic (torch-free; unit-tested) ----
DSPARK_LAYERS = (40, 41, 42)
HC_DIM = 4 * 4096  # 16384 = hc_mult * hidden_size


def slot_for_layer(idx: int) -> int | None:
    """Global layer idx -> buffer slot if it's a capture layer (40/41/42), else None."""
    return DSPARK_LAYERS.index(idx) if idx in DSPARK_LAYERS else None


def capture_into_buffer(buffer, slot, hidden_states, residual, post_mix, res_mix,
                        mhc_post_fn) -> int:
    """Apply mhc_post -> flatten -> copy into buffer[:n, slot]. Returns n tokens.

    hidden_states: [n, hc_mult=4, hidden=4096]; residual/post_mix/res_mix: the layer's
    carried HC-mix state. mhc_post_fn(hs,residual,post_mix,res_mix) -> [n,4,4096] (the
    post-FFN HC residual == ds4's after_ffn_hc). Flattened to [n,16384] into the buffer.
    """
    post = mhc_post_fn(hidden_states, residual, post_mix, res_mix)  # [n, hc_mult, hidden]
    n = post.shape[0]
    buffer[:n, slot] = post.reshape(n, -1)  # [n, hc_mult*hidden = 16384]
    return n


# ---- torch wrappers (real deployment; EngineCore-side) ----
def _resolve_mhc_post():
    """Find vLLM's mhc_post_tilelang (backend-specific module). Returns fn or None."""
    import importlib
    for modpath, name in [
        ("vllm.models.deepseek_v4.nvidia.model", "mhc_post_tilelang"),
        ("vllm.models.deepseek_v4.model", "mhc_post_tilelang"),
        ("vllm.models.deepseek_v4", "mhc_post_tilelang"),
    ]:
        try:
            return getattr(importlib.import_module(modpath), name)
        except Exception:
            continue
    return None


def _resolve_deepseek_v4_model():
    """Find the DeepseekV4Model class (the inner model with .layers)."""
    from vllm.models.deepseek_v4 import DeepSeekV4ForCausalLM
    return DeepSeekV4ForCausalLM.model_cls  # DeepseekV4Model


def register_dspark_hooks(model, max_tokens: int = 8192, device: str = "cuda"):
    """Register forward hooks on model.layers[40,41,42] + init capture lists.

    Captures after_ffn_hc = mhc_post(hs, residual, post_mix, res_mix) on CLONES
    (clones prevent in-place interference). out[0] alone is the reduced FFN hidden
    [tok,4096], NOT the HC residual; mhc_post produces the [tok,4,4096] post-FFN HC.
    """
    import sys, torch
    mod = sys.modules.get(type(model).__module__)
    mhc_post_fn = getattr(mod, "mhc_post_tilelang", None) or _resolve_mhc_post()
    assert mhc_post_fn is not None, "could not resolve mhc_post_tilelang"
    model._dspark_captures = [[] for _ in DSPARK_LAYERS]

    def make_hook(slot):
        def hook(_mod, _inp, out):
            hs, residual, post_mix, res_mix = out
            post = mhc_post_fn(hs.clone(), residual.clone(),
                               post_mix.clone(), res_mix.clone())
            model._dspark_captures[slot].append(post.flatten(1).float().cpu())
        return hook

    model._dspark_handles = [
        model.layers[lyr].register_forward_hook(make_hook(slot))
        for slot, lyr in enumerate(DSPARK_LAYERS)
    ]
    from types import MethodType
    model.get_dspark_hc_buffer = MethodType(get_dspark_hc_buffer, model)
    model.reset_dspark_captures = lambda: setattr(model, '_dspark_captures', [[] for _ in DSPARK_LAYERS])
    return model


def get_dspark_hc_buffer(model):
    """Return the accumulated decode-step HC residuals [n_decode, 3, 16384] (rank-0, CPU).

    Filters the appended captures: decode steps (1 token each, shape[0]==1) are the
    generated positions; prefill (shape[0]>1) is excluded. Stacks across the 3 layers.
    """
    import torch
    per_layer = []
    for slot in range(len(DSPARK_LAYERS)):
        caps = getattr(model, "_dspark_captures", [[]])[slot]
        decode = [c for c in caps if c.shape[0] == 1]   # decode steps only
        if decode:
            per_layer.append(torch.cat(decode, dim=0))    # [n_decode, 16384]
        else:
            per_layer.append(torch.zeros((0, HC_DIM), dtype=torch.float32))
    return torch.stack(per_layer, dim=1)  # [n_decode, 3, 16384]


# DEPRECATED (use post-load apply_model instead): def apply_patch(max_tokens: int = 8192):
    """Monkeypatch DeepseekV4Model.__init__ to register the hooks at instantiation.
    Call this from a module imported inside the EngineCore (sitecustomize/PYTHONPATH)."""
    DSV4Model = _resolve_deepseek_v4_model()
    _orig_init = DSV4Model.__init__

    def _new_init(self, *a, **kw):
        _orig_init(self, *a, **kw)
        try:
            dev = next(self.parameters()).device if list(self.parameters()) else "cuda"
            register_dspark_hooks(self, mhc_post_fn=_resolve_mhc_post(),
                                  max_tokens=max_tokens, device=str(dev))
            print(f"[dspark_hc_patch] hooks registered on layers {DSPARK_LAYERS}", flush=True)
        except Exception as e:
            print(f"[dspark_hc_patch] hook registration FAILED: {e!r}", flush=True)

    DSV4Model.__init__ = _new_init
    DSV4Model.get_dspark_hc_buffer = lambda self: get_dspark_hc_buffer(self)
    print("[dspark_hc_patch] apply_patch done", flush=True)


# ---- auto-apply on import (if RUN_DSPARK_HC_PATCH env set, for EngineCore import) ----
import os
if os.environ.get("RUN_DSPARK_HC_PATCH") == "1":
    apply_patch()
