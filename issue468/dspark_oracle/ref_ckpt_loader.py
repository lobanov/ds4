#!/usr/bin/env python3
"""Reference-checkpoint loader for the numpy DSpark oracle.

Loads the converted source checkpoint (`ref-ckpt/model0-mp1.safetensors`) and
maps its reference-model tensor names onto the older oracle names expected by
`issue468/dspark_oracle/forward.py`.
"""

from __future__ import annotations

import numpy as np


FP4_TABLE = np.array(
    [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ],
    dtype=np.float32,
)


KEY_MAP = {
    "embed.weight": "embed.weight",
    "head.weight": "head.weight",
    "mtp.0.main_proj.weight": "mtp.0.main_proj.weight",
    "mtp.0.main_norm.weight": "mtp.0.main_norm.weight",
    "mtp.{s}.hc_attn_fn": "mtp.{s}.hc_attn_fn.weight",
    "mtp.{s}.hc_attn_scale": "mtp.{s}.hc_attn_scale.weight",
    "mtp.{s}.hc_attn_base": "mtp.{s}.hc_attn_base.weight",
    "mtp.{s}.hc_ffn_fn": "mtp.{s}.hc_ffn_fn.weight",
    "mtp.{s}.hc_ffn_scale": "mtp.{s}.hc_ffn_scale.weight",
    "mtp.{s}.hc_ffn_base": "mtp.{s}.hc_ffn_base.weight",
    "mtp.{s}.attn_norm.weight": "mtp.{s}.attn_norm.weight",
    "mtp.{s}.ffn_norm.weight": "mtp.{s}.ffn_norm.weight",
    "mtp.{s}.attn.wq_a.weight": "mtp.{s}.attn_q_a.weight",
    "mtp.{s}.attn.q_norm.weight": "mtp.{s}.attn_q_a_norm.weight",
    "mtp.{s}.attn.wq_b.weight": "mtp.{s}.attn_q_b.weight",
    "mtp.{s}.attn.wkv.weight": "mtp.{s}.attn_kv.weight",
    "mtp.{s}.attn.kv_norm.weight": "mtp.{s}.attn_kv_a_norm.weight",
    "mtp.{s}.attn.attn_sink": "mtp.{s}.attn_sinks.weight",
    "mtp.{s}.attn.wo_a.weight": "mtp.{s}.attn_output_a.weight",
    "mtp.{s}.attn.wo_b.weight": "mtp.{s}.attn_output_b.weight",
    "mtp.{s}.ffn.gate.weight": "mtp.{s}.ffn_gate_inp.weight",
    "mtp.{s}.ffn.gate.bias": "mtp.{s}.exp_probs_b.bias",
    "mtp.{s}.ffn.shared_experts.w1.weight": "mtp.{s}.ffn_gate_shexp.weight",
    "mtp.{s}.ffn.shared_experts.w2.weight": "mtp.{s}.ffn_down_shexp.weight",
    "mtp.{s}.ffn.shared_experts.w3.weight": "mtp.{s}.ffn_up_shexp.weight",
    "mtp.2.norm.weight": "mtp.2.norm.weight",
    "mtp.2.hc_head_fn": "mtp.2.hc_head_fn.weight",
    "mtp.2.hc_head_scale": "mtp.2.hc_head_scale.weight",
    "mtp.2.hc_head_base": "mtp.2.hc_head_base.weight",
    "mtp.2.markov_head.markov_w1.weight": "mtp.2.markov_head.markov_w1.weight",
    "mtp.2.markov_head.markov_w2.weight": "mtp.2.markov_head.markov_w2.weight",
    "mtp.2.confidence_head.proj.weight": "mtp.2.confidence_head.proj.weight",
}


def _torch():
    import torch

    return torch


def _unpack_fp4_tensor(weight):
    torch = _torch()
    raw = weight.view(torch.uint8)
    low = (raw & 0x0F).long()
    high = ((raw >> 4) & 0x0F).long()
    table = torch.tensor(FP4_TABLE, dtype=torch.float32, device=weight.device)
    vals = torch.stack([table[low], table[high]], dim=-1)
    return vals.flatten(-2)


def _dequant_fp8_tensor(weight, scale):
    expanded = scale.float().repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    expanded = expanded[: weight.size(0), : weight.size(1)]
    return weight.float() * expanded


def _dequant_fp4_tensor(weight, scale):
    unpacked = _unpack_fp4_tensor(weight)
    expanded = scale.float().repeat_interleave(32, dim=1)
    expanded = expanded[:, : unpacked.size(1)]
    return unpacked.float() * expanded


def _tensor_to_f32(f, key: str):
    torch = _torch()
    tensor = f.get_tensor(key)
    if tensor.dtype == torch.float8_e4m3fn:
        return _dequant_fp8_tensor(tensor, f.get_tensor(key.replace(".weight", ".scale"))).cpu().numpy().astype(np.float32)
    if tensor.dtype == torch.float4_e2m1fn_x2:
        return _dequant_fp4_tensor(tensor, f.get_tensor(key.replace(".weight", ".scale"))).cpu().numpy().astype(np.float32)
    return tensor.float().cpu().numpy().astype(np.float32)


def load_ref_ckpt_dense(path: str):
    """Return `(kv, T, infos, data_off, embed_w, lm_head)` for oracle use.

    `T` matches the legacy oracle names and stores tuple-wrapped arrays so
    `forward.py` and `measure_b2_acceptance.py` style helpers can reuse them.
    """
    from safetensors import safe_open

    T = {}
    kv = {}
    infos = {}
    data_off = 0
    with safe_open(path, framework="pt", device="cpu") as f:
        for src_key, dst_key in KEY_MAP.items():
            if "{s}" in src_key:
                for stage in range(3):
                    src = src_key.format(s=stage)
                    dst = dst_key.format(s=stage)
                    T[dst] = (_tensor_to_f32(f, src),)
            else:
                T[dst_key] = (_tensor_to_f32(f, src_key),)
    embed_w = T["embed.weight"][0]
    lm_head = T["head.weight"][0]
    return kv, T, infos, data_off, embed_w, lm_head
