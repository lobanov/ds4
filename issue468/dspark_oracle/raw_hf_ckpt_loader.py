#!/usr/bin/env python3
"""Raw-HF checkpoint loader for the numpy DSpark oracle.

Loads the original DSpark Hugging Face MTP shard files directly from
``hf-dspark`` without going through ``ref-ckpt/model0-mp1.safetensors``.

The oracle still operates on float32 arrays, so mixed-precision source tensors
are dequantized on load using the same FP8/FP4 math as the local C quantizer.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gguf_loader import index_gguf, read_tensor


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HF_DSPARK = (ROOT / ".." / "ds4" / "hf-dspark").resolve()
TARGET_GGUF = str((ROOT / ".." / "ds4" / "gguf" / "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf").resolve())

FP4_TABLE = np.array(
    [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ],
    dtype=np.float32,
)


KEY_MAP = {
    "mtp.0.main_proj.weight": "mtp.0.main_proj.weight",
    "mtp.0.main_norm.weight": "mtp.0.main_norm.weight",
    "mtp.{s}.hc_attn_base": "mtp.{s}.hc_attn_base.weight",
    "mtp.{s}.hc_attn_fn": "mtp.{s}.hc_attn_fn.weight",
    "mtp.{s}.hc_attn_scale": "mtp.{s}.hc_attn_scale.weight",
    "mtp.{s}.hc_ffn_base": "mtp.{s}.hc_ffn_base.weight",
    "mtp.{s}.hc_ffn_fn": "mtp.{s}.hc_ffn_fn.weight",
    "mtp.{s}.hc_ffn_scale": "mtp.{s}.hc_ffn_scale.weight",
    "mtp.{s}.attn.attn_sink": "mtp.{s}.attn_sinks.weight",
    "mtp.{s}.attn.wq_a.weight": "mtp.{s}.attn_q_a.weight",
    "mtp.{s}.attn.q_norm.weight": "mtp.{s}.attn_q_a_norm.weight",
    "mtp.{s}.attn.wq_b.weight": "mtp.{s}.attn_q_b.weight",
    "mtp.{s}.attn.wkv.weight": "mtp.{s}.attn_kv.weight",
    "mtp.{s}.attn.kv_norm.weight": "mtp.{s}.attn_kv_a_norm.weight",
    "mtp.{s}.attn.wo_a.weight": "mtp.{s}.attn_output_a.weight",
    "mtp.{s}.attn.wo_b.weight": "mtp.{s}.attn_output_b.weight",
    "mtp.{s}.attn_norm.weight": "mtp.{s}.attn_norm.weight",
    "mtp.{s}.ffn_norm.weight": "mtp.{s}.ffn_norm.weight",
    "mtp.{s}.ffn.gate.weight": "mtp.{s}.ffn_gate_inp.weight",
    "mtp.{s}.ffn.gate.bias": "mtp.{s}.exp_probs_b.bias",
    "mtp.{s}.ffn.shared_experts.w1.weight": "mtp.{s}.ffn_gate_shexp.weight",
    "mtp.{s}.ffn.shared_experts.w2.weight": "mtp.{s}.ffn_down_shexp.weight",
    "mtp.{s}.ffn.shared_experts.w3.weight": "mtp.{s}.ffn_up_shexp.weight",
    "mtp.2.norm.weight": "mtp.2.norm.weight",
    "mtp.2.hc_head_base": "mtp.2.hc_head_base.weight",
    "mtp.2.hc_head_fn": "mtp.2.hc_head_fn.weight",
    "mtp.2.hc_head_scale": "mtp.2.hc_head_scale.weight",
    "mtp.2.markov_head.markov_w1.weight": "mtp.2.markov_head.markov_w1.weight",
    "mtp.2.markov_head.markov_w2.weight": "mtp.2.markov_head.markov_w2.weight",
    "mtp.2.confidence_head.proj.weight": "mtp.2.confidence_head.proj.weight",
}


@dataclass(frozen=True)
class TensorMeta:
    shard: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]


class RawHFTensorStore:
    def __init__(self, hf_dir: str):
        self.hf_dir = os.path.expanduser(hf_dir)
        self.weight_map = json.loads((Path(self.hf_dir) / "model.safetensors.index.json").read_text())["weight_map"]
        self._headers: dict[str, dict[str, dict[str, object]]] = {}
        self._data_bases: dict[str, int] = {}

    def _ensure_shard(self, shard: str) -> None:
        if shard in self._headers:
            return
        path = Path(self.hf_dir) / shard
        with path.open("rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            self._headers[shard] = json.loads(f.read(header_len))
        self._data_bases[shard] = 8 + header_len

    def meta(self, name: str) -> TensorMeta:
        shard = self.weight_map[name]
        self._ensure_shard(shard)
        info = self._headers[shard][name]
        return TensorMeta(
            shard=shard,
            dtype=str(info["dtype"]),
            shape=tuple(int(x) for x in info["shape"]),
            data_offsets=(int(info["data_offsets"][0]), int(info["data_offsets"][1])),
        )

    def raw_bytes(self, name: str) -> tuple[TensorMeta, bytes]:
        meta = self.meta(name)
        path = Path(self.hf_dir) / meta.shard
        start, end = meta.data_offsets
        with path.open("rb") as f:
            f.seek(self._data_bases[meta.shard] + start)
            data = f.read(end - start)
        return meta, data


def _e8m0_to_f32(data: np.ndarray) -> np.ndarray:
    bits = data.astype(np.uint32) << 23
    bits = np.where(data == 0, np.uint32(0x00400000), bits)
    return bits.view(np.float32)


def _e4m3fn_to_f32(data: np.ndarray) -> np.ndarray:
    x = data.astype(np.uint8)
    abs_x = x & np.uint8(0x7F)
    sign = (x & np.uint8(0x80)) != 0
    exp = ((x >> np.uint8(3)) & np.uint8(0x0F)).astype(np.int32)
    man = (x & np.uint8(0x07)).astype(np.float32)
    denorm = np.ldexp(man, -9)
    norm = np.ldexp(1.0 + man / 8.0, exp - 7)
    out = np.where(abs_x == 0, 0.0, np.where(abs_x == 0x7F, 0.0, np.where(exp == 0, denorm, norm)))
    out = out.astype(np.float32, copy=False)
    out[sign] *= -1.0
    return out


def _bf16_to_f32(data: bytes) -> np.ndarray:
    u16 = np.frombuffer(data, dtype="<u2").astype(np.uint32)
    return (u16 << 16).view(np.float32)


def _f16_to_f32(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<f2").astype(np.float32)


def _dequant_fp8_weight(weight: bytes, weight_shape: tuple[int, int], scale: bytes, scale_shape: tuple[int, int]) -> np.ndarray:
    out_dim, in_dim = weight_shape
    scale_rows, scale_cols = scale_shape
    if out_dim % 128 or in_dim % 128:
        raise RuntimeError(f"FP8 dims not divisible by 128: {weight_shape}")
    if scale_rows != out_dim // 128 or scale_cols != in_dim // 128:
        raise RuntimeError(f"FP8 scale shape mismatch: weight={weight_shape} scale={scale_shape}")
    w = _e4m3fn_to_f32(np.frombuffer(weight, dtype=np.uint8)).reshape(out_dim, in_dim)
    s = _e8m0_to_f32(np.frombuffer(scale, dtype=np.uint8)).reshape(scale_rows, scale_cols)
    expanded = np.repeat(np.repeat(s, 128, axis=0), 128, axis=1)
    return (w * expanded).astype(np.float32, copy=False)


def _dequant_fp4_weight(weight: bytes, weight_shape: tuple[int, int], scale: bytes, scale_shape: tuple[int, int]) -> np.ndarray:
    out_dim, packed_in = weight_shape
    in_dim = packed_in * 2
    if in_dim % 32:
        raise RuntimeError(f"FP4 in_dim not divisible by 32: {weight_shape}")
    n_blocks = in_dim // 32
    if scale_shape != (out_dim, n_blocks):
        raise RuntimeError(f"FP4 scale shape mismatch: weight={weight_shape} scale={scale_shape}")
    packed = np.frombuffer(weight, dtype=np.uint8).reshape(out_dim, n_blocks, 16)
    lo = FP4_TABLE[packed & np.uint8(0x0F)]
    hi = FP4_TABLE[(packed >> np.uint8(4)) & np.uint8(0x0F)]
    vals = np.empty((out_dim, n_blocks, 32), dtype=np.float32)
    vals[:, :, 0::2] = lo
    vals[:, :, 1::2] = hi
    scales = _e8m0_to_f32(np.frombuffer(scale, dtype=np.uint8)).reshape(out_dim, n_blocks)
    vals *= scales[:, :, None]
    return vals.reshape(out_dim, in_dim)


def tensor_to_f32(store: RawHFTensorStore, name: str) -> np.ndarray:
    meta, raw = store.raw_bytes(name)
    if meta.dtype == "F32":
        return np.frombuffer(raw, dtype="<f4").reshape(meta.shape).astype(np.float32, copy=False)
    if meta.dtype == "BF16":
        return _bf16_to_f32(raw).reshape(meta.shape)
    if meta.dtype == "F16":
        return _f16_to_f32(raw).reshape(meta.shape)
    if meta.dtype == "F8_E4M3":
        scale_name = name.replace(".weight", ".scale")
        scale_meta, scale_raw = store.raw_bytes(scale_name)
        return _dequant_fp8_weight(raw, meta.shape, scale_raw, scale_meta.shape)
    raise RuntimeError(f"unsupported direct dtype for {name}: {meta.dtype}")


def load_raw_hf_dense(hf_dir: str = str(DEFAULT_HF_DSPARK)):
    """Return `(kv, T, infos, data_off, embed_w, lm_head)` for oracle use."""
    store = RawHFTensorStore(hf_dir)
    T: dict[str, tuple[np.ndarray]] = {}
    kv = {}
    infos = {}
    data_off = 0
    for src_key, dst_key in KEY_MAP.items():
        if "{s}" in src_key:
            for stage in range(3):
                src = src_key.format(s=stage)
                dst = dst_key.format(s=stage)
                T[dst] = (tensor_to_f32(store, src),)
        else:
            T[dst_key] = (tensor_to_f32(store, src_key),)
    _, target_infos, target_data_off = index_gguf(TARGET_GGUF)
    embed_w = read_tensor(TARGET_GGUF, target_infos, target_data_off, "token_embd.weight").astype(np.float32)
    lm_head = read_tensor(TARGET_GGUF, target_infos, target_data_off, "output.weight").astype(np.float32)
    return kv, T, infos, data_off, embed_w, lm_head
