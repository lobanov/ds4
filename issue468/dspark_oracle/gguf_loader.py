#!/usr/bin/env python3
"""GGUF -> F32 numpy loader for the DSpark numpy oracle.

Loads dspark.gguf (Phase-3 converter output) and dequantizes every tensor to
F32 numpy. Implements F32 / BF16 / Q8_0 / Q4_K dequant using the standard GGML
block layouts (Q4_K super-block of 256: F16 d + F16 dmin + 12 packed 6-bit
scales/mins + 128 packed 4-bit quants).

The Q4_K/Q8_0 dequant is the one place this oracle could diverge from the
converter; it is cross-validated against deepseek4-quantize --compare-tensor
(the authoritative byte-compare) for F32/BF16 tensors, and the Q4_K/Q8_0 path
is validated by the end-to-end token-agreement gate (dequant tolerance allowed).

Tensor orientation: GGUF ne is HF/torch shape reversed (ne[0] innermost). This
loader returns arrays in GGUF ne order (ne[0] fastest); matmuls in the oracle
use that convention consistently.
"""
import struct
import numpy as np


def _f16_to_f32(arr_u16):
    """IEEE 754 half (u16 bits) -> f32 (standard GGML f16, not bf16)."""
    u16 = arr_u16.astype(np.uint32)
    sign = (u16 >> 15) & 0x1
    exp = (u16 >> 10) & 0x1F
    man = u16 & 0x3FF
    out = np.zeros_like(u16, dtype=np.float32)
    # subnormal -> 0 (rare for scales; GGML treats as 0)
    normal = exp != 0
    e32 = (exp - 15 + 127).astype(np.int32)
    bits = (sign << 31) | ((e32.astype(np.uint32) & 0xFF) << 23) | (man << 13)
    out[normal] = bits[normal].view(np.float32)
    return out


def _bf16_to_f32(arr_u16):
    """BF16 (u16 bits) -> f32: BF16 is the top 16 bits of f32."""
    return (arr_u16.astype(np.uint32) << 16).view(np.float32).copy()


def _dequant_q8_0(raw, nelem):
    """Q8_0: blocks of 32. Each block: F16 scale (2B) + 32 int8 (32B) = 34B."""
    nb = nelem // 32
    out = np.empty(nelem, dtype=np.float32)
    arr = np.frombuffer(raw, dtype=np.uint8, count=nb * 34).reshape(nb, 34)
    scales = _f16_to_f32(arr[:, :2].copy().view(np.uint16).reshape(nb))
    qs = arr[:, 2:].view(np.int8).reshape(nb, 32).astype(np.float32)
    out = (qs * scales[:, None]).reshape(-1)
    return out


def _dequant_q4_k(raw, nelem):
    """Q4_K: super-blocks of 256. Faithful port of ggml-quants.c
    dequantize_row_q4_K + get_scale_min_k4. Block layout (144B): F16 d(2) +
    F16 dmin(2) + scales(12) + qs(128). 8 sub-blocks of 32; each qs 32-byte
    chunk yields low-nibbles (even sub-block) then high-nibbles (odd sub-block)."""
    assert nelem % 256 == 0, f"Q4_K nelem {nelem} not divisible by 256"
    nb = nelem // 256
    b = np.frombuffer(raw, dtype=np.uint8, count=nb * 144).reshape(nb, 144)
    d = _f16_to_f32(b[:, 0:2].copy().view(np.uint16).reshape(nb))     # [nb]
    dmin = _f16_to_f32(b[:, 2:4].copy().view(np.uint16).reshape(nb))  # [nb]
    sc = b[:, 4:16].astype(np.uint32)   # [nb, 12] scale bytes
    qs = b[:, 16:144]                    # [nb, 128] quant bytes

    # get_scale_min_k4(j, q, &d, &m): for j<4 d=q[j]&63 m=q[j+4]&63;
    # for j>=4 d=(q[j+4]&0xF)|((q[j-4]>>6)<<4) m=(q[j+4]>>4)|((q[j]>>6)<<4).
    # Precompute the 8 per-sub-block (scale, min) 6-bit codes, then apply d/dmin.
    scales = np.empty((nb, 8), dtype=np.float32)
    mins = np.empty((nb, 8), dtype=np.float32)
    for j in range(8):
        if j < 4:
            sd = sc[:, j] & 63
            sm = sc[:, j + 4] & 63
        else:
            sd = (sc[:, j + 4] & 0xF) | ((sc[:, j - 4] >> 6) << 4)
            sm = (sc[:, j + 4] >> 4) | ((sc[:, j] >> 6) << 4)
        scales[:, j] = d * sd
        mins[:, j] = dmin * sm

    # qs: 256 nibbles across 4 outer iters; iter t uses 32 bytes, low nibbles
    # -> sub-block 2t, high nibbles -> sub-block 2t+1.
    q = qs.astype(np.uint16)
    low = (q & 0xF).astype(np.float32)    # [nb, 128]
    high = (q >> 4).astype(np.float32)     # [nb, 128]
    nib = np.empty((nb, 256), dtype=np.float32)
    for t in range(4):
        nib[:, t * 64 : t * 64 + 32] = low[:, t * 32 : (t + 1) * 32]
        nib[:, t * 64 + 32 : t * 64 + 64] = high[:, t * 32 : (t + 1) * 32]
    # each sub-block spans 32 output elements; broadcast its scale/min.
    sb = np.repeat(np.arange(8, dtype=np.int64), 32)   # [256] sub-block id per elem
    out = nib * scales[:, sb] - mins[:, sb]            # [nb, 256]
    return out.reshape(-1)


def load_gguf(path):
    """Parse a GGUF and return {name: (np.float32 array in GGUF ne order, gguf_type_str)}."""
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        f.read(4)  # version
        n_t = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        def s():
            n = struct.unpack("<Q", f.read(8))[0]
            return f.read(n).decode()

        def rd_val():
            t = struct.unpack("<I", f.read(4))[0]
            if t == 8: return ("str", s())
            elif t == 0: return ("u8", f.read(1))           # UINT8
            elif t == 1: return ("i8", f.read(1))           # INT8
            elif t == 2: return ("u16", struct.unpack("<H", f.read(2))[0])
            elif t == 3: return ("i16", struct.unpack("<h", f.read(2))[0])
            elif t == 4: return ("u32", struct.unpack("<I", f.read(4))[0])  # UINT32
            elif t == 5: return ("i32", struct.unpack("<i", f.read(4))[0])
            elif t == 6: return ("f32", struct.unpack("<f", f.read(4))[0])
            elif t == 7: return ("bool", f.read(1))
            elif t == 10: return ("u64", struct.unpack("<Q", f.read(8))[0])
            elif t == 11: return ("i64", struct.unpack("<q", f.read(8))[0])
            elif t == 12: return ("f64", struct.unpack("<d", f.read(8))[0])
            elif t == 9:
                et = struct.unpack("<I", f.read(4))[0]
                nn = struct.unpack("<Q", f.read(8))[0]
                arr = []
                for _ in range(nn):
                    if et == 4: arr.append(struct.unpack("<I", f.read(4))[0])
                    elif et == 8: arr.append(s())
                    elif et == 10: arr.append(struct.unpack("<Q", f.read(8))[0])
                return ("arr", arr)
            else:
                raise ValueError(f"unhandled KV type {t}")

        kv = {}
        for _ in range(n_kv):
            k = s()
            kv[k] = rd_val()
        infos = {}
        meta_end = f.tell()
        for _ in range(n_t):
            nm = s()
            nd = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
            tt = struct.unpack("<I", f.read(4))[0]
            off = struct.unpack("<Q", f.read(8))[0]
            infos[nm] = (dims, tt, off)
        meta_end = f.tell()
        align = kv.get("general.alignment", ("u32", 32))[1] if isinstance(kv.get("general.alignment"), tuple) else 32
        data_off = (meta_end + align - 1) // align * align

        TYPE_NAME = {0:"f32",1:"f16",30:"bf16",8:"q8_0",12:"q4_k"}
        out = {}
        for nm, (dims, tt, off) in infos.items():
            nelem = 1
            for d in dims: nelem *= d
            f.seek(data_off + off)
            if tt == 0:       # F32
                raw = f.read(nelem * 4); arr = np.frombuffer(raw, dtype=np.float32).copy()
            elif tt == 30:    # BF16
                raw = f.read(nelem * 2)
                arr = _bf16_to_f32(np.frombuffer(raw, dtype=np.uint16).copy())
            elif tt == 1:     # F16
                raw = f.read(nelem * 2)
                arr = _f16_to_f32(np.frombuffer(raw, dtype=np.uint16).copy())
            elif tt == 8:     # Q8_0
                arr = _dequant_q8_0(f.read((nelem // 32) * 34), nelem)
            elif tt == 12:    # Q4_K
                arr = _dequant_q4_k(f.read((nelem // 256) * 144), nelem)
            else:
                raise ValueError(f"unhandled tensor type {tt} for {nm}")
            arr = arr.reshape(dims)
            out[nm] = (arr, TYPE_NAME.get(tt, str(tt)))
        return kv, out


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "../ds4/gguf/dspark.gguf"
    kv, tensors = load_gguf(path)
    print(f"KV: {[(k,v) for k,v in kv.items()]}")
    print(f"\nloaded {len(tensors)} tensors:")
    from collections import Counter
    c = Counter(t for _, t in tensors.values())
    print("  type counts:", dict(c))
    # sanity: a few known tensors
    for nm in ["mtp.0.main_norm.weight", "mtp.2.markov_head.markov_w1.weight",
               "mtp.0.attn_q_a.weight", "mtp.0.ffn_gate_exps.weight"]:
        if nm in tensors:
            a, t = tensors[nm]
            print(f"  {nm}: type={t} shape={a.shape} mean={a.mean():.4f} std={a.std():.4f}")
