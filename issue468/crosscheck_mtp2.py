#!/usr/bin/env python3
"""Cross-check the full DSpark dspark.gguf (3 drafter layers) against the HF source.

Phase 3 conversion validation (extends the iter-1 mtp.2-only check to all 3
layers). The fp4/fp8 dequant math is IDENTICAL to the target's (already
validated by the shipped target GGUFs); the NEW risk for DSpark is name-mapping
+ tensor orientation + the per-tensor read/write path on real DSpark data.

This script does two checks:

  (1) LOSSLESS BYTE-MATCH: for every tensor whose HF dtype EQUALS its GGUF type
      (both F32 or both BF16), the round-trip through the converter is bit-exact,
      so the GGUF bytes must equal the HF source bytes. This validates name
      mapping, the read path, AND orientation on real data, for 32 tensors
      spanning all 3 layers (hc_attn/hc_ffn base/fn/scale, attn_sinks,
      exp_probs_b, hc_head_* mtp.2, main_norm mtp.0, norm mtp.2, markov_w1/w2,
      confidence_head.proj).

  (2) BF16->F32 UPCAST VALUE-MATCH: for the norm/gate tensors that are BF16 in
      HF but upcast to F32 in the GGUF (attn_norm/ffn_norm/q_norm/kv_norm/gate),
      the upcast is value-lossless (BF16 is a subset of F32 precision), so the
      float32 values must be numerically identical element-for-element.

The FP8_E4M3 -> Q8_0 dequant+requant path (attn projections, shared experts,
main_proj) is NOT byte-exact by construction and is validated separately by the
drafter-forward regression gate in Phase 4 (token agreement vs model.py).
"""
import struct, json, os, sys

HF_DIR = os.environ.get("DS4_DSPARK_HF", "../ds4/hf-dspark")
GGUF = os.environ.get("DS4_DSPARK_GGUF", "../ds4/gguf/dspark.gguf")
SHARDS = [
    "model-00046-of-00048.safetensors",  # mtp.0
    "model-00047-of-00048.safetensors",  # mtp.1
    "model-00048-of-00048.safetensors",  # mtp.2
]
BF16, F32 = 30, 0
ELEMSZ = {F32: 4, BF16: 2}  # bytes per element

# --- converter name mapping (gguf leaf after "mtp.N." -> HF leaf) ---
LAYER_MAP = {
    "hc_attn_base.weight": "hc_attn_base", "hc_attn_fn.weight": "hc_attn_fn",
    "hc_attn_scale.weight": "hc_attn_scale", "hc_ffn_base.weight": "hc_ffn_base",
    "hc_ffn_fn.weight": "hc_ffn_fn", "hc_ffn_scale.weight": "hc_ffn_scale",
    "attn_sinks.weight": "attn.attn_sink", "attn_q_a.weight": "attn.wq_a.weight",
    "attn_q_b.weight": "attn.wq_b.weight", "attn_q_a_norm.weight": "attn.q_norm.weight",
    "attn_kv.weight": "attn.wkv.weight", "attn_kv_a_norm.weight": "attn.kv_norm.weight",
    "attn_output_a.weight": "attn.wo_a.weight", "attn_output_b.weight": "attn.wo_b.weight",
    "attn_norm.weight": "attn_norm.weight", "ffn_norm.weight": "ffn_norm.weight",
    "ffn_gate_shexp.weight": "ffn.shared_experts.w1.weight",
    "ffn_up_shexp.weight": "ffn.shared_experts.w3.weight",
    "ffn_down_shexp.weight": "ffn.shared_experts.w2.weight",
    "ffn_gate_inp.weight": "ffn.gate.weight", "exp_probs_b.bias": "ffn.gate.bias",
}
MTP_UNIQUE = {
    "main_proj.weight": "main_proj.weight", "main_norm.weight": "main_norm.weight",
    "norm.weight": "norm.weight", "hc_head_base.weight": "hc_head_base",
    "hc_head_fn.weight": "hc_head_fn", "hc_head_scale.weight": "hc_head_scale",
    "markov_head.markov_w1.weight": "markov_head.markov_w1.weight",
    "markov_head.markov_w2.weight": "markov_head.markov_w2.weight",
    "confidence_head.proj.weight": "confidence_head.proj.weight",
}


def hf_leaf(leaf):
    return LAYER_MAP.get(leaf) or MTP_UNIQUE.get(leaf)


# ---- safetensors index (which shard holds each tensor) ----
def load_index():
    with open(os.path.join(HF_DIR, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


# ---- per-shard header cache (name -> (dtype, shape, (lo, hi))) ----
_shard_hdr = {}
_shard_base = {}


def hf_tensor_meta(name, weight_map):
    shard = weight_map[name]
    if shard not in _shard_hdr:
        path = os.path.join(HF_DIR, shard)
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        base = 8 + n
        _shard_hdr[shard] = hdr
        _shard_base[shard] = base
    hdr = _shard_hdr[shard]
    info = hdr[name]
    return shard, info["dtype"], info["shape"], info["data_offsets"]


def hf_raw(name, weight_map):
    shard, dtype, shape, (lo, hi) = hf_tensor_meta(name, weight_map)
    path = os.path.join(HF_DIR, shard)
    with open(path, "rb") as f:
        f.seek(_shard_base[shard] + lo)
        return dtype, shape, f.read(hi - lo)


# ---- GGUF reader ----
def gguf_index(path):
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
            if t == 8:
                return s()
            elif t in (0, 1, 7):
                return f.read(1)
            elif t in (2, 3, 4, 5, 6):
                return struct.unpack("<I", f.read(4))
            elif t in (10, 11, 12):
                return struct.unpack("<Q", f.read(8))
            elif t == 9:
                et = struct.unpack("<I", f.read(4))[0]
                nn = struct.unpack("<Q", f.read(8))[0]
                for _ in range(nn):
                    if et == 8:
                        s()
                    elif et in (0, 1, 7):
                        f.read(1)
                    elif et in (2, 3, 4, 5, 6):
                        f.read(4)
                    elif et in (10, 11, 12):
                        f.read(8)

        for _ in range(n_kv):
            s()
            rd_val()
        infos = {}
        for _ in range(n_t):
            nm = s()
            nd = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
            tt = struct.unpack("<I", f.read(4))[0]
            off = struct.unpack("<Q", f.read(8))[0]
            infos[nm] = (dims, tt, off)
        # tensor data is 32-byte aligned after the FULL metadata section
        # (KV pairs + all tensor infos), not just after the KV section.
        meta_end = f.tell()
        data_off = ((meta_end + 31) // 32) * 32
        for nm in infos:
            dims, tt, off = infos[nm]
            infos[nm] = (dims, tt, data_off + off)
        return infos


def gguf_raw(path, name, infos):
    dims, tt, off = infos[name]
    n = 1
    for d in dims:
        n *= d
    with open(path, "rb") as f:
        f.seek(off)
        return dims, tt, f.read(n * ELEMSZ[tt])


def bf16_to_f32_bytes(b):
    """Convert a BF16 byte buffer to the equivalent F32 byte buffer (value-lossless)."""
    import array
    u16 = array.array("H")
    u16.frombytes(b)
    u32 = array.array("I", (x << 16 for x in u16))  # BF16 = top 16 bits of F32
    return u32.tobytes()


def main():
    print(f"cross-check: {GGUF}\n          vs: {HF_DIR}/{{46,47,48}}.safetensors")
    weight_map = load_index()
    ginfos = gguf_index(GGUF)
    n_byte_ok = n_val_ok = n_byte_bad = n_val_bad = 0
    failures = []
    for nm in sorted(ginfos):
        leaf = nm.split(".", 2)[2]  # mtp.N.<leaf>
        if leaf.endswith("_exps.weight"):
            continue  # experts: FP4->Q4_K, validated in Phase 4
        hleaf = hf_leaf(leaf)
        if hleaf is None:
            continue
        L = nm.split(".")[1]
        hf_name = f"mtp.{L}.{hleaf}"
        dims, gtt, _ = ginfos[nm]
        # Only F32/BF16 GGUF tensors are byte/value-checkable here; FP8->Q8_0
        # and FP4->Q4_K dequant paths are validated by the Phase-4 forward gate.
        if gtt not in (F32, BF16):
            continue
        try:
            hdt, hshape, hraw = hf_raw(hf_name, weight_map)
        except KeyError:
            failures.append(f"  HF MISSING: {hf_name}")
            n_byte_bad += 1
            continue
        _, _, graw = gguf_raw(GGUF, nm, ginfos)
        hdt_int = {"F32": F32, "BF16": BF16}.get(hdt)
        # Case 1: lossless (same F32/BF16 dtype on both sides) -> exact byte match
        if gtt in (F32, BF16) and hdt_int == gtt:
            if hraw == graw:
                n_byte_ok += 1
            else:
                diff = sum(1 for a, b in zip(hraw, graw) if a != b)
                n_byte_bad += 1
                failures.append(f"  BYTE MISMATCH: {nm} ({hdt}): {diff}/{len(hraw)} bytes differ")
        # Case 2: BF16(HF) -> F32(GGUF) upcast -> value match (lossless precision upcast)
        elif gtt == F32 and hdt == "BF16":
            want = bf16_to_f32_bytes(hraw)
            if want == graw:
                n_val_ok += 1
            else:
                diff = sum(1 for a, b in zip(want, graw) if a != b)
                n_val_bad += 1
                failures.append(f"  VALUE MISMATCH: {nm} (BF16->F32): {diff}/{len(want)} bytes differ")
        # else: FP8->Q8_0 / FP4->Q4_K dequant path; validated by the Phase-4 forward gate
    print()
    print(f"lossless byte-exact:   {n_byte_ok} OK, {n_byte_bad} bad")
    print(f"BF16->F32 value-exact: {n_val_ok} OK, {n_val_bad} bad")
    if failures:
        print("FAILURES:")
        for f in failures:
            print(f)
        print("\nCROSSCHECK FAILED")
        sys.exit(1)
    print("\nCROSSCHECK PASSED")
    print(f"  - {n_byte_ok} tensors byte-exact vs HF (name-mapping + read + orientation validated)")
    print(f"  - {n_val_ok} BF16->F32 upcasts value-exact (lossless precision upcast)")


if __name__ == "__main__":
    main()
