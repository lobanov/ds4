#!/usr/bin/env python3
"""Build a DSpark drafter template GGUF (metadata only, zero data).

The template defines the converter's OUTPUT tensor inventory (names, logical
shapes, target quant types) + KV metadata. deepseek4-quantize regenerates the
bytes from the HF safetensors (shards 46-48 of deepseek-ai/DeepSeek-V4-Flash-
DSpark). See issue468/10_converter_reuse.md.

Naming convention: ds4-renamed (mtp.N.attn_q_a.weight, mtp.N.ffn_gate_exps.weight)
so the Phase-4 loader is structurally parallel to mtp_weights_bind.

Quant recipe (mirrors the existing MTP GGUF):
  F32   - norms, hc_*, attn_sinks, exp_probs_b, gate_inp
  Q8_0  - attn projections (wq_a/b, wkv, wo_a/b), main_proj, shared experts
  Q4_K  - routed experts (gate/up/down), packed 256 experts -> 1 tensor
  BF16  - markov_head, confidence_head, main_norm, output norm (accuracy-critical)
"""
import struct, sys

EMIT_LAYERS = (0, 1, 2)

# GGUF tensor types — NOTE: values follow this repo's ds4q_type enum (quants.h),
# which DIVERGES from standard GGUF for BF16 (30 here, not 27). The converter
# casts template type ids directly to ds4q_type, so they must match.
F32, F16, BF16, Q8_0, Q4_K, I8 = 0, 1, 30, 8, 12, 24

# ---- DSpark config (from inference/config.json) ----
N_EXPERTS = 256
DS4_N_EMBD = 4096
DS4_N_HC = 4
DS4_N_HEAD = 64
DS4_N_HEAD_DIM = 512
DS4_N_LORA_Q = 1024
DS4_N_OUT_GROUP = 8
DS4_N_LORA_O = 1024
DS4_N_FF_EXP = 2048
DS4_N_INDEXER_TOP_K = 0  # DSpark attn has no indexer
DSPARK_BLOCK_SIZE = 5
DSPARK_NOISE_TOKEN = 128799
DSPARK_TARGET_LAYERS = [40, 41, 42]
DSPARK_MARKOV_RANK = 256
VOCAB = 129280
HC_DIM = DS4_N_EMBD * DS4_N_HC              # 16384
HC_MIX_DIM = 2 * DS4_N_HC + DS4_N_HC * DS4_N_HC  # 24
Q_DIM = DS4_N_HEAD * DS4_N_HEAD_DIM         # 32768 (MLA q_b expand target)
OUT_LOW = DS4_N_OUT_GROUP * DS4_N_LORA_O    # 8192

def tensors():
    """Yield (gguf_name, gguf_dims, gguf_type). Shapes are specified in HF/torch
    order (shape[0]=outer) and reversed to GGUF ne[0]=innermost via gguf_dims().
    This avoids the manual orientation errors that bit the first attempt."""
    EMBD, FF, NX = DS4_N_EMBD, DS4_N_FF_EXP, N_EXPERTS
    LQ, QD, OL = DS4_N_LORA_Q, Q_DIM, OUT_LOW
    HD, HCD, HCM = DS4_N_HEAD_DIM, HC_DIM, HC_MIX_DIM
    VH = DS4_N_HEAD
    MR = DSPARK_MARKOV_RANK
    def g(hf_shape):  # HF/torch shape -> GGUF ne (reversed)
        return list(reversed(hf_shape))
    t = []
    def add(name, hf_shape, typ): t.append((name, g(hf_shape), typ))
    for L in EMIT_LAYERS:
        P = f"mtp.{L}"
        # --- block-internal (mini-DeepSeek-V4 block; same each layer) ---
        # NOTE: hc_head_* is the drafter's OUTPUT-stage HC head and exists on
        # mtp.2 ONLY (verified against HF inventory); it is emitted with the
        # other mtp.2 output-stage tensors below, not here.
        add(f"{P}.hc_attn_base.weight",  [HCM],                 F32)
        add(f"{P}.hc_attn_fn.weight",    [HCM, HCD],            F16)  # F16: ds4 matmul_f16 kernel reads these as F16 (matches target)
        add(f"{P}.hc_attn_scale.weight", [3],                   F32)
        add(f"{P}.hc_ffn_base.weight",   [HCM],                 F32)
        add(f"{P}.hc_ffn_fn.weight",     [HCM, HCD],            F16)  # F16: ds4 matmul_f16 kernel reads these as F16 (matches target)
        add(f"{P}.hc_ffn_scale.weight",  [3],                   F32)
        add(f"{P}.attn_sinks.weight",    [VH],                  F32)
        add(f"{P}.attn_q_a.weight",      [LQ, EMBD],            Q8_0)
        add(f"{P}.attn_q_b.weight",      [QD, LQ],              Q8_0)
        add(f"{P}.attn_q_a_norm.weight", [LQ],                  F32)
        add(f"{P}.attn_kv.weight",       [HD, EMBD],            Q8_0)
        add(f"{P}.attn_kv_a_norm.weight",[HD],                  F32)
        add(f"{P}.attn_output_a.weight", [OL, HD * (VH // DS4_N_OUT_GROUP)], Q8_0)
        add(f"{P}.attn_output_b.weight", [EMBD, OL],            Q8_0)
        add(f"{P}.attn_norm.weight",     [EMBD],                F32)
        add(f"{P}.ffn_norm.weight",      [EMBD],                F32)
        add(f"{P}.ffn_gate_shexp.weight",[FF, EMBD],            Q8_0)
        add(f"{P}.ffn_up_shexp.weight",  [FF, EMBD],            Q8_0)
        add(f"{P}.ffn_down_shexp.weight",[EMBD, FF],            Q8_0)
        # routed experts: per-expert HF shape w1/w3=[FF, EMBD packed], w2=[EMBD, FF packed];
        # fused GGUF tensor is 3D ne=[in, out, n_experts] (reversed from [n_experts, out, in]).
        add(f"{P}.ffn_gate_exps.weight", [NX, FF, EMBD],        Q4_K)
        add(f"{P}.ffn_up_exps.weight",   [NX, FF, EMBD],        Q4_K)
        add(f"{P}.ffn_down_exps.weight", [NX, EMBD, FF],        Q4_K)
        add(f"{P}.ffn_gate_inp.weight",  [NX, EMBD],            F16)  # F16: ffn_batch gate matmul uses ds4_gpu_matmul_f16_tensor (matches target)
        add(f"{P}.exp_probs_b.bias",     [NX],                  F32)
        # --- DSpark input/output stage ---
        if L == 0:
            add(f"{P}.main_proj.weight", [EMBD, EMBD * len(DSPARK_TARGET_LAYERS)], Q8_0)
            add(f"{P}.main_norm.weight", [EMBD], F32)  # F32: rmsnorm Metal kernel is F32-only (matches target norms)
        if L == 2:
            # hc_head_* is the output-stage HC head (mtp.2 only; absent on
            # mtp.0/1 in HF). F32 like the target's output_hc_* tensors.
            add(f"{P}.hc_head_base.weight",          [DS4_N_HC],         F32)
            add(f"{P}.hc_head_fn.weight",            [DS4_N_HC, HCD],    F16)  # F16: ds4 batched output head uses matmul_f16 (F32 path is single-token only)
            add(f"{P}.hc_head_scale.weight",         [1],                F32)
            add(f"{P}.norm.weight",                  [EMBD],             F32)  # F32: rmsnorm Metal kernel is F32-only
            add(f"{P}.markov_head.markov_w1.weight", [VOCAB, MR],        BF16)
            add(f"{P}.markov_head.markov_w2.weight", [VOCAB, MR],        BF16)
            add(f"{P}.confidence_head.proj.weight",  [EMBD + MR],     BF16)  # HF [1,4352]; converter strips trailing 1 -> 1D
    return t

# ---- minimal GGUF v3 writer (header + KV + tensor infos, zero data) ----
def w_str(f, s):
    b = s.encode("utf-8")
    f.write(struct.pack("<Q", len(b))); f.write(b)
def w_kv(f, key, vtype, write_val):
    w_str(f, key); f.write(struct.pack("<I", vtype)); write_val(f)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="issue468/baseline/dspark_template.gguf")
    ap.add_argument("--layers", default="0,1,2", help="comma list of drafter layers to emit")
    args = ap.parse_args()
    global EMIT_LAYERS
    EMIT_LAYERS = tuple(int(x) for x in args.layers.split(","))
    out = args.out
    tens = tensors()
    with open(out, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))           # version
        f.write(struct.pack("<Q", len(tens)))   # n_tensors
        # KV
        kv = []
        def kv_str(k, v):  kv.append((k, 8, lambda f: w_str(f, v)))
        def kv_u32(k, v):  kv.append((k, 4, lambda f: f.write(struct.pack("<I", v))))
        def kv_arr(k, vt, vs):
            def w(f):
                f.write(struct.pack("<I", vt))
                f.write(struct.pack("<Q", len(vs)))
                for x in vs:
                    if vt == 4: f.write(struct.pack("<I", x))
                    elif vt == 8: w_str(f, x)
            kv.append((k, 9, w))
        kv_str("general.architecture", "deepseek4")
        kv_str("general.name", "DeepSeek-V4-Flash-DSpark-drafter")
        kv_u32("deepseek4.expert_count", N_EXPERTS)
        kv_u32("deepseek4.mtp_layer_count", 3)
        kv_u32("deepseek4.nextn_predict_layers", 1)
        kv_u32("dspark.block_size", DSPARK_BLOCK_SIZE)
        kv_u32("dspark.noise_token_id", DSPARK_NOISE_TOKEN)
        kv_u32("dspark.markov_rank", DSPARK_MARKOV_RANK)
        kv_arr("dspark.target_layer_ids", 4, DSPARK_TARGET_LAYERS)
        f.write(struct.pack("<Q", len(kv)))     # n_kv
        for k, vt, w in kv:
            w_kv(f, k, vt, w)
        # tensor infos: name, n_dims, dims, type, offset(0)
        for name, dims, typ in tens:
            w_str(f, name)
            f.write(struct.pack("<I", len(dims)))
            for d in dims: f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", typ))
            f.write(struct.pack("<Q", 0))       # offset (placeholder; converter recomputes)
    print(f"wrote {out}: {len(tens)} tensors, {len(kv)} KV")
    # sanity print
    from collections import Counter
    c = Counter(t[2] for t in tens)
    names = {F32:"F32", F16:"F16", BF16:"BF16", Q8_0:"Q8_0", Q4_K:"Q4_K", I8:"I8"}
    print("type counts:", {names.get(k,k): v for k,v in sorted(c.items())})

if __name__ == "__main__":
    main()
