#!/usr/bin/env python3
"""DSpark drafter forward — numpy oracle.

Assembles forward_embed -> 3x DSparkBlock -> forward_head (model.py
Transformer.forward_spec) on F32 weights from dspark.gguf + the shared target
embed/lm_head, fed the captured main_hidden. Produces reference draft tokens for
the Phase-4 regression gate (phase4-refcheck). See issue468/13_phase4_forward_design.md.

Usage:
  python forward.py [--smoke]   # smoke uses random main_hidden; --validate uses
                                # the captured pos152/153 npy as prefill/decode anchors
"""
import os, sys, argparse, json
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_loader import load_gguf_dense_only, index_gguf, read_tensor
from expert_store import ExpertStore
from hc_primitives import rmsnorm, hc_pre, hc_post, hc_head
from attention import (precompute_rope, dspark_attention, dspark_attention_prefill)
from moe import moe

DSPARK_GGUF = os.path.expanduser(os.environ.get(
    "DSPARK_GGUF",
    str((Path(__file__).resolve().parents[3] / "ds4" / "gguf" / "dspark.gguf").resolve()),
))
TARGET_GGUF = os.path.expanduser(os.environ.get(
    "TARGET_GGUF",
    str((Path(__file__).resolve().parents[3] / "ds4" / "gguf" / "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf").resolve()),
))
DIM = 4096
HC = 4
BLOCK = 5
NOISE_TOK = 128799
VOCAB = 129280
MARKOV_RANK = 256
MAX_POS = 4096


def layer_weights(T, stage):
    """Pull one DSparkBlock's weights from the loaded tensor dict (GGUF ne order)."""
    p = f"mtp.{stage}."
    w = {
        "hc_attn_fn": T[p + "hc_attn_fn.weight"][0], "hc_attn_scale": T[p + "hc_attn_scale.weight"][0],
        "hc_attn_base": T[p + "hc_attn_base.weight"][0],
        "hc_ffn_fn": T[p + "hc_ffn_fn.weight"][0], "hc_ffn_scale": T[p + "hc_ffn_scale.weight"][0],
        "hc_ffn_base": T[p + "hc_ffn_base.weight"][0],
        "attn_norm": T[p + "attn_norm.weight"][0], "ffn_norm": T[p + "ffn_norm.weight"][0],
        "q_a": T[p + "attn_q_a.weight"][0], "q_a_norm": T[p + "attn_q_a_norm.weight"][0],
        "q_b": T[p + "attn_q_b.weight"][0], "kv": T[p + "attn_kv.weight"][0],
        "kv_a_norm": T[p + "attn_kv_a_norm.weight"][0], "attn_sinks": T[p + "attn_sinks.weight"][0],
        "output_a": T[p + "attn_output_a.weight"][0], "output_b": T[p + "attn_output_b.weight"][0],
        "ffn_gate_inp": T[p + "ffn_gate_inp.weight"][0], "ffn_exp_probs_b": T[p + "exp_probs_b.bias"][0],
        "ffn_gate_shexp": T[p + "ffn_gate_shexp.weight"][0], "ffn_up_shexp": T[p + "ffn_up_shexp.weight"][0],
        "ffn_down_shexp": T[p + "ffn_down_shexp.weight"][0],
    }
    return w


def forward_embed(main_hidden, anchor_tok, main_proj, main_norm, embed_w):
    """mtp.0.forward_embed. main_hidden [3*dim] -> (x [1,block,hc,dim], main_x [1,1,dim])."""
    mh = main_hidden.reshape(1, 1, 3 * DIM).astype(np.float32)
    main_x = rmsnorm(mh @ main_proj.T, main_norm)                  # [1,1,dim] (main_proj HF [out=dim,in=3*dim])
    draft_ids = np.full((1, BLOCK), NOISE_TOK, dtype=np.int64)
    draft_ids[0, 0] = anchor_tok
    # embed_w GGUF ne [dim, vocab]; row token t = embed_w[:, t]. x = embed[draft_ids]
    x = embed_w[draft_ids[0]]                                      # [block, dim] (embed HF [vocab,dim])
    x = x[None]                                                    # [1,block,dim]
    x = np.repeat(x[:, :, None, :], HC, axis=2)                    # [1,block,hc,dim]
    return x.astype(np.float32), main_x


def block_forward(x, main_x, input_ids, w, store, stage, start_pos, cos, sin, slot0_kv):
    """DSparkBlock.forward decode (hc_pre -> attn -> hc_post -> hc_pre -> ffn -> hc_post).
    x [1,block,hc,dim], main_x [1,1,dim]. Returns (x', slot1_kv)."""
    # --- attention sub-block ---
    residual = x
    yd, post, comb = hc_pre(x, w["hc_attn_fn"], w["hc_attn_scale"], w["hc_attn_base"])
    yd = rmsnorm(yd, w["attn_norm"])
    w_with_cache = dict(w); w_with_cache["_win_slot0_kv"] = slot0_kv[None] if slot0_kv is not None else None
    attn_out, slot1_kv = dspark_attention(yd, main_x, w_with_cache, start_pos, cos, sin)
    x = hc_post(attn_out, residual, post, comb)
    # --- ffn sub-block ---
    residual = x
    yd, post, comb = hc_pre(x, w["hc_ffn_fn"], w["hc_ffn_scale"], w["hc_ffn_base"])
    yd = rmsnorm(yd, w["ffn_norm"])
    ffn_out = moe(yd, input_ids, w, store)
    x = hc_post(ffn_out, residual, post, comb)
    return x.astype(np.float32), slot1_kv


def forward_head(h, anchor_tok, w_head, norm_w, hc_head_fn, hc_head_scale, hc_head_base,
                 markov_w1, markov_w2, conf_proj, lm_head, temp=1.0, return_conf=False,
                 return_full_logits=False):
    """mtp.2.forward_head.

    h [1,block,hc,dim] -> output_ids [block+1], logits [block,vocab]
    Optional confidence output follows the DSpark paper / converter contract:
    one scalar per draft position from the concatenation of normalized hidden h_k
    and the Markov embedding of the previous token.

    When return_full_logits is set, a trailing ``full_logits`` array
    ``[block, vocab]`` of the Markov-bias-applied draft logits ``(base + bias)``
    is appended to the return tuple (at temp=1.0 this is the drafter's full
    per-position distribution before softmax). Existing callers are unaffected.
    """
    x = hc_head(h, hc_head_fn, hc_head_scale, hc_head_base)        # [1,block,dim]
    x = rmsnorm(x, norm_w)
    logits = x[0] @ lm_head.T                                       # [block, vocab] (lm_head HF [vocab,dim])
    output_ids = np.zeros(BLOCK + 1, dtype=np.int64)
    output_ids[0] = anchor_tok
    conf_logits = np.zeros(BLOCK, dtype=np.float32) if return_conf else None
    conf_scores = np.zeros(BLOCK, dtype=np.float32) if return_conf else None
    full_logits = np.zeros((BLOCK, logits.shape[1]), dtype=np.float32) if return_full_logits else None
    for i in range(BLOCK):
        # markov_head(output_ids[i]): markov_w1 embed [rank], markov_w2 -> [vocab]
        emb = markov_w1[output_ids[i]]                             # [rank]  (markov_w1 HF [vocab,rank])
        if return_conf:
            conf_inp = np.concatenate((x[0, i], emb.astype(np.float32)), axis=0)
            clogit = np.float32(conf_inp @ conf_proj.astype(np.float32))
            conf_logits[i] = clogit
            conf_scores[i] = np.float32(1.0 / (1.0 + np.exp(-clogit)))
        bias = emb @ markov_w2.T                                   # [vocab]  (markov_w2 HF [vocab,rank])
        li = (logits[i] + bias) / max(temp, 1e-5)
        if full_logits is not None:
            full_logits[i] = li
        output_ids[i + 1] = int(np.argmax(li))
    if return_conf:
        if full_logits is not None:
            return output_ids, logits, conf_logits, conf_scores, full_logits
        return output_ids, logits, conf_logits, conf_scores
    if full_logits is not None:
        return output_ids, logits, full_logits
    return output_ids, logits


def forward_spec(T, infos, data_off, main_hidden_prefill, main_hidden_decode, anchor_prefill, anchor_decode,
                 embed_w, lm_head, cos, sin, temp=1.0, prefetch_workers=4):
    """Run a full 2-step drafter scenario: prefill (start_pos=0) caches slot0 KV
    from main_hidden_prefill; decode (start_pos=1) drafts from main_hidden_decode.
    Returns output_ids [block+1]."""
    main_proj = T["mtp.0.main_proj.weight"][0]
    main_norm_w = T["mtp.0.main_norm.weight"][0]
    _, main_x_p = forward_embed(main_hidden_prefill, anchor_prefill, main_proj, main_norm_w, embed_w)
    layers = [layer_weights(T, s) for s in range(3)]
    stores = [ExpertStore(DSPARK_GGUF, infos, data_off, s) for s in range(3)]
    slot0_per_layer = []
    for s in range(3):
        slot0_per_layer.append(dspark_attention_prefill(main_x_p, layers[s], cos, sin))
    x, main_x_d = forward_embed(main_hidden_decode, anchor_decode, main_proj, main_norm_w, embed_w)
    input_ids = np.array([anchor_decode], dtype=np.int64)
    # prefetch the experts each layer will need: route the 5 draft tokens' ffn
    # inputs (post-norm) through the gate; dequant the union of selected experts
    # in parallel. The dense weights needed for the ffn_norm are already in w.
    if prefetch_workers and prefetch_workers > 0:
        from moe import gate
        for s in range(3):
            # compute ffn input: hc_pre -> attn (reuse cached prefill attn is too much;
            # for routing purposes, the pre-norm residual is a good proxy). Use the
            # decode x's mean over hc/block as a representative token.
            probe = x[0].mean(axis=(0, 1))  # [dim]
            _, idx = gate(probe[None, :], layers[s]["ffn_gate_inp"],
                          layers[s]["ffn_exp_probs_b"], 6)
            stores[s].prefetch(idx[0], n_workers=prefetch_workers)
    slot1 = None
    for s in range(3):
        x, slot1 = block_forward(x, main_x_d, input_ids, layers[s], stores[s], s, 1, cos, sin, slot0_per_layer[s])
    out, logits = forward_head(
        x, anchor_decode, None, T["mtp.2.norm.weight"][0],
        T["mtp.2.hc_head_fn.weight"][0], T["mtp.2.hc_head_scale.weight"][0], T["mtp.2.hc_head_base.weight"][0],
        T["mtp.2.markov_head.markov_w1.weight"][0], T["mtp.2.markov_head.markov_w2.weight"][0],
        T["mtp.2.confidence_head.proj.weight"][0], lm_head, temp)
    return out, logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--capture-dir")
    ap.add_argument("--prefill-pos", type=int, default=152)
    ap.add_argument("--decode-pos", type=int, default=153)
    ap.add_argument("--out")
    args = ap.parse_args()
    if not args.smoke and not args.validate:
        ap.error("need --smoke or --validate")

    print("=== load dspark.gguf (dense only; experts lazy) ===", flush=True)
    kv, T, infos, data_off, skipped = load_gguf_dense_only(DSPARK_GGUF)
    print(f"  {len(T)} dense tensors loaded, {len(skipped)} packed expert tensors deferred to ExpertStore")
    print("=== load shared embed + lm_head from target GGUF (lazy, 2 tensors) ===", flush=True)
    _, tinfos, tdata_off = index_gguf(TARGET_GGUF)
    embed_w = read_tensor(TARGET_GGUF, tinfos, tdata_off, "token_embd.weight").astype(np.float32)
    lm_head = read_tensor(TARGET_GGUF, tinfos, tdata_off, "output.weight").astype(np.float32)
    print(f"  embed {embed_w.shape}, lm_head {lm_head.shape}")

    cos, sin = precompute_rope(64, MAX_POS)

    if args.validate:
        cap = Path(args.capture_dir) if args.capture_dir else Path("issue468/baseline/dspark_capture")
        mh_p = np.load(cap / f"main_hidden_pos{args.prefill_pos}.npy")
        mh_d = np.load(cap / f"main_hidden_pos{args.decode_pos}.npy")
        selected_path = cap / "target_selected_tokens.json"
        if not selected_path.exists():
            selected_path = cap.parent / "target_selected_tokens.json"
        if not selected_path.exists():
            selected_path = cap / "target_greedy.json"
        if not selected_path.exists():
            selected_path = cap.parent / "target_greedy.json"
        selected = json.loads(selected_path.read_text())
        rel0 = args.prefill_pos - args.prefill_pos
        rel1 = args.decode_pos - args.prefill_pos
        if rel0 < 0 or rel1 < 0 or rel1 >= len(selected):
            raise ValueError("prefill/decode positions are out of range for the retained selected token stream")
        anchor_p = int(selected[rel0])
        anchor_d = int(selected[rel1])
    else:
        rng = np.random.default_rng(args.seed)
        mh_p = (rng.standard_normal(3 * DIM) * 0.5).astype(np.float32)
        mh_d = (rng.standard_normal(3 * DIM) * 0.5).astype(np.float32)
        anchor_p, anchor_d = 366, 366

    print(f"=== forward_spec (prefill anchor {anchor_p}, decode anchor {anchor_d}) ===", flush=True)
    out, logits = forward_spec(T, infos, data_off, mh_p, mh_d, anchor_p, anchor_d, embed_w, lm_head, cos, sin, temp=1.0)
    print(f"  output_ids: {out.tolist()}")
    print(f"  logits shape: {logits.shape}, range [{logits.min():.2f}, {logits.max():.2f}]")
    # confidence (Phase 6 only; compute for completeness)
    print(f"  draft tokens (positions 1..5): {out[1:].tolist()}")
    if args.validate:
        out_path = Path(args.out) if args.out else (cap / "oracle_ref.npz")
        np.savez(out_path, output_ids=out, logits=logits)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
