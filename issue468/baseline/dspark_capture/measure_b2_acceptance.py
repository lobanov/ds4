#!/usr/bin/env python3
"""Measure DSpark B2 rejection-sampling acceptance + committed tokens per cycle.

THE decisive GO/NO-GO measurement for the DSpark project (see issue468/17_*).

Context: doc 15 projected speedup using greedy acceptance (2.79 avg prefix) and a
4x-wrong verify cost. Doc 17 corrected the verify cost (verify(L=5)=75ms measured,
doc 06) -> at greedy acceptance the gate FAILS. But greedy is NOT an upper bound
for B2: rejection sampling accepts any plausible draft (prob min(1,p/q)), so B2
committed-tokens >= greedy. This script measures the real B2 number.

Two estimates, both reusing the validated numpy oracle (issue468/dspark_oracle/):
  (1) Analytical per-position acceptance = 1 - TV(p_i, q_i) with greedy-conditioned
      Markov head. Upper bound (true prev token is best case for drafter). Fast.
  (2) Monte Carlo B2 simulation: drafter SAMPLES from q (temp=1.0), Markov head
      conditions on the sampled token; verifier accepts w.p. min(1,p/q), resamples
      from norm(max(0,p-q)) on reject. Realistic committed tokens per cycle.

Inputs (from issue468/baseline/dspark_capture/):
  - hc_dspark_main_hc-{40,41,42}_pos{152..175}.bin  (main_hidden captures)
  - target_topk200.json                              (target top-128 logprobs/pos)
  - greedy25_tokens.json                             (anchor tokens, true continuation)
"""
import os, sys, json, argparse
import numpy as np

ORACLE = os.path.join(os.path.dirname(__file__), "..", "..", "dspark_oracle")
sys.path.insert(0, os.path.normpath(ORACLE))
from gguf_loader import load_gguf_dense_only, index_gguf, read_tensor
from expert_store import ExpertStore
from forward import layer_weights, forward_head, DIM, BLOCK, HC, NOISE_TOK
from attention import (precompute_rope, apply_rotary, dspark_topk_idxs, sparse_attn,
                       dspark_attention_prefill, ROPE_DIM, HEAD_DIM)
from hc_primitives import rmsnorm, hc_pre, hc_post, hc_head
from moe import moe

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DSPARK = os.path.join(_ROOT, "..", "ds4", "gguf", "dspark.gguf")
TARGET = os.path.join(_ROOT, "..", "ds4", "gguf",
    "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf")
CAP = os.path.dirname(os.path.abspath(__file__))

WIN = 128; N_HEADS = 64; N_GROUPS = 8; O_LORA = 1024
RNG = np.random.default_rng(20260628)


def load_mh(pos):
    parts = []
    for L in [40, 41, 42]:
        f = os.path.join(CAP, f"hc_dspark_main_hc-{L}_pos{pos}.bin")
        if not os.path.exists(f): return None
        parts.append(np.fromfile(f, dtype=np.float32).reshape(HC, DIM).mean(0))
    return np.concatenate(parts).astype(np.float32)


def softmax(x):
    x = x - x.max()
    e = np.exp(x); return e / e.sum()


def dspark_attn(x_draft, main_x, w, win_kv, start_pos, cos, sin):
    """DSpark decode attention with PERSISTENT window KV [win,512] (slots 0..start_pos-1 filled).
    Mirrors attention.py dspark_attention matmuls EXACTLY (.T HF convention), but threads
    a persistent window across decode steps instead of slot0+current only.
    Caches the current anchor KV (from main_x) at slot start_pos%win, then attends."""
    bs = 1; block = x_draft.shape[1]; scale = HEAD_DIM ** -0.5
    n_heads = N_HEADS; head_dim = HEAD_DIM; n_groups = N_GROUPS; o_lora = O_LORA
    # 1. cache current anchor KV from main_x at slot start_pos%win
    main_kv = rmsnorm(main_x @ w["kv"].T, w["kv_a_norm"])            # [1,1,512]
    fc = cos[start_pos], sin[start_pos]
    main_kv[..., -ROPE_DIM:] = apply_rotary(main_kv[..., -ROPE_DIM:], fc[0], fc[1])
    win_kv[start_pos % WIN] = main_kv[0, 0]
    # 2. draft-block q, kv (same matmuls as attention.py)
    qr = rmsnorm(x_draft @ w["q_a"].T, w["q_a_norm"])              # [1,block,1024]
    q = (qr @ w["q_b"].T).reshape(bs, block, n_heads, head_dim)      # [1,block,64,512]
    q = q * (1.0 / np.sqrt(np.mean(q*q, axis=-1, keepdims=True) + 1e-6))
    cs = cos[start_pos+1:start_pos+1+block]; ss = sin[start_pos+1:start_pos+1+block]
    cs_q = cs[None,:,None,:]; ss_q = ss[None,:,None,:]
    cs_kv = cs[None,:,:]; ss_kv = ss[None,:,:]
    q[..., -ROPE_DIM:] = apply_rotary(q[..., -ROPE_DIM:], cs_q, ss_q)
    kv = rmsnorm(x_draft @ w["kv"].T, w["kv_a_norm"])              # [1,block,512]
    kv[..., -ROPE_DIM:] = apply_rotary(kv[..., -ROPE_DIM:], cs_kv, ss_kv)
    kv_all = np.concatenate([win_kv[None], kv], axis=1)             # [1, win+block, 512]
    # 3. DSpark topk pattern (reads slots 0..min(win,start+1)-1 ++ win..win+block-1)
    idx = dspark_topk_idxs(WIN, block, start_pos)
    kv_g = np.broadcast_to(kv_all[:, idx, :], (bs, block, idx.size, head_dim))
    o = sparse_attn(q, kv_g, w["attn_sinks"], scale)               # [1,block,64,512]
    o[..., -ROPE_DIM:] = apply_rotary(o[..., -ROPE_DIM:], cs_q, -ss_q)
    # 4. grouped low-rank output projection (identical to attention.py)
    gd = head_dim * n_heads // n_groups                              # 4096
    o_g = o.reshape(bs, block, n_groups, gd)                        # [1,block,8,4096]
    wo_a = w["output_a"].reshape(n_groups, o_lora, gd)             # [8,1024,4096] (HF order)
    o_lor = np.einsum("bsgd,grd->bsgr", o_g, wo_a)                 # [1,block,8,1024]
    o_flat = o_lor.reshape(bs, block, n_groups * o_lora)            # [1,block,8192]
    return (o_flat @ w["output_b"].T).astype(np.float32)           # [1,block,4096]


def run_blocks(x, main_x, step, layers, stores, win_kv_list, cos, sin):
    """3 DSpark blocks -> h [1,block,hc,dim]. Each block's dspark_attn caches the
    current anchor KV (from main_x) into its persistent win_kv_list[s]. Mirrors
    forward.py block_forward structure with validated hc_pre/post + moe."""
    for s in range(3):
        res = x
        yd, post, comb = hc_pre(x, layers[s]["hc_attn_fn"], layers[s]["hc_attn_scale"], layers[s]["hc_attn_base"])
        yd = rmsnorm(yd, layers[s]["attn_norm"])
        ao = dspark_attn(yd, main_x, layers[s], win_kv_list[s], step, cos, sin)
        x = hc_post(ao, res, post, comb)
        res = x
        yd, post, comb = hc_pre(x, layers[s]["hc_ffn_fn"], layers[s]["hc_ffn_scale"], layers[s]["hc_ffn_base"])
        yd = rmsnorm(yd, layers[s]["ffn_norm"])
        fo = moe(yd, np.array([0]), layers[s], stores[s])
        x = hc_post(fo, res, post, comb)
    return x


def head_base_logits(h, T, lm_head):
    """hc_head + norm + lm_head -> base logits [block, vocab] (pre-Markov)."""
    x = hc_head(h, T["mtp.2.hc_head_fn.weight"][0],
                T["mtp.2.hc_head_scale.weight"][0], T["mtp.2.hc_head_base.weight"][0])
    x = rmsnorm(x, T["mtp.2.norm.weight"][0])
    return (x[0] @ lm_head.T).astype(np.float32)   # [block, vocab]


def markov_bias(prev_tok, markov_w1, markov_w2):
    return (markov_w1[prev_tok] @ markov_w2.T).astype(np.float32)   # [vocab]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-dir", default=CAP, help="dir with hc_dspark_main_hc-*_posN.bin")
    ap.add_argument("--target-json", default=os.path.join(CAP, "target_topk200.json"))
    ap.add_argument("--greedy-json", default=os.path.join(CAP, "greedy25_tokens.json"))
    ap.add_argument("--pos0", type=int, default=152, help="first generated position")
    ap.add_argument("--label", default="low-fill")
    args = ap.parse_args()
    capdir = args.cap_dir; POS0 = args.pos0
    def load_mh_pos(pos):
        parts = []
        for L in [40, 41, 42]:
            f = os.path.join(capdir, f"hc_dspark_main_hc-{L}_pos{pos}.bin")
            if not os.path.exists(f): return None
            parts.append(np.fromfile(f, dtype=np.float32).reshape(HC, DIM).mean(0))
        return np.concatenate(parts).astype(np.float32)
    print("=== load weights ===", flush=True)
    print(f"=== data: {args.label} | cap-dir={capdir} | target={os.path.basename(args.target_json)} ===", flush=True)
    _, T, infos, doff, _ = load_gguf_dense_only(DSPARK)
    _, ti, tdo = index_gguf(TARGET)
    lm_head = read_tensor(TARGET, ti, tdo, "output.weight").astype(np.float32)
    cos, sin = precompute_rope(64, 4096)
    greedy_src = json.load(open(args.greedy_json))
    # support both list (greedy25_tokens.json) and the logprobs-dump shape (dict w/ steps)
    if isinstance(greedy_src, dict):
        greedy = [s["selected"]["id"] for s in greedy_src["steps"]]
    else:
        greedy = greedy_src
    tgt = json.load(open(args.target_json))
    tgt_steps = tgt["steps"]   # tgt_steps[k] = target dist at pos POS0+k

    main_proj = T["mtp.0.main_proj.weight"][0]; main_norm_w = T["mtp.0.main_norm.weight"][0]
    layers = [layer_weights(T, s) for s in range(3)]
    stores = [ExpertStore(DSPARK, infos, doff, s) for s in range(3)]
    mw1 = T["mtp.2.markov_head.markov_w1.weight"][0]   # [vocab, 256]
    mw2 = T["mtp.2.markov_head.markov_w2.weight"][0]   # [vocab, 256]

    # Build target p per position: {tok: prob}, renormalized over captured top-128.
    # logprob is full-vocab log-prob -> prob = exp(logprob). Sum of top-128 ~0.9999+.
    def target_p(step_idx):
        """step_idx = target generation step = pos-POS0. Returns {tok:prob} dict."""
        entries = tgt_steps[step_idx]["top_logprobs"]
        toks = np.array([e["token"]["id"] for e in entries], dtype=np.int64)
        probs = np.exp(np.array([e["logprob"] for e in entries], dtype=np.float64))
        probs /= probs.sum()   # renormalize over captured support
        return toks, probs

    win_kv = [np.zeros((WIN, HEAD_DIM), dtype=np.float32) for _ in range(3)]
    # Prefill anchor KV at slot 0 (pos POS0) using the validated prefill matmul.
    mh0 = load_mh_pos(POS0)
    main_x0 = rmsnorm(mh0.reshape(1,1,3*DIM) @ main_proj.T, main_norm_w)
    for s in range(3):
        slot0 = dspark_attention_prefill(main_x0, layers[s], cos, sin)   # [512]
        win_kv[s][0] = slot0

    # ---- decode steps: greedy-conditioned anchor, full forward to base logits ----
    MAX_STEP = min(len(greedy) - 6, len(tgt_steps) - 6)   # need step+5 in both
    print(f"\nprefill done. Measuring B2 over decode steps 1..{MAX_STEP} (greedy anchor).")
    print("loading embed and running forward per step...", flush=True)
    embed_w = read_tensor(TARGET, ti, tdo, "token_embd.weight").astype(np.float32)
    base_logits_cache = {}
    for step in range(1, MAX_STEP + 1):
        pos = POS0 + step
        mh = load_mh_pos(pos)
        if mh is None: break
        main_x = rmsnorm(mh.reshape(1,1,3*DIM) @ main_proj.T, main_norm_w)
        anchor = greedy[step]
        draft_ids = np.full(BLOCK, NOISE_TOK, dtype=np.int64); draft_ids[0] = anchor
        x = embed_w[draft_ids][None]; x = np.repeat(x[:,:,None,:], HC, axis=2)
        h = run_blocks(x, main_x, step, layers, stores, win_kv, cos, sin)
        base_logits_cache[step] = head_base_logits(h, T, lm_head)
        # advance the persistent window KV: dspark_attn already cached slot `step` for each layer
    print(f"  forward done for {len(base_logits_cache)} steps.")

    # ============ SANITY: greedy acceptance (must reproduce doc-15 ~2.79) ============
    print("\n=== SANITY: greedy acceptance (validates the forward; expect ~2.79 per doc 15) ===")
    g_match=0; g_tot=0; g_hist={}
    for step in base_logits_cache:
        base = base_logits_cache[step]
        prev = greedy[step]
        prefix=0
        for j in range(BLOCK):
            li = base[j] + markov_bias(prev, mw1, mw2)
            dtok = int(np.argmax(li))
            ttok = greedy[step+1+j]
            g_tot+=1
            if dtok==ttok:
                g_match+=1; prefix=j+1; prev=dtok
            else: break
        g_hist[prefix]=g_hist.get(prefix,0)+1
    g_avg = sum(k*v for k,v in g_hist.items())/sum(g_hist.values()) if g_hist else 0
    print(f"  greedy match: {g_match}/{g_tot} = {100*g_match/max(g_tot,1):.1f}%")
    print(f"  greedy accepted prefix hist: {dict(sorted(g_hist.items()))}  avg={g_avg:.2f}")
    print(f"  -> doc 15 reported 57.9% / 2.79. If this differs materially, the forward is off; DO NOT trust B2 below.")

    # ============ (1) Analytical per-position acceptance: 1 - TV(p, q) ============
    # Greedy-conditioned Markov (prev = true target token). UPPER BOUND on drafter.
    print("\n=== (1) Analytical per-position acceptance (1 - TV), greedy-conditioned (UPPER BOUND) ===")
    tv_pos = [[] for _ in range(BLOCK)]   # tv_pos[j] = list of (1-TV) at block pos j
    for step in base_logits_cache:
        base = base_logits_cache[step]
        prev = greedy[step]   # anchor = true token at pos 152+step
        for j in range(BLOCK):
            q_logits = base[j] + markov_bias(prev, mw1, mw2)
            q = softmax(q_logits)
            tstep = step + 1 + j   # target step index for this draft pos
            if tstep >= len(tgt_steps): break
            ptoks, pprobs = target_p(tstep)
            p = np.zeros_like(q)
            p[ptoks] = pprobs
            tv = 0.5 * np.abs(p - q).sum()
            tv_pos[j].append(1.0 - tv)
            prev = greedy[step + 1 + j]   # greedy-conditioned
    print(f"  {'blk_pos':>7} {'E[accept|reach]':>16} {'n':>4}")
    for j in range(BLOCK):
        if tv_pos[j]:
            v = np.mean(tv_pos[j])
            print(f"  {j:>7} {v:>16.4f} {len(tv_pos[j]):>4}")
    # Expected committed prefix from per-position acceptance (geometric-ish, with resample bonus=1)
    pe = [np.mean(tv_pos[j]) if tv_pos[j] else 0.0 for j in range(BLOCK)]
    exp_prefix = 0.0
    reach = 1.0
    for j in range(BLOCK):
        exp_prefix += reach * pe[j]
        reach *= pe[j]
    print(f"  -> E[committed prefix | greedy-cond] = {exp_prefix:.3f} + 1 bonus = {exp_prefix+1:.3f}/cycle (UPPER BOUND)")

    # ============ (2) Monte Carlo B2 simulation (realistic: sample + resample) =====
    print("\n=== (2) Monte Carlo B2 simulation (sample from q, accept min(1,p/q), resample norm(max(0,p-q))) ===")
    N_CYCLES = 200
    committed_hist = {}
    pos_accept = [0,0,0,0,0]; pos_total = [0,0,0,0,0]
    # fill-binned: bin by n_real (= step+1) into buckets to see if high fill lifts acceptance
    FILL_BINS = [(2,10),(11,30),(31,60),(61,100),(101,128)]
    bin_committed = {b: [] for b in FILL_BINS}
    for step in base_logits_cache:
        base = base_logits_cache[step]
        n_real = min(step + 1, WIN)
        bkey = next((b for b in FILL_BINS if b[0] <= n_real <= b[1]), None)
        for _ in range(N_CYCLES):
            prev = greedy[step]
            n_committed = 0
            for j in range(BLOCK):
                q_logits = base[j] + markov_bias(prev, mw1, mw2)
                q = softmax(q_logits)
                x = int(RNG.choice(len(q), p=q))   # drafter samples
                tstep = step + 1 + j
                if tstep >= len(tgt_steps): break
                ptoks, pprobs = target_p(tstep)
                pmap = dict(zip(ptoks.tolist(), pprobs.tolist()))
                px = pmap.get(x, 0.0)
                qx = q[x]
                accept_p = min(1.0, px / qx) if qx > 0 else 0.0
                pos_total[j] += 1
                if RNG.random() < accept_p:
                    n_committed += 1; pos_accept[j] += 1; prev = x
                else:
                    # resample from norm(max(0, p - q)); committed the resample, then STOP
                    n_committed += 1
                    break
            committed_hist[n_committed] = committed_hist.get(n_committed, 0) + 1
            if bkey: bin_committed[bkey].append(n_committed)
    total_cycles = sum(committed_hist.values())
    avg_committed = sum(k*v for k,v in committed_hist.items()) / total_cycles
    print(f"  cycles simulated: {total_cycles}")
    print(f"  committed/cycle histogram: {dict(sorted(committed_hist.items()))}")
    print(f"  per-position empirical accept rate: {[round(pos_accept[j]/max(pos_total[j],1),3) for j in range(BLOCK)]}")
    print(f"  -> E[committed tokens / cycle] = {avg_committed:.3f}")
    print(f"  committed/cycle BY KV-FILL LEVEL (does high fill lift acceptance?):")
    print(f"    {'fill(n_real)':>14} {'mean_committed':>15} {'n_cycles':>9}")
    for b in FILL_BINS:
        vals = bin_committed[b]
        if vals:
            print(f"    {f'{b[0]}-{b[1]}':>14} {np.mean(vals):>15.3f} {len(vals):>9}")

    # ============ Speedup with corrected verify cost ============
    print("\n=== Speedup projection (verify(L=5)=75ms measured doc06; plain=25.7ms/tok) ===")
    V, D_plain = 75.0, 25.7
    for label, draft_est, committed in [
        ("analytic_upper", 15.0, exp_prefix + 1),
        ("montecarlo", 15.0, avg_committed),
        ("montecarlo_pess_draft", 30.0, avg_committed),
    ]:
        if committed <= 0: continue
        msptok = (draft_est + V) / committed
        print(f"  {label:22s} draft={draft_est:.0f}ms verify={V:.0f}ms committed={committed:.2f} -> {msptok:.1f}ms/tok = {D_plain/msptok:.2f}x ({100*(D_plain/msptok-1):+.0f}%) vs plain; gate>20%: {'PASS' if D_plain/msptok > 1.20 else 'FAIL'}")


if __name__ == "__main__":
    main()
