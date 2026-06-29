#!/usr/bin/env python3
"""Assignment 1a — B2 rejection-sampling acceptance on the METAL drafter.

THE decisive measurement for whether Outcome B flips on measurement alone.
The prior verdict (issue468/23) used Metal GREEDY acceptance (1.53). This measures
the actually-selected B2 protocol (issue468/24 observation 3) on Metal's real
drafter distribution (production Q4_K kernels).

Method: the Metal drafter's window KV depends only on target anchors (main_x), not
on sampled drafts (positions 1-4 are noise tokens), so the per-step base_logits
[dumped by DS4_DSPARK_PROBE_DUMP_Q via the persistent-KV sweep] are valid for an
offline B2 MC simulation with many trials per step. B2 protocol (mirrors
measure_b2_acceptance.py but on Metal base_logits):
  q_i = softmax(base_logits[i] + markov_bias(prev)) at temp=1.0   [drafter dist]
  p_i = target distribution (top-128 from --dump-logprobs)        [verifier dist]
  drafter SAMPLES x_i ~ q_i
  accept w.p. min(1, p(x_i)/q(x_i)); on reject resample norm(max(0, p_i - q_i)), STOP
  committed = (accepted count) + 1   [the structural bonus/correction token]

Gate (issue468/24 1d): speedup >20% at 64k needs B2-accepted A >= ~1.78
(committed >= 2.78). Oracle greedy A=2.79. The question: does Metal B2 A reach ~1.78?
"""
import os, sys, json
import numpy as np

ORACLE = os.path.join(os.path.dirname(__file__), "..", "..", "dspark_oracle")
sys.path.insert(0, os.path.normpath(ORACLE))
from gguf_loader import load_gguf_dense_only

CAP = os.path.dirname(os.path.abspath(__file__))
DSPARK = "../ds4/gguf/dspark.gguf" if not os.path.isabs("../ds4/gguf/dspark.gguf") else "../ds4/gguf/dspark.gguf"
DSPARK = os.path.normpath(os.path.join(CAP, "..", "..", "..", "..", "ds4", "gguf", "dspark.gguf"))
if not os.path.exists(DSPARK):
    DSPARK = os.path.normpath(os.path.join(CAP, "..", "..", "..", "ds4", "gguf", "dspark.gguf"))
RNG = np.random.default_rng(20260629)
BLOCK = 5
VOCAB = 129280
RANK = 256
NOISE_TOK = 128799


def bf16_to_f32(u16):
    """BF16 (u16 bits) -> F32: top 16 bits of f32."""
    bits = u16.astype(np.uint32) << 16
    return bits.view(np.float32)


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def main():
    # Load Metal base_logits [n_steps, 5, vocab]
    qpath = os.path.join(CAP, "metal_base_logits_19steps.bin")
    base = np.fromfile(qpath, dtype=np.float32).reshape(19, BLOCK, VOCAB)
    n_steps = base.shape[0]
    print(f"loaded Metal base_logits: {base.shape} (range {base.min():.2f}..{base.max():.2f})")

    # Load markov_w1/w2 [vocab, 256] BF16 from dspark.gguf
    _, T, infos, doff, _ = load_gguf_dense_only(DSPARK)
    mw1_off = infos["mtp.2.markov_head.markov_w1.weight"]  # ([256,vocab], 30, off)
    mw2_off = infos["mtp.2.markov_head.markov_w2.weight"]
    # Read raw BF16 directly
    with open(DSPARK, "rb") as f:
        f.seek(doff + mw1_off[2]); mw1 = bf16_to_f32(np.frombuffer(f.read(RANK*VOCAB*2), dtype=np.uint16).reshape(VOCAB, RANK))  # [vocab,256]
        f.seek(doff + mw2_off[2]); mw2 = bf16_to_f32(np.frombuffer(f.read(RANK*VOCAB*2), dtype=np.uint16).reshape(VOCAB, RANK))  # [vocab,256]

    # Load target greedy anchors + top-128 p(x) per position
    greedy = json.load(open(os.path.join(CAP, "target_greedy_130.json")))
    tgt = json.load(open(os.path.join(CAP, "target_topk200_ext.json")))
    tgt_steps = tgt["steps"]

    def target_p(step_idx):
        """step_idx = pos-152. Returns (toks[v], probs[v]) for the top-128, renormalized."""
        entries = tgt_steps[step_idx]["top_logprobs"]
        toks = np.array([e["token"]["id"] for e in entries], dtype=np.int64)
        probs = np.exp(np.array([e["logprob"] for e in entries], dtype=np.float64))
        probs /= probs.sum()
        return toks, probs

    # B2 MC simulation over all steps, many trials each.
    N_TRIALS = 500
    pos_accept = [0]*BLOCK; pos_total = [0]*BLOCK
    committed_hist = {}
    per_step_committed = []
    print(f"\n=== B2 rejection-sampling MC on METAL base_logits ({N_TRIALS} trials/step) ===")
    print(f"{'step':>4} {'pos':>4} {'greedy[0]':>9} {'B2-E[committed]':>15} {'accept[0..4]':>30}")
    for step in range(n_steps):
        anchor = greedy[step]   # target token at this pos (drafter's anchor / prev seed)
        # Precompute per-position q and target p for this step
        qs = np.empty((BLOCK, VOCAB), dtype=np.float64)
        # (q depends on prev via Markov; but in B2 the draft SEQUENCE varies per trial,
        #  so q must be computed per-trial. However: markov_bias(prev) is cheap; compute
        #  base+markov inside the trial loop. Precompute the base rows.)
        # Target p per position (fixed — depends only on target, not draft)
        tstep0 = step + 1   # target dist for draft position 0 is at pos 152+step+1
        ps = []  # per-position (toks, probs, full-vocab p)
        for j in range(BLOCK):
            tstep = tstep0 + j
            if tstep >= len(tgt_steps):
                ps.append(None); continue
            toks, probs = target_p(tstep)
            p_full = np.zeros(VOCAB, dtype=np.float64); p_full[toks] = probs
            ps.append((toks, probs, p_full))

        trial_committed = []
        for _ in range(N_TRIALS):
            prev = anchor
            n_committed = 0
            for j in range(BLOCK):
                if ps[j] is None: break
                bias = mw1[prev] @ mw2.T   # [vocab] markov bias
                q = softmax((base[step, j] + bias).astype(np.float64))
                x = int(RNG.choice(VOCAB, p=q))   # drafter samples
                toks, probs, p_full = ps[j]
                px = p_full[x] if x < VOCAB else 0.0
                qx = q[x]
                accept_p = min(1.0, px/qx) if qx > 0 else 0.0
                pos_total[j] += 1
                if RNG.random() < accept_p:
                    n_committed += 1; pos_accept[j] += 1; prev = x
                else:
                    n_committed += 1  # resample (norm(max(0,p-q))) counts as committed, then STOP
                    break
            trial_committed.append(n_committed)
        avg_c = float(np.mean(trial_committed))
        per_step_committed.append(avg_c)
        for c in trial_committed: committed_hist[c] = committed_hist.get(c,0)+1
        acc_rate = [round(pos_accept[j]/max(pos_total[j],1),3) for j in range(BLOCK)]
        print(f"{step+1:>4} {152+step+1:>4} {greedy[step+1]:>9} {avg_c:>15.3f} {str(acc_rate):>30}", flush=True)

    overall = np.mean(per_step_committed)
    A = overall - 1   # accepted (excluding structural +1)
    print(f"\n=== SUMMARY (Metal B2, {N_TRIALS} trials/step, {n_steps} steps) ===")
    print(f"  E[committed/cycle] = {overall:.3f}")
    print(f"  E[accepted A]      = {A:.3f}  (committed - 1)")
    print(f"  committed hist     = {dict(sorted(committed_hist.items()))}")
    print(f"  per-position accept rate = {[round(pos_accept[j]/max(pos_total[j],1),3) for j in range(BLOCK)]}")
    print()
    print(f"  COMPARISON:")
    print(f"    Metal GREEDY (issue468/23): A = 1.53, committed = 2.53")
    print(f"    Metal B2 (THIS):            A = {A:.2f}, committed = {overall:.2f}")
    print(f"    Oracle GREEDY (doc 15):     A = 2.79, committed = 3.79")
    print(f"    Oracle B2 (measure_b2):     committed ~2.8-3.0")
    print()
    # Speedup at 64k with committed = overall
    draft, verify, plain64 = 7.2, 75.0, 35.5
    msptok = (draft+verify)/overall
    sp = plain64/msptok
    print(f"  SPEEDUP @64k (committed={overall:.2f}, draft {draft}+verify {verify}={draft+verify}ms, plain {plain64}ms/tok):")
    print(f"    ms/tok = {msptok:.1f}, speedup = {sp:.2f}x ({100*(sp-1):+.0f}%), gate >20%: {'PASS' if sp>1.20 else 'FAIL'}")
    print(f"    (gate needs committed >= 2.78 at 64k; this measures {overall:.2f})")


if __name__ == "__main__":
    main()
