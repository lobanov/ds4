#!/usr/bin/env python3
"""Score one saved reference-path DSpark case against captured target top-k."""
import argparse
import json
import math
import os

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max()
    probs = np.exp(logits, dtype=np.float64)
    return probs / probs.sum()


def load_greedy_tokens(path: str) -> list[int]:
    src = json.load(open(path))
    if isinstance(src, dict):
        return [step["selected"]["id"] for step in src["steps"]]
    return src


def target_dist(step_entry: dict) -> tuple[np.ndarray, np.ndarray]:
    toks = np.array([entry["token"]["id"] for entry in step_entry["top_logprobs"]], dtype=np.int64)
    probs = np.exp(np.array([entry["logprob"] for entry in step_entry["top_logprobs"]], dtype=np.float64))
    probs /= probs.sum()
    return toks, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-npz", required=True)
    ap.add_argument("--target-json", default=os.path.join("issue468", "baseline", "dspark_capture", "target_topk200.json"))
    ap.add_argument("--greedy-json", default=os.path.join("issue468", "baseline", "dspark_capture", "greedy25_tokens.json"))
    ap.add_argument("--decode-step", type=int, default=1, help="decode step index relative to greedy25_tokens baseline")
    args = ap.parse_args()

    ref = np.load(args.ref_npz)
    logits = ref["ref_logits"].astype(np.float64)
    output_ids = ref["ref_output_ids"].astype(np.int64)
    greedy = load_greedy_tokens(args.greedy_json)
    target = json.load(open(args.target_json))["steps"]

    print(f"=== score reference case: {args.ref_npz} ===")
    print(f"decode step index: {args.decode_step}")
    print(f"anchor token: sampled={int(output_ids[0])} greedy={int(greedy[args.decode_step])}")

    sampled_prefix = 0
    greedy_prefix = 0
    sampled_chain_accept = []
    sampled_chain_tvb = []
    sampled_chain_exp = 0.0
    reach = 1.0

    for pos in range(logits.shape[0]):
        q = softmax(logits[pos])
        sampled_tok = int(output_ids[pos + 1])
        greedy_tok = int(np.argmax(logits[pos]))
        target_step = args.decode_step + 1 + pos
        ttoks, tprobs = target_dist(target[target_step])
        p = np.zeros_like(q)
        p[ttoks] = tprobs
        px = float(p[sampled_tok])
        qx = float(q[sampled_tok])
        accept_prob = min(1.0, px / qx) if qx > 0 else 0.0
        tv_accept = 1.0 - 0.5 * np.abs(p - q).sum()
        sampled_chain_accept.append(accept_prob)
        sampled_chain_tvb.append(tv_accept)
        sampled_chain_exp += reach
        reach *= accept_prob

        greedy_truth = int(greedy[target_step])
        if sampled_tok == greedy_truth and sampled_prefix == pos:
            sampled_prefix += 1
        if greedy_tok == greedy_truth and greedy_prefix == pos:
            greedy_prefix += 1

        print(
            f"pos={pos} target_step={target_step} sampled={sampled_tok} greedy={greedy_tok} "
            f"truth={greedy_truth} accept_prob={accept_prob:.4f} tv_accept={tv_accept:.4f}"
        )

    print(f"sampled accepted prefix vs truth: {sampled_prefix}/{logits.shape[0]}")
    print(f"greedy accepted prefix vs truth: {greedy_prefix}/{logits.shape[0]}")
    print(f"sampled-chain accept probs: {[round(x, 4) for x in sampled_chain_accept]}")
    print(f"sampled-chain 1-TV upper bounds: {[round(x, 4) for x in sampled_chain_tvb]}")
    print(f"expected committed tokens on sampled chain: {sampled_chain_exp:.4f}")


if __name__ == "__main__":
    main()
