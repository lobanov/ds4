#!/usr/bin/env python3
"""Measure DSpark B2 rejection-sampling acceptance on Metal q-dumps.

This is the parameterized version of the original one-off script in
``issue468/baseline/dspark_capture``. It consumes:

- target hidden-state captures + top-k logprobs from a target-only run
- one ``metal_base_logits_*.bin`` dump from ``DS4_DSPARK_PROBE_DUMP_Q``
- one DSpark GGUF providing the Markov head weights

The q-dump index is aligned to ``ds4_dspark_probe_accept()``:
``q[0]`` corresponds to probe ``step=1``, anchor ``greedy[1]``, and predicts
target positions ``greedy[2:7]``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ORACLE = os.path.join(os.path.dirname(__file__), "..", "..", "dspark_oracle")
sys.path.insert(0, os.path.normpath(ORACLE))
from gguf_loader import load_gguf_dense_only

CAP = Path(__file__).resolve().parent
DEFAULT_DSPARK = (CAP / ".." / ".." / ".." / ".." / "ds4" / "gguf" / "dspark.gguf").resolve()
if not DEFAULT_DSPARK.exists():
    DEFAULT_DSPARK = (CAP / ".." / ".." / ".." / "ds4" / "gguf" / "dspark.gguf").resolve()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q-path", default=str(CAP / "metal_base_logits_19steps.bin"))
    ap.add_argument("--target-json", default=str(CAP / "target_topk200_ext.json"))
    ap.add_argument("--greedy-json", default=str(CAP / "target_greedy_130.json"))
    ap.add_argument("--dspark", default=str(DEFAULT_DSPARK))
    ap.add_argument("--pos0", type=int, default=152, help="anchor position used by the probe prefill")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260630)
    ap.add_argument("--block", type=int, default=5)
    ap.add_argument("--vocab", type=int, default=129280)
    ap.add_argument("--rank", type=int, default=256)
    ap.add_argument("--label", default="metal-b2")
    ap.add_argument("--plain-ms", type=float, default=35.5)
    ap.add_argument("--draft-ms", type=float, default=7.2)
    ap.add_argument("--verify-ms", type=float, default=75.0)
    ap.add_argument("--json-out")
    return ap.parse_args()


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    bits = u16.astype(np.uint32) << 16
    return bits.view(np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def load_greedy(path: str) -> list[int]:
    src = json.loads(Path(path).read_text())
    if isinstance(src, dict):
        return [int(step["selected"]["id"]) for step in src.get("steps", [])]
    return [int(v) for v in src]


def load_markov_weights(dspark_path: str, vocab: int, rank: int) -> tuple[np.ndarray, np.ndarray]:
    _, _, infos, doff, _ = load_gguf_dense_only(dspark_path)
    mw1_info = infos["mtp.2.markov_head.markov_w1.weight"]
    mw2_info = infos["mtp.2.markov_head.markov_w2.weight"]
    with open(dspark_path, "rb") as f:
        f.seek(doff + mw1_info[2])
        mw1 = bf16_to_f32(
            np.frombuffer(f.read(vocab * rank * 2), dtype=np.uint16).reshape(vocab, rank)
        )
        f.seek(doff + mw2_info[2])
        mw2 = bf16_to_f32(
            np.frombuffer(f.read(vocab * rank * 2), dtype=np.uint16).reshape(vocab, rank)
        )
    return mw1, mw2


def load_q_dump(path: str, block: int, vocab: int) -> np.ndarray:
    flat = np.fromfile(path, dtype=np.float32)
    step_size = block * vocab
    if flat.size == 0 or flat.size % step_size != 0:
        raise ValueError(
            f"q-dump size {flat.size} is not divisible by block*vocab={step_size}"
        )
    return flat.reshape(flat.size // step_size, block, vocab)


def load_target_support(path: str, vocab: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    tgt = json.loads(Path(path).read_text())
    supports = []
    for step in tgt["steps"]:
        entries = step["top_logprobs"]
        toks = np.array([entry["token"]["id"] for entry in entries], dtype=np.int64)
        probs = np.exp(np.array([entry["logprob"] for entry in entries], dtype=np.float64))
        probs /= probs.sum()
        dense = np.zeros(vocab, dtype=np.float64)
        dense[toks] = probs
        supports.append((toks, probs, dense))
    return supports


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    base = load_q_dump(args.q_path, args.block, args.vocab)
    n_steps = base.shape[0]
    greedy = load_greedy(args.greedy_json)
    supports = load_target_support(args.target_json, args.vocab)
    if len(greedy) < n_steps + args.block + 1:
        raise ValueError(
            f"need at least {n_steps + args.block + 1} greedy tokens, got {len(greedy)}"
        )
    if len(supports) < n_steps + args.block + 1:
        raise ValueError(
            f"need at least {n_steps + args.block + 1} target steps, got {len(supports)}"
        )

    print(
        f"loaded {args.label}: q={base.shape} pos0={args.pos0} trials={args.trials} "
        f"range={base.min():.2f}..{base.max():.2f}"
    )
    mw1, mw2 = load_markov_weights(args.dspark, args.vocab, args.rank)
    mw2_t = mw2.T

    pos_accept = [0] * args.block
    pos_total = [0] * args.block
    committed_hist: dict[int, int] = {}
    per_step_committed: list[float] = []
    per_step_accepted: list[float] = []

    print()
    print(
        f"{'step':>4} {'anchor_pos':>10} {'anchor_tok':>10} "
        f"{'E[accepted]':>12} {'E[committed]':>13} {'accept[0..4]':>30}"
    )

    for q_idx in range(n_steps):
        step = q_idx + 1
        anchor_pos = args.pos0 + step
        anchor = greedy[step]
        ps = []
        for j in range(args.block):
            tstep = step + 1 + j
            ps.append(supports[tstep] if tstep < len(supports) else None)

        trial_committed = []
        trial_accepted = []
        for _ in range(args.trials):
            prev = anchor
            n_committed = 0
            n_accepted = 0
            for j in range(args.block):
                support = ps[j]
                if support is None:
                    break
                _, _, p_full = support
                bias = mw1[prev] @ mw2_t
                q = softmax((base[q_idx, j].astype(np.float64) + bias.astype(np.float64)))
                x = int(rng.choice(args.vocab, p=q))
                qx = q[x]
                px = p_full[x]
                accept_p = min(1.0, px / qx) if qx > 0 else 0.0
                pos_total[j] += 1
                if rng.random() < accept_p:
                    n_committed += 1
                    n_accepted += 1
                    pos_accept[j] += 1
                    prev = x
                else:
                    n_committed += 1
                    break
            trial_committed.append(n_committed)
            trial_accepted.append(n_accepted)

        avg_committed = float(np.mean(trial_committed))
        avg_accepted = float(np.mean(trial_accepted))
        per_step_committed.append(avg_committed)
        per_step_accepted.append(avg_accepted)
        for value in trial_committed:
            committed_hist[value] = committed_hist.get(value, 0) + 1
        accept_rate = [round(pos_accept[j] / max(pos_total[j], 1), 3) for j in range(args.block)]
        print(
            f"{step:>4} {anchor_pos:>10} {anchor:>10} "
            f"{avg_accepted:>12.3f} {avg_committed:>13.3f} {str(accept_rate):>30}",
            flush=True,
        )

    overall_committed = float(np.mean(per_step_committed))
    overall_accepted = float(np.mean(per_step_accepted))
    ms_per_tok = (args.draft_ms + args.verify_ms) / overall_committed if overall_committed > 0 else None
    speedup = args.plain_ms / ms_per_tok if ms_per_tok and ms_per_tok > 0 else None

    summary = {
        "label": args.label,
        "q_path": os.path.abspath(args.q_path),
        "target_json": os.path.abspath(args.target_json),
        "greedy_json": os.path.abspath(args.greedy_json),
        "dspark": os.path.abspath(args.dspark),
        "pos0": args.pos0,
        "n_steps": n_steps,
        "trials": args.trials,
        "seed": args.seed,
        "block": args.block,
        "vocab": args.vocab,
        "average_committed": overall_committed,
        "average_accepted": overall_accepted,
        "per_step_committed": per_step_committed,
        "per_step_accepted": per_step_accepted,
        "per_position_accept_rate": [
            pos_accept[j] / max(pos_total[j], 1) for j in range(args.block)
        ],
        "committed_hist": {str(k): v for k, v in sorted(committed_hist.items())},
        "draft_ms": args.draft_ms,
        "verify_ms": args.verify_ms,
        "plain_ms": args.plain_ms,
        "ms_per_token": ms_per_tok,
        "speedup": speedup,
    }

    print()
    print(f"=== SUMMARY {args.label} ===")
    print(f"  E[committed/cycle] = {overall_committed:.3f}")
    print(f"  E[accepted/cycle]  = {overall_accepted:.3f}")
    print(f"  committed hist     = {dict(sorted(committed_hist.items()))}")
    print(
        "  per-position accept rate = "
        f"{[round(v, 3) for v in summary['per_position_accept_rate']]}"
    )
    if ms_per_tok is not None and speedup is not None:
        print(
            f"  speedup projection = {speedup:.2f}x "
            f"({100.0 * (speedup - 1.0):+.0f}%) at plain={args.plain_ms:.1f}ms/tok"
        )

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
