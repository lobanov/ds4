#!/usr/bin/env python3
"""Score a capture bundle with the torch-fallback FP8 reference path.

This runs the official reference-model structure with the torch fallback kernel
across a full captured decode bundle, preserving DSpark KV state as start_pos
advances. It mirrors the existing acceptance methodology closely enough to
compare source-FP8 against the baseline/candidate B2 summaries on the same
bundle.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

from dspark_ref_harness_torch import REF_DIR, build_args, install_kernel_fallback


HC = 4
DIM = 4096
LAYERS = (40, 41, 42)
BLOCK = 5
POS_RE = re.compile(r"hc_dspark_main_hc-\d+_pos(\d+)\.bin$")


def load_greedy_tokens(path: Path) -> list[int]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [int(v) for v in data]
    if isinstance(data, dict) and "steps" in data:
        return [int(step["selected"]["id"]) for step in data["steps"]]
    raise ValueError(f"unsupported greedy token format: {path}")


def infer_pos0(bundle: Path) -> int:
    positions: list[int] = []
    for name in os.listdir(bundle):
        match = POS_RE.match(name)
        if match:
            positions.append(int(match.group(1)))
    if not positions:
        raise FileNotFoundError(f"no hc_dspark_main_hc-* files found in {bundle}")
    return min(positions)


def load_main_hidden(bundle: Path, pos: int) -> np.ndarray:
    parts = []
    for layer in LAYERS:
        path = bundle / f"hc_dspark_main_hc-{layer}_pos{pos}.bin"
        arr = np.fromfile(path, dtype=np.float32)
        if arr.size != HC * DIM:
            raise ValueError(f"unexpected size for {path}: got {arr.size}, want {HC * DIM}")
        parts.append(arr.reshape(HC, DIM).mean(axis=0))
    return np.concatenate(parts, axis=0).astype(np.float32).reshape(1, 1, len(LAYERS) * DIM)


def load_target_steps(path: Path) -> tuple[int, list[dict]]:
    data = json.loads(path.read_text())
    return int(data["prompt_tokens"]), data["steps"]


def target_dist(step_entry: dict, vocab: int) -> np.ndarray:
    probs = np.zeros(vocab, dtype=np.float64)
    toks = np.array([entry["token"]["id"] for entry in step_entry["top_logprobs"]], dtype=np.int64)
    vals = np.exp(np.array([entry["logprob"] for entry in step_entry["top_logprobs"]], dtype=np.float64))
    vals /= vals.sum()
    probs[toks] = vals
    return probs


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max()
    probs = np.exp(logits, dtype=np.float64)
    return probs / probs.sum()


def build_model(temperature: float):
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    install_kernel_fallback()
    sys.path.insert(0, os.path.join(REF_DIR, "inference"))
    from model import Transformer
    from safetensors.torch import load_model
    from safetensors import safe_open

    margs = build_args(temperature)
    with torch.device("cuda"):
        model = Transformer(margs)

    ckpt = os.path.expanduser("~/ds4/ref-ckpt/model0-mp1.safetensors")
    state_dict = model.state_dict()
    with safe_open(ckpt, framework="pt", device="cpu") as f:
        ck_keys = set(f.keys())
    load_model(model, ckpt, strict=False)
    missing = sorted(set(state_dict) - {k for k in state_dict if k in ck_keys})
    return model.eval(), missing


def forward_step_base_logits(model, input_ids: torch.Tensor, main_hidden: torch.Tensor, start_pos: int) -> np.ndarray | None:
    h, main_x = model.mtp[0].forward_embed(main_hidden, input_ids)
    for layer in model.mtp:
        h = layer(h, start_pos, input_ids, main_x)
    if start_pos == 0:
        return None
    stage = model.mtp[-1]
    h = stage.hc_head(h, stage.hc_head_fn, stage.hc_head_scale, stage.hc_head_base)
    logits = stage.head(stage.norm(h), full_logits=True)
    return logits[0].detach().to(torch.float32).cpu().numpy()


def markov_bias_tables(model) -> tuple[np.ndarray, np.ndarray]:
    stage = model.mtp[-1].markov_head
    w1 = stage.markov_w1.weight.detach().to(torch.float32).cpu().numpy()
    w2 = stage.markov_w2.weight.detach().to(torch.float32).cpu().numpy()
    return w1, w2


def greedy_chain(base_logits: np.ndarray, anchor: int, markov_w1: np.ndarray, markov_w2: np.ndarray) -> tuple[list[int], list[np.ndarray]]:
    prev = anchor
    sampled = [anchor]
    q_logits = []
    for pos in range(base_logits.shape[0]):
        bias = markov_w2 @ markov_w1[prev]
        logits = base_logits[pos].astype(np.float64) + bias.astype(np.float64)
        q_logits.append(logits)
        prev = int(np.argmax(logits))
        sampled.append(prev)
    return sampled, q_logits


def score_bundle(
    bundle: Path,
    model,
    *,
    trials: int,
    steps_cap: int,
    seed: int,
) -> dict:
    greedy_tokens = load_greedy_tokens(bundle / "target_greedy.json")
    prompt_tokens, target_steps = load_target_steps(bundle / "target_topk.json")
    pos0 = infer_pos0(bundle)
    n_steps = min(len(greedy_tokens) - 1 - BLOCK, len(target_steps) - 1 - BLOCK)
    if steps_cap > 0:
        n_steps = min(n_steps, steps_cap)
    if n_steps <= 0:
        raise RuntimeError(f"bundle {bundle} has no scoreable steps")

    rng = np.random.default_rng(seed)
    markov_w1, markov_w2 = markov_bias_tables(model)
    vocab = markov_w2.shape[0]
    base_logits_cache: dict[int, np.ndarray] = {}

    prefill_hidden = torch.tensor(load_main_hidden(bundle, pos0), dtype=torch.bfloat16, device="cuda")
    prefill_input = torch.tensor([greedy_tokens[0]], dtype=torch.long, device="cuda")
    forward_step_base_logits(model, prefill_input, prefill_hidden, 0)

    for step in range(1, n_steps + 1):
        hidden = torch.tensor(load_main_hidden(bundle, pos0 + step), dtype=torch.bfloat16, device="cuda")
        token = torch.tensor([greedy_tokens[step]], dtype=torch.long, device="cuda")
        base_logits_cache[step] = forward_step_base_logits(model, token, hidden, step)

    per_step_accepted: list[float] = []
    per_step_committed: list[float] = []
    per_position_accept = np.zeros(BLOCK, dtype=np.float64)
    per_position_total = np.zeros(BLOCK, dtype=np.float64)
    committed_hist: dict[int, int] = {}
    greedy_hist: dict[int, int] = {}
    analytical_accept_by_pos: list[list[float]] = [[] for _ in range(BLOCK)]

    for step in range(1, n_steps + 1):
        anchor = greedy_tokens[step]
        truth = greedy_tokens[step + 1 : step + 1 + BLOCK]
        base_logits = base_logits_cache[step]
        greedy_output, _ = greedy_chain(base_logits, anchor, markov_w1, markov_w2)

        greedy_prefix = 0
        for idx, tok in enumerate(greedy_output[1:]):
            if tok != truth[idx]:
                break
            greedy_prefix += 1
        greedy_hist[greedy_prefix] = greedy_hist.get(greedy_prefix, 0) + 1

        prev_truth = anchor
        for pos in range(BLOCK):
            p = target_dist(target_steps[step + 1 + pos], vocab)
            q = softmax(base_logits[pos].astype(np.float64) + (markov_w2 @ markov_w1[prev_truth]).astype(np.float64))
            tv_accept = 1.0 - 0.5 * np.abs(p - q).sum()
            analytical_accept_by_pos[pos].append(float(tv_accept))
            prev_truth = truth[pos]

        step_accepted = 0.0
        step_committed = 0.0
        for _ in range(trials):
            prev = anchor
            accepted = 0
            committed = 0
            for pos in range(BLOCK):
                q = softmax(base_logits[pos].astype(np.float64) + (markov_w2 @ markov_w1[prev]).astype(np.float64))
                sampled = int(rng.choice(vocab, p=q))
                p = target_dist(target_steps[step + 1 + pos], vocab)
                qx = q[sampled]
                px = p[sampled]
                accept_prob = min(1.0, px / qx) if qx > 0 else 0.0
                per_position_total[pos] += 1
                if rng.random() < accept_prob:
                    accepted += 1
                    committed += 1
                    per_position_accept[pos] += 1
                    prev = sampled
                    continue
                committed += 1
                break
            step_accepted += accepted
            step_committed += committed
            committed_hist[committed] = committed_hist.get(committed, 0) + 1
        per_step_accepted.append(step_accepted / trials)
        per_step_committed.append(step_committed / trials)

    avg_accepted = float(sum(per_step_accepted) / len(per_step_accepted))
    avg_committed = float(sum(per_step_committed) / len(per_step_committed))
    greedy_avg_prefix = float(sum(k * v for k, v in greedy_hist.items()) / sum(greedy_hist.values()))
    analytical_means = [float(sum(vals) / len(vals)) if vals else 0.0 for vals in analytical_accept_by_pos]
    analytical_prefix = 0.0
    reach = 1.0
    for val in analytical_means:
        analytical_prefix += reach * val
        reach *= val

    return {
        "prompt_tokens": prompt_tokens,
        "pos0": pos0,
        "n_steps": n_steps,
        "trials": trials,
        "seed": seed,
        "block": BLOCK,
        "vocab": vocab,
        "average_committed": avg_committed,
        "average_accepted": avg_accepted,
        "per_step_committed": per_step_committed,
        "per_step_accepted": per_step_accepted,
        "per_position_accept_rate": [
            float(per_position_accept[idx] / max(per_position_total[idx], 1.0)) for idx in range(BLOCK)
        ],
        "committed_hist": {str(k): v for k, v in sorted(committed_hist.items())},
        "greedy_avg_prefix": greedy_avg_prefix,
        "greedy_hist": {str(k): v for k, v in sorted(greedy_hist.items())},
        "analytical_accept_upper_bound_by_pos": analytical_means,
        "analytical_committed_upper_bound": float(analytical_prefix + 1.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--label", default="ref-fp8-torch")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--trials", type=int, default=128)
    ap.add_argument("--steps-cap", type=int, default=19)
    ap.add_argument("--seed", type=int, default=20260701)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model, missing = build_model(args.temperature)
    summary = score_bundle(
        Path(args.capture_dir),
        model,
        trials=args.trials,
        steps_cap=args.steps_cap,
        seed=args.seed,
    )
    summary["label"] = args.label
    summary["capture_dir"] = str(Path(args.capture_dir).resolve())
    summary["missing_checkpoint_keys"] = missing[:10]
    out_path = Path(args.out_json)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    print(
        f"avg_committed={summary['average_committed']:.4f} "
        f"avg_accepted={summary['average_accepted']:.4f} "
        f"greedy_avg_prefix={summary['greedy_avg_prefix']:.4f} "
        f"analytic_upper={summary['analytical_committed_upper_bound']:.4f}"
    )


if __name__ == "__main__":
    main()
