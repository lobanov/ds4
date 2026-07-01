#!/usr/bin/env python3
"""Torch-fallback DSpark drafter reference harness.

Loads the official reference model code but injects torch implementations for
the tilelang `kernel` module so the source FP8/FP4 checkpoint can be scored on
captured acceptance bundles without depending on the broken tilelang stack.
"""
import argparse
import importlib.util
import json
import os
import sys
import traceback

import numpy as np
import torch


CKPT = os.path.expanduser("~/ds4/hf-dspark")
REF_DIR = os.path.dirname(os.path.abspath(__file__))


def install_kernel_fallback():
    path = os.path.join(REF_DIR, "kernel_torch_fallback.py")
    spec = importlib.util.spec_from_file_location("kernel", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load torch fallback kernel module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["kernel"] = module


def build_args(temperature: float):
    with open(os.path.join(REF_DIR, "inference", "config.json")) as f:
        c = json.load(f)
    from model import ModelArgs

    return ModelArgs(
        max_batch_size=1,
        max_seq_len=4096,
        temperature=temperature,
        dtype=c["dtype"],
        scale_fmt=c["scale_fmt"],
        expert_dtype=c["expert_dtype"],
        scale_dtype="fp8",
        vocab_size=c["vocab_size"],
        dim=c["dim"],
        moe_inter_dim=c["moe_inter_dim"],
        n_layers=0,
        n_hash_layers=c["n_hash_layers"],
        n_mtp_layers=c["n_mtp_layers"],
        n_heads=c["n_heads"],
        n_routed_experts=c["n_routed_experts"],
        n_shared_experts=c["n_shared_experts"],
        n_activated_experts=c["n_activated_experts"],
        score_func=c["score_func"],
        route_scale=c["route_scale"],
        swiglu_limit=c["swiglu_limit"],
        q_lora_rank=c["q_lora_rank"],
        head_dim=c["head_dim"],
        rope_head_dim=c["rope_head_dim"],
        norm_eps=1e-6,
        o_groups=c["o_groups"],
        o_lora_rank=c["o_lora_rank"],
        window_size=c["window_size"],
        original_seq_len=c["original_seq_len"],
        rope_theta=c["rope_theta"],
        rope_factor=c["rope_factor"],
        beta_fast=c["beta_fast"],
        beta_slow=c["beta_slow"],
        compress_rope_theta=c["compress_rope_theta"],
        compress_ratios=(0, 0, 0),
        hc_mult=c["hc_mult"],
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        dspark_block_size=c["dspark_block_size"],
        dspark_noise_token_id=c["dspark_noise_token_id"],
        dspark_target_layer_ids=tuple(c["dspark_target_layer_ids"]),
        dspark_markov_rank=c["dspark_markov_rank"],
    )


def load_validate_inputs(path: str):
    d = np.load(path)
    if "main_hidden_prefill" in d and "input_ids_prefill" in d:
        return (
            torch.tensor(d["main_hidden_prefill"], dtype=torch.bfloat16, device="cuda"),
            torch.tensor(d["input_ids_prefill"], dtype=torch.long, device="cuda"),
            torch.tensor(d["main_hidden_decode"], dtype=torch.bfloat16, device="cuda"),
            torch.tensor(d["input_ids_decode"], dtype=torch.long, device="cuda"),
        )
    main_hidden = torch.tensor(d["main_hidden"], dtype=torch.bfloat16, device="cuda")
    input_ids = torch.tensor(d["input_ids"], dtype=torch.long, device="cuda")
    return main_hidden, input_ids, main_hidden, input_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--validate", metavar="NPZ", help="npz with main_hidden + input_ids")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()
    if not args.smoke and not args.validate:
        ap.error("need --smoke or --validate NPZ")

    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(args.seed)
    install_kernel_fallback()
    sys.path.insert(0, os.path.join(REF_DIR, "inference"))
    from model import Transformer

    print("=== build drafter-only Transformer (torch fallback kernel) ===", flush=True)
    margs = build_args(args.temperature)
    with torch.device("cuda"):
        model = Transformer(margs)

    try:
        from safetensors.torch import load_model
        from safetensors import safe_open
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing Python dependency 'safetensors'. Install issue468/ref/inference/requirements.txt "
            "into the active environment before running dspark_ref_harness_torch.py."
        ) from exc

    ckpt = os.path.expanduser("~/ds4/ref-ckpt/model0-mp1.safetensors")
    sd = model.state_dict()
    with safe_open(ckpt, framework="pt", device="cpu") as f:
        ck_keys = set(f.keys())
    load_model(model, ckpt, strict=False)
    have = {k for k in sd if k in ck_keys}
    missing = set(sd) - have
    print(f"  ckpt tensors={len(ck_keys)}, model params={len(sd)}, populated={len(have)}, missing={len(missing)}")
    for key in sorted(missing)[:10]:
        print(f"    missing: {key}")

    model.eval()
    dim = margs.dim
    n_target_layers = len(margs.dspark_target_layer_ids)
    if args.validate:
        main_hidden_prefill, input_ids_prefill, main_hidden_decode, input_ids_decode = load_validate_inputs(args.validate)
        print(f"=== validate mode: loaded {args.validate}")
    else:
        main_hidden_prefill = torch.randn(1, 1, n_target_layers * dim, dtype=torch.bfloat16, device="cuda") * 0.1
        main_hidden_decode = torch.randn(1, 1, n_target_layers * dim, dtype=torch.bfloat16, device="cuda") * 0.1
        input_ids_prefill = torch.tensor([100], dtype=torch.long, device="cuda")
        input_ids_decode = torch.tensor([101], dtype=torch.long, device="cuda")
        print("=== smoke mode: synthetic prefill/decode inputs")

    try:
        print("=== forward_spec prefill (start_pos=0) ===", flush=True)
        r0 = model.forward_spec(input_ids_prefill, main_hidden_prefill, start_pos=0)
        print(f"  prefill returned: {r0}")
        print("=== forward_spec decode (start_pos=1) -> draft ===", flush=True)
        output_ids, logits, confidence = model.forward_spec(input_ids_decode, main_hidden_decode, start_pos=1)
        print(f"  draft output_ids shape: {tuple(output_ids.shape)}")
        print(f"  logits shape: {tuple(logits.shape)}")
        print(f"  confidence shape: {tuple(confidence.shape)}")
        print(f"  draft tokens: {output_ids[0].tolist()}")
        if args.validate:
            outp = args.validate + ".ref_torch.npz"
            np.savez(
                outp,
                ref_output_ids=output_ids[0].cpu().to(torch.int32).numpy(),
                ref_logits=logits[0].cpu().to(torch.float32).numpy(),
                ref_confidence=confidence[0].cpu().to(torch.float32).numpy(),
                prefill_input_ids=input_ids_prefill.cpu().to(torch.int32).numpy(),
                decode_input_ids=input_ids_decode.cpu().to(torch.int32).numpy(),
            )
            print(f"  wrote reference output -> {outp}")
        print("=== HARNESS OK — torch fallback path ran ===")
    except Exception:
        print("=== HARNESS FAILED ===")
        traceback.print_exc()


if __name__ == "__main__":
    main()
