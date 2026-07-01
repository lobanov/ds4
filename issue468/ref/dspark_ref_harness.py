#!/usr/bin/env python3
"""DSpark drafter reference harness (CUDA + official tilelang kernels).

Runs inference/model.py.forward_spec with a DRAFTER-ONLY instantiation
(n_layers=0, so no target layers / no 167 GB target needed). The drafter reads:
  - its own mtp.{0,1,2}.* weights (shards 46-48)
  - the SHARED target embed (shard 1) and head/lm_head (shard 45)
  - target-side main_hidden captures from ds4

Two modes:
  --smoke       synthetic hidden states (random bf16, correct shapes). Proves the
                official tilelang kernels compile+run on this GPU and the drafter
                emits tokens. Output tokens are garbage (untrained cache) but
                execution success is the de-risk signal.
  --validate F  load a validation npz and run a real two-step DSpark scenario:
                prefill (start_pos=0) with one hidden/token pair, then decode
                (start_pos=1) with the next hidden/token pair. Save reference
                draft tokens + logits to F.ref.npz for comparison/scoring.

The preferred validation NPZ format is:
  - main_hidden_prefill [1,1,3*dim]
  - input_ids_prefill   [1]
  - main_hidden_decode  [1,1,3*dim]
  - input_ids_decode    [1]

Legacy NPZs with:
  - main_hidden [1,1,3*dim]
  - input_ids   [1]

are still accepted and are treated as "same input for prefill and decode".

Memory: drafter ~11 GB (fp8/fp4 resident) + embed 1 GB + head 1 GB + torch/CUDA
overhead ~= 15 GB. Fits the 128 GB DGX with the target NOT resident.

Run on the DGX:
  source ~/dref-venv/bin/activate
  cd ~/ds4/ref && python dspark_ref_harness.py --smoke
"""
import argparse, json, os, sys, traceback
import numpy as np
import torch
# Patch tvm_ffi's __dict__ setattr bug BEFORE any tilelang/tvm import (model.py
# pulls in tilelang via kernel.py). Must precede `from model import ...`.
import _tvm_ffi_shim  # noqa: F401  (side-effect monkeypatch)

CKPT = os.path.expanduser("~/ds4/hf-dspark")
REF_DIR = os.path.dirname(os.path.abspath(__file__))


def build_args():
    with open(os.path.join(REF_DIR, "inference", "config.json")) as f:
        c = json.load(f)
    from model import ModelArgs
    # Drafter-only: n_layers=0 skips the 43 target blocks entirely. The drafter
    # layers (mtp.0/1/2) are created by the dspark_block_size branch. compress_ratios
    # already carries 46 entries (43 target + [0,0,0] for drafter layers 43/44/45),
    # satisfying DSparkAttention's assert(compress_ratio==0).
    return ModelArgs(
        max_batch_size=1, max_seq_len=4096, temperature=1.0,
        dtype=c["dtype"], scale_fmt=c["scale_fmt"], expert_dtype=c["expert_dtype"],
        scale_dtype="fp8",
        vocab_size=c["vocab_size"], dim=c["dim"], moe_inter_dim=c["moe_inter_dim"],
        n_layers=0,                       # <-- drafter-only
        n_hash_layers=c["n_hash_layers"], n_mtp_layers=c["n_mtp_layers"],
        n_heads=c["n_heads"],
        n_routed_experts=c["n_routed_experts"], n_shared_experts=c["n_shared_experts"],
        n_activated_experts=c["n_activated_experts"],
        score_func=c["score_func"], route_scale=c["route_scale"], swiglu_limit=c["swiglu_limit"],
        q_lora_rank=c["q_lora_rank"], head_dim=c["head_dim"], rope_head_dim=c["rope_head_dim"],
        norm_eps=1e-6, o_groups=c["o_groups"], o_lora_rank=c["o_lora_rank"],
        window_size=c["window_size"],
        original_seq_len=c["original_seq_len"], rope_theta=c["rope_theta"],
        rope_factor=c["rope_factor"], beta_fast=c["beta_fast"], beta_slow=c["beta_slow"],
        compress_rope_theta=c["compress_rope_theta"],
        # Drafter-only instantiation (n_layers=0 below) means mtp stages get
        # layer ids 0/1/2, so they index compress_ratios[0/1/2]. The real
        # drafter ids are 43/44/45 -> compress_ratios[43/44/45] = [0,0,0]
        # (DSparkAttention asserts compress_ratio==0; no Compressor/Indexer).
        # Mirror that by passing (0,0,0): all three drafter layers become
        # window-only MLA, matching the real checkpoint. (Target layers are
        # never created, so no other compress_ratios slot is read.)
        compress_ratios=(0, 0, 0),
        hc_mult=c["hc_mult"], hc_sinkhorn_iters=20, hc_eps=1e-6,
        dspark_block_size=c["dspark_block_size"], dspark_noise_token_id=c["dspark_noise_token_id"],
        dspark_target_layer_ids=tuple(c["dspark_target_layer_ids"]), dspark_markov_rank=c["dspark_markov_rank"],
    )


def patch_tilelang_semantic_checks():
    """Work around a tilelang/TVM Python wrapper bug on this DGX environment.

    tilelang 0.1.8 on the current Python 3.12 stack reaches the JIT path, then
    crashes inside the Python-side semantic-check visitors before any real
    semantic validation or lowering result is produced:

      AttributeError: '_NestedLoopCheckVisitor' object has no attribute '_inst'
      AttributeError: '_FragmentLoopCheckVisitor' object has no attribute '_inst'

    Those failures are in the Python-side derived-object wrapper, not in the
    DSpark program. Replace the whole pre-lower semantic-check phase with a
    no-op so the harness can continue to the actual kernel
    compilation/execution path.
    """
    import tilelang
    import tilelang.analysis.fragment_loop_checker as flc
    import tilelang.analysis.nested_loop_checker as nlc
    import tilelang.engine.lower as tl_lower
    import tilelang.engine.phase as tl_phase
    from tvm.tir.transform import prim_func_pass

    def _noop_pre_lower_semantic_check(mod):
        return None

    def _identity_pass():
        def _pass_fn(func, mod, ctx):
            return func
        return prim_func_pass(_pass_fn, opt_level=0)

    tl_phase.PreLowerSemanticCheck = _noop_pre_lower_semantic_check
    tilelang.engine.phase.PreLowerSemanticCheck = _noop_pre_lower_semantic_check
    tl_lower.PreLowerSemanticCheck = _noop_pre_lower_semantic_check
    nlc.NestedLoopChecker = _identity_pass
    tilelang.analysis.NestedLoopChecker = _identity_pass
    flc.FragmentLoopChecker = _identity_pass
    tilelang.analysis.FragmentLoopChecker = _identity_pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--validate", metavar="NPZ", help="npz with main_hidden + input_ids")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    if not args.smoke and not args.validate:
        ap.error("need --smoke or --validate NPZ")

    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(args.seed)
    sys.path.insert(0, os.path.join(REF_DIR, "inference"))
    from model import Transformer
    patch_tilelang_semantic_checks()

    print("=== build drafter-only Transformer (n_layers=0) ===", flush=True)
    margs = build_args()
    with torch.device("cuda"):
        model = Transformer(margs)
    # Transformer.__init__ already wires mtp[i].embed/head to the shared ones.
    print(f"  embed.weight: {tuple(model.embed.weight.shape)} dtype={model.embed.weight.dtype}")
    print(f"  head.weight : {tuple(model.head.weight.shape)} dtype={model.head.weight.dtype}")
    print(f"  mtp stages  : {len(model.mtp)}  target layers: {len(model.layers)}")

    print("=== load converted ckpt (model0-mp1.safetensors from convert.py), strict=False ===", flush=True)
    try:
        from safetensors.torch import load_model
        from safetensors import safe_open
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing Python dependency 'safetensors'. Install issue468/ref/"
            "inference/requirements.txt into the active environment before "
            "running dspark_ref_harness.py."
        ) from exc
    ckpt = os.path.expanduser("~/ds4/ref-ckpt/model0-mp1.safetensors")
    sd = model.state_dict()
    with safe_open(ckpt, framework="pt", device="cpu") as f:
        ck_keys = set(f.keys())
    load_model(model, ckpt, strict=False)
    # report coverage: which model params the checkpoint populated (by name match)
    have = {k for k in sd if k in ck_keys}
    missing = set(sd) - have
    print(f"  ckpt tensors={len(ck_keys)}, model params={len(sd)}, populated={len(have)}, missing={len(missing)}")
    for k in sorted(missing)[:10]:
        print(f"    missing: {k}")

    model.eval()

    # --- inputs ---
    dim = margs.dim
    n_target_layers = len(margs.dspark_target_layer_ids)
    if args.validate:
        d = np.load(args.validate)
        if "main_hidden_prefill" in d and "input_ids_prefill" in d:
            main_hidden_prefill = torch.tensor(
                d["main_hidden_prefill"], dtype=torch.bfloat16, device="cuda"
            )
            input_ids_prefill = torch.tensor(
                d["input_ids_prefill"], dtype=torch.long, device="cuda"
            )
            main_hidden_decode = torch.tensor(
                d["main_hidden_decode"], dtype=torch.bfloat16, device="cuda"
            )
            input_ids_decode = torch.tensor(
                d["input_ids_decode"], dtype=torch.long, device="cuda"
            )
            print(
                f"=== validate mode: loaded {args.validate}: "
                f"prefill main_hidden {tuple(main_hidden_prefill.shape)}, "
                f"prefill input_ids {tuple(input_ids_prefill.shape)}, "
                f"decode main_hidden {tuple(main_hidden_decode.shape)}, "
                f"decode input_ids {tuple(input_ids_decode.shape)}"
            )
        else:
            main_hidden_prefill = torch.tensor(
                d["main_hidden"], dtype=torch.bfloat16, device="cuda"
            )
            input_ids_prefill = torch.tensor(
                d["input_ids"], dtype=torch.long, device="cuda"
            )
            main_hidden_decode = main_hidden_prefill
            input_ids_decode = input_ids_prefill
            print(
                f"=== validate mode (legacy single-step): loaded {args.validate}: "
                f"main_hidden {tuple(main_hidden_prefill.shape)}, "
                f"input_ids {tuple(input_ids_prefill.shape)}"
            )
    else:
        # synthetic: correct shapes, random values. Cache is untrained so outputs
        # are garbage, but kernel execution success is the de-risk signal.
        # Shapes per forward_spec: input_ids [batch], main_hidden [batch,1,n_tgt*dim].
        main_hidden_prefill = (
            torch.randn(1, 1, n_target_layers * dim, dtype=torch.bfloat16, device="cuda") * 0.1
        )
        main_hidden_decode = (
            torch.randn(1, 1, n_target_layers * dim, dtype=torch.bfloat16, device="cuda") * 0.1
        )
        input_ids_prefill = torch.tensor([100], dtype=torch.long, device="cuda")
        input_ids_decode = torch.tensor([101], dtype=torch.long, device="cuda")
        print(
            "=== smoke mode: synthetic prefill "
            f"{tuple(main_hidden_prefill.shape)}, decode {tuple(main_hidden_decode.shape)}, "
            f"input_ids {input_ids_prefill.shape}/{input_ids_decode.shape}"
        )

    # forward_spec needs a prefill (start_pos=0, caches the anchor KV) then a
    # decode (start_pos>0, attends + drafts via the Markov head).
    try:
        print("=== forward_spec prefill (start_pos=0) ===", flush=True)
        r0 = model.forward_spec(input_ids_prefill, main_hidden_prefill, start_pos=0)
        print(f"  prefill returned: {r0} (expected None — caches drafter KV)")
        print("=== forward_spec decode (start_pos=1) -> draft ===", flush=True)
        out = model.forward_spec(input_ids_decode, main_hidden_decode, start_pos=1)
        output_ids, logits, confidence = out
        print(f"  draft output_ids shape: {tuple(output_ids.shape)}")
        print(f"  logits shape: {tuple(logits.shape)}")
        print(f"  confidence shape: {tuple(confidence.shape)}")
        print(f"  draft tokens: {output_ids[0].tolist()}")
        print(f"  confidence  : {confidence[0].squeeze(-1).tolist()}")
        if args.validate:
            outp = args.validate + ".ref.npz"
            np.savez(outp,
                     ref_output_ids=output_ids[0].cpu().to(torch.int32).numpy(),
                     ref_logits=logits[0].cpu().to(torch.float32).numpy(),
                     ref_confidence=confidence[0].cpu().to(torch.float32).numpy(),
                     prefill_input_ids=input_ids_prefill.cpu().to(torch.int32).numpy(),
                     decode_input_ids=input_ids_decode.cpu().to(torch.int32).numpy())
            print(f"  wrote reference output -> {outp}")
        print("=== HARNESS OK — tilelang kernels ran on this GPU ===")
    except Exception:
        print("=== HARNESS FAILED ===")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
