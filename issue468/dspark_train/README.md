# dspark_train — Stage 2 drafter fine-tune PoC (RESEARCH INSTRUMENTATION)

PyTorch (MPS) environment for the bounded drafter fine-tune described in
`issue468/summaries/stage2_finetune_protocol.md`. Not part of the ds4 engine or
the numpy oracle; this is the Stage-2 training/eval harness.

## Setup

```bash
cd issue468/dspark_train
python3 -m venv .venv            # CPython 3.14 (torch cp314 wheel includes MPS)
source .venv/bin/activate
pip install -r requirements.txt
```

Verify MPS:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"   # True
```

## Scope

- Torch port of the drafter forward (reference: `issue468/dspark_oracle/forward.py`),
  with a fidelity gate (LoRA=0 must reproduce the numpy oracle's draft tokens).
- LoRA fine-tune: Stage 2a head-only (precomputed features), Stage 2b full-drafter.
- Frozen routed MoE experts (dequant to F16) + the target's shared embed/lm_head.
- Reads sharded capture data (few-file, safetensors/.npz) produced by the
  `--capture-dataset` ds4 mode (see the protocol note, Phase A).

## Notes

- The `.venv/` is gitignored (see `issue468/.gitignore`). Do not commit checkpoints
  or large capture shards — keep them gitignored and reference them in
  `issue468/STATUS.md` / inventories.
- 128 GB unified memory on the target machine: the ~23 GB F16 drafter experts,
  activations, and capture shards all fit comfortably.
