#!/usr/bin/env python3
"""Bake the conf_proj calibration LoRA delta into the dspark GGUF.
The conf_proj (mtp.2.confidence_head.proj.weight, [4352], BF16) += the delta (F32),
written back as BF16 (round-to-nearest-even). Drop-in dspark_lora_conf.gguf."""
import sys, shutil, numpy as np
sys.path.insert(0, 'issue468/dspark_oracle')
from gguf_loader import index_gguf, read_tensor

DSPARK = '/Users/lobanov/Projects/ds4/gguf/dspark.gguf'
DELTA = '/tmp/confproj_lora_delta.npy'
OUT = __import__('os').environ.get('OUT_GGUF', '/Users/lobanov/Projects/ds4/gguf/dspark_lora_conf.gguf')
TENSOR = 'mtp.2.confidence_head.proj.weight'

def f32_to_bf16_bits(arr_f32):
    x = np.ascontiguousarray(arr_f32, dtype=np.float32).view(np.uint32)
    lsb = (x >> 16) & 1
    round_bias = np.uint32(0x7FFF) + lsb
    return ((x + round_bias) >> 16).astype(np.uint16)

delta = np.load(DELTA).astype(np.float32)
print(f'delta: shape={delta.shape} abs_mean={np.abs(delta).mean():.6f} max={np.abs(delta).max():.4f}', flush=True)

_, ginfos, gdoff = index_gguf(DSPARK)
ne, dt, off = ginfos[TENSOR]
abs_off = gdoff + off
print(f'{TENSOR}: ne={ne} dtype={dt} abs_off={abs_off}', flush=True)
assert dt == 30, f'expected BF16 (30), got {dt}'
assert list(ne) == [4352], f'expected [4352], got {ne}'

current = read_tensor(DSPARK, ginfos, gdoff, TENSOR).astype(np.float32)
print(f'current conf_proj: abs_mean={np.abs(current).mean():.6f} max={np.abs(current).max():.4f} std={current.std():.6f}', flush=True)

modified = current + delta
rel = np.abs(delta).mean() / np.abs(current).mean()
print(f'modified: rel perturbation = {rel*100:.2f}% of |conf_proj|', flush=True)

# copy + bake
print(f'copying {DSPARK} -> {OUT}', flush=True)
shutil.copy2(DSPARK, OUT)
bf16_bits = f32_to_bf16_bits(modified)
with open(OUT, 'r+b') as f:
    f.seek(abs_off)
    f.write(bf16_bits.tobytes())

# verify: read back + check the delta applied
_, ginfos2, gdoff2 = index_gguf(OUT)
check = read_tensor(OUT, ginfos2, gdoff2, TENSOR).astype(np.float32)
resid = check - modified
print(f'verify: ‖check - modified‖={np.linalg.norm(resid):.6f} (BF16 rounding residual)', flush=True)
print(f'verify: ‖check - current‖={np.linalg.norm(check - current):.4f} (should be ~‖delta‖={np.linalg.norm(delta):.4f})', flush=True)
print(f'\nbaked -> {OUT}', flush=True)
print('Next: ds4-spec-bench --dspark dspark_lora_conf.gguf (M3 stack, STS ON, th=0.15)', flush=True)
