#!/usr/bin/env python3
"""Bake the hc_fn REINFORCE LoRA delta into the conf_proj-baked GGUF (dspark_lora_conf.gguf)
-> dspark_lora_conf_hc.gguf (the combined conf_proj + hc_fn deployment).
The hc_head_fn (mtp.2.hc_head_fn.weight, [4,16384], F16) += the delta (F32) -> F16."""
import sys, shutil, numpy as np
sys.path.insert(0, 'issue468/dspark_oracle')
from gguf_loader import index_gguf, read_tensor

SRC = '/Users/lobanov/Projects/ds4/gguf/dspark_lora_conf.gguf'  # already has conf_proj LoRA
DELTA = '/tmp/hcfn_lora_delta.npy'
OUT = '/Users/lobanov/Projects/ds4/gguf/dspark_lora_conf_hc.gguf'
TENSOR = 'mtp.2.hc_head_fn.weight'

def f32_to_f16(arr_f32):
    return np.ascontiguousarray(arr_f32, dtype=np.float16)

delta = np.load(DELTA).astype(np.float32)
print(f'hc_fn delta: shape={delta.shape} ‖d‖={np.linalg.norm(delta):.4f} max={np.abs(delta).max():.4f}', flush=True)

_, ginfos, gdoff = index_gguf(SRC)
ne, dt, off = ginfos[TENSOR]; abs_off = gdoff + off
print(f'{TENSOR}: ne={ne} dtype={dt} abs_off={abs_off}', flush=True)
current = read_tensor(SRC, ginfos, gdoff, TENSOR).astype(np.float32)
print(f'current: shape={current.shape} ‖c‖={np.linalg.norm(current):.4f}', flush=True)
modified = current + delta
print(f'modified: rel perturbation = {100*np.abs(delta).mean()/np.abs(current).mean():.2f}% of |hc_fn|', flush=True)

print(f'copying {SRC} -> {OUT}', flush=True)
shutil.copy2(SRC, OUT)
with open(OUT, 'r+b') as f:
    f.seek(abs_off); f.write(f32_to_f16(modified).tobytes())
# verify
_, g2, d2 = index_gguf(OUT)
check = read_tensor(OUT, g2, d2, TENSOR).astype(np.float32)
print(f'verify: ‖check-current‖={np.linalg.norm(check-current):.4f} (~‖delta‖={np.linalg.norm(delta):.4f})', flush=True)
print(f'baked -> {OUT}', flush=True)
