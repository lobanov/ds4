#!/usr/bin/env python3
"""Cross-check the DSpark mtp.2 GGUF (converter output) against the HF source.

Iteration-1 validation. The fp4/fp8 dequant math is IDENTICAL to the target's
(already validated by the working target GGUFs); the NEW risk for DSpark is
name-mapping + tensor orientation, which the converter's check_reversed_shape
already validates against real HF shapes. So this script does the one check
check_reversed_shape cannot: a numeric spot-check.

We pick the BF16 markov_head.markov_w1 tensor: BF16 round-trips losslessly
through the converter (read BF16 -> F32 -> write BF16), so the GGUF bytes must
EXACTLY equal the HF source bytes. This validates: name-mapping correctness,
the read path, and orientation — all on real DSpark data.
"""
import struct, json

HF="../ds4/hf-dspark/model-00048-of-00048.safetensors"
GGUF="issue468/baseline/dspark.mtp2.gguf"
VOCAB=129280; MRANK=256

def hf_raw(path, name):
    with open(path,'rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]
        hdr=json.loads(f.read(n))
        base=8+n
        info=hdr[name]
        f.seek(base+info['data_offsets'][0])
        return info, f.read(info['data_offsets'][1]-info['data_offsets'][0])

def gguf_tensor(path, name):
    with open(path,'rb') as f:
        assert f.read(4)==b'GGUF'; f.read(4)
        n_t=struct.unpack('<Q',f.read(8))[0]; n_kv=struct.unpack('<Q',f.read(8))[0]
        def s():
            n=struct.unpack('<Q',f.read(8))[0]; return f.read(n).decode()
        def rd_val():
            t=struct.unpack('<I',f.read(4))[0]
            if t==8: s()
            elif t in(0,1,7): f.read(1)
            elif t in(2,3,4,5,6): f.read(4)
            elif t in(10,11,12): f.read(8)
            elif t==9:
                et=struct.unpack('<I',f.read(4))[0]; nn=struct.unpack('<Q',f.read(8))[0]
                for _ in range(nn):
                    if et==8: s()
                    elif et in(0,1,7): f.read(1)
                    elif et in(2,3,4,5,6): f.read(4)
                    elif et in(10,11,12): f.read(8)
        for _ in range(n_kv): s(); rd_val()
        infos={}
        for _ in range(n_t):
            nm=s(); nd=struct.unpack('<I',f.read(4))[0]
            dims=[struct.unpack('<Q',f.read(8))[0] for _ in range(nd)]
            tt=struct.unpack('<I',f.read(4))[0]; off=struct.unpack('<Q',f.read(8))[0]
            infos[nm]=(dims,tt,off)
        meta_end=f.tell()
        data_off=((meta_end+31)//32)*32
        dims,tt,off=infos[name]
        f.seek(data_off+off)
        nE=1
        for d in dims: nE*=d
        return dims,tt, f.read(nE*2)  # BF16 = 2 bytes/elem

def main():
    print(f"cross-check: {GGUF} vs {HF}")
    hinfo,hraw = hf_raw(HF,"mtp.2.markov_head.markov_w1.weight")
    print(f"  HF markov_w1: dtype={hinfo['dtype']} shape={hinfo['shape']} bytes={len(hraw)}")
    gdims,gtt,graw = gguf_tensor(GGUF,"mtp.2.markov_head.markov_w1.weight")
    print(f"  GGUF markov_w1: type={gtt}(BF16=30) dims={gdims} bytes={len(graw)}")
    assert hinfo['shape']==[VOCAB,MRANK], f"unexpected HF shape {hinfo['shape']}"
    assert gtt==30, f"unexpected GGUF type {gtt}"
    # GGUF dims should be reversed HF shape (per check_reversed_shape)
    assert gdims==[MRANK,VOCAB], f"unexpected GGUF dims {gdims} (want reversed=[{MRANK},{VOCAB}])"
    # BF16 round-trip is lossless: GGUF bytes (laid out in GGUF ne order) must equal
    # HF bytes (laid out in torch shape order). For a 2D [out,in] tensor, GGUF ne=[in,out]
    # stores col-major (in fastest), HF stores row-major (in fastest). Both have 'in' as
    # the fastest-varying dim -> byte layout IDENTICAL. So raw bytes must match exactly.
    if hraw==graw:
        print(f"  EXACT BYTE MATCH: {len(hraw)} bytes identical -> name-mapping + read + orientation OK")
    else:
        mm=sum(1 for a,b in zip(hraw,graw) if a!=b)
        print(f"  MISMATCH: {mm}/{len(hraw)} bytes differ (first diff at byte {next(i for i,(a,b) in enumerate(zip(hraw,graw)) if a!=b)})")

if __name__=="__main__":
    main()
