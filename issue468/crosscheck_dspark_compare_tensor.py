#!/usr/bin/env python3
"""Authoritative DSpark GGUF conversion validation via gguf-tools' --compare-tensor.

Reuses deepseek4-quantize's own `--compare-tensor` mode (the authoritative
byte-compare: regenerates a tensor fresh from HF via the converter's read path,
then byte-compares to the stored GGUF bytes using the converter's GGUF reader,
which correctly handles `general.alignment` and per-tensor padding). For lossless
F32/BF16 tensors the regeneration is identity, so `byte_compare: OK` is equivalent
to stored-GGUF == raw-HF-bytes.

This is the gguf-tools-native validator; crosscheck_mtp2.py provides an
independent (converter-independent) direct HF<->GGUF byte-match as a second opinion.

Usage:
  DS4_DSPARK_HF=../ds4/hf-dspark DS4_DSPARK_GGUF=../ds4/gguf/dspark.gguf \
      python3 issue468/crosscheck_dspark_compare_tensor.py
"""
import os, re, struct, subprocess, sys

HF_DIR = os.environ.get("DS4_DSPARK_HF", "../ds4/hf-dspark")
GGUF = os.environ.get("DS4_DSPARK_GGUF", "../ds4/gguf/dspark.gguf")
CONV = "gguf-tools/deepseek4-quantize"
F32, BF16 = 0, 30


def template_tensor_names(path):
    """Return [(name, gguf_type)] from a GGUF metadata section."""
    out = []
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"; f.read(4)
        n_t = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        def s():
            n = struct.unpack("<Q", f.read(8))[0]; return f.read(n).decode()

        def rd_val():
            t = struct.unpack("<I", f.read(4))[0]
            if t == 8: s()
            elif t in (0, 1, 7): f.read(1)
            elif t in (2, 3, 4, 5, 6): f.read(4)
            elif t in (10, 11, 12): f.read(8)
            elif t == 9:
                et = struct.unpack("<I", f.read(4))[0]; nn = struct.unpack("<Q", f.read(8))[0]
                for _ in range(nn):
                    if et == 8: s()
                    elif et in (0, 1, 7): f.read(1)
                    elif et in (2, 3, 4, 5, 6): f.read(4)
                    elif et in (10, 11, 12): f.read(8)

        for _ in range(n_kv):
            s(); rd_val()
        for _ in range(n_t):
            nm = s()
            nd = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
            tt = struct.unpack("<I", f.read(4))[0]
            off = struct.unpack("<Q", f.read(8))[0]
            out.append((nm, tt))
    return out


def main():
    tens = template_tensor_names(GGUF)
    # Validate every F32/BF16 non-expert tensor (lossless round-trip). Experts
    # (FP4->Q4_K) and FP8->Q8_0 projections are dequant+requant by construction
    # and are validated by the Phase-4 drafter-forward regression gate.
    targets = [nm for nm, tt in tens if tt in (F32, BF16) and "_exps." not in nm]
    print(f"validating {len(targets)} F32/BF16 non-expert tensors via --compare-tensor")
    print(f"  gguf: {GGUF}")
    print(f"  hf:   {HF_DIR}")
    from collections import defaultdict
    by_layer = defaultdict(lambda: {"ok": 0, "fail": 0})
    fails = []
    for nm in targets:
        r = subprocess.run(
            [CONV, "--hf", HF_DIR, "--template", GGUF, "--compare-tensor", nm],
            capture_output=True, text=True,
        )
        m = re.search(r"byte_compare:\s*(\S+)", r.stdout)
        status = m.group(1) if m else f"NO-OUTPUT(rc={r.returncode})"
        L = nm.split(".")[1]
        if status == "OK":
            by_layer[L]["ok"] += 1
        else:
            by_layer[L]["fail"] += 1
            fails.append((nm, status, r.stderr.strip().splitlines()[-1] if r.stderr else ""))
    print("\nper-layer (dspark.gguf self round-trip):")
    for L in sorted(by_layer):
        d = by_layer[L]
        print(f"  mtp.{L}: {d['ok']} OK, {d['fail']} fail")
    tot_ok = sum(d["ok"] for d in by_layer.values())
    tot_fail = sum(d["fail"] for d in by_layer.values())
    print(f"\ntotal: {tot_ok} OK, {tot_fail} fail (of {len(targets)})")
    if fails:
        print("FAILURES:")
        for nm, st, err in fails:
            print(f"  {nm}: {st}  {err}")
        sys.exit(1)
    print("\nCOMPARE-TENSOR VALIDATION PASSED")
    print("  All F32/BF16 non-expert tensors round-trip byte-exact (stored == regenerated-from-HF).")


if __name__ == "__main__":
    main()
