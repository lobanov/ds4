## Verdict On The Revised Verdict

**HOLD is justified, but the revised verdict still overclaims.**

It is **not a GO**: MoE equality alone does not prove full verifier exactness, and the speed cost of exact attention/HC/compressor paths is still unmeasured.

It is **not a NO-GO**: existing byte arithmetic makes the “decode bandwidth may be ≤250 GB/s” cliff unlikely.

The main correction: the two proposed measurements are useful, but **not decisive**. The decisive test is a retained K=4 verifier that is both bit-exact and measures `verify_ms(4) <= 50.45 ms` under deployed accounting.

## Can Decode Bandwidth Be Resolved From Existing Data?

Mostly yes. Conservative byte lower bound from existing artifacts:

```text
mapped model: 82697.67 MiB = 80.76 GiB = 86.71 GB
full routed experts: 72.56 GiB
dense/non-routed: 80.76 - 72.56 = 8.20 GiB
selected routed per decode: 72.56 * 6 / 256 = 1.70 GiB
KV, DS4 compressed 512 fp16 values/layer:
  ctx 128   -> 0.007 GiB
  ctx 8192  -> 0.477 GiB
  ctx 16384 -> 0.953 GiB
```

So per-token bytes are at least:

```text
no KV:   8.20 + 1.70 = 9.91 GiB -> 381 GiB/s = 409 GB/s at 26 ms
8k KV:   10.38 GiB             -> 399 GiB/s = 429 GB/s
16k KV:  10.85 GiB             -> 417 GiB/s = 448 GB/s
```

This is **clearly ≥300 GB/s**, not near 250. The revised verdict’s decode-bandwidth uncertainty is too pessimistic as a reason for HOLD. Caveat: this only resolves the decode-side bandwidth premise, not whether a future K=4 bit-exact fused verifier reaches that efficiency.

## Are The Two Measurements Decisive?

No.

1. **Decode bandwidth ≥275 GB/s** does not prove the fused K=4 verifier reaches that bandwidth. Gate A’s K3→K4 vs K4→K5 slope problem remains.
2. **Identical-input MoE equality** only settles gate/up/MoE kernel equality. It does not settle HC/compressor/attention exactness.
3. Exactness fixes may have speed cost: F32/Q8_0 substitutions, decode-order attention, or per-token attention could push `verify_ms(4)` back above 50.45 ms.
4. The actual decisive measurement is: **bit-exact K=4 verifier timing**, not separate decode bandwidth plus MoE equality.

## The Pos-61 Clean-Input Question

The pos-61 comparison is **not clean for MoE**.

Batch row0 is the aligned row: row1 is obviously a different token. But row0 already differs before MoE:

```text
attn_norm row0 max diff: 0.00467
hc_attn_pre row0 max diff: 0.0860
KVcur row0 max diff: 0.125
ffn_norm row0 max diff: 0.01413
```

Then MoE internals diverge:

```text
gate max diff: 6.736
up max diff:   7.553
moe_out diff:  0.01765
topk: identical [186,104,22,105,195,202]
```

So “gate/up outputs diverge because inherited upstream input diverged” is plausible. The dist-probe does **not** prove a real gate/up reduction diff. But the verdict should not call the pos-61 measurement “identical-input” evidence.

## Remaining Overclaim / Sensitivity

Robust:
- `verify_ms(4) <= 50.45 ms` threshold is arithmetically sound for `decode_ms=26`, `E=2.1976`, `S=0.34019`.
- Host overhead is small; `layer_execute` dominates.

Overclaimed:
- “Phase-B targeted the wrong thing” is too strong. If Phase-B targeted exactness-only gate/up, yes. If it targeted expert-load bandwidth sharing, that is still directly relevant.
- “Two decisive measurements” is false; they are necessary-ish probes, not sufficient.

Second-tightest sensitivity after verifier bandwidth is **acceptance / decode_ms**:
- `decode_ms ±10%` shifts threshold by about `±6.0 ms`.
- `E[a|4] ±10%` shifts threshold by about `±4.8 ms`.
- At `verify=46.6 ms`, pooled speed is `1.270x`; even dolly is `1.227x`.
- At `verify=50.0 ms`, pooled barely clears `1.208x`, but dolly fails at `1.165x`.

## Leading Assessment

**HOLD remains the right verdict**, but not because decode bandwidth is unknowable. Existing bytes imply decode is already comfortably above 300 GB/s. HOLD survives because full bit-exact verifier cost is still unmeasured.

The one test that moves it: run the actual bit-exact K=4 verifier path and profile it. `<=50.45 ms` moves HOLD toward GO; `>50.45 ms` moves it to NO-GO.