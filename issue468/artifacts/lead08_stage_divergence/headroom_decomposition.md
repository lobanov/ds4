# Lead 08 — verify headroom decomposition (8k corpus)

Date: 2026-07-13. Task: `headroom-decomposition`.
Source: `issue468/artifacts/mtp_phaseA_profile/summary.json` (the retained
`DS4_MTP_VERIFY_PROFILE` instrumentation — reused, not re-run; reproduces the known
Phase A headroom). Method: per-stage medians for `code_8k` K=3,4,5 + marginal-bandwidth
analysis from the K-slope.

## Headline

The ~19–22 ms/cycle verify headroom is **almost entirely GPU `layer_execute` bandwidth
inefficiency**, not host-side launch/encode overhead or readback. The verify path runs
the routed-expert stream at ~190 GB/s vs the decode path's ~300 GB/s effective; if a
fused/tuned kernel reached decode bandwidth, `verify_ms(4)` would fall ~66 → ~43 ms.

## Per-stage breakdown (code_8k medians)

| K | verify_ms | layer_execute | host overhead | selected GiB |
|---:|---:|---:|---:|---:|
| 3 | 59.51 | 56.46 | 3.04 | 4.25 |
| 4 | 65.91 | 62.59 | 3.33 | 5.39 |
| 5 | 76.66 | 72.64 | 3.85 | 5.95 |

- `layer_execute` (GPU, 61 layers) = **~95%** of verify_ms.
- host overhead = `upload + layer_encode + head_encode + head_execute + top_read + logits_read` ≈ **3–4 ms (~5%)**; `top_read`/`logits_read` ≈ 0.
- Fit: `verify_ms(K) ≈ 40 + 6.6·K` (8k) — reproduces Phase A; headroom vs the 45 ms target at K=4 = **20.9 ms** (in the Phase A 19–22 ms band). ✓

## Component decomposition of the headroom

| Component | K=4 estimate | Recoverable by fusion? |
|---|---:|---|
| Host-side launch/encode/readback | ~3 ms | yes (fewer kernel launches) — small |
| **GPU `layer_execute` bandwidth inefficiency** | **~20 ms** | **yes (tuned/fused kernels → decode bw)** — the bulk |
| Readback (top/logits) | ~0 ms | n/a |
| **Total headroom vs 45 ms target** | **~21 ms** | — |

The bulk is **not** host overhead — it is the GPU running the batched verify kernels at
~63% of decode's per-byte efficiency.

## Bandwidth analysis (the recoverable term)

- Expert-stream marginal bandwidth (verify path), from the K3→K4 slope:
  Δselected = 1.14 GiB, Δlayer_execute = 6.14 ms → **~190 GB/s**.
- Decode effective bandwidth: **~300 GB/s** (Lead 08 doc; corroborated by the
  `code_4k` K=2 verify ≈ 1.05× decode point — the floor is reachable at low K).
- So the verify path is at ~190/300 ≈ **63%** of decode bandwidth on the expert stream.
- **Floor projection:** if a fused/tuned kernel reached decode bandwidth across
  `layer_execute`, `layer_execute(K=4)` → 62.6 × 190/300 ≈ **39.7 ms**, giving
  `verify_ms(4)` ≈ **43 ms** (recover ~23 ms). That is **at/below the 45 ms floor target.**

| K | verify_ms (now) | projected @ decode bw | recoverable |
|---:|---:|---:|---:|
| 3 | 59.5 | ~39 | ~21 |
| 4 | 65.9 | ~43 | ~23 |
| 5 | 76.7 | ~50 | ~27 |

## Recoverable estimate + key uncertainty

**Recoverable: ~20–25 ms/cycle at K=4** (the GPU bandwidth gap), projecting `verify_ms(4)`
66 → ~43 ms — which the speedup model turns into ~1.3× (clears the +20% gate, per
`spec_speedup_model.md` with anchor-reuse + GPU drafter + oracle acceptance).

**Key uncertainty (load-bearing for the verdict):** whether a multi-token (M=K) fused
kernel can actually reach decode's ~300 GB/s. The expert stream at ~190 GB/s is the
batched-kernel efficiency; the `code_4k` K=2 ≈ decode point shows it is reachable at low K,
but K=4 reachability is unproven. The 300 GB/s decode figure itself is the Lead 08 doc's
assertion (not directly re-measured here) — a direct decode-bandwidth measurement would
firm it up, but the order-of-magnitude (verify ~0.6× decode bw, ~23 ms recoverable) is
robust to that.

## What this decomposition does NOT resolve

- Whether the bit-exactness gap (down/sum6 + attention — see `exactness-gap`) is closable.
- Whether the fused kernel hits 300 GB/s at K=4 (engineering bet, not proven).
Both feed the `floor-clearance-verdict`.
