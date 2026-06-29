# Assignment 1 — Re-establish the true acceptance baseline (measurement integrity)

Date: 2026-06-29. Supersedes issue468/23's terminal verdict. Resolves the five
flaws the independent review (issue468/24) identified.

## Headline

The prior Outcome B verdict (issue468/23) is **not reaffirmed as terminal**. Under
corrected measurement, the >20% gate is **reachable at 64k via the F32-accumulation
path** (committed 2.80), contingent on the F32 drafter's measured cost (Assignment 2,
bar: draft ≤ 7.8ms). The verdict was premature because it (a) dropped the structural
`+1` committed token, and (b) used Metal greedy acceptance while never measuring the
project's actual B2 protocol on Metal.

But the verdict's CORE finding holds: the BF16-accumulation Metal drafter (as-is)
fails decisively (B2 committed 1.31, gate FAILS everywhere). The path to the gate is
fixing Source B (BF16 accumulation) — exactly the review's Assignment 2.

## The four checks

### 1a — B2 on Metal (the number the terminal gate needed, never measured)

Built `measure_metal_b2.py`: dumps the Metal drafter's per-step base_logits
[19,5,vocab] via a persistent-KV sweep (DS4_DSPARK_PROBE_DUMP_Q), then runs a 500-
trial/step B2 MC simulation (drafter samples q_i = softmax(base[i] + markov_bias(prev))
at temp=1.0; verifier accepts w.p. min(1,p(x)/q(x)); resample norm(max(0,p-q)) on
reject; committed = accepted + 1). The drafter's window KV depends only on target
anchors (main_x), not on sampled drafts (positions 1-4 are noise tokens), so the
per-step base_logits are valid for offline B2 MC.

**Metal B2: committed 1.31 (accepted A=0.31).** WORSE than greedy (1.53/committed
2.53). Per-position accept rate pos0 = 0.178: the BF16-noisy drafter puts ~82% of its
probability mass on tokens outside the target's top-128 support, so B2 sampling mostly
rejects. This is Source B made concrete — a flat-on-wrong-tokens distribution.

**Oracle B2 (F32, same captures): committed 2.80 (A=1.80).** The F32 path reaches the
gate marginally. The ~5.8× gap (2.80 vs 1.31) is entirely Source B (precision), since
oracle and Metal use the SAME Q4_K weights, SAME persistent-KV window, SAME target.

Bug found during measurement: the oracle B2 sim (measure_b2_acceptance.py) had a
correct markov weight load (reshape [vocab,rank]); my first Metal B2 attempt copied a
WRONG transpose ([rank,vocab].T) that scrambled BF16 elements → gave committed 1.00.
Fixed; verified the corrected markov-conditioned q argmax matches the greedy probe's
drafts ([1309,304,...] at pos152).

### 1b — single-step vs 19-step / KV reconciliation

ANSWERED by the 1a B2 comparison: oracle multi-step (19 steps, persistent growing KV
window, n_real 2-128) holds at committed 2.80 — the SAME KV bookkeeping the Metal
probe uses. So the Metal 1.53/1.31 collapse is purely Source B (precision), NOT a KV-
ring threading artifact. The window-KV path is correct. (KV fill even helps: oracle
committed by fill 2-10=3.40, 11-30=3.10, ..., 101-128=2.87 — no collapse with growing
n_real.)

### 1c — RAM at 64k

Measured at 64k with model+drafter loaded (prefill ~54k in progress):
- wired-down pages: 89.3 GiB (includes model 80.76 + drafter 10.71 + GPU/context buffers)
- context buffers: 1394 MiB at 64k (SWA keeps it tiny)
- free + inactive (reclaimable): ~19.7 GiB

doc 19's "~7 GB free" was point-in-time free-pages (pessimistic, ignores reclaimable
inactive). Phase 3's itemized 29.5 GiB headroom is the real capacity. **Q8_0 drafter
(+~7 GiB net) and Q6_K (+3.8 GiB) fit comfortably**; F16 (+26 GiB) is borderline.

### 1d — committed accounting audit

doc 23's terminal speedup table used `committed = raw greedy prefix` (1.53),
contradicting its own prose ("committed ≈ prefix + ~1") and doc 15 (2.79 + 1 = 3.79).
Correcting to `committed = accepted + 1` (the structural bonus/correction token,
always present in spec decode — verify over γ drafts yields γ+1 target dists):

| accounting (64k, draft 7.2, verify 75) | committed | speedup | gate |
|---|---|---|---|
| doc 23 (no +1, greedy A) | 1.53 | 0.66× | FAIL (wrong) |
| corrected greedy (+1) | 2.53 | 1.09× | FAIL |
| B2 Metal-BF16 (+1) | 1.31 | 0.57× | FAIL |
| B2 Oracle-F32 (+1) | 2.80 | 1.21× | **PASS** |

Also confirmed: the verify pass runs all 43 layers (metal_graph_encode_layer_batch),
so the layer-40/41/42 hidden (the drafter's next-cycle anchor) is architecturally free
— no separate per-cycle target decode. Per-cycle cost = draft + verify only.

## Gate reachability (the decisive table)

committed = accepted + 1. Break-even at 64k for committed 2.80: draft ≤ 7.8ms.

| ctx | plain | Metal-BF16 (1.31, d7.2) | F32-committed-2.80 d7.2 | d10 | d15 |
|---|---|---|---|---|---|
| 32k | 32.9 | 0.52× fail | 1.12× fail | 1.08× | 1.02× |
| 55k | 34.8 | 0.55× fail | 1.19× fail | 1.15× | 1.08× |
| 64k | 35.5 | 0.57× fail | **1.21× PASS** | 1.17× fail | 1.10× fail |

## Exit verdict

**1.53/1.31 does NOT survive corrected measurement** in the sense that the F32 path
reaches the 64k gate marginally (committed 2.80 → +21% iff F32 draft ≤ 7.8ms). The
prior "fails everywhere decisively" verdict is corrected to: "fails as-is (BF16), but
the F32-accumulation path (Assignment 2) is the necessary fix and reaches 64k
contingent on its measured draft cost."

The gate's decisive variable is now **Assignment 2's measured F32-accumulation draft
cost** (bar: ≤ 7.8ms at 64k for the committed-2.80 acceptance). 32k/55k fail even with
F32 (need draft <2ms / <6ms, implausible) — so the realistic regime is **64k only**.

**Proceed to Assignment 2** (memory-neutral F32-accumulation gathered-dense MoE over
unchanged Q4_K weights) to measure the real F32 draft cost and confirm end-to-end
acceptance + speedup. If F32 draft ≤ 7.8ms and acceptance holds at ~2.80 → Outcome A
at 64k; if F32 draft > 7.8ms or acceptance drops → the verdict is reaffirmed as B with
corrected numbers.

## Files
- `measure_metal_b2.py` — Metal B2 MC on dumped base_logits.
- `ds4_dspark_probe_accept` (ds4.c) extended with `DS4_DSPARK_PROBE_DUMP_Q` to emit
  base_logits [n_steps,5,vocab].
- `metal_base_logits_19steps.bin` — the dumped Metal drafter distribution.
