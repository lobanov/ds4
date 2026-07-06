# MTP verifier benchmark — measured draft/verify/decode split

Date: 2026-07-05. Empirical confirmation of `summaries/mtp_verifier_bandwidth_binding.md`
(which was a source-only roofline assessment). Single-run characterization on
one prompt; the binding conclusion is structural, the net-t/s numbers are
prompt/acceptance dependent.

## Setup

- target: `DeepSeek-V4-Flash-IQ2XXS-...-imatrix.gguf` (87 GiB, IQ2XXS experts)
- mtp head: `DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf`
- prompt: `issue468/prompts/baseline_corpus/code_4k.txt` (~1.7k tokens)
- backend: metal, `ctx=8192`, `n=96`, `temp=0.0`, `seed=1`
- harness: `issue468/run_mtp_verifier_bench.py` (retained, reproducible)
- knobs: `DS4_MTP_TIMING=1` (per-cycle draft/snapshot/verify/total ms),
  `DS4_MTP_SPEC_LOG=1` (first-draft miss count)
- artifacts: `issue468/artifacts/mtp_verifier_bench/{summary.json,summary.csv,*.stderr}`

## Results

| config | K | prefill t/s | gen t/s | Δ vs base | ms/emitted | verify ms (med) | verify/token | draft ms | committed/cycle | first-miss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | — | 363.7 | **38.49** | — | **26.0** | — | — | — | — | — |
| mtp | 2 | 357.4 | 36.47 | −5.2% | 27.4 | 27.3 | 13.7 | 2.0 | 1.35 | 16/34 |
| mtp | 4 | 362.3 | 27.22 | −29.3% | 36.7 | 57.9 | 14.5 | 5.9 | 2.40 | 11/25 |
| mtp | 8 | 363.3 | 19.27 | −49.9% | 51.9 | 86.0 | 10.8 | 13.3 | 2.07 | 10/28 |
| mtp | 16 | 362.2 | 11.27 | −70.7% | 88.7 | 213.0 | 13.3 | 27.2 | 2.07 | 10/28 |

`committed/cycle` = accepted draft tokens per verify cycle (excludes the base
target token); `first-miss` = cycles where the first draft mismatched the target
argmax and the whole suffix was discarded.

## What this confirms

1. **The verifier sits in the decode (bandwidth-bound) regime, not prefill.**
   Prefill holds ~362 t/s across all configs (large batch, compute-amortized);
   plain decode is 38.5 t/s — a **9.4x gap**, the roofline signature of
   bandwidth-bound single-token decode. The verifier's per-token cost
   (verify/token = 10.8-14.5 ms) lands at ~0.4-0.55 of a decode (26.0 ms): it
   amortizes weight loading over the suffix, but it is still bandwidth-floor'd,
   not compute-bound.

2. **Verify(K=2) ~= one decode.** Median verify for K=2 is **27.3 ms vs 26.0 ms
   decode** — the bandwidth floor predicted by the audit, measured directly.

3. **MTP is net-negative at every K tested.** Best case K=2 is −5.2%; K=16 is
   −70.7%. The verifier-dominated cycle the GOAL warns about is real and measured
   here, not just structural.

4. **Draft is not the bottleneck.** Draft cost is 2-6 ms for K=2-4 (vs 27-58 ms
   verify). Consistent with `summaries/dspark_quantization_ceiling.md` — draft
   quality/precision is exhausted; the cost is on the verify side.

## What this refines (vs the source-only assessment)

- **Total verify is NOT flat in K — it grows, and super-linearly at large K.**
  verify median: 27 -> 58 -> 86 -> 213 ms for K = 2/4/8/16. Per-token verify is
  roughly flat (~11-15 ms) up to K=8, but K=16 jumps because the **union of
  activated routed experts grows** with the suffix (up to 6/token), eroding the
  weight-amortization that makes batched verify cheap. This is a MoE-specific
  effect the roofline argument under-weighted.
- **Acceptance peaks at K=4 (2.40 committed/cycle) then declines** (2.07 at
  K=8/16): the drafter is less accurate further out, so a longer suffix means
  more rejected positions paid for in full. Combined with the super-linear verify,
  large K is strictly worse — the operating sweet spot is K=2-4.
- **First-draft misses waste ~30-47% of cycles** (16/34 at K=2, 10-11/25-28
  elsewhere): a full base decode + draft attempt that accepts nothing.

## Breakeven (back-of-envelope)

Per cycle the MTP path pays `~decode (base token, 26 ms) + draft + verify` to
emit `1 + committed` tokens. For K=4: 26 + 5.9 + 57.6 ~= 89.5 ms for ~3.4 emitted
= 26.3 ms/token — right at the 26.0 ms baseline. Measured 36.7 ms/token because
misses waste cycles. So **K=4 needs ~2.5+ accepted drafts/cycle consistently
(i.e. few misses) just to break even**; clearing the GOAL's >=20% gate would need
materially higher acceptance or materially cheaper verification, neither of which
small-K local MTP delivers today.

## Conclusion

The audit's core claim is confirmed by measurement: the bulk draft verifier is
memory-bandwidth-bound, costs ~one decode at K=2 and grows with K, and the MTP
path is verifier-dominated (net-negative t/s at all tested K on this prompt).
The path to a local speedup is not drafter precision (already exhausted) and not
larger K (super-linear verify + declining acceptance); it requires either much
higher draft acceptance or server-side multi-request batching that amortizes the
verify bandwidth floor across requests — consistent with the GOAL's "plausible
server-side gains" lever.

## Longer-prompt fine K sweep (follow-up)

Extends the above to genuinely longer prompts and finer K resolution.
Harness: `issue468/run_mtp_verifier_bench_long.py`; artifacts:
`issue468/artifacts/mtp_verifier_bench_long/`. Setup: 3 prompt families at the
8k length class (`code_8k` ~8.2k tok, `synthesis_8k`, `grounded_8k`), `ctx=16384`,
`n=128`, `temp=0`, K = 2..6, plus per-prompt baseline.

Baselines (plain decode): code_8k 36.5 t/s (27.4 ms/tok), synthesis_8k 37.5
(26.6), grounded_8k 33.8 (29.6).

| prompt | K | gen t/s | Δ vs base | verify ms (med) | verify/token | committed/cycle | first-miss |
|---|---:|---:|---:|---:|---:|---:|---:|
| code_8k | 2 | 34.46 | −5.6% | 31.8 | 15.9 | 1.44 | 11/48 |
| code_8k | 3 | 24.76 | −32% | 59.7 | 19.9 | 1.95 | 13/39 |
| code_8k | 4 | 22.24 | −39% | 65.9 | 16.5 | 2.08 | 16/36 |
| code_8k | 5 | 19.87 | −46% | 74.8 | 15.0 | 2.03 | 18/36 |
| code_8k | 6 | 19.21 | −47% | 80.1 | 13.4 | 2.03 | 18/36 |
| synthesis_8k | 2 | 35.25 | −6.1% | 49.4 | 24.7 | 1.53 | 13/45 |
| synthesis_8k | 3 | 27.35 | −27% | 59.7 | 19.9 | 2.16 | 10/37 |
| synthesis_8k | 4 | 23.14 | −38% | 65.8 | 16.4 | 2.16 | 11/37 |
| synthesis_8k | 5 | 21.27 | −43% | 74.5 | 14.9 | 2.31 | 9/36 |
| synthesis_8k | 6 | 20.18 | −46% | 79.7 | 13.3 | 2.28 | 10/36 |
| grounded_8k | 2 | 33.71 | −0.3% | 49.5 | 24.8 | 1.64 | 12/44 |
| grounded_8k | 3 | 28.45 | −16% | 59.7 | 19.9 | 2.34 | 10/35 |
| grounded_8k | 4 | 24.04 | −29% | 65.8 | 16.5 | 2.56 | 6/34 |
| grounded_8k | 5 | 22.35 | −34% | 74.3 | 14.9 | 2.64 | 7/33 |
| grounded_8k | 6 | 20.14 | −40% | 79.0 | 13.2 | 2.46 | 6/35 |

Findings:

- **Binding holds on longer prompts.** verify median is essentially
  prompt-independent in *trend* (K=2..6 -> ~32-50, 60, 66, 74, 79 ms across all
  three families); per-token verify falls with K (~20 -> ~13 ms/tok), always
  below decode (~27-30 ms). The bandwidth floor is prompt-independent, as
  claimed. Absolute verify varies a little by prompt (code_8k K=2 is cheapest)
  because the activated-expert union depends on token content.
- **verify grows ~linearly with K** in 2..6 (roughly +6-8 ms per +1 K after the
  K=2->3 step, which is larger because K=2 uses the decode2 path and K>=3 the
  batched micro path). No super-linear blow-up in this range (that only appeared
  at K=8/16 in the first sweep, from expert-union growth).
- **Acceptance peaks at K=4-5** then plateaus/declines: 1.4-1.6 (K=2) ->
  2.0-2.6 (K=4-5) -> 2.0-2.5 (K=6). `grounded_8k` has the highest acceptance
  (2.64 at K=5) and the least-bad net t/s.
- **MTP is net-negative at every (prompt, K) cell.** Best case is
  `grounded_8k` K=2 at **−0.3%** (break-even); worst is K=6 at **−40 to −47%**.
  No prompt x K cell beats plain decode. K=2 is break-even-ish; K>=3 is clearly
  negative.
- **First-draft misses waste 17-50% of cycles** (e.g. code_8k K=4: 16/36 = 44%):
  a full base decode + draft that accepts nothing — the largest single leakage.

This refines the headline: on longer prompts the per-token verify amortization
is healthy (down to ~13 ms/tok at K=6), but the cycle still pays a full base
decode + draft + verify + a high miss rate, so accepted tokens (<=2.6/cycle)
never amortize the floor. The least-bad operating point is K=2 (break-even on
the best prompt); nothing in the local single-request MTP path clears baseline,
let alone the >=20% gate.

## Caveats

- Single run per cell, three 8k prompts plus the earlier single `code_4k` run,
  temp=0. Net t/s varies with prompt because acceptance varies; the **binding**
  (bandwidth-bound, verify ~= decode, grows with K) is prompt-independent and is
  the load-bearing conclusion.
- `code_4k` is a structured/repetitive prompt (relatively high acceptance); a
  harder prompt would lower `committed/cycle` and make MTP look worse, not better.
- These are wall timings under `DS4_MTP_TIMING`, which adds minor stderr-printing
  overhead per cycle; the relative split (draft vs verify, K-scaling) is unaffected.
