# Lead 04 — SSD-streaming / RAM-constrained regime benchmark

Date: 2026-07-07. Status: pending. Independent of leads 01–03; requires a scope
decision before spending effort (see caveat).

## Rationale

All measurements to date are on an in-RAM Metal setup where decode costs 26 ms
and the amortizable per-pass component (~16 ms dense streaming) is small relative
to the per-K costs. The GOAL requires +20% on "at least one realistic local
machine/backend setup" — and the economics change qualitatively in any regime
where each forward pass carries a large **per-pass-fixed** cost, because a verify
pass streams that cost once for K positions while plain decode pays it per token.

Sensitivity from the current model at K=4 (draft 10, verify 65.8, S(4)=0.288,
E[a|4]+1=3.175), adding a fixed per-pass cost T to both decode and verify:
`speedup(T) = 3.175·(26+T) / (83.3 + 1.288·T)` → T=10 ms: **+19%**; T=25: +40%;
T=50: +63%; limit 2.47×. The current setup misses the gate by less than 10 ms of
per-pass fixed cost. Candidate realistic setups: the 87 GiB target on a
64–96 GB machine via ds4's SSD path (`ds4_ssd.c`, `g->ssd_streaming`,
`ds4_streaming_hotlist.inc`), or Strix Halo (`STRIXHALO.md`).

**The decisive unknown (why this is a measurement, not a claim):** only
per-pass-fixed traffic amortizes. MoE routed experts are per-token traffic — the
verify union grows with K. Two sub-regimes:

- *Dense weights among what streams per pass* (RAM ≪ model): fixed component
  dominates (~6 GB/pass dense vs ~2 GB/token selected experts) → large win even
  with pessimistic expert overlap.
- *Dense resident, only cold experts stream* (RAM slightly small; the natural
  hotlist behavior, since dense is small and hot): SSD traffic is the scaling
  component; amortization comes only from expert-union overlap, and the
  economics can be marginal or even worse than in-RAM. An inverted residency
  policy (pin experts, stream dense) is a possible design lever here.

## Content of work

1. **Scope check first** (zero cost): confirm the RAM-constrained local setup is
   one the project actually cares to win on — baseline decode there is slow in
   absolute terms (a 100 ms/pass toll ⇒ ~8 t/s baseline), so this clears the
   gate as written but not necessarily the gate as intended.
2. **Instrument expert-union stats** (cheap, in-RAM): log per-cycle union size /
   overlap for K=2..5 from the existing MTP bench knobs — this alone predicts
   which sub-regime SSD lands in.
3. **One K-sweep in the constrained regime**: run the existing
   `run_mtp_verifier_bench_long.py` protocol (baseline + K=2..6) with ds4 SSD
   streaming active (or artificially capped RAM / Strix Halo), recording the
   same draft/verify/decode split plus SSD read volume per cycle.
4. Feed the measured decode' / verify'(K) into the speedup model; report the
   regime-adjusted fixed-K and (with lead 02) scheduled projections.

Estimated effort: instrumentation ~1 day; benchmark runs mostly unattended.

## Success criteria

- **Gate test:** any (prompt, K) cell — or the modeled schedule from lead 02's
  policy — showing ≥ +20% generation t/s vs the same-setup plain baseline, with
  exact greedy output preserved. That satisfies the GOAL's primary gate as
  written.
- **Mechanism test:** measured verify'(K)/decode' ratio ≤ ~1.3 at K=4 (i.e., the
  per-pass cost genuinely dominates), confirming the amortization argument
  rather than a lucky cell.
- **Negative outcome is also decisive:** if the hotlist keeps dense resident and
  expert misses dominate (verify' scaling ≈ K× decode'), record the regime as
  closed and drop the lead — the fixed-cost argument then has no realistic local
  carrier.
