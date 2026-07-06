# Speculative-decode speedup model — DSPark drafter on the ds4 target

Date: 2026-07-06 (cycle-cost corrected per reviewer). Numpy projection of
speculative-decode speedup answering three questions. Model:
`issue468/model_spec_speedup.py`; outputs:
`issue48/artifacts/spec_speedup_model/{summary.json,model_inputs.json}`.

## Model (corrected cycle accounting)

The verifier forward produces the correction/bonus token at the rejection point,
and that token **is the anchor for the next cycle**. A fresh anchor decode is
needed **only on full-block acceptance** (no rejection point). Therefore:

```
cost(reject)      = draft + verify                 (verify yields the anchor)
cost(full accept) = draft + verify + decode
E[cost/cycle]     = draft + verify_ms(K) + decode * P(full accept)
                  = draft + verify_ms(K) + decode * S(K)     # S(K)=P(first K all match)
E[tokens/cycle]   = E[a|K] + 1                      # +1 = correction bonus or fresh decode
speedup(K)        = (E[a|K] + 1) * decode_ms / E[cost]
```

Inputs (all retained/measured): `decode_ms=26` (1000/38.49 t/s), `draft_ms=10`
(DSpark, given), `verify_ms(K)` = cross-prompt median of `mtp_verifier_bench_long`
(K2:43.6, K3:59.7, K4:65.8, K5:74.5, K6:79.6), acceptance = greedy prefix
histogram (temp=0) from `exactness_small_bundles`: S(1..5)=0.8125, 0.65, 0.425,
0.2875, 0.1625; E[a|K]=1.4625/1.8875/2.175/2.3375 for K=2/3/4/5.

**Important distinction:** this is an *optimized-verifier* projection. The
shipped `ds4 --mtp` implementation pays a separate anchor decode **every** cycle
(decode+draft+verify), which is why measured MTP t/s is much worse (see
`mtp_verifier_bench_results.md`). Building the verifier to reuse the
verify-produced anchor is itself worth ~18 pp at K=4 (moves K=4 from ~−19% to
~−1%). That verifier-implementation change is a real lever, independent of the
drafter.

## Adversarial review (codex / gpt-5.5, xhigh) — strong perspective, retained

The findings below (Q1/Q2/Q3) are computed **under the explicit modeling
assumption that the verifier reuses the anchor** (per the brief). A read-only
adversarial review was run with `codex exec -m gpt-5.5 -c model_reasoning_effort=xhigh`;
retained at `issue468/artifacts/spec_speedup_model/codex_review.md` (prompt
`codex_review_prompt.md`, full trace `codex_review_output.txt`). Codex was **not**
given the anchor-reuse context, so its central attack lands on that assumption
itself. Its perspective is recorded here as the primary caveat; it does not
retract the under-assumption findings.

**On the assumption (codex's load-bearing objection).** Codex argues anchor-reuse
is the unverified risk: on rejection the verify forward yields the correction
token's *logits* but has not run the target on that token, so there is no valid
KV/hidden state to anchor the next draft (the verify processed the *rejected*
draft at that position). It verified this against the code: every shipped
speculative cycle starts with `ds4_session_eval(first_token)` (`ds4.c:27202`) and
committed state covers only processed drafts (`ds4.c:27555`), i.e. the current
`--mtp` path decodes every cycle. **Under the shipped verifier the cost reverts to
`decode+draft+verify`, giving K=4 = −18.9% and a ~88% per-position requirement to
beat baseline** (vs −0.9% / ~79% under the reuse assumption). Realizability note:
reusing the anchor is *possible in principle* — folding the correction into the
next verify forward replaces the 26 ms decode with ~verify_ms(K+1)−verify_ms(K)
≈ 9 ms — but it is unbuilt on ds4 and not free of overhead. The codex-proposed
**falsifier**: force a rejection and try to start the next draft without decoding
the correction; divergence ⇒ reuse is not realizable on this graph.

**Sensitivities that hold even under the reuse assumption (in-scope):**
- **Residual per-cycle overhead set to zero.** Measured MTP cycle at K=4 is
  `total − draft − verify ≈ 42 ms`; removing the ~26 ms anchor decode still leaves
  ~15–19 ms of readback / partial-accept rollback / first-miss waste. A
  reuse-optimized verifier still pays some of this; adding ~10 ms drops the
  under-assumption K=4 speedup from 0.991 to ~0.885. So the break-even is fragile
  to overhead even granting anchor reuse.
- **The "~2 pp" precision is statistically indefensible.** Across the 10 prompts
  E[a|4] has sd ≈ 0.43 (prompt-level 95% half-width ±0.3) vs a 0.028 drafts/cycle
  break-even gap; per-prompt modeled K=4 spans −18.5% to +10.6%. The mean figure
  is inside the noise.
- **Acceptance-data caveats:** measured on 67–165-token prompts along the
  drafter's own greedy spine, not real trajectories (post-rejection cycles start
  from an unpredicted correction token — likely lower acceptance); temp 0.5/1.0
  drops E[a|4] from 2.175 to ~1.92 (~7 pp hit).
- **Q3 "cannot help" is too absolute:** a depth-3 4-node tree could clear +20%
  *if* its verify fell to ≤~51 ms and acceptance were near-perfect — neither met
  today, but the ceiling isn't literally zero.
- **draft=10 ms** leaves only ~0.7 ms of cost headroom at K=4 under the reuse
  model (+1 ms draft → ~−2%, +5 ms → ~−6.5%).

**Net:** the Q1/Q2/Q3 results below are valid *conditional on anchor reuse and
low residual overhead*. Codex's review flags those two conditions as the
load-bearing, unverified risks (plus the acceptance/noise caveats). Treat the
under-assumption numbers as the optimistic edge of a band whose pessimistic edge
(~−19% at K=4) is the shipped-verifier reality.

## Q2 — does K=4 give a speedup? ~Break-even (−0.9%), not yet a gain.

| K | verify_ms | P(full) | cycle_ms | E[a\|K] | speedup | vs base |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 26.0 | 0.812 | 57.1 | 0.812 | 0.825 | −17.5% |
| 2 | 43.6 | 0.650 | 70.5 | 1.462 | 0.908 | −9.2% |
| 3 | 59.7 | 0.425 | 80.8 | 1.888 | 0.930 | −7.0% |
| 4 | 65.8 | 0.288 | 83.3 | 2.175 | **0.991** | **−0.9%** |
| 5 | 74.5 | 0.163 | 88.7 | 2.338 | 0.978 | −2.2% |

At current acceptance every K is still net-negative, but **K=4 is essentially at
break-even (−0.9%)** and K=5 close (−2.2%). The optimum is K=4. So with a
correctly-built verifier the current drafter is right at the threshold — a small
acceptance gain flips it positive.

## Q1 — acceptance needed to beat baseline and to clear +20%

`E beat`/`E +20%` = accepted drafts/cycle needed (uses actual S(K) for the
full-accept decode); `p` = uniform per-position match probability (geometric)
achieving it; `-1` = unreachable (ceiling below target).

| K | E beat | E +20% | p beat | p +20% | current E | current p |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1.712 | 2.254 | **unreachable** | **unreachable** | 1.462 | 0.809 |
| 3 | 2.106 | 2.727 | 0.890 | **unreachable** | 1.888 | 0.786 |
| 4 | 2.203 | 2.843 | **0.792** | 0.940 | 2.175 | 0.771 |
| 5 | 2.412 | 3.095 | 0.783 | 0.890 | 2.338 | 0.757 |

- **Beat baseline:** K=4 needs p≈0.79 (current 0.771 → **+2 pp**), i.e. E[a|4]
  2.20 vs current 2.175 (+0.03 drafts/cycle). K=5 needs p≈0.78 (+2.6 pp). K=2
  cannot beat baseline at any acceptance (its perfect-accept ceiling is −2%);
  K=3 can (ceiling +8.7%) but not the +20% gate.
- **+20% gate:** K=4 needs p≈0.94 (+17 pp); **K=5 needs the least lift, p≈0.89
  (+13 pp)**, because its higher ceiling (+41% vs +28%) amortizes the verify
  better. The gate is reachable in principle (K≥4 ceilings exceed +20%) but
  needs a substantially better drafter.

So: **the current drafter is ~2 pp of per-position acceptance away from beating
baseline at K=4**, and ~13-17 pp away from the +20% gate. The gap is acceptance,
not structure.

## Q3 — can a 4-node draft tree give a speedup? NO.

A verify of N nodes costs `verify_ms(N)` regardless of tree shape (bandwidth
floor). A branched tree has max committed depth d < N; a linear chain has d = N.
Ceilings = max speedup at **perfect** acceptance:

| N nodes | verify_ms | linear d=N ceiling | best tree d=N−1 ceiling |
|---:|---:|---:|---:|
| 3 | 59.7 | +8.7% | −18.5% |
| **4** | 65.8 | **+27.7%** | **+2.2%** |
| 5 | 74.5 | +41.2% | +17.6% |
| 6 | 79.6 | +57.4% | +34.9% |

For a **4-node** structure (corrected cost):
- **linear depth-4**: ceiling +27.7%; at current acceptance E=2.175 → **0.991
  (−0.9%)**.
- **tree depth-3 (spine3 + repair at pos 3)**: ceiling **+2.2%** even at perfect
  acceptance. At current top-1 (no repair lift) → 0.864 (−13.6%); **with a
  perfect repair branch** → 0.873 (−12.7%) — i.e. even a flawless repair leaves
  it ~12% below baseline and worse than linear-4.
- **tree depth-2 (spine2 + 2 leaves)**: ceiling −23.4% (always slower).

Two independent reasons trees lose here:
1. **Ceiling:** a 4-node tree's perfect-acceptance ceiling is +2.2% (below the
   +20% gate and barely above baseline), because branches spend verify nodes
   without extending the committed prefix.
2. **Hedging is dominated by spine extension** — and is *doubly* penalized: the
   4th node as a spine extension adds E = S(4) = 0.288; as a position-3 hedge it
   adds only E = S(2)·Δ ≈ 0.065/0.130/0.195 for a +10/20/30% top-2 lift (spine
   wins at every realistic lift). Worse, raising pos-3 coverage *also* raises
   P(full-block-acceptance), so it increases the fresh-anchor-decode penalty —
   which is why the perfect-repair tree (0.873) barely beats the no-repair tree
   (0.864).

**Best tree strategy:** do not branch — use a linear chain. If branching is
forced, maximize depth (spine N−1 + one repair at the deepest node), but no
4-node tree beats the gate or delivers a real speedup.

## Conclusions

1. **Q1:** with an optimized verifier, beating baseline needs only ~79%
   per-position acceptance at K=4 (current ~77% → **~2 pp short**); the +20%
   gate needs ~89-94% (K=5/K=4) → ~13-17 pp short.
2. **Q2:** K=4 is at **break-even (−0.9%)** with the current drafter and an
   optimized verifier — not a gain yet, but a ~2 pp acceptance improvement makes
   it beat baseline. (The shipped `--mtp` implementation is far worse because it
   pays a redundant anchor decode every cycle.)
3. **Q3:** a 4-node draft tree cannot give a speedup — ceiling +2.2%, current
   ~−13%, dominated by a linear chain. Hedging is doubly penalized (wastes verify
   nodes AND raises the full-accept decode penalty).

Net: the binding constraint is **drafter acceptance** (precision is exhausted per
`dspark_quantization_ceiling.md`). Two realistic levers: (a) a verifier that
reuses the verify-produced anchor (the ds4 `--mtp` implementation does not — this
is ~18 pp at K=4), and (b) a materially better drafter. Trees do not help.

## Caveats

- `verify_ms(K)` is the MTP verifier cost; DSPark would reuse the same target
  verifier. Tree-attention verify may be marginally cheaper (sparser masks) but
  the bandwidth floor is unchanged, so the +2.2% tree ceiling is robust.
- Acceptance is the retained greedy (temp=0) prefix distribution over 5-token
  blocks; K>5 is not grounded. The corrected cost uses S(K)=P(first K all match).
- The geometric per-position `p` is a summary statistic; real acceptance decays
  faster with depth (non-geometric), making the gate modestly harder than the
  uniform-p numbers suggest.
- The optimized-verifier projection assumes the verify forward produces a usable
  anchor hidden state at the rejection point; this is standard but is an
  implementation property to confirm against the ds4 graph (the current `--mtp`
  path does not do it).
