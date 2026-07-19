# Lead 05 — Per-pass-fixed-cost regimes: SSD streaming + fast-link distributed

Date: 2026-07-07 (rewritten 2026-07-19 to expand scope from SSD-only to the
generalized per-pass-fixed-cost mechanism with two carriers). Status: **pending.**
Independent of leads 07 / 08 / 09 for its offline phase; the cheapest decisive
experiments (the scope check + the extended speedup-model projection) run
without any new engineering and predict which carrier — if any — warrants a
measurement run.

> **2026-07-19 scope expansion.** The original Lead 05 considered only SSD
> streaming as the source of the per-pass-fixed cost `T`. The mechanism is more
> general: speculative verify amortizes *any* per-pass-fixed cost over the `K`
> drafted positions. The README's distributed-inference section surfaces a
> second carrier — the cross-machine activation round-trip in distributed
> decode — which is the same `T`-amortization mechanism applied to a different
> `T` source. This rewrite keeps the original SSD analysis intact and adds the
> distributed fast-link carrier as a parallel. Slow links (WiFi, VPN) are
> explicitly **out of scope**: the +20% gate is a *local* speedup gate, and
> slow-link distributed decode is already known-bad on its own terms.

## Execution contract

Pursue this lead under the general research methodology in
`.pi/skills/research-lead/SKILL.md`:

- orient and verify all claimed assets before acting (done 2026-07-19 — see
  "Asset verification" below);
- lock the decision rule, the estimator, and the per-carrier threshold before
  trusting measurements;
- fidelity-gate any new acceptance / verify path against retained oracle /
  powered-acceptance / M3-runtime tooling;
- report prompt-clustered CIs and **per-carrier + per-link** results, not a
  single pooled headline;
- propagate the canonical result through `summaries/` and `STATUS.md`;
- archive the lead correctly on completion.

Treat this as decision-grade work with two adversarial review gates following
`.pi/skills/adversarial-codex-review/SKILL.md`:

- **setup gate:** after the extended speedup-model projection exists, but
  before trusting its carrier ranking;
- **verdict gate:** before recording the conclusion for whichever carrier(s)
  reach measurement.

Each gate uses an independent `gpt-5.5` `xhigh` subagent review that re-derives
claims from code and artifacts. Any decisive subagent finding must be
independently verified before acceptance.

## Rationale — the unified mechanism

Every measurement in the dossier to date is on an **in-RAM, single-process
Metal setup** where decode costs ~26 ms and the only per-K cost is the
bandwidth-bound verify forward (Lead 08's `verify_ms(K) ≈ 40 + 6.6·K`). In that
regime the verify dominates and there is no per-pass-fixed cost worth
amortizing — DSpark loses or breaks even (M2: −16 % vs plain on the exact
verifier; M3: +4.9 % over plain only after committing batched verify + anchor
reuse + GPU drafter, and only because the verify is now ~80 % of the cycle
rather than the draft or decode).

The economics change qualitatively in any regime where each forward pass
carries a **per-pass-fixed cost `T`**, because a verify pass pays `T` once for
`K` positions while plain decode pays `T` per token. The carried-forward
sensitivity table (at K=4, `S(4)=0.288`, `E[a|4]+1=3.175`, draft 10 ms, decode
26 ms):

```
speedup(T) = 3.175·(26+T) / (83.3 + 1.288·T)
  T=0   →  +0 %   (the current in-RAM reality, ≈ M3's headline regime)
  T=10  →  +19 %  (just under the gate)
  T=25  →  +40 %
  T=50  →  +63 %  (limit 2.47×)
```

The current in-RAM setup misses the +20 % primary gate by less than 10 ms of
per-pass-fixed cost. Two realistic carriers can supply that `T`:

### Carrier A — SSD streaming (RAM ≪ model, the original Lead 05)

When the model does not fit in RAM, ds4's SSD streaming mode
(`--ssd-streaming`, `ds4_ssd.c`, `ds4_streaming_hotlist.inc`, `g->ssd_streaming`)
keeps the non-routed (dense) weights resident and loads routed MoE experts from
the GGUF on cache misses. The README is explicit that this is a continuous
spectrum, not a binary cutoff:

> "SSD streaming allows to turn the available amount of RAM from a hard cutoff
> (can I run this model or not?) to continuous spectrum of speed levels… Long
> prefills can still be fast; generation is more sensitive to cache misses
> because every new token routes through experts again."

The per-pass-fixed cost `T` here is the dense-stream component on a cache miss
(~6 GB/pass dense vs ~2 GB/token selected experts at full miss). The Lead 05
mechanism applies directly. The load-bearing unknown (unchanged from the
original): only per-pass-fixed traffic amortizes. MoE routed experts are
per-token traffic — the verify union grows with `K`. Two sub-regimes:

- *Dense weights among what streams per pass* (RAM ≪ model): the fixed
  component dominates → large win even with pessimistic expert-union overlap.
- *Dense resident, only cold experts stream* (RAM slightly small; the natural
  hotlist behavior, since dense is small and hot): SSD traffic is the scaling
  component; amortization comes only from expert-union overlap, and the
  economics can be marginal or even worse than in-RAM. An inverted residency
  policy (pin experts, stream dense) is a possible design lever here.

### Carrier B — Fast-link distributed decode (the new carrier)

Distributed inference (`ds4_distributed.c`, `--role coordinator`/`worker`,
`--layers A:B`) splits transformer layers across machines. The README
documents the regime precisely:

> "Generation is strictly autoregressive: token N+1 cannot start until token N
> has produced logits and sampling has selected the next token. That means
> distributed generation cannot use the long prefill pipeline. It pays at least
> one cross-machine activation hop per generated token, so generation is slower
> than a single local process."

That "at least one cross-machine activation hop per generated token" **is** the
per-pass-fixed cost `T`. A verify pass that carries `K` positions through the
layer pipeline pays the hop once for the whole batch (the activation payload
scales `K×`, but the per-message latency and the per-slice weight-streaming are
paid once). The prefill path already proves the transport supports multi-token
batches (chunks of 4096 via `--dist-prefill-chunk`); only the decode entry
point hardcodes single-token.

**Scope: fast local direct link only.** The README's per-link table:

| Link | Ping | Prefill | Generation |
|---|---:|---:|---:|
| Thunderbolt 5 | 0.45 ms | 582.99 t/s | 25.09 t/s |
| WiFi | 77.20 ms | 250.70 t/s | 10.70 t/s |
| Internet / VPN | 152.10 ms | 114.88 t/s | 3.63 t/s |

WiFi and VPN are out of scope: their generation numbers are dominated by
link RTT in a way that is not a "local" speedup story (3.63 t/s on VPN is a
capacity/inspection mode, not a deployment target). The realistic distributed
carrier is **Thunderbolt 5** (documented best case) and equivalent **direct
Ethernet** (10/25/40 GbE, sub-ms ping, switching fabric on a desk or rack).
On TB5 the per-token distributed penalty is ~6 ms (39.85 ms/token distributed
vs ~33 ms/token single-process on the 12 k control run); on direct Ethernet the
shape is similar with bandwidth determined by the link.

**Prize shape (Carrier B, fast link).** A verify(K=4) batch through the
layer pipeline replaces 4 single-token round trips with 1 batched round trip.
Using the Lead 08 fit `verify_ms(K) ≈ 40 + 6.6·K` (single-process, full stack),
split across two layer-slices computed serially with one network round trip:

- Plain distributed decode ≈ 40 ms/token (README, TB5).
- Distributed verify(K=4) ≈ (slice_A + slice_B) + 2× ping + K× activation bw.
  With per-slice verify compute ≈ half of (40 + 6.6·4) = 33 ms each, serial
  → ~66 ms slice compute + ~1 ms RTT + activation bw ≈ ~68 ms/cycle.
- At cycle-jump acceptance E[a|4]≈2.2 (3.2 useful tokens/cycle), that is
  ~3.2 tokens / 68 ms ≈ **47 t/s vs plain distributed 25 t/s ≈ 1.9×**.

This is a back-of-envelope projection to anchor the prize, **not** a
measurement. The load-bearing unknowns: whether the existing distributed
batched transport scales efficiently to K=2..5 (it is tuned for 4096-token
prefill chunks, not 4-token verify batches), whether the activation bandwidth
dominates at small K on the link, and whether the per-slice verify cost
matches the Lead 08 single-process curve (which itself is above its bandwidth
floor at ~190 GB/s vs decode ~410–450 GB/s — Lead 08's open attribution
question).

## Asset verification (2026-07-19 orientation)

Read-only verification of every asset claimed above:

- `ds4_ssd.c` (6 KB), `ds4_streaming_hotlist.inc` (192 KB) — **present.**
- `ds4.h` `engine.{ssd_streaming, ssd_streaming_cold, ssd_streaming_cache_experts,
  ssd_streaming_cache_bytes, ssd_streaming_preload_experts}` — **present**;
  `ds4_cli.c:1582` parses `--ssd-streaming` family; `ds4_gpu.h:74`
  `ds4_gpu_set_ssd_streaming(bool)`.
- `STRIXHALO.md` — **present** (the Strix Halo reference for the RAM-constrained
  local setup).
- `ds4_distributed.c` (322 KB) / `ds4_distributed.h` — **present**;
  `ds4_dist_session_eval` (ds4_distributed.c:5637) hardcodes
  `dist_coordinator_eval_span(..., &token, 1, ...)` (single-token decode).
- `dist_coordinator_eval_span` (ds4_distributed.c:2670) is called from 5 sites
  including the prefill path — **multi-token transport exists and is exercised
  by prefill chunks today.** The decode single-token restriction is at the
  entry point, not in the transport.
- **DSpark is hard-disabled in distributed today.** `ds4.c:29621`
  `ds4_session_eval_speculative_argmax` short-circuits to plain
  `ds4_session_eval` when `s->distributed` is set: the drafter is not loaded,
  the verify is not wired. Enabling Carrier B requires replacing that bail-out
  with a real distributed verify dispatch.
- **M3 spec levers compose with `--ssd-streaming`.** The M3 stack
  (`DS4_DSPARK_VERIFY_BATCHED`, `DS4_DSPARK_ANCHOR_REUSE`,
  `DS4_DSPARK_VERIFY_PREFIX_CHECKPOINT`, `DS4_DSPARK_DRAFT_METAL`,
  `DS4_DSPARK_DRAFT_METAL_STS`) is env-gated at decode time; `ssd_streaming`
  is an engine-load flag. They are orthogonal; grep finds no coupling. So
  Carrier A's measurement is "run the existing M3 bench under
  `--ssd-streaming` with a constrained cache budget" — no new code.
- Retained acceptance tooling (Lead 03 numpy/torch-MPS oracle) and the cycle
  model (`issue468/model_spec_speedup.py`) — **present** and reusable for the
  Phase 1 projection.

## Notation & definitions

- **`T`** — per-pass-fixed cost paid once per forward pass, independent of how
  many positions the pass carries. Source depends on carrier: SSD dense-stream
  on miss (Carrier A) or network round-trip + per-slice setup (Carrier B).
  Always qualified by carrier: `T_ssd` or `T_dist`.
- **`K`**, **`a`**, **`E[a|K]`**, **`S(K)`**, **`verify_ms(K)`**,
  **`decode_ms`**, **`draft_ms`**, **`speedup(K)`** — as in
  `summaries/spec_speedup_model.md`. Two estimators: **sliding** (optimistic,
  diagnostic) and **cycle-jump** (realistic, model currency).
- **Carrier** — which `T` source is in play. Per-carrier results are first-class;
  never pool Carrier A and Carrier B into one headline.
- **Link** (Carrier B only) — the physical network between coordinator and
  worker(s). In-scope: Thunderbolt 5, direct Ethernet. Out-of-scope: WiFi, VPN.
- **Sub-regime** (Carrier A only) — dense-stream-dominated vs
  dense-resident/cold-experts-only. Predicted by the expert-union
  instrumentation step; decides whether SSD amortization is real.
- **Slice compute** (Carrier B) — the per-machine forward over its layer range.
  For an N:output split the verify's slice compute scales sublinearly in `K`
  (Lead 08 mechanism: weights streamed once per pass, not per token).

## Content of work

Phased. Phase 0 + Phase 1 are cheap, no new engineering, and decide whether
either carrier warrants a measurement. Phases 2 and 3 are carrier-specific and
run only if the projection clears.

### Phase 0 — Scope decisions (zero cost, before any projection)

Two questions, both for the user:

1. **Carrier A scope:** is a RAM-constrained local setup (the 87 GiB target on
   a 64–96 GB machine, or Strix Halo) one the project actually cares to win on?
   Baseline decode there is slow in absolute terms (a 100 ms/pass toll ⇒ ~8 t/s
   baseline), so this clears the +20 % gate as written but not necessarily as
   intended. If the answer is "no, the project's local target is fully in-RAM,"
   Carrier A closes at Phase 0.
2. **Carrier B scope:** is fast-link distributed decode (TB5 / direct Ethernet
   between two machines on a desk or rack) a deployment shape the project
   cares about, distinct from the single-machine local gate? The README frames
   distributed as primarily for fitting larger models and speeding up long
   prefills, with generation "slower than single process" by design — so the
   prize here is "rescue distributed generation," not "beat single-process
   generation." If the project does not intend to ship distributed generation,
   Carrier B closes at Phase 0.

If both close at Phase 0, record the lead as scoped-out negative and stop.

### Phase 1 — Extended speedup-model projection (cheap, no engineering)

Extend `issue468/model_spec_speedup.py` with a `T` parameter and re-run the
gate sensitivity for each in-scope carrier. Concretely:

1. Generalize the cycle-cost formula to take `T` on both decode and verify
   (`speedup(T)` as in the rationale). Validate the implementation by
   reproducing the existing in-RAM headline at `T=0` bit-for-bit (fidelity
   gate against the retained `spec_speedup_model` numbers).
2. **Carrier A projection:** plug in `T_ssd` from a measurement-free lower
   bound (the dense-stream bytes at one cache miss / an achievable SSD
   bandwidth ≈ 5–7 GB/s on modern Mac SSDs → `T_ssd ≈ 6 GB / 6 GB/s ≈ 1 s` for
   a *full* dense miss, but in the hotlist sub-regime only the *cold-expert*
   traffic scales with `K` so the effective `T` for amortization is the
   dense-stream component only — pin a low and a high bound). Report the
   per-sub-regime projection (dense-stream-dominated vs cold-experts-only).
3. **Carrier B projection:** plug in `T_dist` from the README's TB5 numbers
   (per-token distributed penalty ≈ 6 ms above single-process; per-slice
   verify compute ≈ half the Lead 08 fit; activation payload bandwidth at the
   link rate). Sweep `K ∈ {2,3,4,5}`, acceptance from the cycle-jump
   corpus-mean and from the jsonex per-source cell (the highest-measured).
   Report per-link (TB5, hypothetical 10/25 GbE).
4. **Codex setup gate** on the extended model + both carrier projections
    before trusting the ranking. Independently verify the `T` derivation for
    each carrier (especially the SSD dense-stream-vs-cold-expert split and the
    distributed slice-compute split).

**Decision rule (locked before Phase 2/3):** a carrier proceeds to
measurement only if its Phase 1 projection shows ≥ +20 % speedup at the
cycle-jump (realistic) acceptance with a non-empty CI upper bound above the
gate. Sliding (optimistic) projections are diagnostic only and do not justify
a measurement run. Both carriers can proceed independently.

### Phase 2 — Carrier A measurement (cheap; no new code)

Only if Phase 1 projects Carrier A ≥ +20 %. The M3 stack is orthogonal to
`--ssd-streaming`, so this is the existing M3 bench under a constrained
expert-cache budget:

1. **Instrument expert-union stats first** (cheap, in-RAM): log per-cycle
   union size / overlap for K=2..5 from the existing MTP bench knobs. This
   alone predicts which SSD sub-regime the constrained-cache run will land in
   and bounds the prize before any SSD run.
2. **One K-sweep in the constrained regime:** run the existing
   `run_mtp_verifier_bench_long.py` / `ds4-spec-bench` protocol (baseline +
   K=2..6) with `--ssd-streaming` active and the cache budget swept (e.g.
   `--ssd-streaming-cache-experts {8,16,32}GB`), recording the same
   draft/verify/decode split plus SSD read volume per cycle. Single heavy
   process; checkpoint per-config so a crash is recoverable; monitor swap.
3. Feed the measured `decode'(T)` / `verify'(K,T)` into the extended speedup
   model; report the regime-adjusted fixed-K and scheduled projections,
   per-sub-regime and per-cache-budget.
4. **Codex verdict gate** on the measurement before recording.

### Phase 3 — Carrier B wiring + measurement (engineering required)

Only if Phase 1 projects Carrier B ≥ +20 %. Unlike Carrier A, this requires
real engine work because DSpark is hard-disabled in distributed today
(`ds4.c:29621`).

1. **Wire the distributed verify path.** Replace the speculative_argmax
   bail-out at `ds4.c:29621` with a real distributed dispatch: the coordinator
   drafts locally (the drafter is single-process, lives on the coordinator),
   then sends the K-token verify batch through the existing
   `dist_coordinator_eval_span` (already multi-token capable — used by prefill).
   Env-gated default-off; the plain single-token distributed path stays the
   fallback. Fidelity-gate against the existing distributed single-token
   decode (the verify must reproduce plain decode's per-token logits when `K=1`
   and acceptance collapses to 0).
2. **Decide drafter placement.** Default: coordinator-side (it already owns
   tokenization, sampling, the prompt). The alternative (drafter on the final
   worker, closer to the output head) trades a smaller activation return for a
   drafter-load cost on the worker — model both in Phase 1 first.
3. **One K-sweep on the in-scope link.** Run `ds4-spec-bench` distributed on
   TB5 (the documented reference setup — two M5 Max 128 GB MacBooks,
   coordinator `--layers 0:19`, worker `--layers 20:output`, the 8192-token
   `promessi_sposi` prompt from the README's per-link table). K=2..5, the
   full M3 stack engaged, byte-diffed vs plain distributed decode.
4. Feed the measured distributed verify(K) curve back into the extended model;
   report the per-link projection (TB5 measured; Ethernet extrapolated from
   the link rate).
5. **Codex verdict gate** on the wiring + the measurement before recording.

Estimated effort: Phase 0 ~0.5 day; Phase 1 ~1–2 days; Phase 2 ~2–3 days
(mostly unattended bench runs); Phase 3 ~1–2 weeks (the distributed verify
wiring + drafter-placement decision + the bench, with a fidelity gate and one
codex bug-hunt cycle budgeted).

## Success criteria

- **Phase 1 gate (the cheap decisive test):** the extended speedup model,
  fidelity-gated against the in-RAM headline at `T=0`, projects at least one
  carrier at ≥ +20 % cycle-jump speedup with a non-empty CI upper bound above
  the gate. If neither carrier projects, record the lead as closed negative
  (the in-RAM verify-dominated regime is the only realistic local regime) and
  stop.
- **Carrier A primary gate:** any (prompt, K, cache-budget) cell — or the
  modeled schedule — showing ≥ +20 % generation t/s vs the same-setup plain
  `--ssd-streaming` baseline, with exact greedy output preserved (the exact
  sequential verify path; the M3 batched verify is score-neutral but not
  byte-exact and is reported separately).
- **Carrier A mechanism test:** measured `verify'(K)/decode'` ratio ≤ ~1.3 at
  K=4 in the constrained regime (the per-pass cost genuinely dominates),
  confirming the amortization argument rather than a lucky cell. Per-sub-regime
  reported.
- **Carrier B primary gate:** any (prompt, K) cell showing ≥ +20 % generation
  t/s vs the same-setup plain distributed decode on the in-scope fast link,
  with exact greedy output preserved.
- **Carrier B mechanism test:** measured distributed `verify(K)` wall time
  grows sublinearly in `K` (the per-slice weight-streaming is amortized; the
  network round trip is paid once per batch, not per token). If distributed
  `verify(K)` grows linearly because the transport or the per-slice compute
  fails to amortize, the carrier closes negative even if a single lucky cell
  clears the gate.
- **Negative outcome is also decisive for either carrier:** record the carrier
  as closed and report the regime that closed it (e.g. "Carrier A closed:
  hotlist keeps dense resident; expert-union overlap insufficient").

## Non-goals

- WiFi and VPN distributed links — out of scope (the +20 % gate is a local
  speedup gate; slow-link distributed decode is a capacity/inspection mode).
- A new drafter or drafter fine-tuning (Lead 07 owns quality); this lead uses
  the existing IQ2XXS drafter.
- Verifier / kernel engineering on the single-process path (Lead 08 owns that;
  this lead consumes Lead 08's `verify_ms(K)` curve as an input).
- Server-side multi-request batching (a separate deployment-shape lead —
  batching amortizes the verify floor across *requests*, not across
  *positions* within a request).
- A polished CLI or stable public flag for either carrier; env-gated
  default-off is sufficient for the measurement.

## Deliverables

1. An extended `model_spec_speedup.py` with the `T` parameter, fidelity-gated
   against the existing in-RAM headline, and per-carrier projection artifacts
   under `issue468/artifacts/lead05_projection/`.
2. A compact summary answering: does either per-pass-fixed-cost carrier (SSD
   streaming, fast-link distributed) clear the +20 % primary gate on a
   realistic setup; if so, which carrier, which sub-regime/link, and at what
   `K`; if not, which mechanism closed it.
3. If Phase 2 and/or Phase 3 run, the per-carrier measurement artifacts and
   the propagated `STATUS.md` / `spec_speedup_model.md` updates.

## Exit conditions

- **Proceed (per carrier):** Phase 1 projects ≥ +20 % cycle-jump for that
  carrier → run the carrier's measurement phase.
- **Narrow (per carrier):** the projection lands in [secondary, +20 %) →
  record the carrier as workload/setup-conditioned secondary material and a
  scope input to the dossier's "narrow DSpark to a `--mtp`-beating path"
  decision.
- **Stop (per carrier):** the projection does not clear secondary → close the
  carrier with the mechanism that closed it.
- **Stop (whole lead):** both carriers close at Phase 0 (scope) or Phase 1
  (projection) → record the lead as closed negative; the in-RAM
  verify-dominated regime is the only realistic local regime for this
  drafter on this target.

## One-line verdict

Lead 05 generalizes from "SSD streaming" to "any per-pass-fixed-cost `T`
amortized by speculative verify"; the two realistic carriers are SSD streaming
(RAM ≪ model) and fast-link distributed decode (TB5 / direct Ethernet), and a
cheap extended-speedup-model projection decides whether either warrants a
measurement run before any engineering investment.
