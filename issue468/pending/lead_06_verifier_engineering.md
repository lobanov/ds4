# Lead 06 — Verifier engineering: anchor reuse + residual cycle overhead

Date: 2026-07-07. Status: pending. **Updated 2026-07-07 by Lead 03:** the modeled K=4
numbers below cite the SLIDING estimate (~−0.9% / 0.99×), which Lead 03 showed is
OPTIMISTIC — the realistic cycle-jump K=4 edge is **0.982× (−1.8%, P(speed<1)=0.999)**
on the 300-prompt corpus (`summaries/acceptance_statistical_power.md`). Treat the
−0.9% figures here as superseded; the verifier must now overcome a ~−1.8% (cycle-jump)
gap, not ~−0.9% (sliding). **Gated:** start only after lead 01 confirms
anchor reuse is acceptance-safe and (ideally) lead 02 sizes the scheduled
ceiling. This is the only lead requiring real ds4 engineering (weeks, not days).

## Rationale

The gap between the shipped `--mtp` path and the modeled optimized verifier is
the largest single quantified lever in the dossier:

- **Redundant anchor decode, ~18 pp at K=4.** Every shipped cycle starts with
  `ds4_session_eval(first_token)` (`ds4.c:27202`) — a full 26 ms decode — even
  though the verify forward already produced the correction/bonus token's
  logits. Folding the correction into the *next* verify pass (verify K+1 inputs,
  first being the correction) replaces that decode with the marginal verify
  cost of one position, ~7–9 ms. This moves modeled K=4 from −18.9% to −0.9%.
- **~15–19 ms residual per-cycle overhead.** Measured MTP cycle at K=4 is
  ~42 ms above draft+verify; removing the 26 ms anchor decode still leaves
  readback, partial-accept rollback, and scheduling waste. The model sets this
  to zero; the codex review showed +10 ms of residual drops K=4 from 0.99 to
  ~0.89 — so this overhead is the difference between the model and reality even
  after anchor reuse.
- Secondary target: the K=2→3 verify step (43.6→59.7 ms) is partly a path
  artifact (decode2-exact vs batched-micro path); a batch path that is bit-exact
  with decode (or falls back on the rare greedy flip) may shave the low-K cells.

Together with lead 02's scheduler this is the implementation half of the only
modeled route to the primary gate on the in-RAM setup; even alone it converts
the shipped path from clearly-negative to break-even-competitive, which already
satisfies the "beat or clearly match `--mtp`" secondary gate trivially and gives
a real substrate for measuring everything else end-to-end (including the true
DSpark draft cost — the model's draft=10 ms is an assumption, never measured on
Metal).

## Content of work

1. **Anchor-reuse cycle restructure** in `ds4_session_eval_speculative_argmax`
   (`ds4.c:27167`): on rejection, carry the correction token as the first input
   of the next verify batch instead of a standalone decode; commit KV/state for
   it inside that pass. Preserve the exact-greedy contract (the correction is
   sampled from the same logits either way; verify numerics must match the
   decode path or use the exact path as fallback).
2. **Residual-overhead audit + trim**: per-stage timing (`DS4_MTP_TIMING`,
   `DS4_METAL_GRAPH_TOKEN_PROFILE`) of the non-draft/non-verify ~15–19 ms;
   attack readback and partial-accept rollback first (prefix-attn capture at
   `ds4.c:13166` is the existing partial-accept mitigation to extend).
3. **Exactness + performance measurement** on the retained corpora: greedy
   output byte-identical to target-only decode; K-sweep with the same protocol
   as `run_mtp_verifier_bench_long.py` for direct before/after comparison.
4. (Stretch, if lead 02 cleared) wire the confidence gate / adaptive ℓ into the
   restructured cycle.

## Success criteria

- **Exactness preserved:** greedy output stream identical to baseline decode on
  the full exactness corpus (hard requirement from GOAL).
- **Cycle accounting matches the model:** measured cost/cycle ≈ draft +
  verify(K) + decode·S(K) within ~5 ms, i.e., the redundant decode is actually
  gone and residual overhead ≤ ~10 ms.
- **Net t/s:** ≥ break-even vs plain decode at K=4 on the 8k corpus (model
  prediction −0.9% ± residuals) — this alone clears the secondary gate vs
  `--mtp` (currently −29 to −39% at K=4). With lead 02's scheduler wired in,
  the target is the realized-policy number lead 02 projected; ≥ +20% on any
  corpus cell would clear the primary gate.
- **Abort condition:** if restructuring cannot preserve exact greedy output
  without reintroducing a per-cycle decode (e.g., batch-vs-decode numerics
  can't be reconciled), record that as the engineering falsification of anchor
  reuse on ds4 — the counterpart to lead 01's representational falsifier.
