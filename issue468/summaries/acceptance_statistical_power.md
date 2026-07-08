# Lead 03 — acceptance statistical power + realistic-trajectory measurement

Date: 2026-07-07. Status: **resolved** (both codex gates passed; verdict recorded with
GATE-2 edits). Harness: `run_lead03_torch_measure.py` (torch/MPS, precision-gated) +
`run_lead03_cyclejump.py` (realistic trajectory) + `run_lead03_aggregate.py`. Artifacts:
`issue468/artifacts/acceptance_powered/` (combined300/{aggregate,cyclejump,trajectory}.json,
stage2_torch_measure/, torch_measure/torch_precision_gate.json, codex_reviews/). Worklog:
`issue468/archive/leads/lead_03_acceptance_statistical_power.md`.

## Question

Power the DSpark acceptance measurement to a decision-grade CI (every prior number rested
on 10 prompts × ~8 steps, sd≈0.43, CI ±0.3 vs a 0.028 break-even gap — inside the noise),
and quantify the realistic-trajectory bias the speedup model flagged but never measured.

## Method

- **Corpus:** 300 prompts (Stage 2's 240 dolly/codealpaca/jsonex + 60 newly captured via the
  committed `--capture-dataset`), 128-token temp=0 greedy spines. Few-file sharded safetensors.
- **Measurement:** torch/MPS port of the drafter (`dspark_train/drafter_body.py`+`drafter_head.py`),
  **precision-fidelity-gated at 100% draft-token agreement vs the numpy oracle** (25 prompts).
  The numpy oracle was too memory-heavy (per-expert re-dequant crashed the machine at scale);
  torch loads experts once and runs ~3× faster.
- **Realistic trajectory:** K-cycle-jump simulation (advance by accepted+1/cycle), the
  per-cycle acceptance the speedup model actually needs — NOT the sliding (position-uniform)
  average. (GATE-1 caught that my initial sliding-based headline was an overclaim.)

## Result

| estimator | E[a\|4] | S(4) | K=4 speedup (optimized verifier) |
|---|---|---|---|
| sliding (position-uniform; the model's Q-tables) | 2.337 | 0.382 | **+1.2%** (optimistic, diagnostic only) |
| **cycle-jump (realistic per-cycle trajectory)** | **2.198** | **0.340** | **0.982× (−1.8%)** |

- 300-prompt sliding CI half-width on E[a|4] = **0.046 (< 0.05 floor, met)**; sd = 0.418.
  (The 128-token generations did NOT collapse the sd vs the old 0.43 — real prompt-to-prompt
  heterogeneity dominates, not within-prompt noise.)
- **Cycle-jump K=4 is significantly below baseline:** speedup CI [0.971, 0.993],
  **P(speed<1) = 0.999**. The dynamic break-even (at S(4)=0.340) is E[a|4]=2.256; the deficit
  is 0.058 accepted drafts/cycle. (The old "2.203 break-even" was sliding-S; stale for cycle-jump.)
- **Corpus-mix-dependent** (cycle-jump per-source speedup): jsonex +2.3%, codealpaca −1.9%,
  dolly −5.6%. The old 10 code/synthesis exactness prompts were E[a|4]=2.175 (harder than all
  three Stage-2 families) — so the dolly/codealpaca/jsonex mix is on the easy side.
- **Trajectory bias is real:** correction-anchor cycles (drafter mispredicted the anchor) have
  E[a|4]=1.994 vs predicted-anchor 2.427; real trajectories interleave these, pulling the
  per-cycle acceptance down from the greedy-spine 2.337 to the cycle-jump 2.198.

## Verdict (GATE-2 corrected)

**Under the realistic cycle-jump trajectory, K=4 gives speedup 0.982× — significantly below
baseline (P<1=0.999) on this corpus mix, and corpus-dependent.** The sliding estimator the
model and the initial headline used is optimistic by ~3pp. Beating baseline at K=4 is NOT
achieved on this corpus under the realistic trajectory. The honest band: sliding +1.2%
(optimistic) to cycle-jump −1.8% (realistic), with the cycle-jump the model-currency value.

**Scope (what this does NOT settle):** the cycle-jump measures anchor-token difficulty along a
*linear* trajectory. It does NOT model drafter-state pollution after rejected drafts, draft
trees/branching, residual verifier overhead, or adaptive policies — those are Lead 05. The
corpus is dolly/codealpaca/jsonex; a code/synthesis-heavy deployment would be harder (the old
exactness 2.175).

## Side-finding: the Stage 2 torch body port had a bug

GATE-1's precision bug-hunt found `drafter_body.py::hc_post` used `residual.unsqueeze(-3)`
where the numpy oracle uses `unsqueeze(-2)` — a broadcast-axis error, tiny at layer 0 but
amplifying to max 113 by layer 2. **Fixed** (verified: body-fidelity now passes at 1.4e-5 mean;
F32 MPS reproduces numpy 100%). Implication: Stage 2's torch-body features were computed with
the wrong hc_post; Stage 2's *self-consistent* relative verdict (head-LoRA hurts) likely
stands, but its absolute p=1 numbers were on buggy-body features. Recorded for the Stage 2
dossier.

## What was built (reusable)

- `dspark_oracle/stage2_capture_store.py`: reads sharded captures (now multi-dir merge).
- `run_lead03_torch_measure.py`: torch/MPS acceptance harness (gate + measure modes).
- `run_lead03_cyclejump.py`: realistic-trajectory (cycle-jump) E[a|K]/S(K)/speedup + bootstrap.
- `run_lead03_aggregate.py` / `run_lead03_trajectory.py`: sliding aggregate + predicted/correction partition.
- `run_lead03_sample_corpus.py`: non-overlapping corpus sampler.
- `model_spec_speedup.py`: refreshed (powered sliding inputs + `lead03_cyclejump_realistic`).
- Engine: `--capture-dataset` committed on `dspark-research` (3510497); `drafter_body.py` hc_post fixed.

## Implications for the other leads

- **Lead 01 (anchor reuse):** still holds — reuse does not collapse acceptance — but the
  *realistic* K=4 edge it feeds is now ~0.98× (cycle-jump), not the ~−0.9% sliding estimate.
- **Lead 02 (confidence-scheduled verification):** the cycle-jump acceptance (2.198) is the
  honest input, not the sliding 2.337.
- **Lead 05 (verifier engineering):** the binding question is now sharper — K=4 is ~0.98× on
  the realistic trajectory, so the verifier must either raise acceptance (drafter) or cut
  residual overhead to flip it positive; and the cycle-jump/drafter-state-pollution gap
  (unmodeled here) is the next thing to measure. (Note: `pending/lead_05_*.md` still cites the
  stale sliding −0.9%; treat as superseded by this cycle-jump 0.982×.)
