You are a skeptical senior engineer doing an ADVERSARIAL review of a powered acceptance
measurement about to refresh a speedup model. Challenge the methodology AND the headline.
Re-derive from the data; run read-only checks. Be terse.

WORKTREE: /Users/lobanov/Projects/ds4-dspark-research . Python env: issue468/dspark_train/.venv/bin/python
(numpy 2.5.1, torch 2.12.1).

READ FIRST:
- issue468/artifacts/acceptance_powered/stage2_torch_measure/aggregate.json  (powered E[a|4], CI, sd, power-calc N)
- issue468/artifacts/acceptance_powered/stage2_torch_measure/trajectory.json  (predicted vs correction anchor)
- issue468/artifacts/acceptance_powered/torch_measure/torch_precision_gate.json  (torch-vs-numpy parity)
- issue468/summaries/spec_speedup_model.md  (the model being refreshed; K=4 break-even E[a|4]=2.203)
- issue468/archive/leads/lead_03_acceptance_statistical_power.md  (worklog: the torch hc_post bug fix + findings)

CODE: issue468/run_lead03_torch_measure.py (torch acceptance harness), run_lead03_aggregate.py
(aggregate + power calc), run_lead03_trajectory.py (trajectory partition), dspark_train/drafter_body.py
(the port; hc_post bug was fixed: unsqueeze(-3)->unsqueeze(-2)), dspark_oracle/measure_acceptance_bundle.py
(the numpy reference), dspark_oracle/stage2_capture_store.py (corpus loader). Per-prompt checkpoints:
issue468/artifacts/acceptance_powered/stage2_torch_measure/per_prompt/*.json (240 files; schema:
prompt_id, n_steps, drafts[[BLOCK] per step], E_a_5block, p1, per_position_match[BLOCK], prefix_hist[0..5]).

RECAP: Powered measurement of 240 Stage-2 prompts (dolly/codealpaca/jsonex x80, 128-tok, temp=0) via a
torch/MPS port that was precision-gated at 100% draft-token agreement vs the numpy oracle (15 prompts;
after fixing a hc_post broadcast bug). The investigator is about to claim the K=4 break-even is now
SIGNED POSITIVE.

THE INVESTIGATOR'S RESULT + CLAIMS (verify, do not trust):
1. E[a|4] = 2.366 (greedy-spine), clustered 95% CI [2.315, 2.419] (half-width 0.052); prompt sd 0.408.
   The model's K=4 break-even (beat baseline with the optimized/anchor-reuse verifier) is E[a|4]=2.203.
   => "break-even SIGNED POSITIVE: CI entirely above 2.203 (margin 0.09)."
2. The 128-token generations did NOT collapse the prompt sd (0.408 vs old 10-prompt 0.43) -> real
   prompt-to-prompt heterogeneity dominates.
3. Trajectory partition: predicted-anchor cycles E[a|4]=2.427 (n=21712, 79%), correction-anchor cycles
   E[a|4]=1.994 (n=5768, 21%, BELOW break-even). Real trajectories interleave correction cycles ->
   trajectory-weighted E[a|4] is below the greedy-spine 2.366, near break-even.
4. Per-source: jsonex E[a|4]=2.51, codealpaca=2.36, dolly=2.22 (dolly ~at break-even). Old 10-prompt
   exactness (code/synthesis) was 2.175 (below break-even).
5. Power calc: N for clustered half-width <=0.05 is 256 (need +16 over the 240 measured; current 0.052
   just MISSES the 0.05 floor), <=0.028 is 816.

PREMISES / SCOPE (exogenous; assess how the conclusions swing if a premise fails, do not call them bugs):
- P1: the torch/MPS port at 100% draft-token agreement on 15 prompts is a valid stand-in for the numpy
  oracle across all 240 prompts (could a subtle divergence hide in the un-sampled 225?).
- P2: the Stage 2 corpus (dolly/codealpaca/jsonex) is representative of the deployment workload the
  speedup model targets.
- P3: the trajectory partition (greedy-spine, prior-step draft[0] match) is a valid proxy for real
  speculative-decode trajectory bias.

YOUR MANDATE (prioritize runnable checks over assertion):
1. PRECISION SOUNDNESS: is the 100% torch-vs-numpy agreement on 15 prompts real, or could a subset of
   the 240 diverge? Spot-check: re-derive a few per-prompt E_a_5block from the stored drafts vs the
   stored E_a_5block (internal consistency); optionally re-measure 2-3 of the 240 via the numpy oracle
   (single-process, DS4_EXPERT_MAX_CACHE=64) and compare drafts. Is the absolute scale trustworthy?
2. THE HEADLINE: is "K=4 break-even SIGNED POSITIVE" justified? The CI half-width (0.052) MISSES the
   0.05 floor; the trajectory bias pulls E[a|4] toward break-even; dolly is already ~at break-even;
   the old code/synthesis corpus was below. Does the corpus-composition variance (jsonex 2.51 vs dolly
   2.22 vs old code/synthesis 2.175) undermine a single "E[a|4]=2.366, beats break-even" headline?
3. POWER CALC + N: re-derive the prompt-clustered CI and the N formula from the per-prompt checkpoints
   (prefix_hist). Is sd=0.408 / N=256/816 correct? Is the clustered bootstrap right (resample prompts,
   not steps)? Is treating the 0.052 half-width as "just misses the floor" honest?
4. TRAJECTORY: is the partition indexing correct (step t correction-anchored iff prior step draft[0] !=
   anchor)? Re-derive from the drafts+target stream for one prompt. Does "correction-anchor E[a|4]=1.994"
   actually mean real-trajectory acceptance is lower, or is the greedy-spine mix (79/21) already the
   right weighting? What's the honest trajectory-weighted E[a|4] range?
5. SCOPE/CAPTURE DECISION: given (a) the floor isn't met at 240 (need 256), (b) the break-even margin is
   0.09 (already clear), (c) corpus variance is large — is capturing to N=816 (the +/-0.028 stretch)
   worthwhile, or should capture instead target CORPUS DIVERSITY (code/synthesis/grounded, which the
   Stage-2 corpus lacks) over raw N?

CONSTRAINTS: read-only; do NOT load the 81 GB expert npz or run ds4. Per-prompt drafts + a numpy
single-process spot-check (DS4_EXPERT_MAX_CACHE=64) are the heaviest things you should run.

OUTPUT (terse, evidence-based):
## Verdict per claim (sound / questionable / likely-wrong + 1-line why)
## Is the K=4 "break-even signed positive" headline honest, or overclaim? (definite read)
## Corrected E[a|4] picture (greedy vs trajectory-weighted range; corpus-mix sensitivity)
## Capture recommendation (N target + diversity vs raw-N, with reasoning)
## Decisive test (the one check that confirms/refutes the headline)
