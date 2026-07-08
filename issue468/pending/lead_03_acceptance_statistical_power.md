# Lead 03 — Acceptance statistical power + realistic-trajectory measurement

Date: 2026-07-07. Status: **in progress** (worklog; goal `mras0t7x-e5byoo`). De-risks
every other lead; can run in parallel with leads 01/02.

## Rationale

Every decision number in this dossier rests on 10 prompts at temp 0 (80 measure
steps). The adversarial review of the speedup model quantified the consequence:
E[a|4] has prompt-level sd ≈ 0.43 (95% half-width ±0.3) against a **0.028**
drafts/cycle break-even gap; per-prompt modeled K=4 spans −18.5% to +10.6%. The
headline "~2 pp short of beating baseline" is therefore inside the noise — and so
would be "1 pp over." Stage 1 was likewise declared underpowered (bootstrap CI
[−0.16, +0.41] straddling zero). No stop/go decision on this branch is defensible
at n=10.

A second, independent bias is unmeasured: all acceptance was taken along the
drafter's own greedy spine. Real speculative trajectories restart each cycle from
a *correction* token the drafter just mispredicted; acceptance on post-rejection
cycles is plausibly lower (the model's own caveat). The direction and size of
this bias directly moves the break-even arithmetic and is measurable with the
existing protocol.

## Content of work

1. **Widen the corpus.** Extend the prompt corpus to 100+ prompts spanning the
   existing families (code / synthesis / grounded / creative) and length classes;
   re-run the capture + acceptance pipeline (`run_exactness_small_bundles.py` +
   `dspark_oracle/measure_acceptance_bundle.py`) at temp 0, with a 0.5/1.0 subset.
   The tooling exists; this is compute time, not new code.
2. **Realistic-trajectory acceptance.** From the captures, measure acceptance
   conditioned on cycle type: cycles anchored on a token the drafter had
   predicted vs cycles anchored on a correction (i.e., positions immediately
   following a p=1 miss). Report E[a|K] and S(K) for both populations and the
   trajectory-weighted mixture.
3. Re-emit the speedup-model inputs (`artifacts/spec_speedup_model/model_inputs.json`)
   from the widened corpus with CIs tight enough that the K=4 break-even verdict
   has a sign.

Estimated effort: mostly unattended compute; ~1–2 days of attended work.

## Success criteria

- 95% CI half-width on E[a|4] ≤ ~0.05 drafts/cycle (vs ±0.3 today), so the
  fixed-K break-even question and the lead 02 policy simulation both get a
  signed answer instead of a band.
- Post-rejection-cycle acceptance quantified; the speedup model updated with the
  trajectory-weighted acceptance (if it drops E[a|4] materially, that is itself
  a decision-grade negative finding and should be recorded as such).
- Updated per-position histogram feeding leads 01/02 re-runs at scale.

---

## Worklog (goal `mras0t7x-e5byoo`)

### 2026-07-07 — task-1 start: converter + fidelity + power calc

**Corpus asset confirmed.** Stage 2 shards (`issue468/dspark_train/data/shards/`):
6 safetensors, 29,160 positions across 240 prompts (temp=0, 128-tok generations).
Schema (per `index.json`): `main_hidden [N,12288] f16`, `token_ids [N] i32`,
`prompt_idx [N] i32`, `pos_in_prompt [N] i32`; per-prompt `prompt_tokens`,
`pos_range [P, P+128]`, `source` (dolly/codealpaca/jsonex ×80 each), `split`.

**Index-note semantics (verified vs measure_acceptance_bundle, no off-by-one):**
"anchor = token_ids at p; predicts token[p+1..p+K]; drafter input = main_hidden[p]"
— i.e. main_hidden[pos] is the POST-token hidden at absolute position pos, and
token_ids at pos_in_prompt=j is the target token at pos0+j. This is exactly the
bundle convention (Lead 01 confirmed main_hidden is post-token via dump_hc_ffn_post).

**Storage decision (few-file):** measurement reads main_hidden directly from the
6 shards via a `Stage2CaptureStore` (additive load path in measure_acceptance_bundle;
bundle-dir baseline path UNCHANGED -> fidelity-gated). No 29k per-position .npy /
240 bundle dirs materialized.

**F16 precision:** Stage 2 stores main_hidden as F16 (vs exactness F32). Fidelity
gate (task-1b): re-capture 2-3 prompts at F32 via the exactness path, compare
per-prompt E[a|4]/p1.

(to be appended: N calc, capture log, per-task outcomes)

### 2026-07-07 — task-1b: F16-vs-F32 fidelity gate PASSED

Re-captured 2 Stage 2 prompts (codealpaca_0000, dolly_0000) at TRUE F32 via the
exactness path (./ds4 + DS4_METAL_GRAPH_DUMP, 3 per-layer runs) and compared to
their F16-stored Stage 2 captures: **draft-token agreement 100% on both prompts**,
main_hidden max rel diff ~0.0003, d_avg_prefix <= 0.006. F16 storage does NOT
materially shift drafter acceptance -> Stage 2 F16 captures usable as-is.
Artifact: `artifacts/acceptance_powered/f32_fidelity/f32_fidelity.json`.

### 2026-07-07 — task-2: --capture-dataset committed on dspark-research

**Branch decision (user):** all research work in the `dspark-research` worktree
(`/Users/lobanov/Projects/ds4-dspark-research`), consistent with all other
issue468 commits. The patch had been left uncommitted in the MAIN worktree
(`/Users/lobanov/Projects/ds4`, branch `local-gen-with-dist-prefill`) by Stage 2.

- Applied the patch (ds4.c comma-list layer parse; ds4_cli.c `run_capture_dataset` +
  `--capture-dataset/--capture-out/--capture-layers`; ds4_help.c docs) to the research
  worktree source; built the research binary; **golden-parity VERIFIED on the research
  binary: 42/42 .bin byte-identical** to the proven per-layer path (+ token streams
  identical) on 2 prompts.
- Committed: `3510497 issue468: commit --capture-dataset engine mode (Lead 03)`
  on `dspark-research`.
- Reverted the main worktree's uncommitted copy + rebuilt it clean (capture-dataset
  no longer in its --help).
- **task-3 capture must use the research binary**
  `/Users/lobanov/Projects/ds4-dspark-research/ds4` (NOT the main-worktree binary,
  which no longer has the patch).

**Memory incident (recorded for the worklog):** the machine restarted mid-run when
the 6-worker Python measurement (~45 GB) ran CONCURRENTLY with ds4 loading the 87 GB
IQ2XXS model -> oversubscribed 128 GB. **Rule going forward: never run the Python
measurement workers and ds4 (87 GB model) concurrently; ds4 runs alone; the Python
measurement runs alone with fewer workers + memory monitoring.**

### task-1a: converter (Stage2CaptureStore) DONE

`dspark_oracle/stage2_capture_store.py` reads the 6 shards directly (few-file;
no per-position .npy materialized). Additive `store` path in `measure_acceptance_bundle`
(bundle-dir baseline UNCHANGED -> bit-for-bit reproduces code_histogram__t0p0 retained
summary). Store path verified on Stage 2 prompts (codealpaca_0000: p1=0.828, 122 steps).

### 2026-07-07 — task-1c: torch port SEVERELY precision-degraded vs numpy (codex bug-hunt triggered)

After the memory-driven goal tweak (measure via torch/MPS instead of the numpy
oracle), built the torch acceptance harness (`run_lead03_torch_measure.py`) on the
Stage 2 port (drafter_body.py + drafter_head.py, F16 MPS). Precision gate vs the
numpy oracle FAILED hard:
- **Draft-token gate (F16 MPS vs live numpy):** mean step-agree **17.2%**, token-agree
  **43.1%**; torch E[a|5] roughly HALVED (0.59–1.97 vs numpy 2.2–3.2). Required ≥99%.
- **Body-fidelity gate (CPU F32 torch-x vs stored numpy-x, run_body_fidelity.py):**
  mean abs diff **2.09**, max **174** (criteria < 2e-3) — a STRUCTURAL divergence, not
  F16/precision. Reconciles with Stage 2's "layer-0 ~2e-4, deeper-layer chaos accepted
  as inherent" — the chaos is severe enough to flip draft tokens, which Stage 2 could
  ignore (self-consistent use) but Lead 03 cannot (needs the numpy-absolute scale).

Memory was fine (torch F16 MPS ~43 GB + numpy ~10 GB fit; free 101 GB after). The
blocker is PRECISION, not memory. Per the contract: adversarial codex bug-hunt on the
torch body port to localize fixable bug vs inherent chaos; if unfixable, STOP + ask user.

### 2026-07-07 — task-1c: codex bug-hunt FOUND + FIXED the torch body bug; F32 MPS now 100%

Codex bug-hunt (gpt-5.5 xhigh) localized the divergence to a FIXABLE porting bug in
`drafter_body.py::hc_post`: torch used `residual.unsqueeze(-3)` but the numpy oracle
(`hc_primitives.hc_post`) uses `residual[..., None, :]` = `unsqueeze(-2)` — a broadcast-
axis error (mixes the residual across the HC dim). Tiny at L0, amplified to max 113 by L2.
**Independently verified** (numpy `r[...,None,:]` shape == `unsqueeze(-2)`, NOT `-3`),
then applied the fix and re-ran:
- Body-fidelity gate (F32 CPU): now **PASS** (max abs diff 0.00125 < 2e-3; was 174).
- F16 MPS draft-token gate: still ~43% (F16 breaks the precision-sensitive head/lm_head
  argmax over 129k vocab) -> **use F32, not F16**.
- **F32 MPS draft-token gate: 100% step-agree + 100% token-agree on 6 prompts** — the
  torch port (F32 MPS) reproduces the numpy oracle EXACTLY. Memory ~80 GB single-process
  (F32 experts via mmap npz), feasible. Stage 2's "deeper-layer BLAS chaos" was in fact
  THIS bug mis-attributed to BLAS.

Fix committed to `drafter_body.py` (hc_post: unsqueeze(-3) -> unsqueeze(-2)). Powered
measurement (task-1d) will use **F32 MPS**, single process, no concurrent ds4/numpy.

### 2026-07-07 — task-1d: 240-prompt torch measurement DONE; K=4 break-even now SIGNED

F32 MPS measurement of all 240 Stage 2 prompts (27,720 cycles), single process, ~54 min
wall (13.6 s/prompt), swap steady ~3-4 GB. Aggregate (artifacts/acceptance_powered/
stage2_torch_measure/aggregate.json):
- **E[a|4] = 2.345** (E[a|5]=2.646), p1 = 0.793
- per-position match: [0.790, 0.683, 0.567, 0.452, 0.355]; S(k)=[.790,.653,.516,.385,.273]
- **prompt-level sd = 0.408** — the 128-tok generations did NOT collapse the sd vs the old
  10-prompt 0.43; real prompt-to-prompt heterogeneity dominates (not within-prompt noise).
- **clustered 95% CI on E[a|4]: [2.315, 2.419], half-width 0.052** (just over the 0.05 floor).
- power calc: N_floor(±0.05)=256, N_stretch(±0.028)=816.

**HEADLINE CHANGE:** the model's K=4 break-even is E[a|4]=2.203 (beat baseline with the
optimized/anchor-reuse verifier). Powered CI [2.315,2.419] is ENTIRELY above 2.203 -> the
K=4 break-even is now SIGNED positive (beats baseline), at ±0.052. The old "~2 pp short /
E[a|4]=2.175" was the 10 harder EXACTNESS prompts (code/synthesis); the 240-prompt
dolly/codealpaca/jsonex corpus has HIGHER deep-position acceptance (S(3)=.52, S(4)=.385 vs
old .43/.29) -> E[a|4]=2.345. **Caveat:** corpus-composition driven (Stage 2 families may be
easier than deployment); the trajectory partition (task-4) and codex gate will scrutinize.

Implication for task-3: the break-even is already signed at n=240 (margin 0.09 >> CI 0.052),
so the ±0.028 stretch (N=816) is over-powered for THIS question; capturing to ~300 clears the
≤0.05 floor. Will confirm scope with the user at task-3.

### 2026-07-07 — GATE 1 (codex): CORRECTED a headline overclaim (cycle-jump vs sliding)

Codex GATE 1 (gpt-5.5 xhigh) found my "K=4 break-even SIGNED POSITIVE" headline was an
OVERCLAIM based on the wrong acceptance estimator. **Independently verified + fixed:**
- My sliding-position E[a|4]=2.345 overestimates: it uniformly samples positions,
  under-weighting post-rejection correction cycles. Real speculative decode advances by
  (accepted+1) per cycle. The correct estimator is a **K-cycle-jump simulation** from drafts.
- Cycle-jump (cycle-pooled, the model's per-cycle currency): **E[a|4]=2.202, S(4)=0.344,
  speedup=0.982x (-1.8%, BELOW baseline)** — at/slightly-below the 2.203 break-even, NOT
  positive. (Per-prompt-mean E[a|4]=2.282; pooled=2.202 exactly matches codex's number.)
- Per-source cycle-jump speedup: jsonex +2.8%, codealpaca -2.2%, **dolly -5.6%**.
  Corpus mix dominates the verdict.
- Fixes applied: (1) `run_lead03_cyclejump.py` — the corrected trajectory artifact (replaces
  the sliding `run_lead03_trajectory.py` as the model input); (2) precision gate re-saved at
  10 prompts 100% (codex noted only 6 saved) + 15-prompt run earlier = 25 total, all 100%;
  (3) headline corrected.
- Codex capture advice: do NOT chase N=816 on the same Stage-2 mix (shrinks CI around an
  easy-corpus estimator); capture to ~256-300 to clear the ≤0.05 floor, OR spend on corpus
  DIVERSITY (code/synthesis/grounded, which corpus_source lacks).

**Corrected Lead 03 finding:** K=4 is at/slightly-below break-even (-1.8%) under the realistic
cycle-jump trajectory, corpus-mix-dependent (jsonex positive, dolly negative). The verdict is
CORPUS-limited, not statistical-precision-limited — so more N from the same 3 families won't
change it. artifacts: stage2_torch_measure/{aggregate,trajectory,cyclejump}.json +
torch_measure/torch_precision_gate.json.

### 2026-07-07 — task-3/4: captured +60 (to 300), floor MET; cycle-jump verdict holds

Captured 60 new prompts (dolly/codealpaca/jsonex x20, ids 0080-0099, non-overlapping) via
the committed --capture-dataset (research binary, ~5min) + consolidated to lead3_new_shards
(2 shards, 7300 pos; 22k scratch .bin deleted). Measured the combined 300 via torch F32 MPS
(Stage2CaptureStore extended to merge shard dirs). Results (artifacts/acceptance_powered/
combined300/{aggregate,cyclejump}.json):
- **Sliding E[a|4]=2.362, clustered CI half-width 0.0463 (< 0.05 -> FLOOR MET)**, sd=0.418.
- **Cycle-jump (realistic) E[a|4]=2.198, S(4)=0.340, speedup=0.982x (-1.8%, below baseline)**
  — confirms the 240-prompt cycle-jump verdict at n=300 with the floor cleared.
- Per-source cycle-jump speedup: jsonex +2.3%, codealpaca -1.9%, dolly -5.6%.

**Decision-grade Lead 03 finding:** under the realistic cycle-jump trajectory, K=4 gives
speedup ~0.98x (at/slightly-below break-even), corpus-mix-dependent. The sliding estimator
(used by the model + my initial overclaim) is optimistic by ~3pp (sliding +1.3% vs cycle-jump
-1.8%). The model refresh (task-6) will use the cycle-jump E[a|4]=2.198 / S(4)=0.340 and
flag the sliding-vs-cycle-jump distinction.

### 2026-07-07 — task-6: model refreshed (sliding + cycle-jump, corrected verdict)

Re-ran model_spec_speedup.py with the 300-prompt powered SLIDING prefix_hist
(HIST5 updated; assert fixed). model_inputs.json now carries BOTH:
- sliding (the Q1/Q2/Q3 currency): E[a|4]=2.337, S(4)=0.382 -> K=4 speedup **+1.2%**.
- lead03_cyclejump_realistic: E[a|4]=2.198, S(4)=0.340 -> K=4 speedup **0.982x (-1.8%)**.
spec_speedup_model.md: added a prominent "Lead 03 powered refresh" section flagging that
the sliding estimator is OPTIMISTIC (~3pp), the realistic cycle-jump edge is 0.982x
(below baseline), and the K=4 verdict is corpus-dependent (jsonex +2.3%, dolly -5.6%).
Diff vs prior model_inputs recorded (25 insertions).

### 2026-07-07 — GATE 2 (codex): RECORD with edits; verified the stronger verdict

GATE 2 (gpt-5.5 xhigh): RECORD with edits. Independently verified the decisive claims:
- cycle-jump K=4 speedup=0.9822x, bootstrap CI95=[0.9711,0.9931], **P(speed<1)=0.9990**
  (codex 0.9989) -> significantly below baseline, not just "at break-even."
- **dynamic break-even E[a|4]=2.256** (at S(4)=0.340), NOT the stale 2.203 (sliding-S);
  deficit 0.058, not 0.005. My initial framing understated the deficit.
Edits applied to spec_speedup_model.md: corrected break-even framing (2.256/deficit 0.058/
P<1=0.999), labelled sliding as diagnostic-only/optimistic, added the drafter-state-pollution
+ Lead-05 scope caveat. Per-source already 300-prompt (jsonex+2.3%, codealpaca-1.9%, dolly-5.6%).
Downstream stale -0.9% language flagged in lead_05 (sweep in writeup). Reviews retained at
artifacts/acceptance_powered/codex_reviews/.
