You are a skeptical senior ML engineer doing an ADVERSARIAL independent review of a
*plan* (not yet executed). Challenge the approach, recommend on the open questions,
catch methodological bugs. Be terse. Your decisive claims will be INDEPENDENTLY
re-verified by the dispatcher before acceptance, so prioritize runnable checks and
precise citations.

WORKTREE: /Users/lobanov/Projects/ds4-dspark-research. Python envs:
`issue468/dspark_train/.venv` (torch 2.12.1 MPS, numpy, pyarrow, safetensors),
`issue468/dspark_oracle/.venv` (numpy oracle). Engine repo: /Users/lobanov/Projects/ds4.

READ FIRST (the plan under review): `issue468/summaries/stage2_finetune_protocol.md`

CONTEXT (already codex-reviewed, take as valid premises — do NOT re-litigate;
assess how the plan SWINGS if a premise is weaker than claimed):
- Stage 0 (`issue468/summaries/quant_mismatch_diagnostic.md`): DSpark drafter p=1
  misses vs IQ2XXS target are shallow — median target-rank 1.0; 100% within top-10;
  top-2 coverage 0.8125→0.9125; only ~27% are Q2 near-ties.
- Stage 1 (`issue468/summaries/stage1_tap_precision.md`): raising layers 37–42 to
  Q4_K did not materially improve p=1 (0.8125→0.7875, underpowered).
- Speedup model (`issue468/summaries/spec_speedup_model.md`): beating baseline at
  K=4 needs E[a|4]≈2.20 (current 2.175); +20% gate needs ~0.94 p-acceptance (K=4);
  the shipped --mtp verifier pays a redundant anchor decode (~−18 pp at K=4).
- Drafter architecture (`issue468/dspark_oracle/forward.py`,
  `measure_acceptance_bundle.py`): borrows the TARGET's embed_w + lm_head; own
  output path = hc_head + norm + markov_w1/w2 (33M each). 518.5M dense + frozen
  routed MoE experts. main_hidden = concat(layers 40/41/42), 12288-dim.

RECAP OF THE PLAN: A bounded PoC to decide whether fine-tuning the drafter can
materially raise p=1 acceptance. Staged: **2a head-only LoRA** on precomputed
frozen-drafter features (hc_head+markov+norm), then **2b full-drafter LoRA**
(main_proj+dense attn/FFN-shexp+head, experts frozen) if 2a stalls. Decision rule:
held-out p=1 ≥+5pp ⇒ reaffirm; <+1pp (with overfit-to-eval passing) ⇒ falsify.
Data: ~180 train / ~30 eval prompts sampled from Dolly-15k + CodeAlpaca-Python +
json-extraction, captured via a new `--capture-dataset` ds4 mode (load-once loop +
multi-layer patch) storing few-file sharded safetensors. Eval on ≥1k p=1 positions.

THE PLAN'S CLAIMS / CHOICES (verify, do not trust):
- P1 DECISION RULE: +5pp reaffirm / <+1pp falsify is conclusive for "would a more
  extensive fine-tune help?", given the overfit-to-eval sanity check.
- P2 STAGED 2a→2b: head-only first is a fair, fast decisive probe; 2b covers the
  input-side. Together they conclusively falsify or reaffirm.
- P3 HEAD-ONLY CEILING = Stage 0 top-2 coverage (~0.91): a retrained head "can't
  beat 0.91"; pushing toward 0.88–0.91 ⇒ output-ordering deficit.
- P4 FROZEN-EXPERTS BOUNDARY: a null with experts frozen falsifies "drafter
  fine-tuning" sufficiently (Stage 2c expert-tuning left as a possible extra).
- P5 FEW-FILE CAPTURE: in-engine per-prompt buffer → sharded safetensors is sound
  and avoids ~57k .bin; the multi-layer comma-list patch is trivial+safe.
- P6 FIDELITY GATE: torch at LoRA=0 reproducing numpy draft tokens ≈100% is
  sufficient to trust training results.
- P7 STATISTICAL POWER: ≥1k (planned ~3.6k) held-out p=1 positions detects +5pp.
- P8 SOFT vs HARD: hard-CE + label smoothing primary is the right default.

YOUR MANDATE — challenge the approach; recommend on the open questions; find bugs.
Decisive claims must be runnable/citable. Not exhaustive:
1. CONCLUSIVENESS (P1/P4): is a bounded LoRA PoC (experts frozen) able to
   *conclusively falsify* "a more extensive fine-tune would help"? Enumerate the
   escape hatches a null would leave (expert tuning, full Q8/FP target, more data,
   higher rank, longer training). Is the overfit-to-eval check enough to close
   them? What would make "stop" actually trustworthy?
2. THE HEAD-ONLY CEILING (P3): "a retrained head can't beat the frozen top-2
   coverage 0.91" — is this right? A retrained head produces a DIFFERENT
   distribution (not the frozen one); couldn't it move the target from rank 5→1 if
   the frozen pre-head features h carry the info? Is 0.91 a real ceiling or a loose
   proxy? What is the *actual* ceiling of head-only retraining (e.g. a from-scratch
   linear head on h, trained to convergence)?
3. THE 2a→2b LOGIC (P2): if 2a stalls, does 2b genuinely isolate "input-side" vs
   "output-side"? Could 2b's extra capacity (more LoRA params) be the real driver
   rather than input adaptation? Is there a cleaner ablation?
4. THE METRIC (P7 + exposure bias): p=1 is clean (prev=anchor). But the PoC trains
   on K=5 teacher-forced and decides on p=1. Is p=1 improvement a reliable proxy
   for deployment acceptance (which includes rollout + the verifier)? Is ≥1k/3.6k
   positions enough — recompute the power for +5pp and +3pp at α=0.05 (McNemar).
5. SOFT vs HARD LABELS (P8): for the SPECIFIC goal (raise argmax match rate, exact
   greedy preservation), is hard-CE optimal, or does top-K KL/distillation give
   better generalization that matters here? Recommend concretely.
6. FROZEN lm_head/embed (architecture): the drafter borrows the target's lm_head.
   Training hc_head (pre-lm_head) + markov but freezing lm_head — any risk the
   fine-tune fights the frozen lm_head's geometry? Is there a better parameterization?
7. CAPTURE/STORAGE (P5): is the few-file safetensors schema complete for training
   (K-step targets, prompt boundaries, soft labels)? Any correctness risk in the
   in-engine buffer vs the proven per-(layer,pos) dump? Is the multi-layer patch
   (comma-list in metal_graph_debug_wants, ds4.c:10827) actually safe given the
   "dumping synchronizes and restarts the command batch" comment?
8. CORPUS REPRESENTATIVENESS: instruction-tuning datasets (Dolly/CodeAlpaca/json)
   vs deployment (local single-request chat decode). Is the eval a fair proxy? Any
   genre that's missing/over-weighted vs the exactness corpus?
9. ALTERNATIVES THE PLAN MISSES: e.g. is there a cheaper/more decisive test than
   the full PoC (the crossed-oracle diagnostic; a from-scratch head upper-bound on
   frozen features; distilling on a Q8 target if one existed)? Should any precede
   the LoRA training?

CONSTRAINTS: read-only; run python -c / rg / jq to verify. Do NOT run ds4 or
training. The torch + numpy venvs are available.

OUTPUT (terse, evidence-based):
## Verdict per claim P1..P8 (sound / questionable / likely-wrong + 1-line why)
## Open-questions recommendations (for each of the plan's 8 open questions: a
   concrete recommendation + confidence)
## Methodological bugs / risks found (ranked; each: the issue, the fix, effort)
## Cheaper/more-decisive alternatives the plan should consider
## Bottom line: does the plan, as written, produce a TRUSTWORTHY reaffirm/falsify?
   What is the single highest-value change before execution?
