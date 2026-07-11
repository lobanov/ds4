# Stage 2 — bounded drafter fine-tune PoC (plan)

Date: 2026-07-06. Status: **planning — setup decisions locked; activities below are
tasks, not yet executed.** Purpose: decide whether **non-expert** fine-tuning of the
DSpark drafter can materially raise p=1 acceptance vs the IQ2XXS target. Follow-on to
`summaries/quant_mismatch_recommendation.md` (NARROW).

Revised with feedback from an adversarial codex/gpt-5.5 review (prompt, verdicts, and
trace retained at `issue468/artifacts/stage2_plan_review/`).

## Scope

- **In scope:** drafter fine-tuning with the routed MoE **experts frozen** (the Q4_K
  bulk) — head + dense-body LoRA / retraining. Decision metric: held-out p=1 argmax
  acceptance vs the IQ2XXS target.
- **Out of scope (explored separately, regardless of this PoC's outcome):** expert
  tuning — the expensive part and a separate workstream; not a hedge on the verdict.
- This PoC measures **acceptance only**. A throughput reaffirm would additionally
  require a verifier much cheaper than the current exact anchor-reuse path
  (`spec_speedup_model.md`), so acceptance
  gain is treated as necessary-but-not-sufficient for the GOAL's throughput gate.

## Decision rule

Primary metric: **held-out p=1 argmax match rate** (clean, no-rollout). Eval set:
**~60 prompts → ≥3.6 k p=1 positions** (McNemar paired test; detects +5 pp at α≈0.05
comfortably, +3 pp at the margin). Decision condition is **temp=0 (greedy)**; a
temp>0 eval is secondary if deployment samples. Report p=1 primary, prefix-hist /
E[a|4,5] secondary under the existing speed model.

- **Reaffirm:** held-out p=1 **≥ +5 pp** with the overfit-to-eval and fidelity gates
  passing ⇒ non-expert fine-tuning works; scale data + capacity.
- **Not justified:** p=1 still **< +1 pp** after the **Activity 6** gate AND rank
  32/64/128 AND a data/seed sweep ⇒ non-expert drafter fine-tuning does not justify
  scale-up.
- **Diagnostic:** the Activity 6 upper bound and the Activity 7-vs-8 contrast localize
  *where* the deficit sits (head-side / input-side / upstream of the frozen body) —
  often more informative than the single verdict.

Within its scope the verdict is conclusive; expert tuning is out of scope.

## Setup

1. **Engine change** — efficient `--capture-dataset` ds4 mode (engine repo, this
   branch): load-once loop mirroring `--imatrix-dataset` (ds4.c:~25060) +
   `metal_graph_reset_prefill_state` per prompt + a **multi-layer comma-list patch** to
   `metal_graph_debug_wants` (ds4.c:10827; currently single `strtoul`/`"all"`). ~9×
   faster capture (~110 min → ~13 min). A **golden-parity test** is required (new
   multi-layer capture vs the proven per-layer dumps on 2 prompts, all
   layers/positions/tokens/topk), since "dumping synchronizes and restarts the command
   batch" (ds4.c:10816).
2. **Few-file sharded storage** — in-engine per-prompt buffer → consolidated
   **safetensors** shards (no ~57 k `.bin`); schema below.
3. **Machine: 128 GB unified memory**, no competing workloads.
4. **Corpus = three datasets** in `issue468/data/corpus_source/` (gitignored) —
   Dolly-15k (general, 8 categories), CodeAlpaca-Python (code), json-extraction
   (structured). **Final for this PoC.**
5. **Training env: `issue468/dspark_train/`** — torch 2.12.1 MPS verified + numpy,
   pyarrow, safetensors, tqdm (CPython 3.14).

## Architecture — trainable vs frozen

518.5 M dense + 9 packed routed-expert groups. The drafter borrows the **target's
`embed_w` + `lm_head` (frozen)**; its own output path is `hc_head` + `norm` +
`markov_w1`/`markov_w2` (33 M each). Per the DSpark paper, training updates the
backbone + sequential block + head (embed/lm_head frozen); **this PoC additionally
freezes the routed experts** (the scope boundary above).

## Loss design (DSpark §3.3)

`L = Lce + Ltv` (the paper's `Lconf` is dropped — the confidence head serves the
scheduler's admission, not argmax acceptance). **Exponential position weights**
`w_k = exp(−(k−1)/γ)`, γ=5, emphasize p=1 — the decision metric and highest-leverage
position.
- **Lce**: cross-entropy to the target argmax (teacher-forced prev tokens). No label
  smoothing.
- **Ltv**: total-variation distance between draft and target distributions — *directly
  maximizes expected acceptance* (paper). Computed over the captured **top-128
  logprobs** (+ tail bucket); TV is the principled loss for this metric.

## Activities (tasks — none executed)

- **Activity 1 — Sample the corpus.** ~60/source → 180 train; ~60 eval prompts; 128
  tok. Stratified; tracked `prompts/stage2_corpus/`. Keep the 10 exactness prompts as
  a locked legacy benchmark.
- **Activity 2 — Build `--capture-dataset` + multi-layer patch + golden-parity test**
  (engine repo; research instrumentation).
- **Activity 3 — Capture data → sharded safetensors** (~13 min; 180 prompts × 128 tok).
- **Activity 4 — Crossed hidden/label oracle (mandatory).** From existing IQ2XXS +
  Q4-tap captures: Q4-hidden + IQ2XXS-tokens and IQ2XXS-hidden + Q4-tokens, re-measure.
  Separates input-shift (representation) from label-shift (argmax drift) *before*
  training and tells the LoRA where to focus.
- **Activity 5 — Torch port + fidelity gate.** Port `dspark_oracle/forward.py`→torch
  (MPS); experts dequant to F16 frozen. **Gate: torch vs numpy full-score + top-128
  parity (not argmax-only)** on the exactness corpus; no-LoRA eval reproduces baseline p=1.
- **Activity 6 — From-scratch head upper bound [gate before LoRA].** Train a
  from-scratch head (hc_head+markov, full-rank) on the frozen pre-head features `h` to
  convergence; report held-out p=1. This measures the **true head-only ceiling**
  directly — the Stage-0 top-2 coverage (~0.91) is only a loose indicator, not a
  ceiling, since a retrained head produces a different distribution and could move the
  target from rank-5→1 if `h` carries the info. If it caps low (~0.83), `h` lacks the
  info ⇒ the deficit is upstream of the head ⇒ move toward Activity 8. If it is high
  (~0.90) but Activity 7 is low, LoRA rank is the bottleneck.
- **Activity 7 — Head-only LoRA** (hc_head + markov + norm; rank 32; cheap; compare to
  the Activity 6 ceiling).
- **Activity 8 — Full-drafter LoRA** if Activity 7 < Activity 6 ceiling (add main_proj
  + dense attn/FFN-shexp). **Ablate to isolate input-side** — head-only /
  main_proj-only / dense-block-only / head+main_proj / head+blocks at matched rank —
  so the gain isn't confounded with extra capacity. Rank 32, then 64/128 rescue if needed.
- **Activity 9 — Eval + decision.** McNemar paired p=1; apply the decision rule.

## Storage format

Sharded **safetensors**. Per shard: `main_hidden [N,12288] f16`; `token_ids [N] i32`
(target greedy stream; K-step targets via windowing); `topk_ids [N,128] i32` +
**`topk_logprobs [N,128] f16`** (+ optional `topk_logits`) for Ltv; `selected_id [N]
i32`; `prompt_idx [N] i32`; `pos_in_prompt [N] i32`. Plus `index.json` (prompt→source,
text ref, prompt_tokens). **Explicit no-cross-prompt windowing** at train time. Layout:
train ~4 shards, eval 1 shard + index.

## Sanity / fidelity checks

- **Overfit-to-eval** (train on eval; p=1→~1.0; proves the port + capacity can fit).
- **Activity 6 upper bound** (caps the head-only story before any LoRA).
- **Fidelity gate:** torch-vs-numpy **full-score + top-128 parity**, not argmax-only.
- **Capture golden-parity:** new multi-layer mode vs proven per-layer dumps (2 prompts).
- **No-LoRA baseline** reproduces ~0.81 p=1 on the new eval set.

## Budget (128 GB box)

| activity | wall-time |
|---|---|
| 1 sample corpus | minutes |
| 2 build capture mode + multi-layer patch + golden test | ~0.5–1 day |
| 3 capture (180 prompts × 128 tok) | ~13 min |
| 4 crossed-oracle diagnostic | minutes |
| 5 torch port + fidelity gate | ~0.5–1 day (main eng risk) |
| 6 from-scratch head upper bound | minutes–1 h |
| 7 head-only LoRA | minutes–1 h |
| 8 full-drafter LoRA (+ ablations) | a few h |
| 9 eval + decision | minutes |

## What each outcome means

- **Reaffirm:** scale up — larger corpus, higher rank, integrate with the anchor-reuse
  verifier; realistic target the secondary gate.
- **Not justified:** close the non-expert fine-tuning direction; redirect to
  server-side batching (amortize the verify bandwidth floor). Expert tuning remains a
  separate exploration, independent of this result.

## Execution shape

When greenlit, run as a **staged `/sisyphus` goal**: Activity 1 → 2 → 3 → 4 → 5 →
**6 (gate)** → 7 → (8 + ablations if needed) → 9 + decision.
