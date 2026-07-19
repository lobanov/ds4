---
name: pytorch-numerical-modelling
description: Patterns + pitfalls for PyTorch-based numerical modelling in the ds4-dspark-research dossier (the drafter oracle, the acceptance measurement, the cycle-economics). Use before writing or adapting torch code that loads GGUF weights, runs the drafter forward, computes acceptance/economics, or compares to the live engine.
---

# PyTorch numerical modelling

A reference for the PyTorch patterns + the pitfalls learned across Leads 03–11 in
the ds4-dspark-research dossier. The torch code (`dspark_train/drafter_body.py` +
`drafter_head.py`) + the numpy oracle (`dspark_oracle/forward.py`) are the canonical
implementations; this skill captures how to use them correctly + what goes wrong.

Pair with `research-lead` (the methodology) + `adversarial-codex-review` (the gates).

## When to use

- Writing or adapting torch code that loads the drafter weights from the GGUF + runs
  the forward (the drafter oracle, the acceptance measurement, the crossed oracle).
- Computing acceptance / block-acceptance / E[a|K] / the cycle-economics from captured
  hidden states.
- Comparing the offline drafter (torch) to the live engine (Metal) — the dtype +
  the block-divergence pitfalls.
- Running a bootstrap CI on a per-prompt metric.

## Prerequisites

- `issue468/.venv/bin/python` (numpy, torch with MPS).
- The drafter GGUF: `/Users/lobanov/Projects/ds4/gguf/dspark.gguf` (~10.7 GB).
- The target GGUF (for the embed): the IQ2XXS target at
  `/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-...gguf` (~87 GB).
- `issue468/dspark_train/drafter_body.py` + `drafter_head.py` (the torch drafter).
- `issue468/dspark_oracle/forward.py` (the numpy reference oracle).
- `issue468/dspark_oracle/gguf_loader.py` (the GGUF loader).
- Apple Silicon MPS backend (the `dev='mps'` path).

## Core principles (the load-bearing lessons)

1. **Fidelity-gate the torch port against the numpy oracle (bit-token-for-token).**
   Before trusting ANY torch drafter number, reproduce the numpy oracle's drafts at
   100% draft-token agreement on a known-good input. The torch port had a real
   `hc_post` broadcast-axis bug (Lead 03) that made it 43% wrong until the gate
   caught it. The gate: run both on the same `main_hidden` + anchors → compare the
   draft token IDs. If they diverge, the torch port has a bug — fix it before
   trusting any powered number.

2. **The offline drafter ≠ the live engine — don't trust the offline throughput.**
   The torch drafter (D_f32 or D_f16 on captured H) does NOT reproduce the live Metal
   engine's acceptance. The Lead-04 dtype-invariance ("f32-Q2 == f16-Q2, drafts
   identical") held for **p1 (the first draft)** but **NOT for the block** (the
   autoregressive continuation). Measured (Lead 11): D_f32 p1=0.835, D_f16 p1=0.525,
   vs the live Metal p1~0.70; the offline baselines 59.54/35.72 t/s vs the live 40.04.
   **For runtime/engine verdicts, measure on the real engine** (`ds4-spec-bench` +
   `DS4_DSPARK_TIMING`); the torch drafter is a diagnostic, not the deployable result.

3. **Choose the dtype deliberately (F32 vs F16) — and know the block-divergence.**
   - **F32** is the evaluation carrier (Lead 04: D_f16 on FP hiddens is numerically
     unstable → insane p1 ~0.62 vs F32's sane ~0.85).
   - **F16** is the deployment precision (the Metal drafter runs at F16/Q4_K).
   - On Q2 (IQ2XXS) hiddens, the Lead-04 dtype-invariance holds for p1 (F32 == F16,
     drafts identical) — but **the block diverges** (the autoregressive errors compound
     differently). If you measure the block acceptance, the dtype matters: **always
     report which dtype + validate the block-level dtype-invariance** (don't assume it
     from the p1 result).

4. **The GGUF weight loading is NOT a torch.load — it's a custom loader.**
   The drafter weights live in a GGUF (not a .pt/.safetensors). The loader
   (`gguf_loader.py`):
   - `load_gguf_dense_only(path)` → loads the dense tensors (the non-expert weights)
     into a flat array + a tensor index. Returns `(params, T, infos, doff, _)`.
   - `index_gguf(path)` → returns a lightweight index (no data loaded); use with
     `read_tensor(path, infos, doff, name)` to read a SINGLE tensor (e.g. the target's
     `token_embd.weight`) without loading the full 87 GB target.
   - The expert weights: `dequant_q4_k_expert` (if needed — the drafter body uses the
     dense weights only; the MoE experts are in the target, not the drafter).
   - **build_body/build_head** call `load_gguf_dense_only(dspark_path)` for the drafter
     + `index_gguf(target_path)` + `read_tensor(target_path, ..., "token_embd.weight")`
     for the embed. The target is NOT fully loaded — only the embed tensor is read.

5. **The HC (Manifold-constrained Hyper-Connections) representation.**
   The drafter's input is `main_hidden [12288]` = `concat(mean(hc_ffn_post[40]),
   mean(hc_ffn_post[41]), mean(hc_ffn_post[42]))`. Each layer's HC residual is
   `[HC=4, DIM=4096]` = 16384 floats. The drafter consumes the **mean over HC=4**
   (→ [4096] per layer), concatenated across 3 layers (→ [12288]). The HC=4 components
   are **DISTINCT** (not copies — `mean|hc−mean|/|mean| ≈ 0.71`). The
   `forward_embed` (numpy) + `forward_prompt` (torch) both apply this reduction.

6. **The forward_prompt processes the hidden SEQUENCE (not one position at a time).**
   `body.forward_prompt(main_hidden_seq, anchors)` where `main_hidden_seq` is
   `[max_step+1, 12288]` (the hidden states at positions 0..max_step) + `anchors` is
   `[max_step]` (the anchor token IDs at positions 1..max_step). Returns `xs
   [max_step, BLOCK, HC, DIM]` (the per-anchor block features). The drafter's KV
   accumulates across the sequence (the sliding window `WIN` + the `n_real` counter).
   **The anchor token is the drafter's embedding seed** — it's the greedy token at
   each position (from the target's greedy trajectory). NOT the drafter's own draft.

7. **The head is autoregressive — the block drafts condition on the PRIOR drafts.**
   `head.forward(xs, anchors, return_conf=False)` → `(out [b, BLOCK+1], base
   [b, BLOCK, VOCAB])`. The loop: `prev_tok[:, i] = out[:, i]`; `out[:, i+1] =
   (base[:, i] + markov_emb @ mw2_T).argmax(-1)`. So `out[:, i+1]` (the i-th draft)
   is conditioned on `out[:, i]` (the (i-1)-th draft). **The block acceptance =
   the longest prefix of drafts matching the target's greedy trajectory** — a
   mismatch at position j stops the count (the subsequent drafts are conditioned on
   the wrong token).

8. **The confidence head: `return_conf=True` gives the conf_logits per block position.**
   `head.forward(xs, anchors, return_conf=True)` → `(out, base, conf_logits,
   conf_scores)`. The `conf_logits [b, BLOCK]` are the per-block-position confidence
   logits (the drafter's predicted acceptance probability). The STS survival schedule
   uses these: `verify_n = the largest K such that Π sigmoid(conf_logits[i]/temp[i])
   ≥ θ_sts`. The `temp` array is `[1.057, 0.758, 1.038, 1.369, 1.295]` (per block
   position). **Don't conflate the drafter's confidence (post-draft) with the target's
   margin (pre-draft) — they're different signals** (Lead 11).

9. **The prompt-clustered bootstrap CI (resample prompts, not positions).**
   The acceptance / throughput metric is per-PROMPT (e.g. the per-prompt p1 or the
   per-prompt Δt/s). The bootstrap resamples PROMPTS (the independent unit), not
   positions/anchors (they're correlated within a prompt). Use:
   ```python
   rng = np.random.default_rng(0); n = len(deltas)
   means = [deltas[rng.integers(0, n, size=n)].mean() for _ in range(10000)]
   lo, hi = np.percentile(means, [2.5, 97.5])
   ```
   Report per-source (codealpaca/dolly/jsonex) — the verdict can be corpus-limited.

10. **The indexing: anchor s consumes H[s] → predicts token s+1.**
    The drafter at anchor s (s=1..max_step) consumes `H[s]` (the hidden at position s,
    post-token-s, predicting token s+1). The draft at anchor s predicts token s+1. The
    target's label for anchor s = `greedy[s+1]`. In the measure loop: `drafts[i]` for
    anchor `s=i+1`, predicts token `s+1=i+2`. The target's margin predicting token i+2
    = `topk[i+2]` (from the logprobs capture). **Off-by-one here is the #1 indexing
    bug** (Lead 11's signal off-by-one; Lead 03's position-uniform vs cycle-jump).

## Workflow (the torch measurement)

1. **Load the drafter** (once): `body = build_body(DSPARK, TARGET, dev, dtype=DTYPE)`;
   `head = build_head(DSPARK, TARGET, dev, lora_rank=0, dtype=DTYPE)`. Choose DTYPE
   deliberately (F32 for evaluation; F16 for the deployment proxy). The load takes
   ~30s on MPS (the dspark dense-only + the target embed).
2. **Load the captures**: the main_hidden (H) from the `.bin` dump OR the oracle
   bundle; the greedy tokens (Y) from the target_selected_tokens.json; the topk
   logprobs (if needed) from the logprobs jsonl. Verify the alignment (the fidelity
   gate: H + Y on the same trajectory).
3. **Run the forward**: `xs = body.forward_prompt(mh_seq, anchors)`; then `drafts, _,
   conf, _ = head(xs, anc_t, return_conf=True)`. Chunk the head call (CHUNK=24) to
   avoid MPS OOM on large batches.
4. **Compute the metric**: the p1 (first-draft match), the block-acceptance (the
   longest prefix), the E[a|K] (the mean accepted at verify-depth K), the margin/entropy
   (from the topk). Name the estimator explicitly.
5. **The bootstrap CI**: prompt-clustered (resample prompts). Report per-source.
6. **The fidelity gate** (if a new implementation): reproduce the numpy oracle (forward.py)
   at 100% draft-token agreement on a known-good input before trusting any number.

## Patterns

### The measure-crossed pattern (separate H source / anchors / labels)

For the crossed oracle (Lead 07) — the 2×2 H × Y factorial:
```python
def measure_crossed(mh, anchors_seq, labels_seq, body, head, dev):
    # anchors_seq = the context tokens (always the trajectory's greedy).
    # labels_seq = the judgment source (Y_iq2 OR Y_fp).
    # mh = the hidden source (H_iq2 OR H_fp).
    max_step = min(len(labels_seq) - BLOCK - 1, len(anchors_seq) - 1, mh.shape[0] - 1)
    anc = [int(anchors_seq[s]) for s in range(1, max_step + 1)]
    xs = body.forward_prompt(mh[:max_step + 1], anc)
    drafts = head(xs, anc_t)[:, 1:]  # [max_step, BLOCK]
    p1 = mean(drafts[i, 0] == labels_seq[i + 2] for i in range(max_step))
```

### The block-acceptance + E[a|K]
```python
def block_acceptance(drafts_i, labels, start):
    acc = 0
    for b in range(len(drafts_i)):
        if start + b < len(labels) and drafts_i[b] == labels[start + b]: acc += 1
        else: break
    return acc  # the longest prefix match
# E[a|K] = mean(min(block_acceptance, K) over anchors)
```

### The STS survival schedule
```python
STS_TEMP = [1.057018, 0.757858, 1.037660, 1.369200, 1.295342]
def survival_verify_n(conf_logits, theta_sts):
    survive = 1.0; keep = 0
    for i in range(min(5, len(conf_logits))):
        survive *= 1.0 / (1.0 + np.exp(-conf_logits[i] / STS_TEMP[i]))
        if survive < theta_sts: break
        keep = i + 1
    return keep
```

### The target margin (from the topk logprobs)
```python
def margin_of(top):  # top = [[id, logprob], ...] from the logprobs jsonl
    return top[0][1] - top[1][1] if len(top) > 1 else 99.0
def entropy_of(top):
    lp = np.array([t[1] for t in top]); p = np.exp(lp - lp.max()); p /= p.sum()
    return float(-(p * np.log(p + 1e-12)).sum())
```

## Guardrails

- **Never trust a torch drafter number without the fidelity gate** (100% draft-token
  agreement vs the numpy oracle on a known-good input). The `hc_post` broadcast-axis
  bug (Lead 03) made the torch port 43% wrong until the gate caught it.
- **Never use the offline drafter's throughput as the deployable verdict** for an
  engine/runtime lead (Lead 11: D_f32/D_f16 diverged ±50% from the live Metal). Use
  the real-engine measurement (`ds4-spec-bench` + `DS4_DSPARK_TIMING`).
- **Never assume the dtype-invariance holds for the BLOCK from the p1 result** (Lead 04:
  F32 == F16 for p1 on Q2, but the block diverges). Validate the block separately.
- **Never conflate the drafter's confidence (post-draft) with the target's margin
  (pre-draft)** (Lead 11: they're different signals; the pre-draft signal is off-by-one).
- **Never run two heavy torch processes concurrently** on the MPS (the single-process
  discipline — the drafter load + the target embed ~30 GB on the MPS).
- **Never compute the bootstrap CI by resampling positions/anchors** (they're correlated
  within a prompt). Resample PROMPTS.
- **Never forget the indexing**: anchor s → H[s] → predicts token s+1 → label
  greedy[s+1]. Off-by-one is the #1 bug.

## Anti-patterns (from experience)

- **Trusting the torch port without the numpy fidelity gate** → the `hc_post` bug (Lead
  03). Always gate: 100% draft-token agreement on a known-good input.
- **Headlining the offline simulation's throughput** → the D_f32/D_f16 divergence from
  the live Metal (Lead 11). The offline drafter is a diagnostic; the live engine is the
  verdict.
- **Assuming F16 == F32 for the block from the p1 result** → the block diverges (Lead
  04 + Lead 11). Validate the block-level dtype-invariance separately.
- **Using the drafter's confidence as a pre-draft signal** → it's a post-draft product
  (Lead 11). The pre-draft signal (the target's margin) is different (off-by-one).
- **Resampling positions for the bootstrap CI** → the positions within a prompt are
  correlated. Resample prompts.
- **Forgetting the anchor-token-as-embedding-seed** → the drafter's `forward_prompt`
  seeds from the anchor token (the greedy), NOT from the drafter's own draft. The
  anchor is the target's greedy at each position.
