# Lead 07 - Upstream quality ceiling

Date: 2026-07-07. Status: **proposed diagnostic protocol; not yet executed.**
Purpose: quantify whether the remaining acceptance deficit is recoverable by more
extensive fine-tuning, or whether it is upstream of the local fine-tune levers:
target/base-model quantization, missing original DSpark drafter quality, or lack of
usable information in the vendored drafter features.

## Why this is needed

The existing dossier closed several narrower questions, but not the full upstream
quality ceiling:

- `dspark_quantization_ceiling.md` shows that converting the **vendored DSpark
  drafter** from Q4_K to F16/F32 does not materially improve acceptance. This
  proves the local GGUF conversion precision is not the bottleneck. It does **not**
  prove the vendored DSpark source weights are as good as an original BF16 drafter,
  if such a checkpoint exists.
- `stage1_tap_precision.md` tests a partial higher-precision target variant
  (layers 37-42 experts at Q4_K). It does **not** measure a full Q4/Q8/BF16 target
  ceiling, so it does not fully quantify quality lost from the served IQ2XXS base
  model.
- `stage2_finetune_result.md` shows that the tested non-expert head LoRA did not
  improve held-out p=1 acceptance. It does **not** by itself prove that a better
  drafter, a full-precision target trajectory, expert tuning, or a larger
  high-capacity adaptation could not recover acceptance.

The missing question is therefore:

> Is the current acceptance ceiling caused by trainable calibration error, or by an
> upstream quality loss that the local PoC could never recover?

## Definitions

Separate acceptance into three inputs:

`A(D, H, Y)` = drafter acceptance when drafter `D` consumes hidden-state trajectory
`H` and is judged against target labels `Y`.

Useful instances:

- `D_cur`: current vendored DSpark drafter as served locally. The only available
  source for this drafter is the vendored safetensors/MXFP4-derived weight set
  used to build `dspark.gguf` and its F16-dequant ceiling.
- `D_orig`: original BF16/FP DSpark drafter checkpoint, if one can be obtained
  externally. It is **not available in the current local dossier**; if it remains
  unavailable, this ceiling cannot be directly measured.
- `H_iq2`, `Y_iq2`: hidden states and greedy labels from the served IQ2XXS target.
- `H_hp`, `Y_hp`: hidden states and greedy labels from the highest-precision target
  we can practically run (full Q4, Q8, BF16, or a smaller prefix miss-set if the
  full model does not fit).
- `D_probe`: a diagnostic high-capacity probe trained only to estimate whether the
  relevant target token is recoverable from available features. It is not intended
  to be deployable.

The upstream quality ceiling is the best acceptance that could be reached without
changing the verifier economics. It has three components:

1. **Current-drafter information ceiling:** does `D_cur` already place the target
   token in a small candidate neighborhood?
2. **Target/base quantization ceiling:** does moving from `H_iq2/Y_iq2` to
   `H_hp/Y_hp` materially improve acceptance?
3. **Drafter-source ceiling:** does `D_orig`, if available, materially outperform
   the vendored MXFP4-derived drafter under the same target trajectory?

## Experiment 1 — broad top-k rank ceiling for the current drafter

### Rationale

Before training anything larger, measure whether the current drafter distribution
contains the correct served-target token at all. If the target token is often not
in the drafter's top-k neighborhood, a head reranker or modest LoRA cannot reliably
recover it. If it is usually present, then the deficit is more plausibly an
ordering/calibration problem.

This extends the Stage-0 diagnostic from the 10 exactness prompts to the larger
Stage-2 train/eval/test corpus.

### Method

For every held-out temp=0 anchor and draft position `p=1..5`:

1. Run the current drafter and compute the full decision score:
   `base_logits + markov_bias(prev_token)`.
2. Record the drafter argmax, target token, target rank in the drafter score
   distribution, score margin, and whether the target is in top-2/top-5/top-10/top-128.
3. Compute top-k oracle prefix acceptance:
   for each step, accept a draft position if the target token is within the
   drafter's top-k at that position along the greedy-spine rollout.
4. Report `p=1` top-k coverage, per-position coverage, `E[a|4]`, `E[a|5]`,
   full-accept rates `S(4)`, `S(5)`, and prompt-bootstrap confidence intervals.
5. Feed the oracle prefix histograms into `model_spec_speedup.py` or the same
   speed model formulas used by `spec_speedup_model.md`.

### Decision rule

- If top-5/top-10 oracle acceptance still cannot reach the secondary speed-model
  threshold, then extensive non-expert fine-tuning is unlikely to help. The correct
  token is often not available to a local reranker.
- If top-k oracle acceptance is high but trained LoRA is flat, the direction is not
  fully falsified. The failure is then in the trainable parameterization, objective,
  optimization, or generalization, not necessarily in the current features.
- If top-k oracle is high only at `p=1` but collapses at deeper positions, then the
  improvement path must explicitly address rollout compounding; a p=1-only gain will
  not automatically become a useful K=4/5 prefix gain.

## Experiment 2 — high-capacity probe ceiling on served IQ2XXS data

### Rationale

Top-k rank answers whether the *current scoring function* already ranks the target
near the top. It does not answer whether the information is present in hidden states
or pre-head features but inaccessible to the current head. A high-capacity probe
estimates this learnability ceiling without committing to a deployable DSpark
fine-tune.

This experiment does **not** require another drafter checkpoint. It uses only the
available vendored drafter and the captured served-target data, then asks whether
a diagnostic model can extract more target-token signal from the same features than
the deployed head does.

The probe is a falsifier, not a product path. It can be larger or less efficient
than the real drafter, as long as train/eval/test splits are strict.

### Method

Train probes on the Stage-2 train split and report only locked test results:

- Inputs:
  - pre-head drafter features `h` from the frozen current body;
  - optionally raw `main_hidden` (`layers 40/41/42` concat) for an input-side
    information probe.
- Candidate set:
  - `drafter top-128` union `target top-128` union the true target token.
  - This avoids full-vocab training cost while preserving the relevant candidates.
- Targets:
  - served IQ2XXS greedy next token for `p=1`, and optionally `p=2..5` with
    teacher-forced previous tokens.
- Models:
  - multinomial logistic probe;
  - one or two hidden-layer MLP probe;
  - optionally a per-anchor reranker over candidate embeddings.
- Metrics:
  - held-out p=1 match;
  - candidate recall ceiling;
  - `E[a|4/5]` under teacher-forced and real rollout evaluation;
  - calibration/error overlap with baseline misses.

### Decision rule

- If the high-capacity probe cannot beat the current drafter on locked held-out
  p=1, then the available non-expert features do not carry enough generalizable
  signal. More extensive head/body fine-tuning is unlikely to recover the needed
  acceptance.
- If the probe beats the current drafter by `>=5 pp` p=1 but deployable LoRA does
  not, then the fine-tune direction is not falsified; the LoRA class, loss, or
  training scale was too weak.
- If `main_hidden` probes beat `h` probes but `h` probes do not, the information is
  lost inside the frozen drafter body. That points toward body/expert tuning or a
  better drafter source, not head-only adaptation.

## Experiment 3 — high-precision target crossed oracle

### Rationale

The served target is IQ2XXS. A drafter distilled against a higher-precision teacher
may be good for that teacher but mismatched to IQ2XXS hidden states and IQ2XXS
argmax labels. To quantify base-model quantization loss, run the same drafter
against higher-precision target trajectories.

The earlier Q4-tap experiment is useful but partial: lower layers remained IQ2XXS.
The decisive version needs the highest-precision target that can be run locally or
on an available larger machine.

For this dossier, "high precision" should be interpreted carefully. The public
`deepseek-ai/DeepSeek-V4-Flash-DSpark` repository is the official upstream DSpark
artifact, but it is **not** an all-BF16 DeepSeek-V4-Flash target. The model card
reports a 167 GB repository / 165B-parameter safetensors view with mixed tensor
types, and the config uses `expert_dtype: fp4` plus FP8 quantization metadata. It is
therefore the correct **official upstream mixed-precision ceiling** relative to the
local IQ2XXS target, not a true BF16 full-precision ceiling. A true BF16 284B target
would require a different checkpoint that is not currently available in the local
dossier.

### Method

Capture aligned temp=0 trajectories for IQ2XXS and the high-precision target.
For every prefix-aligned anchor, compute:

| acceptance cell | label |
|---|---|
| `A(D_cur, H_iq2, Y_iq2)` | current served baseline |
| `A(D_cur, H_hp, Y_iq2)` | hidden/input-side effect only |
| `A(D_cur, H_iq2, Y_hp)` | label/argmax-side effect only |
| `A(D_cur, H_hp, Y_hp)` | high-precision target ceiling for current drafter |

Use p=1 as the primary clean metric, then repeat for prefix acceptance where
prefixes remain aligned. Report input main effect, label main effect, interaction,
and target-label shift rate.

If full high-precision capture is too expensive, run a targeted miss-set:

1. Select IQ2XXS p=1 misses from the Stage-2 eval/test corpus.
2. Re-run only those prompts/positions with the high-precision target.
3. Measure whether the high-precision target agrees with the drafter, with IQ2XXS,
   or with neither.

### Decision rule

- If `A(D_cur, H_hp, Y_hp)` is close to `A(D_cur, H_iq2, Y_iq2)`, then target
  quantization is not the main upstream ceiling for the current drafter.
- If `A(D_cur, H_hp, Y_hp)` is materially higher, then the current drafter may be
  good for a higher-precision teacher but mismatched to the served IQ2XXS target.
- If swapping labels changes acceptance but swapping hidden states does not, the
  problem is target argmax drift; output/head-side adaptation is the relevant lever.
- If swapping hidden states changes acceptance, the problem is representation/input
  shift; main-projection/body adaptation is more relevant.

### Remote execution plan on Modal

The local 128 GB Apple Silicon machine should not attempt this run. Use Modal only
for short data gathering: load the official upstream DSpark repository, replay the
Stage-2 prompt set at temp=0, and emit compact artifacts back to the local dossier.

#### Required machine class

Official upstream mixed-precision DSpark artifact (`deepseek-ai/DeepSeek-V4-Flash-DSpark`):

- Model facts from the public repo:
  - DeepSeek-V4-Flash is 284B total / 13B activated parameters.
  - The DSpark repository is the same checkpoint with the speculative decoding
    module attached.
  - The repo file tree is ~167 GB and the model card reports mixed tensor types
    (`BF16`, `F32`, `F8_E8M0`, `F8_E4M3`, `I8`, `I64`).
  - `config.json` has `expert_dtype: fp4`, FP8 quantization metadata, 43 layers,
    `hidden_size=4096`, DSpark target layers `[40,41,42]`, and block size 5.
  - The official inference README uses model-parallel size `MP=4`.
- Practical Modal minimum: `gpu="H100:4"` or `gpu="A100-80GB:4"` should fit the
  167 GB mixed-precision artifact for short-context data gathering, but with less
  headroom for conversion buffers and instrumentation.
- Recommended Modal setup: `gpu="H200:4"` for headroom (4 x 141 GB = 564 GB VRAM),
  better memory bandwidth, and lower risk of OOM during instrumented capture.
- Higher-margin alternative: `gpu="B200:4"` if availability is good. Modal supports
  up to 8 GPUs per container for B200/H200/H100/A100 families, so `B200:4` and
  `H200:4` stay within a single-node setup.
- Host resources: 16 physical CPU cores, 256 GiB host memory, and a persistent
  Modal Volume sized 500-700 GiB. The volume must hold the HF cache (~167 GB), the
  converted model-parallel checkpoint if using the official `inference/convert.py`
  path (another ~167 GB), Python environment/cache, and compact outputs.

Hypothetical true all-BF16 target + F16 drafter:

- DeepSeek-V4-Flash target alone would be about `284B * 2 bytes ~= 568 GB` of
  weights. The local F16 DSpark ceiling drafter is ~40 GiB, so target+drafter is
  roughly 608 GiB before KV, activations, runtime buffers, and instrumentation.
- Minimum practical class would be `B200:4` (nominal 768 GB VRAM) for short-context
  inference, or `H200:8` for comfortable headroom. `H100:8` is only ~640 GB and is
  too tight once overhead is included.
- This is **not recommended now** because no true BF16 DeepSeek-V4-Flash + DSpark
  checkpoint is identified. Dequantizing the public FP4/FP8 artifact into BF16 would
  not restore lost information, so it would not answer the original-weight ceiling
  question.

#### Modal implementation shape

Use the official PyTorch inference code rather than vLLM for this diagnostic,
because we need internal target-layer taps and drafter outputs, not only API text.

1. Create a Modal image with CUDA PyTorch, `transformers`, `safetensors`,
   `huggingface_hub`, `numpy`, and `pyarrow`/`safetensors` for outputs.
2. Mount a persistent volume, e.g. `/cache`, for HF files, converted MP=4 weights,
   and outputs.
3. Define three functions:
   - `prepare_dspark_weights()`: download `deepseek-ai/DeepSeek-V4-Flash-DSpark`
     into the volume and run `inference/convert.py --model-parallel 4`.
   - `pilot_capture()`: run 5-10 Stage-2 prompts, verify tokenizer alignment,
     greedy labels, top-k shape, and DSpark draft semantics against expected local
     tokenization.
   - `full_capture()`: run all Stage-2 prompts at temp=0 and emit compact shards.
4. Patch or wrap the official `inference/model.py` / `generate.py` path to emit:
   - upstream greedy selected token per generated position;
   - upstream top-128 token ids and logits/logprobs;
   - optional exact DSpark tap hidden states for layers 40/41/42 if the hook is
     confirmed to match `dspark_target_layer_ids`;
   - official DSpark p=1..5 draft tokens and, if cheap, top-k scores.
5. Keep the first full run label-only if hidden hooks are uncertain. Label-only
   already answers whether upstream target labels agree with IQ2XXS or with the
   current drafter on the miss set. Add hidden-state crossed oracle only after the
   tap representation is validated.

Pseudo-Modal resource declaration:

```python
@app.function(
    gpu="H200:4",
    cpu=16,
    memory=262144,  # MiB
    timeout=6 * 60 * 60,
    volumes={"/cache": dspark_volume},
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def full_capture(...):
    ...
```

#### Workload size

The current Stage-2 dataset has 240 prompts and 29,160 captured target positions.
Prompt prefill is small (22,316 total prompt tokens; mean prompt length ~93 tokens),
so the exhaustive upstream run is only about 51k total target-model token positions
including prefill. The compute is small compared with cold start, model download,
conversion, and model load.

Recommended run set:

- **Pilot:** 10 prompts across the three sources (`codealpaca`, `dolly`, `jsonex`).
- **Full label-only:** all 240 prompts; output selected ids + top-128 logits.
- **Optional full hidden/drafter:** same 240 prompts; output layer 40/41/42 taps and
  official DSpark drafts after hook validation.
- **Retry contingency:** budget one extra pilot/full pass if instrumentation is wrong.

#### Cost estimate

Modal pricing as of 2026-07-07:

- H200: `$0.001261/sec` = `$4.54/GPU-hour`; `H200:4` GPU cost ~= `$18.16/hour`.
- H100: `$0.001097/sec` = `$3.95/GPU-hour`; `H100:4` GPU cost ~= `$15.80/hour`.
- B200: `$0.001736/sec` = `$6.25/GPU-hour`; `B200:4` GPU cost ~= `$25.00/hour`.
- A100-80GB: `$0.000694/sec` = `$2.50/GPU-hour`; `A100-80GB:4` GPU cost ~=
  `$10.00/hour`.
- Add host resources for the recommended shape: 16 physical cores ~= `$0.75/hour`;
  256 GiB memory ~= `$2.05/hour`.
- Modal Volumes are `$0.09/GiB-month` with 1 TiB/month included, so the proposed
  500-700 GiB experiment volume should usually add no extra storage charge within
  the free included tier. Region selection adds 1.5-1.75x; non-preemptible execution
  is 3x and is not recommended for this short experiment.

Recommended budget on `H200:4`:

| phase | expected wall time | cost at ~$20.96/hour |
|---|---:|---:|
| first download + convert + model-load smoke | 1-2 h | $21-$42 |
| pilot capture | 0.25-0.5 h | $5-$11 |
| full label-only capture | 0.5-2 h | $11-$42 |
| optional hidden + official-drafter capture | 0.5-2 h | $11-$42 |
| one retry/contingency pass | 1-2 h | $21-$42 |

Practical envelope:

- **Label-only answer:** ~2-4 hours on `H200:4`, about **$40-$85** after the first
  successful setup.
- **Full labels + hidden taps + official drafter outputs:** ~3-6 hours, about
  **$65-$125**.
- **Conservative budget with retries:** **$150-$250**.

Cheaper `A100-80GB:4` could reduce hourly cost but is not the recommended first run:
the experiment is dominated by setup/debug risk, and H200's memory headroom is worth
more than the small absolute cost saving. `B200:4` is a good fast fallback if H200
queue time is poor, at roughly `$28/hour` including host resources.

## Experiment 4 — original drafter checkpoint A/B, if available

### Rationale

The current F16 ceiling is only the F16 dequantization of the vendored MXFP4 source.
If the DSpark drafter originally existed as BF16/FP weights before MXFP4 export, the
vendored weights may simply have lost too much quality. This is a separate question
from Q4_K GGUF conversion.

This experiment is conditional. The current local materials include only the
vendored safetensors/MXFP4-derived drafter, so this A/B cannot be executed unless an
external original BF16/FP checkpoint is found.

### Method

If an original BF16/FP DSpark drafter checkpoint can be obtained:

1. Convert or load it into the same oracle path.
2. Verify tensor semantics against the current DSpark forward.
3. Run the same retained acceptance harness:
   - exactness small corpus;
   - Stage-2 locked eval/test corpus;
   - same IQ2XXS target captures;
   - optionally high-precision target captures from Experiment 3.
4. Compare against `D_cur` and the F16 vendored ceiling.

### Decision rule

- If `D_orig` improves held-out p=1 and `E[a|4/5]` by `<1 pp`, the original drafter
  source quality is not the missing ceiling.
- If `D_orig` improves by `>=5 pp`, the vendored MXFP4-derived drafter is a real
  upstream bottleneck, and local fine-tuning of the shipped weights is solving a
  harder problem than intended.
- If no original checkpoint exists, this ceiling remains unmeasured. The dossier
  should state that explicitly rather than treating the F16 vendored ceiling as a
  true original-weight ceiling.

## Recommended execution order

1. **Broad top-k rank ceiling on current Stage-2 held-out data.** Cheapest and most
   directly tied to acceptance. This determines whether a local reranking/fine-tune
   story is even plausible.
2. **High-capacity probe ceiling.** Still local and fast on Apple Silicon. This
   separates "information absent" from "our LoRA/loss was too weak."
3. **High-precision target crossed oracle.** More expensive because it needs a
   better target run, but decisive for base-model quantization loss.
4. **Original drafter A/B.** Most decisive for vendored drafter quality, but only
   possible if true pre-MXFP4 DSpark weights are available.

## Efficient falsification criteria

Close the "more extensive non-expert fine-tuning will recover acceptance" direction
if all of the following hold:

- broad top-k oracle acceptance is too low to meet at least the secondary speed-model
  threshold;
- high-capacity probes on served IQ2XXS data cannot materially beat the current
  drafter on a locked held-out split;
- high-precision target crossed oracle, if available, does not materially raise
  current-drafter acceptance;
- no original BF16/FP drafter A/B is available, or it fails to improve materially.

Keep the direction open, but localize it, if any of the following hold:

- top-k oracle is high but LoRA is flat: trainable adapter/loss/scale was too weak;
- high-capacity probe is high from `main_hidden` but not from `h`: frozen drafter
  body loses useful information;
- high-precision target is much better: served IQ2XXS quantization is the mismatch;
- original BF16 drafter is much better: vendored MXFP4 source quality is the ceiling.

## Reporting requirements

For every experiment, report both acceptance and speed-model relevance:

- p=1 match rate with paired confidence / McNemar where applicable;
- `E[a|4]`, `E[a|5]`, prefix histograms, and full-accept rates;
- prompt-bootstrap intervals, not only token-level aggregate means;
- miss-set overlap with Stage-0 shallow-rank cases;
- whether the result changes the projected ability to beat baseline or the `+20%`
  primary gate under `spec_speedup_model.md`.

This prevents a repeat of the main ambiguity in Stage 2: a diagnostic can be
locally correct but still not answer whether the acceptance gain is large enough to
matter for DSpark throughput.
