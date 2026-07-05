# DSpark drafter quantization ceiling — Q4_K vs F16 vs F32

Date: 2026-07-05.

## Question

Earlier implementation attempts concluded the current Q4_K drafter cannot produce
high-quality draft blocks and that DSpark decode is a verifier-dominated cycle.
This note measures the **theoretical draft-quality ceiling** of the vendored
drafter with routed-expert quantization removed, to decide whether Q4_K is the
quality bottleneck.

## Setup

Built unquantized drafter GGUFs directly from the vendored HuggingFace
safetensors (`/Users/lobanov/Projects/ds4/hf-dspark`, shards 46-48) using the
retained converter copy at `issue468/dspark_converter/` (see its README for
provenance — it is the newer `mtp.`-aware converter, byte-verified to reproduce
the canonical `dspark.gguf` for Q4_K/BF16/Q8_0 drafter tensors):

- `dspark_f16.gguf` (39.69 GiB) — all categories F16
- `dspark_f32.gguf` (79.38 GiB) — all categories F32 (built, then **deleted** as
  redundant; see F16≡F32 below)

Ran the trusted numpy oracle over all 30 retained exactness bundles
(`issue468/artifacts/exactness_small_bundles/`, temps 0.0/0.5/1.0) via the bulk
harness `issue468/run_ceiling_bulk.py`, comparing draft tokens and accepted-prefix
length against the Q4_K baseline.

## Results

### F16 ≡ F32 at the weight level

The drafter ships as **MXFP4 (4-bit weights + F8_E8M0 block scales)**. Dequantizing
MXFP4 yields values that are already F16-exact, so the F16 and F32 GGUFs carry
**identical** expert bytes (measured: 0.000% mean diff). F32 adds no information;
the F32 GGUF was deleted. The drafter's "unquantized ceiling" is therefore the
**MXFP4-dequant ceiling**, and F16 reaches it.

### Q4_K vs F16 drafts: rare flips, net-zero acceptance

Across the full corpus (30 bundles × 8 steps × 5 positions = 1200 draft tokens):

- **42 draft-token flips (3.5%)**, spread over 13/30 bundles.
- **26/30 bundles have identical avg accepted prefix.**

Mean accepted prefix over the 5-token block:

| temp | Q4_K | F16 | Δ |
|---|---:|---:|---:|
| 0.0 | 2.3375 | 2.3500 | +0.0125 |
| 0.5 | 2.0875 | 2.1125 | +0.0250 |
| 1.0 | 2.0875 | 2.0750 | −0.0125 |
| **overall** | **2.1708** | **2.1792** | **+0.0083 (+0.38%)** |

The sign of the delta is not even consistent across temperatures, and the four
bundles where avg prefix changes move in both directions (e.g.
`code_sort_pairs__t1p0` 1.500→1.250 with F16, `synthesis_incident_json__t0p5`
2.875→3.125 with F16). This is within noise: **removing Q4_K yields no material
acceptance gain.**

## Why this is not an oracle bug (diagnostic)

The oracle was instrumented (`issue468/archive/diag_*.py`) to confirm every stage
of the forward genuinely differs between Q4_K and F16/F32, then the argmax absorbs
the perturbation:

| stage | Q4_K vs F32/F16 difference |
|---|---|
| routed-expert weights | **5.4% rel** (mean 0.0021 vs absmean 0.039) — Q4_K is not lossless |
| drafter hidden `x` into head | **2.84% rel** — backbone really uses the experts |
| logits | **2.24% rel**, max single-logit Δ **0.53** |
| argmax (draft token) | flips only where margin(top1−top2) ≲ 0.53 — ~3.5% of positions |

Controls ruling out dead paths / loader bugs:

- `read_dense_expert` matches a full-tensor slice exactly (diff 0.0 for experts
  0/1/127/255); F32 expert bytes are genuinely F32 (not Q4_K data).
- Zeroing the experts changes drafts drastically (code_histogram__t0p0
  avg prefix 3.125 → 1.125), proving the MoE path is live and decisive.
- The converter reproduces the canonical Q4_K `dspark.gguf` byte-for-byte
  (`--compare-tensor` OK for a drafter expert, a `mtp_unique_map` tensor, and an
  MX-FP8 attention projection).

So the oracle is correct; the result is physical: Q4_K quantization noise
perturbs logits by ~2%, which flips ~3.5% of draft tokens, but those flips net to
~zero acceptance change.

## Interpretation

**Q4_K is not the draft-quality bottleneck.** The vendored drafter's quality
ceiling is the MXFP4 source itself, and Q4_K already reaches it for draft tokens
(within noise). No precision available for these weights — F16 or F32 — yields a
material acceptance gain.

This **falsifies the hypothesis that Q4_K drafter quality is the cause of the
verifier-dominated DSpark decode cycle.** The cycle overhead must come from
elsewhere (verification / KV-replay / scheduling), not draft precision. Research
should redirect away from drafter quantization.

This supersedes the prior (now-known-invalid) dossier notes
`22_phase4_q4k_ceiling.md` and `51_numpy_oracle_q4k_ceiling.md` from the sibling
`ds4-dspark` tree: those used the buggy oracle and measured only Q4_K-dequant-to-F32
(not the HF source), and note 51 explicitly flagged the HF-source ceiling as
"unmeasured." This note measures it and closes the question.

## Reproducibility

Build the F16 ceiling GGUF (retained converter):

```sh
cd issue468/dspark_converter && make
python3 build_dspark_template.py dspark_template.gguf
./deepseek4-quantize --hf /Users/lobanov/Projects/ds4/hf-dspark \
  --template dspark_template.gguf \
  --out ../artifacts/dspark_ceiling/dspark_f16.gguf \
  --experts f16 --attention-proj f16 --attention f16 --shared f16 --dense f16 \
  --threads 16 --overwrite
```

Measure F16 against the retained bundles (reuses target captures; Q4_K baseline
already retained under `q4k_baseline/`):

```sh
. issue468/dspark_oracle/.venv/bin/activate
python issue468/run_ceiling_bulk.py \
  --candidate f16_ceiling=issue468/artifacts/dspark_ceiling/dspark_f16.gguf
```

## Retained artifacts

- `issue468/artifacts/dspark_ceiling/dspark_f16.gguf` — F16 ceiling drafter (39.69 GiB, gitignored, reproducible)
- `issue468/artifacts/exactness_small_acceptance/f16_ceiling/` — 30 per-bundle acceptance JSONs + summary
- `issue468/artifacts/exactness_small_acceptance/q4k_baseline/` — Q4_K comparison baseline (same harness)
- `issue468/dspark_converter/` — retained converter tooling (research-scoped copy)
- `issue468/run_ceiling_bulk.py` — in-process bulk acceptance harness
- `issue468/archive/diag_ceiling.py`, `diag_weights.py`, `diag_logits.py` — diagnostic scripts used for this note
