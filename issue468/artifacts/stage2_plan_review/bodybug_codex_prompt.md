You are a skeptical senior engineer hunting a SUBTLE BUG in a torch port of a numpy
ML drafter forward. Find the systematic divergence. Be terse. Your decisive claims
will be INDEPENDENTLY re-verified by the dispatcher, so prioritize runnable checks.

WORKTREE: /Users/lobanov/Projects/ds4-dspark-research. Venvs:
- `issue468/dspark_train/.venv` (torch 2.12.1 + numpy) — the TORCH PORT.
- `issue468/dspark_oracle/.venv` (numpy) — the NUMPY ORACLE (ground truth).

FILES:
- TORCH PORT (suspect): `issue468/dspark_train/drafter_body.py`  (DrafterBody + helpers)
- NUMPY GROUND TRUTH (read these, trust them):
  - `issue468/dspark_oracle/measure_acceptance_bundle.py`  (the forward loop + inline `dspark_attn`)
  - `issue468/dspark_oracle/forward.py`            (forward_embed/forward_head, constants HC/DIM/BLOCK)
  - `issue468/dspark_oracle/attention.py`          (apply_rotary, sparse_attn, dspark_attention)
  - `issue486/dspark_oracle/moe.py`                (gate, swiglu_expert, moe)
  - `issue468/dspark_oracle/hc_primitives.py`      (rmsnorm, hc_pre/hc_post/hc_split_sinkhorn, hc_head)
  - `issue468/dspark_oracle/expert_store.py` + `gguf_loader.py` (dequant_q4_k_expert)
- DEBUG HARNESS + ARTIFACTS:
  - `issue468/dspark_train/run_body_debug.py`      (per-layer x diff: numpy vs torch)
  - `issue468/dspark_train/run_body_fidelity.py`, `run_body_acceptance.py`
  - `issue468/dspark_train/data/exactness_features.npz`  (numpy oracle's true x per anchor)
  - `issue468/dspark_train/data/drafter_body_weights.npz` (cached loaded weights)

SYMPTOM (measured):
- The torch body forward runs end-to-end and is STRUCTURALLY close, but on the
  exactness corpus, per-anchor pre-head feature x diverges from the numpy oracle:
  layer 0 max-abs-diff 0.14 (mean 0.005, |x|~2), AMPLIFYING across the 3 residual
  layers -> layer 1 max 4.8, layer 2 max 110. So layer-0 has a small but NON-zero
  systematic error that the residual stack amplifies.
- Consequence: torch-drafter (body+head) p=1 = 0.875 vs numpy 0.8125 (6.25 pp
  HIGHER — systematic, not noise).
- The HEAD port alone is bit-faithful (argmax 100%, base max-abs 3e-5 vs numpy
  forward_head) — so the bug is in the BODY, not the head.

YOUR MANDATE — find the SYSTEMATIC bug at layer 0 (the 0.14 max-abs source). The
likely candidates (verify each by re-deriving from numpy and running an isolated
component test on identical input):
1. RoPE (apply_rotary): cos/sin shaping for q [1,block,heads,32] vs kv [1,block,32]
   vs window-kv [32]; the per-position slice `cos[step+1:step+1+block]`; the inverse
   rotary on the output (`-sin`). Off-by-one in position index? Wrong broadcast axis?
2. hc_post: the comb/residual mixing axis semantics. numpy: `(comb[...,None] *
   residual[...,None,:]).sum(axis=-2)`. torch port: `(comb.unsqueeze(-1) *
   residual.unsqueeze(-3)).sum(-2)`. Is the einsum/axis equivalent? (comb is
   [b,s,hc,hc]; residual [b,s,hc,d]; output [b,s,hc,d]).
3. hc_pre / hc_split_sinkhorn: pre/post/comb slicing, sigmoid, the Sinkhorn
   iteration count/axes, the rsqrt normalization.
4. MLA attention (`dspark_attn`): q_a/q_b/kv projections, per-head RMSNorm on q
   (`q * 1/sqrt(mean(q^2))`), sparse_attn with sinks (the sink max/softmax denom),
   the grouped output projection `output_a.reshape(8,1024,4096)` + einsum
   `bsgd,grd->bsgr` + output_b. Numpy uses measure_acceptance_bundle's INLINE
   dspark_attn (NOT attention.py's dspark_attention) — compare to the RIGHT one.
5. MoE: gate routing (softplus->sqrt, bias for topk only, weights from ORIGINAL
   scores, *1.5), SwiGLU clamp (gate max-only, up both), and CRITICALLY the expert
   weight orientation — dequant_q4_k_expert returns [out,in] (applied as x@W.T);
   check the torch port's `exp_gate[ids].transpose(1,2)` etc. match (gate/up are
   [inter,dim], down is [dim,inter]). Shared expert.
6. main_proj / embed / KV-window init and per-step update (positions, rope index
   per step).

HOW TO VERIFY (run read-only python; both venvs available):
- Isolate ONE component (e.g., feed identical hc_pre input to numpy hc_pre and
  torch hc_pre, diff). Or instrument run_body_debug.py to dump x AFTER the
  attention (before MoE) at layer 0 for one anchor, and AFTER the MoE, to bisect
  whether the 0.14 originates in the attn/hc path or the MoE path.
- The numpy side MUST use the oracle primitives verbatim (import from
  dspark_oracle) so the reference is the real ground truth.
- GGUF paths: /Users/lobanov/Projects/ds4/gguf/{dspark.gguf,
  DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf}.

CONSTRAINTS: read-only; do not modify tracked files. Run python -c / inline scripts.

OUTPUT (terse, evidence-based):
## The bug (the single most likely systematic divergence, with the exact line in
   drafter_body.py vs the numpy reference, and the one-line fix)
## Evidence (the isolated component test you ran + the diff it produced)
## Confidence + any secondary candidates
## If you believe it is NOT a logic bug but pure float32 BLAS amplification, say so
   explicitly and give the test that distinguishes bug-vs-chaos (e.g., float64).
