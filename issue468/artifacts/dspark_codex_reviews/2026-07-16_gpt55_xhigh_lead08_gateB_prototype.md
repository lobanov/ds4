**Verdict: NO-GO stands.** I found one real self-inflicted performance bug in the M2 kernel, but even an optimistic bound on fixing it leaves M2 about 3x slower than M1x2. This is not a bounded bug hunt away from beating `metal_graph_verify_suffix_tops`.

**Findings**

1. **M2 does extra token work for singleton union slots.**
   In [metal/moe.metal](/Users/lobanov/Projects/ds4-dspark-research/metal/moe.metal:1236), the M2 kernel always loads `yl_a` and `yl_b` and accumulates `sum*_a` and `sum*_b` for every union slot. The `sa/sb` checks only gate the final writes at [metal/moe.metal](/Users/lobanov/Projects/ds4-dspark-research/metal/moe.metal:1291).
   
   For `selA={0..5}`, `selB={3..8}`, there are 9 union experts but only 12 real token-expert pairs. Current M2 computes 18 token-expert gate/up MAC streams. So the artifact claim “same compute as M1x2” is false.

   Bound: `13.334 ms * 12/18 = 8.889 ms`, still `2.93x` slower than `M1x2=3.039 ms`. If only gate/up scales and down is held constant, the bound is still about `3.0x` slower. So this bug matters, but does not rescue the prototype.

2. **Register/occupancy diagnosis is directionally right, but incomplete.**
   M2 has `yl_a[32] + yl_b[32] + 4 accumulator arrays` at [metal/moe.metal](/Users/lobanov/Projects/ds4-dspark-research/metal/moe.metal:1213), versus one token’s state in M1. The measured slowdown is better described as:
   
   `1.5x wasted singleton compute * ~3x slower per token-expert unit`, not simply “6x per expert.”

   A serious redesign could split shared slots from singleton slots or reduce row tiling/register state, but that is a new kernel design, not a bounded fix.

3. **Correctness indexing is mostly sound in the current worktree.**
   The union maps token slots correctly in [ds4_metal.m](/Users/lobanov/Projects/ds4-dspark-research/ds4_metal.m:22541). M2 writes `mid_a[sa]` / `mid_b[sb]`, and the existing down kernel reads `mid[expert_slot]` in `expert_slot=0..5` order at [metal/moe.metal](/Users/lobanov/Projects/ds4-dspark-research/metal/moe.metal:2803). That preserves down/sum6 selection order.

   The dirty worktree also contains an important fix: route weights are now read as `weights_[slot]`, not `weights_[sa/sb]`, at [metal/moe.metal](/Users/lobanov/Projects/ds4-dspark-research/metal/moe.metal:1295). Without that, shared/B-only experts would be misweighted.

4. **Fidelity localization is acceptable, with one wording caveat.**
   The mid diff appears before down, and down uses the same per-token `id_q2_K_sum6` path, so “localized to gate+up” is a fair engineering conclusion. It is not bit-exact, and the “bit-exact by construction” comment is too strong because the distinct M2 kernel changes compiler scheduling/register pressure.

5. **The timing harness is valid enough to reject this prototype.**
   The cold all-layer sweep at [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:22135) does avoid immediate same-layer L2 reuse. The order is not ideal: M2 always runs before M1x2, so a repeat should reverse/alternate order. But a 4.4x gap is too large to plausibly be explained by that artifact.

   Also, M1x2 does not have “2x begin/end overhead” at the harness level; both M1 calls sit inside one begin/end block at [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:22148). The broader point still holds: M1x2 is not the production batch verifier and is not an absolute baseline.

6. **M1x2 is not “batch-verifier-equivalent,” but it is a lenient bar.**
   Production `metal_graph_verify_suffix_tops` batches all layers in one command stream via [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:21863), and its batch MoE path differs from single-token `routed_moe_one_tensor`. So the artifact should not call M1x2 batch-equivalent. But losing badly to M1x2 still means losing harder to the real batch verifier.

**Other latent issues**

- [ds4_metal.m](/Users/lobanov/Projects/ds4-dspark-research/ds4_metal.m:22700) always dispatches the Q2_K down-sum pipeline; `down_type` is not enforced as Q2_K. Fine for the tested model, unsafe as a general API.
- [metal/moe.metal](/Users/lobanov/Projects/ds4-dspark-research/metal/moe.metal:1197) hard-codes `384` experts instead of using `args.ne02`.

I could not rerun the GPU harness because the target GGUF is not present in this workspace. The read-only static bound above is the verdict-changing check: it catches a real M2 performance bug, and still leaves the prototype far from viable.