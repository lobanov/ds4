**Gate B Verdict**

I would pass the core Metal STS composition, but I would not sign off on the bench package as “current `ds4.c` evidence” without a rerun or commit pin.

**Findings**

1. **Code/artifact mismatch: current HEAD is already Lever 4.**  
   Current `ds4.c` no longer disables anchor reuse for Metal; `anchor_reuse` has no `!dspark_draft_metal_enabled()` guard at [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:29647), and the comment says Metal anchor reuse now composes at [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:29653).  
   The artifacts clearly come from a no-Metal-anchor-reuse run: in `metal_sts.jsonl`, real drafted cycles have `accepted - verified = 1` for 100% of cycles, and `draft_ids[0] == anchor_id` only 3/2373 times. So the `decode_ms=25.8ms` claim is valid for the artifact binary, not current HEAD `571bebe`.

2. **`exclude_eos=true` is not consistently applied in speculative mode.**  
   Plain argmax excludes EOS every token at [ds4_spec_bench.c](/Users/lobanov/Projects/ds4-dspark-research/ds4_spec_bench.c:1099). Speculative mode excludes EOS only for the first token at [ds4_spec_bench.c](/Users/lobanov/Projects/ds4-dspark-research/ds4_spec_bench.c:1132); later draft/verify positions use normal argmax, e.g. [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:29814).  
   That is why spec runs hit EOS while plain emits all 64 tokens. It does not erase the Metal-over-CPU speedup: excluding the 3 affected rows gives Metal +12.0% over reuse+prefix-ckp. But it is a benchmark fidelity caveat.

3. **Cycle arrays include zeroed anchor-only tail metrics.**  
   `dspark_last_cycle.valid` is set before early anchor-only returns at [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:29637), and `ds4-spec-bench` records metrics before checking `produced <= 0`/cycle shape at [ds4_spec_bench.c](/Users/lobanov/Projects/ds4-dspark-research/ds4_spec_bench.c:1153).  
   Artifacts contain 32 CPU and 29 Metal zeroed DSpark metric rows. Filter `drafted > 0` for acceptance/cycle distributions.

**Checks That Pass**

The `conf_logits` computation looks correct. CPU computes `norm` via DSpark HC head + DSpark `w->norm`, then adds the Markov latent projection at [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:28467). Metal’s DSpark output head writes the same rms-normed hidden into `g->batch_ffn_norm` at [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:29435), then reads that buffer and applies the same dot products at [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:29583). `conf_proj` loading is also consistent at [ds4.c](/Users/lobanov/Projects/ds4-dspark-research/ds4.c:26985).

The acceptance analysis survives after cleanup. Filtering real drafted cycles:
- CPU continuation accepts: `verified - 1 = 1.480`
- Metal continuation accepts: `verified = 1.484`
- First continuation rejection, tested positions only: CPU `30.6%`, Metal `31.0%`

So the lower raw Metal `verified` is mostly the anchor-reuse accounting artifact, not worse draft quality.

The +11.8% speedup over reuse+prefix-ckp is real on the matched 93-row subset. Paired bootstrap on prompt row t/s gives Metal/reuse ratio `1.118`, CI `1.087..1.150`. It is still below plain: Metal/plain ratio `0.826`, CI `0.798..0.855`.

Main remaining caveat: the 93 rows are a biased valid-frontier subset. The dropped rows are mostly `jsonex` and have much longer prompts/frontiers, so scope the claim to “matched valid subset,” not the intended full corpus.