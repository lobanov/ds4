# Lead 08 iteration 22: mapped tiny-pair ALU separator

Date: 2026-07-18. Status: **INVALID / near-threshold diagnostic; audit COMMIT.**
No binding claim and no prototype authorization.

## Preflight correction

The independent preflight returned **REDESIGN**. T2b can only be a one-sided arithmetic sensitivity
separator: a strong retained-FMA response may authorize one arithmetic/instruction-reduction
prototype, but a weak response cannot identify DRAM, cache, or address latency. The predeclared
production/companion-zero paired 95% interval must be contained within +/-2%; R32 `gate_up` must
have a lower bound of at least 25% in both selection strata for a compute/issue-sensitive result.

The measured carrier is mapped IQ2 K=4 `tiny_pair_mv`, backed by
`kernel_mul_mv_id_iq2_xxs_pair_f32`. It writes F32 gate/up and uses the separate activation and down
path; it is not the SSD selected-address kernel and not an F16 fused-mid path.

## Implementation and validity evidence

- Selector 0 uses the literal production entry point and pipeline. Selectors `{1,8,32,128}` use one
  research-only companion; selector 1 supplies actual zero rounds.
- A separate 16-byte buffer-7 argument supplies runtime rounds, multiply 1, and add 0. Bytes,
  mapped buffers and offsets, inputs, IDs, route weights, 24 physical pairs, 18 unique experts,
  NSG 2, NR0 4, grid, threadgroup memory, outputs, activation/down work, and dispatch count remain
  fixed within each stratum.
- The overlap stratum uses three contiguous six-expert groups with the first shared by two tokens.
  The dispersed stratum applies coprime stride 37 modulo 384 to the same 18 logical experts.
- Every companion arm passes 86/86 cases: all 43 routed layers times two strata have zero differing
  gate, up, weighted-mid, or routed-output F32 bits and zero output-argmax flips.
- Every measured record reports IQ2, tokens 4, pairs 24, `path=tiny_pair_mv`, NSG 2, NR0 4, and the
  expected production/companion identity. The 20-round Latin/reversed schedule places each arm in
  each order position four times per stratum.
- Production and companion report thread width 32, maximum 1024 threads, and zero static
  threadgroup bytes. Metal exposes no register/spill metadata here.
- `metal -S` on the dumped runtime source gives production no probe argument and zero probe FMAs.
  The companion loads rounds/mul/add and retains two `air.fma.f32` calls plus a backedge comparing
  the incremented loop counter with runtime rounds.

## Results

Paired intervals resample same-round relative differences with 20,000 deterministic bootstrap
draws. Full point statistics and median intervals are in the CSV.

| timing / stratum | production vs companion-zero, paired 95% CI | R8 vs zero | R32 vs zero | R128 vs zero |
|---|---:|---:|---:|---:|
| wall / overlap | -0.17% [-2.51, +0.84] | +6.92% [+5.90, +7.58] | +21.08% [+20.38, +22.39] | +96.24% [+94.46, +97.98] |
| wall / dispersed | -1.50% [-2.64, -0.51] | +6.57% [+5.81, +8.00] | +21.96% [+20.45, +22.75] | +97.16% [+95.83, +98.80] |
| `gate_up` / overlap | -1.51% [-2.70, +0.05] | +5.54% [+4.56, +6.41] | +26.00% [+24.96, +27.82] | +131.43% [+130.27, +135.31] |
| `gate_up` / dispersed | -0.34% [-1.30, +1.90] | +5.62% [+4.80, +6.18] | +27.05% [+25.64, +28.23] | +132.87% [+131.57, +134.04] |

The Theil-Sen slope is 0.00446/0.00452 ms per layer per round for wall time and
0.00398/0.00410 for synchronized `gate_up` in overlap/dispersed order. Direction and magnitude are
consistent across strata and timing modes.

## Decision

The result is **INVALID** under the predeclared control gate: three of four production-versus-zero
paired intervals escape the required +/-2% band. It also narrowly misses positive classification:
the overlap R32 `gate_up` lower bound is 24.958%, below 25%. The strong monotonic response is retained
only as near-threshold diagnostic evidence. It does not authorize dequant/instruction reduction and
cannot authorize a traffic/latency prototype.

The companion-zero `gate_up` median is about 0.427 ms/layer, or 18.4 ms across 43 layers. Saving the
required 8.7 ms/cycle from this stage alone would require about a 47% whole-stage reduction. Even a
valid T2c classification may authorize a prototype only if the proposed instruction/dequant removal
has a credible path to that prize; the generic 15% stage floor is not sufficient here.

Close this T2b measurement design. The next preflight should challenge a control-stabilized direct
replay of the same mapped K4 pair kernel, eliminating per-layer diagnostic logging from the timed
interval and increasing precision without changing the carrier. If that control cannot pass, stop
this separator family and design an independent byte/latency discriminator.

## Reproduction

```sh
MODEL=/Users/lobanov/Projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
DSPARK=/Users/lobanov/Projects/ds4/gguf/dspark.gguf
ONE=issue468/artifacts/lead08_reassessment/19_iter11_code_sort_pairs_config.jsonl

make -j4 ds4-spec-bench
env DS4_M2_FIDELITY_TEST=1 DS4_LEAD08_MAPPED_ALU_PROBE=1 \
  DS4_METAL_DUMP_SOURCE=/tmp/lead08_iter22_full.metal \
  ./ds4-spec-bench --metal -m "$MODEL" --dspark "$DSPARK" --bulk-config "$ONE"

env DS4_M2_FIDELITY_TEST=1 DS4_LEAD08_MAPPED_ALU_PROBE=1 \
  DS4_METAL_MOE_STAGE_PROFILE=1 DS4_METAL_MOE_STAGE_PROFILE_FILTER=gate_up \
  ./ds4-spec-bench --metal -m "$MODEL" --dspark "$DSPARK" --bulk-config "$ONE"

xcrun metal -S -gline-tables-only -D DS4_METAL_HAS_TENSOR=1 \
  -D DS4_METAL_HC_STABLE=1 -D DS4_METAL_NORM_RSQRT_DISABLE=1 \
  /tmp/lead08_iter22_full.metal -o /tmp/lead08_iter22_full.air.s
```

The harness's legacy nonzero exit after its older fast-math comparison is expected.

The independent post-iteration audit reproduced the schedule, fidelity matrix, dispatch identities,
AIR structure, statistics, thresholds, and economic calculation and returned **COMMIT**.
