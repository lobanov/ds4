# Lead 08 iteration 21: selected-address ALU-headroom separator

Date: 2026-07-18. Status: **T2a AMBIGUOUS, live-verifier scope fail; repeat audit COMMIT.**
No DRAM-bandwidth or dequant-bound claim is made.

## Contract and scope correction

The independent preflight rejected a purported "dequant/FMA" test and authorized a narrower
arithmetic-headroom separator. A research-only companion to
`kernel_mul_mv_addr_iq2_xxs_pair_swiglu_f32` preserves the production 24 physical rows, 18 selected
experts, addresses, IQ2 weight and activation reads, NSG 2, row tile 4, reductions, outputs, and
dispatch count. Literal production is selector 0. Selector 1 uses the companion with zero injected
rounds, and candidates `{8,32,128}` add two dependent identity FMA chains after each gate/up dot
partial and before accumulation. The production/companion-zero control must agree within 2%.

The first mapped-weight run was discarded: the harness did not dispatch the address-table kernel.
That exposed an important scope limit obscured by the preflight. This kernel is active only on the
SSD selected-address path, and `--ssd-streaming` cannot compose with DSpark. Therefore this is T2a,
not a characterization of the live M3 verifier. A follow-up mapped run emits 688/688 K4 stage
records as `path=tiny_pair_mv`, backed by `kernel_mul_mv_id_iq2_xxs_pair_f32`; that family still
requires T2b.

## Validity gates

- The final binary reports identical production/probe pipeline metadata: thread width 32, maximum
  1024 threads, and zero static threadgroup bytes. Metal exposes no register/spill metadata here.
- Gate, up, weighted mid, routed output, and output argmax are bit-exact for the discovered layer-0
  fidelity pattern for every companion arm against literal production before timing. This is not an
  all-layer output check.
- `metal -S` on the exact dumped runtime source shows zero probe-field loads and no probe FMA in
  the production `<4,false>` address specialization.
- The companion loads runtime rounds/mul/add. Its `<4,true>` math specialization retains two
  `air.fma.f32` calls and a dynamic backedge comparing the incremented counter with runtime rounds.
- All arms run 10 samples over all 43 routed layers with the same cold-cache preparation. Five
  ascending rotations followed by five reversed rotations put every arm in every order position
  twice. Median confidence intervals are deterministic 20,000-resample bootstrap intervals; paired
  intervals resample same-round differences.

## Results

| arm | total median ms/layer | delta | paired delta, 95% CI | gate/up median ms/layer | delta | paired delta, 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| production | 10.5510 | +1.15% vs companion-zero | +0.071 [-0.061, +0.229] | 4.9598 | +0.19% | -0.012 [-0.085, +0.110] |
| companion-zero | 10.4310 | -1.14% vs production | -0.071 [-0.229, +0.061] | 4.9503 | -0.19% | +0.012 [-0.110, +0.085] |
| 8 rounds | 10.6265 | +1.87% | +0.107 [-0.017, +0.474] | 5.0240 | +1.49% | +0.087 [-0.004, +0.154] |
| 32 rounds | 10.9955 | +5.41% | +0.557 [+0.417, +0.678] | 5.4502 | +10.10% | +0.465 [+0.312, +0.563] |
| 128 rounds | 12.9050 | +23.72% | +2.407 [+2.056, +2.653] | 7.4936 | +51.38% | +2.538 [+2.298, +2.669] |

The production/companion-zero control passes the predeclared 2% total and stage gates. The Theil-Sen
slope across companion-zero and candidate medians is 0.01916 ms/layer per injected chain round for
total cost and 0.01881 ms/layer for synchronized gate/up. The response is monotonic and approximately
linear from 8 rounds onward, but at 32 rounds it lies in the predeclared 5-25% ambiguous band, while
the 8-round paired intervals include zero. This is neither the under-5% ALU-headroom pattern nor the
predeclared >=25% strong compute/issue pattern. It does not prove DRAM bandwidth binding and does
not distinguish dequant, instruction throughput, address latency, occupancy, or unexcluded
loop/code-generation effects.

## Decision and ledger reassessment

Close T2a as **AMBIGUOUS** for the incompatible SSD carrier. Do not optimize away IQ2 dequant/FMA
on its strength, and do not reopen grouped, layout, geometry, or residency mechanisms:
their economic/carrier falsifiers are unchanged. Keep V14 open but unauthorized for a prototype.
The next P0 is T2b: apply the same compiler-retained separator to the mapped
`kernel_mul_mv_id_iq2_xxs_pair_f32` gate/up kernel actually dispatched as `tiny_pair_mv` by the
current K=4 verifier. Only T2b may select a traffic/latency or
ALU/instruction mechanism for the one permitted >=15% fixed-work prototype.

## Reproduction

```sh
make -j4 ds4-spec-bench
env DS4_M2_FIDELITY_TEST=1 DS4_LEAD08_ADDR_ALU_PROBE=1 \
  DS4_METAL_ENABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR=1 \
  ./ds4-spec-bench --metal --ssd-streaming --ssd-streaming-cold \
  --ssd-streaming-cache-experts 256 -m "$MODEL" --bulk-config "$ONE"

env DS4_M2_FIDELITY_TEST=1 DS4_LEAD08_ADDR_ALU_PROBE=1 \
  DS4_METAL_ENABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR=1 \
  DS4_METAL_MOE_STAGE_PROFILE=1 DS4_METAL_MOE_STAGE_PROFILE_FILTER=gate_up \
  ./ds4-spec-bench --metal --ssd-streaming --ssd-streaming-cold \
  --ssd-streaming-cache-experts 256 -m "$MODEL" --bulk-config "$ONE"

DS4_LEAD08_ADDR_ALU_PROBE=1 DS4_METAL_DUMP_SOURCE=/tmp/lead08_iter21_full.metal \
  ./ds4-spec-bench --metal -m "$MODEL" --dspark "$DSPARK" --bulk-config "$ONE"
xcrun metal -S -gline-tables-only -D DS4_METAL_HAS_TENSOR=1 \
  -D DS4_METAL_HC_STABLE=1 -D DS4_METAL_NORM_RSQRT_DISABLE=1 \
  /tmp/lead08_iter21_full.metal -o /tmp/lead08_iter21_full.air.s

DS4_M2_FIDELITY_TEST=1 DS4_LEAD08_BATCH_OVERLAP_PROBE=1 \
  DS4_METAL_MOE_STAGE_PROFILE=1 DS4_METAL_MOE_STAGE_PROFILE_FILTER=gate \
  ./ds4-spec-bench --metal -m "$MODEL" --dspark "$DSPARK" --bulk-config "$ONE"
```

`MODEL` is the retained 81 GiB IQ2XXS target, `DSPARK` is the retained support model, and `ONE` is
`19_iter11_code_sort_pairs_config.jsonl`. The harness's legacy nonzero exit after completing its
older fast-math comparison is expected.
