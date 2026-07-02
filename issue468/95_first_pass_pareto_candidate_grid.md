# First-Pass Pareto Candidate Grid for Mixed Routed `Q2/Q4`

Date: 2026-07-02

## Purpose

Turn the high-level Pareto goal from `94` into a concrete first-pass candidate
grid with explicit expected artifact sizes before generating new mixed drafter
GGUFs.

This note answers:

- what are the first useful size points to measure?
- what is the expected size of each point if routed experts move from `Q4_K`
  toward the imatrix-style routed `Q2` recipe?

## Basis

Used the current shipped drafter:

- `../ds4/gguf/dspark.gguf`

and the new helper:

- `issue468/estimate_q2_q4_frontier_sizes.py`

Command:

```sh
python3 issue468/estimate_q2_q4_frontier_sizes.py \
  --baseline-gguf ../ds4/gguf/dspark.gguf
```

The helper treats the imatrix-style routed `Q2` recipe as:

- `ffn_gate_exps` -> `IQ2_XXS`
- `ffn_up_exps` -> `IQ2_XXS`
- `ffn_down_exps` -> `Q2_K`

which matches the repo’s documented 2-bit routed-expert recipe family.

## Per-layer size decomposition

| layer | non-expert GiB | expert `Q4_K` GiB | expert routed `Q2` GiB | all-`Q4` layer total GiB | all-routed-`Q2` layer total GiB |
|---|---:|---:|---:|---:|---:|
| `mtp.0` | `0.184013` | `3.375000` | `1.687500` | `3.559013` | `1.871513` |
| `mtp.1` | `0.134193` | `3.375000` | `1.687500` | `3.509193` | `1.821693` |
| `mtp.2` | `0.257629` | `3.375000` | `1.687500` | `3.632629` | `1.945129` |

Read:

- non-expert bytes are small and unequal across layers
- routed expert bytes dominate and are symmetric across layers
- moving one layer’s routed experts from `Q4_K` to routed `Q2` saves exactly:
  - `1.687500 GiB`

So the first-pass frontier is naturally quantized into a small set of size
levels.

## First-pass candidate size levels

| recipe | `Q4` routed layers | estimated total GiB |
|---|---|---:|
| baseline all-`Q4` | `mtp.0,mtp.1,mtp.2` | `10.700835` |
| two-layer `Q4`, one-layer routed `Q2` | any 2 of 3 | `9.013335` |
| one-layer `Q4`, two-layer routed `Q2` | any 1 of 3 | `7.325835` |
| all routed `Q2` | none | `5.638335` |

This is the coarse size ladder the first measurement pass should cover.

## Why this grid is enough for a first pass

Because routed expert bytes are symmetric across the three layers, the first
coarse frontier does **not** need every possible byte level. It only needs to
distinguish:

1. all routed experts at `Q4`
2. one routed layer kept at `Q4`
3. two routed layers kept at `Q4`
4. all routed experts at routed `Q2`

The quality differences between recipes at the same size will come from:

- which specific layers stay at `Q4`

not from different byte counts.

That means the first pass should test:

1. all-`Q4`
2. all-routed-`Q2`
3. `mtp.2 = Q4`, `mtp.0/1 = Q2`
4. `mtp.0 = Q4`, `mtp.1/2 = Q2`
5. `mtp.1 = Q4`, `mtp.0/2 = Q2`
6. `mtp.0/1 = Q4`, `mtp.2 = Q2`
7. `mtp.0/2 = Q4`, `mtp.1 = Q2`
8. `mtp.1/2 = Q4`, `mtp.0 = Q2`

So the first pass is eight candidate points:

- one baseline
- one full compression endpoint
- six layer-identity variants across the two mixed size levels

## Expected shape of the frontier

Most likely:

- all-`Q4` remains the quality edge
- all-routed-`Q2` is the size edge
- the useful deployment knees, if they exist, are the mixed points

Most plausible quality-preserving mixed point:

- `mtp.2 = Q4`, `mtp.0/1 = Q2`

Reason:

- prior branch work repeatedly made `mtp.2` the most sensitive routed-local
  layer
- if only one layer keeps the larger routed-expert budget, `mtp.2` is the best
  first bet

Most important falsification point:

- all-routed-`Q2`

Reason:

- it defines the true compressed endpoint
- if its quality loss is small, the frontier shifts sharply downward in size
- if its quality loss is large, then the branch should focus only on the mixed
  knees

## Immediate measurement plan

1. obtain or build the routed-`Q2` drafter endpoint
2. generate mixed `Q2/Q4` candidates by routed layer selection
3. record exact artifact bytes for each built GGUF
4. measure all candidates on the `8192/16384/24576/32768` sweep
5. identify the non-dominated points

## Decision use

This first grid is enough to answer the main product question:

- how much acceptance is retained per GiB as routed-expert precision is reduced?

If the `7.33 GiB` or `9.01 GiB` knees are only slightly below baseline, the
frontier is promising.

If both mixed knees are sharply worse, then the current `Q4_K` baseline should
be treated as the practical size/quality knee and the compression branch should
stop.
