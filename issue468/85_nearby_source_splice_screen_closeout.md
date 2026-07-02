# Nearby Source-Splice Screen Closeout

Date: 2026-07-02

## Purpose

Record the first small family of direct source-splice screens after
`84_ffn_gate_inp_source_splice_probe.md`.

The question here was:

- if `ffn_gate_inp` is flat, do nearby attention-path source splices show a
  stronger oracle signal?

## Screened tensors

### 1. `ffn_gate_inp` family

Already recorded in `84`:

- `mtp.{0,1,2}.ffn_gate_inp.weight`

Result:

- flat / slightly negative on `ctx_08192` and `ctx_16384`
- baseline vs source arrays differ only at BF16-rounding scale

### 2. `attn_kv.weight` family

Screened:

- `mtp.0.attn_kv.weight`
- `mtp.1.attn_kv.weight`
- `mtp.2.attn_kv.weight`

Direct tensor-gap stats vs baseline GGUF:

- stage 0 relative L2:
  - `0.005504963919520378`
- stage 1 relative L2:
  - `0.005558535922318697`
- stage 2 relative L2:
  - `0.0054986379109323025`

Probe results:

- `ctx_08192`
  - committed `4.442023026315789`
  - vs baseline `4.473684210526316`
  - delta `-0.7077205882352944%`
- `ctx_16384`
  - committed `4.493421052631579`
  - vs baseline `4.519736842105263`
  - delta `-0.5822416302765587%`

Interpretation:

- slightly less negative than the `ffn_gate_inp` splice on `ctx_08192`
- but still negative on both tested contexts
- not a credible next lead

### 3. `attn_q_a.weight` family

Screened:

- `mtp.0.attn_q_a.weight`
- `mtp.1.attn_q_a.weight`
- `mtp.2.attn_q_a.weight`

Direct tensor-gap stats vs baseline GGUF:

- stage 0 relative L2:
  - `0.005478631239384413`
- stage 1 relative L2:
  - `0.0055007413029670715`
- stage 2 relative L2:
  - `0.005504245404154062`

Probe result:

- `ctx_08192`
  - committed `4.433388157894737`
  - vs baseline `4.473684210526316`
  - delta `-0.9007352941176383%`

Interpretation:

- also flat / slightly negative
- no sign that nearby attention-splice source copies are a better lead than
  `ffn_gate_inp`

## Pattern across the screen

At `ctx_08192`, the screened source-splice candidates cluster tightly:

- `ffn_gate_inp` all-stage splice:
  - `4.4300986842105265`
- `attn_q_a` all-stage splice:
  - `4.433388157894737`
- `attn_kv` all-stage splice:
  - `4.442023026315789`
- baseline:
  - `4.473684210526316`

All remain below baseline.

So the current conclusion is:

- nearby direct source-copy splices are **not** exposing a useful positive
  signal

## Decision impact

This closes out the current nearby source-splice screen family as a lead
branch.

What should **not** lead next:

- more direct source-copy variants of:
  - `ffn_gate_inp`
  - `attn_kv`
  - `attn_q_a`

What should lead next instead:

- a different mechanism than direct source-copy splicing, for example:
  - acceptance-local perturbation or search
  - routed-expert local block/objective work
  - another tensor class chosen for mechanism reasons rather than just source
    difference
