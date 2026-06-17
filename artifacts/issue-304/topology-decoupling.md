# Issue #304 Topology Decoupling

This artifact records follow-on topology work after the first correct local-generation handoff is proven.

## Current Status

- Topology decoupling is deferred until after the fused Phase 4 residency-plus-practical-workflow work.
- The current coordinator-first route model remains acceptable for the first practical handoff path unless Phase 4 shows it blocks a viable runnable workflow or benchmarkable design.

## Current Constraints

- The coordinator currently owns control-plane state, prompt/session state, and the first route slice.
- The route is expressed as ordered layer ownership, commonly `0:M -> N:output`.
- `N:output` makes the final worker own output/logit production.
- Local decode placement is not yet an explicit route property.
- KV return destination is coupled to the coordinator/session flow.

## Desired Future Properties

- Control-plane coordination can be separated from execution role.
- Prefill layer ownership can be assigned to the fastest available devices.
- Local decode can be assigned to the machine with full model/KV residency and appropriate memory bandwidth.
- Output-head/logit ownership can be expressed explicitly instead of only through `N:output`.
- KV return destination can differ from the final prefill worker and from the control coordinator.

## Phase 8 Direction

Phase 8 should document whether topology flexibility remains deferred or requires protocol/API changes. It should preserve compatibility with the current coordinator-first workflow as shorthand if new topology controls are added.
