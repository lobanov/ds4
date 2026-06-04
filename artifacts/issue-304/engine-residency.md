# Issue #304 Engine Residency

This artifact records the model residency decision for turning the Phase 2 proof path into a practical local-generation handoff workflow.

## Current Status

- Phase 2 proved distributed prefill can hand off through a merged `DSV4` payload and continue local decode after the distributed engine is released.
- That proof path opens a fresh full local engine/session for decode. It is acceptable as a research harness, but not yet a user-facing workflow.
- Phase 3 is now scoped to correctness classification: same-backend distributed handoff parity against fully local inference and the official/local vector gates.
- Phase 4 owns the residency decision.

## Known Constraints

- Distributed `--layers` currently influences both route ownership and local model layer residency.
- Local decode needs full model weights, output/logit capability, and full KV state.
- A coordinator that maps only its route slice cannot directly perform full local decode without changing residency or opening another full engine.
- The existing validated handoff boundary is the merged `DSV4` payload.

## Candidate Directions For Phase 4

- Keep the current two-engine transition for research and measure whether the memory/runtime cost is acceptable only for validation.
- Decouple distributed route ownership from local weight residency so one coordinator engine can own a route slice during prefill but still retain enough state for full local decode.
- Add lazy or staged residency expansion if full upfront residency is too expensive.
- Keep merged `DSV4` as the correctness boundary while improving the user-facing transition mechanics.

## Acceptance Direction

The Phase 4 residency design should not weaken the Phase 3 acceptance envelope: distributed prefill -> local decode must continue to match fully local inference on the same decode backend and pass the official/local vector checks.
