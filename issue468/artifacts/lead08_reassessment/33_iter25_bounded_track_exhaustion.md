# Lead 08 iteration 25: bounded-track exhaustion audit

Date: 2026-07-18. Status: **STOP / EXHAUSTED; corrected audit COMMIT.**

After corrected U1 closed arithmetic/dequant with a maximum-removal saving upper confidence bound
below 4.74 ms, an independent preflight systematically reassessed every remaining ledger family
against the unchanged 8.7 ms prize. No independent, composable mechanism retains a defensible
upper bound large enough to admit another experiment.

| family | retained evidence / upper bound | independence and admission |
|---|---|---|
| arithmetic/dequant | U1 removes activation/LUT/dequant/dot but retains traffic; saving upper CI <4.74 ms (`32_iter24_weight_floor.md`) | below 8.7 alone; closed |
| packed traffic/address | U1 residual about 5.96 ms; locality <2%, geometry 1-3%, grouped +22.9% (`11_iter3_grouped_gateup_prototype.md`, `13_iter5_address_locality.md`, `14_iter6_addr_nsg_geometry.md`, `15_iter7_addr_row_tile.md`) | independent in principle, but no surviving realizable traffic mechanism |
| residency | mapped replay -0.04%; SSD replay -87.8% (`16_iter8_cache_residency.md`) | mapped too small; SSD runtime rejects DSpark |
| launch/host/readahead | retained decomposition about 3 ms; readahead no-op (`summaries/lead08_phaseB_floor_clearance_verdict.md`) | below 8.7 and cannot be added to nonsemantic U1 as production saving |
| top-r / margin | top-r saves 2.09-3.52 ms and r=5 harms acceptance; margin adds >=7 ms (`08_iter1_topr_budgeting.md`, `12_iter4_margin_guard.md`) | too small or quality/economic failure |
| down/attention/dense | historical shares are diagnostic; activation fusion sub-ms; dense already batched, attention sublinear (`02_sublead1_fusion_prospect_and_probe.md`, `03_probe_results_verify_ms_curve_and_pairvsunique.md`, `summaries/mtp_verifier_bandwidth_binding.md`) | no decision-grade >=8.7 carrier plus concrete bounded-quality mechanism |

U1's 4.74 ms removable-work bound and 5.96 ms retained traffic residue are distinct, not
overlapping. They cannot simply be summed: U1 is nonsemantic, while K1/K2/K3 falsify the concrete
load-sharing/layout/geometry routes that could capture traffic without deleting the computation.
Likewise, launch, top-r, and marginal geometry numbers come from different carriers/protocols and
do not define one composable, quality-qualified package. No remaining concrete package has a
defensible >=8.7 ms production upper bound.

Decision: close Lead 08 `INTERIM_BOUNDED` as exhausted on the current in-RAM M5/Metal runtime. Do
not compose U1's nonsemantic 4.7 ms with marginal knobs as if it were a realizable production
saving. Possible redirects are Lead 05 only if SSD/RAM-constrained compatibility is in scope, or
the deferred V13/V5 exact track for the primary exact-output goal without a current speed promise.
Lead 10 remains conditional on a verifier/runtime win.

The first audit required the traffic/U1 logic and stale-current-state corrections above. The repeat
audit accepted the mechanism matrix, double-counting rule, scoped conclusion, and redirects and
returned **COMMIT**.
