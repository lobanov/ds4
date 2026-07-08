#!/usr/bin/env python3
"""Activity 9 finalize — assemble a self-consistent activity9_final.json from the
committed sweep (activity7_summary.json) + the committed Lce+Ltv McNemar
(mcnemar_lce_ltv.json). Makes the verdict numbers reproducible from committed code:
  - run_stage2_head_lora_kloss.py  -> activity7_summary.json (rank x seed sweep)
  - run_stage2_mcnemar_kloss.py    -> mcnemar_lce_ltv.json (single-run McNemar)
  - this script                    -> activity9_final.json (consistent merge)
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
R = HERE / "artifacts" / "stage2_results"
sweep = json.loads((R / "activity7_summary.json").read_text())
mcn = json.loads((R / "mcnemar_lce_ltv.json").read_text())

out = {
  "verdict": sweep["verdict"],
  "loss": sweep["loss"],
  "no_lora_baseline_p1": sweep["baseline_p1"],
  "rank_seed_sweep": sweep["rank_seed_sweep"],
  "best_rank": sweep["best_rank"],
  "best_p1_2seed_mean": sweep["best_p1"],
  "best_delta_pp": sweep["best_delta_pp"],
  "decision_rule_satisfied": ("NOT-JUSTIFIED: best delta -1.46 pp < +1 pp, after rank 32/64/128 "
                              "x 2-seed sweep (all ranks negative, seed-stable)"),
  "mcnemar_single_run": {
    "source": "run_stage2_mcnemar_kloss.py (rank32 seed42, Lce+Ltv, 6 epochs)",
    "pre_p1": mcn["pre_p1"], "post_p1": mcn["post_p1"], "delta_pp": mcn["delta_pp"],
    "mcnemar_pre_only": mcn["mcnemar_pre_only"], "mcnemar_post_only": mcn["mcnemar_post_only"],
    "mcnemar_p": mcn["mcnemar_p"], "n": mcn["n"],
    "note": ("post_p1 (0.7961) is THIS single seed-42 run; it equals the sweep's rank-32 seed-42 "
             "value. best_p1_2seed_mean (0.7979) is the 2-seed mean. Both negative; McNemar "
             "p=5.3e-5 (significant HARM). MPS is mildly nondeterministic so exact discordants "
             "vary slightly run-to-run; the significant-harm verdict is robust.")
  },
  "interpretation": ("head-LoRA with the specified Lce+Ltv loss gives -1.46/-1.75/-3.08 pp "
                     "(rank 32/64/128, 2 seeds each, stable std ~0.0017); single-run McNemar "
                     "p=5.3e-5 (significant HARM). Non-expert head LoRA does NOT improve (significantly "
                     "degrades) acceptance. Body LoRA (Activity 8) skipped per contract (7>6ceiling) + "
                     "Activity 4 (input-invariant). Expert tuning out of scope."),
}
(R / "activity9_final.json").write_text(json.dumps(out, indent=2) + "\n")
print("wrote consistent activity9_final.json; verdict:", out["verdict"],
      "| best_delta:", out["best_delta_pp"], "| mcnemar_p:", mcn["mcnemar_p"])
