#!/usr/bin/env bash
# Branch B precision baseline — accounts for Metal temp=0 nondeterminism.
#
# The reference run (reference_nothink_4096.log) is ONE target-only sample, not
# a fixed ground truth: greedy argmax is not bit-reproducible on Metal because
# GPU reduction order is nondeterministic, so near-tied tokens can flip run-to-
# run independent of the verifier path.
#
# To separate "Metal nondeterminism" from "batch verifier drift" we run each
# path R times and compare flip rates:
#   exact (--quality) = same single-token decode kernel as target-only.
#                       Its run-to-run flips IS the noise floor.
#   batch (default)   = batched verifier kernel + nondeterminism + logit drift.
#                       batch flips > exact flips  =>  drift is harmful.
#                       batch flips ≈ exact flips  =>  Branch B quality-neutral.
#
# Usage: run_quality_baseline.sh [R]   (R = runs per config, default 2)
# One ds4 process at a time (instance lock).
set -u
cd "$(dirname "$0")/.."
R="${1:-2}"
MODEL="../ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
MTP="../ds4/gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"
OUT=issue468/baseline/quality
mkdir -p "$OUT"
COMMON=(--plain --nothink -n 4096 --seed 1 --questions 92 --power 100)

echo "=== start $(date +%H:%M:%S)  R=$R runs/config  (each ~40-55 min) ==="

for r in $(seq 1 "$R"); do
  echo "=== [batch]   run $r/$R  $(date +%H:%M:%S) ==="
  ./ds4-eval -m "$MODEL" --mtp "$MTP" --mtp-draft 2 \
       "${COMMON[@]}" --trace "$OUT/batch.r$r.trace" \
       > "$OUT/batch.r$r.out" 2> "$OUT/batch.r$r.log"
  echo "[batch.r$r] $(grep -E 'passed' "$OUT/batch.r$r.out" | tail -1)"

  echo "=== [exact]   run $r/$R  $(date +%H:%M:%S) ==="
  ./ds4-eval -m "$MODEL" --mtp "$MTP" --mtp-draft 2 --quality \
       "${COMMON[@]}" --trace "$OUT/exact.r$r.trace" \
       > "$OUT/exact.r$r.out" 2> "$OUT/exact.r$r.log"
  echo "[exact.r$r] $(grep -E 'passed' "$OUT/exact.r$r.out" | tail -1)"
done

echo "=== ALL DONE $(date +%H:%M:%S) ==="
