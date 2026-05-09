#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SAMPLES_PER_LEVEL="${SAMPLES_PER_LEVEL:-300}"
BATCH_SIZE="${BATCH_SIZE:-384}"
CNN_EPOCHS="${CNN_EPOCHS:-20}"
NUM_WORKERS="${NUM_WORKERS:-0}"
POLLUTION="${POLLUTION:-core}"
FAST="${FAST:-0}"

args=(
  main_eval.py
  --pollution "${POLLUTION}"
  --samples-per-level "${SAMPLES_PER_LEVEL}"
  --batch-size "${BATCH_SIZE}"
  --cnn-epochs "${CNN_EPOCHS}"
  --num-workers "${NUM_WORKERS}"
  --split-mode group
  --mchn-topk 10
  --mchn-maxsim-weight 0.50
  --skip-e2e
)

if [[ "${FAST}" == "1" ]]; then
  args+=(
    --skip-confusion
    --skip-balanced-eval
    --skip-ablation
    --skip-beta-ablation
    --skip-attention-errors
    --skip-capacity
    --skip-random-capacity
  )
fi

echo "Running final MCHN experiments from ${PROJECT_ROOT}"
python "${args[@]}"
