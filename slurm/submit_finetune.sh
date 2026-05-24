#!/bin/bash
# Phase-3 sweep: fine-tune Instructor-large on each dataset's 1024 train
# triplets. One sbatch array, 7 tasks. Re-running is safe — orchestrate's
# finetune() short-circuits when checkpoint/final/ exists.
#
#   export SLURM_ACCOUNT=economics
#   ./slurm/submit_finetune.sh
#
# Optional:
#   export SCRATCH_BASE=<override>
#   export ARRAY_RANGE=0-6          # subset for re-runs, e.g. 5,6
#   export PARTITION=gpu_test       # default; gpu_requeue if you can
#                                   # tolerate restarts
#   export WALLTIME=2:00:00         # default; bump after seff-based calibration
#   export CONSTRAINT=""            # set "a100" on gpu_requeue (heterogeneous)
#
# Per /fasrc gotchas (May 2026):
#  - gpu_test caps at 2 concurrent submissions per user (QOSMaxSubmitJobPerUser).
#    For an array of 7 tasks this is fine — they queue under one job_id.
#  - gpu_requeue is heterogeneous; use --constraint=a100 to dodge older cards
#    without cu128 kernels.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT, e.g. export SLURM_ACCOUNT=economics}"
SCRATCH_BASE="${SCRATCH_BASE:-$PROJECT_ROOT}"
ARRAY_RANGE="${ARRAY_RANGE:-0-6}"
PARTITION="${PARTITION:-gpu_test}"
WALLTIME="${WALLTIME:-2:00:00}"
CONSTRAINT="${CONSTRAINT:-}"

logdir="$SCRATCH_BASE/logs/finetune"
mkdir -p "$logdir"

echo "Submitting clusterllm finetune sweep"
echo "  array  = $ARRAY_RANGE"
echo "  part   = $PARTITION"
echo "  time   = $WALLTIME"
echo "  constr = ${CONSTRAINT:-(none)}"
echo "  logs   = $logdir"

extra_args=()
if [[ -n "$CONSTRAINT" ]]; then
    extra_args+=(--constraint="$CONSTRAINT")
fi

sbatch \
    --partition="$PARTITION" \
    --time="$WALLTIME" \
    --account="$SLURM_ACCOUNT" \
    --array="$ARRAY_RANGE" \
    --output="$logdir/%a.out" \
    --error="$logdir/%a.err" \
    --job-name="clusterllm_finetune" \
    --export=ALL,SCRATCH_BASE="$SCRATCH_BASE" \
    "${extra_args[@]}" \
    slurm/run_clusterllm_finetune.sbatch

echo
echo "Watch:"
echo "  squeue -u \$USER -o '%.10i %.30j %.8T %.10M %.6D %R'"
echo "  ls -la $PROJECT_ROOT/data/clusterllm/*/checkpoint/final/ 2>/dev/null | grep model_safetensors | wc -l"
