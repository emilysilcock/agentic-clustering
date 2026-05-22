#!/bin/bash
# Phase-2 calibration: run task 0 (banking77, smallest dataset) on gpu_test.
# Read seff after it finishes to size the full-sweep walltime in submit_all.sh.
#
#   export SLURM_ACCOUNT=economics
#   ./slurm/submit_calibration.sh
#
# Optional:
#   export SCRATCH_BASE=<override>

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT, e.g. export SLURM_ACCOUNT=economics}"
SCRATCH_BASE="${SCRATCH_BASE:-$PROJECT_ROOT}"

logdir="$SCRATCH_BASE/logs/calibration"
mkdir -p "$logdir"

echo "Submitting calibration (banking77, task 0) on gpu_test"
sbatch \
    --partition=gpu_test \
    --time=1:00:00 \
    --account="$SLURM_ACCOUNT" \
    --array=0 \
    --output="$logdir/%a.out" \
    --error="$logdir/%a.err" \
    --job-name="clusterllm_cal" \
    --export=ALL,SCRATCH_BASE="$SCRATCH_BASE" \
    slurm/run_clusterllm_prep.sbatch

echo
echo "Watch with: squeue -u \$USER"
echo "After it finishes:"
echo "  sacct -u \$USER --name=clusterllm_cal -X --format=JobID,State,Elapsed,MaxRSS"
echo "  seff <jobid>"
