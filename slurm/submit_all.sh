#!/bin/bash
# Phase-3 full sweep: 7 datasets in parallel as a SLURM array.
#
# Re-runs are safe — orchestrate.embed_base / sample_triplets cache outputs
# under data/clusterllm/<dataset>/ and skip on cache hit.
#
#   export SLURM_ACCOUNT=economics
#   ./slurm/submit_all.sh
#
# Optional:
#   export SCRATCH_BASE=<override>
#   export ARRAY_RANGE=0-6        # subset for re-runs, e.g. 5,6
#   export PARTITION=gpu          # default; flip to gpu_test if 12h is plenty

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT, e.g. export SLURM_ACCOUNT=economics}"
SCRATCH_BASE="${SCRATCH_BASE:-$PROJECT_ROOT}"
ARRAY_RANGE="${ARRAY_RANGE:-0-6}"
PARTITION="${PARTITION:-gpu_test}"
WALLTIME="${WALLTIME:-2:00:00}"

logdir="$SCRATCH_BASE/logs/sweep"
mkdir -p "$logdir"

echo "Submitting clusterllm prep sweep"
echo "  array  = $ARRAY_RANGE"
echo "  part   = $PARTITION"
echo "  time   = $WALLTIME"
echo "  logs   = $logdir"

sbatch \
    --partition="$PARTITION" \
    --time="$WALLTIME" \
    --account="$SLURM_ACCOUNT" \
    --array="$ARRAY_RANGE" \
    --output="$logdir/%a.out" \
    --error="$logdir/%a.err" \
    --job-name="clusterllm_prep" \
    --export=ALL,SCRATCH_BASE="$SCRATCH_BASE" \
    slurm/run_clusterllm_prep.sbatch

echo
echo "Watch with:"
echo "  squeue -u \$USER -o '%.10i %.30j %.8T %.10M %.6D %R'"
echo "  ls -la $PROJECT_ROOT/data/clusterllm/*/base_embeds.hdf5 2>/dev/null | wc -l"
echo "  ls -la $PROJECT_ROOT/data/clusterllm/*/triplets.json    2>/dev/null | wc -l"
