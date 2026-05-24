#!/bin/bash
# Phase-4 sweep: encode each dataset with its finetuned checkpoint, k-means,
# persist results to results/predictions/clusterllm/<dataset>/seed=0.*. One
# sbatch array, 7 tasks. Safe to re-run — embed_base caches final_embeds.hdf5;
# write_run_artifacts overwrites the prediction artefacts deterministically
# (single seed per SPEC §5.7).
#
#   export SLURM_ACCOUNT=economics
#   ./slurm/submit_cluster.sh
#
# Same env-var knobs as submit_finetune.sh: SCRATCH_BASE, ARRAY_RANGE,
# PARTITION, WALLTIME, CONSTRAINT.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT, e.g. export SLURM_ACCOUNT=economics}"
SCRATCH_BASE="${SCRATCH_BASE:-$PROJECT_ROOT}"
ARRAY_RANGE="${ARRAY_RANGE:-0-6}"
PARTITION="${PARTITION:-gpu_test}"
WALLTIME="${WALLTIME:-1:00:00}"
CONSTRAINT="${CONSTRAINT:-}"

logdir="$SCRATCH_BASE/logs/cluster"
mkdir -p "$logdir"

echo "Submitting clusterllm cluster (phase 4) sweep"
echo "  array  = $ARRAY_RANGE"
echo "  part   = $PARTITION"
echo "  time   = $WALLTIME"
echo "  constr = ${CONSTRAINT:-(none)}"

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
    --job-name="clusterllm_cluster" \
    --export=ALL,SCRATCH_BASE="$SCRATCH_BASE" \
    "${extra_args[@]}" \
    slurm/run_clusterllm_cluster.sbatch

echo
echo "Watch:"
echo "  squeue -u \$USER -o '%.10i %.30j %.8T %.10M %.6D %R'"
echo "  ls $PROJECT_ROOT/results/predictions/clusterllm/*/seed=0.meta.json 2>/dev/null | wc -l"
