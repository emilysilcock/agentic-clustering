#!/bin/bash
# ClusterLLM phase 0+1: Instructor-large embedding + triplet sampling.
#
# Submitted by submit_all.sh as a 7-task array (one per dataset). Per-task
# wall time is set by the submitter; defaults below are headroom for the
# largest dataset (GoEmotions ~50k, StackExchange ~50k).
#
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

# SLURM copies the script to a spool dir; use SLURM_SUBMIT_DIR.
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]:-$0}")/..}"
PROJECT_ROOT="$(pwd)"

SCRATCH_BASE="${SCRATCH_BASE:-$PROJECT_ROOT}"
export HF_HOME="$SCRATCH_BASE/hf_cache"
mkdir -p "$HF_HOME"

# shellcheck disable=SC1091
source slurm/setup_env.sh

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID:-?} host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
echo "[$(date)] HF_HOME=$HF_HOME PROJECT_ROOT=$PROJECT_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no nvidia-smi)"

python -m slurm.array_task

echo "[$(date)] done"
