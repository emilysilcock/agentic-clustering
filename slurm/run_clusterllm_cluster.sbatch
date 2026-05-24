#!/bin/bash
# ClusterLLM phase 4: re-encode with finetuned checkpoint + k-means + persist.
#
# Submitted by submit_cluster.sh as a 7-task array. Faster per task than
# phase 3 (no training) — bounded by encoding the corpus once and a
# single k-means run at k_in_scope. Estimate: ~5–15 min per dataset on A100.
#
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

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

CLUSTERLLM_PHASE=cluster python -m slurm.array_task

echo "[$(date)] done"
