#!/bin/bash
# ClusterLLM phase 3: fine-tune Instructor-large per dataset (GPU).
#
# Submitted by submit_finetune.sh as a 7-task array. Walltime is set by the
# submitter; pinning here would override --array-level overrides.
#
# Per-task resource ask: 4 CPU, 32G RAM, 1 GPU. Instructor-large is ~330M
# T5 params; 15 epochs × 1024 triplets / batch 4 ≈ 3840 steps. With FP16
# disabled (faithful to upstream) and BF16 on A100 the step rate is ~3–4/s
# → ~15–25 min per dataset, comfortably under gpu_test's 12h cap.
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

CLUSTERLLM_PHASE=finetune python -m slurm.array_task

echo "[$(date)] done"
