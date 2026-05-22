#!/bin/bash
# Prewarm Instructor-large into HF_HOME so the 7 array tasks don't race on
# the same 1.3 GB download. Submit once before submit_all.sh.
#
#   sbatch slurm/prewarm.sh
#
#SBATCH -p gpu_test
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 0:30:00
#SBATCH --gres=gpu:1
#SBATCH -J clusterllm_prewarm

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]:-$0}")/..}"
PROJECT_ROOT="$(pwd)"

SCRATCH_BASE="${SCRATCH_BASE:-$PROJECT_ROOT}"
export HF_HOME="$SCRATCH_BASE/hf_cache"
mkdir -p "$HF_HOME"

# shellcheck disable=SC1091
source slurm/setup_env.sh

echo "[$(date)] Prewarming Instructor-large into HF_HOME=$HF_HOME"

python - <<'PY'
import os
from sentence_transformers import SentenceTransformer

print(f"HF_HOME={os.environ.get('HF_HOME')}")
print("Loading hkunlp/instructor-large (downloads ~1.3 GB on first call)...")
model = SentenceTransformer("hkunlp/instructor-large")
print(f"Loaded. Encoder max_seq_length={model.max_seq_length}")
# Quick smoke-encode to make sure the model is fully materialised on disk.
out = model.encode(["hello world"])
print(f"Smoke encode ok, shape={out.shape}")
PY

echo "[$(date)] Prewarm complete."
