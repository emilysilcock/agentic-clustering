# `slurm/` — ClusterLLM phase 0+1 on FASRC

This directory is the FASRC harness for **only** ClusterLLM phases 0 and 1:

- Phase 0: encode each dataset's docs with `hkunlp/instructor-large`.
- Phase 1: entropy-rank ambiguous points, sample 1024 triplets per dataset.

Phase 2 (Claude Code Opus 4.7 judges the triplets) runs **locally** on
Emily's laptop — Claude Code lives there under the Max subscription.
Phase 3 (encoder fine-tune) and Phase 4 (re-cluster) will land here later
as a separate scaffold.

## Sweep shape

Single axis: 7 datasets. SLURM array maps `SLURM_ARRAY_TASK_ID` →
`DATASETS[task_id]` (see `array_task.py`).

| task_id | dataset            | n_docs (approx) |
|--------:|--------------------|----------------:|
| 0       | banking77          | 3,080           |
| 1       | clinc150           | 4,500           |
| 2       | massive_intent     | 12,000          |
| 3       | massive_domain     | 12,000          |
| 4       | twenty_newsgroups  | 18,000          |
| 5       | goemotions         | 50,000          |
| 6       | stackexchange      | 50,000          |

## Layout

```
slurm/
├── README.md                       # this file
├── array_task.py                   # SLURM_ARRAY_TASK_ID → dataset dispatcher
├── setup_env.sh                    # uv install + uv python install + uv sync (sourced)
├── prewarm.sh                      # one-shot: download instructor-large into HF_HOME
├── run_clusterllm_prep.sbatch      # the array-task sbatch script
├── submit_calibration.sh           # array=0 only (banking77, smallest), on gpu_test
└── submit_all.sh                   # array=0-6, on gpu (or gpu_test if --time<=12h)
```

## Operator workflow

Assuming the project is cloned at
`/n/netscratch/economics/Lab/esilcock/agentic-clustering` and `data/derived/`
has been scp'd alongside it (it's gitignored):

```bash
ssh fasrc
cd /n/netscratch/economics/Lab/esilcock/agentic-clustering

# 1. Build env (once after a fresh clone or pyproject change)
source slurm/setup_env.sh
python --version  # 3.11.x via uv

# 2. Prewarm instructor-large into hf_cache/
sbatch slurm/prewarm.sh
# wait for COMPLETED in sacct before step 3.

# 3. Calibrate on banking77 to read off walltime
export SLURM_ACCOUNT=economics
./slurm/submit_calibration.sh
# wait, then:
sacct -u $USER --name=clusterllm_cal -X --format=JobID,State,Elapsed,MaxRSS
seff <jobid>

# 4. Bump WALLTIME in submit_all.sh if needed, then full sweep
./slurm/submit_all.sh

# 5. After all 7 tasks finish, pull outputs back (run from laptop)
# (gitignored side outputs only — base_embeds.hdf5 + triplets.json)
# rsync -avh --include='*/' --include='base_embeds.hdf5' --include='triplets.json' \
#   --exclude='*' \
#   fasrc:/n/netscratch/economics/Lab/esilcock/agentic-clustering/data/clusterllm/ \
#   ./data/clusterllm/
```

## What runs inside one array task

```
slurm/run_clusterllm_prep.sbatch
  ├─ source slurm/setup_env.sh           # uv sync, activate .venv
  ├─ export HF_HOME=$PROJECT/hf_cache    # prewarmed
  └─ python -m slurm.array_task
        ├─ benchmarking.baselines.clusterllm.orchestrate.embed_base(<ds>)
        │     subprocess → _vendored/perspective/2_finetune/get_embedding.py
        │     writes data/clusterllm/<ds>/base_embeds.hdf5
        └─ benchmarking.baselines.clusterllm.orchestrate.sample_triplets(<ds>)
              subprocess → _vendored/perspective/1_predict_triplet/triplet_sampling.py
              writes data/clusterllm/<ds>/triplets.json
```

## Walltime expectations (to refine after calibration)

Instructor-large on a single A100 encodes roughly 1k docs/sec for short
sequences, so phase 0 should be a few minutes even for the 50k-doc
datasets. Phase 1 is CPU-bound (agglomerative or minibatch k-means on
embeddings + a Python loop) and adds <1 min per dataset.

Initial guess: 30 min per task is comfortable headroom. The calibration
job reads off the actual elapsed time so we can tighten this before the
full sweep.
