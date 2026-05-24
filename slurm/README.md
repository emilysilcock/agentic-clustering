# `slurm/` — ClusterLLM phases 0/1/3/4 on FASRC

This directory is the FASRC harness for the GPU-bound phases of ClusterLLM:

- **Phase 0 (prep)**: encode each dataset with `hkunlp/instructor-large`.
- **Phase 1 (prep)**: entropy-rank ambiguous points, sample 1,024 triplets per dataset.
- **Phase 3 (finetune)**: fine-tune Instructor-large on the LLM-judged triplets.
- **Phase 4 (cluster)**: re-encode with the finetuned checkpoint, k-means at `k_in_scope`, persist predictions.

**Not on FASRC**: phase 2 (gpt-5-mini triplet judging via OpenAI Batch API)
and phase 2.5 (`convert_triplets`, a tiny CPU step). Both run locally and the
outputs are scp'd here.

## Sweep shape

Single axis: 7 datasets. SLURM array maps `SLURM_ARRAY_TASK_ID` → `DATASETS[task_id]`
(see `array_task.py`). The same array enumerates every phase.

| task_id | dataset            | n_docs |
|--------:|--------------------|-------:|
| 0       | banking77          | 3,080  |
| 1       | clinc150           | 5,500  |
| 2       | massive_intent     | 2,974  |
| 3       | massive_domain     | 2,974  |
| 4       | twenty_newsgroups  | 18,331 |
| 5       | goemotions         | 45,446 |
| 6       | stackexchange      | 4,156  |

## Layout

```
slurm/
├── README.md                            # this file
├── array_task.py                        # SLURM_ARRAY_TASK_ID + --phase → dispatch
├── setup_env.sh                         # uv install + uv sync (sourced from each sbatch)
├── prewarm.sh                           # one-shot: pre-download instructor-large into HF_HOME
├── run_clusterllm_prep.sbatch           # phase 0+1 (encode + sample)
├── run_clusterllm_finetune.sbatch       # phase 3 (triplet fine-tune)
├── run_clusterllm_cluster.sbatch        # phase 4 (re-encode + k-means + persist)
├── submit_calibration.sh                # single-task calibration on gpu_test
├── submit_all.sh                        # full sweep, phase 0+1
├── submit_finetune.sh                   # full sweep, phase 3
└── submit_cluster.sh                    # full sweep, phase 4
```

## Operator workflow

Assuming the project is at `/n/netscratch/economics/Lab/esilcock/agentic-clustering`
and `data/derived/` has been scp'd over once (gitignored):

```bash
# === Local laptop steps (already done as of 2026-05-23) ===
# - Phase 0+1 ran on FASRC; outputs are in data/clusterllm/<ds>/{base_embeds.hdf5, triplets.json}.
# - Phase 2 (gpt-5-mini Batch judging) ran locally; data/clusterllm/<ds>/triplets_judged.jsonl.
# - Phase 2.5 (convert_triplets) ran locally; data/clusterllm/<ds>/train_triplets.json.

# === Ship phase-2 outputs to FASRC ===
# Local laptop — copy only the new train_triplets.json files (the rest of
# data/clusterllm/ on FASRC is already complete from the earlier prep run).
for ds in banking77 clinc150 massive_intent massive_domain goemotions twenty_newsgroups stackexchange; do
  scp data/clusterllm/$ds/train_triplets.json \
      fasrc:/n/netscratch/economics/Lab/esilcock/agentic-clustering/data/clusterllm/$ds/
done

# === FASRC: env build (only after pyproject changes — sentence-transformers, torch, etc.) ===
ssh fasrc
cd /n/netscratch/economics/Lab/esilcock/agentic-clustering
export FORCE_UV_SYNC=1
source slurm/setup_env.sh
python -c "import sentence_transformers; print(sentence_transformers.__version__)"

# === Calibrate phase 3 on banking77 (smallest dataset) ===
export SLURM_ACCOUNT=economics
ARRAY_RANGE=0 ./slurm/submit_finetune.sh
# wait, then:
sacct -u $USER --name=clusterllm_finetune -X --format=JobID,State,Elapsed,MaxRSS
seff <jobid>

# === Full phase-3 sweep ===
# Bump WALLTIME in submit_finetune.sh if calibration exceeded 2h.
./slurm/submit_finetune.sh

# === After all 7 finetunes finish, kick off phase 4 ===
./slurm/submit_cluster.sh

# === Pull results back to laptop ===
# Just the predictions metadata + final embeds (checkpoints are heavy, leave on FASRC):
scp -r fasrc:/n/netscratch/economics/Lab/esilcock/agentic-clustering/results/predictions/clusterllm/ \
       ./results/predictions/
```

## Walltime expectations (to refine after calibration)

| Phase | Per-task estimate | Why |
|---|---|---|
| 0+1 (prep) | 1–10 min | Encode + sample; dominated by big datasets. Calibrated 2026-05-21 at 54s for banking77. |
| 3 (finetune) | 15–25 min | 15 epochs × 1024 triplets, batch 4 → ~3,800 steps. Pinned `--time=2:00:00` for safety. |
| 4 (cluster) | 5–15 min | Encode 3k–50k docs + one k-means at `k_in_scope`. |

Tested partition mix (per FASRC ops update 2026-05-23): `gpu_test` and
`gpu_requeue` are working better than `gpu`. The submit scripts default to
`gpu_test` (12h cap, sufficient for all three phases). Add
`CONSTRAINT=a100` when using `gpu_requeue` because its node set is
heterogeneous and the cu128 torch wheel doesn't include kernels for the
oldest GPUs there.

## What runs inside one array task per phase

```
slurm/run_clusterllm_finetune.sbatch
  ├─ source slurm/setup_env.sh           # uv sync, activate .venv
  ├─ export HF_HOME=$PROJECT/hf_cache    # prewarmed; instructor-large already on disk
  └─ CLUSTERLLM_PHASE=finetune python -m slurm.array_task
        └─ benchmarking.baselines.clusterllm.orchestrate.finetune(<ds>)
              ├─ loads train_triplets.json (from local convert step)
              ├─ SentenceTransformer("hkunlp/instructor-large") + include_prompt=False
              ├─ BiDirectionalInBatchContrastiveLoss (cl_temperature=0.01)
              ├─ SequentialSampler (upstream-faithful)
              └─ writes data/clusterllm/<ds>/checkpoint/final/
```

Phase 4 follows the same shape but calls `cluster(<ds>)` which chains
`embed_base(checkpoint_dir=...)` → `run_kmeans` → `write_run_artifacts`.
