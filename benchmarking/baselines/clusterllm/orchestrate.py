"""Phase orchestrator for the ClusterLLM baseline.

Drives the vendored author code (phases 0/1/2.5/3/4) via subprocess, and our
own ``triplet_judge`` (phase 2) in-process. Each phase is idempotent and
caches its artifact under ``data/clusterllm/<dataset>/``:

| Phase | Output                              | Runner                     |
|-------|-------------------------------------|----------------------------|
| 0     | base_embeds.hdf5                    | vendored get_embedding.py  |
| 1     | triplets.json                       | vendored triplet_sampling  |
| 2     | triplets_judged.jsonl               | this repo's triplet_judge  |
| 2.5   | train_triplets.json                 | vendored convert_triplet   |
| 3     | checkpoints/instructor-large/...    | vendored finetune.py (GPU) |
| 4     | final_embeds.hdf5 + results/        | vendored get_embedding.py  |

Tonight's overnight call site is phase 2 only. Phases 0+1 prepare its input;
phases 3+4 are tomorrow on FASRC GPU.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from benchmarking.baselines.clusterllm.dataset_adapter import adapt
from benchmarking.baselines.clusterllm.triplet_judge import judge_triplets
from benchmarking.llm_clients.claude_code import DEFAULT_MODEL
from benchmarking.paths import DATA

VENDORED_PERSPECTIVE = Path(__file__).resolve().parent / "_vendored" / "perspective"
TRIPLET_SAMPLING_PY = VENDORED_PERSPECTIVE / "1_predict_triplet" / "triplet_sampling.py"
FINETUNE_DIR = VENDORED_PERSPECTIVE / "2_finetune"
GET_EMBEDDING_PY = FINETUNE_DIR / "get_embedding.py"
CONVERT_TRIPLET_PY = FINETUNE_DIR / "convert_triplet.py"
FINETUNE_PY = FINETUNE_DIR / "finetune.py"

CLUSTERLLM_ROOT = DATA / "clusterllm"

INSTRUCTOR_MODEL = "hkunlp/instructor-large"


_VENDORED_WRAPPER = "benchmarking.baselines.clusterllm._run_vendored"


def _run_vendored(script: Path, args: list[str], *, cwd: Path) -> None:
    """Dispatch a vendored script via _run_vendored so truststore is in place."""
    cmd = [sys.executable, "-m", _VENDORED_WRAPPER, str(script), *args]
    print(f"  $ (cwd={cwd}) {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=cwd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"subprocess failed (exit {proc.returncode}): {' '.join(cmd)}"
        )


def embed_base(dataset_name: str, *, overwrite: bool = False) -> Path:
    """Phase 0: Instructor-large embeddings of all docs."""
    adapted = adapt(dataset_name)
    out_dir = CLUSTERLLM_ROOT / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    result_file = out_dir / "base_embeds.hdf5"

    if result_file.exists() and not overwrite:
        print(f"[clusterllm/{dataset_name}/phase=embed] cache hit -> {result_file}", flush=True)
        return result_file

    print(f"[clusterllm/{dataset_name}/phase=embed] encoding {adapted.n_docs} docs with {INSTRUCTOR_MODEL}", flush=True)
    args = [
        "--model_name", INSTRUCTOR_MODEL,
        "--task_name", dataset_name,
        "--data_path", str(adapted.jsonl_path),
        "--result_file", str(result_file),
        "--prompt", INSTRUCTOR_MODEL,
        "--scale", "large",
    ]
    _run_vendored(GET_EMBEDDING_PY, args, cwd=FINETUNE_DIR)
    return result_file


def sample_triplets(
    dataset_name: str,
    *,
    seed: int = 100,
    max_query: int = 1024,
    overwrite: bool = False,
) -> Path:
    """Phase 1: entropy-rank ambiguous points, sample triplets.

    Uses the author's default sampling hyperparameters
    (max_distance=67, large_ent_prop=0.20, close_cluster_prop=0.02, shuffle).
    """
    embed_file = CLUSTERLLM_ROOT / dataset_name / "base_embeds.hdf5"
    if not embed_file.exists():
        raise FileNotFoundError(f"phase 1 needs phase 0 output: {embed_file}")

    adapted = adapt(dataset_name)
    out_dir = CLUSTERLLM_ROOT / dataset_name
    triplets_path = out_dir / "triplets.json"

    if triplets_path.exists() and not overwrite:
        print(f"[clusterllm/{dataset_name}/phase=sample] cache hit -> {triplets_path}", flush=True)
        return triplets_path

    work_dir = out_dir / "_sampling_work"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    args = [
        "--dataset", dataset_name,
        "--data_path", str(adapted.jsonl_path),
        "--feat_path", str(embed_file),
        "--embed_method", "instructor",
        "--scale", "large",
        "--max_query", str(max_query),
        "--shuffle_inds",
        "--out_dir", str(work_dir),
        "--seed", str(seed),
    ]
    _run_vendored(TRIPLET_SAMPLING_PY, args, cwd=TRIPLET_SAMPLING_PY.parent)

    produced = list(work_dir.glob("*.json"))
    if len(produced) != 1:
        raise RuntimeError(
            f"phase 1 expected 1 .json output under {work_dir}, got {len(produced)}: {produced}"
        )
    shutil.move(str(produced[0]), str(triplets_path))
    shutil.rmtree(work_dir)
    print(f"[clusterllm/{dataset_name}/phase=sample] -> {triplets_path}", flush=True)
    return triplets_path


def judge(
    dataset_name: str,
    *,
    concurrency: int = 4,
    model: str = DEFAULT_MODEL,
) -> Path:
    """Phase 2: judge each triplet via Claude Code (Opus 4.7).

    Resumable — appends to ``triplets_judged.jsonl`` and skips records the
    file already contains. Usage limits are absorbed inside ``call_claude``.
    """
    triplets_path = CLUSTERLLM_ROOT / dataset_name / "triplets.json"
    if not triplets_path.exists():
        raise FileNotFoundError(f"phase 2 needs phase 1 output: {triplets_path}")

    out_path = CLUSTERLLM_ROOT / dataset_name / "triplets_judged.jsonl"
    summary = judge_triplets(
        triplets_path=triplets_path,
        out_path=out_path,
        dataset=dataset_name,
        model=model,
        concurrency=concurrency,
    )
    print(f"[clusterllm/{dataset_name}/phase=judge] -> {out_path} | {summary}", flush=True)
    return out_path
