"""Phase orchestrator for the ClusterLLM baseline.

| Phase | Output                              | Runner                                |
|-------|-------------------------------------|---------------------------------------|
| 0     | base_embeds.hdf5                    | ``embed_base`` (SentenceTransformer)  |
| 1     | triplets.json                       | vendored triplet_sampling.py          |
| 2     | triplets_judged.jsonl               | this repo's triplet_judge             |
| 2.5   | train_triplets.json                 | vendored convert_triplet.py           |
| 3     | checkpoints/instructor-large/...    | vendored finetune.py (GPU)            |
| 4     | final_embeds.hdf5 + results/        | ``embed_base`` (with finetuned ckpt)  |

Tonight's overnight call site is phase 2 only. Phases 0+1 prepare its input;
phases 3+4 are tomorrow on FASRC GPU.

Encoder note (phase 0/4): we use ``SentenceTransformer("hkunlp/instructor-large")``
with the modern ``prompt=`` argument rather than subprocessing into the
vendored ``get_embedding.py``. The vendored encoder targets sentence-
transformers 2.2 internals and breaks against current ST 5.x; rather than
patch the science-adjacent code we drive the same model weights through
the ST 5.x prompt API, with the per-dataset Instructor instructions from
``instructor_prompts.json`` (copied verbatim from the authors' phase-3
prompts.json). Documented in SPEC §5.6.3.
"""

from __future__ import annotations

import json
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
CONVERT_TRIPLET_PY = FINETUNE_DIR / "convert_triplet.py"
FINETUNE_PY = FINETUNE_DIR / "finetune.py"

INSTRUCTOR_PROMPTS_PATH = Path(__file__).resolve().parent / "instructor_prompts.json"

CLUSTERLLM_ROOT = DATA / "clusterllm"
INSTRUCTOR_MODEL = "hkunlp/instructor-large"
ENCODE_BATCH_SIZE = 32


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


def _load_instructor_prompt(dataset_name: str) -> str:
    prompts = json.loads(INSTRUCTOR_PROMPTS_PATH.read_text(encoding="utf-8"))
    if dataset_name not in prompts:
        raise KeyError(
            f"No Instructor prompt for dataset {dataset_name!r} in "
            f"{INSTRUCTOR_PROMPTS_PATH}. Available: "
            f"{[k for k in prompts if not k.startswith('_')]}"
        )
    return prompts[dataset_name]


def embed_base(
    dataset_name: str,
    *,
    overwrite: bool = False,
    checkpoint_dir: Path | None = None,
) -> Path:
    """Phase 0 (or phase 4 with ``checkpoint_dir``): Instructor-large embeddings.

    Uses ``SentenceTransformer`` with the modern ``prompt=`` API instead of the
    vendored Instructor wrapper — see module docstring for the why. Output
    layout matches what the vendored ``triplet_sampling.py`` expects:
    HDF5 with a single dataset named ``embeds`` of shape ``(N, D)``.
    """
    import h5py
    import numpy as np
    from sentence_transformers import SentenceTransformer

    adapted = adapt(dataset_name)
    out_dir = CLUSTERLLM_ROOT / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    result_file = (
        out_dir / "final_embeds.hdf5" if checkpoint_dir is not None else out_dir / "base_embeds.hdf5"
    )

    if result_file.exists() and not overwrite:
        print(
            f"[clusterllm/{dataset_name}/phase=embed] cache hit -> {result_file}",
            flush=True,
        )
        return result_file

    prompt = _load_instructor_prompt(dataset_name)
    print(
        f"[clusterllm/{dataset_name}/phase=embed] encoding {adapted.n_docs} docs "
        f"with {INSTRUCTOR_MODEL} | prompt={prompt!r}"
        + (f" | checkpoint={checkpoint_dir}" if checkpoint_dir is not None else ""),
        flush=True,
    )

    texts = [json.loads(line)["input"] for line in adapted.jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(texts) != adapted.n_docs:
        raise RuntimeError(
            f"adapter said {adapted.n_docs} docs but loaded {len(texts)} from {adapted.jsonl_path}"
        )

    model_id = str(checkpoint_dir) if checkpoint_dir is not None else INSTRUCTOR_MODEL
    model = SentenceTransformer(model_id)
    embeddings = model.encode(
        texts,
        prompt=prompt,
        batch_size=ENCODE_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    with h5py.File(result_file, "w") as f:
        f.create_dataset("embeds", data=embeddings)
    print(
        f"[clusterllm/{dataset_name}/phase=embed] wrote {embeddings.shape} -> {result_file}",
        flush=True,
    )
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
