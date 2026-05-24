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
from benchmarking.baselines.clusterllm.triplet_judge import (
    judge_triplets,
    judge_triplets_openai_batch,
)
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
    model: str | None = None,
    backend: str = "openai_batch",
) -> Path:
    """Phase 2: judge each triplet via the cheap-tier LLM.

    Default ``backend="openai_batch"`` routes through ``gpt-5-mini`` on the
    OpenAI Batch API, per SPEC §5.6.3 (>1,000-text rule). ``backend="claude"``
    is the legacy Opus path, kept for the archived ``triplets_judged.opus.jsonl``
    records and not used for new runs.

    Resumable — appends to ``triplets_judged.jsonl`` and skips records the
    file already contains.
    """
    triplets_path = CLUSTERLLM_ROOT / dataset_name / "triplets.json"
    if not triplets_path.exists():
        raise FileNotFoundError(f"phase 2 needs phase 1 output: {triplets_path}")

    out_path = CLUSTERLLM_ROOT / dataset_name / "triplets_judged.jsonl"
    if backend == "openai_batch":
        summary = judge_triplets_openai_batch(
            triplets_path=triplets_path,
            out_path=out_path,
            dataset=dataset_name,
            model=model or "gpt-5-mini",
        )
    elif backend == "claude":
        summary = judge_triplets(
            triplets_path=triplets_path,
            out_path=out_path,
            dataset=dataset_name,
            model=model or DEFAULT_MODEL,
            concurrency=concurrency,
        )
    else:
        raise ValueError(f"unknown judging backend: {backend!r}")

    print(f"[clusterllm/{dataset_name}/phase=judge] -> {out_path} | {summary}", flush=True)
    return out_path


def convert_triplets(dataset_name: str, *, overwrite: bool = False) -> Path:
    """Phase 2.5: turn judged triplets into (anchor, pos, neg) training rows.

    Inline equivalent of vendored ``convert_triplet.py``. We do it in-process
    rather than calling the vendored script for two reasons: (a) the vendored
    script hardcodes a path to its own ``prompts.json`` whose dataset keys
    don't match ours; (b) it's 30 lines of pure data shuffling — no point in
    a subprocess.

    Output JSON shape matches upstream exactly so ``finetune.py`` can consume
    either source identically. Each row is
    ``{'query': [prompt, anchor], 'pos': [prompt, pos], 'neg': [prompt, neg], 'task_name': dataset, ...}``.
    Records with ambiguous predictions (``len(prediction) != 1``) are skipped,
    matching upstream's behaviour.
    """
    judged_path = CLUSTERLLM_ROOT / dataset_name / "triplets_judged.jsonl"
    if not judged_path.exists():
        raise FileNotFoundError(f"phase 2.5 needs phase 2 output: {judged_path}")

    out_path = CLUSTERLLM_ROOT / dataset_name / "train_triplets.json"
    if out_path.exists() and not overwrite:
        print(f"[clusterllm/{dataset_name}/phase=convert] cache hit -> {out_path}", flush=True)
        return out_path

    adapted = adapt(dataset_name)
    with adapted.jsonl_path.open(encoding="utf-8") as f:
        inp_data = [json.loads(line) for line in f if line.strip()]
    prompt = _load_instructor_prompt(dataset_name)

    out_rows: list[dict] = []
    skipped = 0
    with judged_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            preds = rec.get("prediction", [])
            if len(preds) != 1:
                skipped += 1
                continue
            qi, c1i, c2i = int(rec["query_idx"]), int(rec["choice1_idx"]), int(rec["choice2_idx"])
            if preds[0] == " 1":
                pos_text = inp_data[c1i]["input"]
                neg_text = inp_data[c2i]["input"]
            elif preds[0] == " 2":
                pos_text = inp_data[c2i]["input"]
                neg_text = inp_data[c1i]["input"]
            else:
                skipped += 1
                continue
            out_rows.append({
                "query": [prompt, inp_data[qi]["input"]],
                "pos":   [prompt, pos_text],
                "neg":   [prompt, neg_text],
                "task_name": dataset_name,
                "query_idx": qi, "choice1_idx": c1i, "choice2_idx": c2i,
            })

    out_path.write_text(json.dumps(out_rows), encoding="utf-8")
    print(
        f"[clusterllm/{dataset_name}/phase=convert] {len(out_rows)} train triplets "
        f"({skipped} skipped as ambiguous) -> {out_path}",
        flush=True,
    )
    return out_path


def finetune(
    dataset_name: str,
    *,
    overwrite: bool = False,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
) -> Path:
    """Phase 3: fine-tune Instructor-large on this dataset's train triplets.

    Returns the path to the final checkpoint directory, suitable for passing
    to ``embed_base(checkpoint_dir=...)`` in phase 4.
    """
    from benchmarking.baselines.clusterllm.finetune import (
        UPSTREAM_BATCH_SIZE,
        UPSTREAM_EPOCHS,
        UPSTREAM_LR,
        finetune_one,
    )

    train_triplets_path = CLUSTERLLM_ROOT / dataset_name / "train_triplets.json"
    if not train_triplets_path.exists():
        raise FileNotFoundError(f"phase 3 needs phase 2.5 output: {train_triplets_path}")

    out_dir = CLUSTERLLM_ROOT / dataset_name / "checkpoint"
    final_dir = out_dir / "final"
    if final_dir.exists() and not overwrite:
        print(f"[clusterllm/{dataset_name}/phase=finetune] cache hit -> {final_dir}", flush=True)
        return final_dir

    return finetune_one(
        train_triplets_path=train_triplets_path,
        output_dir=out_dir,
        learning_rate=learning_rate if learning_rate is not None else UPSTREAM_LR,
        num_train_epochs=epochs if epochs is not None else UPSTREAM_EPOCHS,
        per_device_train_batch_size=batch_size if batch_size is not None else UPSTREAM_BATCH_SIZE,
    )


def cluster(
    dataset_name: str,
    *,
    seed: int = 0,
    overwrite: bool = False,
) -> dict:
    """Phase 4: re-encode with the finetuned checkpoint and k-means at k_in_scope.

    Persists predictions + metrics via ``write_run_artifacts`` to
    ``results/predictions/clusterllm/<dataset>/seed=<seed>.*``, matching the
    shape used by every other baseline in this repo.
    """
    from benchmarking.baselines.kmeans import run_kmeans
    from benchmarking.data_processing.load import load_processed
    from benchmarking.evaluation.cost import CostAccumulator, WallClock
    from benchmarking.evaluation.metrics import compute_partition_metrics
    from benchmarking.evaluation.persistence import (
        DocPrediction, TaxonomyEntry, write_run_artifacts,
    )

    import h5py
    import numpy as np

    final_ckpt = CLUSTERLLM_ROOT / dataset_name / "checkpoint" / "final"
    if not final_ckpt.exists():
        raise FileNotFoundError(f"phase 4 needs phase 3 output: {final_ckpt}")

    final_embeds_path = embed_base(dataset_name, checkpoint_dir=final_ckpt, overwrite=overwrite)

    ds = load_processed(dataset_name)
    k = int(ds.meta["k_in_scope"])
    gold_ids = [int(d["gold_label_id"]) for d in ds.documents]

    cost = CostAccumulator()
    with WallClock(cost):
        with h5py.File(final_embeds_path, "r") as fh:
            emb = np.array(fh["embeds"])
        result = run_kmeans(embeddings=emb, k=k, seed=seed)

    metrics = compute_partition_metrics(pred_ids=result.pred_ids, gold_ids=gold_ids)

    taxonomy = [
        TaxonomyEntry(cluster_id=i, label=f"cluster_{i}", description="")
        for i in range(k)
    ]
    predictions = [
        DocPrediction(
            doc_id=doc["doc_id"],
            text=doc["text"],
            gold_label=doc["gold_label_name"],
            gold_label_id=int(doc["gold_label_id"]),
            is_none=bool(doc["is_none"]),
            predicted_cluster_id=cid,
            predicted_cluster_label=f"cluster_{cid}",
            confidence=None,
            iteration=0,
        )
        for doc, cid in zip(ds.documents, result.pred_ids)
    ]
    hyperparameters = {
        **result.hyperparameters,
        "encoder": INSTRUCTOR_MODEL,
        "encoder_finetuned": True,
        "checkpoint_dir": str(final_ckpt),
    }
    write_run_artifacts(
        method="clusterllm",
        dataset=dataset_name,
        seed=seed,
        predictions=predictions,
        taxonomy=taxonomy,
        cost=cost,
        model_versions={"encoder": INSTRUCTOR_MODEL},
        iterations=0,
        metrics=metrics.to_dict(),
        hyperparameters=hyperparameters,
        extra_meta={"k_used": k, "n_docs": len(ds.documents)},
    )

    summary = {"dataset": dataset_name, "n": len(ds.documents), "k": k, **metrics.to_dict()}
    print(f"[clusterllm/{dataset_name}/phase=cluster] {summary}", flush=True)
    return summary
