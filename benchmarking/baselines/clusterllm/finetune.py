"""Phase-3 of ClusterLLM: fine-tune Instructor-large with triplet supervision.

Modern ``sentence-transformers`` 5.x re-implementation of the vendored
``perspective/2_finetune/finetune.py``. Faithful to upstream:

- Encoder weights:        ``hkunlp/instructor-large`` (same checkpoint).
- Per-dataset instruction: same strings, prepended via ST's ``prompt=`` mechanism.
- Pooling:                separated pooling via ``include_prompt=False`` — the
                          instruction tokens participate in encoder self-attention
                          but are masked out of the final mean-pool, matching the
                          legacy ``InstructorEmbedding.forward``'s
                          ``attention_mask[:context_masks] = 0``.
- Loss:                   bi-directional in-batch contrastive cross-entropy with
                          ``cl_temperature``, identical to
                          ``InstructorTrainer.compute_loss``.
- Sampler:                ``SequentialSampler`` (upstream's choice — see
                          ``_get_train_sampler``).
- Hyperparameters:        ``lr=2e-6``, ``epochs=15``, ``per_device_batch=4``,
                          ``max_seq=512`` — pinned from upstream
                          ``scripts/finetune.sh``.

The only divergences from upstream are unavoidable PyTorch / transformers
version drift (float arithmetic, attention-implementation defaults).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Pinned hyperparameters (from upstream scripts/finetune.sh; flagged in SPEC).
# ---------------------------------------------------------------------------
UPSTREAM_LR = 2e-6
UPSTREAM_EPOCHS = 15
UPSTREAM_BATCH_SIZE = 4
UPSTREAM_MAX_SOURCE_LENGTH = 512
UPSTREAM_CL_TEMPERATURE = 0.01

BASE_MODEL = "hkunlp/instructor-large"


# ---------------------------------------------------------------------------
# Custom loss: byte-for-byte match to InstructorTrainer.compute_loss
# ---------------------------------------------------------------------------
class BiDirectionalInBatchContrastiveLoss(nn.Module):
    """Mirrors ``InstructorTrainer.compute_loss`` from vendored finetune.py.

    For a batch of ``n`` (anchor, positive, negative) triples:

    Direction A — anchor → (positive + ALL negatives in batch):
        For each i in 0..n-1, score row =
            [cos(a_i, p_i),  cos(a_i, n_0),  cos(a_i, n_1), ...,  cos(a_i, n_{n-1})]
        scaled by 1/cl_temperature, then softmax cross-entropy with label=0
        (the positive sits at column 0).

    Direction B — positive → (anchor + OTHER anchors in batch, excluding self):
        For each i, row =
            [cos(p_i, a_i)] + [cos(p_i, a_j) for j != i]
        same softmax CE with label=0.

    Final loss = (loss_A + loss_B) / 2 (upstream returns both then sums; ST
    expects a single scalar — equivalent because they share grads).
    """

    def __init__(self, model, cl_temperature: float = UPSTREAM_CL_TEMPERATURE):
        super().__init__()
        self.model = model
        self.cl_temperature = float(cl_temperature)

    def forward(
        self,
        sentence_features: list[dict[str, torch.Tensor]],
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # ST passes one features dict per input column, in the column order
        # registered on the Dataset. We always pass (anchor, positive, negative).
        if len(sentence_features) != 3:
            raise RuntimeError(
                f"expected 3 sentence_features (anchor/pos/neg); got {len(sentence_features)}"
            )
        emb_a = self.model(sentence_features[0])["sentence_embedding"]
        emb_p = self.model(sentence_features[1])["sentence_embedding"]
        emb_n = self.model(sentence_features[2])["sentence_embedding"]
        return self._bidirectional_loss(emb_a, emb_p, emb_n)

    def _bidirectional_loss(self, emb_a, emb_p, emb_n) -> torch.Tensor:
        n = emb_a.size(0)
        if n == 0:
            return emb_a.sum() * 0.0  # degenerate batch

        # Cosine sims; / cl_temperature for the InfoNCE-style scaling.
        # cos(a_i, p_i): shape (n,)
        ap_diag = F.cosine_similarity(emb_a, emb_p, dim=-1) / self.cl_temperature
        # cos(a_i, n_j): shape (n, n)
        an_full = (
            F.cosine_similarity(emb_a.unsqueeze(1), emb_n.unsqueeze(0), dim=-1)
            / self.cl_temperature
        )
        # Row i = [ap_diag[i], an_full[i, 0..n-1]] → shape (n, n+1); label=0
        logits_a = torch.cat([ap_diag.unsqueeze(1), an_full], dim=1)
        labels_a = torch.zeros(n, dtype=torch.long, device=emb_a.device)
        loss_a = F.cross_entropy(logits_a, labels_a)

        # Direction B: anchor as the "negative" set for each positive, skipping
        # the diagonal so a positive doesn't appear as its own negative.
        pa_diag = F.cosine_similarity(emb_p, emb_a, dim=-1) / self.cl_temperature
        pa_full = (
            F.cosine_similarity(emb_p.unsqueeze(1), emb_a.unsqueeze(0), dim=-1)
            / self.cl_temperature
        )
        # Mask out the diagonal (j == i) by replacing with -inf so it contributes
        # nothing to the softmax. Matches upstream's `if i == j: continue`.
        mask = torch.eye(n, dtype=torch.bool, device=emb_p.device)
        pa_off_diag = pa_full.masked_fill(mask, float("-inf"))
        logits_b = torch.cat([pa_diag.unsqueeze(1), pa_off_diag], dim=1)
        labels_b = torch.zeros(n, dtype=torch.long, device=emb_p.device)
        loss_b = F.cross_entropy(logits_b, labels_b)

        return (loss_a + loss_b) / 2.0


# ---------------------------------------------------------------------------
# Training driver
# ---------------------------------------------------------------------------
def _load_train_triplets(path: Path) -> tuple[list[str], list[str], list[str], str]:
    """Load convert_triplets output → (anchor, positive, negative, prompt)."""
    with path.open(encoding="utf-8") as f:
        rows = json.load(f)
    if not rows:
        raise RuntimeError(f"empty triplet file: {path}")

    # convert_triplets stores rows as {'query': [prompt, text], 'pos': [...], 'neg': [...]}
    # The prompt is the same across all rows in one dataset.
    prompt = rows[0]["query"][0]
    for r in rows:
        for k in ("query", "pos", "neg"):
            if r[k][0] != prompt:
                raise RuntimeError(
                    f"inconsistent prompt in {path}: expected {prompt!r}, "
                    f"got {r[k][0]!r}"
                )
    anchors = [r["query"][1] for r in rows]
    positives = [r["pos"][1] for r in rows]
    negatives = [r["neg"][1] for r in rows]
    return anchors, positives, negatives, prompt


def finetune_one(
    train_triplets_path: Path,
    output_dir: Path,
    *,
    base_model: str = BASE_MODEL,
    learning_rate: float = UPSTREAM_LR,
    num_train_epochs: int = UPSTREAM_EPOCHS,
    per_device_train_batch_size: int = UPSTREAM_BATCH_SIZE,
    max_seq_length: int = UPSTREAM_MAX_SOURCE_LENGTH,
    cl_temperature: float = UPSTREAM_CL_TEMPERATURE,
    seed: int = 42,
    log_prefix: str = "[clusterllm/finetune]",
) -> Path:
    """Fine-tune one dataset; return the saved checkpoint path."""
    # Local imports keep CLI/help fast and surface missing deps lazily.
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.training_args import BatchSamplers
    from torch.utils.data import SequentialSampler

    anchors, positives, negatives, prompt = _load_train_triplets(train_triplets_path)
    print(
        f"{log_prefix} loaded {len(anchors)} triplets from {train_triplets_path}; "
        f"prompt={prompt!r}",
        flush=True,
    )

    train_ds = Dataset.from_dict(
        {"anchor": anchors, "positive": positives, "negative": negatives}
    )

    model = SentenceTransformer(base_model)
    # Force the pooling module to mask out the instruction tokens from the
    # final mean-pool, matching legacy InstructorEmbedding's separated pooling.
    # The instruction still flows through encoder self-attention so document
    # tokens can attend to it; only the final aggregate excludes it.
    for module in model:
        if hasattr(module, "include_prompt"):
            module.include_prompt = False
    # Cap encoder seq length to match upstream's 512.
    model.max_seq_length = max_seq_length

    # Apply the same Instructor instruction to all three columns at encode time.
    # set_pooling_include_prompt isn't on the trainer — the per-column prompt
    # passes through ST's tokenization layer directly via ``args.prompts``.
    prompts_for_columns = {"anchor": prompt, "positive": prompt, "negative": prompt}

    output_dir.mkdir(parents=True, exist_ok=True)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        # Upstream's scripts/finetune.sh sets only the flags listed above; the
        # rest are Seq2SeqTrainingArguments defaults — warmup_ratio=0,
        # weight_decay=0, gradient_accumulation_steps=1.
        warmup_ratio=0.0,
        weight_decay=0.0,
        gradient_accumulation_steps=1,
        # ST 5.x: prompts get prepended at tokenization; include_prompt=False
        # above makes pooling ignore them.
        prompts=prompts_for_columns,
        # No in-batch contamination protection needed — we explicitly handle
        # the diagonal in our loss.
        batch_sampler=BatchSamplers.BATCH_SAMPLER,
        report_to="none",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=50,
        seed=seed,
        dataloader_drop_last=False,
        fp16=False,  # mixed-precision changes loss scale; keep upstream-faithful
        bf16=torch.cuda.is_available()
        and torch.cuda.get_device_capability(0)[0] >= 8,  # A100+ only
        # Re-using the existing dataset cache speeds re-runs.
        dataloader_num_workers=2,
    )

    loss = BiDirectionalInBatchContrastiveLoss(model, cl_temperature=cl_temperature)

    # Upstream's _get_train_sampler returns SequentialSampler when
    # world_size <= 1 — we mirror that by subclassing the trainer.
    class _SequentialTrainer(SentenceTransformerTrainer):
        def _get_train_sampler(self, *_args, **_kwargs):  # type: ignore[override]
            return SequentialSampler(self.train_dataset)

    trainer = _SequentialTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        loss=loss,
    )
    trainer.train()

    # Save the fine-tuned model in ST's standard layout so phase-4 embedding can
    # load it via SentenceTransformer(checkpoint_dir).
    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))
    print(f"{log_prefix} saved checkpoint -> {final_dir}", flush=True)
    return final_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-triplets", type=Path, required=True,
        help="convert_triplets output JSON for one dataset.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Checkpoint output dir. Final model lands under <output_dir>/final/.",
    )
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--lr", type=float, default=UPSTREAM_LR)
    parser.add_argument("--epochs", type=int, default=UPSTREAM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=UPSTREAM_BATCH_SIZE)
    parser.add_argument("--cl-temperature", type=float, default=UPSTREAM_CL_TEMPERATURE)
    parser.add_argument("--max-seq-length", type=int, default=UPSTREAM_MAX_SOURCE_LENGTH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    finetune_one(
        train_triplets_path=args.train_triplets,
        output_dir=args.output_dir,
        base_model=args.base_model,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        cl_temperature=args.cl_temperature,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
