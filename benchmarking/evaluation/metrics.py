"""Partition-quality metrics for clustering: ARI, NMI, ACC (Hungarian).

All three are computed on the full document set, including "none" gold docs
(gold_label_id == -1). Per SPEC §5.5, methods without native "none" support
will see their rejected-equivalent docs distributed across the k in-scope
clusters and be penalised here accordingly.

ARI / NMI are partition metrics and treat "none" as just another gold class.
ACC uses scipy's Hungarian solver on a rectangular contingency matrix
(predicted clusters × gold classes), which handles the k vs k+1 case
natively — see SPEC §5.5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


@dataclass(frozen=True)
class PartitionMetrics:
    ari: float
    nmi: float
    acc: float

    def to_dict(self) -> dict:
        return {"ari": self.ari, "nmi": self.nmi, "acc": self.acc}


def _contingency(pred_ids: np.ndarray, gold_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a (n_pred_clusters × n_gold_classes) contingency matrix.

    Returns (matrix, unique_pred, unique_gold) so callers can map matrix rows/cols
    back to the original label ids if needed.
    """
    unique_pred, pred_idx = np.unique(pred_ids, return_inverse=True)
    unique_gold, gold_idx = np.unique(gold_ids, return_inverse=True)
    matrix = np.zeros((unique_pred.size, unique_gold.size), dtype=np.int64)
    np.add.at(matrix, (pred_idx, gold_idx), 1)
    return matrix, unique_pred, unique_gold


def hungarian_accuracy(pred_ids: np.ndarray, gold_ids: np.ndarray) -> float:
    """Optimal-assignment accuracy (a.k.a. cluster purity under Hungarian alignment)."""
    if pred_ids.size == 0:
        return 0.0
    matrix, _, _ = _contingency(pred_ids, gold_ids)
    row_ind, col_ind = linear_sum_assignment(matrix, maximize=True)
    return float(matrix[row_ind, col_ind].sum()) / float(pred_ids.size)


def compute_partition_metrics(pred_ids, gold_ids) -> PartitionMetrics:
    pred = np.asarray(pred_ids)
    gold = np.asarray(gold_ids)
    assert pred.shape == gold.shape, f"shape mismatch: pred={pred.shape} gold={gold.shape}"
    return PartitionMetrics(
        ari=float(adjusted_rand_score(gold, pred)),
        nmi=float(normalized_mutual_info_score(gold, pred)),
        acc=hungarian_accuracy(pred, gold),
    )
