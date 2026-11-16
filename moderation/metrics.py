"""Evaluation helpers - every model is scored with the same functions."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)


def classification_metrics(y_true, p_off, threshold: float = 0.5) -> dict:
    """Macro-F1 (main metric, classes are imbalanced) plus supporting numbers."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(p_off) >= threshold).astype(int)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "off_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "off_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "off_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "threshold": float(threshold),
    }


def best_threshold(y_true, p_off, grid=None) -> float:
    """Pick the decision threshold that maximises macro-F1 on the VALIDATION set."""
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2) if grid is None else grid
    scores = [f1_score(y_true, (np.asarray(p_off) >= t).astype(int), average="macro") for t in grid]
    return float(grid[int(np.argmax(scores))])
