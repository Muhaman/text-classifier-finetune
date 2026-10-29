"""Evaluation metrics for the Trainer and the final report."""

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import EvalPrediction


def compute_metrics(eval_pred: EvalPrediction | tuple[Any, Any]) -> dict[str, float]:
    """Accuracy and macro-F1 from ``(logits_or_predictions, labels)``."""
    if isinstance(eval_pred, tuple):
        predictions, labels = eval_pred
    else:
        predictions, labels = eval_pred.predictions, eval_pred.label_ids
    predicted = np.asarray(predictions)
    if predicted.ndim > 1:
        predicted = predicted.argmax(axis=-1)
    expected = np.asarray(labels)
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1": float(f1_score(expected, predicted, average="macro", zero_division=0)),
    }
