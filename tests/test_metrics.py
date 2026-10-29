import numpy as np
import pytest

from clf.metrics import compute_metrics


def test_given_preds_and_labels_when_computing_then_accuracy_and_macro_f1():
    preds = np.array([0, 1, 1])
    labels = np.array([0, 1, 0])

    result = compute_metrics((preds, labels))

    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["macro_f1"] == pytest.approx(0.6667, abs=1e-4)


def test_given_logits_when_computing_then_argmax_is_used():
    logits = np.array([[2.0, 0.1], [0.2, 1.5], [0.0, 3.0]])
    labels = np.array([0, 1, 1])

    result = compute_metrics((logits, labels))

    assert result == {"accuracy": 1.0, "macro_f1": 1.0}
