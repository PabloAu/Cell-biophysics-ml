from __future__ import annotations

import numpy as np
import pytest

from cell_biophysics_benchmark.evaluation import classification_report


def test_perfect_classification_report() -> None:
    classes = ["brownian", "directed"]
    truth = ["brownian", "directed", "brownian"]
    probabilities = np.asarray([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]])
    report = classification_report(truth, probabilities, classes, confidence_threshold=0.85)
    assert report["accuracy"] == 1.0
    assert report["macro_f1"] == 1.0
    assert report["coverage"] == pytest.approx(2 / 3)
    assert report["selective_accuracy"] == 1.0


def test_probability_rows_must_be_normalized() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        classification_report(["a"], np.asarray([[0.4, 0.4]]), ["a", "b"])
