"""Metrics that foreground calibration, abstention, and measurement regimes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..types import TrajectorySample


def _confusion(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> np.ndarray:
    index = {label: position for position, label in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for truth, prediction in zip(y_true, y_pred, strict=True):
        matrix[index[truth], index[prediction]] += 1
    return matrix


def _expected_calibration_error(
    correct: np.ndarray, confidence: np.ndarray, bins: int
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(correct)
    result = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            result += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return float(result) if total else np.nan


def classification_report(
    y_true: Sequence[str],
    probabilities: np.ndarray,
    classes: Sequence[str],
    *,
    confidence_threshold: float = 0.70,
    calibration_bins: int = 10,
) -> dict[str, Any]:
    """Compute classification, probabilistic, and selective metrics."""

    classes_array = np.asarray(classes, dtype=object)
    truth = np.asarray(y_true, dtype=object)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(truth), len(classes_array)):
        raise ValueError("Probability matrix shape does not match labels/classes")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Each probability row must sum to one")
    predictions = classes_array[np.argmax(probabilities, axis=1)]
    confidence = probabilities.max(axis=1)
    correct = predictions == truth
    confusion = _confusion(truth, predictions, classes_array)

    recalls = np.divide(
        np.diag(confusion),
        confusion.sum(axis=1),
        out=np.zeros(len(classes_array), dtype=float),
        where=confusion.sum(axis=1) > 0,
    )
    precisions = np.divide(
        np.diag(confusion),
        confusion.sum(axis=0),
        out=np.zeros(len(classes_array), dtype=float),
        where=confusion.sum(axis=0) > 0,
    )
    f1 = np.divide(
        2 * precisions * recalls,
        precisions + recalls,
        out=np.zeros(len(classes_array), dtype=float),
        where=(precisions + recalls) > 0,
    )
    class_index = {label: i for i, label in enumerate(classes_array)}
    target = np.zeros_like(probabilities)
    for row, label in enumerate(truth):
        target[row, class_index[label]] = 1.0
    selected = confidence >= confidence_threshold
    return {
        "n_samples": int(len(truth)),
        "accuracy": float(correct.mean()),
        "balanced_accuracy": float(recalls.mean()),
        "macro_f1": float(f1.mean()),
        "multiclass_brier": float(np.square(probabilities - target).sum(axis=1).mean()),
        "negative_log_likelihood": float(
            -np.log(np.clip(probabilities[target.astype(bool)], 1e-12, 1.0)).mean()
        ),
        "expected_calibration_error": _expected_calibration_error(
            correct, confidence, calibration_bins
        ),
        "confidence_threshold": float(confidence_threshold),
        "coverage": float(selected.mean()),
        "selective_accuracy": float(correct[selected].mean()) if selected.any() else None,
        "per_class": {
            str(label): {
                "precision": float(precisions[index]),
                "recall": float(recalls[index]),
                "f1": float(f1[index]),
                "support": int(confusion[index].sum()),
            }
            for index, label in enumerate(classes_array)
        },
        "classes": classes_array.tolist(),
        "confusion_matrix": confusion.tolist(),
    }


def identifiability_table(
    samples: Sequence[TrajectorySample],
    probabilities: np.ndarray,
    classes: Sequence[str],
    *,
    confidence_threshold: float = 0.70,
) -> pd.DataFrame:
    """Return per-state/per-acquisition-bin accuracy and selective coverage."""

    classes_array = np.asarray(classes, dtype=object)
    predicted = classes_array[np.argmax(probabilities, axis=1)]
    confidence = probabilities.max(axis=1)
    rows: list[dict[str, Any]] = []
    for sample, prediction, score in zip(samples, predicted, confidence, strict=True):
        acquisition = sample.acquisition_params
        sigma_nm = 1000.0 * acquisition.localization_sigma_um
        exposure_fraction = acquisition.exposure_time_s / acquisition.frame_interval_s
        rows.append(
            {
                "split": sample.split,
                "state": sample.state_label,
                "localization_bin_nm": pd.cut(
                    [sigma_nm], bins=[0, 20, 40, 80, 150, np.inf], right=False
                )[0],
                "length_bin": pd.cut(
                    [acquisition.n_frames], bins=[0, 50, 100, 200, 500, np.inf], right=False
                )[0],
                "exposure_bin": pd.cut(
                    [exposure_fraction], bins=[0, 0.25, 0.5, 0.8, 1.01], right=False
                )[0],
                "missing_bin": pd.cut(
                    [acquisition.missing_probability],
                    bins=[0, 0.1, 0.2, 0.4, 1.0],
                    right=False,
                )[0],
                "correct": float(prediction == sample.state_label),
                "confidence": float(score),
                "selected": float(score >= confidence_threshold),
            }
        )
    frame = pd.DataFrame(rows)
    group_columns = [
        "split",
        "state",
        "localization_bin_nm",
        "length_bin",
        "exposure_bin",
        "missing_bin",
    ]
    grouped = frame.groupby(group_columns, observed=True, dropna=False)
    result = grouped.agg(
        n=("correct", "size"),
        accuracy=("correct", "mean"),
        mean_confidence=("confidence", "mean"),
        coverage=("selected", "mean"),
    ).reset_index()
    return result
