"""Physics-feature logistic-regression baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..features import FEATURE_VERSION, feature_matrix
from ..types import TrajectorySample


def train_classical(
    samples: list[TrajectorySample],
    *,
    measurement_aware: bool = True,
    regularization_c: float = 1.0,
    max_iter: int = 3000,
    seed: int = 20260904,
) -> dict[str, Any]:
    """Fit a standardized multinomial logistic-regression baseline."""

    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("Install the 'ml' optional dependencies to train models") from error

    matrix, labels, names, _ = feature_matrix(samples, measurement_aware=measurement_aware)
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=regularization_c,
                    max_iter=max_iter,
                    class_weight="balanced",
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    pipeline.fit(matrix, labels)
    return {
        "pipeline": pipeline,
        "measurement_aware": measurement_aware,
        "feature_names": names,
        "feature_version": FEATURE_VERSION,
        "classes": pipeline.classes_.tolist(),
        "training_samples": len(samples),
        "seed": seed,
    }


def predict_classical(
    artifact: dict[str, Any], samples: list[TrajectorySample]
) -> tuple[np.ndarray, np.ndarray]:
    matrix, _, names, _ = feature_matrix(
        samples, measurement_aware=bool(artifact["measurement_aware"])
    )
    if list(artifact["feature_names"]) != names:
        raise ValueError("Feature schema does not match the trained artifact")
    probabilities = artifact["pipeline"].predict_proba(matrix)
    classes = np.asarray(artifact["pipeline"].classes_, dtype=object)
    return probabilities, classes


def save_classical_model(artifact: dict[str, Any], output: str | Path) -> None:
    try:
        import joblib
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("Install joblib to save models") from error
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_classical_model(path: str | Path) -> dict[str, Any]:
    try:
        import joblib
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("Install joblib to load models") from error
    return joblib.load(path)
