"""Classical and temporal baselines."""

from .classical import (
    load_classical_model,
    predict_classical,
    save_classical_model,
    train_classical,
)

__all__ = [
    "load_classical_model",
    "predict_classical",
    "save_classical_model",
    "train_classical",
]
