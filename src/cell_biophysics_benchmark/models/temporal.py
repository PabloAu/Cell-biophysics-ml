"""Small masked temporal-convolution baseline.

PyTorch is imported lazily so simulation and feature extraction remain usable
without the heavy ML dependency.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..types import TrajectorySample


def temporal_inputs(sample: TrajectorySample) -> tuple[np.ndarray, np.ndarray]:
    """Create displacement/mask channels and acquisition metadata.

    Displacements are scaled by ``sqrt(frame_interval)`` to reduce trivial time
    discretization shift while retaining diffusion-scale information. Invalid
    displacement pairs are zeroed and represented by the third channel.
    """

    sample.validate()
    dt = sample.acquisition_params.frame_interval_s
    valid = sample.observed_mask[:-1] & sample.observed_mask[1:]
    displacement = np.zeros((len(sample.t_s) - 1, 2), dtype=np.float32)
    displacement[valid] = (
        sample.observed_xy_um[1:][valid] - sample.observed_xy_um[:-1][valid]
    ) / np.sqrt(dt)
    sequence = np.vstack(
        [displacement.T, valid.astype(np.float32, copy=False)[None, :]]
    ).astype(np.float32)
    acquisition = sample.acquisition_params
    metadata = np.asarray(
        [
            np.log(dt),
            acquisition.exposure_time_s / dt,
            np.log(acquisition.localization_sigma_um + 1e-6),
            acquisition.missing_probability,
            np.log(acquisition.n_frames),
        ],
        dtype=np.float32,
    )
    return sequence, metadata


def build_temporal_cnn(
    *,
    n_classes: int,
    metadata_features: int = 0,
    hidden_channels: int = 64,
    kernel_size: int = 5,
    dropout: float = 0.15,
) -> Any:
    try:
        import torch
        from torch import nn
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Install the 'ml' optional dependencies for the temporal model"
        ) from error

    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd to preserve sequence length")

    class TemporalCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(3, hidden_channels, kernel_size, padding=kernel_size // 2),
                nn.GELU(),
                nn.Conv1d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size,
                    padding=kernel_size // 2,
                ),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.classifier = nn.Sequential(
                nn.Linear(hidden_channels + metadata_features, hidden_channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, n_classes),
            )

        def forward(self, sequence: Any, metadata: Any | None = None) -> Any:
            # Channels are dx, dy, and validity. Pool only valid displacements.
            encoded = self.encoder(sequence)
            mask = sequence[:, 2:3, :].clamp(0, 1)
            pooled = (encoded * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
            if metadata_features:
                if metadata is None:
                    raise ValueError("metadata is required for this model")
                pooled = torch.cat([pooled, metadata], dim=1)
            return self.classifier(pooled)

    return TemporalCNN()
