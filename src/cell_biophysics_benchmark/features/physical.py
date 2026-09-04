"""Interpretable trajectory summaries with explicit missing-data handling."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..types import TrajectorySample

FEATURE_VERSION = "0.1.0"

TRAJECTORY_FEATURES = (
    "n_frames",
    "n_observed",
    "observed_fraction",
    "duration_s",
    "mean_step_um",
    "step_std_um",
    "step_q90_um",
    "max_step_um",
    "net_displacement_um",
    "path_length_um",
    "straightness",
    "efficiency",
    "radius_gyration_um",
    "asymmetry",
    "mean_turn_cosine",
    "velocity_autocorrelation_lag1",
    "msd_alpha",
    "apparent_diffusion_um2_s",
    "msd_plateau_ratio",
)

ACQUISITION_FEATURES = (
    "frame_interval_s",
    "exposure_fraction",
    "localization_sigma_um",
    "declared_missing_probability",
)


def time_averaged_msd(
    xy: np.ndarray, mask: np.ndarray, dt_s: float, max_lag: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a gap-aware time-averaged MSD.

    A pair contributes at lag ``k`` only when both endpoint localizations are
    observed. The number of contributing pairs is returned for uncertainty and
    quality control.
    """

    xy = np.asarray(xy, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if xy.ndim != 2 or xy.shape[1] != 2 or mask.shape != (len(xy),):
        raise ValueError("Expected xy shape (T, 2) and mask shape (T,)")
    if dt_s <= 0:
        raise ValueError("dt_s must be positive")
    if max_lag is None:
        max_lag = min(20, max(1, len(xy) // 4))
    max_lag = min(int(max_lag), len(xy) - 1)
    lags: list[float] = []
    values: list[float] = []
    counts: list[int] = []
    for lag in range(1, max_lag + 1):
        valid = mask[:-lag] & mask[lag:]
        if not valid.any():
            continue
        displacement = xy[lag:][valid] - xy[:-lag][valid]
        squared = np.einsum("ij,ij->i", displacement, displacement)
        lags.append(lag * dt_s)
        values.append(float(squared.mean()))
        counts.append(int(valid.sum()))
    return np.asarray(lags), np.asarray(values), np.asarray(counts)


def _safe_float(value: float, default: float = np.nan) -> float:
    return float(value) if np.isfinite(value) else float(default)


def extract_features(
    sample: TrajectorySample, *, measurement_aware: bool = False
) -> dict[str, float]:
    """Extract an interpretable, fixed-width feature vector from one sample."""

    sample.validate()
    xy = sample.observed_xy_um
    mask = sample.observed_mask
    acquisition = sample.acquisition_params
    valid_xy = xy[mask]
    valid_times = sample.t_s[mask]

    consecutive = mask[:-1] & mask[1:]
    steps = xy[1:][consecutive] - xy[:-1][consecutive]
    step_lengths = np.linalg.norm(steps, axis=1) if len(steps) else np.asarray([])
    if len(valid_xy) >= 2:
        net = float(np.linalg.norm(valid_xy[-1] - valid_xy[0]))
        duration = float(valid_times[-1] - valid_times[0])
    else:
        net = 0.0
        duration = 0.0
    path_length = float(step_lengths.sum()) if len(step_lengths) else 0.0
    straightness = net / path_length if path_length > 0 else 0.0
    denominator = float(np.square(step_lengths).sum()) if len(step_lengths) else 0.0
    efficiency = net * net / (len(step_lengths) * denominator) if denominator > 0 else 0.0

    centered = valid_xy - valid_xy.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, len(centered))
    eigenvalues = np.linalg.eigvalsh(covariance)
    radius_gyration = float(np.sqrt(max(0.0, eigenvalues.sum())))
    asymmetry = float((eigenvalues[-1] - eigenvalues[0]) / (eigenvalues.sum() + 1e-15))

    turn_cosines: list[float] = []
    velocity_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for index in range(1, len(xy) - 1):
        if mask[index - 1 : index + 2].all():
            previous = xy[index] - xy[index - 1]
            following = xy[index + 1] - xy[index]
            scale = float(np.linalg.norm(previous) * np.linalg.norm(following))
            if scale > 0:
                turn_cosines.append(float(previous @ following / scale))
                velocity_pairs.append((previous, following))
    mean_turn = float(np.mean(turn_cosines)) if turn_cosines else np.nan
    if velocity_pairs:
        left = np.vstack([pair[0] for pair in velocity_pairs])
        right = np.vstack([pair[1] for pair in velocity_pairs])
        velocity_autocorrelation = float(
            np.mean(np.einsum("ij,ij->i", left, right))
            / np.sqrt(np.mean(np.square(left)) * np.mean(np.square(right)) + 1e-15)
        )
    else:
        velocity_autocorrelation = np.nan

    lag_s, msd, _ = time_averaged_msd(xy, mask, acquisition.frame_interval_s)
    positive = (lag_s > 0) & (msd > 0)
    if positive.sum() >= 2:
        msd_alpha, _ = np.polyfit(np.log(lag_s[positive]), np.log(msd[positive]), deg=1)
    else:
        msd_alpha = np.nan
    apparent_diffusion = (
        float(msd[0] / (4.0 * lag_s[0])) if len(msd) and lag_s[0] > 0 else np.nan
    )
    plateau_ratio = float(msd[-1] / (np.max(msd) + 1e-15)) if len(msd) >= 2 else np.nan

    values = {
        "n_frames": float(acquisition.n_frames),
        "n_observed": float(mask.sum()),
        "observed_fraction": float(mask.mean()),
        "duration_s": duration,
        "mean_step_um": float(step_lengths.mean()) if len(step_lengths) else np.nan,
        "step_std_um": float(step_lengths.std()) if len(step_lengths) else np.nan,
        "step_q90_um": float(np.quantile(step_lengths, 0.9)) if len(step_lengths) else np.nan,
        "max_step_um": float(step_lengths.max()) if len(step_lengths) else np.nan,
        "net_displacement_um": net,
        "path_length_um": path_length,
        "straightness": straightness,
        "efficiency": efficiency,
        "radius_gyration_um": radius_gyration,
        "asymmetry": asymmetry,
        "mean_turn_cosine": mean_turn,
        "velocity_autocorrelation_lag1": velocity_autocorrelation,
        "msd_alpha": float(msd_alpha),
        "apparent_diffusion_um2_s": apparent_diffusion,
        "msd_plateau_ratio": plateau_ratio,
    }
    if measurement_aware:
        values.update(
            {
                "frame_interval_s": acquisition.frame_interval_s,
                "exposure_fraction": (
                    acquisition.exposure_time_s / acquisition.frame_interval_s
                ),
                "localization_sigma_um": acquisition.localization_sigma_um,
                "declared_missing_probability": acquisition.missing_probability,
            }
        )
    return {key: _safe_float(value) for key, value in values.items()}


def feature_matrix(
    samples: Sequence[TrajectorySample], *, measurement_aware: bool = False
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    names = list(TRAJECTORY_FEATURES)
    if measurement_aware:
        names.extend(ACQUISITION_FEATURES)
    rows = [extract_features(sample, measurement_aware=measurement_aware) for sample in samples]
    matrix = np.asarray([[row[name] for name in names] for row in rows], dtype=float)
    labels = np.asarray([sample.state_label for sample in samples], dtype=object)
    identifiers = [sample.sample_id for sample in samples]
    return matrix, labels, names, identifiers
