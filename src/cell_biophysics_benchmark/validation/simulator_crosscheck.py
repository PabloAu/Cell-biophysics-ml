"""Statistics for comparing independently generated latent trajectory ensembles."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _validate_paths(paths: np.ndarray) -> np.ndarray:
    values = np.asarray(paths, dtype=float)
    if values.ndim != 3 or values.shape[2] != 2:
        raise ValueError("paths must have shape (trajectories, time points, 2)")
    if values.shape[0] < 2 or values.shape[1] < 3:
        raise ValueError("at least two trajectories and three time points are required")
    if not np.isfinite(values).all():
        raise ValueError("paths must contain only finite values")
    return values


def time_averaged_msd(paths: np.ndarray, lags: Sequence[int]) -> np.ndarray:
    """Return the ensemble mean of time-averaged squared displacements."""

    values = _validate_paths(paths)
    results: list[float] = []
    for lag in lags:
        if lag < 1 or lag >= values.shape[1]:
            raise ValueError("lags must be positive and shorter than each trajectory")
        displacement = values[:, lag:, :] - values[:, :-lag, :]
        results.append(float(np.mean(np.sum(displacement**2, axis=2))))
    return np.asarray(results)


def increment_correlation(paths: np.ndarray) -> float:
    """Return the pooled lag-one Pearson correlation of coordinate increments."""

    values = _validate_paths(paths)
    increments = np.diff(values, axis=1)
    left = increments[:, :-1, :].reshape(-1)
    right = increments[:, 1:, :].reshape(-1)
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(np.sqrt(np.sum(left**2) * np.sum(right**2)))
    if denominator == 0:
        raise ValueError("increment correlation is undefined for zero-variance paths")
    return float(np.sum(left * right) / denominator)


def summarize_ensemble(
    paths: np.ndarray,
    *,
    diffusion: float,
    frame_interval: float,
    alpha: float,
    lags: Sequence[int],
) -> dict[str, object]:
    """Summarize MSD scaling and the characteristic increment correlation."""

    lag_array = np.asarray(lags, dtype=int)
    measured = time_averaged_msd(paths, lag_array)
    expected = 4.0 * diffusion * (lag_array * frame_interval) ** alpha
    relative_error = np.abs(measured - expected) / expected
    expected_correlation = 0.5 * (2.0**alpha - 2.0)
    measured_correlation = increment_correlation(paths)
    return {
        "lags_frames": lag_array.tolist(),
        "msd_measured_um2": measured.tolist(),
        "msd_expected_um2": expected.tolist(),
        "msd_relative_error": relative_error.tolist(),
        "max_msd_relative_error": float(relative_error.max()),
        "increment_correlation_lag1": measured_correlation,
        "increment_correlation_expected": expected_correlation,
        "increment_correlation_absolute_error": abs(
            measured_correlation - expected_correlation
        ),
    }


def compare_ensembles(
    first: np.ndarray,
    second: np.ndarray,
    *,
    diffusion: float,
    frame_interval: float,
    alpha: float,
    lags: Sequence[int],
    theory_relative_tolerance: float,
    between_relative_tolerance: float,
    correlation_absolute_tolerance: float,
) -> dict[str, object]:
    """Compare two simulators against theory and against each other."""

    first_summary = summarize_ensemble(
        first,
        diffusion=diffusion,
        frame_interval=frame_interval,
        alpha=alpha,
        lags=lags,
    )
    second_summary = summarize_ensemble(
        second,
        diffusion=diffusion,
        frame_interval=frame_interval,
        alpha=alpha,
        lags=lags,
    )
    first_msd = np.asarray(first_summary["msd_measured_um2"])
    second_msd = np.asarray(second_summary["msd_measured_um2"])
    midpoint = 0.5 * (first_msd + second_msd)
    between_error = np.abs(first_msd - second_msd) / midpoint
    checks = {
        "first_matches_theory": (
            first_summary["max_msd_relative_error"] <= theory_relative_tolerance
        ),
        "second_matches_theory": (
            second_summary["max_msd_relative_error"] <= theory_relative_tolerance
        ),
        "simulators_agree": float(between_error.max()) <= between_relative_tolerance,
        "first_increment_correlation_matches_theory": (
            first_summary["increment_correlation_absolute_error"]
            <= correlation_absolute_tolerance
        ),
        "second_increment_correlation_matches_theory": (
            second_summary["increment_correlation_absolute_error"]
            <= correlation_absolute_tolerance
        ),
    }
    return {
        "first": first_summary,
        "second": second_summary,
        "between_simulator_msd_relative_error": between_error.tolist(),
        "max_between_simulator_msd_relative_error": float(between_error.max()),
        "thresholds": {
            "theory_relative_tolerance": theory_relative_tolerance,
            "between_relative_tolerance": between_relative_tolerance,
            "correlation_absolute_tolerance": correlation_absolute_tolerance,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
